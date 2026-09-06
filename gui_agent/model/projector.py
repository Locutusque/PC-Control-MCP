"""Vision -> LM bridge (plan section 3.1).

Two options, and the choice is a real latency decision rather than a
preference:

* :class:`MLPProjector` keeps all ``grid**2`` patch tokens (576 by default).
  Spatial structure is preserved exactly, which is what the coordinate
  vocabulary wants, but the LM pays attention over 576 image tokens *every
  control tick* and the image prefix cannot be cached because the screen
  changes.
* :class:`PerceiverResampler` compresses those to a fixed ``num_latents``
  (144 by default), cutting the per-tick prefill roughly 4x at the cost of
  making the patch-to-token correspondence learned instead of exact.

Start with the MLP: it makes the coordinate grounding as easy as possible to
learn.  Move to the resampler only if the latency pass (plan section 7.2, step
7) says the image prefix is the bottleneck -- and re-measure click accuracy when
you do, because that is the metric the compression puts at risk.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ProjectorConfig

__all__ = ["MLPProjector", "PerceiverResampler", "build_projector"]


class MLPProjector(nn.Module):
    """Per-patch MLP into the LM embedding space; token count unchanged."""

    def __init__(self, in_dim: int, out_dim: int, config: ProjectorConfig | None = None) -> None:
        super().__init__()
        config = config or ProjectorConfig()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(1, config.depth) - 1):
            layers += [nn.Linear(dim, config.hidden_dim), nn.GELU(), nn.Dropout(config.dropout)]
            dim = config.hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(patches))

    def output_tokens(self, n_patches: int) -> int:
        return n_patches


class PerceiverResampler(nn.Module):
    """Cross-attention from learned latents into the patch grid."""

    def __init__(self, in_dim: int, out_dim: int, config: ProjectorConfig | None = None) -> None:
        super().__init__()
        config = config or ProjectorConfig()
        self.num_latents = config.num_latents
        self.latents = nn.Parameter(torch.randn(1, config.num_latents, out_dim) * 0.02)
        self.kv_proj = nn.Linear(in_dim, out_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_latents": nn.LayerNorm(out_dim),
                        "norm_context": nn.LayerNorm(out_dim),
                        "attn": nn.MultiheadAttention(
                            out_dim, config.num_heads, dropout=config.dropout, batch_first=True
                        ),
                        "norm_ff": nn.LayerNorm(out_dim),
                        "ff": nn.Sequential(
                            nn.Linear(out_dim, config.hidden_dim), nn.GELU(),
                            nn.Linear(config.hidden_dim, out_dim),
                        ),
                    }
                )
                for _ in range(max(1, config.depth))
            ]
        )
        self.norm = nn.LayerNorm(out_dim)
        self.out_dim = out_dim

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        context = self.kv_proj(patches)
        latents = self.latents.expand(patches.shape[0], -1, -1)
        for layer in self.layers:
            q = layer["norm_latents"](latents)
            kv = layer["norm_context"](context)
            latents = latents + layer["attn"](q, kv, kv, need_weights=False)[0]
            latents = latents + layer["ff"](layer["norm_ff"](latents))
        return self.norm(latents)

    def output_tokens(self, n_patches: int) -> int:
        return self.num_latents


def build_projector(in_dim: int, out_dim: int, config: ProjectorConfig | None = None) -> nn.Module:
    config = config or ProjectorConfig()
    if config.kind == "perceiver":
        return PerceiverResampler(in_dim, out_dim, config)
    return MLPProjector(in_dim, out_dim, config)
