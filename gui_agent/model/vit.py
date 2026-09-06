"""Small ViT over a screenshot (plan section 3.1).

Two deliberate departures from a classification ViT:

* **No CLS token and no pooling.**  The action vocabulary addresses locations by
  patch index, so ``<ROW_r> <COL_c>`` has to correspond to a real patch
  embedding.  Pooling to a single vector would destroy exactly the structure the
  coordinate tokens rely on.
* **Learned position embeddings sized to the patch grid**, interpolated if the
  input resolution changes, so a checkpoint trained at one resolution can still
  be evaluated at another without the grid silently shifting under the
  coordinate vocabulary.

Defaults are ~86M parameters, inside the 50-90M budget of plan section 3.5.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import VisionConfig

__all__ = ["ViT", "PatchEmbed", "IMAGENET_MEAN", "IMAGENET_STD", "preprocess_screenshot"]

# Screenshots are not natural images, but the ViT is initialised from scratch,
# so the normalisation constants only need to be *consistent* between training
# and inference.  These are kept because pretrained ViT weights, if someone
# initialises from them, expect this scaling.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class PatchEmbed(nn.Module):
    """Non-overlapping patch projection via a strided conv."""

    def __init__(self, config: VisionConfig) -> None:
        super().__init__()
        self.config = config
        self.proj = nn.Conv2d(
            config.channels, config.dim,
            kernel_size=config.patch_size, stride=config.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, N, D), N in row-major order so index r*cols + c
        # is exactly the patch <ROW_r> <COL_c> addresses.
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """Patch-token encoder.  Output is ``(B, num_patches, dim)``."""

    def __init__(self, config: VisionConfig | None = None) -> None:
        super().__init__()
        self.config = config or VisionConfig()
        c = self.config
        self.patch_embed = PatchEmbed(c)
        self.pos_embed = nn.Parameter(torch.zeros(1, c.num_patches, c.dim))
        self.dropout = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(
            [Block(c.dim, c.heads, c.mlp_ratio, c.dropout) for _ in range(c.depth)]
        )
        self.norm = nn.LayerNorm(c.dim)
        self.apply(self._init)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def output_dim(self) -> int:
        return self.config.dim

    @property
    def grid_side(self) -> int:
        return self.config.grid_side

    def _interpolate_pos_embed(self, n_patches: int) -> torch.Tensor:
        if n_patches == self.config.num_patches:
            return self.pos_embed
        side = int(math.sqrt(n_patches))
        if side * side != n_patches:
            raise ValueError(f"{n_patches} patches is not a square grid")
        src = self.config.grid_side
        pos = self.pos_embed.reshape(1, src, src, self.config.dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(side, side), mode="bicubic", align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, n_patches, self.config.dim)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.dim() != 4:
            raise ValueError(f"expected (B, C, H, W), got {tuple(pixels.shape)}")
        x = self.patch_embed(pixels)
        x = self.dropout(x + self._interpolate_pos_embed(x.shape[1]))
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def preprocess_screenshot(image, config: VisionConfig | None = None) -> torch.Tensor:
    """Screenshot (numpy HWC uint8 or PIL image) -> normalised ``(1, C, H, W)``.

    Aspect ratio is *not* preserved: the whole screen is squashed to a square.
    That is intentional -- the coordinate vocabulary maps grid cells linearly
    back to screen pixels, so letterboxing would leave dead rows in the
    vocabulary and shift every coordinate.
    """
    config = config or VisionConfig()
    tensor = _to_tensor(image)
    if tensor.shape[-2:] != (config.image_size, config.image_size):
        tensor = F.interpolate(
            tensor, size=(config.image_size, config.image_size),
            mode="bilinear", align_corners=False, antialias=True,
        )
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _to_tensor(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.float()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] not in (1, 3):  # channels-last input
            tensor = tensor.permute(0, 3, 1, 2)
        return tensor / 255.0 if tensor.max() > 1.5 else tensor

    import numpy as np

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 4:  # BGRA/RGBA from the grabber
        array = array[:, :, :3]
    if array.ndim != 3:
        raise ValueError(f"expected an HWC image, got shape {array.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float().permute(2, 0, 1).unsqueeze(0)
    return tensor / 255.0
