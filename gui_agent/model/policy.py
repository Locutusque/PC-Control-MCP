"""The GUI control policy: ViT -> projector -> pretrained LM (plan section 3).

Context layout per control tick
-------------------------------
::

    [ instruction ][ form_data ]  [ image patches ][ action history ]  -> action
    \\------- static prefix ------/ \\------ recomputed every tick -----/

The ordering is forced by caching (plan 3.4).  Image tokens change every frame,
and a causal LM's KV cache is only reusable up to the first token that changes,
so anything cacheable must come *before* the image.  Putting the instruction
first is what lets it be encoded once per subtask instead of 15 times a second.

Tuning (plan 3.2)
-----------------
Base LM weights stay frozen.  Only LoRA adapters, the projector, the ViT and
the freshly added action-token embedding rows train.  The GUI corpus is narrow
next to the LM's pretraining data, and full fine-tuning would overwrite the
language priors the LM was brought in for -- which are exactly what free-compose
typing and instruction understanding depend on.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..actions import ActionCodec
from ..config import PolicyConfig
from .decoding import ConstrainedActionDecoder, DecodeResult, SamplingConfig
from .projector import build_projector
from .tokenizer_ext import ActionTokenizer
from .vit import ViT, preprocess_screenshot

log = logging.getLogger(__name__)

__all__ = ["GuiPolicy", "PolicyBatch", "PolicySession", "IGNORE_INDEX"]

IGNORE_INDEX = -100


@dataclass
class PolicyBatch:
    """A padded training batch."""

    pixels: torch.Tensor           # (B, C, H, W)
    prefix_ids: torch.Tensor       # (B, P) left-padded static context
    prefix_mask: torch.Tensor      # (B, P)
    suffix_ids: torch.Tensor       # (B, S) history + target
    suffix_mask: torch.Tensor      # (B, S)
    labels: torch.Tensor           # (B, S) IGNORE_INDEX except target positions
    loss_weights: torch.Tensor     # (B,) per-example CE weight

    def to(self, device, dtype=None) -> PolicyBatch:
        pixels = self.pixels.to(device=device, dtype=dtype) if dtype else self.pixels.to(device)
        return PolicyBatch(
            pixels=pixels,
            prefix_ids=self.prefix_ids.to(device),
            prefix_mask=self.prefix_mask.to(device),
            suffix_ids=self.suffix_ids.to(device),
            suffix_mask=self.suffix_mask.to(device),
            labels=self.labels.to(device),
            loss_weights=self.loss_weights.to(device),
        )

    def __len__(self) -> int:
        return self.pixels.shape[0]


class GuiPolicy(nn.Module):
    """Vision-language policy that emits action tokens."""

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: ActionTokenizer,
        language_model: nn.Module,
        vision: ViT | None = None,
        projector: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.codec = ActionCodec(config.action_space)
        self.lm = language_model
        self.vision = vision or ViT(config.vision)

        lm_dim = self.lm.get_input_embeddings().weight.shape[1]
        self.projector = projector or build_projector(
            self.vision.output_dim, lm_dim, config.projector
        )
        self.decoder = ConstrainedActionDecoder(tokenizer, self.codec)

    # -- construction -----------------------------------------------------
    @classmethod
    def from_pretrained_lm(
        cls, config: PolicyConfig | None = None, tokenizer: ActionTokenizer | None = None
    ) -> GuiPolicy:
        """Build a fresh policy around a pretrained LM, with LoRA attached."""
        from transformers import AutoModelForCausalLM

        config = config or PolicyConfig()
        tokenizer = tokenizer or ActionTokenizer.from_pretrained(
            config.decoder.base_model, config.action_space
        )
        dec = config.decoder
        kwargs: dict = {"dtype": getattr(torch, dec.torch_dtype)}
        if dec.attn_implementation:
            kwargs["attn_implementation"] = dec.attn_implementation
        if dec.load_in_4bit or dec.load_in_8bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=dec.load_in_4bit, load_in_8bit=dec.load_in_8bit,
                bnb_4bit_compute_dtype=getattr(torch, dec.torch_dtype),
            )

        lm = AutoModelForCausalLM.from_pretrained(dec.base_model, **kwargs)
        tokenizer.resize_embeddings(lm)

        if dec.freeze_base:
            lm = _attach_lora(lm, dec)
            _freeze_embeddings_except(lm, tokenizer.new_token_slice(), dec.train_new_embeddings)

        return cls(config, tokenizer, lm)

    # -- context ----------------------------------------------------------
    def build_prefix(self, instruction: str, form_data: dict | None = None) -> list[int]:
        """The cacheable static context for one subtask.

        form_data keys are numbered so ``<FIELD_k>`` has a visible referent in
        the prompt; the *values* are shown too, because free-compose has to be
        able to tell that a field already covers what it would otherwise write.
        """
        parts = [f"instruction: {instruction.strip() or 'continue'}"]
        if form_data:
            fields = list(form_data.items())[: self.config.action_space.max_form_fields]
            rendered = ", ".join(f"{i}={k!r}: {v!r}" for i, (k, v) in enumerate(fields))
            parts.append(f"form_data: {{{rendered}}}")
        text = "\n".join(parts) + "\nactions:"
        return self.tokenizer.encode_text(
            text, self.config.max_instruction_tokens + self.config.max_form_data_tokens
        )

    def build_history(self, history: Sequence[Sequence[str]]) -> list[int]:
        ids: list[int] = []
        for atoms in list(history)[-self.config.max_history :]:
            ids.extend(self.tokenizer.encode_atoms(atoms))
        return ids

    # -- embeddings -------------------------------------------------------
    def encode_image(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projector(self.vision(pixels))

    def _embed_ids(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(ids)

    def _assemble(self, batch: PolicyBatch) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Concatenate prefix, image and suffix into one embedded sequence."""
        image = self.encode_image(batch.pixels).to(self._embed_ids(batch.prefix_ids).dtype)
        prefix = self._embed_ids(batch.prefix_ids)
        suffix = self._embed_ids(batch.suffix_ids)
        embeds = torch.cat([prefix, image, suffix], dim=1)
        image_mask = torch.ones(
            image.shape[:2], dtype=batch.prefix_mask.dtype, device=batch.prefix_mask.device
        )
        mask = torch.cat([batch.prefix_mask, image_mask, batch.suffix_mask], dim=1)
        return embeds, mask, image.shape[1]

    # -- training ---------------------------------------------------------
    def forward(self, batch: PolicyBatch) -> dict:
        """Weighted next-action-token cross-entropy."""
        embeds, mask, n_image = self._assemble(batch)
        outputs = self.lm(inputs_embeds=embeds, attention_mask=mask)
        logits = outputs.logits

        # Only suffix positions carry labels.  Predicting token t happens at
        # position t-1, so the logits are shifted by one relative to labels.
        n_prefix = batch.prefix_ids.shape[1]
        start = n_prefix + n_image
        suffix_logits = logits[:, start - 1 : -1, :]
        labels = batch.labels

        loss_per_token = nn.functional.cross_entropy(
            suffix_logits.reshape(-1, suffix_logits.shape[-1]).float(),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(labels.shape)

        valid = (labels != IGNORE_INDEX).float()
        # Normalise per example before weighting, so a long free-compose target
        # does not outweigh a 4-token click purely by length.
        per_example = (loss_per_token * valid).sum(1) / valid.sum(1).clamp(min=1)
        weights = batch.loss_weights.to(per_example.dtype)
        loss = (per_example * weights).sum() / weights.sum().clamp(min=1e-6)

        with torch.no_grad():
            predicted = suffix_logits.argmax(-1)
            correct = ((predicted == labels) & (labels != IGNORE_INDEX)).float().sum()
            token_accuracy = correct / valid.sum().clamp(min=1)

        return {
            "loss": loss,
            "per_example_loss": per_example.detach(),
            "token_accuracy": token_accuracy,
            "n_target_tokens": valid.sum().detach(),
        }

    # -- inference --------------------------------------------------------
    @torch.no_grad()
    def act(
        self,
        frame,
        instruction: str,
        form_data: dict | None = None,
        history: Sequence[Sequence[str]] = (),
        session: PolicySession | None = None,
        sampling: SamplingConfig | None = None,
    ) -> DecodeResult:
        """One control tick: frame in, action out."""
        if sampling is not None:
            self.decoder.sampling = sampling
        device = next(self.parameters()).device
        dtype = self._embed_ids(torch.zeros(1, 1, dtype=torch.long, device=device)).dtype

        pixels = preprocess_screenshot(frame, self.config.vision).to(device=device, dtype=dtype)
        image = self.encode_image(pixels).to(dtype)

        if session is None:
            session = PolicySession(self, instruction, form_data)
            session.warm(device)

        history_ids = self.build_history(history)
        history_embeds = (
            self._embed_ids(torch.tensor([history_ids], device=device))
            if history_ids
            else image.new_zeros((1, 0, image.shape[-1]))
        )

        cache = session.fresh_cache()
        prompt = torch.cat([image, history_embeds], dim=1)
        state = {"first": True, "past": cache, "seen": session.prefix_len}

        def step(generated_ids: list[int]) -> torch.Tensor:
            if state["first"]:
                embeds = prompt
                state["first"] = False
            else:
                embeds = self._embed_ids(
                    torch.tensor([[generated_ids[-1]]], device=device)
                )
            n_new = embeds.shape[1]
            mask = torch.ones((1, state["seen"] + n_new), dtype=torch.long, device=device)
            out = self.lm(
                inputs_embeds=embeds,
                attention_mask=mask,
                past_key_values=state["past"],
                use_cache=True,
            )
            state["past"] = out.past_key_values
            state["seen"] += n_new
            return out.logits[0, -1]

        return self.decoder.decode(step, form_fields=list((form_data or {}).keys()))

    # -- persistence ------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return {
            "total": total,
            "trainable": trainable,
            "trainable_fraction": round(trainable / total, 5) if total else 0.0,
            "vision": sum(p.numel() for p in self.vision.parameters()),
            "projector": sum(p.numel() for p in self.projector.parameters()),
            "lm": sum(p.numel() for p in self.lm.parameters()),
        }

    def save(self, path: str | Path) -> None:
        """Save the trainable parts plus the vocabulary signature.

        The signature is what stops a checkpoint being loaded against a
        differently-ordered action vocabulary, which would remap the action
        embeddings and produce confident wrong clicks with no error.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.state_dict().items() if not k.startswith("lm.")}
        torch.save(state, path / "policy.pt")
        if hasattr(self.lm, "save_pretrained"):
            self.lm.save_pretrained(path / "lm")
        self.tokenizer.save_pretrained(path / "tokenizer")
        (path / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        (path / "action_vocab.json").write_text(
            json.dumps(
                {
                    "signature": self.tokenizer.signature(),
                    "action_id_start": self.tokenizer.action_id_start,
                    "tokens": list(self.tokenizer.vocab.tokens),
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> GuiPolicy:
        path = Path(path)
        config = PolicyConfig.from_dict(json.loads((path / "config.json").read_text()))
        tokenizer = ActionTokenizer.from_pretrained(
            str(path / "tokenizer"), config.action_space
        )
        expected = json.loads((path / "action_vocab.json").read_text()).get("signature")
        tokenizer.check_signature(expected)

        from transformers import AutoModelForCausalLM

        lm_path = path / "lm"
        lm = AutoModelForCausalLM.from_pretrained(
            str(lm_path) if lm_path.exists() else config.decoder.base_model,
            dtype=getattr(torch, config.decoder.torch_dtype),
        )
        tokenizer.resize_embeddings(lm)
        policy = cls(config, tokenizer, lm)
        missing, unexpected = policy.load_state_dict(
            torch.load(path / "policy.pt", map_location=device), strict=False
        )
        unexpected = [k for k in unexpected if not k.startswith("lm.")]
        if unexpected:
            log.warning("unexpected keys in checkpoint: %s", unexpected[:8])
        return policy.to(device)


class PolicySession:
    """Per-subtask KV cache for the static instruction prefix (plan 3.4).

    The instruction and form_data do not change while a subtask runs, so their
    keys and values are computed once and reused for every tick.  Only the
    image and the action history are recomputed -- which is unavoidable, since
    the screen is what changed.
    """

    _warned_crop = False

    def __init__(self, policy: GuiPolicy, instruction: str, form_data: dict | None = None) -> None:
        self.policy = policy
        self.instruction = instruction
        self.form_data = dict(form_data or {})
        self.prefix_ids = policy.build_prefix(instruction, form_data)
        self.prefix_len = len(self.prefix_ids)
        self._cache = None

    @torch.no_grad()
    def warm(self, device=None) -> PolicySession:
        """Run the prefix through the LM once and keep its cache."""
        device = device or next(self.policy.parameters()).device
        ids = torch.tensor([self.prefix_ids], device=device)
        out = self.policy.lm(input_ids=ids, use_cache=True)
        self._cache = out.past_key_values
        return self

    def fresh_cache(self):
        """The prefix cache, trimmed back to prefix length.

        Each tick appends image and action tokens to the cache; those entries
        describe a screen that no longer exists, so they are dropped before the
        next tick rather than accumulating.

        ``crop`` changed meaning across transformers versions -- it took a
        target length, and now takes a negative count of tokens to drop -- so
        the result is *verified* rather than assumed.  A cache left one token
        too long would misalign every position that follows and produce
        plausible, wrong actions with no error.
        """
        if self._cache is None:
            self.warm()
        cache = self._cache
        extra = self._cache_length(cache) - self.prefix_len
        if extra <= 0:
            return cache

        crop = getattr(cache, "crop", None)
        if crop is not None:
            for argument in (-extra, self.prefix_len):
                try:
                    crop(argument)
                except (TypeError, ValueError):
                    continue
                if self._cache_length(cache) == self.prefix_len:
                    return cache

        # Either there is no usable crop, or it did not land on the expected
        # length.  Re-encoding the prefix is slower but always correct.
        if not self._warned_crop:
            log.warning(
                "KV cache could not be trimmed to %d tokens; re-encoding the prefix "
                "each tick (slower, but correct)", self.prefix_len,
            )
            PolicySession._warned_crop = True
        self.warm()
        return self._cache

    @staticmethod
    def _cache_length(cache) -> int:
        getter = getattr(cache, "get_seq_length", None)
        if getter is not None:
            return int(getter())
        return len(cache)  # pragma: no cover - legacy tuple caches

    def reset(self) -> None:
        self._cache = None


# --------------------------------------------------------------------------
# Parameter freezing
# --------------------------------------------------------------------------


def _attach_lora(lm, decoder_config):
    from peft import LoraConfig, get_peft_model

    lora = LoraConfig(
        r=decoder_config.lora_r,
        lora_alpha=decoder_config.lora_alpha,
        lora_dropout=decoder_config.lora_dropout,
        target_modules=list(decoder_config.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(lm, lora)


def _freeze_embeddings_except(lm, new_slice: slice, train_new: bool) -> None:
    """Train only the newly added embedding rows (plan 3.2).

    ``modules_to_save=["embed_tokens"]`` would make the whole matrix trainable
    -- hundreds of millions of parameters, and every one of them a chance to
    drift the language priors.  Instead the weight stays trainable but a
    gradient hook zeroes every row outside the action-token block, so the
    optimiser can only move the ~250 rows that started from noise.
    """
    if not train_new:
        return

    embeddings = lm.get_input_embeddings()
    weight = embeddings.weight
    weight.requires_grad_(True)

    def mask_grad(grad: torch.Tensor) -> torch.Tensor:
        masked = torch.zeros_like(grad)
        masked[new_slice] = grad[new_slice]
        return masked

    weight.register_hook(mask_grad)

    output = lm.get_output_embeddings()
    if output is not None and output.weight is not weight:
        # Untied lm_head: it has its own rows for the new tokens and needs the
        # same treatment, or the model can embed an action token but never
        # score one.
        output.weight.requires_grad_(True)
        output.weight.register_hook(mask_grad)
    log.info(
        "training embedding rows [%s, %s) only", new_slice.start, new_slice.stop
    )
