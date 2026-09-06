"""Torch dataset and collation for training examples.

Turns a :class:`~gui_agent.data.schema.TrainingExample` into the exact tensor
layout :meth:`~gui_agent.model.policy.GuiPolicy.forward` expects: a cacheable
prefix, the observation frame, and a suffix of history plus target where only
the target positions carry labels.

The prefix is left-padded and the suffix right-padded.  That is not cosmetic:
the image embeddings are spliced between them, so the boundary has to be at a
fixed index across the batch for the label shift in ``forward`` to line up.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import torch
from torch.utils.data import Dataset, Sampler

from ..config import PolicyConfig
from ..model.policy import IGNORE_INDEX, PolicyBatch
from ..model.tokenizer_ext import ActionTokenizer
from ..model.vit import preprocess_screenshot
from .schema import TrainingExample
from .sources import FrameLoader, SegmentSource, iter_segments

log = logging.getLogger(__name__)

__all__ = ["GuiExampleDataset", "collate_examples", "SegmentGroupedSampler"]


class GuiExampleDataset(Dataset):
    """Examples plus the frames they reference."""

    def __init__(
        self,
        examples: Sequence[TrainingExample],
        tokenizer: ActionTokenizer,
        config: PolicyConfig,
        pool_root: str | None = None,
        raw_root: str | None = None,
        frame_loader: FrameLoader | None = None,
        skip_missing_frames: bool = True,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.config = config
        self.skip_missing_frames = skip_missing_frames
        self.frames = frame_loader or FrameLoader(
            config.vision.image_size, config.vision.image_size
        )
        self._sources: dict[str, SegmentSource] = {}
        if pool_root:
            self._sources = {s.segment_id: s for s in iter_segments(pool_root, raw_root)}
        self._missing = 0

    def __len__(self) -> int:
        return len(self.examples)

    def _pixels(self, example: TrainingExample) -> torch.Tensor:
        source = self._sources.get(example.segment_id)
        if source is None:
            raise FileNotFoundError(f"no source for segment {example.segment_id}")
        return preprocess_screenshot(
            self.frames.load(source, example.frame_index), self.config.vision
        )[0]

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        try:
            pixels = self._pixels(example)
        except (FileNotFoundError, IndexError, OSError) as exc:
            if not self.skip_missing_frames:
                raise
            # One unreadable segment must not kill a training run; substitute a
            # blank frame and count it, so a systematically broken pool shows up
            # in the log rather than as a mysteriously flat loss.
            self._missing += 1
            if self._missing in (1, 10, 100) or self._missing % 1000 == 0:
                log.warning("frame unavailable (%d so far): %s", self._missing, exc)
            size = self.config.vision.image_size
            pixels = torch.zeros(self.config.vision.channels, size, size)

        prefix_text = _render_prefix(example, self.config)
        prefix_ids = self.tokenizer.encode_text(
            prefix_text,
            self.config.max_instruction_tokens + self.config.max_form_data_tokens,
        )
        history_ids: list[int] = []
        for atoms in example.history[-self.config.max_history :]:
            history_ids.extend(self.tokenizer.encode_atoms(atoms))
        target_ids = self.tokenizer.encode_atoms(example.target)

        return {
            "pixels": pixels,
            "prefix_ids": prefix_ids,
            "history_ids": history_ids,
            "target_ids": target_ids,
            "loss_weight": example.loss_weight,
        }

    @property
    def missing_frames(self) -> int:
        return self._missing


def _render_prefix(example: TrainingExample, config: PolicyConfig) -> str:
    """Must match :meth:`GuiPolicy.build_prefix` exactly.

    Training and inference share a prompt format; drifting them apart is a
    silent train/serve skew that shows up only as degraded rollouts.
    """
    parts = [f"instruction: {example.instruction.strip() or 'continue'}"]
    if example.form_data:
        fields = list(example.form_data.items())[: config.action_space.max_form_fields]
        rendered = ", ".join(f"{i}={k!r}: {v!r}" for i, (k, v) in enumerate(fields))
        parts.append(f"form_data: {{{rendered}}}")
    return "\n".join(parts) + "\nactions:"


def collate_examples(items: Sequence[dict], pad_id: int) -> PolicyBatch:
    """Pad a list of dataset items into a :class:`PolicyBatch`."""
    n = len(items)
    max_prefix = max(len(i["prefix_ids"]) for i in items)
    suffixes = [i["history_ids"] + i["target_ids"] for i in items]
    max_suffix = max(len(s) for s in suffixes)

    prefix_ids = torch.full((n, max_prefix), pad_id, dtype=torch.long)
    prefix_mask = torch.zeros((n, max_prefix), dtype=torch.long)
    suffix_ids = torch.full((n, max_suffix), pad_id, dtype=torch.long)
    suffix_mask = torch.zeros((n, max_suffix), dtype=torch.long)
    labels = torch.full((n, max_suffix), IGNORE_INDEX, dtype=torch.long)

    for row, (item, suffix) in enumerate(zip(items, suffixes)):
        prefix = item["prefix_ids"]
        # Left-pad the prefix so the image always begins at column max_prefix.
        prefix_ids[row, max_prefix - len(prefix) :] = torch.tensor(prefix)
        prefix_mask[row, max_prefix - len(prefix) :] = 1

        suffix_ids[row, : len(suffix)] = torch.tensor(suffix)
        suffix_mask[row, : len(suffix)] = 1
        start = len(item["history_ids"])
        labels[row, start : len(suffix)] = torch.tensor(item["target_ids"])

    return PolicyBatch(
        pixels=torch.stack([i["pixels"] for i in items]),
        prefix_ids=prefix_ids,
        prefix_mask=prefix_mask,
        suffix_ids=suffix_ids,
        suffix_mask=suffix_mask,
        labels=labels,
        loss_weights=torch.tensor([float(i["loss_weight"]) for i in items]),
    )


class SegmentGroupedSampler(Sampler[int]):
    """Shuffles while keeping each segment's examples together.

    Frames are decoded per segment, so fully random shuffling re-decodes a
    video once per example drawn from it.  Grouping keeps
    :class:`~gui_agent.data.sources.FrameLoader`'s cache useful while still
    shuffling segment order and the examples within each segment.
    """

    def __init__(self, examples: Sequence[TrainingExample], seed: int = 0, shuffle: bool = True) -> None:
        self.groups: dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(examples):
            self.groups[example.segment_id].append(index)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return sum(len(v) for v in self.groups.values())

    def __iter__(self):
        import random

        keys = list(self.groups)
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(keys)
            for key in keys:
                indices = list(self.groups[key])
                rng.shuffle(indices)
                yield from indices
        else:
            for key in keys:
                yield from self.groups[key]
