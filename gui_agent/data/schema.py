"""Training-example schema shared by all three stages (plan sections 4.2-4.4).

One example is one control tick: an observation frame plus the context the
policy sees, and the action atoms it should emit.  Examples are stored as JSONL
next to the capture pool and reference frames by ``(segment_id, frame_index)``
rather than embedding pixels, so a dataset stays small enough to shuffle and
inspect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Iterator

from ..capture.schema import read_jsonl, write_jsonl

__all__ = ["ExampleKind", "TrainingExample", "load_examples", "save_examples"]


class ExampleKind(str, Enum):
    """What the example teaches; also selects the loss weighting."""

    MOVEMENT = "movement"          # stage 1: clicks/scrolls/keys, no instruction
    CONTROL = "control"            # stage 2: instruction-conditioned control tick
    FIELD_FILL = "field_fill"      # TYPE that copies a supplied form_data value
    FREE_COMPOSE = "free_compose"  # TYPE that generates novel text
    TERMINAL = "terminal"          # DONE
    ESCALATE = "escalate"          # stage 3: the policy should have handed back


@dataclass
class TrainingExample:
    """One (observation, context) -> action supervision pair."""

    segment_id: str
    frame_index: int
    target: list[str]                       # action atoms to predict
    kind: ExampleKind = ExampleKind.CONTROL
    # Empty for stage 1: movement pretraining is unconditioned (plan 4.2).
    instruction: str = ""
    form_data: dict[str, str] = field(default_factory=dict)
    # Preceding actions, most recent last, each already serialised to atoms.
    history: list[list[str]] = field(default_factory=list)
    # Where to find the frame.
    video_path: str | None = None
    source_root: str | None = None
    screen_w: int | None = None
    screen_h: int | None = None
    app_context: str | None = None
    # Extra CE weight, applied per-example.  Field-fill spans get more, because
    # copying a supplied value exactly is a correctness requirement rather than
    # a quality one (plan 2.3 / 5).
    loss_weight: float = 1.0
    # Stage 3 provenance: which rollout this correction came from.
    rollout_id: str | None = None
    corrected_from: list[str] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        return {k: v for k, v in d.items() if v not in (None, [], {})} | {
            "segment_id": self.segment_id,
            "frame_index": self.frame_index,
            "target": self.target,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingExample":
        d = dict(d)
        if "kind" in d:
            d["kind"] = ExampleKind(d["kind"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def form_fields(self) -> list[str]:
        """Ordered form_data keys; ``<FIELD_k>`` indexes into this."""
        return list(self.form_data.keys())


def save_examples(path: str | Path, examples, append: bool = False) -> int:
    return write_jsonl(path, examples, append=append)


def load_examples(path: str | Path) -> Iterator[TrainingExample]:
    for row in read_jsonl(path):
        yield TrainingExample.from_dict(row)
