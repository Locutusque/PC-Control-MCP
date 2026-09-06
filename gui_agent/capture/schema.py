"""Recording schema for the capture daemon (plan section 4.1.2).

Records are append-only JSONL next to a per-session video segment.  The daemon
writes them in the hot path, so serialisation stays dumb: dataclasses,
``json.dumps``, no validation frameworks.

Because we control collection we get ground-truth action labels for free --
unlike VPT, which had to train an inverse-dynamics model to recover actions
from unlabelled video.  The price is that this file describes some of the most
sensitive data on the machine; see :mod:`gui_agent.capture.privacy` and
``docs/privacy.md`` before adding a field.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class EventType(str, Enum):
    MOUSE_MOVE = "mouse_move"
    MOUSE_DOWN = "mouse_down"
    MOUSE_UP = "mouse_up"
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    SCROLL = "scroll"
    KEY_DOWN = "key_down"
    KEY_UP = "key_up"
    TEXT = "text"                 # a coalesced run of printable keystrokes
    REDACTED_TEXT = "redacted_text"  # secure field: count only, never content
    APP_SWITCH = "app_switch"
    MARKER = "marker"             # user pressed the label hotkey
    PAUSE = "pause"
    RESUME = "resume"


@dataclass
class InputEvent:
    """One input event, timestamped against the same clock as the frames."""

    type: EventType
    x: int | None = None
    y: int | None = None
    button: str | None = None
    dx: int | None = None
    dy: int | None = None
    key: str | None = None
    text: str | None = None
    # Set instead of ``text`` when the focused field is secure: we record that
    # N characters were typed and nothing about what they were.
    n_chars: int | None = None

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        d["type"] = self.type.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> InputEvent:
        d = dict(d)
        d["type"] = EventType(d["type"])
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class FrameRecord:
    """A frame plus whatever input happened in its interval, plus context.

    ``frame_index`` addresses the encoded video segment; ``frame_path`` is only
    populated in PNG debug mode.  One of the two is always set.
    """

    timestamp: float
    session_id: str
    segment_id: str
    frame_index: int
    event: dict | None = None
    frame_path: str | None = None
    # Foreground-window context (plan 4.1.1).
    app_context: str | None = None
    window_title: str | None = None
    url: str | None = None
    screen_w: int | None = None
    screen_h: int | None = None
    # Privacy state at capture time.
    is_password_field: bool = False
    sensitive_flagged: bool = False
    redaction_reason: str | None = None
    # Values visible on screen that a later field-fill example can copy from
    # (plan 4.1.2).  Populated from the accessibility tree, not from OCR.
    form_data_visible: dict[str, str] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, {}, False) or k in _ALWAYS}

    @classmethod
    def from_dict(cls, d: dict) -> FrameRecord:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def input_event(self) -> InputEvent | None:
        return InputEvent.from_dict(self.event) if self.event else None


_ALWAYS = {"timestamp", "session_id", "segment_id", "frame_index", "schema_version"}


@dataclass
class SegmentMeta:
    """A trajectory: one contiguous run of frames with stable app context.

    The daemon cuts these on app switch, idle gap or explicit marker (plan
    4.1.4) so the stage-1/2 dataset builders consume clean trajectories rather
    than one undifferentiated stream.
    """

    segment_id: str
    session_id: str
    started_at: float
    ended_at: float | None = None
    n_frames: int = 0
    fps: float = 15.0
    video_path: str | None = None
    records_path: str | None = None
    app_context: str | None = None
    cut_reason: str | None = None  # app_switch | idle | max_duration | marker | stop
    # Optional in-the-moment label from the label hotkey (plan 4.1.5).
    user_label: str | None = None
    # Filled in later by hindsight relabelling (plan 4.3).
    hindsight_instruction: str | None = None
    hindsight_confidence: float | None = None
    # Privacy pipeline state: raw -> redacted -> promoted into the pool.
    redaction_status: str = "pending"  # pending | clean | redacted | quarantined
    redaction_findings: list[str] = field(default_factory=list)
    promoted_at: float | None = None
    schema_version: int = SCHEMA_VERSION

    @property
    def duration_s(self) -> float:
        return (self.ended_at or self.started_at) - self.started_at

    @property
    def is_promotable(self) -> bool:
        """Only segments that cleared the redaction sweep enter the pool."""
        return self.redaction_status in ("clean", "redacted")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SegmentMeta:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------
# JSONL io
# --------------------------------------------------------------------------


def write_jsonl(path: str | Path, rows: Iterable[Any], append: bool = False) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonable(row), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                # A daemon killed mid-write leaves one torn final line; that
                # should cost the last frame, not the whole session.
                raise ValueError(f"{path}:{lineno}: malformed JSONL: {exc}") from exc


def read_frames(path: str | Path) -> Iterator[FrameRecord]:
    for row in read_jsonl(path):
        yield FrameRecord.from_dict(row)


def _jsonable(row: Any) -> Any:
    if hasattr(row, "to_dict"):
        return row.to_dict()
    if isinstance(row, Enum):
        return row.value
    return row
