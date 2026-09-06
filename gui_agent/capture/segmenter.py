"""Cutting the continuous stream into trajectories (plan section 4.1.4).

The daemon records one long stream; training wants discrete trajectories.  Cuts
happen on an app switch, an idle gap over threshold, an explicit user marker, or
a hard duration cap, and the boundaries are stored with the segment metadata so
the stage-1/2 dataset builders never see one undifferentiated stream.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import Enum

from ..config import CaptureConfig

__all__ = ["CutReason", "Segmenter", "SegmentState"]


class CutReason(str, Enum):
    APP_SWITCH = "app_switch"
    IDLE = "idle"
    MAX_DURATION = "max_duration"
    MARKER = "marker"
    BLOCKED = "blocked"
    STOP = "stop"


@dataclass
class SegmentState:
    segment_id: str
    started_at: float
    app_context: str | None
    n_frames: int = 0
    user_label: str | None = None


class Segmenter:
    """Decides when the current trajectory ends and the next begins."""

    def __init__(self, config: CaptureConfig | None = None) -> None:
        self.config = config or CaptureConfig()
        self.current: SegmentState | None = None
        self._last_input_at: float | None = None

    def _new_id(self) -> str:
        return f"seg_{uuid.uuid4().hex[:12]}"

    def open(self, now: float, app_context: str | None, label: str | None = None) -> SegmentState:
        self.current = SegmentState(self._new_id(), now, app_context, user_label=label)
        self._last_input_at = now
        return self.current

    def note_input(self, now: float) -> None:
        self._last_input_at = now

    def observe(
        self,
        now: float,
        app_context: str | None,
        marker: bool = False,
        blocked: bool = False,
    ) -> CutReason | None:
        """Reason to cut before recording this frame, or ``None`` to continue."""
        seg = self.current
        if seg is None:
            return None
        if blocked:
            return CutReason.BLOCKED
        if marker:
            return CutReason.MARKER
        if self.config.segment_on_app_switch and app_context != seg.app_context:
            return CutReason.APP_SWITCH
        if now - seg.started_at >= self.config.segment_max_duration_s:
            return CutReason.MAX_DURATION
        if (
            self._last_input_at is not None
            and now - self._last_input_at >= self.config.segment_idle_gap_s
        ):
            return CutReason.IDLE
        return None

    def close(self) -> SegmentState | None:
        seg, self.current = self.current, None
        return seg

    def is_useful(self, seg: SegmentState, min_frames: int = 5) -> bool:
        """Whether a finished segment is worth keeping.

        A two-frame stub from a rapid app switch costs storage and teaches
        nothing.
        """
        return seg.n_frames >= min_frames
