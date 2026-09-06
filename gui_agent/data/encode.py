"""Captured events -> action token sequences.

This is where ground-truth labels are made.  A captured trajectory is a stream
of raw input events; the policy predicts one :class:`~gui_agent.actions.Action`
per control tick, so the events have to be collapsed into that form:

* a press/release pair that moved far enough is a drag, not a click;
* a run of printable keystrokes is one TYPE action;
* a TYPE whose text matches a value visible on screen is *field-fill*, and
  everything else is *free-compose* -- the split plan section 4.3 requires to
  exist in the data rather than only in the training script;
* gaps between actions become explicit WAIT actions, which is how the policy
  learns to let a page finish loading instead of clicking into a stale frame;
* every trajectory ends in DONE, which is the only supervision that token gets.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..actions import Action, ActionCodec, ActionType
from ..capture.schema import EventType, FrameRecord
from .schema import ExampleKind

log = logging.getLogger(__name__)

__all__ = ["EncodeConfig", "TimedAction", "TrajectoryEncoder"]


@dataclass(frozen=True)
class EncodeConfig:
    # A press/release separated by more than this is a drag, not a click.
    drag_min_px: int = 8
    # Standalone cursor motion below this is noise, not a deliberate move.
    min_move_px: int = 48
    # Emit MOVE_REL for hovering/scanning motion.  This is the "cursor physics"
    # signal stage-1 pretraining is for (plan 4.2); turn it off to train a
    # purely click-driven policy.
    emit_standalone_moves: bool = True
    # Gaps longer than this become an explicit WAIT action.
    wait_threshold_s: float = 0.4
    # Ignore gaps longer than this: the user went for coffee, they did not
    # decide to wait.
    max_wait_s: float = 5.0
    # Free-compose targets longer than this are dropped rather than truncated;
    # a half-typed target teaches the model to stop mid-word.
    max_type_chars: int = 512
    # Minimum length before a typed string is considered for field-fill
    # matching, so "a" does not match every value on screen.
    min_field_match_chars: int = 3
    append_done: bool = True


@dataclass
class TimedAction:
    """One action with the frame it should be predicted from."""

    frame_index: int
    timestamp: float
    action: Action
    kind: ExampleKind = ExampleKind.CONTROL
    form_data: dict[str, str] = field(default_factory=dict)
    app_context: str | None = None
    screen_w: int | None = None
    screen_h: int | None = None


@dataclass
class _Pending:
    """An open mouse press, waiting to resolve into a click or a drag."""

    x: int
    y: int
    button: str
    frame_index: int
    timestamp: float


class TrajectoryEncoder:
    """Turns one segment's :class:`FrameRecord` stream into actions."""

    def __init__(self, codec: ActionCodec | None = None, config: EncodeConfig | None = None) -> None:
        self.codec = codec or ActionCodec()
        self.config = config or EncodeConfig()

    def encode_segment(self, records: Iterable[FrameRecord]) -> list[TimedAction]:
        cfg = self.config
        out: list[TimedAction] = []
        pending: _Pending | None = None
        suppress_click = False
        last_action_ts: float | None = None
        last_move: tuple[int, int] | None = None
        screen: tuple[int, int] | None = None

        def emit(action: Action, rec: FrameRecord, kind: ExampleKind = ExampleKind.CONTROL) -> None:
            nonlocal last_action_ts
            out.append(
                TimedAction(
                    frame_index=rec.frame_index,
                    timestamp=rec.timestamp,
                    action=action,
                    kind=kind,
                    form_data=dict(rec.form_data_visible or {}),
                    app_context=rec.app_context,
                    screen_w=screen[0] if screen else None,
                    screen_h=screen[1] if screen else None,
                )
            )
            last_action_ts = rec.timestamp

        for rec in records:
            if rec.screen_w and rec.screen_h:
                screen = (rec.screen_w, rec.screen_h)
            event = rec.input_event
            if event is None:
                continue
            if screen is None and event.x is not None:
                log.debug("skipping event before any screen size was recorded")
                continue

            # A gap since the last action becomes an explicit WAIT, so the
            # policy can learn to pause rather than hammer a loading page.
            if last_action_ts is not None and event.type in _ACTION_EVENTS:
                gap = rec.timestamp - last_action_ts
                if cfg.wait_threshold_s <= gap <= cfg.max_wait_s:
                    emit(
                        Action(ActionType.WAIT, wait_ms=self.codec.bucket_wait(gap * 1000)),
                        rec,
                    )

            if event.type is EventType.MOUSE_DOWN:
                pending = _Pending(event.x, event.y, event.button or "left",
                                   rec.frame_index, rec.timestamp)

            elif event.type is EventType.MOUSE_UP:
                if pending is not None and _dist(pending.x, pending.y, event.x, event.y) >= cfg.drag_min_px:
                    emit(self._point(ActionType.DRAG_START, pending.x, pending.y, screen), rec)
                    emit(self._point(ActionType.DRAG_END, event.x, event.y, screen), rec)
                    suppress_click = True  # the synthetic CLICK that follows is the same gesture
                pending = None

            elif event.type in (EventType.CLICK, EventType.DOUBLE_CLICK, EventType.RIGHT_CLICK):
                if suppress_click:
                    suppress_click = False
                else:
                    emit(self._point(_CLICK_TYPES[event.type], event.x, event.y, screen), rec)
                last_move = (event.x, event.y)

            elif event.type is EventType.MOUSE_MOVE:
                if not cfg.emit_standalone_moves:
                    continue
                if last_move is None:
                    last_move = (event.x, event.y)
                    continue
                dx, dy = event.x - last_move[0], event.y - last_move[1]
                if max(abs(dx), abs(dy)) < cfg.min_move_px:
                    continue
                emit(
                    Action(
                        ActionType.MOVE_REL,
                        dx=self.codec.bucket_delta(dx),
                        dy=self.codec.bucket_delta(dy),
                    ),
                    rec,
                )
                last_move = (event.x, event.y)

            elif event.type is EventType.SCROLL:
                emit(
                    Action(
                        ActionType.SCROLL,
                        scroll_dx=self.codec.bucket_scroll(event.dx or 0),
                        scroll_dy=self.codec.bucket_scroll(event.dy or 0),
                    ),
                    rec,
                )

            elif event.type is EventType.KEY_DOWN and event.key:
                if event.key in self.codec.config.keys:
                    emit(Action(ActionType.KEY, key=event.key), rec)

            elif event.type is EventType.TEXT and event.text:
                typed = self._type_action(event.text, rec)
                if typed is not None:
                    action, kind = typed
                    emit(action, rec, kind)

            # REDACTED_TEXT deliberately produces no action: we know a password
            # was typed and refuse to reconstruct it, so the trajectory keeps a
            # hole rather than a guess.

        if out and self.config.append_done:
            last = out[-1]
            out.append(
                TimedAction(
                    frame_index=last.frame_index,
                    timestamp=last.timestamp,
                    action=Action(ActionType.DONE),
                    kind=ExampleKind.TERMINAL,
                    form_data=dict(last.form_data),
                    app_context=last.app_context,
                    screen_w=last.screen_w,
                    screen_h=last.screen_h,
                )
            )
        return out

    # -- helpers ----------------------------------------------------------
    def _point(self, atype: ActionType, x: int, y: int, screen: tuple[int, int] | None) -> Action:
        if screen is None:
            raise ValueError("cannot ground a pointer action without a screen size")
        return self.codec.click(x, y, screen[0], screen[1], type=atype)

    def _type_action(self, text: str, rec: FrameRecord) -> tuple[Action, ExampleKind] | None:
        """Route a typed string to field-fill or free-compose (plan 2.3)."""
        if len(text) > self.config.max_type_chars:
            log.debug("dropping over-long typed span (%d chars)", len(text))
            return None
        field_name = self._match_field(text, rec.form_data_visible or {})
        if field_name is not None:
            return Action(ActionType.TYPE, field=field_name), ExampleKind.FIELD_FILL
        return Action(ActionType.TYPE, text=text), ExampleKind.FREE_COMPOSE

    def _match_field(self, text: str, form_data: dict) -> str | None:
        """The form_data key this text copies, if any.

        Exact match only.  A fuzzy match here would mislabel free-compose as
        field-fill and teach the model to copy when it should generate, which
        is exactly the routing failure section 2.3 warns about.
        """
        if len(text) < self.config.min_field_match_chars:
            return None
        stripped = text.strip()
        for key, value in form_data.items():
            if stripped and stripped == str(value).strip():
                return key
        return None

    def with_history(
        self, actions: Sequence[TimedAction], max_history: int
    ) -> list[list[list[str]]]:
        """Rolling window of the previous actions, serialised to atoms."""
        histories: list[list[list[str]]] = []
        encoded: list[list[str]] = []
        for ta in actions:
            histories.append([list(h) for h in encoded[-max_history:]])
            try:
                encoded.append(self.codec.encode(ta.action, list(ta.form_data.keys())))
            except Exception as exc:  # a bad label must not poison the window
                log.debug("could not encode %s for history: %s", ta.action.summary(), exc)
                encoded.append([])
        return histories


_CLICK_TYPES = {
    EventType.CLICK: ActionType.MOVE_CLICK,
    EventType.DOUBLE_CLICK: ActionType.DOUBLE_CLICK,
    EventType.RIGHT_CLICK: ActionType.RIGHT_CLICK,
}

_ACTION_EVENTS = frozenset(
    {
        EventType.CLICK, EventType.DOUBLE_CLICK, EventType.RIGHT_CLICK,
        EventType.SCROLL, EventType.KEY_DOWN, EventType.TEXT, EventType.MOUSE_DOWN,
    }
)


def _dist(x0: int, y0: int, x1: int, y1: int) -> float:
    return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
