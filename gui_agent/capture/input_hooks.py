"""Global input capture (plan section 4.1.1).

``pynput`` is the cross-platform baseline.  Native hooks (Win32
``SetWindowsHookEx``, macOS ``CGEventTap``, Linux XRecord/evdev) have lower
latency and are the upgrade path, but they need elevated privileges and a
per-OS event loop; put them behind :class:`InputRecorder` rather than teaching
the daemon about two capture stacks.

The recorder does three jobs beyond forwarding events:

* coalesces printable keystrokes into ``TEXT`` events, so a typed field is one
  training example instead of forty;
* routes every keystroke through
  :class:`~gui_agent.capture.privacy.SecureFieldTracker`, so password content
  never enters the queue in the first place;
* collapses the mouse-move firehose to one sample per frame interval.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

from ..config import PrivacyConfig
from .privacy import SecureFieldTracker
from .schema import EventType, InputEvent

log = logging.getLogger(__name__)

__all__ = ["InputRecorder", "InputUnavailable", "KEY_NAME_MAP"]


class InputUnavailable(RuntimeError):
    """pynput is missing, or the OS denied the hook."""


# pynput key names -> our fixed key vocabulary (gui_agent.config.DEFAULT_KEYS).
KEY_NAME_MAP = {
    "enter": "ENTER", "return": "ENTER", "tab": "TAB", "esc": "ESC",
    "backspace": "BACKSPACE", "delete": "DELETE", "space": "SPACE",
    "up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT",
    "home": "HOME", "end": "END", "page_up": "PAGE_UP", "page_down": "PAGE_DOWN",
    "ctrl": "CTRL", "ctrl_l": "CTRL", "ctrl_r": "CTRL",
    "alt": "ALT", "alt_l": "ALT", "alt_r": "ALT", "alt_gr": "ALT",
    "shift": "SHIFT", "shift_l": "SHIFT", "shift_r": "SHIFT",
    "cmd": "META", "cmd_l": "META", "cmd_r": "META", "super": "META",
    **{f"f{i}": f"F{i}" for i in range(1, 13)},
}

# How long a run of printable characters stays open before being flushed as one
# TEXT event.  Long enough to catch normal typing cadence, short enough that a
# pause reads as a separate field.
_TEXT_FLUSH_S = 1.2
_DOUBLE_CLICK_S = 0.4
_DOUBLE_CLICK_PX = 6


@dataclass
class _ClickState:
    t: float = 0.0
    x: int = 0
    y: int = 0
    button: str = ""


class InputRecorder:
    """Collects input events onto a thread-safe queue.

    The daemon drains the queue once per frame, so the listener threads never
    block on disk.  A full queue drops the oldest events and logs it -- a
    stalled writer must not turn into unbounded memory growth on a machine the
    user is trying to work on.
    """

    def __init__(
        self,
        privacy: PrivacyConfig | None = None,
        move_interval_s: float = 1 / 15,
        max_queue: int = 4096,
        strict_secure: bool = True,
    ) -> None:
        self.privacy = privacy or PrivacyConfig()
        self.move_interval_s = move_interval_s
        self.secure = SecureFieldTracker(self.privacy, strict=strict_secure)
        self.events: queue.Queue[InputEvent] = queue.Queue(maxsize=max_queue)

        self._lock = threading.Lock()
        self._text_buf: list[str] = []
        self._text_started = 0.0
        self._text_secure_chars = 0
        self._last_move = 0.0
        self._last_click = _ClickState()
        self._last_input = time.monotonic()
        self._mouse_listener = None
        self._key_listener = None
        self._dropped = 0

    # -- lifecycle --------------------------------------------------------
    def start(self) -> InputRecorder:
        try:
            from pynput import keyboard, mouse  # type: ignore
        except ImportError as exc:
            raise InputUnavailable(
                "pynput is not installed; install it (pip install pynput) to capture "
                "input events"
            ) from exc

        self._pynput_keyboard = keyboard
        self._mouse_listener = mouse.Listener(
            on_move=self._on_move, on_click=self._on_click, on_scroll=self._on_scroll
        )
        self._key_listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self._mouse_listener.start()
        self._key_listener.start()

        # pynput reports failure asynchronously; give the hooks a moment and
        # check, so a denied Accessibility permission surfaces here rather than
        # as a silently empty dataset.
        time.sleep(0.2)
        for listener, kind in ((self._mouse_listener, "mouse"), (self._key_listener, "keyboard")):
            if not listener.running:
                self.stop()
                raise InputUnavailable(
                    f"{kind} hook failed to start -- on macOS grant Accessibility "
                    "permission, on Linux check X11/uinput access"
                )
        return self

    def stop(self) -> None:
        self.flush_text()
        for listener in (self._mouse_listener, self._key_listener):
            if listener is not None:
                listener.stop()
        self._mouse_listener = self._key_listener = None

    def __enter__(self) -> InputRecorder:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- state ------------------------------------------------------------
    def update_focus(self, is_secure: bool | None) -> None:
        """Feed accessibility focus state; flushes on a secure/insecure change."""
        was = self.secure.is_secure
        self.secure.update(is_secure)
        if was != self.secure.is_secure:
            self.flush_text()

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_input

    @property
    def dropped_events(self) -> int:
        return self._dropped

    def drain(self, max_events: int = 512) -> list[InputEvent]:
        """Pop everything queued since the last call."""
        # A long typing run that has gone quiet is flushed here rather than
        # waiting for the next keystroke, which may never come.
        with self._lock:
            stale = self._text_buf and time.monotonic() - self._text_started > _TEXT_FLUSH_S
        if stale:
            self.flush_text()

        out: list[InputEvent] = []
        for _ in range(max_events):
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                break
        return out

    # -- emission ---------------------------------------------------------
    def _emit(self, event: InputEvent) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()  # drop oldest, keep the recent past
                self.events.put_nowait(event)
            except (queue.Empty, queue.Full):  # pragma: no cover - racy drain
                pass
            self._dropped += 1
            if self._dropped % 100 == 1:
                log.warning("input queue full; dropped %d events", self._dropped)

    def _touch(self) -> None:
        self._last_input = time.monotonic()

    # -- mouse ------------------------------------------------------------
    def _on_move(self, x: int, y: int) -> None:
        self._touch()
        now = time.monotonic()
        if now - self._last_move < self.move_interval_s:
            return  # one sample per frame; the raw stream is ~1000Hz
        self._last_move = now
        self._emit(InputEvent(EventType.MOUSE_MOVE, x=int(x), y=int(y)))

    def _on_click(self, x: int, y: int, button, pressed: bool) -> None:
        self._touch()
        self.flush_text()  # a click ends the current typing run
        name = getattr(button, "name", str(button))
        x, y = int(x), int(y)
        if pressed:
            self._emit(InputEvent(EventType.MOUSE_DOWN, x=x, y=y, button=name))
            return

        self._emit(InputEvent(EventType.MOUSE_UP, x=x, y=y, button=name))
        now = time.monotonic()
        prev = self._last_click
        is_double = (
            name == "left"
            and prev.button == name
            and now - prev.t <= _DOUBLE_CLICK_S
            and abs(x - prev.x) <= _DOUBLE_CLICK_PX
            and abs(y - prev.y) <= _DOUBLE_CLICK_PX
        )
        if is_double:
            self._emit(InputEvent(EventType.DOUBLE_CLICK, x=x, y=y, button=name))
            self._last_click = _ClickState()  # a triple click is not two doubles
        else:
            kind = EventType.RIGHT_CLICK if name == "right" else EventType.CLICK
            self._emit(InputEvent(kind, x=x, y=y, button=name))
            self._last_click = _ClickState(now, x, y, name)

    def _on_scroll(self, x: int, y: int, dx: int, dy: int) -> None:
        self._touch()
        self._emit(InputEvent(EventType.SCROLL, x=int(x), y=int(y), dx=int(dx), dy=int(dy)))

    # -- keyboard ---------------------------------------------------------
    def _on_press(self, key) -> None:
        self._touch()
        char = getattr(key, "char", None)
        if char is not None and char.isprintable():
            self._append_text(char)
            return

        name = getattr(key, "name", None)
        if name is None:
            return
        if name == "space":
            self._append_text(" ")
            return
        self.flush_text()
        mapped = KEY_NAME_MAP.get(name)
        if mapped:
            self._emit(InputEvent(EventType.KEY_DOWN, key=mapped))

    def _on_release(self, key) -> None:
        self._touch()

    def _append_text(self, char: str) -> None:
        with self._lock:
            if self.secure.is_secure:
                # Never buffer secure characters: only their count survives.
                self._text_secure_chars += 1
                return
            if not self._text_buf:
                self._text_started = time.monotonic()
            self._text_buf.append(char)

    def flush_text(self) -> None:
        """Close the open typing run and emit it as a single event."""
        with self._lock:
            text = "".join(self._text_buf)
            secure_chars = self._text_secure_chars
            self._text_buf = []
            self._text_secure_chars = 0
        if secure_chars:
            self._emit(InputEvent(EventType.REDACTED_TEXT, n_chars=secure_chars))
        if text:
            self._emit(InputEvent(EventType.TEXT, text=text, n_chars=len(text)))
