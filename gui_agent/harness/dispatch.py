"""Turning actions into operating-system input (plan section 6.2).

Every dispatch goes through :class:`~gui_agent.harness.safety.ActionGuard`
first.  The guard is not a wrapper the caller may skip: :meth:`Dispatcher.send`
consults it and refuses, because the point of putting guardrails at this layer
is that they cannot be bypassed by a policy that has learned something
unexpected.

Backends:

* :class:`PyAutoGuiBackend` -- cross-platform default.
* :class:`XdotoolBackend` -- X11, useful when pyautogui's own X bindings are
  unavailable.
* :class:`NullBackend` -- records intent without touching the desktop; used by
  ``dry_run`` and by every test in this repo.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, field

from ..actions import Action, ActionCodec, ActionType
from ..config import SafetyConfig
from .safety import ActionGuard, Verdict

log = logging.getLogger(__name__)

__all__ = ["DispatchResult", "Dispatcher", "InputBackend", "NullBackend",
           "PyAutoGuiBackend", "XdotoolBackend", "build_backend"]

# Modifier chords are expressed as one key token; this expands them.
_CHORD_KEYS = {
    "CTRL_A": ("ctrl", "a"), "CTRL_C": ("ctrl", "c"), "CTRL_V": ("ctrl", "v"),
    "CTRL_X": ("ctrl", "x"), "CTRL_Z": ("ctrl", "z"), "CTRL_Y": ("ctrl", "y"),
    "CTRL_S": ("ctrl", "s"), "CTRL_F": ("ctrl", "f"), "CTRL_W": ("ctrl", "w"),
    "CTRL_T": ("ctrl", "t"), "CTRL_L": ("ctrl", "l"), "CTRL_R": ("ctrl", "r"),
    "ALT_TAB": ("alt", "tab"), "SHIFT_TAB": ("shift", "tab"),
}

_SINGLE_KEYS = {
    "ENTER": "enter", "TAB": "tab", "ESC": "esc", "BACKSPACE": "backspace",
    "DELETE": "delete", "SPACE": "space", "UP": "up", "DOWN": "down",
    "LEFT": "left", "RIGHT": "right", "HOME": "home", "END": "end",
    "PAGE_UP": "pageup", "PAGE_DOWN": "pagedown",
    "CTRL": "ctrl", "ALT": "alt", "SHIFT": "shift", "META": "win",
    **{f"F{i}": f"f{i}" for i in range(1, 13)},
}


@dataclass
class DispatchResult:
    action: Action
    dispatched: bool
    verdict: Verdict | None = None
    error: str | None = None
    pixels: tuple[int, int] | None = None

    @property
    def blocked(self) -> bool:
        return not self.dispatched and self.verdict is not None and not self.verdict.allowed


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class InputBackend:
    """What a dispatcher needs from the OS."""

    name = "base"

    def screen_size(self) -> tuple[int, int]:
        raise NotImplementedError

    def move(self, x: int, y: int) -> None: ...
    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None: ...
    def move_relative(self, dx: int, dy: int) -> None: ...
    def mouse_down(self, x: int, y: int, button: str = "left") -> None: ...
    def mouse_up(self, x: int, y: int, button: str = "left") -> None: ...
    def scroll(self, dx: int, dy: int) -> None: ...
    def key(self, keys: tuple[str, ...]) -> None: ...
    def type_text(self, text: str) -> None: ...


class NullBackend(InputBackend):
    """Records calls instead of performing them.

    Each method is overridden explicitly rather than caught by ``__getattr__``:
    the base class defines them, so attribute lookup would succeed and the
    fallback would never fire, leaving a dry run that silently records nothing.
    """

    name = "null"

    def __init__(self, screen: tuple[int, int] = (1920, 1080)) -> None:
        self._screen = screen
        self.calls: list[tuple] = []

    def screen_size(self) -> tuple[int, int]:
        return self._screen

    def _record(self, name: str, *args, **kwargs) -> None:
        self.calls.append((name, args, kwargs))

    def move(self, x: int, y: int) -> None:
        self._record("move", x, y)

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        self._record("click", x, y, button, clicks)

    def move_relative(self, dx: int, dy: int) -> None:
        self._record("move_relative", dx, dy)

    def mouse_down(self, x: int, y: int, button: str = "left") -> None:
        self._record("mouse_down", x, y, button)

    def mouse_up(self, x: int, y: int, button: str = "left") -> None:
        self._record("mouse_up", x, y, button)

    def scroll(self, dx: int, dy: int) -> None:
        self._record("scroll", dx, dy)

    def key(self, keys: tuple[str, ...]) -> None:
        self._record("key", keys)

    def type_text(self, text: str) -> None:
        self._record("type_text", text)


class PyAutoGuiBackend(InputBackend):
    """Cross-platform input injection."""

    name = "pyautogui"

    def __init__(self, move_duration: float = 0.0) -> None:
        try:
            import pyautogui  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyautogui is not installed (pip install 'gui-agent[harness]')"
            ) from exc
        # The library's own failsafe (slam the cursor into a corner to abort)
        # is a genuine safety feature when an autonomous policy is driving.
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0  # we pace the loop ourselves
        self.gui = pyautogui
        self.move_duration = move_duration

    def screen_size(self) -> tuple[int, int]:
        size = self.gui.size()
        return int(size[0]), int(size[1])

    def move(self, x: int, y: int) -> None:
        self.gui.moveTo(x, y, duration=self.move_duration)

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        self.gui.click(x=x, y=y, button=button, clicks=clicks,
                       interval=0.05 if clicks > 1 else 0.0)

    def move_relative(self, dx: int, dy: int) -> None:
        self.gui.moveRel(dx, dy, duration=self.move_duration)

    def mouse_down(self, x: int, y: int, button: str = "left") -> None:
        self.gui.moveTo(x, y, duration=self.move_duration)
        self.gui.mouseDown(button=button)

    def mouse_up(self, x: int, y: int, button: str = "left") -> None:
        self.gui.moveTo(x, y, duration=self.move_duration)
        self.gui.mouseUp(button=button)

    def scroll(self, dx: int, dy: int) -> None:
        if dy:
            self.gui.scroll(dy)
        if dx:
            self.gui.hscroll(dx)

    def key(self, keys: tuple[str, ...]) -> None:
        self.gui.hotkey(*keys) if len(keys) > 1 else self.gui.press(keys[0])

    def type_text(self, text: str) -> None:
        self.gui.typewrite(text, interval=0.01)


class XdotoolBackend(InputBackend):
    """X11 via the xdotool CLI."""

    name = "xdotool"

    def __init__(self) -> None:
        self.exe = shutil.which("xdotool")
        if not self.exe:
            raise RuntimeError("xdotool not found on PATH")

    def _run(self, *args: str) -> None:
        subprocess.run([self.exe, *args], check=True, capture_output=True, timeout=5.0)

    def screen_size(self) -> tuple[int, int]:
        out = subprocess.run([self.exe, "getdisplaygeometry"], capture_output=True,
                             text=True, timeout=5.0).stdout.split()
        return int(out[0]), int(out[1])

    def move(self, x: int, y: int) -> None:
        self._run("mousemove", str(x), str(y))

    def click(self, x: int, y: int, button: str = "left", clicks: int = 1) -> None:
        self.move(x, y)
        self._run("click", "--repeat", str(clicks), _X_BUTTONS.get(button, "1"))

    def move_relative(self, dx: int, dy: int) -> None:
        self._run("mousemove_relative", "--", str(dx), str(dy))

    def mouse_down(self, x: int, y: int, button: str = "left") -> None:
        self.move(x, y)
        self._run("mousedown", _X_BUTTONS.get(button, "1"))

    def mouse_up(self, x: int, y: int, button: str = "left") -> None:
        self.move(x, y)
        self._run("mouseup", _X_BUTTONS.get(button, "1"))

    def scroll(self, dx: int, dy: int) -> None:
        for _ in range(abs(dy)):
            self._run("click", "4" if dy > 0 else "5")
        for _ in range(abs(dx)):
            self._run("click", "6" if dx < 0 else "7")

    def key(self, keys: tuple[str, ...]) -> None:
        self._run("key", "+".join(keys))

    def type_text(self, text: str) -> None:
        self._run("type", "--delay", "10", "--", text)


_X_BUTTONS = {"left": "1", "middle": "2", "right": "3"}


def build_backend(name: str = "auto", **kwargs) -> InputBackend:
    if name == "null":
        return NullBackend(**kwargs)
    if name == "xdotool":
        return XdotoolBackend()
    if name == "pyautogui":
        return PyAutoGuiBackend(**kwargs)
    for factory in (PyAutoGuiBackend, XdotoolBackend):
        try:
            return factory()
        except Exception as exc:
            log.debug("%s unavailable: %s", factory.__name__, exc)
    raise RuntimeError(
        "no input backend available; install pyautogui or xdotool, or pass "
        "backend='null' for a dry run"
    )


# --------------------------------------------------------------------------
# Dispatcher
# --------------------------------------------------------------------------


@dataclass
class DragState:
    """A DRAG_START with no DRAG_END yet."""

    x: int
    y: int
    button: str = "left"
    started_at: float = field(default_factory=time.monotonic)


class Dispatcher:
    """Executes guarded actions against an input backend."""

    def __init__(
        self,
        backend: InputBackend | None = None,
        guard: ActionGuard | None = None,
        codec: ActionCodec | None = None,
        config: SafetyConfig | None = None,
    ) -> None:
        self.config = config or SafetyConfig()
        self.backend = backend or (NullBackend() if self.config.dry_run else build_backend())
        self.codec = codec or ActionCodec()
        self.guard = guard or ActionGuard(self.config, self.codec, self.backend.screen_size())
        self.drag: DragState | None = None

    @property
    def screen_size(self) -> tuple[int, int]:
        return self.backend.screen_size()

    def send(
        self,
        action: Action,
        foreground_app: str | None = None,
        form_data: dict | None = None,
    ) -> DispatchResult:
        """Guard, then execute.  The guard cannot be skipped."""
        if action.type.is_terminal:
            # DONE and ESCALATE end the subtask; they never touch the OS.
            return DispatchResult(action, dispatched=False)

        verdict = self.guard.check(action, foreground_app, form_data)
        if not verdict.allowed:
            log.warning("blocked %s: %s (%s)", action.summary(), verdict.reason, verdict.detail)
            return DispatchResult(action, dispatched=False, verdict=verdict)

        try:
            pixels = self._execute(action, form_data)
        except Exception as exc:
            log.exception("dispatch failed for %s", action.summary())
            return DispatchResult(action, dispatched=False, error=str(exc))

        self.guard.note_dispatch(action, foreground_app=foreground_app)
        return DispatchResult(action, dispatched=True, verdict=verdict, pixels=pixels)

    def _execute(self, action: Action, form_data: dict | None) -> tuple[int, int] | None:
        w, h = self.screen_size
        t = action.type

        if t.is_absolute_pointer:
            x, y = self.codec.to_pixels(action, w, h)
            if t is ActionType.MOVE_CLICK:
                self.backend.click(x, y, "left", 1)
            elif t is ActionType.DOUBLE_CLICK:
                self.backend.click(x, y, "left", 2)
            elif t is ActionType.RIGHT_CLICK:
                self.backend.click(x, y, "right", 1)
            elif t is ActionType.DRAG_START:
                self.backend.mouse_down(x, y, "left")
                self.drag = DragState(x, y)
            elif t is ActionType.DRAG_END:
                if self.drag is None:
                    # A drag that never started: press and release so the
                    # gesture is at least well-formed rather than leaving a
                    # button stuck down.
                    log.warning("DRAG_END with no open drag; treating as a click")
                    self.backend.mouse_down(x, y, "left")
                self.backend.mouse_up(x, y, "left")
                self.drag = None
            return (x, y)

        if t is ActionType.MOVE_REL:
            self.backend.move_relative(action.dx, action.dy)
        elif t is ActionType.SCROLL:
            self.backend.scroll(action.scroll_dx, action.scroll_dy)
        elif t is ActionType.KEY:
            self.backend.key(_expand_key(action.key))
        elif t is ActionType.WAIT:
            time.sleep(action.wait_ms / 1000.0)
        elif t is ActionType.TYPE:
            # resolve_text copies form_data verbatim for field-fill, so the
            # value cannot drift between what was supplied and what is typed.
            self.backend.type_text(action.resolve_text(form_data))
        return None

    def release(self) -> None:
        """Release a drag left open by a timeout or an escalation.

        Leaving a mouse button held down would keep affecting the desktop after
        the harness has handed control back.
        """
        if self.drag is None:
            return
        log.warning("releasing drag left open at (%d, %d)", self.drag.x, self.drag.y)
        try:
            self.backend.mouse_up(self.drag.x, self.drag.y, self.drag.button)
        except Exception as exc:  # pragma: no cover
            log.error("could not release drag: %s", exc)
        self.drag = None

    def reset_subtask(self) -> None:
        self.release()
        self.guard.reset_subtask()


def _expand_key(key: str) -> tuple[str, ...]:
    if key in _CHORD_KEYS:
        return _CHORD_KEYS[key]
    mapped = _SINGLE_KEYS.get(key)
    if mapped:
        return (mapped,)
    return (key.lower(),)
