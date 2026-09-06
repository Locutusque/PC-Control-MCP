"""Per-OS window, focus and idle queries (plan section 4.1.1).

Each backend wraps the native API for its platform and returns ``None`` for
anything it cannot determine.  Callers must treat ``None`` as "unknown" and
fail closed -- :class:`~gui_agent.capture.privacy.BlocklistGuard` refuses to
record an unattributed window, and
:class:`~gui_agent.capture.privacy.SecureFieldTracker` treats unknown focus as
secure.

Window titles are read but never mined for structure: URLs come from the
accessibility tree or a browser extension, not from scraping whatever the title
bar happens to contain (plan 4.1.1).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

__all__ = ["WindowInfo", "PlatformBackend", "get_backend"]


@dataclass(frozen=True)
class WindowInfo:
    app: str | None = None
    title: str | None = None
    url: str | None = None
    pid: int | None = None
    # None means the accessibility tree could not be consulted, which is
    # different from "the focused field is not a password box".
    is_secure_field: bool | None = None

    @property
    def is_known(self) -> bool:
        return bool(self.app or self.title)


class PlatformBackend:
    """Interface every OS backend implements."""

    name = "base"

    def foreground_window(self) -> WindowInfo:
        return WindowInfo()

    def idle_seconds(self) -> float | None:
        """Seconds since the last user input, or ``None`` if unavailable.

        The daemon falls back to timing its own input hooks when this is
        ``None``, which is accurate enough for idle cut-off.
        """
        return None

    def screen_size(self) -> tuple[int, int] | None:
        return None

    def capabilities(self) -> dict[str, bool]:
        info = self.foreground_window()
        return {
            "window_info": info.is_known,
            "secure_field_detection": info.is_secure_field is not None,
            "idle_detection": self.idle_seconds() is not None,
            "screen_size": self.screen_size() is not None,
        }


# --------------------------------------------------------------------------
# Linux (X11 / Wayland)
# --------------------------------------------------------------------------


class LinuxBackend(PlatformBackend):
    """X11 via xdotool/wmctrl; secure fields via AT-SPI when installed.

    Under Wayland most of this is unavailable by design -- compositors do not
    let arbitrary clients read the foreground window.  ``capabilities()`` will
    show that, and the daemon refuses to start rather than recording without
    the blocklist being enforceable.
    """

    name = "linux"

    def __init__(self) -> None:
        self._xdotool = shutil.which("xdotool")
        self._atspi = self._load_atspi()

    @staticmethod
    def _load_atspi():
        try:
            import gi  # type: ignore

            gi.require_version("Atspi", "2.0")
            from gi.repository import Atspi  # type: ignore

            Atspi.init()
            return Atspi
        except Exception:  # pragma: no cover - depends on desktop stack
            log.debug("AT-SPI unavailable; secure-field detection disabled")
            return None

    def foreground_window(self) -> WindowInfo:
        if not self._xdotool:
            return WindowInfo(is_secure_field=self._secure_field())
        try:
            wid = _run([self._xdotool, "getactivewindow"])
            if wid is None:
                return WindowInfo(is_secure_field=self._secure_field())
            title = _run([self._xdotool, "getwindowname", wid])
            pid_s = _run([self._xdotool, "getwindowpid", wid])
            pid = int(pid_s) if pid_s and pid_s.isdigit() else None
            app = None
            if pid:
                try:
                    app = (
                        subprocess.run(
                            ["ps", "-p", str(pid), "-o", "comm="],
                            capture_output=True, text=True, timeout=1.0,
                        ).stdout.strip()
                        or None
                    )
                except Exception:
                    app = None
            return WindowInfo(app=app, title=title, pid=pid,
                              is_secure_field=self._secure_field())
        except Exception as exc:  # pragma: no cover
            log.debug("foreground_window failed: %s", exc)
            return WindowInfo(is_secure_field=self._secure_field())

    def _secure_field(self) -> bool | None:
        if self._atspi is None:
            return None
        try:  # pragma: no cover - needs a live desktop
            focused = self._atspi.get_desktop(0)
            state = focused.get_state_set()
            return bool(state.contains(self._atspi.StateType.EDITABLE)) and bool(
                state.contains(self._atspi.StateType.PASSWORD_TEXT)
            )
        except Exception:
            return None

    def idle_seconds(self) -> float | None:
        xprintidle = shutil.which("xprintidle")
        if not xprintidle:
            return None
        out = _run([xprintidle])
        return int(out) / 1000.0 if out and out.isdigit() else None

    def screen_size(self) -> tuple[int, int] | None:
        if not self._xdotool:
            return None
        out = _run([self._xdotool, "getdisplaygeometry"])
        if not out:
            return None
        parts = out.split()
        return (int(parts[0]), int(parts[1])) if len(parts) == 2 else None


# --------------------------------------------------------------------------
# macOS
# --------------------------------------------------------------------------


class MacBackend(PlatformBackend):
    """NSWorkspace for the app, AXIsSecureTextField for password fields.

    Requires Accessibility permission (System Settings -> Privacy & Security ->
    Accessibility).  Without it ``is_secure_field`` stays ``None`` and the
    daemon will not record keystroke content.
    """

    name = "darwin"

    def __init__(self) -> None:
        self._ok = self._load()

    def _load(self) -> bool:
        try:  # pragma: no cover - macOS only
            import Quartz  # type: ignore
            from AppKit import NSWorkspace  # type: ignore

            self._NSWorkspace = NSWorkspace
            self._Quartz = Quartz
            return True
        except ImportError:
            log.debug("pyobjc not installed; macOS backend degraded")
            return False

    def foreground_window(self) -> WindowInfo:
        if not self._ok:
            return WindowInfo()
        try:  # pragma: no cover - macOS only
            active = self._NSWorkspace.sharedWorkspace().frontmostApplication()
            app = active.localizedName()
            pid = int(active.processIdentifier())
            title = None
            windows = self._Quartz.CGWindowListCopyWindowInfo(
                self._Quartz.kCGWindowListOptionOnScreenOnly
                | self._Quartz.kCGWindowListExcludeDesktopElements,
                self._Quartz.kCGNullWindowID,
            )
            for w in windows or []:
                if w.get("kCGWindowOwnerPID") == pid and w.get("kCGWindowName"):
                    title = w["kCGWindowName"]
                    break
            return WindowInfo(app=app, title=title, pid=pid,
                              is_secure_field=self._secure_field())
        except Exception as exc:
            log.debug("foreground_window failed: %s", exc)
            return WindowInfo()

    def _secure_field(self) -> bool | None:
        try:  # pragma: no cover - macOS only
            from ApplicationServices import (  # type: ignore
                AXUIElementCopyAttributeValue,
                AXUIElementCreateSystemWide,
            )

            system = AXUIElementCreateSystemWide()
            err, focused = AXUIElementCopyAttributeValue(system, "AXFocusedUIElement", None)
            if err or focused is None:
                return None
            err, role = AXUIElementCopyAttributeValue(focused, "AXRole", None)
            if err:
                return None
            # AXSecureTextField is the role AppKit gives NSSecureTextField.
            return role == "AXSecureTextField"
        except Exception:
            return None

    def idle_seconds(self) -> float | None:
        if not self._ok:
            return None
        try:  # pragma: no cover - macOS only
            return float(
                self._Quartz.CGEventSourceSecondsSinceLastEventType(
                    self._Quartz.kCGEventSourceStateHIDSystemState,
                    self._Quartz.kCGAnyInputEventType,
                )
            )
        except Exception:
            return None

    def screen_size(self) -> tuple[int, int] | None:
        if not self._ok:
            return None
        try:  # pragma: no cover - macOS only
            frame = self._Quartz.CGDisplayBounds(self._Quartz.CGMainDisplayID())
            return int(frame.size.width), int(frame.size.height)
        except Exception:
            return None


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------


class WindowsBackend(PlatformBackend):
    """Win32 GetForegroundWindow + UI Automation ``IsPassword``."""

    name = "win32"

    def __init__(self) -> None:
        self._ok = self._load()

    def _load(self) -> bool:
        try:  # pragma: no cover - Windows only
            import ctypes

            import psutil  # type: ignore

            self._user32 = ctypes.windll.user32
            self._ctypes = ctypes
            self._psutil = psutil
            return True
        except Exception:
            log.debug("win32 backend unavailable")
            return False

    def foreground_window(self) -> WindowInfo:
        if not self._ok:
            return WindowInfo()
        try:  # pragma: no cover - Windows only
            ctypes = self._ctypes
            hwnd = self._user32.GetForegroundWindow()
            if not hwnd:
                return WindowInfo()
            length = self._user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            self._user32.GetWindowTextW(hwnd, buf, length + 1)
            pid = ctypes.c_ulong()
            self._user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            app = None
            try:
                app = self._psutil.Process(pid.value).name()
            except Exception:
                pass
            return WindowInfo(app=app, title=buf.value or None, pid=pid.value or None,
                              is_secure_field=self._secure_field())
        except Exception as exc:
            log.debug("foreground_window failed: %s", exc)
            return WindowInfo()

    def _secure_field(self) -> bool | None:
        try:  # pragma: no cover - Windows only
            import comtypes.client  # type: ignore

            uia = comtypes.client.CreateObject(
                "{ff48dba4-60ef-4201-aa87-54103eef594e}",
                interface=comtypes.gen.UIAutomationClient.IUIAutomation,
            )
            focused = uia.GetFocusedElement()
            return bool(focused.CurrentIsPassword)
        except Exception:
            return None

    def idle_seconds(self) -> float | None:
        if not self._ok:
            return None
        try:  # pragma: no cover - Windows only
            import ctypes

            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_ulong)]

            info = LASTINPUTINFO()
            info.cbSize = ctypes.sizeof(info)
            if not self._user32.GetLastInputInfo(ctypes.byref(info)):
                return None
            millis = ctypes.windll.kernel32.GetTickCount() - info.dwTime
            return millis / 1000.0
        except Exception:
            return None

    def screen_size(self) -> tuple[int, int] | None:
        if not self._ok:
            return None
        try:  # pragma: no cover - Windows only
            return int(self._user32.GetSystemMetrics(0)), int(self._user32.GetSystemMetrics(1))
        except Exception:
            return None


class NullBackend(PlatformBackend):
    """Used in tests and on unsupported platforms; knows nothing, admits it."""

    name = "null"


def _run(cmd: list[str], timeout: float = 1.0) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


_BACKENDS = {"linux": LinuxBackend, "darwin": MacBackend, "win32": WindowsBackend}


def get_backend(platform: str | None = None) -> PlatformBackend:
    """Backend for ``platform`` (defaults to the running OS)."""
    key = platform or sys.platform
    key = "linux" if key.startswith("linux") else key
    cls = _BACKENDS.get(key)
    if cls is None:
        log.warning("no capture backend for platform %r; using NullBackend", key)
        return NullBackend()
    try:
        return cls()
    except Exception as exc:  # pragma: no cover
        log.warning("backend %s failed to initialise (%s); using NullBackend", key, exc)
        return NullBackend()


class InputIdleClock:
    """Idle tracking from our own input hooks, for platforms with no API."""

    def __init__(self) -> None:
        self._last = time.monotonic()

    def touch(self) -> None:
        self._last = time.monotonic()

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last
