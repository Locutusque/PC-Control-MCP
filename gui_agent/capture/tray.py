"""Recording indicator and global hotkeys (plan section 4.1.1).

The daemon must never be a silent background process.  ``pystray`` gives a real
tray icon when it is installed; when it is not, we fall back to a periodic
console banner rather than showing nothing -- degrading to invisible would
defeat the point.
"""

from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)

__all__ = ["Indicator", "HotkeyListener", "run_with_indicator"]

_RECORDING = (0xE0, 0x3B, 0x3B)  # red
_PAUSED = (0x9A, 0x9A, 0x9A)     # grey


class Indicator:
    """Tray icon showing whether capture is live."""

    def __init__(self, daemon, title: str = "GUI Agent Capture") -> None:
        self.daemon = daemon
        self.title = title
        self._icon = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> Indicator:
        try:
            import pystray  # type: ignore
            from PIL import Image, ImageDraw  # type: ignore
        except ImportError:
            log.warning(
                "pystray/Pillow not installed; falling back to a console recording "
                "banner. Recording is never silent."
            )
            self._thread = threading.Thread(target=self._console, name="indicator", daemon=True)
            self._thread.start()
            return self

        def image(color):
            img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
            ImageDraw.Draw(img).ellipse((8, 8, 56, 56), fill=color + (255,))
            return img

        menu = pystray.Menu(
            pystray.MenuItem(
                lambda _: "Pause" if self.daemon.is_recording else "Resume",
                lambda: self.daemon.toggle_pause(),
            ),
            pystray.MenuItem("Mark task…", lambda: self.daemon.mark()),
            pystray.MenuItem("Quit", lambda: self.daemon.stop()),
        )
        self._icon = pystray.Icon("gui-agent", image(_RECORDING), self.title, menu)

        def refresh():
            while not self._stop.wait(1.0):
                if self._icon is None:
                    return
                self._icon.icon = image(_RECORDING if self.daemon.is_recording else _PAUSED)
                self._icon.title = (
                    f"{self.title} — {'recording' if self.daemon.is_recording else 'paused'}"
                )

        threading.Thread(target=refresh, name="indicator-refresh", daemon=True).start()
        self._thread = threading.Thread(target=self._icon.run, name="indicator", daemon=True)
        self._thread.start()
        return self

    def _console(self) -> None:
        while not self._stop.wait(10.0):
            state = "RECORDING" if self.daemon.is_recording else "paused"
            print(f"[gui-agent capture] {state} — {self.daemon.stats.as_dict()}", flush=True)

    def stop(self) -> None:
        self._stop.set()
        if self._icon is not None:
            self._icon.stop()
            self._icon = None


class HotkeyListener:
    """Global pause and label hotkeys."""

    def __init__(self, daemon, pause_hotkey: str, label_hotkey: str | None = None) -> None:
        self.daemon = daemon
        self.pause_hotkey = pause_hotkey
        self.label_hotkey = label_hotkey
        self._listener = None

    def start(self) -> HotkeyListener:
        try:
            from pynput import keyboard  # type: ignore
        except ImportError:
            log.warning("pynput not installed; global hotkeys unavailable")
            return self
        bindings = {self.pause_hotkey: self.daemon.toggle_pause}
        if self.label_hotkey:
            bindings[self.label_hotkey] = self._prompt_label
        try:
            self._listener = keyboard.GlobalHotKeys(bindings)
            self._listener.start()
        except Exception as exc:  # pragma: no cover - depends on desktop stack
            log.warning("could not register hotkeys: %s", exc)
        return self

    def _prompt_label(self) -> None:
        """Optional, skippable one-line 'what am I about to do' (plan 4.1.5).

        Runs off-thread so the hotkey handler never blocks input delivery.  Even
        sparse coverage gives a calibration set for hindsight relabelling.
        """

        def ask():
            try:
                label = input("\n[gui-agent] What are you about to do? (blank to skip) ").strip()
            except (EOFError, KeyboardInterrupt):
                label = ""
            self.daemon.mark(label or None)

        threading.Thread(target=ask, name="label-prompt", daemon=True).start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None


def run_with_indicator(daemon, config=None, max_frames: int | None = None):
    """Run ``daemon`` with the tray indicator and hotkeys attached."""
    config = config or daemon.config
    indicator = Indicator(daemon).start() if config.tray_indicator else None
    hotkeys = HotkeyListener(daemon, config.pause_hotkey, config.label_hotkey).start()
    try:
        return daemon.run(max_frames=max_frames)
    finally:
        hotkeys.stop()
        if indicator is not None:
            indicator.stop()
        time.sleep(0.1)
