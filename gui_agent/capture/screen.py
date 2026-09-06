"""Screen capture and segment encoding (plan section 4.1.1).

Frames are captured at a fixed rate matching the target control-tick rate, so
frame/action alignment downstream is an index rather than an interpolation
problem, and streamed straight into a compressed video segment via ffmpeg.
Screenshots are low-entropy and compress extremely well; keeping raw PNGs for
hours of continuous recording a day is not viable.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

log = logging.getLogger(__name__)

__all__ = ["ScreenGrabber", "SegmentWriter", "VideoUnavailable", "read_segment_frames"]


class VideoUnavailable(RuntimeError):
    """ffmpeg or a capture backend is missing."""


@dataclass
class Frame:
    data: bytes           # raw BGRA/RGB bytes
    width: int
    height: int
    timestamp: float
    channels: int = 4


class ScreenGrabber:
    """Fixed-rate screen capture.

    ``mss`` is the cross-platform baseline.  Platform-native paths (Windows
    Desktop Duplication, macOS ScreenCaptureKit, PipeWire on Wayland) are worth
    it for throughput, but only once the pipeline is proven -- swap the backend
    behind this interface rather than threading a second one through the
    daemon.
    """

    def __init__(self, monitor: int = 1, scale: float = 1.0) -> None:
        self.monitor_index = monitor
        self.scale = scale
        self._sct = None
        self._monitor = None

    def open(self) -> "ScreenGrabber":
        try:
            import mss  # type: ignore
        except ImportError as exc:
            raise VideoUnavailable(
                "mss is not installed; install it (pip install mss) or supply a "
                "platform-native grabber"
            ) from exc
        self._sct = mss.mss()
        monitors = self._sct.monitors
        if self.monitor_index >= len(monitors):
            raise VideoUnavailable(
                f"monitor {self.monitor_index} not found (have {len(monitors) - 1})"
            )
        self._monitor = monitors[self.monitor_index]
        return self

    def close(self) -> None:
        if self._sct is not None:
            self._sct.close()
            self._sct = None

    def __enter__(self) -> "ScreenGrabber":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def size(self) -> tuple[int, int]:
        if self._monitor is None:
            raise VideoUnavailable("grabber is not open")
        return self._monitor["width"], self._monitor["height"]

    @property
    def output_size(self) -> tuple[int, int]:
        w, h = self.size
        # ffmpeg's yuv420p path needs even dimensions.
        return max(2, int(w * self.scale) // 2 * 2), max(2, int(h * self.scale) // 2 * 2)

    def grab(self) -> Frame:
        if self._sct is None or self._monitor is None:
            raise VideoUnavailable("grabber is not open")
        shot = self._sct.grab(self._monitor)
        return Frame(bytes(shot.raw), shot.width, shot.height, time.time())

    def stream(self, fps: float, stop=lambda: False) -> Iterator[Frame]:
        """Yield frames at ``fps``, skipping rather than accumulating lag."""
        period = 1.0 / fps
        next_at = time.monotonic()
        while not stop():
            now = time.monotonic()
            if now < next_at:
                time.sleep(next_at - now)
            yield self.grab()
            next_at += period
            if next_at < time.monotonic():  # fell behind: resync, don't burst
                next_at = time.monotonic() + period


class SegmentWriter:
    """Pipes raw frames into one ffmpeg process per segment.

    Defaults to FFV1 (mathematically lossless): UI screenshots have hard edges
    and flat regions that lossy codecs smear exactly where small click targets
    live.  ``libx264`` with a low CRF is the near-lossless fallback when disk
    is the binding constraint.
    """

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        codec: str = "ffv1",
        crf: int = 18,
        pix_fmt_in: str = "bgra",
    ) -> None:
        self.path = Path(path)
        self.width, self.height, self.fps = width, height, fps
        self.codec, self.crf, self.pix_fmt_in = codec, crf, pix_fmt_in
        self._proc: subprocess.Popen | None = None
        self.n_frames = 0

    @staticmethod
    def ffmpeg_path() -> str:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise VideoUnavailable(
                "ffmpeg not found on PATH; it is required to encode capture segments"
            )
        return exe

    def open(self) -> "SegmentWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_path(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", self.pix_fmt_in,
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
        ]
        if self.codec == "ffv1":
            cmd += ["-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "bgr0"]
        else:
            cmd += ["-c:v", self.codec, "-crf", str(self.crf),
                    "-preset", "veryfast", "-pix_fmt", "yuv420p"]
        cmd.append(str(self.path))
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def write(self, frame: Frame) -> int:
        """Append a frame; returns its index within the segment."""
        if self._proc is None or self._proc.stdin is None:
            raise VideoUnavailable("segment writer is not open")
        try:
            self._proc.stdin.write(frame.data)
        except BrokenPipeError as exc:
            stderr = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            raise VideoUnavailable(f"ffmpeg died mid-segment: {stderr.strip()}") from exc
        index = self.n_frames
        self.n_frames += 1
        return index

    def close(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        if proc.stdin:
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass
        if proc.wait(timeout=30) != 0:
            stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
            log.error("ffmpeg exited %s: %s", proc.returncode, stderr.strip())

    def __enter__(self) -> "SegmentWriter":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


def read_segment_frames(path: str | Path, width: int, height: int,
                        indices: list[int] | None = None):
    """Decode a segment back to numpy RGB arrays.

    ``indices`` selects specific frames; the datasets use it to pull the single
    frame a training example needs without decoding the whole segment.
    """
    import numpy as np

    exe = SegmentWriter.ffmpeg_path()
    select = ""
    if indices is not None:
        if not indices:
            return np.empty((0, height, width, 3), dtype=np.uint8)
        expr = "+".join(f"eq(n\\,{i})" for i in sorted(set(indices)))
        select = f"select='{expr}',"
    cmd = [exe, "-loglevel", "error", "-i", str(path),
           "-vf", f"{select}scale={width}:{height}",
           "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    frame_bytes = width * height * 3
    if len(out) % frame_bytes:
        raise VideoUnavailable(
            f"decoded {len(out)} bytes, not a multiple of {frame_bytes}; "
            "width/height do not match the segment"
        )
    return np.frombuffer(out, dtype=np.uint8).reshape(-1, height, width, 3)
