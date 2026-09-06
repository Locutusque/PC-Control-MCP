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
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["ScreenGrabber", "SegmentWriter", "VideoUnavailable", "read_segment_frames",
           "probe_video_size", "frame_hash", "hamming"]


_BYTES_PER_PIXEL = {"bgra": 4, "rgba": 4, "rgb24": 3, "bgr24": 3}


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
        self._frame_size: tuple[int, int] | None = None

    def open(self) -> ScreenGrabber:
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
        self._frame_size = None

    def __enter__(self) -> ScreenGrabber:
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def size(self) -> tuple[int, int]:
        """The **input coordinate space**, in points.

        This is the space mouse events are reported in, so it -- not the pixel
        buffer size -- is what click coordinates must be grounded against.  On a
        HiDPI display (any Retina Mac, and fractional scaling on Linux) the two
        differ, and grounding against the wrong one silently mislabels every
        click by the scale factor.
        """
        if self._monitor is None:
            raise VideoUnavailable("grabber is not open")
        return self._monitor["width"], self._monitor["height"]

    @property
    def frame_size(self) -> tuple[int, int]:
        """The **pixel buffer** size, probed from a real grab.

        mss reports monitor geometry in points but returns a backing buffer at
        the display's true pixel density, so on a 2x Retina panel a frame
        carries four times the bytes the monitor dict implies.  ffmpeg is fed
        raw bytes with no header to correct it, so this must come from an
        actual frame rather than from the monitor geometry.
        """
        if self._frame_size is None:
            shot = self.grab()
            self._frame_size = (shot.width, shot.height)
        return self._frame_size

    @property
    def pixel_ratio(self) -> float:
        """Pixels per point. 2.0 on a Retina display, 1.0 on a plain one."""
        return self.frame_size[0] / max(1, self.size[0])

    @property
    def output_size(self) -> tuple[int, int]:
        """Encoded size after ``scale``, with even dimensions for ffmpeg."""
        w, h = self.frame_size
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
        output_width: int | None = None,
        output_height: int | None = None,
    ) -> None:
        self.path = Path(path)
        # width/height describe the raw bytes arriving on stdin and must match
        # the frames exactly; output_width/height are what gets encoded, so
        # downscaling happens in ffmpeg rather than in the capture hot path.
        self.width, self.height, self.fps = width, height, fps
        self.output_width = output_width or width
        self.output_height = output_height or height
        self.codec, self.crf, self.pix_fmt_in = codec, crf, pix_fmt_in
        self._proc: subprocess.Popen | None = None
        self.n_frames = 0

    @property
    def is_scaling(self) -> bool:
        return (self.output_width, self.output_height) != (self.width, self.height)

    @staticmethod
    def ffmpeg_path() -> str:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise VideoUnavailable(
                "ffmpeg not found on PATH; it is required to encode capture segments"
            )
        return exe

    def open(self) -> SegmentWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.ffmpeg_path(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", self.pix_fmt_in,
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
        ]
        if self.is_scaling:
            cmd += ["-vf", f"scale={self.output_width}:{self.output_height}:flags=area"]
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
        expected = self.width * self.height * _BYTES_PER_PIXEL[self.pix_fmt_in]
        if len(frame.data) != expected:
            # ffmpeg reads a headerless byte stream, so a size mismatch does not
            # error -- it silently desynchronises and every later frame is
            # garbage. Catch it on the first frame instead.
            raise VideoUnavailable(
                f"frame is {len(frame.data)} bytes ({frame.width}x{frame.height}) but the "
                f"encoder was opened for {self.width}x{self.height} ({expected} bytes). "
                "On a HiDPI display the pixel buffer is larger than the monitor "
                "geometry; open the writer with ScreenGrabber.frame_size."
            )
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

    def __enter__(self) -> SegmentWriter:
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


def probe_video_size(path: str | Path) -> tuple[int, int] | None:
    """The stored dimensions of a segment, via ffprobe.

    The redaction sweep needs to decode at native resolution: forcing an
    arbitrary width distorts the aspect ratio and downscaling loses exactly
    the small text OCR is there to catch.
    """
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        width, height = (int(v) for v in out.split("x")[:2])
        return width, height
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def frame_hash(frame, size: int = 16) -> int:
    """Difference hash of a frame.

    Perceptual rather than exact, so a blinking text caret or one pixel of
    antialiasing does not read as "the screen changed".
    """
    import numpy as np

    array = np.asarray(frame)
    if array.ndim == 3:
        array = array[:, :, :3].mean(axis=2)
    h, w = array.shape[:2]
    rows = np.linspace(0, h - 1, size).astype(int)
    cols = np.linspace(0, w - 1, size + 1).astype(int)
    small = array[np.ix_(rows, cols)]
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
