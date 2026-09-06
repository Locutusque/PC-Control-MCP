"""The always-on recording daemon (plan section 4.1).

Passive capture of ordinary usage is the highest-leverage source of volume and
diversity for stage-1 pretraining, and because we control collection the action
labels are ground truth rather than inferred.

The loop, per frame:

1. consult the foreground monitor -- blocked, stale or unknown means pause
   *before* the frame is grabbed, so blocked pixels never reach disk;
2. skip while idle (idle screen time is pure storage waste);
3. grab, encode into the open segment, and write one record per frame;
4. ask the segmenter whether this frame starts a new trajectory.

This is deliberately not a hidden process: a tray indicator is shown whenever
recording is live, and a global hotkey pauses everything.
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CaptureConfig
from .platform_ import PlatformBackend, WindowInfo, get_backend
from .privacy import BlocklistGuard, PrivacyError, SegmentEncryptor
from .schema import EventType, FrameRecord, InputEvent, SegmentMeta, write_jsonl
from .screen import ScreenGrabber, SegmentWriter, VideoUnavailable
from .segmenter import CutReason, Segmenter

log = logging.getLogger(__name__)

__all__ = ["ForegroundMonitor", "CaptureDaemon", "DaemonStats"]

# Foreground state older than this counts as unknown.  On Linux the query
# shells out to xdotool, so it is polled on its own thread rather than in the
# frame loop; if that thread stalls we must not keep recording against a stale
# "not blocked" answer.
_FOREGROUND_MAX_STALE_S = 0.5


class ForegroundMonitor:
    """Polls the foreground window on a background thread.

    Frame capture needs the answer 15 times a second but the query is far too
    slow for that, so it runs here and the loop reads the latest value.  Reads
    older than :data:`_FOREGROUND_MAX_STALE_S` are reported as unknown, which
    the blocklist guard treats as blocked.
    """

    def __init__(self, backend: PlatformBackend, poll_hz: float = 8.0) -> None:
        self.backend = backend
        self.period = 1.0 / poll_hz
        self._info = WindowInfo()
        self._at = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> ForegroundMonitor:
        self._poll()  # prime, so the first frame is not rejected as stale
        self._thread = threading.Thread(target=self._run, name="foreground", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.period):
            try:
                self._poll()
            except Exception as exc:  # pragma: no cover - defensive
                log.debug("foreground poll failed: %s", exc)
                with self._lock:
                    self._info, self._at = WindowInfo(), 0.0

    def _poll(self) -> None:
        info = self.backend.foreground_window()
        with self._lock:
            self._info, self._at = info, time.monotonic()

    def read(self) -> tuple[WindowInfo, bool]:
        """Latest window info plus whether it is fresh enough to trust."""
        with self._lock:
            info, at = self._info, self._at
        return info, (time.monotonic() - at) <= _FOREGROUND_MAX_STALE_S


@dataclass
class DaemonStats:
    frames: int = 0
    segments: int = 0
    blocked_frames: int = 0
    idle_frames: int = 0
    dropped_events: int = 0
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "frames": self.frames,
            "segments": self.segments,
            "blocked_frames": self.blocked_frames,
            "idle_frames": self.idle_frames,
            "dropped_events": self.dropped_events,
            "uptime_s": round(time.time() - self.started_at, 1),
        }


class CaptureDaemon:
    """Owns the capture loop and the lifetime of each segment."""

    def __init__(
        self,
        config: CaptureConfig | None = None,
        backend: PlatformBackend | None = None,
        recorder=None,
        grabber: ScreenGrabber | None = None,
        session_id: str | None = None,
    ) -> None:
        self.config = config or CaptureConfig()
        self.backend = backend or get_backend()
        self.guard = BlocklistGuard(self.config.privacy)
        self.segmenter = Segmenter(self.config)
        self.stats = DaemonStats()
        self.session_id = session_id or f"sess_{int(time.time())}"

        self._recorder = recorder
        self._grabber = grabber
        self._monitor: ForegroundMonitor | None = None
        self._writer: SegmentWriter | None = None
        self._records: list[FrameRecord] = []
        self._segments: list[SegmentMeta] = []
        self._encryptor: SegmentEncryptor | None = None

        self._paused = threading.Event()
        self._stop = threading.Event()
        self._pending_label: str | None = None
        self._marker = False

        self.root = Path(self.config.root) / self.session_id

    # -- control ----------------------------------------------------------
    def pause(self) -> None:
        """Toggled by the global hotkey; closes the open segment cleanly."""
        if not self._paused.is_set():
            log.info("capture paused")
            self._paused.set()
            self._close_segment(CutReason.STOP)

    def resume(self) -> None:
        if self._paused.is_set():
            log.info("capture resumed")
            self._paused.clear()

    def toggle_pause(self) -> None:
        self.resume() if self._paused.is_set() else self.pause()

    def stop(self) -> None:
        self._stop.set()

    def mark(self, label: str | None = None) -> None:
        """Label hotkey: cut here and tag the next segment (plan 4.1.5)."""
        self._marker = True
        self._pending_label = label

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    @property
    def is_recording(self) -> bool:
        return not self._paused.is_set() and not self._stop.is_set()

    # -- preflight --------------------------------------------------------
    def preflight(self) -> list[str]:
        """Problems that must be fixed before recording.

        Refusing to start beats recording without the privacy guarantees the
        config claims to provide.
        """
        problems: list[str] = []
        caps = self.backend.capabilities()
        if not caps["window_info"]:
            problems.append(
                "cannot read the foreground window, so the app/domain blocklist cannot "
                "be enforced (Linux: install xdotool and use X11; macOS: grant "
                "Accessibility permission)"
            )
        if self.config.privacy.redact_password_fields and not caps["secure_field_detection"]:
            log.warning(
                "secure-field detection unavailable; every keystroke will be treated as "
                "secure and typed text will not be recorded"
            )
        if self.config.privacy.encrypt_at_rest:
            try:
                self._encryptor = SegmentEncryptor()
            except PrivacyError as exc:
                problems.append(str(exc))
        try:
            SegmentWriter.ffmpeg_path()
        except VideoUnavailable as exc:
            problems.append(str(exc))
        return problems

    # -- main loop --------------------------------------------------------
    def run(self, max_frames: int | None = None) -> DaemonStats:
        problems = self.preflight()
        if problems:
            raise PrivacyError(
                "refusing to start capture:\n  - " + "\n  - ".join(problems)
            )

        if self._recorder is None:
            from .input_hooks import InputRecorder

            self._recorder = InputRecorder(
                self.config.privacy, move_interval_s=1.0 / self.config.fps
            )
        if self._grabber is None:
            self._grabber = ScreenGrabber(scale=self.config.capture_scale)

        self.root.mkdir(parents=True, exist_ok=True)
        self._monitor = ForegroundMonitor(self.backend).start()
        self._recorder.start()
        self._grabber.open()
        self._install_signal_handlers()

        log.info(
            "capture started: session=%s root=%s input=%dx%d pixels=%dx%d (%.2gx) "
            "encoding=%dx%d",
            self.session_id, self.root, *self._grabber.size, *self._grabber.frame_size,
            self._grabber.pixel_ratio, *self._grabber.output_size,
        )
        try:
            self._loop(max_frames)
        finally:
            self._close_segment(CutReason.STOP)
            self._grabber.close()
            self._recorder.stop()
            self._monitor.stop()
            self._write_manifest()
            log.info("capture stopped: %s", self.stats.as_dict())
        return self.stats

    def _loop(self, max_frames: int | None) -> None:
        period = 1.0 / self.config.fps
        next_at = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic()
            if now < next_at:
                time.sleep(min(next_at - now, period))
                continue
            next_at = max(next_at + period, now - period)  # resync, never burst

            if max_frames is not None and self.stats.frames >= max_frames:
                return
            if self._paused.is_set():
                continue

            info, fresh = self._monitor.read()
            decision = (
                self.guard.check(info.app, info.title, info.url)
                if fresh
                else self.guard.check()  # stale == unknown == blocked
            )
            if decision.blocked:
                self.stats.blocked_frames += 1
                self._close_segment(CutReason.BLOCKED)
                continue

            self._recorder.update_focus(info.is_secure_field)
            events = self._recorder.drain()
            if events:
                self.segmenter.note_input(time.time())

            if self._is_idle(events):
                self.stats.idle_frames += 1
                continue

            self._record_frame(info, events)

    def _is_idle(self, events: list[InputEvent]) -> bool:
        if events:
            return False
        idle = self.backend.idle_seconds()
        if idle is None:
            idle = self._recorder.idle_seconds
        return idle >= self.config.idle_timeout_s

    def _record_frame(self, info: WindowInfo, events: list[InputEvent]) -> None:
        wall = time.time()
        app = info.app or "unknown"
        marker, self._marker = self._marker, False

        reason = self.segmenter.observe(wall, app, marker=marker)
        if reason is not None:
            self._close_segment(reason)
        if self.segmenter.current is None:
            self._open_segment(wall, app)

        frame = self._grabber.grab()
        index = self._writer.write(frame)
        self.segmenter.current.n_frames += 1
        self.stats.frames += 1

        secure = bool(info.is_secure_field)
        base = dict(
            timestamp=wall,
            session_id=self.session_id,
            segment_id=self.segmenter.current.segment_id,
            frame_index=index,
            app_context=app,
            window_title=info.title,
            url=info.url,
            # The *input* coordinate space, which is what mouse events are
            # reported in. Recording the pixel buffer size here would mislabel
            # every click on a HiDPI display by the scale factor, silently.
            screen_w=self._grabber.size[0],
            screen_h=self._grabber.size[1],
            is_password_field=secure,
        )
        # One record per event so the action stream keeps its ordering; a frame
        # with no input still gets a record, because "nothing happened here" is
        # signal the movement-pretraining objective needs.
        if events:
            self._records.extend(
                FrameRecord(event=e.to_dict(), **base)
                for e in events
                if not (secure and e.type is EventType.TEXT)
            )
        else:
            self._records.append(FrameRecord(**base))

        if len(self._records) >= 256:
            self._flush_records()

    # -- segments ---------------------------------------------------------
    def _open_segment(self, now: float, app: str) -> None:
        label, self._pending_label = self._pending_label, None
        seg = self.segmenter.open(now, app, label)
        # Raw dimensions come from a real frame, not the monitor geometry: on a
        # HiDPI display those differ and ffmpeg would desynchronise silently.
        frame_w, frame_h = self._grabber.frame_size
        out_w, out_h = self._grabber.output_size
        self._writer = SegmentWriter(
            self.root / "video" / f"{seg.segment_id}.mkv",
            width=frame_w,
            height=frame_h,
            fps=self.config.fps,
            codec=self.config.video_codec,
            crf=self.config.video_crf,
            output_width=out_w,
            output_height=out_h,
        ).open()
        self.stats.segments += 1
        log.debug("segment %s opened (app=%s)", seg.segment_id, app)

    def _close_segment(self, reason: CutReason) -> None:
        seg = self.segmenter.close()
        if seg is None:
            return
        self._flush_records()
        if self._writer is not None:
            self._writer.close()
            video_path = self._writer.path
            self._writer = None
        else:  # pragma: no cover - defensive
            video_path = None

        records_path = self.root / "records" / f"{seg.segment_id}.jsonl"
        if not self.segmenter.is_useful(seg):
            log.debug("discarding short segment %s (%d frames)", seg.segment_id, seg.n_frames)
            for path in (video_path, records_path):
                if path and path.exists():
                    path.unlink()
            return

        meta = SegmentMeta(
            segment_id=seg.segment_id,
            session_id=self.session_id,
            started_at=seg.started_at,
            ended_at=time.time(),
            n_frames=seg.n_frames,
            fps=self.config.fps,
            video_path=str(video_path.relative_to(self.root)) if video_path else None,
            records_path=str(records_path.relative_to(self.root)),
            app_context=seg.app_context,
            cut_reason=reason.value,
            user_label=seg.user_label,
        )
        if self._encryptor is not None:
            for path in (video_path, records_path):
                if path and path.exists():
                    self._encryptor.encrypt_file(path)
        self._segments.append(meta)
        self._write_manifest()

    def _flush_records(self) -> None:
        if not self._records:
            return
        segment_id = self._records[0].segment_id
        write_jsonl(
            self.root / "records" / f"{segment_id}.jsonl", self._records, append=True
        )
        self._records = []

    def _write_manifest(self) -> None:
        self.stats.dropped_events = getattr(self._recorder, "dropped_events", 0)
        write_jsonl(self.root / "segments.jsonl", self._segments)

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return  # signal handlers can only be set from the main thread
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, lambda *_: self.stop())
            except (ValueError, OSError):  # pragma: no cover
                pass
