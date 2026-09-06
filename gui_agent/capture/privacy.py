"""Privacy and redaction for the capture daemon (plan section 4.1.3).

The daemon records everything the user does, so this layer is not a post-hoc
filter that somebody might forget to run -- it is checked *before* each frame
is written, and again before any segment is promoted into the training pool.

Three defences, in order of how early they fire:

1. :class:`BlocklistGuard` -- pauses recording entirely while a blocked app,
   domain or window title owns the foreground.  Evaluated before the frame is
   grabbed, so blocked pixels never reach disk.
2. :class:`SecureFieldTracker` -- keystrokes into a field the accessibility
   tree marks secure are recorded as a character *count* and nothing else.
3. :class:`RedactionScanner` / :func:`promote_segments` -- an asynchronous
   OCR + regex sweep that a segment must clear before it is promoted from raw
   capture into the training pool.  Deliberately out of the hot path.

Everything here fails closed: if a check cannot be performed (no accessibility
permission, no OCR backend, no encryption library) the daemon refuses to record
rather than recording unprotected.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..config import PrivacyConfig
from .schema import SegmentMeta, read_jsonl, write_jsonl

log = logging.getLogger(__name__)

__all__ = [
    "PrivacyError",
    "BlocklistDecision",
    "BlocklistGuard",
    "SecureFieldTracker",
    "Finding",
    "RedactionScanner",
    "SegmentEncryptor",
    "promote_segments",
    "prune_expired",
]


class PrivacyError(RuntimeError):
    """A privacy guarantee could not be upheld.  Never swallow this."""


# --------------------------------------------------------------------------
# 1. Foreground blocklist
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BlocklistDecision:
    blocked: bool
    reason: str | None = None
    matched: str | None = None

    def __bool__(self) -> bool:
        return self.blocked


class BlocklistGuard:
    """Decides whether recording is permitted for a foreground window.

    Checked *before* capturing each frame (plan 4.1.3), which is the whole
    point: a password manager's contents must never land on disk, not even for
    the milliseconds before a filter runs.
    """

    def __init__(self, config: PrivacyConfig | None = None) -> None:
        self.config = config or PrivacyConfig()
        self._app = _lowered(self.config.app_blocklist)
        self._domain = _lowered(self.config.domain_blocklist)
        self._title = _lowered(self.config.title_blocklist)

    def check(
        self,
        app: str | None = None,
        window_title: str | None = None,
        url: str | None = None,
    ) -> BlocklistDecision:
        # Unknown foreground window: fail closed.  We cannot prove the window
        # is safe, and an unattributed frame is worth less than the risk.
        if not app and not window_title and not url:
            return BlocklistDecision(True, "unknown_foreground")

        for value, patterns, reason in (
            (app, self._app, "blocked_app"),
            (url, self._domain, "blocked_domain"),
            (window_title, self._title, "blocked_title"),
        ):
            if not value:
                continue
            hay = value.lower()
            for pat in patterns:
                if pat in hay:
                    return BlocklistDecision(True, reason, pat)
        return BlocklistDecision(False)


def _lowered(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(s.lower() for s in items)


# --------------------------------------------------------------------------
# 2. Secure text fields
# --------------------------------------------------------------------------


class SecureFieldTracker:
    """Suppresses keystroke content while a secure text field has focus.

    Secure-field state comes from the platform accessibility tree
    (``AXIsSecureTextField`` on macOS, UI Automation ``IsPassword`` on Windows,
    AT-SPI ``PASSWORD`` state on Linux) -- see
    :mod:`gui_agent.capture.platform_`.

    If ``strict`` is set and the platform cannot answer, every keystroke is
    treated as secure.  That loses typing data on machines without
    accessibility permission, which is the correct trade.
    """

    def __init__(self, config: PrivacyConfig | None = None, strict: bool = True) -> None:
        self.config = config or PrivacyConfig()
        self.strict = strict
        self._secure = False
        self._unknown = False
        self._suppressed_chars = 0

    def update(self, is_secure: bool | None) -> None:
        """Feed the current focus state.  ``None`` means 'could not determine'."""
        if is_secure is None:
            self._unknown = True
            self._secure = self.strict
        else:
            self._unknown = False
            self._secure = bool(is_secure)

    @property
    def is_secure(self) -> bool:
        return self.config.redact_password_fields and self._secure

    @property
    def state_unknown(self) -> bool:
        return self._unknown

    def observe_text(self, text: str) -> tuple[str | None, int]:
        """Returns ``(text_to_record, n_chars)``.

        Inside a secure field the text is dropped and only the count survives,
        so the trajectory keeps its shape without leaking the password.
        """
        if self.is_secure:
            self._suppressed_chars += len(text)
            return None, len(text)
        return text, len(text)

    @property
    def suppressed_chars(self) -> int:
        return self._suppressed_chars


# --------------------------------------------------------------------------
# 3. Asynchronous redaction sweep
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Finding:
    """A sensitive-looking match, with the region to blur if it came from OCR."""

    kind: str
    text: str
    frame_index: int | None = None
    bbox: tuple[int, int, int, int] | None = None

    def redacted(self) -> str:
        """The finding as it is safe to write into a log or metadata file."""
        return f"{self.kind}:{len(self.text)}chars"


class RedactionScanner:
    """Regex sweep over text pulled off captured frames.

    Runs as a batch job over newly captured segments, never synchronously in
    the capture path -- OCR at 15Hz would eat the frame budget whole.

    ``ocr`` is any callable taking an image and returning
    ``[(text, bbox), ...]``; :func:`default_ocr` wires up Tesseract when it is
    installed.  With no OCR backend the scanner still checks recorded *typed*
    text, and marks frames unverified so :func:`promote_segments` can refuse to
    promote them.
    """

    def __init__(
        self,
        config: PrivacyConfig | None = None,
        ocr: Callable[[object], Sequence[tuple[str, tuple[int, int, int, int]]]] | None = None,
    ) -> None:
        self.config = config or PrivacyConfig()
        self.ocr = ocr
        self._patterns = [
            (kind, re.compile(pattern)) for kind, pattern in self.config.redaction_patterns
        ]

    @property
    def has_ocr(self) -> bool:
        return self.ocr is not None

    def scan_text(self, text: str, frame_index: int | None = None) -> list[Finding]:
        out: list[Finding] = []
        for kind, rx in self._patterns:
            for m in rx.finditer(text):
                matched = m.group(0)
                if kind == "credit_card" and not _luhn(matched):
                    continue  # digit runs are common on screen; Luhn cuts the noise
                out.append(Finding(kind, matched, frame_index))
        return out

    def scan_frame(self, image, frame_index: int | None = None) -> list[Finding]:
        """OCR one frame and scan the recognised text."""
        if self.ocr is None:
            return []
        findings: list[Finding] = []
        for text, bbox in self.ocr(image):
            for f in self.scan_text(text, frame_index):
                findings.append(Finding(f.kind, f.text, frame_index, bbox))
        return findings

    def scan_video(
        self,
        video_path: str | Path,
        max_frames: int = 200,
        change_threshold: int = 6,
    ) -> list[Finding]:
        """OCR the recorded frames and scan what is visible on screen.

        Nothing else in the pipeline reads the video, so without this the
        sweep only ever saw text the user *typed* -- a card number rendered in
        a web page went straight into the training pool.

        Frames are deduplicated by perceptual hash before OCR rather than
        sampled every Nth. At 15Hz consecutive frames are near-identical, so
        deduplication cuts a 300-frame segment to a few dozen distinct screens
        while still examining every screen that actually appeared; blind
        sampling would skip whole screens that happened to fall between
        samples.

        This is still not a proof: ``max_frames`` caps the work, and OCR
        misses text it cannot read. It is a filter, not a guarantee.
        """
        if self.ocr is None:
            return []
        from .screen import frame_hash, hamming, probe_video_size, read_segment_frames

        size = probe_video_size(video_path)
        if size is None:
            log.warning("could not determine the size of %s; skipping frame OCR", video_path)
            return []
        try:
            frames = read_segment_frames(video_path, size[0], size[1])
        except Exception as exc:
            log.warning("could not decode %s for the redaction sweep: %s", video_path, exc)
            raise

        findings: list[Finding] = []
        last_hash: int | None = None
        scanned = 0
        for index, frame in enumerate(frames):
            digest = frame_hash(frame)
            if last_hash is not None and hamming(digest, last_hash) <= change_threshold:
                continue
            last_hash = digest
            findings.extend(self.scan_frame(frame, index))
            scanned += 1
            if scanned >= max_frames:
                log.warning(
                    "stopped the frame sweep of %s at %d distinct frames (max_frames); "
                    "later frames were not examined", video_path, scanned,
                )
                break
        log.info("OCR swept %d distinct frames of %s", scanned, video_path)
        return findings

    def scan_records(self, records_path: str | Path) -> list[Finding]:
        """Scan the typed text recorded alongside a segment."""
        findings: list[Finding] = []
        for row in read_jsonl(records_path):
            event = row.get("event") or {}
            text = event.get("text")
            if text:
                findings.extend(self.scan_text(text, row.get("frame_index")))
            for value in (row.get("form_data_visible") or {}).values():
                findings.extend(self.scan_text(str(value), row.get("frame_index")))
        return findings


def default_ocr():
    """Tesseract-backed OCR, or ``None`` when it is unavailable.

    Checks that the tesseract *binary* actually runs, not merely that the
    Python wrapper imports. ``pip install pytesseract`` does not install
    tesseract itself, and reporting OCR as available when every call would
    raise is the worst of the three outcomes: the sweep's gate passes, no text
    is ever read, and segments are promoted as clean having been scanned by
    nothing.
    """
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # noqa: F401
    except ImportError:
        log.warning("pytesseract not installed; OCR redaction sweep unavailable")
        return None

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        log.warning(
            "pytesseract is installed but the tesseract binary is not usable (%s). "
            "Install it (macOS: brew install tesseract, Debian/Ubuntu: apt install "
            "tesseract-ocr); until then segments cannot clear the redaction sweep.",
            exc,
        )
        return None

    def _ocr(image):
        import numpy as np

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        out = []
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            out.append(
                (text, (data["left"][i], data["top"][i], data["width"][i], data["height"][i]))
            )
        return out

    return _ocr


def _luhn(candidate: str) -> bool:
    digits = [int(c) for c in candidate if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# --------------------------------------------------------------------------
# Encryption at rest
# --------------------------------------------------------------------------


class SegmentEncryptor:
    """Fernet encryption for captured segments (plan 4.1.3: encrypt at rest).

    Constructing this without the ``cryptography`` package raises rather than
    falling back to plaintext -- a capture archive quietly written in the clear
    is the exact failure this setting exists to prevent.
    """

    ENV_KEY = "GUI_AGENT_CAPTURE_KEY"

    def __init__(self, key: bytes | None = None) -> None:
        # Catch broadly: a half-installed cryptography (missing _cffi_backend,
        # mismatched Rust bindings) raises things that are not ImportError, and
        # every one of them must land as a fail-closed PrivacyError rather than
        # a crash the caller might paper over.
        try:
            from cryptography.fernet import Fernet  # type: ignore
        except Exception as exc:
            raise PrivacyError(
                "encrypt_at_rest is enabled but the 'cryptography' package is not "
                f"usable ({type(exc).__name__}: {exc}). Install or repair it "
                "(pip install --force-reinstall cryptography) or explicitly set "
                "privacy.encrypt_at_rest=false to accept plaintext capture storage."
            ) from exc
        self._Fernet = Fernet
        key = key or self._key_from_env()
        self._f = Fernet(key)

    @classmethod
    def _key_from_env(cls) -> bytes:
        key = os.environ.get(cls.ENV_KEY)
        if not key:
            raise PrivacyError(
                f"{cls.ENV_KEY} is not set. Generate one with "
                "`python -m gui_agent.capture.cli keygen` and store it in your OS "
                "keychain; the capture archive cannot be encrypted without it."
            )
        return key.encode()

    @staticmethod
    def generate_key() -> str:
        try:
            from cryptography.fernet import Fernet  # type: ignore
        except Exception as exc:
            raise PrivacyError(f"cryptography is unusable: {exc}") from exc
        return Fernet.generate_key().decode()

    def encrypt_file(self, path: str | Path, remove_plaintext: bool = True) -> Path:
        path = Path(path)
        out = path.with_suffix(path.suffix + ".enc")
        out.write_bytes(self._f.encrypt(path.read_bytes()))
        if remove_plaintext:
            path.unlink()
        return out

    def decrypt_file(self, path: str | Path, dest: str | Path | None = None) -> Path:
        path = Path(path)
        dest = Path(dest) if dest else path.with_suffix("")
        dest.write_bytes(self._f.decrypt(path.read_bytes()))
        return dest


# --------------------------------------------------------------------------
# Promotion and retention
# --------------------------------------------------------------------------


def promote_segments(
    segments: Sequence[SegmentMeta],
    scanner: RedactionScanner,
    raw_root: str | Path,
    pool_root: str | Path,
    require_ocr: bool = True,
    now: float | None = None,
    encryptor: SegmentEncryptor | None = None,
    max_ocr_frames: int = 200,
    change_threshold: int = 6,
) -> tuple[list[SegmentMeta], list[SegmentMeta]]:
    """Run the redaction sweep and promote segments that clear it.

    Returns ``(promoted, quarantined)``.  A segment whose findings cannot be
    localised to a region -- typed text, or any finding at all when no OCR
    backend is available -- is quarantined rather than promoted, because we
    cannot prove which pixels to blur.
    """
    now = now if now is not None else time.time()
    raw_root, pool_root = Path(raw_root), Path(pool_root)
    promoted: list[SegmentMeta] = []
    quarantined: list[SegmentMeta] = []

    def quarantine(seg: SegmentMeta, reasons: list[str]) -> None:
        seg.redaction_status = "quarantined"
        seg.redaction_findings = reasons
        quarantined.append(seg)

    for seg in segments:
        if not seg.records_path:
            quarantine(seg, ["missing_records"])
            continue

        if require_ocr and not scanner.has_ocr:
            quarantine(seg, ["ocr_unavailable"])
            continue

        try:
            # Segments are encrypted at rest by default, which deletes the
            # plaintext. Reading the ciphertext path directly used to find
            # nothing, scan nothing, and promote the segment as clean -- so a
            # missing or unreadable file is now a quarantine, never a pass.
            with _readable(raw_root / seg.records_path, encryptor) as records:
                if records is None:
                    quarantine(seg, ["records_unreadable"])
                    continue
                findings = scanner.scan_records(records)

            if scanner.has_ocr and seg.video_path:
                with _readable(raw_root / seg.video_path, encryptor) as video:
                    if video is None:
                        quarantine(seg, ["video_unreadable"])
                        continue
                    findings += scanner.scan_video(
                        video, max_frames=max_ocr_frames, change_threshold=change_threshold
                    )
        except PrivacyError:
            raise
        except Exception as exc:
            # Never let a failed scan read as a clean one.
            log.warning("redaction sweep failed for %s: %s", seg.segment_id, exc)
            quarantine(seg, [f"scan_failed:{type(exc).__name__}"])
            continue

        if findings:
            # Typed-text findings have no bbox, so blurring cannot fix them.
            quarantine(seg, sorted({f.redacted() for f in findings}))
            continue

        seg.redaction_status = "clean"
        seg.promoted_at = now
        (pool_root / "segments").mkdir(parents=True, exist_ok=True)
        write_jsonl(pool_root / "segments" / f"{seg.segment_id}.json", [seg])
        promoted.append(seg)

    return promoted, quarantined


@contextmanager
def _readable(path: Path, encryptor: SegmentEncryptor | None):
    """Yield a readable plaintext path for ``path``, or ``None``.

    Handles the encrypted-at-rest case by decrypting to a temporary file that
    is removed on exit, so the sweep never leaves a decrypted copy of a
    capture segment lying around after it finishes.
    """
    if path.exists():
        yield path
        return

    ciphertext = path.with_suffix(path.suffix + ".enc")
    if not ciphertext.exists():
        yield None
        return
    if encryptor is None:
        raise PrivacyError(
            f"{ciphertext.name} is encrypted but no key was supplied, so the redaction "
            f"sweep cannot read it. Set {SegmentEncryptor.ENV_KEY} and try again -- "
            "promoting an unscanned segment is not an option."
        )

    with tempfile.TemporaryDirectory(prefix="gui-agent-sweep-") as tmp:
        destination = Path(tmp) / path.name
        encryptor.decrypt_file(ciphertext, destination)
        yield destination


def prune_expired(
    segments: Sequence[SegmentMeta],
    raw_root: str | Path,
    retention_days: int,
    now: float | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Delete raw segments processed more than ``retention_days`` ago.

    Retention is deliberate (plan 4.1.3): without it the sensitive archive
    grows forever.  Only *processed* segments expire -- an unpromoted segment
    is never silently discarded, since that would hide a stuck pipeline.
    """
    now = now if now is not None else time.time()
    cutoff = now - retention_days * 86400
    raw_root = Path(raw_root)
    removed: list[str] = []

    for seg in segments:
        if seg.promoted_at is None or seg.promoted_at > cutoff:
            continue
        for rel in (seg.video_path, seg.records_path):
            if not rel:
                continue
            path = raw_root / rel
            for candidate in (path, path.with_suffix(path.suffix + ".enc")):
                if candidate.exists():
                    if not dry_run:
                        candidate.unlink()
                    removed.append(str(candidate))
    return removed
