"""Command line for the capture daemon and the privacy pipeline.

    python -m gui_agent.capture.cli keygen
    python -m gui_agent.capture.cli record --config capture.json
    python -m gui_agent.capture.cli promote --session sess_123
    python -m gui_agent.capture.cli prune --dry-run
    python -m gui_agent.capture.cli doctor
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ..config import CaptureConfig
from .daemon import CaptureDaemon
from .platform_ import get_backend
from .privacy import (
    ENVIRONMENTAL_QUARANTINE_REASONS,
    PrivacyError,
    RedactionScanner,
    SegmentEncryptor,
    default_ocr,
    is_retryable_quarantine,
    promote_segments,
    prune_expired,
)
from .schema import SegmentMeta, read_jsonl, write_jsonl
from .tray import run_with_indicator


def _load_config(path: str | None) -> CaptureConfig:
    return CaptureConfig.load(path) if path else CaptureConfig()


def _load_segments(root: Path) -> list[SegmentMeta]:
    manifest = root / "segments.jsonl"
    if not manifest.exists():
        return []
    return [SegmentMeta.from_dict(row) for row in read_jsonl(manifest)]


def cmd_keygen(args) -> int:
    print(SegmentEncryptor.generate_key())
    print(
        f"\nExport this as {SegmentEncryptor.ENV_KEY} and store it in your OS keychain.\n"
        "Losing it makes every encrypted capture segment unreadable.",
        file=sys.stderr,
    )
    return 0


def cmd_doctor(args) -> int:
    """Report whether this machine can record with the guarantees configured.

    Deliberately verbose: most of what can go wrong here is a permission that
    was never granted, and the symptom is a dataset that looks fine but is
    empty or mislabelled rather than an error.
    """
    config = _load_config(args.config)
    backend = get_backend()
    caps = backend.capabilities()

    print(f"platform backend : {backend.name}")
    for name, ok in caps.items():
        print(f"  {name:<24} {'yes' if ok else 'NO'}")

    warnings: list[str] = []
    print("\nscreen capture")
    geometry = _probe_screen(config, warnings)
    print("\ninput hooks")
    _probe_input(config, warnings)

    daemon = CaptureDaemon(config, backend=backend)
    problems = daemon.preflight()

    if not caps["secure_field_detection"] and config.privacy.redact_password_fields:
        warnings.append(
            "secure-field detection is unavailable, so every keystroke is treated as "
            "secure and NO typed text will be recorded. The capture is still useful for "
            "stage 1 (movement), but stage 2 will have no field-fill or free-compose "
            "examples at all."
        )

    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print(f"  - {w}")
    if problems:
        print("\nblocking problems:")
        for p in problems:
            print(f"  - {p}")
    _print_platform_help(backend.name, caps, geometry)

    if problems:
        return 1
    print("\nready to record" + (" (with the warnings above)" if warnings else ""))
    return 0


def _probe_screen(config: CaptureConfig, warnings: list[str]) -> dict | None:
    """Open the grabber and take one real frame."""
    from ..capture.screen import ScreenGrabber, VideoUnavailable

    try:
        grabber = ScreenGrabber(scale=config.capture_scale).open()
    except VideoUnavailable as exc:
        print(f"  unavailable: {exc}")
        return None
    try:
        frame = grabber.grab()
        points, pixels = grabber.size, grabber.frame_size
        output = grabber.output_size
        print(f"  input coordinate space   {points[0]}x{points[1]} (points)")
        print(f"  captured pixel buffer    {pixels[0]}x{pixels[1]}")
        print(f"  pixel ratio              {grabber.pixel_ratio:.2g}x")
        print(f"  encoded at               {output[0]}x{output[1]} "
              f"(capture_scale={config.capture_scale})")
        raw = frame.data
        # Sampled rather than scanned: a full 4K frame comparison per byte is
        # slow and adds nothing over a sample this large.
        sample = raw[: min(len(raw), 400_000)]
        if len(set(sample[::997])) <= 1:
            warnings.append(
                "the captured frame is a single flat colour. On macOS this usually "
                "means Screen Recording permission has not been granted; on Wayland it "
                "means the compositor refused the capture."
            )
        if grabber.pixel_ratio != 1.0:
            print(f"  note: HiDPI display. Clicks are grounded against the "
                  f"{points[0]}x{points[1]} point space, video is encoded from the "
                  f"{pixels[0]}x{pixels[1]} buffer.")
        return {"points": points, "pixels": pixels, "ratio": grabber.pixel_ratio}
    finally:
        grabber.close()


def _probe_input(config: CaptureConfig, warnings: list[str]) -> None:
    """Start the input hooks briefly to see whether the OS allows them."""
    from ..capture.input_hooks import InputRecorder, InputUnavailable

    recorder = InputRecorder(config.privacy)
    try:
        recorder.start()
    except InputUnavailable as exc:
        print(f"  unavailable: {exc}")
        warnings.append(f"input capture will not work: {exc}")
        return
    finally:
        try:
            recorder.stop()
        except Exception:
            pass
    print("  keyboard and mouse hooks started successfully")


def _print_platform_help(backend_name: str, caps: dict, geometry: dict | None) -> None:
    if backend_name != "darwin":
        return
    print(
        "\nmacOS permissions (System Settings -> Privacy & Security):\n"
        "  Accessibility     -> your terminal, or the Python binary. Required for the\n"
        "                       keyboard/mouse hooks and for password-field detection.\n"
        "  Screen Recording  -> same application. Required for frame capture; macOS\n"
        "                       returns a desktop-only image rather than an error when\n"
        "                       this is missing, so it can look like it is working.\n"
        "  After granting either one, fully quit and reopen the terminal. macOS does\n"
        "  not apply the change to an already-running process."
    )


def cmd_record(args) -> int:
    config = _load_config(args.config)
    daemon = CaptureDaemon(config)
    try:
        stats = run_with_indicator(daemon, config, max_frames=args.max_frames)
    except PrivacyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(stats.as_dict(), indent=2))
    return 0


def _session_dirs(config: CaptureConfig, session: str | None) -> list[Path]:
    """The session directories to operate on.

    The daemon writes each session's manifest to ``<root>/<session_id>/``, so
    the root itself never holds one -- reading it directly finds nothing and
    silently reports there is no work to do.
    """
    root = Path(config.root)
    if session:
        return [root / session]
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*") if p.is_dir())


def cmd_promote(args) -> int:
    """Run the redaction sweep and promote clean segments into the pool."""
    config = _load_config(args.config)
    session_dirs = _session_dirs(config, args.session)

    scanner = RedactionScanner(
        config.privacy, ocr=default_ocr() if config.privacy.ocr_redaction else None
    )
    if config.privacy.ocr_redaction and not scanner.has_ocr:
        print(
            "warning: OCR isn't available, so nothing can clear the sweep. Install the\n"
            "         tesseract binary (macOS: brew install tesseract), or set\n"
            "         privacy.ocr_redaction=false to scan typed text only.\n"
        )

    encryptor = None
    if config.privacy.encrypt_at_rest:
        try:
            encryptor = SegmentEncryptor()
        except PrivacyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    total_promoted = total_quarantined = considered = retryable_skipped = 0
    for session_dir in session_dirs:
        all_segments = _load_segments(session_dir)
        segments = [s for s in all_segments if s.redaction_status == "pending"]
        if args.retry:
            # Re-examine segments quarantined because the sweep could not run,
            # now that whatever blocked it may be fixed.
            segments += [s for s in all_segments if is_retryable_quarantine(s)]
        else:
            retryable_skipped += sum(1 for s in all_segments if is_retryable_quarantine(s))
        if not segments:
            continue
        considered += len(segments)

        promoted, quarantined = promote_segments(
            segments, scanner, session_dir, config.pool_root,
            require_ocr=config.privacy.ocr_redaction, encryptor=encryptor,
        )
        by_id = {s.segment_id: s for s in promoted + quarantined}
        updated = [by_id.get(s.segment_id, s) for s in _load_segments(session_dir)]
        write_jsonl(session_dir / "segments.jsonl", updated)

        total_promoted += len(promoted)
        total_quarantined += len(quarantined)
        for seg in quarantined:
            print(f"  {session_dir.name}/{seg.segment_id}: {', '.join(seg.redaction_findings)}")

    if not considered:
        print("nothing to promote")
        if retryable_skipped:
            # The failure this message used to hide: a segment quarantined
            # because OCR was missing looks identical to having no work at all.
            print(
                f"\n{retryable_skipped} segment(s) are quarantined because the sweep "
                "could not run (missing OCR, or no key), not because anything was\n"
                "found in them. Re-examine those with:\n\n"
                "    python -m gui_agent.capture.cli promote --retry\n"
            )
        return 0
    print(f"promoted {total_promoted}, quarantined {total_quarantined}")
    return 0


def cmd_prune(args) -> int:
    """Delete raw segments past the retention window (plan 4.1.3)."""
    config = _load_config(args.config)
    removed: list[str] = []
    for session_dir in _session_dirs(config, None):
        removed += prune_expired(
            _load_segments(session_dir), session_dir,
            config.privacy.raw_retention_days, dry_run=args.dry_run,
        )
    verb = "would remove" if args.dry_run else "removed"
    print(f"{verb} {len(removed)} files")
    for path in removed:
        print(f"  {path}")
    return 0


def cmd_status(args) -> int:
    config = _load_config(args.config)
    root = Path(config.root)
    if not root.exists():
        print(f"no capture root at {root}")
        return 0
    total = by_status = 0
    counts: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for session_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        segments = _load_segments(session_dir)
        frames = sum(s.n_frames for s in segments)
        hours = sum(s.duration_s for s in segments) / 3600
        total += frames
        for seg in segments:
            counts[seg.redaction_status] = counts.get(seg.redaction_status, 0) + 1
            by_status += 1
            for reason in seg.redaction_findings:
                reasons[reason] = reasons.get(reason, 0) + 1
        print(f"{session_dir.name}: {len(segments)} segments, {frames} frames, {hours:.2f}h")
    print(f"\ntotal frames: {total}")
    print("redaction status: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if reasons:
        print("quarantined because: " + ", ".join(f"{k} ({v})" for k, v in sorted(reasons.items())))
        if any(r in ENVIRONMENTAL_QUARANTINE_REASONS or r.startswith("scan_failed:")
               for r in reasons):
            print("  -> some of these mean the sweep could not run, not that something was "
                  "found.\n     Retry them with: promote --retry")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gui-agent-capture", description=__doc__)
    parser.add_argument("--config", help="path to a CaptureConfig JSON file")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("keygen", help="generate a capture encryption key").set_defaults(func=cmd_keygen)
    sub.add_parser("doctor", help="check this machine can record safely").set_defaults(func=cmd_doctor)

    p = sub.add_parser("record", help="run the capture daemon")
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("promote", help="redaction sweep + promote into the training pool")
    p.add_argument("--session", default=None)
    p.add_argument(
        "--retry", action="store_true",
        help="also re-examine segments quarantined because the sweep could not run "
             "(missing OCR, unreadable files). Content findings are never retried.",
    )
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("prune", help="delete raw segments past the retention window")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_prune)

    sub.add_parser("status", help="summarise captured data").set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
