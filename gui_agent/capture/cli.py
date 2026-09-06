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
    PrivacyError, RedactionScanner, SegmentEncryptor, default_ocr, promote_segments,
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
    """Report whether this machine can record with the guarantees configured."""
    config = _load_config(args.config)
    backend = get_backend()
    caps = backend.capabilities()
    print(f"platform backend : {backend.name}")
    for name, ok in caps.items():
        print(f"  {name:<24} {'yes' if ok else 'NO'}")
    daemon = CaptureDaemon(config, backend=backend)
    problems = daemon.preflight()
    if problems:
        print("\nblocking problems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nready to record")
    return 0


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


def cmd_promote(args) -> int:
    """Run the redaction sweep and promote clean segments into the pool."""
    config = _load_config(args.config)
    root = Path(config.root) / args.session if args.session else Path(config.root)
    segments = [s for s in _load_segments(root) if s.redaction_status == "pending"]
    if not segments:
        print("nothing to promote")
        return 0

    scanner = RedactionScanner(config.privacy, ocr=default_ocr() if config.privacy.ocr_redaction else None)
    promoted, quarantined = promote_segments(
        segments, scanner, root, config.pool_root, require_ocr=config.privacy.ocr_redaction
    )
    by_id = {s.segment_id: s for s in promoted + quarantined}
    updated = [by_id.get(s.segment_id, s) for s in _load_segments(root)]
    write_jsonl(root / "segments.jsonl", updated)

    print(f"promoted {len(promoted)}, quarantined {len(quarantined)}")
    for seg in quarantined:
        print(f"  {seg.segment_id}: {', '.join(seg.redaction_findings)}")
    return 0


def cmd_prune(args) -> int:
    """Delete raw segments past the retention window (plan 4.1.3)."""
    config = _load_config(args.config)
    root = Path(config.root)
    removed: list[str] = []
    for session_dir in sorted(p for p in root.glob("*") if p.is_dir()):
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
    for session_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        segments = _load_segments(session_dir)
        frames = sum(s.n_frames for s in segments)
        hours = sum(s.duration_s for s in segments) / 3600
        total += frames
        for seg in segments:
            counts[seg.redaction_status] = counts.get(seg.redaction_status, 0) + 1
            by_status += 1
        print(f"{session_dir.name}: {len(segments)} segments, {frames} frames, {hours:.2f}h")
    print(f"\ntotal frames: {total}")
    print("redaction status: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
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
