"""Capture schema, privacy layer and segmentation (plan section 4.1)."""

from __future__ import annotations

import pytest

from gui_agent.capture.privacy import (
    BlocklistGuard,
    PrivacyError,
    RedactionScanner,
    SecureFieldTracker,
    promote_segments,
    prune_expired,
)
from gui_agent.capture.schema import (
    EventType,
    FrameRecord,
    InputEvent,
    SegmentMeta,
    read_jsonl,
    write_jsonl,
)
from gui_agent.capture.segmenter import CutReason, Segmenter
from gui_agent.config import CaptureConfig, PrivacyConfig


class TestSchema:
    def test_frame_record_roundtrip(self):
        event = InputEvent(EventType.CLICK, x=10, y=20, button="left")
        record = FrameRecord(1.0, "s", "seg", 3, event=event.to_dict(), app_context="chrome")
        restored = FrameRecord.from_dict(record.to_dict())
        assert restored.frame_index == 3
        assert restored.input_event.type is EventType.CLICK

    def test_jsonl_roundtrip(self, tmp_path):
        rows = [SegmentMeta(f"seg{i}", "s", 0.0, 1.0, 10) for i in range(3)]
        path = tmp_path / "segments.jsonl"
        assert write_jsonl(path, rows) == 3
        assert len(list(read_jsonl(path))) == 3

    def test_torn_final_line_is_reported_with_its_location(self, tmp_path):
        path = tmp_path / "torn.jsonl"
        path.write_text('{"a": 1}\n{"b": ')
        with pytest.raises(ValueError, match="malformed JSONL"):
            list(read_jsonl(path))

    def test_only_swept_segments_are_promotable(self):
        segment = SegmentMeta("seg", "s", 0.0)
        assert not segment.is_promotable      # pending
        segment.redaction_status = "clean"
        assert segment.is_promotable
        segment.redaction_status = "quarantined"
        assert not segment.is_promotable


class TestBlocklist:
    @pytest.fixture
    def guard(self):
        return BlocklistGuard(PrivacyConfig())

    def test_blocked_app_pauses_recording(self, guard):
        assert guard.check(app="1Password 8").blocked

    def test_blocked_domain_pauses_recording(self, guard):
        assert guard.check(app="chrome", url="https://chase.com/login").blocked

    def test_private_browsing_title_pauses_recording(self, guard):
        assert guard.check(app="firefox", window_title="Private Browsing").blocked

    def test_ordinary_window_is_allowed(self, guard):
        assert not guard.check(app="chrome", url="docs.python.org", window_title="Docs").blocked

    def test_unknown_foreground_fails_closed(self, guard):
        # An unattributable window cannot be proven safe, and an unattributed
        # frame is worth less than the risk of recording a password manager.
        decision = guard.check()
        assert decision.blocked and decision.reason == "unknown_foreground"


class TestSecureFields:
    def test_secure_field_content_is_never_kept(self):
        tracker = SecureFieldTracker(PrivacyConfig())
        tracker.update(True)
        text, count = tracker.observe_text("hunter2")
        assert text is None and count == 7
        assert tracker.suppressed_chars == 7

    def test_ordinary_field_content_is_kept(self):
        tracker = SecureFieldTracker(PrivacyConfig())
        tracker.update(False)
        assert tracker.observe_text("hello") == ("hello", 5)

    def test_unknown_focus_is_treated_as_secure_in_strict_mode(self):
        tracker = SecureFieldTracker(PrivacyConfig(), strict=True)
        tracker.update(None)
        assert tracker.is_secure and tracker.state_unknown
        assert tracker.observe_text("secret")[0] is None

    def test_redaction_can_be_disabled_deliberately(self):
        tracker = SecureFieldTracker(PrivacyConfig(redact_password_fields=False))
        tracker.update(True)
        assert tracker.observe_text("hunter2") == ("hunter2", 7)


class TestRedactionScanner:
    @pytest.fixture
    def scanner(self):
        return RedactionScanner(PrivacyConfig())

    def test_finds_a_valid_card_number(self, scanner):
        kinds = {f.kind for f in scanner.scan_text("pay with 4111 1111 1111 1111")}
        assert "credit_card" in kinds

    def test_luhn_rejects_a_plain_digit_run(self, scanner):
        # Long digit runs are common on screen; without the Luhn check every
        # order number would quarantine a segment.
        assert not [f for f in scanner.scan_text("order 1234567890123456") if f.kind == "credit_card"]

    def test_finds_ssn_and_email(self, scanner):
        kinds = {f.kind for f in scanner.scan_text("123-45-6789 j@x.com")}
        assert {"ssn", "email"} <= kinds

    def test_findings_never_carry_the_matched_text(self, scanner):
        finding = scanner.scan_text("123-45-6789")[0]
        assert "6789" not in finding.redacted()

    def test_scans_typed_text_in_records(self, tmp_path, scanner):
        path = tmp_path / "records.jsonl"
        write_jsonl(path, [
            FrameRecord(0.0, "s", "seg", 0,
                        event=InputEvent(EventType.TEXT, text="ssn 123-45-6789").to_dict())
        ])
        assert [f.kind for f in scanner.scan_records(path)] == ["ssn"]


class TestPromotion:
    def _segment(self, tmp_path, text: str) -> SegmentMeta:
        write_jsonl(
            tmp_path / "records" / "seg.jsonl",
            [FrameRecord(0.0, "s", "seg", 0, event=InputEvent(EventType.TEXT, text=text).to_dict())],
        )
        return SegmentMeta("seg", "s", 0.0, 1.0, 10, records_path="records/seg.jsonl")

    def test_clean_segment_is_promoted(self, tmp_path):
        segment = self._segment(tmp_path, "hello world")
        scanner = RedactionScanner(PrivacyConfig(), ocr=lambda image: [])
        promoted, quarantined = promote_segments(
            [segment], scanner, tmp_path, tmp_path / "pool"
        )
        assert len(promoted) == 1 and not quarantined
        assert promoted[0].redaction_status == "clean"

    def test_sensitive_segment_is_quarantined(self, tmp_path):
        segment = self._segment(tmp_path, "ssn 123-45-6789")
        scanner = RedactionScanner(PrivacyConfig(), ocr=lambda image: [])
        promoted, quarantined = promote_segments([segment], scanner, tmp_path, tmp_path / "pool")
        assert not promoted and len(quarantined) == 1
        assert "ssn" in quarantined[0].redaction_findings[0]

    def test_missing_ocr_backend_quarantines_rather_than_promotes(self, tmp_path):
        segment = self._segment(tmp_path, "hello world")
        scanner = RedactionScanner(PrivacyConfig(), ocr=None)
        promoted, quarantined = promote_segments(
            [segment], scanner, tmp_path, tmp_path / "pool", require_ocr=True
        )
        assert not promoted
        assert quarantined[0].redaction_findings == ["ocr_unavailable"]


class TestRetention:
    def test_expired_processed_segments_are_pruned(self, tmp_path):
        video = tmp_path / "video" / "seg.mkv"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"x")
        segment = SegmentMeta("seg", "s", 0.0, video_path="video/seg.mkv", promoted_at=0.0)
        removed = prune_expired([segment], tmp_path, retention_days=14, now=100 * 86400)
        assert removed and not video.exists()

    def test_unpromoted_segments_are_never_pruned(self, tmp_path):
        video = tmp_path / "video" / "seg.mkv"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"x")
        # Silently deleting these would hide a stuck redaction pipeline.
        segment = SegmentMeta("seg", "s", 0.0, video_path="video/seg.mkv", promoted_at=None)
        assert prune_expired([segment], tmp_path, retention_days=1, now=999 * 86400) == []
        assert video.exists()

    def test_dry_run_reports_without_deleting(self, tmp_path):
        video = tmp_path / "video" / "seg.mkv"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"x")
        segment = SegmentMeta("seg", "s", 0.0, video_path="video/seg.mkv", promoted_at=0.0)
        assert prune_expired([segment], tmp_path, 14, now=100 * 86400, dry_run=True)
        assert video.exists()


class TestSegmenter:
    @pytest.fixture
    def segmenter(self):
        return Segmenter(CaptureConfig(segment_idle_gap_s=4.0, segment_max_duration_s=100.0))

    def test_no_cut_while_the_app_is_stable(self, segmenter):
        segmenter.open(0.0, "chrome")
        assert segmenter.observe(1.0, "chrome") is None

    def test_app_switch_cuts(self, segmenter):
        segmenter.open(0.0, "chrome")
        assert segmenter.observe(1.0, "slack") is CutReason.APP_SWITCH

    def test_idle_gap_cuts(self, segmenter):
        segmenter.open(0.0, "chrome")
        assert segmenter.observe(9.0, "chrome") is CutReason.IDLE

    def test_input_resets_the_idle_gap(self, segmenter):
        segmenter.open(0.0, "chrome")
        segmenter.note_input(8.0)
        assert segmenter.observe(9.0, "chrome") is None

    def test_max_duration_cuts(self, segmenter):
        segmenter.open(0.0, "chrome")
        segmenter.note_input(100.0)
        assert segmenter.observe(101.0, "chrome") is CutReason.MAX_DURATION

    def test_marker_and_block_cut(self, segmenter):
        segmenter.open(0.0, "chrome")
        assert segmenter.observe(1.0, "chrome", marker=True) is CutReason.MARKER
        assert segmenter.observe(1.0, "chrome", blocked=True) is CutReason.BLOCKED

    def test_short_segments_are_not_worth_keeping(self, segmenter):
        state = segmenter.open(0.0, "chrome")
        state.n_frames = 2
        assert not segmenter.is_useful(state)
        state.n_frames = 50
        assert segmenter.is_useful(state)


class TestEncryptionFailsClosed:
    def test_missing_key_is_an_error_not_a_silent_plaintext_write(self, monkeypatch):
        from gui_agent.capture.privacy import SegmentEncryptor

        monkeypatch.delenv(SegmentEncryptor.ENV_KEY, raising=False)
        with pytest.raises(PrivacyError, match=SegmentEncryptor.ENV_KEY):
            SegmentEncryptor()
