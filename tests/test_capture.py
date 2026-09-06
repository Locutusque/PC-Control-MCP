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


class TestFrameSweep:
    """The sweep must actually read the video (plan 4.1.3).

    Before this existed, `promote_segments` checked that an OCR backend was
    *available* and then never used it: only typed text was ever scanned, so a
    card number rendered in a web page went into the training pool untouched.
    """

    @staticmethod
    def _frames(n_distinct: int, repeats: int = 5):
        """`repeats` identical copies of each of `n_distinct` screens.

        Noise rather than flat fills: a difference hash compares adjacent
        pixels, so uniform images collapse to the same hash and would make
        this test pass or fail for reasons unrelated to the deduplication
        being tested.
        """
        import numpy as np

        out = []
        for i in range(n_distinct):
            frame = np.random.RandomState(i).randint(0, 255, (32, 32, 3), dtype=np.uint8)
            out.extend([frame] * repeats)
        return out

    def _patched_scanner(self, monkeypatch, frames, ocr_text="clean"):
        seen = []

        def fake_ocr(image):
            seen.append(image)
            return [(ocr_text, (0, 0, 10, 10))]

        monkeypatch.setattr(
            "gui_agent.capture.screen.probe_video_size", lambda path: (32, 32)
        )
        monkeypatch.setattr(
            "gui_agent.capture.screen.read_segment_frames", lambda p, w, h: frames
        )
        return RedactionScanner(PrivacyConfig(), ocr=fake_ocr), seen

    def test_near_identical_frames_are_only_ocred_once(self, monkeypatch, tmp_path):
        # At 15Hz consecutive frames are near-identical; OCRing all 300 would
        # take minutes and tell you nothing new.
        scanner, seen = self._patched_scanner(monkeypatch, self._frames(4, repeats=10))
        scanner.scan_video(tmp_path / "seg.mkv")
        assert len(seen) == 4

    def test_findings_on_screen_are_reported(self, monkeypatch, tmp_path):
        scanner, _ = self._patched_scanner(
            monkeypatch, self._frames(2), ocr_text="card 4111 1111 1111 1111"
        )
        findings = scanner.scan_video(tmp_path / "seg.mkv")
        assert findings and all(f.kind == "credit_card" for f in findings)
        assert all(f.bbox is not None for f in findings)

    def test_max_frames_caps_the_work(self, monkeypatch, tmp_path):
        scanner, seen = self._patched_scanner(monkeypatch, self._frames(20))
        scanner.scan_video(tmp_path / "seg.mkv", max_frames=3)
        assert len(seen) == 3

    def test_no_ocr_backend_scans_nothing(self, tmp_path):
        assert RedactionScanner(PrivacyConfig(), ocr=None).scan_video(tmp_path / "s.mkv") == []

    def test_unknown_video_size_is_skipped_not_guessed(self, monkeypatch, tmp_path):
        # Decoding at a guessed size distorts the image and wrecks OCR.
        monkeypatch.setattr("gui_agent.capture.screen.probe_video_size", lambda path: None)
        scanner = RedactionScanner(PrivacyConfig(), ocr=lambda i: [])
        assert scanner.scan_video(tmp_path / "seg.mkv") == []


class TestPromotionReadsWhatItScans:
    """Encrypted-at-rest is the default, and the sweep must handle it.

    Reading the plaintext path of an encrypted segment used to find nothing,
    scan nothing, and promote the segment as clean.
    """

    def _encrypted_segment(self, tmp_path, name, text, encryptor):
        write_jsonl(
            tmp_path / "records" / f"{name}.jsonl",
            [FrameRecord(0.0, "s", name, 0,
                         event=InputEvent(EventType.TEXT, text=text).to_dict())],
        )
        encryptor.encrypt_file(tmp_path / "records" / f"{name}.jsonl")
        return SegmentMeta(name, "s", 0.0, 1.0, 10, records_path=f"records/{name}.jsonl")

    @pytest.fixture
    def encryptor(self, monkeypatch):
        from gui_agent.capture.privacy import SegmentEncryptor

        monkeypatch.setenv(SegmentEncryptor.ENV_KEY, SegmentEncryptor.generate_key())
        return SegmentEncryptor()

    def test_encrypted_clean_segment_is_decrypted_and_promoted(self, tmp_path, encryptor):
        segment = self._encrypted_segment(tmp_path, "seg", "hello world", encryptor)
        scanner = RedactionScanner(PrivacyConfig(), ocr=lambda i: [])
        promoted, quarantined = promote_segments(
            [segment], scanner, tmp_path, tmp_path / "pool", encryptor=encryptor
        )
        assert [s.segment_id for s in promoted] == ["seg"] and not quarantined

    def test_secret_inside_an_encrypted_segment_is_found(self, tmp_path, encryptor):
        segment = self._encrypted_segment(tmp_path, "seg", "ssn 123-45-6789", encryptor)
        scanner = RedactionScanner(PrivacyConfig(), ocr=lambda i: [])
        promoted, quarantined = promote_segments(
            [segment], scanner, tmp_path, tmp_path / "pool", encryptor=encryptor
        )
        assert not promoted
        assert "ssn" in quarantined[0].redaction_findings[0]

    def test_no_key_refuses_rather_than_promoting_unscanned(self, tmp_path, encryptor):
        from gui_agent.capture.privacy import PrivacyError

        segment = self._encrypted_segment(tmp_path, "seg", "hello", encryptor)
        scanner = RedactionScanner(PrivacyConfig(), ocr=lambda i: [])
        with pytest.raises(PrivacyError, match="cannot read"):
            promote_segments([segment], scanner, tmp_path, tmp_path / "pool", encryptor=None)

    def test_missing_records_are_quarantined_not_promoted(self, tmp_path):
        segment = SegmentMeta("seg", "s", 0.0, 1.0, 10, records_path="records/gone.jsonl")
        scanner = RedactionScanner(PrivacyConfig(), ocr=lambda i: [])
        promoted, quarantined = promote_segments(
            [segment], scanner, tmp_path, tmp_path / "pool"
        )
        assert not promoted
        assert quarantined[0].redaction_findings == ["records_unreadable"]

    def test_a_failed_scan_never_reads_as_a_clean_one(self, tmp_path):
        write_jsonl(tmp_path / "records" / "seg.jsonl",
                    [FrameRecord(0.0, "s", "seg", 0,
                                 event=InputEvent(EventType.TEXT, text="hi").to_dict())])
        segment = SegmentMeta("seg", "s", 0.0, 1.0, 10, records_path="records/seg.jsonl")

        def exploding_ocr(image):
            raise RuntimeError("tesseract died")

        scanner = RedactionScanner(PrivacyConfig(), ocr=exploding_ocr)
        scanner.scan_video = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("decode failed"))
        segment.video_path = "video/seg.mkv"
        (tmp_path / "video").mkdir(exist_ok=True)
        (tmp_path / "video" / "seg.mkv").write_bytes(b"not a video")

        promoted, quarantined = promote_segments(
            [segment], scanner, tmp_path, tmp_path / "pool"
        )
        assert not promoted
        assert quarantined[0].redaction_findings[0].startswith("scan_failed")

    def test_decrypted_plaintext_is_not_left_behind(self, tmp_path, encryptor):
        segment = self._encrypted_segment(tmp_path, "seg", "hello world", encryptor)
        scanner = RedactionScanner(PrivacyConfig(), ocr=lambda i: [])
        promote_segments([segment], scanner, tmp_path, tmp_path / "pool", encryptor=encryptor)
        # The sweep must not leave a decrypted capture segment on disk.
        assert not (tmp_path / "records" / "seg.jsonl").exists()


class TestOcrAvailability:
    def test_missing_tesseract_binary_reports_unavailable(self, monkeypatch):
        """pip installs the wrapper, not the binary.

        Reporting OCR as available when every call would raise is the worst
        outcome: the gate passes, nothing is read, everything looks clean.
        """
        import sys
        import types

        fake = types.ModuleType("pytesseract")
        fake.get_tesseract_version = lambda: (_ for _ in ()).throw(
            RuntimeError("tesseract is not installed")
        )
        fake.Output = types.SimpleNamespace(DICT="dict")
        monkeypatch.setitem(sys.modules, "pytesseract", fake)

        from gui_agent.capture.privacy import default_ocr

        assert default_ocr() is None


class TestSessionDiscovery:
    """`promote` reported "nothing to promote" while `status` showed a pending
    segment, because it read `<root>/segments.jsonl` while the daemon writes
    `<root>/<session_id>/segments.jsonl`."""

    def test_sessions_are_found_under_the_root(self, tmp_path):
        from gui_agent.capture.cli import _session_dirs
        from gui_agent.config import CaptureConfig

        for name in ("sess_1", "sess_2"):
            (tmp_path / name).mkdir()
        (tmp_path / "loose_file.txt").write_text("not a session")

        found = _session_dirs(CaptureConfig(root=str(tmp_path)), None)
        assert [p.name for p in found] == ["sess_1", "sess_2"]

    def test_explicit_session_is_used_directly(self, tmp_path):
        from gui_agent.capture.cli import _session_dirs
        from gui_agent.config import CaptureConfig

        found = _session_dirs(CaptureConfig(root=str(tmp_path)), "sess_9")
        assert found == [tmp_path / "sess_9"]

    def test_missing_root_is_empty_not_an_error(self, tmp_path):
        from gui_agent.capture.cli import _session_dirs
        from gui_agent.config import CaptureConfig

        assert _session_dirs(CaptureConfig(root=str(tmp_path / "nope")), None) == []

    def test_promote_and_status_agree_on_where_segments_live(self, tmp_path):
        """The two commands must not disagree about what exists."""
        from gui_agent.capture.cli import _load_segments, _session_dirs
        from gui_agent.config import CaptureConfig

        session = tmp_path / "sess_1788676375"
        session.mkdir()
        write_jsonl(session / "segments.jsonl",
                    [SegmentMeta("seg_a", "sess_1788676375", 0.0, 20.0, 300,
                                 records_path="records/seg_a.jsonl")])

        config = CaptureConfig(root=str(tmp_path))
        pending = [s for d in _session_dirs(config, None)
                   for s in _load_segments(d) if s.redaction_status == "pending"]
        assert len(pending) == 1


class TestQuarantineRetry:
    """A segment quarantined because the sweep *could not run* was stuck forever.

    `promote` only looks at pending segments, so installing the missing OCR
    backend afterwards changed nothing -- the segment stayed quarantined for a
    reason that no longer applied, with no path back.
    """

    @staticmethod
    def _quarantined(findings):
        segment = SegmentMeta("seg", "s", 0.0, 1.0, 10, records_path="records/seg.jsonl")
        segment.redaction_status = "quarantined"
        segment.redaction_findings = findings
        return segment

    def test_environmental_reasons_are_retryable(self):
        from gui_agent.capture.privacy import is_retryable_quarantine

        for reason in ("ocr_unavailable", "records_unreadable", "video_unreadable",
                       "missing_records", "scan_failed:OSError"):
            assert is_retryable_quarantine(self._quarantined([reason])), reason

    def test_content_findings_are_never_retryable(self):
        # Retrying would either re-find the same thing, or -- if the patterns
        # were since loosened -- quietly promote flagged material.
        from gui_agent.capture.privacy import is_retryable_quarantine

        assert not is_retryable_quarantine(self._quarantined(["ssn:11chars"]))
        assert not is_retryable_quarantine(self._quarantined(["credit_card:16chars"]))

    def test_a_mix_is_not_retryable(self):
        from gui_agent.capture.privacy import is_retryable_quarantine

        assert not is_retryable_quarantine(
            self._quarantined(["ocr_unavailable", "ssn:11chars"])
        )

    def test_non_quarantined_segments_are_not_retryable(self):
        from gui_agent.capture.privacy import is_retryable_quarantine

        for status in ("pending", "clean", "redacted"):
            segment = self._quarantined([])
            segment.redaction_status = status
            assert not is_retryable_quarantine(segment)

    def test_retry_rescans_and_can_promote(self, tmp_path):
        """The end-to-end path: quarantined for missing OCR, then OCR works."""
        from gui_agent.capture.privacy import promote_segments

        write_jsonl(tmp_path / "records" / "seg.jsonl",
                    [FrameRecord(0.0, "s", "seg", 0,
                                 event=InputEvent(EventType.TEXT, text="hello").to_dict())])
        segment = self._quarantined(["ocr_unavailable"])
        scanner = RedactionScanner(PrivacyConfig(), ocr=lambda i: [])

        promoted, quarantined = promote_segments(
            [segment], scanner, tmp_path, tmp_path / "pool"
        )
        assert [s.segment_id for s in promoted] == ["seg"]
        assert segment.redaction_status == "clean"

    def test_retry_still_quarantines_if_the_content_is_bad(self, tmp_path):
        from gui_agent.capture.privacy import promote_segments

        write_jsonl(tmp_path / "records" / "seg.jsonl",
                    [FrameRecord(0.0, "s", "seg", 0,
                                 event=InputEvent(EventType.TEXT,
                                                  text="ssn 123-45-6789").to_dict())])
        segment = self._quarantined(["ocr_unavailable"])
        scanner = RedactionScanner(PrivacyConfig(), ocr=lambda i: [])

        promoted, quarantined = promote_segments(
            [segment], scanner, tmp_path, tmp_path / "pool"
        )
        assert not promoted
        assert "ssn" in quarantined[0].redaction_findings[0]
