"""The guided capture-setup wizard.

Only the pure decision functions are tested here -- the wizard's `input()`,
subprocess and OS-open calls are deliberately thin and hold no logic of their
own, per the module's design note. What matters is that every message a
non-technical user reads is accurate and every artifact it writes is usable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gui_agent.capture.onboarding import (
    KeyStorage,
    PackageManager,
    consent_prompt_text,
    default_config,
    detect_package_manager,
    ffmpeg_install_hint,
    generate_and_store_key,
    merge_blocklist,
    permission_panes,
    plain_english_report,
    write_launcher_scripts,
)
from gui_agent.config import PrivacyConfig


class TestPackageManagerDetection:
    def test_finds_the_first_manager_present(self):
        present = {"apt-get"}
        manager = detect_package_manager(which=lambda name: name if name in present else None)
        assert manager.name == "apt-get"

    def test_none_when_nothing_is_present(self):
        assert detect_package_manager(which=lambda name: None) is None

    def test_brew_is_preferred_when_multiple_are_present(self):
        # The common case for someone likely to run a one-click Mac installer.
        present = {"brew", "apt-get"}
        manager = detect_package_manager(which=lambda name: name if name in present else None)
        assert manager.name == "brew"


class TestFfmpegHint:
    def test_known_manager_gets_its_own_command(self):
        assert "brew install" in ffmpeg_install_hint(PackageManager("brew", "brew install ffmpeg"))

    def test_no_manager_gets_a_manual_download_link(self):
        hint = ffmpeg_install_hint(None)
        assert "ffmpeg.org" in hint


class TestPermissionPanes:
    def test_macos_has_two_panes(self):
        panes = permission_panes("darwin")
        names = {name for name, _ in panes}
        assert names == {"Accessibility", "Screen Recording"}

    def test_pane_urls_are_well_formed_apple_deep_links(self):
        for _, url in permission_panes("darwin"):
            assert url.startswith("x-apple.systempreferences:")

    @pytest.mark.parametrize("platform_name", ["linux", "win32", "freebsd"])
    def test_no_panes_outside_macos(self, platform_name):
        # There is no equivalent lockdown to walk through elsewhere; opening
        # nothing is correct, not a missing feature.
        assert permission_panes(platform_name) == ()


class TestBlocklistMerge:
    def test_user_additions_are_appended(self):
        merged = merge_blocklist(("signal",), ["Notion", "my-journal"])
        assert "notion" in merged and "my-journal" in merged

    def test_duplicates_are_not_added_twice(self):
        merged = merge_blocklist(("signal",), ["Signal", "SIGNAL", "signal "])
        assert merged.count("signal") == 1

    def test_blank_and_whitespace_entries_are_dropped(self):
        merged = merge_blocklist(("signal",), ["", "   ", "notion"])
        assert merged == ("signal", "notion")

    def test_empty_additions_leave_defaults_untouched(self):
        defaults = PrivacyConfig().app_blocklist
        assert merge_blocklist(defaults, []) == defaults


class TestPlainEnglishReport:
    def test_no_findings_is_a_clear_go_ahead(self):
        assert plain_english_report([], []) == "Everything checks out. It's ready to record."

    def test_known_problem_gets_translated(self):
        report = plain_english_report(["ffmpeg not found on PATH; it is required"], [])
        assert "ffmpeg" in report.lower()
        assert "tool that saves the video" in report

    def test_unrecognised_message_is_shown_rather_than_hidden(self):
        # A message this function doesn't know how to translate must still
        # reach the user -- silently swallowing it would be worse than a
        # slightly technical sentence.
        report = plain_english_report(["some future problem nobody wrote a translation for"], [])
        assert "some future problem nobody wrote a translation for" in report

    def test_problems_and_warnings_are_both_present_and_distinguished(self):
        report = plain_english_report(["ffmpeg not found"], ["secure-field detection is unavailable"])
        assert "need to be fixed" in report
        assert "you should know" in report

    def test_warnings_alone_do_not_claim_a_problem(self):
        report = plain_english_report([], ["secure-field detection is unavailable"])
        assert "need to be fixed" not in report


class TestConsentPrompt:
    def test_states_where_recordings_are_stored_not_where_the_key_is(self):
        # Regression: an earlier version of this text described the key's
        # storage location as if it were where screen recordings are saved.
        text = consent_prompt_text(3, Path("/home/x/.gui-agent/raw_capture"), "in the Keychain")
        assert "/home/x/.gui-agent/raw_capture" in text
        assert "in the Keychain" in text
        recording_sentence = text.split("Once running,")[1].split(".")[0]
        assert "Keychain" not in recording_sentence

    def test_states_nothing_is_uploaded(self):
        assert "NOT upload" in consent_prompt_text(0, Path("/x"), "on disk")

    def test_reports_the_actual_blocklist_count(self):
        assert "17 entries" in consent_prompt_text(17, Path("/x"), "on disk")

    def test_points_at_the_privacy_doc(self):
        assert "docs/privacy.md" in consent_prompt_text(0, Path("/x"), "on disk")


class TestDefaultConfig:
    def test_storage_lives_under_home_not_the_repo(self, tmp_path):
        config = default_config(tmp_path)
        assert str(tmp_path) in config.root
        assert str(tmp_path) in config.pool_root

    def test_user_blocklist_entries_reach_both_lists(self, tmp_path):
        config = default_config(tmp_path, ["my-therapy-app", "myclinic.example"])
        assert "my-therapy-app" in config.privacy.app_blocklist
        assert "myclinic.example" in config.privacy.domain_blocklist

    def test_defaults_survive_with_no_additions(self, tmp_path):
        config = default_config(tmp_path, None)
        assert config.privacy.app_blocklist == PrivacyConfig().app_blocklist


class TestKeyStorage:
    def test_file_fallback_when_keychain_is_unavailable(self, tmp_path):
        storage = generate_and_store_key("linux", tmp_path)
        key_file = tmp_path / "capture.key"
        assert key_file.exists()
        assert key_file.read_text() == storage.key

    def test_key_file_is_owner_only(self, tmp_path):
        generate_and_store_key("linux", tmp_path)
        mode = (tmp_path / "capture.key").stat().st_mode & 0o777
        assert mode == 0o600

    def test_generated_key_is_actually_usable(self, tmp_path, monkeypatch):
        # Round-trip through the real encryptor, not just a string comparison.
        from gui_agent.capture.privacy import SegmentEncryptor

        storage = generate_and_store_key("linux", tmp_path)
        monkeypatch.setenv(SegmentEncryptor.ENV_KEY, storage.key)
        encryptor = SegmentEncryptor()
        path = tmp_path / "secret.txt"
        path.write_bytes(b"hello")
        encryptor.encrypt_file(path)
        assert encryptor.decrypt_file(path.with_suffix(".txt.enc")).read_bytes() == b"hello"

    def test_export_line_actually_recovers_the_key(self, tmp_path):
        import subprocess

        storage = generate_and_store_key("linux", tmp_path)
        output = subprocess.run(
            ["sh", "-c", storage.env_export_line + " && echo $GUI_AGENT_CAPTURE_KEY"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert output == storage.key


class TestLauncherScripts:
    def test_launchers_are_executable_on_unix(self, tmp_path):
        repo, home = tmp_path / "repo", tmp_path / "home"
        storage = KeyStorage("k", "on disk", 'export GUI_AGENT_CAPTURE_KEY="k"')
        written = write_launcher_scripts(repo, home, storage, "linux")
        for path in written.values():
            assert path.stat().st_mode & 0o111

    def test_launchers_reference_the_venv_python_not_bare_python(self, tmp_path):
        repo, home = tmp_path / "repo", tmp_path / "home"
        storage = KeyStorage("k", "on disk", 'export GUI_AGENT_CAPTURE_KEY="k"')
        written = write_launcher_scripts(repo, home, storage, "linux")
        content = written["Start Recording"].read_text()
        assert str(repo / ".venv" / "bin" / "python") in content

    def test_windows_launcher_uses_batch_syntax(self, tmp_path):
        repo, home = tmp_path / "repo", tmp_path / "home"
        storage = KeyStorage("k", "on disk", 'export GUI_AGENT_CAPTURE_KEY="k"')
        written = write_launcher_scripts(repo, home, storage, "win32")
        content = written["Start Recording"].read_text()
        assert content.startswith("@echo off")
        assert "set GUI_AGENT_CAPTURE_KEY" in content
        assert ".venv\\Scripts\\python.exe" in content or "Scripts" in content

    def test_start_and_status_launchers_both_exist(self, tmp_path):
        repo, home = tmp_path / "repo", tmp_path / "home"
        storage = KeyStorage("k", "on disk", 'export GUI_AGENT_CAPTURE_KEY="k"')
        written = write_launcher_scripts(repo, home, storage, "linux")
        assert {"Start Recording", "Check Status"} == set(written)

    def test_launcher_config_path_matches_where_default_config_is_saved(self, tmp_path):
        # If these ever drift apart, "Start Recording" launches against a
        # config that setup never wrote.
        repo, home = tmp_path / "repo", tmp_path / "home"
        storage = KeyStorage("k", "on disk", 'export GUI_AGENT_CAPTURE_KEY="k"')
        written = write_launcher_scripts(repo, home, storage, "linux")
        assert str(home / "capture_config.json") in written["Start Recording"].read_text()
