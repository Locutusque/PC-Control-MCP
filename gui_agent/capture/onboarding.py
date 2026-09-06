"""Guided setup for the capture daemon: as close to "one click" as it gets.

This is meant for someone who is not going to read `docs/privacy.md` line by
line or hand-edit a `CaptureConfig` JSON file. It automates everything that is
*safe* to automate, and is explicit about the two things it deliberately does
not:

* **It cannot grant Accessibility / Screen Recording permission on macOS, or
  register a Windows global input hook past UAC.** Those are OS security
  boundaries that exist specifically so a script cannot silently turn on
  keystroke and screen capture. Scripting around them is not a feature this
  project will ever add. The wizard opens the exact settings pane and waits
  for a human to flip the switch -- see :func:`permission_panes` and
  :func:`PERMISSION_INSTRUCTIONS`.
* **It cannot decide what belongs on your blocklist.** It proposes the
  defaults from :class:`~gui_agent.config.PrivacyConfig` and asks what to add.
  It has no way to know which of your own apps and sites are sensitive.

Everything else -- creating an isolated environment, installing dependencies,
generating and storing an encryption key, writing a working config, running
the safety preflight, and leaving behind simple launchers -- happens without
asking the user to touch a terminal command by hand.

Run it directly (``python -m gui_agent.capture.onboarding``) or via one of the
double-clickable launchers in ``install/``.

Design note: the interactive parts of this module (``input()``, opening a
browser pane, running the daemon) are deliberately thin. Every decision they
make is pulled out into a plain function below so it can be tested without
mocking a terminal.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from ..config import CaptureConfig, PrivacyConfig
from .platform_ import get_backend

__all__ = [
    "HOME_DIR",
    "PackageManager",
    "detect_package_manager",
    "ffmpeg_install_hint",
    "permission_panes",
    "PERMISSION_INSTRUCTIONS",
    "merge_blocklist",
    "KeyStorage",
    "plain_english_report",
    "consent_prompt_text",
    "default_config",
    "write_launcher_scripts",
    "run_wizard",
    "main",
]

# Everything this wizard creates lives here, not inside the repo checkout --
# re-cloning or moving the repo must not lose the capture key or the pool.
HOME_DIR = Path.home() / ".gui-agent"


# --------------------------------------------------------------------------
# ffmpeg
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PackageManager:
    name: str
    install_ffmpeg: str
    needs_sudo: bool = False


_PACKAGE_MANAGERS = (
    PackageManager("brew", "brew install ffmpeg"),
    PackageManager("winget", "winget install --id Gyan.FFmpeg -e"),
    PackageManager("choco", "choco install ffmpeg -y"),
    PackageManager("apt-get", "sudo apt-get install -y ffmpeg", needs_sudo=True),
    PackageManager("dnf", "sudo dnf install -y ffmpeg", needs_sudo=True),
    PackageManager("pacman", "sudo pacman -S --noconfirm ffmpeg", needs_sudo=True),
)


def detect_package_manager(which=shutil.which) -> PackageManager | None:
    """The first package manager found on PATH, checked in a fixed order.

    Order matters only where more than one could be present (e.g. a Mac with
    both Homebrew and MacPorts) -- brew is checked first because it is by far
    the common case among people likely to run a one-click installer.
    """
    for manager in _PACKAGE_MANAGERS:
        if which(manager.name):
            return manager
    return None


def ffmpeg_install_hint(manager: PackageManager | None) -> str:
    if manager is not None:
        return manager.install_ffmpeg
    return "download a build from https://ffmpeg.org/download.html and add it to your PATH"


# --------------------------------------------------------------------------
# macOS permission panes
# --------------------------------------------------------------------------

# These deep-link URLs have been stable across Ventura/Sonoma/Sequoia. If a
# future macOS version renames the pane, the fallback instruction (open
# System Settings > Privacy & Security by hand) still works -- the wizard
# always prints it alongside the link rather than depending on the link alone.
PERMISSION_PANES = (
    ("Accessibility", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"),
    ("Screen Recording", "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"),
)

PERMISSION_INSTRUCTIONS = (
    "macOS will not let any script grant these for you -- that protection exists "
    "specifically so a program cannot silently turn on your keyboard, mouse and "
    "screen capture, and this installer is not going to try to get around it.\n\n"
    "For each pane that opens:\n"
    "  1. Click the lock icon and enter your password if it's locked.\n"
    "  2. Click the '+' button.\n"
    "  3. Add Terminal (or iTerm, if that's what you're running this from).\n"
    "  4. Make sure the checkbox next to it is turned on.\n\n"
    "If a pane doesn't open, go to System Settings > Privacy & Security and find "
    "it in the list yourself."
)


def permission_panes(platform_name: str) -> tuple[tuple[str, str], ...]:
    """The (name, URL) panes to open. Empty outside macOS -- there is no
    equivalent lockdown to walk through on Windows or Linux for this daemon."""
    return PERMISSION_PANES if platform_name == "darwin" else ()


def open_permission_panes(platform_name: str, opener=subprocess.run) -> None:  # pragma: no cover - OS call
    for _, url in permission_panes(platform_name):
        try:
            opener(["open", url], check=False, timeout=5.0)
        except (OSError, subprocess.SubprocessError):
            pass


# --------------------------------------------------------------------------
# Blocklist
# --------------------------------------------------------------------------


def merge_blocklist(defaults: tuple[str, ...], additions: list[str]) -> tuple[str, ...]:
    """Add the user's own entries to a default blocklist, deduplicated.

    Order-preserving and case-normalised, so a user typing "Signal" doesn't
    end up alongside an existing "signal" as two separate entries that a
    substring match would treat identically anyway.
    """
    cleaned = (a.strip().lower() for a in additions if a and a.strip())
    return tuple(dict.fromkeys((*defaults, *cleaned)))


# --------------------------------------------------------------------------
# Key storage
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyStorage:
    key: str
    description: str
    env_export_line: str


def store_key_in_macos_keychain(key: str, account: str, runner=subprocess.run) -> bool:  # pragma: no cover - OS call
    if not shutil.which("security"):
        return False
    result = runner(
        ["security", "add-generic-password", "-a", account, "-s", "gui-agent-capture-key",
         "-w", key, "-U"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def generate_and_store_key(platform_name: str, home: Path = HOME_DIR,
                           account: str | None = None) -> KeyStorage:
    """Generate a capture key and put it somewhere the launchers can find it.

    macOS gets Keychain storage when the ``security`` CLI is available,
    because that is where a non-technical Mac user's other app secrets already
    live and it survives independently of any file on disk. Everywhere else,
    and as the macOS fallback, the key goes to a file the launcher scripts
    read at start -- permissioned 0600, which is the same protection your SSH
    key gets.
    """
    from .privacy import SegmentEncryptor

    key = SegmentEncryptor.generate_key()
    account = account or os.environ.get("USER") or os.environ.get("USERNAME") or "gui-agent"

    if platform_name == "darwin" and store_key_in_macos_keychain(key, account):
        return KeyStorage(
            key=key,
            description="stored in the macOS Keychain (service: gui-agent-capture-key)",
            env_export_line=(
                f'export {SegmentEncryptor.ENV_KEY}="$(security find-generic-password '
                f'-a {account} -s gui-agent-capture-key -w)"'
            ),
        )

    key_path = home / "capture.key"
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(key)
    key_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return KeyStorage(
        key=key,
        description=f"stored at {key_path} (readable only by you)",
        env_export_line=f'export {SegmentEncryptor.ENV_KEY}="$(cat "{key_path}")"',
    )


# --------------------------------------------------------------------------
# Plain-English reporting
# --------------------------------------------------------------------------

_FRIENDLY_PROBLEMS = (
    ("cannot read the foreground window",
     "Your computer can't tell which application is in front, so it can't reliably "
     "avoid recording things like a password manager. This is usually the "
     "Accessibility permission -- see the steps above."),
    ("GUI_AGENT_CAPTURE_KEY is not set",
     "The encryption key isn't set up yet. Re-run this installer, or open a "
     "terminal and paste the export line it gave you before starting a recording."),
    ("ffmpeg not found",
     "ffmpeg (the tool that saves the video) isn't installed. See the install "
     "command shown above."),
    ("cryptography",
     "The encryption library isn't installed properly. Re-run this installer; "
     "if that doesn't fix it, something on this machine is blocking the install."),
)

_FRIENDLY_WARNINGS = (
    ("secure-field detection is unavailable",
     "Your computer can't yet tell a password box from an ordinary text box. "
     "Until the Accessibility permission above is granted, it will play it safe "
     "and record NO typed text at all -- your mouse movements and clicks will "
     "still be captured."),
    ("single flat colour",
     "The screenshot it just took was a single solid colour, which almost always "
     "means Screen Recording permission hasn't been granted yet -- see the steps "
     "above."),
    ("input capture will not work",
     "It can't watch your keyboard and mouse yet. See the Accessibility steps "
     "above."),
)


def _friendly(message: str, table: tuple[tuple[str, str], ...]) -> str:
    for needle, friendly in table:
        if needle in message:
            return friendly
    return message  # never hide an unrecognised message; just show it plainly


def plain_english_report(problems: list[str], warnings: list[str]) -> str:
    """Turn `doctor`'s engineering-facing findings into something a
    non-technical user can act on without looking anything up."""
    if not problems and not warnings:
        return "Everything checks out. It's ready to record."

    lines: list[str] = []
    if problems:
        lines.append("Before it can start, these need to be fixed:")
        lines.extend(f"  - {_friendly(p, _FRIENDLY_PROBLEMS)}" for p in problems)
    if warnings:
        lines.append("" if not lines else "")
        lines.append("It can start, but you should know:")
        lines.extend(f"  - {_friendly(w, _FRIENDLY_WARNINGS)}" for w in warnings)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Consent
# --------------------------------------------------------------------------


def consent_prompt_text(blocklist_count: int, recordings_dir: Path, key_description: str) -> str:
    """What the user reads immediately before the daemon can be started.

    This is the actual safety mechanism -- informed consent -- so it is a
    named, tested function rather than an inline string that a later edit
    could quietly weaken. Keep the recording location and the key's storage
    location as separate facts: conflating them once already produced a
    consent message that told the user their screen recordings would be
    saved to a small text file containing an encryption key.
    """
    return (
        "One more thing before this starts recording.\n\n"
        f"Once running, it will save your screen, mouse and keyboard activity, "
        f"encrypted, to {recordings_dir} on this computer -- except while a "
        f"blocked app or site has focus ({blocklist_count} entries on that list "
        "right now) or a password field is in view. The encryption key is "
        f"{key_description}.\n\n"
        "It will NOT upload anything anywhere. Nothing leaves this computer.\n\n"
        "The plan this project follows is explicit that you should read a short "
        "recording back before trusting it with a real workday -- see "
        "docs/privacy.md. This installer will not do that check for you.\n\n"
        "Type YES to start recording now, or press Enter to finish setup "
        "without starting it: "
    )


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def default_config(home: Path = HOME_DIR, blocklist_additions: list[str] | None = None) -> CaptureConfig:
    privacy = PrivacyConfig()
    if blocklist_additions:
        privacy = PrivacyConfig(
            **{**privacy.to_dict(),
               "app_blocklist": merge_blocklist(privacy.app_blocklist, blocklist_additions),
               "domain_blocklist": merge_blocklist(privacy.domain_blocklist, blocklist_additions)}
        )
    return CaptureConfig(
        root=str(home / "raw_capture"),
        pool_root=str(home / "data_pool"),
        privacy=privacy,
    )


# --------------------------------------------------------------------------
# Launcher scripts
# --------------------------------------------------------------------------


def _venv_python(repo_root: Path, platform_name: str) -> Path:
    if platform_name == "win32":
        return repo_root / ".venv" / "Scripts" / "python.exe"
    return repo_root / ".venv" / "bin" / "python"


def write_launcher_scripts(
    repo_root: Path, home: Path, key_storage: KeyStorage, platform_name: str,
) -> dict[str, Path]:
    """Write small double-clickable scripts for ongoing, day-to-day use.

    Kept deliberately dumb: each one sources the key, then hands off entirely
    to `python -m gui_agent.capture.cli`, which is the one place the real
    behaviour (and its tests) live.
    """
    home.mkdir(parents=True, exist_ok=True)
    python = _venv_python(repo_root, platform_name)
    config_path = home / "capture_config.json"
    written: dict[str, Path] = {}

    if platform_name == "win32":
        for name, args in (("Start Recording", "record"), ("Check Status", "status")):
            path = home / f"{name}.bat"
            path.write_text(
                "@echo off\r\n"
                f'{key_storage.env_export_line.replace("export ", "set ")}\r\n'
                f'"{python}" -m gui_agent.capture.cli {args} --config "{config_path}"\r\n'
                "pause\r\n"
            )
            written[name] = path
    else:
        for name, args in (("Start Recording", "record"), ("Check Status", "status")):
            path = home / f"{name}.command"
            path.write_text(
                "#!/bin/sh\n"
                f"{key_storage.env_export_line}\n"
                f'exec "{python}" -m gui_agent.capture.cli {args} --config "{config_path}"\n'
            )
            path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            written[name] = path

    return written


# --------------------------------------------------------------------------
# The wizard
# --------------------------------------------------------------------------


BANNER = """
==============================================================
 GUI Agent -- Capture Setup
==============================================================
This sets up the background recorder that will later teach a
model to use your computer. It takes a few minutes, and two
steps along the way need you to click something in a macOS
settings window that no script is allowed to click for you.
"""


def _ask(prompt: str, default: str = "") -> str:  # pragma: no cover - terminal I/O
    try:
        return input(prompt).strip()
    except EOFError:
        return default


def run_wizard(repo_root: Path | None = None, auto_start: bool | None = None) -> int:  # pragma: no cover - interactive
    """The interactive setup flow. See module docstring for what it can't do."""
    repo_root = repo_root or Path(__file__).resolve().parents[2]
    platform_name = get_backend().name
    print(BANNER)

    # The wizard has its own plain-English reporting (step 6); a raw library
    # warning bypassing that straight to stderr is exactly the jargon this
    # exists to hide, and on a non-tty pipe it can interleave with prompts in
    # a way that looks like the installer glitched.
    logging.getLogger("gui_agent").setLevel(logging.ERROR)

    # 1. ffmpeg
    if not shutil.which("ffmpeg"):
        manager = detect_package_manager()
        hint = ffmpeg_install_hint(manager)
        print(f"ffmpeg is required and wasn't found. Install it with:\n\n    {hint}\n")
        if manager and not manager.needs_sudo and _ask("Run that now? [y/N] ").lower() == "y":
            subprocess.run(hint.split(), check=False)
        if not shutil.which("ffmpeg"):
            print("\nffmpeg still isn't on your PATH. Install it, then run this installer again.")
            return 1

    # 2. dependencies, in an isolated environment inside the repo
    python_exe = _venv_python(repo_root, platform_name)
    if not python_exe.exists():
        print("Setting up an isolated Python environment (this can take a minute)...")
        subprocess.run([sys.executable, "-m", "venv", str(repo_root / ".venv")], check=True)
    extras = {"darwin": "capture,capture-macos,redaction",
              "win32": "capture,capture-windows,redaction",
              "linux": "capture,capture-linux,redaction"}.get(platform_name, "capture,redaction")
    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "--quiet", "-e", f"{repo_root}[{extras}]"],
        check=True,
    )

    # 3. macOS permissions -- opened, never bypassed
    panes = permission_panes(platform_name)
    if panes:
        print(PERMISSION_INSTRUCTIONS)
        open_permission_panes(platform_name)
        _ask("\nPress Enter once you've added Terminal in both panes... ")

    # 4. encryption key
    key_storage = generate_and_store_key(platform_name)
    # The launcher scripts export this on every future run by re-reading it
    # from where it was just stored; for the preflight check a few lines down
    # (running in *this* process) to see anything other than "not set", it
    # needs to be exported here too.
    from .privacy import SegmentEncryptor

    os.environ[SegmentEncryptor.ENV_KEY] = key_storage.key
    print(f"\nEncryption key generated and {key_storage.description}.")

    # 5. blocklist
    print(
        "\nBy default this will never record certain apps or sites (password "
        "managers, banking, health portals). Add anything else you want it to "
        "skip -- messaging apps, a journal, anything -- separated by commas."
    )
    additions = [a for a in _ask("Add to the blocklist (or press Enter to skip): ").split(",") if a.strip()]
    config = default_config(HOME_DIR, additions)
    HOME_DIR.mkdir(parents=True, exist_ok=True)
    config.save(HOME_DIR / "capture_config.json")

    # 6. preflight, in plain English
    from .daemon import CaptureDaemon

    daemon = CaptureDaemon(config)
    problems = daemon.preflight()
    caps = get_backend().capabilities()
    warnings: list[str] = []
    if not caps.get("secure_field_detection") and config.privacy.redact_password_fields:
        warnings.append("secure-field detection is unavailable")
    print("\n" + plain_english_report(problems, warnings))

    # 7. launchers for ongoing use
    launchers = write_launcher_scripts(repo_root, HOME_DIR, key_storage, platform_name)
    print("\nCreated, in " + str(HOME_DIR) + ":")
    for name, path in launchers.items():
        print(f"  {name}  ->  double-click {path.name} any time")

    if problems:
        print("\nFix the items above, then double-click 'Start Recording' when ready.")
        return 1

    # 8. explicit, informed consent before the first run
    should_start = auto_start
    if should_start is None:
        answer = _ask(
            "\n" + consent_prompt_text(
                len(config.privacy.app_blocklist), Path(config.root), key_storage.description
            )
        )
        should_start = answer.strip().upper() == "YES"

    if should_start:
        print("\nStarting capture. A tray/menu-bar icon will show it's recording; "
              "use it, or Ctrl+Alt+P, to pause.")
        from .cli import cmd_record  # GUI_AGENT_CAPTURE_KEY was exported in step 4

        return cmd_record(_Args(config=str(HOME_DIR / "capture_config.json"), max_frames=None))
    print("\nSetup complete. Double-click 'Start Recording' whenever you're ready.")
    return 0


@dataclass
class _Args:
    """Mimics the argparse.Namespace cmd_record expects, without importing argparse here."""
    config: str
    max_frames: int | None = None


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - entry point
    import argparse

    parser = argparse.ArgumentParser(description="Guided setup for the capture daemon")
    parser.add_argument("--yes", action="store_true",
                        help="start recording immediately after setup, skipping the prompt")
    parser.add_argument("--no-start", action="store_true",
                        help="finish setup without starting a recording")
    args = parser.parse_args(argv)
    auto_start = True if args.yes else (False if args.no_start else None)
    return run_wizard(auto_start=auto_start)


if __name__ == "__main__":
    raise SystemExit(main())
