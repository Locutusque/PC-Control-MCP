#!/bin/sh
# Double-click this in Finder to set up the capture daemon on a Mac.
#
# This installs into an isolated Python environment inside this folder and
# never touches your system Python. It cannot grant the Accessibility or
# Screen Recording permissions macOS requires -- no script can, by design --
# but it will open the exact settings panes for you.
set -e
cd "$(dirname "$0")/.."

if [ -t 1 ]; then :; else
  # Double-clicked rather than run from a terminal: relaunch inside
  # Terminal.app so the user can see progress and answer prompts.
  osascript -e "tell application \"Terminal\" to do script \"cd '$(pwd)' && sh '$0'\""
  exit 0
fi

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
done

if [ -z "$PYTHON" ]; then
  echo "Python 3.10 or newer wasn't found on this Mac."
  echo "Install it from https://www.python.org/downloads/macos/ (or, if you have"
  echo "Homebrew: brew install python@3.12), then run this installer again."
  read -p "Press Enter to close..." _
  exit 1
fi

VERSION_OK=$("$PYTHON" -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')
if [ "$VERSION_OK" != "1" ]; then
  echo "Found $PYTHON, but it's older than the required Python 3.10."
  echo "Install a newer Python from https://www.python.org/downloads/macos/."
  read -p "Press Enter to close..." _
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Setting up an isolated Python environment..."
  "$PYTHON" -m venv .venv
fi

echo "Installing dependencies (this can take a few minutes on first run)..."
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e ".[capture,capture-macos,redaction]"

.venv/bin/python -m gui_agent.capture.onboarding

echo
read -p "Press Enter to close this window..." _
