#!/bin/sh
# Run this to set up the capture daemon on Linux.
#
#   sh "Linux - Setup Capture.sh"
#
# It installs into an isolated Python environment inside this folder. Global
# keyboard/mouse capture and reading the active window generally work best
# under X11; under Wayland the compositor may refuse both by design, and the
# setup will tell you if that's the case on your system.
set -e
cd "$(dirname "$0")/.."

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
done

if [ -z "$PYTHON" ]; then
  echo "Python 3.10 or newer wasn't found."
  echo "Install it with your distribution's package manager (e.g."
  echo "'sudo apt install python3 python3-venv' on Debian/Ubuntu), then run this"
  echo "installer again."
  exit 1
fi

VERSION_OK=$("$PYTHON" -c 'import sys; print(1 if sys.version_info >= (3, 10) else 0)')
if [ "$VERSION_OK" != "1" ]; then
  echo "Found $PYTHON, but it's older than the required Python 3.10."
  exit 1
fi

if [ ! -d .venv ]; then
  echo "Setting up an isolated Python environment..."
  "$PYTHON" -m venv .venv
fi

echo "Installing dependencies (this can take a few minutes on first run)..."
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -e ".[capture,capture-linux,redaction]"

.venv/bin/python -m gui_agent.capture.onboarding
