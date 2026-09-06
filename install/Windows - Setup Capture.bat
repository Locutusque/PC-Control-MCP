@echo off
REM Double-click this to set up the capture daemon on Windows.
REM
REM Installs into an isolated Python environment inside this folder. It
REM cannot grant Windows any permission on your behalf -- if your keyboard or
REM mouse hooks are blocked by security software, this will tell you, but you
REM will need to allow it yourself in that software's settings.

cd /d "%~dp0.."

where python >nul 2>nul
if errorlevel 1 (
    echo Python 3.10 or newer wasn't found.
    echo Install it from https://www.python.org/downloads/windows/
    echo IMPORTANT: on the first install screen, check "Add python.exe to PATH".
    echo Then run this installer again.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)"') do set VERSION_OK=%%v
if not "%VERSION_OK%"=="1" (
    echo Found Python, but it's older than the required Python 3.10.
    echo Install a newer version from https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

if not exist .venv (
    echo Setting up an isolated Python environment...
    python -m venv .venv
)

echo Installing dependencies ^(this can take a few minutes on first run^)...
.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
.venv\Scripts\python.exe -m pip install --quiet -e ".[capture,capture-windows,redaction]"

.venv\Scripts\python.exe -m gui_agent.capture.onboarding

echo.
pause
