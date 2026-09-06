# Setup for non-technical users

These get the **capture daemon** running — the background recorder that
collects the data everything else in this project trains on. That's the only
thing worth "one-click installing" today: there's no trained policy yet, so
there's nothing else here a non-technical user would run day to day.

## What to do

Download or clone this repository, then double-click the file for your
system:

| System | File |
|---|---|
| macOS | `macOS - Setup Capture.command` |
| Windows | `Windows - Setup Capture.bat` |
| Linux | run `sh "Linux - Setup Capture.sh"` in a terminal |

Each one installs everything into an isolated environment inside this folder
— it never touches your system Python — and then walks you through the rest:
generating an encryption key, choosing what never to record, and checking
that your computer is actually ready.

## What it can't do for you

**Two clicks on macOS, and they're not a bug.** macOS requires a human to
grant Accessibility and Screen Recording permission — no script is allowed to
flip those switches, on purpose, because that's exactly the protection that
stops a program from silently turning on your keyboard and screen capture.
The installer opens the right settings panes; you click Allow.

**It won't decide what's private for you.** It skips password managers,
banking and health sites by default, but it has no way to know which of
*your* apps you consider sensitive. It asks.

**It won't start recording without you explicitly saying so**, every time,
even after everything else is set up — see `docs/privacy.md` for why that
confirmation exists and isn't a step worth skipping.

## After setup

Two shortcuts are created next to your home folder's `.gui-agent` directory:

- **Start Recording** — starts the daemon. A tray/menu-bar icon shows while
  it's live; the same icon (or Ctrl+Alt+P) pauses it.
- **Check Status** — how much has been recorded so far.

**Read a short recording back before you leave it running for real.** Every
guarantee in `docs/privacy.md` is a claim about code; check it against your
own machine once.

## If something's wrong

Whatever the installer reports, it's telling you in plain language. If it
still won't start, re-run it — most problems are permissions or a missing
`ffmpeg`, and both are safe to retry after fixing.

For what happens after you've captured some data — building datasets,
training, evaluating — see the main [README](../README.md). Those still need
a command line and, eventually, a GPU; there isn't a one-click path for them
yet because there isn't a finished product on the other end of them yet.
