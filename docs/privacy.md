# Privacy handling for the capture pool

The capture daemon records everything you do on your computer: the screen, the
mouse, the keyboard, and which application had focus. In practice that archive
will contain material comparable to a password manager's contents — messages,
documents, account pages, anything that was on screen.

Treat it that way. This document is the handling guide; the enforcement lives in
`gui_agent/capture/privacy.py` and runs whether or not anyone reads this.

---

## The three defences, and when each fires

### 1. Foreground blocklist — before the frame is grabbed

`BlocklistGuard` is consulted **before** each capture, not after. Blocked pixels
never reach disk, not even for the milliseconds a post-hoc filter would take.

Blocked by default: password managers, banking and health domains, and window
titles containing "incognito", "private browsing", "password", "sign in".

**An unidentifiable foreground window counts as blocked.** If the daemon cannot
tell which application has focus, it cannot prove the window is safe, and an
unattributed frame is worth less than the risk of recording a credential
vault. The same rule applies to *stale* window information: the foreground
monitor runs on its own thread, and a reading older than 500ms is treated as
unknown.

Add your own before you record anything real:

```python
PrivacyConfig(
    app_blocklist=(*PrivacyConfig().app_blocklist, "signal", "my-therapy-app"),
    domain_blocklist=(*PrivacyConfig().domain_blocklist, "myclinic.example"),
)
```

### 2. Secure fields — content is never buffered

Keystrokes into a field the accessibility tree marks secure are recorded as a
character *count* and nothing else. The characters are never buffered, so they
cannot be flushed by a later code path.

Secure-field state comes from `AXIsSecureTextField` (macOS), UI Automation
`IsPassword` (Windows), or AT-SPI (Linux). **If the platform cannot answer, every
keystroke is treated as secure.** That loses typing data on a machine without
Accessibility permission, which is the correct trade — the alternative is
recording passwords on exactly the machines least able to tell you they are
passwords.

The trajectory keeps a `redacted_text` event with the count, so the shape of the
interaction survives. The encoder emits no action for it: we know a password was
typed and refuse to reconstruct it. A hole in the trajectory beats a guess.

### 3. Redaction sweep — before promotion, never in the hot path

A segment goes `pending -> clean | quarantined`, and only `clean` or `redacted`
segments are ever promoted into the training pool. The sweep reads both the
recorded keystrokes and the recorded *frames* — OCR plus regex (card numbers
with a Luhn check, SSNs, phone numbers, emails, IBANs, API keys) — and runs as
a batch job, because OCR at 15Hz would eat the entire frame budget.

Frames are deduplicated by perceptual hash before OCR rather than sampled
every Nth: at 15Hz consecutive frames are near-identical, so deduplication
cuts a 300-frame segment to a few dozen distinct screens while still examining
every screen that actually appeared. Blind sampling would skip whole screens
that happened to fall between samples. It is still a filter and not a proof —
`ocr_max_frames` caps the work, and OCR misses text it cannot read.

**The sweep refuses to pass anything it could not read.** Segments are
encrypted at rest by default, so it decrypts them to a temporary file, scans,
and discards the plaintext; a missing key, an unreadable file, or a crashed
scan all quarantine the segment. It never treats "scanned nothing" as
"found nothing".

Findings quarantine the segment rather than being blurred out, because a
finding in *typed text* has no screen region to blur. With no OCR backend
installed, every segment is quarantined rather than promoted: it is better to
collect nothing than to promote unscanned frames.

Note that `pip install pytesseract` installs the Python wrapper, **not**
tesseract itself. Install the binary too (`brew install tesseract` on macOS,
`apt install tesseract-ocr` on Debian/Ubuntu) — until you do, `promote` will
tell you OCR is unavailable and quarantine everything.

Dataset builders default to `require_promoted=True`. Overriding that is a
one-flag mistake, which is why it defaults the safe way.

---

## Storage

**Encrypted at rest, local only.** `SegmentEncryptor` uses Fernet with a key from
`GUI_AGENT_CAPTURE_KEY`.

```bash
python -m gui_agent.capture.cli keygen   # store the output in your OS keychain
```

If `encrypt_at_rest` is on and `cryptography` is missing, broken, or the key is
unset, the daemon **refuses to start**. It does not fall back to plaintext. A
capture archive quietly written in the clear is precisely the failure the
setting exists to prevent — including when the library is half-installed and
raises something that is not `ImportError`.

`allow_network_upload` defaults to `False`. There is no code path in this
repository that uploads capture data.

---

## Retention

Raw segments are deleted `raw_retention_days` (default 14) after they were
processed into training examples:

```bash
python -m gui_agent.capture.cli prune --dry-run
python -m gui_agent.capture.cli prune
```

Only *promoted* segments expire. An unpromoted segment is never silently
deleted — that would hide a stuck redaction pipeline behind a shrinking disk.

Without retention the sensitive archive grows forever, and its worst day is the
day it is largest.

---

## The daemon is visible on purpose

A tray indicator shows whenever recording is live, `Ctrl+Alt+P` pauses globally,
and if `pystray` is unavailable the daemon prints a periodic console banner
rather than showing nothing. Do not remove this. Even for personal-only use it
is the right default, and it makes debugging far easier — "was it recording?" is
otherwise a question you cannot answer.

---

## Before you record anything real

```bash
python -m gui_agent.capture.cli doctor
```

This reports whether the blocklist is enforceable, whether secure-field
detection works, and whether segments can be encrypted, and it **refuses to
start** rather than recording without a guarantee the configuration claims to
provide. Take its output seriously; on a machine where it reports
`secure_field_detection NO`, the daemon will record no typed text at all, which
is the correct behaviour but not an obviously useful dataset.

Then, in order:

1. Add your own apps and domains to the blocklists.
2. Generate and store a capture key.
3. Install an OCR backend (`pip install 'gui-agent[redaction]'`) or accept that
   everything will quarantine.
4. Record a short session and **read it back** — `capture.cli status`, then open
   the records. Confirm with your own eyes that nothing you did not intend to
   record is in there.
5. Only then leave it running.

Step 4 is not optional. Every guarantee here is a claim about code; check it
against your own machine once before you trust it with months of your life.

---

## Sharing

Don't. If you must:

- never share raw segments, only derived training examples;
- re-run the redaction sweep with a stricter pattern set first;
- remember that screenshots contain everything that was on screen, including
  windows behind the one you were working in and notifications that arrived
  mid-frame — the sweep looks for patterns it knows, not for everything you
  would not want seen.

`eval_sets/*/frames/` is gitignored for this reason: screenshots of real
applications are captured data and fall under the same rules as the pool.
