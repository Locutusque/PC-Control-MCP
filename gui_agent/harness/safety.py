"""Guardrails at the input-dispatch layer (plan section 6.4).

The orchestrating LLM sees a request and a result, never the individual
actions.  Nothing reviews what the policy emits before it reaches the operating
system -- so these checks are the last line, and they are enforced on the way
*out* regardless of what the model predicted or what it was trained to do.

Every check is a property of the action and the current screen state, not of
the model's confidence.  A policy that has generalised badly is exactly the
case where its own judgement cannot be trusted, so the guard never consults it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..actions import Action, ActionCodec, ActionType
from ..config import SafetyConfig

log = logging.getLogger(__name__)

__all__ = ["Verdict", "SafetyViolation", "ActionGuard", "detect_sandbox"]


class SafetyViolation(RuntimeError):
    """A guarantee the harness must not run without."""


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str | None = None
    detail: str | None = None

    def __bool__(self) -> bool:
        return self.allowed


ALLOW = Verdict(True)


@dataclass
class GuardStats:
    dispatched: int = 0
    blocked: int = 0
    by_reason: dict = field(default_factory=dict)

    def record_block(self, reason: str) -> None:
        self.blocked += 1
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1


class ActionGuard:
    """Screens actions before they reach the OS."""

    def __init__(
        self,
        config: SafetyConfig | None = None,
        codec: ActionCodec | None = None,
        screen_size: tuple[int, int] | None = None,
    ) -> None:
        self.config = config or SafetyConfig()
        self.codec = codec or ActionCodec()
        self.screen_size = screen_size
        self.stats = GuardStats()
        self._text_patterns = [re.compile(p, re.IGNORECASE) for p in self.config.blocked_text_patterns]
        self._blocked_apps = tuple(a.lower() for a in self.config.blocked_apps)
        self._recent: deque[float] = deque(maxlen=256)
        self._audit_path = Path(self.config.audit_log) if self.config.audit_log else None
        if self._audit_path:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)

    # -- preflight --------------------------------------------------------
    def preflight(self) -> None:
        """Refuse to run outside a sandbox unless explicitly allowed."""
        if not self.config.require_sandbox:
            return
        sandbox = detect_sandbox()
        if not sandbox.allowed:
            raise SafetyViolation(
                f"{sandbox.reason}. The policy dispatches un-reviewed input to the real "
                "desktop, so run it in a VM or a scoped browser profile. To override "
                "deliberately, set safety.require_sandbox=false or export "
                "GUI_AGENT_ALLOW_UNSANDBOXED=1."
            )

    # -- per-action checks ------------------------------------------------
    def check(
        self,
        action: Action,
        foreground_app: str | None = None,
        form_data: dict | None = None,
        now: float | None = None,
    ) -> Verdict:
        now = now if now is not None else time.monotonic()

        for verdict in (
            self._check_budget(),
            self._check_rate(now),
            self._check_app(foreground_app),
            self._check_region(action),
            self._check_key(action),
            self._check_text(action, form_data),
        ):
            if not verdict.allowed:
                self.stats.record_block(verdict.reason or "unknown")
                self._audit(action, verdict, foreground_app)
                return verdict
        return ALLOW

    def _check_budget(self) -> Verdict:
        if self.stats.dispatched >= self.config.max_actions_per_subtask:
            return Verdict(
                False, "action_budget",
                f"{self.stats.dispatched} actions dispatched this subtask "
                f"(limit {self.config.max_actions_per_subtask})",
            )
        return ALLOW

    def _check_rate(self, now: float) -> Verdict:
        limit = self.config.max_actions_per_second
        if limit <= 0:
            return ALLOW
        window = [t for t in self._recent if now - t < 1.0]
        if len(window) >= limit:
            return Verdict(False, "rate_limit", f"{len(window)} actions in the last second")
        return ALLOW

    def _check_app(self, foreground_app: str | None) -> Verdict:
        if foreground_app is None:
            # Unknown foreground means the app blocklist cannot be enforced.
            # Refusing here is what stops a mis-click into a terminal from
            # being dispatched simply because we could not see where we were.
            return Verdict(False, "unknown_foreground", "cannot identify the foreground window")
        needle = foreground_app.lower()
        for blocked in self._blocked_apps:
            if blocked in needle:
                return Verdict(False, "blocked_app", f"{foreground_app!r} matches {blocked!r}")
        return ALLOW

    def _check_region(self, action: Action) -> Verdict:
        region = self.config.allowed_region
        if region is None or not action.type.is_absolute_pointer:
            return ALLOW
        if self.screen_size is None:
            return Verdict(False, "unknown_screen", "allowed_region set but screen size unknown")
        x, y = self.codec.to_pixels(action, *self.screen_size)
        left, top, right, bottom = region
        if not (left <= x < right and top <= y < bottom):
            return Verdict(False, "outside_region", f"({x}, {y}) is outside {region}")
        return ALLOW

    def _check_key(self, action: Action) -> Verdict:
        if action.type is ActionType.KEY and action.key in self.config.blocked_keys:
            return Verdict(False, "blocked_key", action.key)
        return ALLOW

    def _check_text(self, action: Action, form_data: dict | None) -> Verdict:
        if action.type is not ActionType.TYPE:
            return ALLOW
        try:
            text = action.resolve_text(form_data)
        except Exception as exc:
            return Verdict(False, "unresolvable_text", str(exc))
        for pattern in self._text_patterns:
            if pattern.search(text):
                # Log the pattern, never the text: the text may be a password
                # the policy was handed as form_data.
                return Verdict(False, "blocked_text", f"matched {pattern.pattern!r}")
        return ALLOW

    # -- bookkeeping ------------------------------------------------------
    def note_dispatch(self, action: Action, now: float | None = None,
                      foreground_app: str | None = None) -> None:
        now = now if now is not None else time.monotonic()
        self._recent.append(now)
        self.stats.dispatched += 1
        self._audit(action, ALLOW, foreground_app)

    def reset_subtask(self) -> None:
        self.stats = GuardStats()
        self._recent.clear()

    def _audit(self, action: Action, verdict: Verdict, foreground_app: str | None) -> None:
        """Append-only log of every dispatched and blocked action.

        This is what makes stage-3 rollout collection reviewable after the
        fact, which matters most precisely when the policy is at its worst.
        """
        if self._audit_path is None:
            return
        record = {
            "t": round(time.time(), 3),
            "action": action.summary(),
            "type": action.type.value,
            "allowed": verdict.allowed,
            "reason": verdict.reason,
            "detail": verdict.detail,
            "app": foreground_app,
        }
        try:
            with self._audit_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        except OSError as exc:  # pragma: no cover - disk full, read-only fs
            log.warning("could not write audit log: %s", exc)


# --------------------------------------------------------------------------
# Sandbox detection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxVerdict:
    allowed: bool
    reason: str


def detect_sandbox() -> SandboxVerdict:
    """Best-effort check that we are not driving someone's real desktop.

    Deliberately conservative and deliberately overridable: it cannot prove
    isolation, so it makes the operator state their intent rather than
    pretending to have verified something it has not.
    """
    if os.environ.get("GUI_AGENT_ALLOW_UNSANDBOXED") == "1":
        return SandboxVerdict(True, "explicitly allowed via GUI_AGENT_ALLOW_UNSANDBOXED")
    if os.environ.get("GUI_AGENT_SANDBOX"):
        return SandboxVerdict(True, "GUI_AGENT_SANDBOX is set")

    if Path("/.dockerenv").exists():
        return SandboxVerdict(True, "running inside a container")
    try:
        vendor = Path("/sys/class/dmi/id/sys_vendor").read_text().strip().lower()
        if any(v in vendor for v in ("qemu", "vmware", "virtualbox", "innotek", "xen", "kvm")):
            return SandboxVerdict(True, f"virtual machine ({vendor})")
    except OSError:
        pass

    return SandboxVerdict(False, "could not confirm this is a VM, container or scoped profile")
