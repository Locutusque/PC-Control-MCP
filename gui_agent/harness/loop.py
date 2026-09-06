"""The closed-loop control harness (plan section 6.2).

Runs the policy against the live screen until the subtask is done, stuck, or
out of time, and returns a single result.  The orchestrating LLM never sees the
per-frame trace -- that is the whole token and latency win.

Stop conditions (plan 6.3)
--------------------------
* ``DONE`` from the policy, then an independent verifier check.
* ``ESCALATE`` from the policy.
* No visual change for K consecutive ticks -- a heuristic deliberately
  independent of the model's own judgement, since a policy stuck in a loop is
  the case where its judgement is least reliable.
* The same action repeated N times, which catches a policy hammering a control
  on a screen that is animating and so never hashes equal.
* Timeout.
* Repeated safety blocks: if the guard keeps refusing, the policy is trying to
  do something it must not, and grinding away at it is worse than escalating.

Every tick is recorded in a :class:`RolloutTrace`.  That trace is the raw
material for stage-3 DAgger (plan 4.4): closed-loop failures are exactly the
states pure imitation learning never sees.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Sequence

from ..actions import Action, ActionType
from ..config import HarnessConfig
from .dispatch import DispatchResult, Dispatcher
from .verifier import AlwaysTrueVerifier, Verification, Verifier

log = logging.getLogger(__name__)

__all__ = ["SubtaskStatus", "TickRecord", "RolloutTrace", "SubtaskResult", "ControlLoop",
           "frame_hash", "hamming"]


class SubtaskStatus(str, Enum):
    DONE = "done"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class TickRecord:
    """One control tick, kept for DAgger relabelling."""

    index: int
    timestamp: float
    atoms: list[str]
    action_summary: str | None
    dispatched: bool
    blocked_reason: str | None = None
    decode_ms: float = 0.0
    dispatch_ms: float = 0.0
    frame_changed: bool = True
    app_context: str | None = None
    # Frames are large; the trace stores indices and the loop keeps only a
    # bounded ring of actual pixels.
    frame_ref: int | None = None


@dataclass
class RolloutTrace:
    rollout_id: str
    instruction: str
    form_data: dict = field(default_factory=dict)
    ticks: list[TickRecord] = field(default_factory=list)
    status: SubtaskStatus | None = None
    stop_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "rollout_id": self.rollout_id,
            "instruction": self.instruction,
            "form_data": self.form_data,
            "status": self.status.value if self.status else None,
            "stop_reason": self.stop_reason,
            "ticks": [t.__dict__ for t in self.ticks],
        }

    def latency_percentiles(self) -> dict:
        """p50/p95 decode latency -- the metric plan section 5 asks for."""
        values = sorted(t.decode_ms for t in self.ticks)
        if not values:
            return {}
        return {
            "n": len(values),
            "p50_ms": round(_percentile(values, 0.50), 2),
            "p95_ms": round(_percentile(values, 0.95), 2),
            "max_ms": round(values[-1], 2),
        }


@dataclass
class SubtaskResult:
    """Exactly what the MCP tool returns (plan 6.1)."""

    status: SubtaskStatus
    summary: str
    final_screenshot: object = None
    stop_reason: str | None = None
    n_ticks: int = 0
    elapsed_s: float = 0.0
    verification: Verification | None = None
    trace: RolloutTrace | None = None

    def to_payload(self, include_screenshot: bool = True) -> dict:
        payload = {
            "status": self.status.value,
            "summary": self.summary,
            "stop_reason": self.stop_reason,
            "n_ticks": self.n_ticks,
            "elapsed_s": round(self.elapsed_s, 2),
        }
        if self.verification is not None:
            payload["verified"] = {
                "checked": self.verification.checked,
                "met": self.verification.met,
                "reason": self.verification.reason,
            }
        if include_screenshot and self.final_screenshot is not None:
            payload["final_screenshot"] = self.final_screenshot
        return payload


class ControlLoop:
    """Drives the policy against the live screen."""

    def __init__(
        self,
        policy,
        dispatcher: Dispatcher,
        capture: Callable[[], object],
        config: HarnessConfig | None = None,
        verifier: Verifier | None = None,
        foreground: Callable[[], str | None] | None = None,
    ) -> None:
        self.policy = policy
        self.dispatcher = dispatcher
        self.capture = capture
        self.config = config or HarnessConfig()
        self.verifier = verifier or AlwaysTrueVerifier()
        self.foreground = foreground or _default_foreground()

    def run(
        self,
        instruction: str,
        form_data: dict | None = None,
        timeout_s: float | None = None,
        success_criteria: str = "",
    ) -> SubtaskResult:
        cfg = self.config
        timeout_s = timeout_s if timeout_s is not None else cfg.default_timeout_s
        form_data = dict(form_data or {})
        trace = RolloutTrace(f"roll_{uuid.uuid4().hex[:12]}", instruction, form_data)

        self.dispatcher.reset_subtask()
        session = self._make_session(instruction, form_data)

        history: list[list[str]] = []
        last_hash = None
        unchanged = 0
        repeat_run = 0
        last_atoms: list[str] | None = None
        blocked_run = 0
        frame = None
        started = time.monotonic()
        period = 1.0 / cfg.target_hz if cfg.target_hz > 0 else 0.0
        index = 0

        try:
            while True:
                elapsed = time.monotonic() - started
                if elapsed >= timeout_s:
                    return self._finish(
                        SubtaskStatus.TIMEOUT, trace, frame, index, elapsed,
                        f"timed out after {elapsed:.1f}s",
                    )

                tick_started = time.monotonic()
                frame = self.capture()
                app = self.foreground()

                decode_started = time.monotonic()
                result = self.policy.act(
                    frame, instruction, form_data, history, session=session
                )
                decode_ms = (time.monotonic() - decode_started) * 1000

                if not result.ok:
                    # The grammar makes malformed actions unreachable, so this
                    # means the decoder hit its budget -- treat it as being
                    # stuck rather than retrying forever.
                    log.warning("undecodable action at tick %d: %s", index, result.error)
                    trace.ticks.append(
                        TickRecord(index, time.time(), result.atoms, None, False,
                                   blocked_reason=result.error, decode_ms=decode_ms,
                                   app_context=app)
                    )
                    return self._finish(
                        SubtaskStatus.ESCALATED, trace, frame, index + 1,
                        time.monotonic() - started,
                        f"policy produced no valid action: {result.error}",
                    )

                action = result.action
                current_hash = frame_hash(frame)
                changed = (
                    last_hash is None
                    or hamming(current_hash, last_hash) > cfg.frame_hash_threshold
                )

                if action.type is ActionType.DONE:
                    trace.ticks.append(
                        TickRecord(index, time.time(), result.atoms, action.summary(),
                                   False, decode_ms=decode_ms, frame_changed=changed,
                                   app_context=app)
                    )
                    verification = self._verify(success_criteria, frame, instruction)
                    status = (
                        SubtaskStatus.DONE if verification.met else SubtaskStatus.ESCALATED
                    )
                    reason = (
                        "policy reported done"
                        if verification.met
                        else f"policy reported done but the criteria were not met: {verification.reason}"
                    )
                    return self._finish(
                        status, trace, frame, index + 1, time.monotonic() - started,
                        reason, verification,
                    )

                if action.type is ActionType.ESCALATE:
                    trace.ticks.append(
                        TickRecord(index, time.time(), result.atoms, action.summary(),
                                   False, decode_ms=decode_ms, frame_changed=changed,
                                   app_context=app)
                    )
                    return self._finish(
                        SubtaskStatus.ESCALATED, trace, frame, index + 1,
                        time.monotonic() - started, "policy escalated",
                    )

                dispatch_started = time.monotonic()
                dispatched: DispatchResult = self.dispatcher.send(action, app, form_data)
                dispatch_ms = (time.monotonic() - dispatch_started) * 1000

                trace.ticks.append(
                    TickRecord(
                        index, time.time(), result.atoms, action.summary(),
                        dispatched.dispatched,
                        # `is not None`, not truthiness: Verdict.__bool__ is
                        # its `allowed` flag, so a blocked verdict is falsy and
                        # a truthiness test would drop the very reason we need.
                        blocked_reason=_block_reason(dispatched),
                        decode_ms=decode_ms, dispatch_ms=dispatch_ms,
                        frame_changed=changed, app_context=app,
                    )
                )
                history.append(result.atoms)
                index += 1

                # -- stop conditions ---------------------------------------
                blocked_run = 0 if dispatched.dispatched else blocked_run + 1
                if blocked_run >= _MAX_BLOCKED_RUN:
                    return self._finish(
                        SubtaskStatus.ESCALATED, trace, frame, index,
                        time.monotonic() - started,
                        f"{blocked_run} consecutive actions blocked by the safety guard "
                        f"({_block_reason(dispatched)})",
                    )

                # A WAIT is *supposed* to leave the screen alone; counting it
                # as "stuck" would escalate out of every page load.
                if action.type is ActionType.WAIT:
                    unchanged = 0
                elif changed:
                    unchanged = 0
                else:
                    unchanged += 1
                    if unchanged >= cfg.stuck_frames:
                        return self._finish(
                            SubtaskStatus.ESCALATED, trace, frame, index,
                            time.monotonic() - started,
                            f"no visual change over {unchanged} consecutive ticks",
                        )

                # Identical actions on a screen that keeps animating hash as
                # "changed" forever, so repetition needs its own detector.
                repeat_run = repeat_run + 1 if result.atoms == last_atoms else 0
                if repeat_run >= _MAX_REPEAT_RUN and action.type is not ActionType.WAIT:
                    return self._finish(
                        SubtaskStatus.ESCALATED, trace, frame, index,
                        time.monotonic() - started,
                        f"repeated {action.summary()} {repeat_run + 1} times with no progress",
                    )
                last_atoms = result.atoms
                last_hash = current_hash

                remaining = period - (time.monotonic() - tick_started)
                if remaining > 0:
                    time.sleep(remaining)

        except Exception as exc:
            log.exception("control loop failed")
            return self._finish(
                SubtaskStatus.ERROR, trace, frame, index, time.monotonic() - started,
                f"harness error: {exc}",
            )
        finally:
            # Never leave a mouse button held down or a drag open once the
            # harness has handed control back.
            self.dispatcher.reset_subtask()

    # -- helpers ----------------------------------------------------------
    def _make_session(self, instruction: str, form_data: dict):
        factory = getattr(self.policy, "make_session", None)
        if factory is not None:
            return factory(instruction, form_data)
        try:
            from ..model.policy import PolicySession

            return PolicySession(self.policy, instruction, form_data).warm()
        except Exception as exc:
            log.debug("could not build a cached session (%s); running uncached", exc)
            return None

    def _verify(self, criteria: str, frame, instruction: str) -> Verification:
        if not self.config.verify_success or not criteria:
            return Verification.unchecked("success_criteria not checked")
        return self.verifier.check(criteria, frame, instruction)

    def _finish(
        self,
        status: SubtaskStatus,
        trace: RolloutTrace,
        frame,
        n_ticks: int,
        elapsed: float,
        reason: str,
        verification: Verification | None = None,
    ) -> SubtaskResult:
        trace.status = status
        trace.stop_reason = reason
        log.info("subtask %s after %d ticks (%.1fs): %s", status.value, n_ticks, elapsed, reason)
        return SubtaskResult(
            status=status,
            summary=_summarise(status, reason, trace),
            final_screenshot=frame,
            stop_reason=reason,
            n_ticks=n_ticks,
            elapsed_s=elapsed,
            verification=verification,
            trace=trace,
        )


_MAX_BLOCKED_RUN = 5
_MAX_REPEAT_RUN = 8


def _block_reason(result: DispatchResult) -> str | None:
    """Why a dispatch did not happen."""
    if result.verdict is not None and not result.verdict.allowed:
        return result.verdict.reason
    return result.error


def _default_foreground() -> Callable[[], str | None]:
    """Query the platform for the foreground app.

    The safety guard refuses to act on an unidentifiable window, so a provider
    that always answers ``None`` would block every action. Ask the platform;
    only genuinely unavailable window info should fail closed.
    """
    from ..capture.platform_ import get_backend

    backend = get_backend()

    def foreground() -> str | None:
        try:
            return backend.foreground_window().app
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("foreground lookup failed: %s", exc)
            return None

    return foreground


def _summarise(status: SubtaskStatus, reason: str, trace: RolloutTrace) -> str:
    """The one line the orchestrating LLM actually reads."""
    kinds: dict[str, int] = {}
    for tick in trace.ticks:
        if tick.action_summary:
            kind = tick.action_summary.split("(")[0]
            kinds[kind] = kinds.get(kind, 0) + 1
    breakdown = ", ".join(f"{n}x {k}" for k, n in sorted(kinds.items(), key=lambda kv: -kv[1])[:4])
    return f"{status.value}: {reason}" + (f" [{breakdown}]" if breakdown else "")


# --------------------------------------------------------------------------
# Frame hashing
# --------------------------------------------------------------------------


def frame_hash(frame, size: int = 16) -> int:
    """Difference hash of a frame.

    A perceptual hash rather than an exact one, so a blinking text caret or a
    one-pixel antialiasing difference does not read as "the screen changed" and
    defeat the stuck detector.
    """
    import numpy as np

    array = np.asarray(frame)
    if array.ndim == 3:
        array = array[:, :, :3].mean(axis=2)
    h, w = array.shape[:2]
    # Nearest-neighbour downsample: cheap, and adequate for a change detector.
    rows = np.linspace(0, h - 1, size).astype(int)
    cols = np.linspace(0, w - 1, size + 1).astype(int)
    small = array[np.ix_(rows, cols)]
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(q * (len(sorted_values) - 1) + 0.5), len(sorted_values) - 1)
    return sorted_values[index]
