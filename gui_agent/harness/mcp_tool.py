"""The ``execute_low_level_task`` MCP tool (plan section 6.1).

This is the entire interface between the orchestrating LLM and the policy.  The
orchestrator sends one request and gets one result back; the hundreds of control
ticks in between never enter its context.  That is the token and latency win the
whole design exists for, so the payload is deliberately small: a status, a
one-line summary, and the final screenshot.

Returning the per-frame trace here would undo the benefit, so it is written to
disk for stage-3 DAgger instead and referenced by id.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import HarnessConfig
from .loop import ControlLoop, SubtaskResult, SubtaskStatus

log = logging.getLogger(__name__)

__all__ = ["TOOL_NAME", "TOOL_DESCRIPTION", "TOOL_SCHEMA", "LowLevelTaskExecutor"]

TOOL_NAME = "execute_low_level_task"

TOOL_DESCRIPTION = (
    "Hand a GUI subtask to the fast local policy, which runs closed-loop against the "
    "screen until it is done, gets stuck, or times out. Use this for concrete, "
    "short-horizon interactions ('fill in the checkout form', 'open the settings "
    "panel and enable dark mode'), not for open-ended goals. Supply exact values the "
    "task needs through form_data: those are copied verbatim rather than retyped from "
    "memory, so they cannot drift. Returns a status, a one-line summary and the final "
    "screenshot; a status of 'escalated' means the policy handed control back and the "
    "screenshot shows where it stopped."
)

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": "The subtask to perform, phrased as an instruction.",
        },
        "form_data": {
            "type": "object",
            "description": (
                "Exact values for fields the task must fill, e.g. "
                '{"email": "j.doe@example.com"}. Copied verbatim into the matching '
                "field; anything the policy has to compose itself should not go here."
            ),
            "additionalProperties": {"type": "string"},
        },
        "timeout_s": {
            "type": "number",
            "description": "Give up after this many seconds. Defaults to 60.",
            "minimum": 1,
            "maximum": 600,
        },
        "success_criteria": {
            "type": "string",
            "description": (
                "What the screen should show when the task is complete. Checked "
                "independently before a result of 'done' is returned."
            ),
        },
    },
    "required": ["instruction"],
    "additionalProperties": False,
}


@dataclass
class ExecutorStats:
    calls: int = 0
    done: int = 0
    escalated: int = 0
    timeout: int = 0
    error: int = 0
    total_ticks: int = 0

    def record(self, result: SubtaskResult) -> None:
        self.calls += 1
        self.total_ticks += result.n_ticks
        setattr(self, result.status.value, getattr(self, result.status.value) + 1)

    def as_dict(self) -> dict:
        return {
            "calls": self.calls, "done": self.done, "escalated": self.escalated,
            "timeout": self.timeout, "error": self.error,
            "escalation_rate": round(self.escalated / self.calls, 4) if self.calls else None,
            "mean_ticks": round(self.total_ticks / self.calls, 1) if self.calls else None,
        }


class LowLevelTaskExecutor:
    """Owns the control loop and turns its result into a tool payload."""

    def __init__(
        self,
        loop: ControlLoop,
        config: HarnessConfig | None = None,
        trace_dir: str | Path | None = "runs/rollouts",
        return_screenshot: bool = True,
        screenshot_max_width: int = 1280,
    ) -> None:
        self.loop = loop
        self.config = config or HarnessConfig()
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.return_screenshot = return_screenshot
        self.screenshot_max_width = screenshot_max_width
        self.stats = ExecutorStats()
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        instruction: str,
        form_data: dict | None = None,
        timeout_s: float | None = None,
        success_criteria: str = "",
    ) -> dict:
        if not instruction or not instruction.strip():
            return {
                "status": SubtaskStatus.ERROR.value,
                "summary": "error: instruction is required",
            }

        started = time.time()
        result = self.loop.run(
            instruction=instruction,
            form_data=form_data,
            timeout_s=timeout_s,
            success_criteria=success_criteria,
        )
        self.stats.record(result)

        payload = result.to_payload(include_screenshot=False)
        trace_id = self._persist_trace(result)
        if trace_id:
            payload["rollout_id"] = trace_id
        if self.return_screenshot and result.final_screenshot is not None:
            try:
                payload["final_screenshot"] = self._encode_screenshot(result.final_screenshot)
            except Exception as exc:
                log.warning("could not encode the final screenshot: %s", exc)
                payload["final_screenshot_error"] = str(exc)
        log.info(
            "%s(%r) -> %s in %.1fs", TOOL_NAME, instruction[:60],
            result.status.value, time.time() - started,
        )
        return payload

    # -- helpers ----------------------------------------------------------
    def _persist_trace(self, result: SubtaskResult) -> str | None:
        """Write the per-frame trace to disk for stage-3 DAgger.

        Kept out of the return payload on purpose: the orchestrator must not
        pay for the trace it is the point of this tool to hide from it.
        """
        if self.trace_dir is None or result.trace is None:
            return None
        path = self.trace_dir / f"{result.trace.rollout_id}.json"
        try:
            path.write_text(json.dumps(result.trace.to_dict(), indent=2))
        except OSError as exc:  # pragma: no cover
            log.warning("could not write rollout trace: %s", exc)
            return None
        return result.trace.rollout_id

    def _encode_screenshot(self, frame) -> dict:
        from PIL import Image  # type: ignore
        import numpy as np

        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[2] == 4:
            array = array[:, :, :3]
        image = Image.fromarray(array.astype("uint8"))
        if image.width > self.screenshot_max_width:
            # A full-resolution screenshot is a large number of image tokens
            # for the orchestrator; it needs enough to see where things stand,
            # not pixel-perfect detail.
            ratio = self.screenshot_max_width / image.width
            image = image.resize(
                (self.screenshot_max_width, max(1, int(image.height * ratio))),
                Image.LANCZOS,
            )
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=True)
        return {
            "type": "image",
            "media_type": "image/png",
            "width": image.width,
            "height": image.height,
            "data": base64.standard_b64encode(buf.getvalue()).decode(),
        }

    def tool_definition(self) -> dict:
        """Anthropic Messages API tool definition for this executor."""
        return {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "input_schema": TOOL_SCHEMA,
        }
