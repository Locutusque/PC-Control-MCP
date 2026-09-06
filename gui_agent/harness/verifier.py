"""Checking ``success_criteria`` before reporting a subtask done (plan 6.1).

``DONE`` is the policy's own opinion, and a policy that has drifted will emit
it confidently at the wrong moment.  The verifier is a second, independent
check: it looks at the final screen and decides whether the stated criteria are
actually met.

Three implementations, in increasing cost:

* :class:`AlwaysTrueVerifier` -- trusts the policy.  The default only because a
  verifier needs an API key; say so in the returned summary rather than letting
  a caller assume verification happened.
* :class:`CallableVerifier` -- wrap your own check (a DOM query, an HTTP call,
  a database read).  Prefer this where the task has a machine-checkable
  outcome; it is cheaper and far more reliable than looking at pixels.
* :class:`LLMVerifier` -- a vision model reads the criteria and the final
  screenshot.  The general fallback, and the one that makes end-to-end task
  success measurable on tasks with no programmatic check.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

__all__ = ["Verification", "Verifier", "AlwaysTrueVerifier", "CallableVerifier", "LLMVerifier"]

DEFAULT_MODEL = "claude-opus-5"

_SCHEMA = {
    "type": "object",
    "properties": {
        "met": {"type": "boolean", "description": "Whether the criteria are satisfied."},
        "reason": {"type": "string", "description": "One sentence of evidence from the screen."},
    },
    "required": ["met", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Verification:
    met: bool
    reason: str = ""
    checked: bool = True

    @classmethod
    def unchecked(cls, reason: str) -> Verification:
        return cls(met=True, reason=reason, checked=False)


class Verifier:
    def check(self, criteria: str, screenshot, instruction: str = "") -> Verification:
        raise NotImplementedError


class AlwaysTrueVerifier(Verifier):
    """No verification; the policy's DONE is taken at face value."""

    def check(self, criteria: str, screenshot, instruction: str = "") -> Verification:
        return Verification.unchecked("no verifier configured; DONE was not independently checked")


class CallableVerifier(Verifier):
    """Delegates to a user-supplied predicate."""

    def __init__(self, fn: Callable[[str, object, str], bool | Verification]) -> None:
        self.fn = fn

    def check(self, criteria: str, screenshot, instruction: str = "") -> Verification:
        try:
            result = self.fn(criteria, screenshot, instruction)
        except Exception as exc:
            log.warning("verifier raised: %s", exc)
            return Verification(False, f"verifier error: {exc}")
        if isinstance(result, Verification):
            return result
        return Verification(bool(result), "custom verifier")


class LLMVerifier(Verifier):
    """Vision-model check of the final screen against the criteria."""

    def __init__(self, model: str = DEFAULT_MODEL, client=None, max_tokens: int = 512) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic()
        return self._client

    def check(self, criteria: str, screenshot, instruction: str = "") -> Verification:
        if not criteria:
            return Verification.unchecked("no success_criteria supplied")
        try:
            png = _to_png(screenshot)
        except Exception as exc:
            return Verification(False, f"could not encode the final screenshot: {exc}")

        prompt = (
            "A GUI automation policy has just finished a subtask and claims it is done.\n\n"
            f"Subtask: {instruction or '(not given)'}\n"
            f"Success criteria: {criteria}\n\n"
            "The image is the final state of the screen. Decide whether the success "
            "criteria are met by what is visible. Judge only the criteria as written; "
            "do not infer intermediate steps you cannot see. If the screen shows an "
            "error, an unsaved form, or a dialog still open, the criteria are not met."
        )
        try:
            response = self._get_client().messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64.standard_b64encode(png).decode(),
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
            )
        except Exception as exc:
            log.warning("verification request failed: %s", exc)
            # A failed check is not a passed check: report it as unmet with the
            # reason, so a broken API key cannot silently turn every run green.
            return Verification(False, f"verification unavailable: {exc}")

        if response.stop_reason == "refusal":
            return Verification(False, "verification declined by the safety classifier")
        payload = next(b.text for b in response.content if b.type == "text")
        data = json.loads(payload)
        return Verification(bool(data.get("met")), str(data.get("reason", "")))


def _to_png(image) -> bytes:
    from PIL import Image  # type: ignore

    if isinstance(image, bytes):
        return image
    array = image
    if hasattr(array, "detach"):  # torch tensor
        array = array.detach().cpu().numpy()
    import numpy as np

    array = np.asarray(array)
    if array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[2] > 4:
        array = array.transpose(1, 2, 0)  # CHW -> HWC
    if array.dtype != np.uint8:
        array = (array.clip(0, 1) * 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()
