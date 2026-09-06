"""Control-tick latency benchmark (plan sections 5 and 8).

Plan section 8 says the base LM should be chosen by benchmarking quantised
latency on the target hardware, not by parameter count on paper.  This is that
benchmark, and it is meant to be run *before* committing to a base model.

It separates the three costs that make up a tick, because they have different
fixes:

* **vision** -- the ViT forward.  Shrink the patch grid or the ViT.
* **prefill** -- attending over the image tokens.  This is what the Perceiver
  resampler exists to cut.
* **decode** -- the 2-4 forced action tokens.  This is what quantisation helps.

A single aggregate number hides which one is the problem.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

log = logging.getLogger(__name__)

__all__ = ["LatencyReport", "benchmark_policy"]


@dataclass
class LatencyReport:
    samples_ms: list[float] = field(default_factory=list)
    vision_ms: list[float] = field(default_factory=list)
    device: str = "cpu"
    dtype: str = "float32"
    n_image_tokens: int = 0
    target_hz: float = 15.0

    def summary(self) -> dict:
        total = sorted(self.samples_ms)
        vision = sorted(self.vision_ms)
        if not total:
            return {}
        p50, p95 = _percentile(total, 0.5), _percentile(total, 0.95)
        budget_ms = 1000.0 / self.target_hz if self.target_hz else float("inf")
        return {
            "device": self.device,
            "dtype": self.dtype,
            "n_image_tokens": self.n_image_tokens,
            "n_samples": len(total),
            "p50_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "max_ms": round(total[-1], 2),
            "vision_p50_ms": round(_percentile(vision, 0.5), 2) if vision else None,
            "achievable_hz_p50": round(1000.0 / p50, 1) if p50 else None,
            "achievable_hz_p95": round(1000.0 / p95, 1) if p95 else None,
            "target_hz": self.target_hz,
            # p95, not p50: a control loop that misses its budget one tick in
            # twenty is a control loop that stutters.
            "meets_target": p95 <= budget_ms,
            "budget_ms": round(budget_ms, 2),
        }


@torch.no_grad()
def benchmark_policy(
    policy,
    instruction: str = "click the submit button",
    form_data: dict | None = None,
    n_warmup: int = 5,
    n_samples: int = 50,
    target_hz: float = 15.0,
    frame=None,
) -> LatencyReport:
    """Measure per-tick latency the way the harness will actually run it."""
    import numpy as np

    from ..model.policy import PolicySession

    device = next(policy.parameters()).device
    policy.eval()

    if frame is None:
        # Random pixels rather than a flat image: a constant frame can be
        # unrepresentatively fast on hardware that skips work on uniform input.
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    session = PolicySession(policy, instruction, form_data).warm(device)
    report = LatencyReport(
        device=str(device),
        dtype=str(next(policy.parameters()).dtype).replace("torch.", ""),
        target_hz=target_hz,
    )

    history: list[list[str]] = []
    for index in range(n_warmup + n_samples):
        _sync(device)
        started = time.perf_counter()

        vision_started = time.perf_counter()
        from ..model.vit import preprocess_screenshot

        pixels = preprocess_screenshot(frame, policy.config.vision).to(
            device=device, dtype=next(policy.parameters()).dtype
        )
        image = policy.encode_image(pixels)
        _sync(device)
        vision_ms = (time.perf_counter() - vision_started) * 1000

        result = policy.act(frame, instruction, form_data, history, session=session)
        _sync(device)
        total_ms = (time.perf_counter() - started) * 1000

        if index >= n_warmup:
            report.samples_ms.append(total_ms)
            report.vision_ms.append(vision_ms)
            report.n_image_tokens = image.shape[1]
        if result.ok and len(history) < policy.config.max_history:
            history.append(result.atoms)

    return report


def _sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(q * (len(sorted_values) - 1) + 0.5), len(sorted_values) - 1)
    return sorted_values[index]
