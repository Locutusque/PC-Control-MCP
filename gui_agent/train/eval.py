"""Offline evaluation against the held-out sets (plan section 5).

Three suites, matching ``eval_sets/``:

* ``click_accuracy`` -- screenshots with the intended element's bounding box.
* ``field_fill`` -- forms with ``form_data`` and the exact value expected.
* ``end_to_end_tasks`` -- run live through the harness; the only suite that
  measures whether the thing actually works.

The first two run offline against a checkpoint and are cheap enough for every
training run.  The third needs a live desktop.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import torch

from ..actions import Action, ActionCodec
from ..capture.schema import read_jsonl
from .metrics import (
    ClickTarget, action_type_accuracy, click_accuracy, escalation_metrics,
    field_fill_exactness, task_success_rate,
)

log = logging.getLogger(__name__)

__all__ = ["evaluate_click_accuracy", "evaluate_field_fill", "evaluate_end_to_end", "evaluate_all"]


def _load_frame(path: str | Path):
    from PIL import Image
    import numpy as np

    return np.asarray(Image.open(path).convert("RGB"))


@torch.no_grad()
def evaluate_click_accuracy(policy, eval_dir: str | Path) -> dict:
    """Each row: ``{screenshot, instruction, bbox, [form_data]}``."""
    rows = _load_suite(eval_dir, "click_accuracy")
    if not rows:
        return {}

    predictions: list[Action | None] = []
    targets: list[ClickTarget] = []
    reference: list[Action] = []
    codec = ActionCodec(policy.config.action_space)
    policy.eval()

    for row in rows:
        frame = _load_frame(Path(eval_dir) / row["screenshot"])
        height, width = frame.shape[:2]
        result = policy.act(frame, row["instruction"], row.get("form_data"), row.get("history", []))
        predictions.append(result.action if result.ok else None)
        targets.append(
            ClickTarget(
                screen_w=row.get("screen_w", width),
                screen_h=row.get("screen_h", height),
                bbox=tuple(row["bbox"]) if row.get("bbox") else None,
                point=tuple(row["point"]) if row.get("point") else None,
                tolerance_px=row.get("tolerance_px", 24.0),
                element_id=row.get("element_id"),
            )
        )
        if row.get("target"):
            reference.append(codec.decode(row["target"], list((row.get("form_data") or {}).keys())))

    out = click_accuracy(predictions, targets, codec)
    if reference:
        out["action_type"] = action_type_accuracy(predictions[: len(reference)], reference)
    return out


@torch.no_grad()
def evaluate_field_fill(policy, eval_dir: str | Path) -> dict:
    """Each row: ``{screenshot, instruction, form_data, expected_value}``.

    Exactness below 1.0 is a bug, not a quality shortfall -- the ``<FIELD_k>``
    routing is supposed to make drift impossible, so a miss here means the
    model routed to free-compose instead.
    """
    rows = _load_suite(eval_dir, "field_fill")
    if not rows:
        return {}

    predictions: list[Action | None] = []
    expected: list[str] = []
    form_data: list[dict] = []
    policy.eval()

    for row in rows:
        frame = _load_frame(Path(eval_dir) / row["screenshot"])
        data = row.get("form_data", {})
        result = policy.act(frame, row["instruction"], data, row.get("history", []))
        predictions.append(result.action if result.ok else None)
        expected.append(row["expected_value"])
        form_data.append(data)

    return field_fill_exactness(predictions, expected, form_data)


def evaluate_end_to_end(loop, eval_dir: str | Path) -> dict:
    """Run each task live through the harness.

    Also produces the escalation slice: a task that failed should have been
    escalated, one that succeeded should not have been.
    """
    from .stage3_dagger import Task

    rows = _load_suite(eval_dir, "end_to_end_tasks")
    if not rows:
        return {}

    statuses: list[str] = []
    verified: list[bool] = []
    escalated: list[bool] = []
    should_escalate: list[bool] = []

    for row in rows:
        task = Task.from_dict(row)
        result = loop.run(task.instruction, task.form_data, task.timeout_s, task.success_criteria)
        statuses.append(result.status.value)
        verified.append(bool(result.verification.met) if result.verification else True)
        escalated.append(result.status.value == "escalated")
        # Ground truth from the eval set where given, otherwise: a task that
        # did not complete is one the policy should have handed back.
        should_escalate.append(
            bool(row["should_escalate"]) if "should_escalate" in row
            else result.status.value != "done"
        )

    out = task_success_rate(statuses, verified)
    out["escalation"] = escalation_metrics(escalated, should_escalate)
    return out


def evaluate_all(policy, eval_root: str | Path = "eval_sets", loop=None) -> dict:
    results: dict = {}
    results["click_accuracy"] = evaluate_click_accuracy(policy, eval_root)
    results["field_fill"] = evaluate_field_fill(policy, eval_root)
    if loop is not None:
        results["end_to_end"] = evaluate_end_to_end(loop, eval_root)
    else:
        results["end_to_end"] = {
            "skipped": "needs a live harness; pass loop= to measure task success"
        }
    return results


def _load_suite(eval_dir: str | Path, name: str) -> list[dict]:
    path = Path(eval_dir) / name / "eval.jsonl"
    if not path.exists():
        log.warning("no evaluation suite at %s", path)
        return []
    return list(read_jsonl(path))
