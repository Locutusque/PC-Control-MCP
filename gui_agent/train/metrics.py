"""Evaluation metrics (plan section 5).

Defined before training rather than after, because each one measures a
different failure and they disagree in useful ways: a policy can have excellent
next-token accuracy and terrible click accuracy (it learned the action *types*
and not the grounding), or high task success and terrible escalation precision
(it never hands back, so failures are silent).

The four that matter most:

* :func:`click_accuracy` -- does the click land on the intended element?  Token
  accuracy will not tell you: being one grid cell off is one wrong token and a
  completely missed button.
* :func:`field_fill_exactness` -- should be 1.0.  Anything else is a bug, not a
  quality shortfall.  The ``<FIELD_k>`` routing makes it structural, so this
  metric mostly guards against the *routing* going wrong.
* :func:`escalation_metrics` -- both directions.  Under-escalating means silent
  task failures; over-escalating defeats the point of the whole system.
* latency p50/p95 on the quantised target hardware, from
  :func:`gui_agent.train.benchmark_latency`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from ..actions import Action, ActionCodec, ActionType

__all__ = [
    "ClickTarget", "click_accuracy", "field_fill_exactness", "action_type_accuracy",
    "escalation_metrics", "task_success_rate", "MetricAccumulator",
]


@dataclass(frozen=True)
class ClickTarget:
    """Ground truth for one click in the evaluation set.

    ``bbox`` is the intended UI element in screen pixels.  Without it the check
    degrades to a distance tolerance, which is weaker: a 40px miss is fine on a
    large button and fatal on a toolbar icon, and only the bbox knows which.
    """

    screen_w: int
    screen_h: int
    bbox: tuple[int, int, int, int] | None = None  # (left, top, right, bottom)
    point: tuple[int, int] | None = None
    tolerance_px: float = 24.0
    element_id: str | None = None

    def contains(self, x: float, y: float) -> bool:
        if self.bbox is not None:
            left, top, right, bottom = self.bbox
            return left <= x < right and top <= y < bottom
        if self.point is not None:
            dx, dy = x - self.point[0], y - self.point[1]
            return (dx * dx + dy * dy) ** 0.5 <= self.tolerance_px
        raise ValueError("ClickTarget needs a bbox or a point")


def click_accuracy(
    predictions: Sequence[Action | None],
    targets: Sequence[ClickTarget],
    codec: ActionCodec | None = None,
) -> dict:
    """Fraction of predicted clicks landing on the intended element.

    Also reports the miss distance distribution, which separates "the model is
    looking at the wrong widget" from "the grid resolution is too coarse" --
    two problems with completely different fixes.
    """
    codec = codec or ActionCodec()
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must be the same length")

    hits = 0
    distances: list[float] = []
    non_pointer = 0
    missing = 0

    for prediction, target in zip(predictions, targets):
        if prediction is None:
            missing += 1
            continue
        if not prediction.type.is_absolute_pointer:
            # Predicting a key press where a click was needed is a miss, and a
            # different kind of miss than a bad coordinate.
            non_pointer += 1
            continue
        x, y = codec.to_pixels(prediction, target.screen_w, target.screen_h)
        if target.contains(x, y):
            hits += 1
        else:
            distances.append(_distance_to_target(x, y, target))

    n = len(predictions)
    quant_x, quant_y = codec.quantization_error_px(
        targets[0].screen_w, targets[0].screen_h
    ) if targets else (0.0, 0.0)
    return {
        "n": n,
        "accuracy": round(hits / n, 4) if n else None,
        "hits": hits,
        "wrong_action_type": non_pointer,
        "no_prediction": missing,
        "median_miss_px": round(_median(distances), 1) if distances else None,
        "p90_miss_px": round(_percentile(sorted(distances), 0.9), 1) if distances else None,
        # If median_miss_px is close to this, the grid resolution is the
        # bottleneck and raising offset_grid will help more than more training.
        "quantization_error_px": round(max(quant_x, quant_y), 1),
    }


def _distance_to_target(x: float, y: float, target: ClickTarget) -> float:
    if target.bbox is not None:
        left, top, right, bottom = target.bbox
        dx = max(left - x, 0, x - (right - 1))
        dy = max(top - y, 0, y - (bottom - 1))
        return (dx * dx + dy * dy) ** 0.5
    px, py = target.point  # type: ignore[misc]
    return ((x - px) ** 2 + (y - py) ** 2) ** 0.5


def field_fill_exactness(
    predictions: Sequence[Action | None],
    expected_values: Sequence[str],
    form_data: Sequence[dict],
) -> dict:
    """Exact-match rate on copied form values.

    Should be 1.0.  ``routing_errors`` is the interesting column: it counts
    cases where the model generated text instead of selecting the supplied
    field, which is the failure the field-fill design exists to prevent.
    """
    exact = routing_errors = wrong_field = missing = 0

    for prediction, expected, data in zip(predictions, expected_values, form_data):
        if prediction is None or prediction.type is not ActionType.TYPE:
            missing += 1
            continue
        if prediction.is_free_compose:
            routing_errors += 1
            # Free-composing the right string still counts as exact -- it
            # produced the correct value -- but the routing error is recorded
            # because it means the guarantee is not being relied upon.
            if prediction.text == expected:
                exact += 1
            continue
        try:
            value = prediction.resolve_text(data)
        except Exception:
            wrong_field += 1
            continue
        if value == expected:
            exact += 1
        else:
            wrong_field += 1

    n = len(predictions)
    return {
        "n": n,
        "exactness": round(exact / n, 4) if n else None,
        "routing_errors": routing_errors,
        "wrong_field": wrong_field,
        "not_a_type_action": missing,
    }


def action_type_accuracy(
    predictions: Sequence[Action | None], targets: Sequence[Action]
) -> dict:
    """Confusion over action types, ignoring arguments."""
    correct = 0
    confusion: Counter = Counter()
    for prediction, target in zip(predictions, targets):
        predicted = prediction.type.value if prediction else "<none>"
        if prediction is not None and prediction.type is target.type:
            correct += 1
        else:
            confusion[f"{target.type.value}->{predicted}"] += 1
    n = len(predictions)
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else None,
        "top_confusions": dict(confusion.most_common(10)),
    }


def escalation_metrics(
    escalated: Sequence[bool], should_escalate: Sequence[bool]
) -> dict:
    """Precision and recall of the ESCALATE decision.

    Plan section 8 calls escalation calibration a genuine ML problem rather
    than a hyperparameter, and this is why it needs its own eval slice: the two
    error directions have opposite costs and a single accuracy number hides
    the trade entirely.
    """
    tp = sum(1 for e, s in zip(escalated, should_escalate) if e and s)
    fp = sum(1 for e, s in zip(escalated, should_escalate) if e and not s)
    fn = sum(1 for e, s in zip(escalated, should_escalate) if not e and s)
    tn = sum(1 for e, s in zip(escalated, should_escalate) if not e and not s)

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )
    return {
        "n": len(escalated),
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "true_positive": tp, "false_positive": fp,
        "false_negative": fn, "true_negative": tn,
        # Silent failures: the policy pressed on when it should have handed
        # back. These are the expensive ones.
        "silent_failures": fn,
        # Handing back when it could have finished: annoying, not dangerous.
        "spurious_escalations": fp,
    }


def task_success_rate(statuses: Sequence[str], verified: Sequence[bool] | None = None) -> dict:
    """End-to-end success over a held-out task set.

    A run only counts as a success if the verifier agreed; a policy that emits
    DONE early would otherwise score perfectly.
    """
    counts = Counter(statuses)
    n = len(statuses)
    if verified is None:
        successes = counts.get("done", 0)
    else:
        successes = sum(1 for s, v in zip(statuses, verified) if s == "done" and v)
    return {
        "n": n,
        "success_rate": round(successes / n, 4) if n else None,
        "by_status": dict(counts),
        "escalation_rate": round(counts.get("escalated", 0) / n, 4) if n else None,
        "timeout_rate": round(counts.get("timeout", 0) / n, 4) if n else None,
    }


@dataclass
class MetricAccumulator:
    """Running totals during a training loop."""

    loss: float = 0.0
    token_accuracy: float = 0.0
    steps: int = 0
    by_kind: dict = field(default_factory=dict)

    def update(self, loss: float, token_accuracy: float, kind: str | None = None) -> None:
        self.loss += loss
        self.token_accuracy += token_accuracy
        self.steps += 1
        if kind:
            entry = self.by_kind.setdefault(kind, {"loss": 0.0, "n": 0})
            entry["loss"] += loss
            entry["n"] += 1

    def summary(self) -> dict:
        if not self.steps:
            return {}
        out = {
            "loss": round(self.loss / self.steps, 4),
            "token_accuracy": round(self.token_accuracy / self.steps, 4),
            "steps": self.steps,
        }
        for kind, entry in self.by_kind.items():
            out[f"loss/{kind}"] = round(entry["loss"] / max(entry["n"], 1), 4)
        return out

    def reset(self) -> None:
        self.loss = self.token_accuracy = 0.0
        self.steps = 0
        self.by_kind = {}


def _median(values: Sequence[float]) -> float:
    return _percentile(sorted(values), 0.5)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(q * (len(sorted_values) - 1) + 0.5), len(sorted_values) - 1)
    return sorted_values[index]
