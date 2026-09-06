"""Metrics, escalation labelling and DAgger example construction (plan 5 / 4.4)."""

from __future__ import annotations

from gui_agent.actions import Action, ActionType
from gui_agent.data.schema import ExampleKind
from gui_agent.harness.loop import RolloutTrace, SubtaskResult, SubtaskStatus, TickRecord
from gui_agent.train.metrics import (
    ClickTarget,
    MetricAccumulator,
    click_accuracy,
    escalation_metrics,
    field_fill_exactness,
    task_success_rate,
)
from gui_agent.train.stage3_dagger import (
    Correction,
    RolloutRecord,
    Task,
    build_dagger_examples,
    escalation_labels,
)


class TestClickAccuracy:
    def test_hits_and_misses(self, codec):
        targets = [
            ClickTarget(1920, 1080, bbox=(800, 600, 900, 650)),
            ClickTarget(1920, 1080, bbox=(100, 100, 140, 130)),
        ]
        predictions = [codec.click(850, 620, 1920, 1080), codec.click(1500, 900, 1920, 1080)]
        report = click_accuracy(predictions, targets, codec)
        assert report["accuracy"] == 0.5
        assert report["median_miss_px"] > 0

    def test_wrong_action_type_is_counted_separately(self, codec):
        # Predicting a key press where a click was needed is a different
        # failure from a bad coordinate.
        report = click_accuracy(
            [Action(ActionType.KEY, key="ENTER")],
            [ClickTarget(1920, 1080, bbox=(0, 0, 10, 10))],
            codec,
        )
        assert report["wrong_action_type"] == 1 and report["accuracy"] == 0.0

    def test_no_prediction_is_counted_separately(self, codec):
        report = click_accuracy([None], [ClickTarget(1920, 1080, bbox=(0, 0, 10, 10))], codec)
        assert report["no_prediction"] == 1

    def test_quantization_error_is_reported_alongside_the_miss(self, codec):
        # If the median miss is near the quantization error, more training will
        # not help and a finer offset grid will.
        report = click_accuracy(
            [codec.click(0, 0, 1920, 1080)], [ClickTarget(1920, 1080, bbox=(0, 0, 4, 4))], codec
        )
        assert report["quantization_error_px"] > 0

    def test_point_targets_use_a_tolerance(self, codec):
        target = ClickTarget(1920, 1080, point=(960, 540), tolerance_px=60)
        assert click_accuracy([codec.click(960, 540, 1920, 1080)], [target], codec)["accuracy"] == 1.0


class TestFieldFillExactness:
    def test_field_selection_is_exact(self):
        report = field_fill_exactness(
            [Action(ActionType.TYPE, field="email")],
            ["j@x.com"], [{"email": "j@x.com"}],
        )
        assert report["exactness"] == 1.0 and report["routing_errors"] == 0

    def test_free_composing_the_right_value_still_records_a_routing_error(self):
        # It produced the right string, but the structural guarantee was not
        # used - which is the failure the field-fill design exists to prevent.
        report = field_fill_exactness(
            [Action(ActionType.TYPE, text="j@x.com")], ["j@x.com"], [{"email": "j@x.com"}]
        )
        assert report["exactness"] == 1.0 and report["routing_errors"] == 1

    def test_free_composing_a_wrong_value_is_not_exact(self):
        report = field_fill_exactness(
            [Action(ActionType.TYPE, text="typo@x.com")], ["j@x.com"], [{"email": "j@x.com"}]
        )
        assert report["exactness"] == 0.0 and report["routing_errors"] == 1

    def test_a_non_type_action_is_counted(self):
        report = field_fill_exactness(
            [Action(ActionType.KEY, key="TAB")], ["j@x.com"], [{"email": "j@x.com"}]
        )
        assert report["not_a_type_action"] == 1


class TestEscalationMetrics:
    def test_both_error_directions_are_reported(self):
        report = escalation_metrics([True, False, True, False], [True, True, False, False])
        assert report["silent_failures"] == 1       # should have escalated, did not
        assert report["spurious_escalations"] == 1  # escalated when it need not have
        assert report["precision"] == 0.5 and report["recall"] == 0.5

    def test_never_escalating_has_no_precision_and_zero_recall(self):
        # The silent-failure regime: an accuracy number alone would look fine.
        report = escalation_metrics([False] * 4, [True, True, False, False])
        assert report["precision"] is None and report["recall"] == 0.0
        assert report["silent_failures"] == 2

    def test_always_escalating_has_perfect_recall_and_poor_precision(self):
        report = escalation_metrics([True] * 4, [True, True, False, False])
        assert report["recall"] == 1.0 and report["precision"] == 0.5


class TestTaskSuccess:
    def test_unverified_done_does_not_count_as_success(self):
        report = task_success_rate(["done", "done"], [True, False])
        assert report["success_rate"] == 0.5

    def test_status_breakdown(self):
        report = task_success_rate(["done", "escalated", "timeout", "done"])
        assert report["escalation_rate"] == 0.25 and report["timeout_rate"] == 0.25


class TestAccumulator:
    def test_averages_over_steps(self):
        accumulator = MetricAccumulator()
        accumulator.update(2.0, 0.5)
        accumulator.update(4.0, 0.7)
        summary = accumulator.summary()
        assert summary["loss"] == 3.0 and summary["steps"] == 2
        accumulator.reset()
        assert accumulator.summary() == {}


def _rollout(status: SubtaskStatus, n: int, reason: str, rollout_id="roll_1", blocked=None):
    trace = RolloutTrace(rollout_id, "click submit", {})
    for i in range(n):
        trace.ticks.append(
            TickRecord(
                i, 0.0, ["<MOVE_CLICK>", f"<ROW_{i % 8}>", "<COL_1>", "<OFFSET_0>"],
                f"MOVE_CLICK(r{i % 8})", True, blocked_reason=blocked,
            )
        )
    trace.status, trace.stop_reason = status, reason
    return RolloutRecord(
        Task("t1", "click submit"),
        SubtaskResult(status, "", None, reason, n, 1.0, trace=trace),
    )


class TestEscalationLabels:
    def test_a_successful_rollout_labels_every_tick_negative(self):
        record = _rollout(SubtaskStatus.DONE, 4, "policy reported done")
        assert all(not should for _, should in escalation_labels(record))

    def test_a_timeout_labels_the_final_ticks_positive(self):
        record = _rollout(SubtaskStatus.TIMEOUT, 8, "timed out")
        labels = dict(escalation_labels(record))
        assert labels[0] is False and labels[7] is True

    def test_a_self_escalation_labels_only_the_last_tick_positive(self):
        record = _rollout(SubtaskStatus.ESCALATED, 5, "policy escalated")
        labels = dict(escalation_labels(record))
        assert labels[4] is True and sum(labels.values()) == 1


class TestDaggerExamples:
    def test_uncorrected_ticks_of_a_failure_are_not_cloned(self):
        # Training on them teaches the behaviour that just failed.
        record = _rollout(SubtaskStatus.TIMEOUT, 8, "timed out")
        examples = build_dagger_examples([record], [])
        assert all(e.kind is ExampleKind.ESCALATE for e in examples)
        assert {e.frame_index for e in examples} == {3, 4, 5, 6, 7}

    def test_successful_rollouts_supply_the_negative_signal(self):
        record = _rollout(SubtaskStatus.DONE, 4, "policy reported done")
        examples = build_dagger_examples([record], [])
        assert len(examples) == 4
        assert all(e.kind is ExampleKind.CONTROL for e in examples)
        assert not any(e.target == ["<ESCALATE>"] for e in examples)

    def test_corrections_replace_the_action_and_record_what_was_wrong(self):
        record = _rollout(SubtaskStatus.TIMEOUT, 8, "timed out")
        correction = Correction("roll_1", 2, ["<DONE>"], "already complete")
        example = next(
            e for e in build_dagger_examples([record], [correction]) if e.frame_index == 2
        )
        assert example.target == ["<DONE>"]
        assert example.corrected_from == record.result.trace.ticks[2].atoms

    def test_history_is_the_actions_actually_taken(self):
        # The policy has to learn from the states it really visited, including
        # the ones it reached by getting things wrong.
        record = _rollout(SubtaskStatus.DONE, 4, "policy reported done")
        examples = sorted(build_dagger_examples([record], []), key=lambda e: e.frame_index)
        assert examples[0].history == []
        assert examples[3].history == [t.atoms for t in record.result.trace.ticks[:3]]

    def test_near_misses_are_detected(self):
        record = _rollout(SubtaskStatus.DONE, 4, "done", blocked="blocked_key")
        assert record.is_near_miss

    def test_corrections_are_scoped_to_their_own_rollout(self):
        a = _rollout(SubtaskStatus.DONE, 3, "done", rollout_id="roll_a")
        b = _rollout(SubtaskStatus.DONE, 3, "done", rollout_id="roll_b")
        examples = build_dagger_examples([a, b], [Correction("roll_a", 1, ["<DONE>"], "x")])
        corrected = [e for e in examples if e.target == ["<DONE>"]]
        assert len(corrected) == 1 and corrected[0].rollout_id == "roll_a"
