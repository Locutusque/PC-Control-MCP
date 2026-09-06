"""Trajectory encoding and dataset construction (plan sections 4.2-4.3)."""

from __future__ import annotations

from gui_agent.actions import ActionType
from gui_agent.capture.schema import EventType, InputEvent, SegmentMeta
from gui_agent.data.encode import EncodeConfig, TrajectoryEncoder
from gui_agent.data.finetune_dataset import (
    balance_examples,
    dataset_stats,
    split_examples,
)
from gui_agent.data.hindsight_relabel import (
    RelabelResult,
    evaluate_against_human_labels,
    sample_frame_indices,
)
from gui_agent.data.schema import ExampleKind, TrainingExample, load_examples, save_examples


class TestTrajectoryEncoding:
    def test_encodes_a_realistic_trajectory(self, codec, click_trajectory):
        actions = TrajectoryEncoder(codec).encode_segment(click_trajectory)
        types = [a.action.type for a in actions]
        assert ActionType.MOVE_CLICK in types
        assert ActionType.SCROLL in types
        assert ActionType.KEY in types
        assert types[-1] is ActionType.DONE

    def test_a_short_press_is_a_click_not_a_drag(self, codec, make_record):
        records = [
            make_record(0, 0.0, InputEvent(EventType.MOUSE_DOWN, x=100, y=100, button="left")),
            make_record(1, 0.05, InputEvent(EventType.MOUSE_UP, x=102, y=101, button="left")),
            make_record(2, 0.06, InputEvent(EventType.CLICK, x=102, y=101, button="left")),
        ]
        actions = TrajectoryEncoder(codec).encode_segment(records)
        types = [a.action.type for a in actions if a.action.type is not ActionType.DONE]
        assert types == [ActionType.MOVE_CLICK]

    def test_a_long_press_is_a_drag_and_suppresses_the_click(self, codec, make_record):
        records = [
            make_record(0, 0.0, InputEvent(EventType.MOUSE_DOWN, x=100, y=100, button="left")),
            make_record(1, 0.3, InputEvent(EventType.MOUSE_UP, x=500, y=100, button="left")),
            make_record(2, 0.31, InputEvent(EventType.CLICK, x=500, y=100, button="left")),
        ]
        actions = TrajectoryEncoder(codec).encode_segment(records)
        types = [a.action.type for a in actions if a.action.type is not ActionType.DONE]
        assert types == [ActionType.DRAG_START, ActionType.DRAG_END]

    def test_typed_text_matching_a_visible_value_is_field_fill(self, codec, make_record):
        records = [
            make_record(0, 0.0, InputEvent(EventType.TEXT, text="j.doe@example.com"),
                        form_data_visible={"email": "j.doe@example.com"})
        ]
        actions = TrajectoryEncoder(codec).encode_segment(records)
        assert actions[0].kind is ExampleKind.FIELD_FILL
        assert actions[0].action.field == "email"

    def test_novel_text_is_free_compose(self, codec, make_record):
        records = [
            make_record(0, 0.0, InputEvent(EventType.TEXT, text="where is my order"),
                        form_data_visible={"email": "j.doe@example.com"})
        ]
        actions = TrajectoryEncoder(codec).encode_segment(records)
        assert actions[0].kind is ExampleKind.FREE_COMPOSE
        assert actions[0].action.text == "where is my order"

    def test_field_matching_is_exact_not_fuzzy(self, codec, make_record):
        # A fuzzy match would label free-compose as field-fill and teach the
        # model to copy where it should generate.
        records = [
            make_record(0, 0.0, InputEvent(EventType.TEXT, text="j.doe@example.com.uk"),
                        form_data_visible={"email": "j.doe@example.com"})
        ]
        actions = TrajectoryEncoder(codec).encode_segment(records)
        assert actions[0].kind is ExampleKind.FREE_COMPOSE

    def test_redacted_text_produces_no_action(self, codec, make_record):
        # We know a password was typed and refuse to reconstruct it: the
        # trajectory keeps a hole rather than a guess.
        records = [make_record(0, 0.0, InputEvent(EventType.REDACTED_TEXT, n_chars=8))]
        actions = TrajectoryEncoder(codec).encode_segment(records)
        assert actions == []

    def test_gaps_become_explicit_waits(self, codec, make_record):
        records = [
            make_record(0, 0.0, InputEvent(EventType.CLICK, x=10, y=10, button="left")),
            make_record(1, 1.0, InputEvent(EventType.CLICK, x=20, y=20, button="left")),
        ]
        actions = TrajectoryEncoder(codec).encode_segment(records)
        assert any(a.action.type is ActionType.WAIT for a in actions)

    def test_very_long_gaps_are_not_waits(self, codec, make_record):
        # The user went for coffee; they did not decide to wait.
        records = [
            make_record(0, 0.0, InputEvent(EventType.CLICK, x=10, y=10, button="left")),
            make_record(1, 600.0, InputEvent(EventType.CLICK, x=20, y=20, button="left")),
        ]
        actions = TrajectoryEncoder(codec).encode_segment(records)
        assert not any(a.action.type is ActionType.WAIT for a in actions)

    def test_small_cursor_movement_is_ignored(self, codec, make_record):
        records = [
            make_record(0, 0.0, InputEvent(EventType.MOUSE_MOVE, x=100, y=100)),
            make_record(1, 0.1, InputEvent(EventType.MOUSE_MOVE, x=104, y=102)),
        ]
        config = EncodeConfig(min_move_px=48, append_done=False)
        assert TrajectoryEncoder(codec, config).encode_segment(records) == []

    def test_large_cursor_movement_becomes_move_rel(self, codec, make_record):
        records = [
            make_record(0, 0.0, InputEvent(EventType.MOUSE_MOVE, x=100, y=100)),
            make_record(1, 0.1, InputEvent(EventType.MOUSE_MOVE, x=400, y=100)),
        ]
        config = EncodeConfig(min_move_px=48, append_done=False)
        actions = TrajectoryEncoder(codec, config).encode_segment(records)
        assert [a.action.type for a in actions] == [ActionType.MOVE_REL]

    def test_unknown_keys_are_dropped_rather_than_guessed(self, codec, make_record):
        records = [make_record(0, 0.0, InputEvent(EventType.KEY_DOWN, key="NOT_A_REAL_KEY"))]
        config = EncodeConfig(append_done=False)
        assert TrajectoryEncoder(codec, config).encode_segment(records) == []

    def test_history_windows_are_rolling(self, codec, click_trajectory):
        encoder = TrajectoryEncoder(codec)
        actions = encoder.encode_segment(click_trajectory)
        histories = encoder.with_history(actions, max_history=2)
        assert len(histories) == len(actions)
        assert histories[0] == []
        assert all(len(h) <= 2 for h in histories)


class TestDatasetTools:
    def _examples(self):
        return (
            [TrainingExample("s1", i, ["<DONE>"], app_context="chrome") for i in range(40)]
            + [
                TrainingExample(
                    "s2", i, ["<TYPE_START>", "<FIELD_0>", "<TYPE_END>"],
                    kind=ExampleKind.FIELD_FILL, form_data={"email": "a@b.c"},
                    app_context="slack",
                )
                for i in range(4)
            ]
        )

    def test_stats_report_coverage(self):
        stats = dataset_stats(self._examples())
        assert stats["n_examples"] == 44
        assert stats["n_segments"] == 2
        assert stats["field_fill_keys"] == {"email": 4}
        assert stats["n_distinct_apps"] == 2

    def test_balancing_caps_the_dominant_kind(self):
        balanced = balance_examples(self._examples(), max_ratio=2.0, seed=0)
        kinds = {}
        for example in balanced:
            kinds[example.kind] = kinds.get(example.kind, 0) + 1
        assert kinds[ExampleKind.CONTROL] <= 2 * kinds[ExampleKind.FIELD_FILL]

    def test_split_is_by_segment_not_by_example(self):
        # Splitting by example puts consecutive frames of one trajectory on
        # both sides and reports training accuracy as validation.
        train, val = split_examples(self._examples(), val_fraction=0.5, seed=0)
        assert {e.segment_id for e in train}.isdisjoint({e.segment_id for e in val})

    def test_examples_roundtrip_through_jsonl(self, tmp_path):
        path = tmp_path / "examples.jsonl"
        examples = self._examples()
        save_examples(path, examples)
        loaded = list(load_examples(path))
        assert len(loaded) == len(examples)
        assert loaded[-1].kind is ExampleKind.FIELD_FILL
        assert loaded[-1].form_data == {"email": "a@b.c"}


class TestHindsightCalibration:
    def test_agreement_is_measured_against_human_labels(self):
        results = [
            RelabelResult("s1", "Search the docs for the refund policy", 0.9),
            RelabelResult("s2", "Do something in a browser", 0.2),
        ]
        segments = [
            SegmentMeta("s1", "x", 0.0, user_label="find the refund policy in the docs"),
            SegmentMeta("s2", "x", 0.0, user_label="book a flight to Tokyo"),
        ]
        report = evaluate_against_human_labels(results, segments)
        assert report["n_compared"] == 2
        assert report["agreement"] == 0.5
        assert report["confident_agreement"] == 1.0
        assert report["confidence_is_informative"] is True

    def test_unlabelled_segments_are_not_compared(self):
        results = [RelabelResult("s1", "anything", 0.9)]
        report = evaluate_against_human_labels(results, [SegmentMeta("s1", "x", 0.0)])
        assert report["n_compared"] == 0 and report["agreement"] is None

    def test_frame_sampling_spans_the_trajectory(self):
        from gui_agent.actions import Action
        from gui_agent.data.encode import TimedAction

        actions = [
            TimedAction(i, float(i), Action(ActionType.DONE)) for i in range(20)
        ]
        sampled = sample_frame_indices(actions, 4)
        assert sampled[0] == 0 and sampled[-1] == 19 and len(sampled) == 4
