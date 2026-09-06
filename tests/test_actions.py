"""The action vocabulary, grammar and codec (plan section 2)."""

from __future__ import annotations

import pytest

from gui_agent.actions import (
    Action,
    ActionParseError,
    ActionType,
    ActionVocab,
    DecodeState,
    TokenClass,
)
from gui_agent.config import ActionSpaceConfig


class TestVocabulary:
    def test_tokens_are_unique_and_ordered_deterministically(self, action_space):
        a, b = ActionVocab(action_space), ActionVocab(action_space)
        assert a.tokens == b.tokens
        assert len(set(a.tokens)) == len(a.tokens)

    def test_signature_changes_with_the_configuration(self, action_space):
        other = ActionSpaceConfig(**{**action_space.to_dict(), "offset_grid": 8})
        assert ActionVocab(action_space).signature() != ActionVocab(other).signature()

    def test_every_action_type_has_a_token(self, action_space):
        vocab = ActionVocab(action_space)
        for action_type in ActionType:
            token = vocab.action_token(action_type)
            assert token in vocab
            assert vocab.action_type_of(token) is action_type

    def test_class_membership_partitions_the_vocabulary(self, action_space):
        vocab = ActionVocab(action_space)
        covered = {t for cls in TokenClass for t in vocab.of_class(cls)}
        assert covered == set(vocab.tokens)


class TestGeometry:
    def test_pixel_roundtrip_is_stable(self, codec):
        for row in range(0, 8, 3):
            for col in range(0, 8, 3):
                for offset in (0, 5, 15):
                    x, y = codec.grid_to_pixels(row, col, offset, 1920, 1080)
                    assert codec.pixels_to_grid(x, y, 1920, 1080) == (row, col, offset)

    def test_clicks_stay_on_screen_at_the_far_corner(self, codec):
        cfg = codec.config
        x, y = codec.grid_to_pixels(
            cfg.grid_rows - 1, cfg.grid_cols - 1, cfg.offset_grid**2 - 1, 1920, 1080
        )
        assert 0 <= x < 1920 and 0 <= y < 1080

    def test_pixels_outside_the_screen_are_clamped(self, codec):
        assert codec.pixels_to_grid(-50, -50, 1920, 1080) == (0, 0, 0)
        row, col, _ = codec.pixels_to_grid(9999, 9999, 1920, 1080)
        assert (row, col) == (codec.config.grid_rows - 1, codec.config.grid_cols - 1)

    def test_quantization_error_matches_the_grid(self, codec):
        ex, _ = codec.quantization_error_px(1920, 1080)
        assert ex == pytest.approx(1920 / (codec.config.grid_cols * codec.config.offset_grid) / 2)

    def test_out_of_range_coordinates_are_rejected(self, codec):
        with pytest.raises(ActionParseError):
            codec.grid_to_pixels(99, 0, 0, 1920, 1080)


class TestCodec:
    @pytest.mark.parametrize(
        "action",
        [
            Action(ActionType.MOVE_CLICK, row=3, col=4, offset=5),
            Action(ActionType.DOUBLE_CLICK, row=0, col=0, offset=0),
            Action(ActionType.RIGHT_CLICK, row=7, col=7, offset=15),
            Action(ActionType.DRAG_START, row=1, col=2, offset=3),
            Action(ActionType.DRAG_END, row=4, col=5, offset=6),
            Action(ActionType.MOVE_REL, dx=16, dy=-32),
            Action(ActionType.SCROLL, scroll_dx=0, scroll_dy=-3),
            Action(ActionType.KEY, key="ENTER"),
            Action(ActionType.KEY, key="CTRL_C"),
            Action(ActionType.WAIT, wait_ms=250),
            Action(ActionType.WAIT, wait_ms=2000),
            Action(ActionType.DONE),
            Action(ActionType.ESCALATE),
        ],
    )
    def test_roundtrip(self, codec, action):
        assert codec.decode(codec.encode(action)) == action

    def test_field_fill_roundtrip(self, codec):
        action = Action(ActionType.TYPE, field="email")
        fields = ["name", "email"]
        atoms = codec.encode(action, fields)
        assert atoms == ["<TYPE_START>", "<FIELD_1>", "<TYPE_END>"]
        assert codec.decode(atoms, fields) == action

    def test_free_compose_roundtrip(self, codec):
        action = Action(ActionType.TYPE, text="hello world")
        assert codec.decode(codec.encode(action)) == action

    def test_field_fill_needs_the_field_to_exist(self, codec):
        with pytest.raises(ActionParseError):
            codec.encode(Action(ActionType.TYPE, field="ghost"), ["email"])

    def test_field_index_beyond_max_form_fields_is_rejected(self, codec):
        fields = [f"f{i}" for i in range(codec.config.max_form_fields + 1)]
        with pytest.raises(ActionParseError):
            codec.encode(Action(ActionType.TYPE, field=fields[-1]), fields)

    def test_decode_stream_parses_consecutive_actions(self, codec):
        actions = [
            Action(ActionType.MOVE_CLICK, row=1, col=1, offset=1),
            Action(ActionType.KEY, key="TAB"),
            Action(ActionType.DONE),
        ]
        atoms = [a for action in actions for a in codec.encode(action)]
        assert list(codec.decode_stream(atoms)) == actions

    def test_trailing_partial_action_is_an_error(self, codec):
        with pytest.raises(ActionParseError, match="incomplete"):
            list(codec.decode_stream(["<MOVE_CLICK>", "<ROW_1>"]))

    def test_click_helper_grounds_pixels(self, codec):
        action = codec.click(960, 540, 1920, 1080)
        assert action.type is ActionType.MOVE_CLICK
        x, y = codec.to_pixels(action, 1920, 1080)
        assert abs(x - 960) < 40 and abs(y - 540) < 40


class TestBucketing:
    def test_delta_snaps_to_the_nearest_bucket(self, codec):
        assert codec.bucket_delta(0) == 0
        assert codec.bucket_delta(17) == 16
        assert codec.bucket_delta(-100) == -128
        assert codec.bucket_delta(10_000) == max(codec.config.delta_buckets)

    def test_scroll_clamps_to_range(self, codec):
        limit = codec.config.scroll_range
        assert codec.bucket_scroll(99) == limit
        assert codec.bucket_scroll(-99) == -limit

    def test_wait_snaps_to_the_nearest_supported_duration(self, codec):
        assert codec.bucket_wait(120) == 100
        assert codec.bucket_wait(1_000_000) == max(codec.config.wait_ms)


class TestGrammar:
    def test_a_control_action_completes_within_the_budget(self, codec):
        grammar = codec.grammar
        for action in (
            Action(ActionType.MOVE_CLICK, row=1, col=1, offset=1),
            Action(ActionType.MOVE_REL, dx=8, dy=8),
            Action(ActionType.KEY, key="ENTER"),
            Action(ActionType.DONE),
        ):
            atoms = codec.encode(action)
            assert len(atoms) <= grammar.max_control_atoms()

    def test_illegal_continuations_are_rejected(self, codec):
        grammar = codec.grammar
        state, chain = grammar.step(grammar.start(), "<MOVE_CLICK>", [])
        assert state is DecodeState.AWAIT_ROW
        with pytest.raises(ActionParseError):
            grammar.step(state, "<COL_1>", chain)

    def test_type_body_permits_field_or_text(self, codec):
        grammar = codec.grammar
        state, chain = grammar.step(grammar.start(), "<TYPE_START>", [])
        assert grammar.allowed_classes(state) == frozenset({TokenClass.FIELD, TokenClass.TEXT})

    def test_free_text_can_only_end_with_type_end(self, codec):
        grammar = codec.grammar
        state, chain = grammar.step(grammar.start(), "<TYPE_START>", [])
        state, chain = grammar.step(state, "some text", chain)
        assert grammar.allowed_classes(state) == frozenset({TokenClass.TEXT, TokenClass.TYPE_END})
        state, chain = grammar.step(state, "<TYPE_END>", chain)
        assert grammar.is_complete(state)

    def test_allowed_tokens_are_non_empty_for_every_reachable_state(self, codec):
        grammar = codec.grammar
        for state in DecodeState:
            if state is DecodeState.COMPLETE:
                continue
            assert grammar.allowed_classes(state)


class TestActionValidation:
    def test_missing_arguments_are_rejected(self):
        with pytest.raises(ActionParseError):
            Action(ActionType.MOVE_CLICK, row=1)
        with pytest.raises(ActionParseError):
            Action(ActionType.KEY)

    def test_type_needs_exactly_one_of_field_or_text(self):
        with pytest.raises(ActionParseError):
            Action(ActionType.TYPE)
        with pytest.raises(ActionParseError):
            Action(ActionType.TYPE, field="a", text="b")

    def test_field_fill_resolves_verbatim(self):
        action = Action(ActionType.TYPE, field="email")
        assert action.resolve_text({"email": "j@x.com"}) == "j@x.com"

    def test_field_fill_without_the_value_raises(self):
        with pytest.raises(ActionParseError):
            Action(ActionType.TYPE, field="email").resolve_text({})

    def test_terminal_actions_are_flagged(self):
        assert Action(ActionType.DONE).type.is_terminal
        assert Action(ActionType.ESCALATE).type.is_terminal
        assert not Action(ActionType.KEY, key="TAB").type.is_terminal
