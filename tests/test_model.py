"""Tokenizer extension, constrained decoding and the policy (plan section 3).

These run against a tiny real policy rather than mocks, so they exercise the
actual shapes, cache handling and grammar masking.
"""

from __future__ import annotations

import pytest
import torch

from gui_agent.actions import Action, ActionType, DecodeState, TokenClass
from gui_agent.config import ProjectorConfig, VisionConfig
from gui_agent.model.decoding import ConstrainedActionDecoder
from gui_agent.model.policy import IGNORE_INDEX, GuiPolicy, PolicyBatch, PolicySession
from gui_agent.model.projector import build_projector
from gui_agent.model.tokenizer_ext import ActionTokenizer, TokenizerMismatch
from gui_agent.model.vit import ViT, preprocess_screenshot


class TestVit:
    def test_output_is_a_patch_grid_not_a_pooled_vector(self):
        # <ROW_r> <COL_c> has to address a real patch embedding.
        config = VisionConfig(image_size=128, patch_size=32, dim=96, depth=2, heads=3)
        vit = ViT(config)
        out = vit(torch.randn(2, 3, 128, 128))
        assert out.shape == (2, config.num_patches, config.dim)
        assert config.num_patches == 16

    def test_default_size_is_within_the_budget(self):
        # Plan section 3.5 budgets 50-90M for the vision encoder.
        assert 50e6 <= ViT().num_parameters() <= 90e6

    def test_preprocessing_squashes_rather_than_letterboxes(self):
        import numpy as np

        # Letterboxing would leave dead rows in the coordinate vocabulary.
        image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        pixels = preprocess_screenshot(image, VisionConfig(image_size=128, patch_size=32))
        assert pixels.shape == (1, 3, 128, 128)

    def test_bgra_frames_are_accepted(self):
        import numpy as np

        image = np.random.randint(0, 255, (100, 200, 4), dtype=np.uint8)
        assert preprocess_screenshot(image, VisionConfig(image_size=64, patch_size=32)).shape[1] == 3

    def test_position_embeddings_interpolate_to_a_new_resolution(self):
        vit = ViT(VisionConfig(image_size=128, patch_size=32, dim=96, depth=2, heads=3))
        assert vit(torch.randn(1, 3, 256, 256)).shape == (1, 64, 96)


class TestProjector:
    def test_mlp_preserves_the_token_count(self):
        projector = build_projector(96, 64, ProjectorConfig(kind="mlp", hidden_dim=128))
        assert projector(torch.randn(2, 16, 96)).shape == (2, 16, 64)

    def test_perceiver_compresses_to_a_fixed_number_of_latents(self):
        config = ProjectorConfig(kind="perceiver", hidden_dim=128, num_latents=8, num_heads=4)
        projector = build_projector(96, 64, config)
        assert projector(torch.randn(2, 16, 96)).shape == (2, 8, 64)


class TestActionTokenizer:
    def test_action_ids_form_one_contiguous_block(self, action_tokenizer):
        # Constrained decoding masks by slice and the embedding parameter group
        # is a contiguous range; both break silently otherwise.
        span = action_tokenizer.action_id_end - action_tokenizer.action_id_start
        assert span == action_tokenizer.n_action_tokens

    def test_class_spans_do_not_overlap(self, action_tokenizer):
        spans = [
            action_tokenizer.class_span(cls)
            for cls in TokenClass
            if cls is not TokenClass.TEXT
        ]
        flat = sorted(spans)
        for (_, end), (start, _) in zip(flat, flat[1:], strict=False):
            assert end <= start

    def test_atoms_roundtrip(self, action_tokenizer):
        atoms = ["<MOVE_CLICK>", "<ROW_3>", "<COL_2>", "<OFFSET_5>"]
        assert action_tokenizer.decode_atoms(action_tokenizer.encode_atoms(atoms)) == atoms

    def test_out_of_range_coordinate_raises_rather_than_becoming_text(self, action_tokenizer):
        # Silently tokenizing this as prose would produce a garbage training
        # target with nothing to show anything had gone wrong.
        with pytest.raises(KeyError, match="action space"):
            action_tokenizer.encode_atoms(["<MOVE_CLICK>", "<ROW_999>"])

    def test_free_text_roundtrips_through_the_base_vocabulary(self, action_tokenizer):
        atoms = ["<TYPE_START>", "hello world", "<TYPE_END>"]
        decoded = action_tokenizer.decode_atoms(action_tokenizer.encode_atoms(atoms))
        assert decoded[0] == "<TYPE_START>" and decoded[-1] == "<TYPE_END>"
        assert "hello" in decoded[1]

    def test_signature_mismatch_is_loud(self, action_tokenizer):
        # A reordered vocabulary remaps the action embeddings and produces
        # confident wrong clicks with no error, so it must never pass quietly.
        with pytest.raises(TokenizerMismatch):
            action_tokenizer.check_signature("999:deadbeefdeadbeef@0")

    def test_matching_signature_passes(self, action_tokenizer):
        action_tokenizer.check_signature(action_tokenizer.signature())

    def test_reloading_a_saved_tokenizer_is_idempotent(self, base_tokenizer, policy_config):
        # A tokenizer saved from a trained policy already carries the whole
        # block; GuiPolicy.load goes down this path on every checkpoint load.
        from copy import deepcopy

        tokenizer = deepcopy(base_tokenizer)
        first = ActionTokenizer(tokenizer, policy_config.action_space)
        second = ActionTokenizer(tokenizer, policy_config.action_space)
        assert second.reloaded and not first.reloaded
        assert second.signature() == first.signature()
        assert second.action_id_start == first.action_id_start

    def test_partial_overlap_is_refused(self, base_tokenizer, policy_config):
        # Appending the rest would leave the block non-contiguous, which the
        # decoding masks and the embedding parameter group both rely on.
        from copy import deepcopy

        tokenizer = deepcopy(base_tokenizer)
        tokenizer.add_special_tokens({"additional_special_tokens": ["<DONE>"]})
        with pytest.raises(TokenizerMismatch, match="non-contiguous"):
            ActionTokenizer(tokenizer, policy_config.action_space)


class TestConstrainedDecoding:
    @pytest.fixture
    def decoder(self, action_tokenizer, policy_config):
        from gui_agent.actions import ActionCodec

        return ConstrainedActionDecoder(action_tokenizer, ActionCodec(policy_config.action_space))

    def _stepper(self, decoder, preferred):
        """A model that ranks the given atoms highest and noise elsewhere."""
        size = len(decoder.tokenizer)

        def step(ids):
            logits = torch.randn(size)
            if len(ids) < len(preferred):
                logits[decoder.tokenizer.atom_id(preferred[len(ids)])] = 100.0
            return logits

        return step

    def test_control_action_decodes_in_four_tokens(self, decoder):
        result = decoder.decode(
            self._stepper(decoder, ["<MOVE_CLICK>", "<ROW_2>", "<COL_3>", "<OFFSET_1>"])
        )
        assert result.ok and result.n_steps == 4
        assert result.action.type is ActionType.MOVE_CLICK
        assert not result.used_type_path

    def test_field_fill_decodes_in_three_tokens(self, decoder):
        result = decoder.decode(
            self._stepper(decoder, ["<TYPE_START>", "<FIELD_1>", "<TYPE_END>"]),
            form_fields=["name", "email"],
        )
        assert result.ok and result.n_steps == 3
        assert result.action.field == "email"
        assert result.used_type_path

    def test_terminal_action_decodes_in_one_token(self, decoder):
        result = decoder.decode(self._stepper(decoder, ["<DONE>"]))
        assert result.ok and result.n_steps == 1

    def test_illegal_first_token_is_unreachable(self, decoder):
        # Even when the model overwhelmingly prefers it.
        size = len(decoder.tokenizer)
        row_id = decoder.tokenizer.atom_id("<ROW_1>")

        def step(ids):
            logits = torch.full((size,), -50.0)
            logits[row_id] = 1000.0
            return logits

        result = decoder.decode(step)
        assert result.atoms[0] != "<ROW_1>"

    def test_free_text_cannot_emit_an_action_token(self, decoder):
        mask = decoder.mask_for(DecodeState.IN_FREE_TEXT)
        start, end = decoder.tokenizer.text_ids_excluded()
        action_ids = mask[start:end].clone()
        type_end = decoder.tokenizer.atom_id("<TYPE_END>") - start
        action_ids[type_end] = False  # <TYPE_END> is the one legal action token
        assert not action_ids.any()

    def test_decoding_stops_at_the_budget(self, decoder):
        size = len(decoder.tokenizer)
        type_start = decoder.tokenizer.atom_id("<TYPE_START>")

        def never_ends(ids):
            logits = torch.randn(size)
            if not ids:
                logits[type_start] = 100.0
            else:
                logits[decoder.tokenizer.atom_id("<TYPE_END>")] = -1000.0
            return logits

        result = decoder.decode(never_ends, max_type_steps=6)
        assert not result.ok and result.truncated

    def test_allowed_sets_match_the_grammar(self, decoder):
        n_actions = len(decoder.codec.vocab.of_class(TokenClass.ACTION))
        assert len(decoder.allowed_ids(DecodeState.AWAIT_ACTION)) == n_actions
        assert len(decoder.allowed_ids(DecodeState.AWAIT_ROW)) == decoder.codec.config.grid_rows


class TestPolicy:
    def _batch(self, policy, batch_size=2, weights=(1.0, 4.0)):
        codec = policy.codec
        tokenizer = policy.tokenizer
        prefix = policy.build_prefix("click the submit button", {"email": "j@x.com"})
        history = tokenizer.encode_atoms(
            codec.encode(Action(ActionType.SCROLL, scroll_dx=0, scroll_dy=-2))
        )
        target = tokenizer.encode_atoms(
            codec.encode(Action(ActionType.MOVE_CLICK, row=1, col=2, offset=3))
        )
        suffix = history + target
        size = policy.config.vision.image_size
        return PolicyBatch(
            pixels=torch.randn(batch_size, 3, size, size),
            prefix_ids=torch.tensor([prefix] * batch_size),
            prefix_mask=torch.ones(batch_size, len(prefix), dtype=torch.long),
            suffix_ids=torch.tensor([suffix] * batch_size),
            suffix_mask=torch.ones(batch_size, len(suffix), dtype=torch.long),
            labels=torch.tensor([[IGNORE_INDEX] * len(history) + target] * batch_size),
            loss_weights=torch.tensor(list(weights[:batch_size])),
        )

    def test_forward_produces_a_finite_loss(self, tiny_policy):
        out = tiny_policy(self._batch(tiny_policy))
        assert torch.isfinite(out["loss"])
        assert out["n_target_tokens"] > 0

    def test_gradients_reach_the_vision_tower_and_projector(self, tiny_policy):
        tiny_policy(self._batch(tiny_policy))["loss"].backward()
        assert tiny_policy.vision.pos_embed.grad is not None
        assert any(p.grad is not None for p in tiny_policy.projector.parameters())

    def test_loss_weights_change_the_loss(self, tiny_policy):
        low = tiny_policy(self._batch(tiny_policy, weights=(1.0, 1.0)))["loss"]
        high = tiny_policy(self._batch(tiny_policy, weights=(1.0, 100.0)))["loss"]
        # Same data, different weighting: the weighted mean must move.
        assert not torch.isclose(low, high)

    def test_act_returns_a_valid_action(self, tiny_policy, frame):
        tiny_policy.eval()
        result = tiny_policy.act(frame, "click the submit button", {"email": "j@x.com"})
        assert result.ok
        assert result.n_steps <= tiny_policy.codec.grammar.max_control_atoms()

    def test_prefix_cache_is_reused_and_trimmed(self, tiny_policy, frame):
        # The cache must not grow across ticks: those entries describe a screen
        # that no longer exists, and a cache one token too long misaligns every
        # position after it.
        tiny_policy.eval()
        session = PolicySession(tiny_policy, "click submit", {"email": "j@x.com"}).warm(
            torch.device("cpu")
        )
        lengths = []
        for _ in range(4):
            # Same (empty) history each tick, so any growth is the previous
            # tick's image and action tokens failing to be trimmed away.
            assert session.fresh_cache() is not None
            assert session._cache_length(session._cache) == session.prefix_len
            tiny_policy.act(frame, "click submit", {"email": "j@x.com"}, [], session=session)
            lengths.append(session._cache_length(session._cache))
        assert len(set(lengths)) == 1
        assert lengths[0] > session.prefix_len

    def test_training_prefix_matches_the_inference_prefix(self, tiny_policy):
        # Train/serve prompt skew shows up only as degraded rollouts.
        from gui_agent.data.dataset import _render_prefix
        from gui_agent.data.schema import TrainingExample

        example = TrainingExample(
            "s", 0, ["<DONE>"], instruction="click submit", form_data={"email": "j@x.com"}
        )
        rendered = _render_prefix(example, tiny_policy.config)
        expected = tiny_policy.tokenizer.encode_text(rendered)
        assert expected == tiny_policy.build_prefix("click submit", {"email": "j@x.com"})

    def test_save_and_load_roundtrip(self, tiny_policy, tmp_path, frame):
        tiny_policy.eval()
        tiny_policy.save(tmp_path / "ckpt")
        assert (tmp_path / "ckpt" / "action_vocab.json").exists()
        restored = GuiPolicy.load(tmp_path / "ckpt")
        assert restored.tokenizer.signature() == tiny_policy.tokenizer.signature()
        assert restored.config.action_space == tiny_policy.config.action_space

    def test_embedding_gradients_are_masked_to_the_new_tokens(self, action_tokenizer):
        from transformers import LlamaConfig, LlamaForCausalLM

        from gui_agent.model.policy import _freeze_embeddings_except

        lm = LlamaForCausalLM(
            LlamaConfig(vocab_size=len(action_tokenizer), hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
        )
        _freeze_embeddings_except(lm, action_tokenizer.new_token_slice(), True)
        weight = lm.get_input_embeddings().weight
        ids = torch.tensor([[1, 2, 3]])
        lm(input_ids=ids, labels=ids).loss.backward()
        # Everything below the action block must be exactly zero: the language
        # priors are what the LM was brought in for.
        assert weight.grad[: action_tokenizer.action_id_start].abs().sum() == 0
