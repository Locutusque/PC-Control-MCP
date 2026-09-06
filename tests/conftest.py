"""Shared fixtures.

The heavy fixtures build a *tiny* real policy -- a 2-layer ViT and a 2-layer
Llama with a locally-trained byte-BPE tokenizer -- rather than mocking the
model.  Nothing is downloaded, it runs on CPU in a couple of seconds, and it
exercises the real shapes, the real cache handling and the real grammar, which
mocks would not.
"""

from __future__ import annotations

import pytest

from gui_agent.actions import ActionCodec
from gui_agent.capture.schema import EventType, FrameRecord, InputEvent
from gui_agent.config import (
    ActionSpaceConfig,
    DecoderConfig,
    PolicyConfig,
    ProjectorConfig,
    SafetyConfig,
    VisionConfig,
)


@pytest.fixture
def action_space() -> ActionSpaceConfig:
    return ActionSpaceConfig(
        grid_rows=8, grid_cols=8, offset_grid=4, max_form_fields=4, scroll_range=3
    )


@pytest.fixture
def codec(action_space) -> ActionCodec:
    return ActionCodec(action_space)


@pytest.fixture
def safety_config() -> SafetyConfig:
    # Tests never touch a real desktop, so the sandbox gate and the audit log
    # are both off; every other guard stays on so the tests exercise them.
    return SafetyConfig(audit_log=None, require_sandbox=False)


@pytest.fixture
def make_record():
    """Build a FrameRecord carrying one input event."""

    def _make(index: int, timestamp: float, event: InputEvent, **kwargs) -> FrameRecord:
        kwargs.setdefault("screen_w", 1920)
        kwargs.setdefault("screen_h", 1080)
        kwargs.setdefault("app_context", "chrome")
        return FrameRecord(
            timestamp=timestamp, session_id="sess_test", segment_id="seg_test",
            frame_index=index, event=event.to_dict(), **kwargs,
        )

    return _make


@pytest.fixture
def click_trajectory(make_record):
    """A short, realistic trajectory: click, type a known value, scroll, enter."""
    form = {"email": "j.doe@example.com", "name": "Jane Doe"}
    return [
        make_record(0, 0.00, InputEvent(EventType.MOUSE_DOWN, x=400, y=300, button="left")),
        make_record(1, 0.05, InputEvent(EventType.MOUSE_UP, x=401, y=300, button="left")),
        make_record(2, 0.06, InputEvent(EventType.CLICK, x=401, y=300, button="left")),
        make_record(3, 0.60, InputEvent(EventType.TEXT, text="j.doe@example.com"),
                    form_data_visible=form),
        make_record(4, 1.20, InputEvent(EventType.TEXT, text="where is my order")),
        make_record(5, 1.60, InputEvent(EventType.SCROLL, x=500, y=500, dx=0, dy=-2)),
        make_record(6, 1.80, InputEvent(EventType.KEY_DOWN, key="ENTER")),
    ]


# --------------------------------------------------------------------------
# Model fixtures (torch)
# --------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="model tests need torch")


@pytest.fixture(scope="session")
def base_tokenizer():
    """A tiny byte-level BPE tokenizer, trained here so nothing is downloaded."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["instruction form_data actions click type email name submit button hello world"] * 64,
        trainers.BpeTrainer(
            vocab_size=400, special_tokens=["<unk>", "<s>", "</s>", "<pad>"]
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>",
        eos_token="</s>", pad_token="<pad>",
    )


@pytest.fixture
def policy_config(action_space) -> PolicyConfig:
    vision = VisionConfig(image_size=128, patch_size=32, dim=96, depth=2, heads=3)
    return PolicyConfig(
        vision=vision,
        action_space=ActionSpaceConfig.for_vision(
            vision, offset_grid=action_space.offset_grid,
            max_form_fields=action_space.max_form_fields,
            scroll_range=action_space.scroll_range,
        ),
        projector=ProjectorConfig(kind="mlp", hidden_dim=128),
        decoder=DecoderConfig(torch_dtype="float32"),
        max_history=4,
    )


@pytest.fixture
def action_tokenizer(base_tokenizer, policy_config):
    from copy import deepcopy

    from gui_agent.model.tokenizer_ext import ActionTokenizer

    # Deep-copied: ActionTokenizer mutates the tokenizer it is given, and the
    # base fixture is session-scoped.
    return ActionTokenizer(deepcopy(base_tokenizer), policy_config.action_space)


@pytest.fixture
def tiny_policy(policy_config, action_tokenizer):
    from transformers import LlamaConfig, LlamaForCausalLM

    from gui_agent.model.policy import GuiPolicy

    # Seeded: an untrained policy emits essentially random actions, and an
    # unseeded one makes any assertion about its behaviour intermittent.
    torch.manual_seed(0)
    lm = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(action_tokenizer), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=2048,
        )
    )
    return GuiPolicy(policy_config, action_tokenizer, lm)


@pytest.fixture
def frame():
    import numpy as np

    return np.random.RandomState(0).randint(0, 255, (240, 320, 3), dtype=np.uint8)
