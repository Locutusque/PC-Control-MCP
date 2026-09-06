"""End-to-end: captured events -> dataset -> training step -> closed-loop rollout.

Each stage is covered on its own elsewhere; this checks that they connect --
that the tokens the encoder emits are the ones the tokenizer accepts, that the
prompt the dataset builds is the prompt the policy builds, and that a
checkpoint written by the trainer can be loaded and driven by the harness.

Those seams are where a refactor breaks things quietly, because every unit
test still passes.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gui_agent.actions import Action
from gui_agent.capture.schema import (
    EventType,
    FrameRecord,
    InputEvent,
    SegmentMeta,
    write_jsonl,
)
from gui_agent.config import HarnessConfig, TrainConfig
from gui_agent.data.dataset import collate_examples
from gui_agent.data.finetune_dataset import FinetuneOptions, build_finetune_examples
from gui_agent.data.pretrain_dataset import PRETRAIN_INSTRUCTION, build_pretrain_examples
from gui_agent.data.schema import ExampleKind
from gui_agent.harness.dispatch import Dispatcher, NullBackend
from gui_agent.harness.loop import ControlLoop, SubtaskStatus
from gui_agent.harness.mcp_tool import LowLevelTaskExecutor
from gui_agent.harness.safety import ActionGuard
from gui_agent.model.policy import GuiPolicy
from gui_agent.train.common import build_optimizer

FORM = {"email": "j.doe@example.com", "name": "Jane Doe"}


@pytest.fixture
def pool(tmp_path):
    """A capture pool with one promoted, labelled segment."""
    root = tmp_path / "pool"
    records = [
        (0, 0.00, InputEvent(EventType.MOUSE_DOWN, x=400, y=300, button="left"), {}),
        (1, 0.05, InputEvent(EventType.MOUSE_UP, x=401, y=300, button="left"), {}),
        (2, 0.06, InputEvent(EventType.CLICK, x=401, y=300, button="left"), {}),
        (3, 0.70, InputEvent(EventType.TEXT, text="j.doe@example.com"), FORM),
        (4, 1.30, InputEvent(EventType.TEXT, text="please expedite"), FORM),
        (5, 1.70, InputEvent(EventType.SCROLL, x=500, y=500, dx=0, dy=-2), {}),
        (6, 1.90, InputEvent(EventType.KEY_DOWN, key="ENTER"), {}),
    ]
    write_jsonl(
        root / "records" / "seg_1.jsonl",
        [
            FrameRecord(
                timestamp=t, session_id="sess_1", segment_id="seg_1", frame_index=i,
                event=event.to_dict(), screen_w=1920, screen_h=1080, app_context="chrome",
                form_data_visible=form,
            )
            for i, t, event, form in records
        ],
    )
    write_jsonl(
        root / "segments" / "seg_1.json",
        [
            SegmentMeta(
                segment_id="seg_1", session_id="sess_1", started_at=0.0, ended_at=2.0,
                n_frames=7, records_path="records/seg_1.jsonl", app_context="chrome",
                user_label="fill in the support form and submit it",
                redaction_status="clean", promoted_at=1.0,
            )
        ],
    )
    return root


class TestCaptureToDataset:
    def test_stage1_drops_the_instruction_and_the_visible_values(self, pool, codec):
        examples = build_pretrain_examples(pool, codec=codec)
        assert examples
        assert all(e.instruction == PRETRAIN_INSTRUCTION for e in examples)
        # Supplying form_data here would train the routing decision against a
        # context stage 2 never reproduces.
        assert all(e.form_data == {} for e in examples)
        assert all(e.kind is ExampleKind.MOVEMENT for e in examples)
        # DONE has no meaning without a goal, so it is not supervised here.
        assert not any(e.target == ["<DONE>"] for e in examples)

    def test_stage2_carries_the_goal_and_splits_typing(self, pool, codec):
        examples = build_finetune_examples(pool, codec=codec, options=FinetuneOptions())
        assert examples
        assert all(e.instruction == "fill in the support form and submit it" for e in examples)

        kinds = {e.kind for e in examples}
        assert ExampleKind.FIELD_FILL in kinds
        assert ExampleKind.FREE_COMPOSE in kinds
        assert ExampleKind.TERMINAL in kinds

        field_fill = next(e for e in examples if e.kind is ExampleKind.FIELD_FILL)
        assert field_fill.target == ["<TYPE_START>", "<FIELD_0>", "<TYPE_END>"]
        assert field_fill.form_data["email"] == FORM["email"]
        # Correctness requirement, so it is weighted above the rest.
        assert field_fill.loss_weight > 1.0

        free = next(e for e in examples if e.kind is ExampleKind.FREE_COMPOSE)
        assert free.target[1] == "please expedite"

    def test_form_data_is_the_union_across_the_trajectory(self, pool, codec):
        # At inference the orchestrator supplies the whole dict up front;
        # training on a per-frame subset would train a context that never occurs.
        examples = build_finetune_examples(pool, codec=codec, options=FinetuneOptions())
        assert all(e.form_data == FORM for e in examples)


class TestDatasetToModel:
    def test_every_encoded_target_is_accepted_by_the_tokenizer(self, pool, tiny_policy):
        """The seam that silently breaks: encoder output vs tokenizer input."""
        examples = build_finetune_examples(
            pool, codec=tiny_policy.codec, options=FinetuneOptions()
        )
        assert examples
        for example in examples:
            ids = tiny_policy.tokenizer.encode_atoms(example.target)
            assert ids
            decoded = tiny_policy.tokenizer.decode_atoms(ids)
            # Round-tripping must recover a parseable action.
            action = tiny_policy.codec.decode(decoded, example.form_fields)
            assert isinstance(action, Action)

    def test_a_full_training_step_runs_on_real_examples(self, pool, tiny_policy):
        examples = build_finetune_examples(
            pool, codec=tiny_policy.codec, options=FinetuneOptions()
        )
        size = tiny_policy.config.vision.image_size
        items = []
        for example in examples[:4]:
            items.append(
                {
                    "pixels": torch.randn(3, size, size),
                    "prefix_ids": tiny_policy.build_prefix(
                        example.instruction, example.form_data
                    ),
                    "history_ids": [
                        i
                        for atoms in example.history
                        for i in tiny_policy.tokenizer.encode_atoms(atoms)
                    ],
                    "target_ids": tiny_policy.tokenizer.encode_atoms(example.target),
                    "loss_weight": example.loss_weight,
                }
            )
        batch = collate_examples(items, tiny_policy.tokenizer.pad_token_id)

        config = TrainConfig(stage="stage2", lr=1e-3, embedding_lr=1e-2)
        optimizer = build_optimizer(tiny_policy, config)
        before = tiny_policy(batch)["loss"].item()
        for _ in range(6):
            optimizer.zero_grad()
            tiny_policy(batch)["loss"].backward()
            optimizer.step()
        after = tiny_policy(batch)["loss"].item()
        # Six steps on four examples must at least move the loss down; if it
        # does not, the labels are not reaching the objective.
        assert after < before

    def test_optimizer_separates_the_new_embeddings(self, tiny_policy):
        groups = build_optimizer(tiny_policy, TrainConfig(lr=2e-4, embedding_lr=1e-3)).param_groups
        assert len(groups) == 2
        rates = sorted(g["lr"] for g in groups)
        assert rates == [2e-4, 1e-3]
        # Shrinking an embedding toward zero is forgetting, not regularisation.
        embedding_group = max(groups, key=lambda g: g["lr"])
        assert embedding_group["weight_decay"] == 0.0


class TestModelToHarness:
    def test_a_saved_checkpoint_drives_the_harness(
        self, tmp_path, tiny_policy, safety_config, codec
    ):
        tiny_policy.eval()
        tiny_policy.save(tmp_path / "ckpt")
        policy = GuiPolicy.load(tmp_path / "ckpt")
        policy.eval()

        frames = [
            np.random.RandomState(i).randint(0, 255, (120, 160, 3), dtype=np.uint8)
            for i in range(40)
        ]
        iterator, last = iter(frames), [None]

        def capture():
            try:
                last[0] = next(iterator)
            except StopIteration:
                pass
            return last[0]

        dispatcher = Dispatcher(
            NullBackend((1920, 1080)),
            ActionGuard(safety_config, policy.codec, (1920, 1080)),
            policy.codec,
            safety_config,
        )
        loop = ControlLoop(
            policy, dispatcher, capture,
            HarnessConfig(target_hz=0, stuck_frames=4, safety=safety_config),
            foreground=lambda: "chrome",
        )
        executor = LowLevelTaskExecutor(
            loop, trace_dir=tmp_path / "rollouts", return_screenshot=False
        )

        payload = executor.execute(
            "fill in the support form", form_data=FORM, timeout_s=8.0
        )

        # An untrained policy will not succeed; what must hold is that the loop
        # terminates cleanly through a defined stop condition, every action it
        # produced was well-formed, and the trace survived for stage 3.
        assert payload["status"] in {s.value for s in SubtaskStatus}
        assert payload["status"] != SubtaskStatus.ERROR.value
        assert payload["stop_reason"]
        assert "rollout_id" in payload

        assert (tmp_path / "rollouts" / f"{payload['rollout_id']}.json").exists()

    def test_actions_reaching_the_os_are_always_well_formed(
        self, tmp_path, tiny_policy, safety_config
    ):
        """The grammar is what guarantees this; the harness never parses text."""
        tiny_policy.eval()
        backend = NullBackend((1920, 1080))
        dispatcher = Dispatcher(
            backend, ActionGuard(safety_config, tiny_policy.codec, (1920, 1080)),
            tiny_policy.codec, safety_config,
        )
        frame = np.random.RandomState(3).randint(0, 255, (120, 160, 3), dtype=np.uint8)
        for _ in range(12):
            result = tiny_policy.act(frame, "do the thing", FORM, [])
            assert result.ok, result.error
            dispatcher.send(result.action, "chrome", FORM)
        for name, args, _ in backend.calls:
            if name == "click":
                x, y = args[0], args[1]
                assert 0 <= x < 1920 and 0 <= y < 1080
            if name == "type_text":
                # Field-fill copies verbatim, so anything typed that matches a
                # form key must be exactly the supplied value.
                assert args[0] in FORM.values() or args[0] not in FORM
