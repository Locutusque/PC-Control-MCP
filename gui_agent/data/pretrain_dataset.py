"""Stage 1: unconditioned movement pretraining (plan section 4.2).

Same captured trajectories as stage 2, but the instruction slot is empty.  The
objective is pure behaviour cloning of UI dynamics -- cursor physics, affordance
recognition, what a control on screen does when you click it -- which is to this
model what raw-text pretraining is to a language model.

No goal labels are needed, so every promoted segment is usable and this stage
consumes the bulk of what passive capture produces.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..actions import ActionCodec
from .encode import EncodeConfig, TrajectoryEncoder
from .schema import ExampleKind, TrainingExample, save_examples
from .sources import iter_segments

log = logging.getLogger(__name__)

__all__ = ["PRETRAIN_INSTRUCTION", "build_pretrain_examples", "write_pretrain_dataset"]

# A constant placeholder rather than an empty string: the LM still sees a
# well-formed prompt, and the token is a consistent "no goal given" signal
# instead of a hole whose meaning drifts with the tokenizer.
PRETRAIN_INSTRUCTION = "continue"


def build_pretrain_examples(
    pool_root: str | Path,
    raw_root: str | Path | None = None,
    codec: ActionCodec | None = None,
    encode_config: EncodeConfig | None = None,
    max_history: int = 8,
    include_terminal: bool = False,
) -> list[TrainingExample]:
    """Every action in every promoted segment, with no instruction.

    ``include_terminal`` is off by default: without a goal there is nothing for
    DONE to mean, and training it here teaches the policy to stop at arbitrary
    moments.  DONE is supervised in stage 2, where an instruction defines what
    finishing is.
    """
    codec = codec or ActionCodec()
    encoder = TrajectoryEncoder(codec, encode_config)
    examples: list[TrainingExample] = []

    for source in iter_segments(pool_root, raw_root):
        actions = encoder.encode_segment(source.read_records())
        if not actions:
            continue
        histories = encoder.with_history(actions, max_history)

        for ta, history in zip(actions, histories, strict=True):
            if ta.kind is ExampleKind.TERMINAL and not include_terminal:
                continue
            try:
                target = codec.encode(ta.action, list(ta.form_data.keys()))
            except Exception as exc:
                log.debug("skipping unencodable action %s: %s", ta.action.summary(), exc)
                continue
            examples.append(
                TrainingExample(
                    segment_id=source.segment_id,
                    frame_index=ta.frame_index,
                    target=target,
                    kind=ExampleKind.MOVEMENT,
                    instruction=PRETRAIN_INSTRUCTION,
                    # Values on screen are deliberately dropped: with no
                    # instruction there is no field to fill, and supplying
                    # form_data here would train the routing decision against
                    # a context stage 2 never reproduces.
                    form_data={},
                    history=history,
                    video_path=str(source.video) if source.video else None,
                    source_root=str(source.root),
                    screen_w=ta.screen_w,
                    screen_h=ta.screen_h,
                    app_context=ta.app_context,
                )
            )

    log.info("built %d stage-1 examples", len(examples))
    return examples


def write_pretrain_dataset(out_path: str | Path, **kwargs) -> int:
    examples = build_pretrain_examples(**kwargs)
    return save_examples(out_path, examples)
