"""Stage 1: unconditioned movement pretraining (plan sections 4.2 / 5).

Behaviour cloning with the instruction slot held constant.  The model learns UI
dynamics -- where controls are, what clicking one does, how a page settles --
without any notion of a goal.  This is to the policy what raw-text pretraining
is to a language model, and it is the stage passive capture feeds directly,
since no goal labels are needed.

DONE is deliberately not trained here: with no instruction there is nothing for
it to mean, and supervising it would teach the policy to stop at arbitrary
moments.  It is learned in stage 2, where a goal defines what finishing is.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..config import PolicyConfig, TrainConfig
from ..data.finetune_dataset import split_examples
from ..data.schema import load_examples
from .common import Trainer, build_dataloader, resolve_device, set_seed

log = logging.getLogger(__name__)

__all__ = ["run_stage1"]


def run_stage1(
    dataset_path: str | Path,
    policy_config: PolicyConfig | None = None,
    train_config: TrainConfig | None = None,
    pool_root: str | None = "data_pool",
    raw_root: str | None = None,
    val_path: str | Path | None = None,
    device: str | None = None,
):
    from ..model.policy import GuiPolicy

    policy_config = policy_config or PolicyConfig()
    config = train_config or TrainConfig(stage="stage1", output_dir="runs/stage1")
    set_seed(config.seed)

    examples = list(load_examples(dataset_path))
    if not examples:
        raise SystemExit(f"no examples in {dataset_path}")
    if val_path:
        val_examples = list(load_examples(val_path))
    else:
        examples, val_examples = split_examples(examples, val_fraction=0.02, seed=config.seed)
    log.info("stage 1: %d train / %d val examples", len(examples), len(val_examples))

    policy = (
        GuiPolicy.load(config.resume_from)
        if config.resume_from
        else GuiPolicy.from_pretrained_lm(policy_config)
    )

    trainer = Trainer(
        policy,
        config,
        build_dataloader(examples, policy, config, pool_root, raw_root),
        build_dataloader(val_examples, policy, config, pool_root, raw_root, shuffle=False)
        if val_examples
        else None,
        device=resolve_device(device),
    )
    return trainer.train()
