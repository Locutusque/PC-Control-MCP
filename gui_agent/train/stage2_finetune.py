"""Stage 2: instruction fine-tuning (plan sections 4.3 / 5).

Conditions the policy on a goal and, critically, teaches it to *route* typing:
select a supplied form value, or compose new text.  Field-fill spans carry
extra loss weight because copying a value exactly is a correctness requirement
rather than a quality one.

The eval hook runs :func:`field_fill_exactness` on the validation split at every
evaluation, not just at the end.  Field-fill exactness that is not ~1.0 is a bug,
and finding out at the end of a long run wastes the run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..actions import Action, ActionType
from ..config import PolicyConfig, TrainConfig
from ..data.finetune_dataset import dataset_stats
from ..data.schema import ExampleKind, TrainingExample, load_examples
from .common import Trainer, build_dataloader, resolve_device, set_seed, stage_loss_weights
from .metrics import field_fill_exactness

log = logging.getLogger(__name__)

__all__ = ["run_stage2", "field_fill_eval_hook"]


def run_stage2(
    dataset_path: str | Path,
    policy_config: PolicyConfig | None = None,
    train_config: TrainConfig | None = None,
    pool_root: str | None = "data_pool",
    raw_root: str | None = None,
    val_path: str | Path | None = None,
    device: str | None = None,
    init_from: str | None = None,
):
    from ..model.policy import GuiPolicy

    policy_config = policy_config or PolicyConfig()
    config = train_config or TrainConfig(stage="stage2", output_dir="runs/stage2")
    set_seed(config.seed)

    examples = list(load_examples(dataset_path))
    if not examples:
        raise SystemExit(f"no examples in {dataset_path}")
    stage_loss_weights(examples, config)

    val_examples = list(load_examples(val_path)) if val_path else []
    stats = dataset_stats(examples)
    log.info("stage 2 dataset: %s", stats)
    if not stats.get("field_fill_fraction"):
        log.warning(
            "no field-fill examples: the copy-vs-generate routing decision has no "
            "training signal at all, and TYPE will always generate"
        )

    # Stage 2 continues from stage 1 by default: starting fresh discards the UI
    # dynamics the movement stage exists to learn.
    start = init_from or config.resume_from
    policy = GuiPolicy.load(start) if start else GuiPolicy.from_pretrained_lm(policy_config)
    if not start:
        log.warning(
            "starting stage 2 from a fresh LM; pass init_from=<stage1 checkpoint> to "
            "keep the movement pretraining"
        )

    trainer = Trainer(
        policy,
        config,
        build_dataloader(examples, policy, config, pool_root, raw_root),
        build_dataloader(val_examples, policy, config, pool_root, raw_root, shuffle=False)
        if val_examples
        else None,
        device=resolve_device(device),
        on_eval=field_fill_eval_hook(val_examples, pool_root, raw_root),
    )
    return trainer.train()


def field_fill_eval_hook(
    val_examples: list[TrainingExample],
    pool_root: str | None,
    raw_root: str | None,
    max_examples: int = 64,
):
    """Measure field-fill exactness on the validation split during training."""
    subset = [e for e in val_examples if e.kind is ExampleKind.FIELD_FILL][:max_examples]
    if not subset:
        return None

    from ..data.dataset import GuiExampleDataset
    from ..model.vit import preprocess_screenshot

    def hook(policy, step: int) -> dict:
        dataset = GuiExampleDataset(
            subset, policy.tokenizer, policy.config, pool_root=pool_root, raw_root=raw_root
        )
        predictions: list[Action | None] = []
        expected: list[str] = []
        form_data: list[dict] = []

        policy.eval()
        with torch.no_grad():
            for i, example in enumerate(subset):
                pixels = dataset[i]["pixels"]
                result = policy.act(
                    pixels.unsqueeze(0), example.instruction, example.form_data,
                    example.history,
                )
                predictions.append(result.action if result.ok else None)
                target = policy.codec.decode(example.target, example.form_fields)
                expected.append(
                    target.resolve_text(example.form_data)
                    if target.type is ActionType.TYPE
                    else ""
                )
                form_data.append(example.form_data)
        policy.train()

        metrics = field_fill_exactness(predictions, expected, form_data)
        if metrics["exactness"] is not None and metrics["exactness"] < 1.0:
            log.warning(
                "field-fill exactness %.3f at step %d (%d routing errors): this is a "
                "correctness failure, not a quality one",
                metrics["exactness"], step, metrics["routing_errors"],
            )
        return {f"field_fill/{k}": v for k, v in metrics.items()}

    return hook
