"""Shared training machinery for all three stages (plan section 5).

All three stages optimise the same objective -- next-action-token cross-entropy
with per-example weights -- and differ only in which dataset they draw from and
which weights they attach.  That commonality lives here so a change to the
optimiser or the checkpoint format cannot drift between stages.

Two details worth knowing:

* **Two parameter groups.**  The new action-token embeddings start from noise
  and need a hotter learning rate than LoRA adapters sitting on top of already
  meaningful pretrained weights.  Running them at one rate either wastes the
  first epoch waiting for the embeddings to become useful, or shakes the
  adapters apart.
* **Checkpoints carry the action vocabulary.**  A checkpoint loaded against a
  differently-ordered vocabulary silently remaps the action embeddings, so
  every load checks the signature.
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import TrainConfig
from ..data.dataset import GuiExampleDataset, SegmentGroupedSampler, collate_examples
from ..data.schema import ExampleKind, TrainingExample
from ..model.policy import GuiPolicy, PolicyBatch
from .metrics import MetricAccumulator

log = logging.getLogger(__name__)

__all__ = ["Trainer", "set_seed", "build_dataloader", "build_optimizer", "resolve_device"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str | None = None) -> torch.device:
    if preference:
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloader(
    examples: list[TrainingExample],
    policy: GuiPolicy,
    config: TrainConfig,
    pool_root: str | None = None,
    raw_root: str | None = None,
    shuffle: bool = True,
) -> DataLoader:
    dataset = GuiExampleDataset(
        examples, policy.tokenizer, policy.config, pool_root=pool_root, raw_root=raw_root
    )
    pad_id = policy.tokenizer.pad_token_id
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=SegmentGroupedSampler(examples, seed=config.seed, shuffle=shuffle),
        num_workers=config.num_workers,
        collate_fn=lambda items: collate_examples(items, pad_id),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_optimizer(policy: GuiPolicy, config: TrainConfig):
    """Two groups: freshly-initialised embeddings, and everything else."""
    embedding_params, other_params = [], []
    embedding_weights = {
        id(policy.lm.get_input_embeddings().weight),
    }
    output = policy.lm.get_output_embeddings()
    if output is not None:
        embedding_weights.add(id(output.weight))

    for param in policy.parameters():
        if not param.requires_grad:
            continue
        (embedding_params if id(param) in embedding_weights else other_params).append(param)

    groups = [{"params": other_params, "lr": config.lr, "weight_decay": config.weight_decay}]
    if embedding_params:
        # No weight decay on embeddings: shrinking a token embedding toward
        # zero is not regularisation, it is forgetting what the token means.
        groups.append(
            {"params": embedding_params, "lr": config.embedding_lr, "weight_decay": 0.0}
        )
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def build_scheduler(optimizer, config: TrainConfig, total_steps: int):
    warmup = max(1, int(total_steps * config.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@dataclass
class TrainState:
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    history: list = field(default_factory=list)


class Trainer:
    """One training stage."""

    def __init__(
        self,
        policy: GuiPolicy,
        config: TrainConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: torch.device | None = None,
        on_eval: Callable[[GuiPolicy, int], dict] | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or resolve_device()
        self.on_eval = on_eval
        self.state = TrainState()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.policy.to(self.device)
        self.optimizer = build_optimizer(policy, config)
        self.total_steps = self._total_steps()
        self.scheduler = build_scheduler(self.optimizer, config, self.total_steps)
        self.metrics = MetricAccumulator()

        summary = policy.parameter_summary()
        log.info(
            "training %s: %.1fM trainable of %.1fM total (%.2f%%) over %d steps on %s",
            config.stage, summary["trainable"] / 1e6, summary["total"] / 1e6,
            summary["trainable_fraction"] * 100, self.total_steps, self.device,
        )

    def _total_steps(self) -> int:
        per_epoch = max(1, len(self.train_loader) // max(1, self.config.grad_accum))
        if self.config.max_steps > 0:
            return self.config.max_steps
        return per_epoch * max(1, self.config.epochs)

    # -- loop -------------------------------------------------------------
    def train(self) -> TrainState:
        self.policy.train()
        started = time.time()
        accumulated = 0

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            sampler = getattr(self.train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            for batch in self.train_loader:
                if self.state.step >= self.total_steps:
                    break
                self._train_step(batch)
                accumulated += 1

                if accumulated >= self.config.grad_accum:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.trainable_parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.step += 1
                    accumulated = 0
                    self._maybe_log(started)
                    self._maybe_eval()
                    self._maybe_save()

            if self.state.step >= self.total_steps:
                break

        self.save("final")
        if self.val_loader is not None:
            final = self.evaluate()
            log.info("final validation: %s", final)
        return self.state

    def _train_step(self, batch: PolicyBatch) -> float:
        batch = batch.to(self.device)
        outputs = self.policy(batch)
        loss = outputs["loss"] / self.config.grad_accum
        loss.backward()
        self.metrics.update(
            float(outputs["loss"].detach()), float(outputs["token_accuracy"])
        )
        return float(outputs["loss"].detach())

    @torch.no_grad()
    def evaluate(self) -> dict:
        if self.val_loader is None:
            return {}
        self.policy.eval()
        accumulator = MetricAccumulator()
        for batch in self.val_loader:
            outputs = self.policy(batch.to(self.device))
            accumulator.update(
                float(outputs["loss"]), float(outputs["token_accuracy"])
            )
        self.policy.train()
        summary = accumulator.summary()
        if self.on_eval is not None:
            summary.update(self.on_eval(self.policy, self.state.step))
        return summary

    # -- hooks ------------------------------------------------------------
    def _maybe_log(self, started: float) -> None:
        if self.state.step % self.config.log_every:
            return
        summary = self.metrics.summary()
        elapsed = time.time() - started
        summary.update(
            {
                "step": self.state.step,
                "lr": round(self.scheduler.get_last_lr()[0], 8),
                "steps_per_s": round(self.state.step / elapsed, 3) if elapsed else None,
            }
        )
        log.info("train %s", json.dumps(summary))
        self.state.history.append({"phase": "train", **summary})
        self.metrics.reset()

    def _maybe_eval(self) -> None:
        if self.config.eval_every <= 0 or self.state.step % self.config.eval_every:
            return
        summary = self.evaluate()
        if not summary:
            return
        log.info("eval %s", json.dumps({"step": self.state.step, **summary}))
        self.state.history.append({"phase": "eval", "step": self.state.step, **summary})
        if summary.get("loss", float("inf")) < self.state.best_val_loss:
            self.state.best_val_loss = summary["loss"]
            self.save("best")

    def _maybe_save(self) -> None:
        if self.config.save_every > 0 and self.state.step % self.config.save_every == 0:
            self.save(f"step_{self.state.step}")

    def save(self, name: str) -> Path:
        path = self.output_dir / name
        self.policy.save(path)
        (path / "train_state.json").write_text(
            json.dumps(
                {
                    "step": self.state.step,
                    "epoch": self.state.epoch,
                    "best_val_loss": self.state.best_val_loss,
                    "config": self.config.to_dict(),
                },
                indent=2,
            )
        )
        (self.output_dir / "history.json").write_text(json.dumps(self.state.history, indent=2))
        log.info("saved %s", path)
        return path


def stage_loss_weights(examples: Iterable[TrainingExample], config: TrainConfig) -> None:
    """Apply the stage's loss weights in place.

    Field-fill spans are weighted up because copying a supplied value exactly
    is a correctness requirement (plan 5); escalation examples are weighted in
    stage 3 to calibrate handing back.
    """
    for example in examples:
        if example.kind is ExampleKind.FIELD_FILL:
            example.loss_weight = config.copy_loss_weight
        elif example.kind is ExampleKind.ESCALATE:
            example.loss_weight = config.escalate_loss_weight
