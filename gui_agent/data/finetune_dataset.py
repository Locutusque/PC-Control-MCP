"""Stage 2: instruction fine-tuning (plan section 4.3).

Same trajectories as stage 1, now labelled with the goal they accomplished and
conditioned on it.  The load-bearing part is the **field-fill / free-compose
split**: it has to exist in the data, because it is what teaches the model to
*route* between copying a supplied value and generating novel text (plan 2.3).

Field-fill examples carry an extra loss weight.  Copying ``form_data["email"]``
correctly is a correctness requirement, not a quality one -- any drift there is
a bug -- so the training signal says so, and at inference the ``<FIELD_k>``
token makes the copy structural rather than generated.

:func:`dataset_stats` exists because of the risk flagged in plan section 8:
narrow form coverage shows up as confident-but-wrong typing on unfamiliar
layouts, and the only way to see that coming is to look at the distribution
before training rather than after.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..actions import ActionCodec, ActionType
from .encode import EncodeConfig, TrajectoryEncoder
from .schema import ExampleKind, TrainingExample, save_examples
from .sources import iter_segments

log = logging.getLogger(__name__)

__all__ = [
    "FinetuneOptions", "build_finetune_examples", "write_finetune_dataset",
    "dataset_stats", "balance_examples", "split_examples",
]


@dataclass(frozen=True)
class FinetuneOptions:
    max_history: int = 8
    # Extra CE weight on field-fill targets (plan section 5).
    field_fill_loss_weight: float = 4.0
    free_compose_loss_weight: float = 1.0
    terminal_loss_weight: float = 2.0
    # Segments with no goal label are unusable here; stage 1 still takes them.
    require_instruction: bool = True
    # Drop hindsight labels the relabeller was not confident about, rather than
    # training instruction-following against a guess.
    min_hindsight_confidence: float = 0.5
    # Cap on how much of the dataset one segment may contribute, so a single
    # long session cannot dominate.
    max_examples_per_segment: int | None = None


def build_finetune_examples(
    pool_root: str | Path,
    raw_root: str | Path | None = None,
    codec: ActionCodec | None = None,
    encode_config: EncodeConfig | None = None,
    options: FinetuneOptions | None = None,
) -> list[TrainingExample]:
    codec = codec or ActionCodec()
    opts = options or FinetuneOptions()
    encoder = TrajectoryEncoder(codec, encode_config)
    examples: list[TrainingExample] = []
    skipped_unlabelled = 0

    for source in iter_segments(pool_root, raw_root):
        instruction = source.instruction
        if opts.require_instruction and not instruction:
            skipped_unlabelled += 1
            continue
        confidence = source.meta.hindsight_confidence
        if (
            confidence is not None
            and not source.meta.user_label       # a human label needs no confidence check
            and confidence < opts.min_hindsight_confidence
        ):
            skipped_unlabelled += 1
            continue

        actions = encoder.encode_segment(source.read_records())
        if not actions:
            continue
        histories = encoder.with_history(actions, opts.max_history)

        # form_data is the union of values visible across the trajectory: at
        # inference the orchestrator supplies the whole dict up front, so
        # training on a per-frame subset would train a context that never
        # occurs.
        form_data: dict[str, str] = {}
        for ta in actions:
            form_data.update(ta.form_data)

        built = 0
        for ta, history in zip(actions, histories):
            if opts.max_examples_per_segment and built >= opts.max_examples_per_segment:
                break
            kind = _kind_for(ta.kind, ta.action.type)
            try:
                target = codec.encode(ta.action, list(form_data.keys()))
            except Exception as exc:
                log.debug("skipping unencodable action %s: %s", ta.action.summary(), exc)
                continue
            examples.append(
                TrainingExample(
                    segment_id=source.segment_id,
                    frame_index=ta.frame_index,
                    target=target,
                    kind=kind,
                    instruction=instruction,
                    form_data=form_data,
                    history=history,
                    video_path=str(source.video) if source.video else None,
                    source_root=str(source.root),
                    screen_w=ta.screen_w,
                    screen_h=ta.screen_h,
                    app_context=ta.app_context,
                    loss_weight=_weight_for(kind, opts),
                )
            )
            built += 1

    if skipped_unlabelled:
        log.info(
            "skipped %d unlabelled segments; run hindsight_relabel to recover them",
            skipped_unlabelled,
        )
    log.info("built %d stage-2 examples", len(examples))
    return examples


def _kind_for(encoded_kind: ExampleKind, action_type: ActionType) -> ExampleKind:
    if encoded_kind in (ExampleKind.FIELD_FILL, ExampleKind.FREE_COMPOSE, ExampleKind.TERMINAL):
        return encoded_kind
    return ExampleKind.CONTROL


def _weight_for(kind: ExampleKind, opts: FinetuneOptions) -> float:
    return {
        ExampleKind.FIELD_FILL: opts.field_fill_loss_weight,
        ExampleKind.FREE_COMPOSE: opts.free_compose_loss_weight,
        ExampleKind.TERMINAL: opts.terminal_loss_weight,
    }.get(kind, 1.0)


def write_finetune_dataset(out_path: str | Path, **kwargs) -> int:
    return save_examples(out_path, build_finetune_examples(**kwargs))


# --------------------------------------------------------------------------
# Inspection and splitting
# --------------------------------------------------------------------------


def dataset_stats(examples: list[TrainingExample]) -> dict:
    """Coverage summary, meant to be read before training rather than after.

    ``field_fill_keys`` and ``apps`` are the two that matter most: narrow
    coverage on either is the failure mode plan section 8 calls out, and it is
    invisible in the loss curve.
    """
    kinds = Counter(e.kind.value for e in examples)
    apps = Counter(e.app_context or "unknown" for e in examples)
    field_keys: Counter = Counter()
    for e in examples:
        if e.kind is ExampleKind.FIELD_FILL:
            for atom in e.target:
                if atom.startswith("<FIELD_"):
                    idx = int(atom[len("<FIELD_") : -1])
                    fields = e.form_fields
                    field_keys[fields[idx] if idx < len(fields) else f"<oob:{idx}>"] += 1
    n = len(examples) or 1
    return {
        "n_examples": len(examples),
        "n_segments": len({e.segment_id for e in examples}),
        "kinds": dict(kinds),
        "field_fill_fraction": round(kinds.get("field_fill", 0) / n, 4),
        "free_compose_fraction": round(kinds.get("free_compose", 0) / n, 4),
        "field_fill_keys": dict(field_keys.most_common(30)),
        "n_distinct_field_keys": len(field_keys),
        "apps": dict(apps.most_common(20)),
        "n_distinct_apps": len(apps),
        "n_distinct_instructions": len({e.instruction for e in examples}),
    }


def balance_examples(
    examples: list[TrainingExample],
    max_ratio: float = 8.0,
    seed: int = 0,
) -> list[TrainingExample]:
    """Downsample over-represented kinds.

    Passive capture is overwhelmingly clicks and scrolls, so an unbalanced
    dataset drowns the TYPE routing decision -- the one thing stage 2 exists to
    teach -- in a sea of movement.
    """
    rng = random.Random(seed)
    by_kind: dict[ExampleKind, list[TrainingExample]] = {}
    for e in examples:
        by_kind.setdefault(e.kind, []).append(e)
    if not by_kind:
        return []

    smallest = min(len(v) for v in by_kind.values())
    cap = max(1, int(smallest * max_ratio))
    out: list[TrainingExample] = []
    for kind, items in by_kind.items():
        if len(items) > cap:
            log.info("downsampling %s from %d to %d", kind.value, len(items), cap)
            items = rng.sample(items, cap)
        out.extend(items)
    rng.shuffle(out)
    return out


def split_examples(
    examples: list[TrainingExample],
    val_fraction: float = 0.05,
    seed: int = 0,
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Train/val split **by segment**.

    Splitting by example would put consecutive frames of the same trajectory on
    both sides and report a validation number that is really training accuracy.
    """
    segments = sorted({e.segment_id for e in examples})
    rng = random.Random(seed)
    rng.shuffle(segments)
    n_val = max(1, int(len(segments) * val_fraction)) if segments else 0
    val_ids = set(segments[:n_val])
    train = [e for e in examples if e.segment_id not in val_ids]
    val = [e for e in examples if e.segment_id in val_ids]
    return train, val
