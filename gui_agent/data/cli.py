"""Command line for building datasets from the capture pool.

    python -m gui_agent.data.cli relabel   --pool data_pool
    python -m gui_agent.data.cli stage1    --pool data_pool --out data/stage1.jsonl
    python -m gui_agent.data.cli stage2    --pool data_pool --out data/stage2.jsonl
    python -m gui_agent.data.cli stats     data/stage2.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..actions import ActionCodec
from ..config import ActionSpaceConfig
from .encode import EncodeConfig
from .finetune_dataset import (
    FinetuneOptions,
    balance_examples,
    build_finetune_examples,
    dataset_stats,
    split_examples,
)
from .pretrain_dataset import build_pretrain_examples
from .schema import load_examples, save_examples

log = logging.getLogger(__name__)


def _codec(args) -> ActionCodec:
    if args.action_space:
        return ActionCodec(ActionSpaceConfig.load(args.action_space))
    return ActionCodec()


def cmd_stage1(args) -> int:
    examples = build_pretrain_examples(
        pool_root=args.pool, raw_root=args.raw, codec=_codec(args),
        encode_config=EncodeConfig(), max_history=args.max_history,
    )
    if not examples:
        print("no examples built; check that segments have been promoted into the pool")
        return 1
    save_examples(args.out, examples)
    print(f"wrote {len(examples)} stage-1 examples to {args.out}")
    return 0


def cmd_stage2(args) -> int:
    examples = build_finetune_examples(
        pool_root=args.pool, raw_root=args.raw, codec=_codec(args),
        encode_config=EncodeConfig(),
        options=FinetuneOptions(max_history=args.max_history),
    )
    if not examples:
        print("no examples built; run `relabel` first if segments are unlabelled")
        return 1
    if args.balance:
        examples = balance_examples(examples, max_ratio=args.max_ratio)

    train, val = split_examples(examples, val_fraction=args.val_fraction)
    out = Path(args.out)
    save_examples(out, train)
    val_path = out.with_name(out.stem + "_val" + out.suffix)
    save_examples(val_path, val)

    stats = dataset_stats(train)
    print(f"wrote {len(train)} train / {len(val)} val examples to {out}, {val_path}")
    print(json.dumps(stats, indent=2))
    _warn_on_coverage(stats)
    return 0


def cmd_relabel(args) -> int:
    from .hindsight_relabel import HindsightRelabeler

    results = HindsightRelabeler().relabel_pool(
        args.pool, args.raw, overwrite=args.overwrite, use_batch=not args.no_batch
    )
    usable = sum(1 for r in results if r.usable)
    failed = [r for r in results if r.error]
    print(f"relabelled {len(results)} segments; {usable} usable, {len(failed)} failed")
    for r in failed[:10]:
        print(f"  {r.segment_id}: {r.error}")
    return 0


def cmd_stats(args) -> int:
    examples = list(load_examples(args.path))
    stats = dataset_stats(examples)
    print(json.dumps(stats, indent=2))
    _warn_on_coverage(stats)
    return 0


def _warn_on_coverage(stats: dict) -> None:
    """Surface the coverage risks from plan section 8 before training starts."""
    warnings = []
    if stats.get("n_distinct_field_keys", 0) < 5:
        warnings.append(
            f"only {stats.get('n_distinct_field_keys')} distinct field-fill keys: expect "
            "confident-but-wrong typing on unfamiliar form layouts"
        )
    if stats.get("field_fill_fraction", 0) < 0.01:
        warnings.append(
            "under 1% of examples are field-fill; the model has almost no signal for the "
            "copy-vs-generate routing decision"
        )
    if stats.get("n_distinct_apps", 0) < 3:
        warnings.append(
            f"only {stats.get('n_distinct_apps')} distinct apps: this will not generalise "
            "beyond the captured workflows"
        )
    if stats.get("n_distinct_instructions", 0) < 20:
        warnings.append(
            f"only {stats.get('n_distinct_instructions')} distinct instructions: instruction "
            "conditioning will overfit"
        )
    for w in warnings:
        print(f"warning: {w}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gui-agent-data", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--pool", default="data_pool")
        p.add_argument("--raw", default=None, help="raw capture root, if videos live there")
        p.add_argument("--action-space", default=None, help="ActionSpaceConfig JSON")
        return p

    p = common(sub.add_parser("stage1", help="build the movement-pretraining dataset"))
    p.add_argument("--out", default="data/stage1.jsonl")
    p.add_argument("--max-history", type=int, default=8)
    p.set_defaults(func=cmd_stage1)

    p = common(sub.add_parser("stage2", help="build the instruction-tuning dataset"))
    p.add_argument("--out", default="data/stage2.jsonl")
    p.add_argument("--max-history", type=int, default=8)
    p.add_argument("--balance", action="store_true")
    p.add_argument("--max-ratio", type=float, default=8.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.set_defaults(func=cmd_stage2)

    p = common(sub.add_parser("relabel", help="LLM hindsight instruction relabelling"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-batch", action="store_true", help="one request per segment")
    p.set_defaults(func=cmd_relabel)

    p = sub.add_parser("stats", help="summarise a built dataset")
    p.add_argument("path")
    p.set_defaults(func=cmd_stats)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
