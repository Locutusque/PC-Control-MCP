"""Command line for training, evaluation and benchmarking.

    python -m gui_agent.train.cli benchmark --policy runs/stage2/best
    python -m gui_agent.train.cli stage1 --data data/stage1.jsonl
    python -m gui_agent.train.cli stage2 --data data/stage2.jsonl --init runs/stage1/best
    python -m gui_agent.train.cli collect --policy runs/stage2/best --tasks tasks.jsonl
    python -m gui_agent.train.cli stage3 --data data/stage3.jsonl --init runs/stage2/best
    python -m gui_agent.train.cli eval --policy runs/stage2/best
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..config import HarnessConfig, PolicyConfig, TrainConfig

log = logging.getLogger(__name__)


def _train_config(args, stage: str) -> TrainConfig:
    if args.train_config:
        return TrainConfig.load(args.train_config)
    return TrainConfig(
        stage=stage,
        output_dir=args.output or f"runs/{stage}",
        epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.workers,
    )


def _policy_config(args) -> PolicyConfig:
    if args.policy_config:
        return PolicyConfig.load(args.policy_config)
    return PolicyConfig()


def cmd_stage1(args) -> int:
    from .stage1_pretrain import run_stage1

    state = run_stage1(
        args.data, _policy_config(args), _train_config(args, "stage1"),
        pool_root=args.pool, raw_root=args.raw, val_path=args.val, device=args.device,
    )
    print(json.dumps({"steps": state.step, "best_val_loss": state.best_val_loss}, indent=2))
    return 0


def cmd_stage2(args) -> int:
    from .stage2_finetune import run_stage2

    state = run_stage2(
        args.data, _policy_config(args), _train_config(args, "stage2"),
        pool_root=args.pool, raw_root=args.raw, val_path=args.val,
        device=args.device, init_from=args.init,
    )
    print(json.dumps({"steps": state.step, "best_val_loss": state.best_val_loss}, indent=2))
    return 0


def cmd_stage3(args) -> int:
    from .stage3_dagger import run_stage3

    state = run_stage3(
        args.data, base_dataset_path=args.mix, train_config=_train_config(args, "stage3"),
        pool_root=args.pool, raw_root=args.raw, init_from=args.init, device=args.device,
    )
    print(json.dumps({"steps": state.step}, indent=2))
    return 0


def cmd_collect(args) -> int:
    """Run the current policy on a task set and write corrected examples."""
    from ..capture.schema import read_jsonl
    from ..harness.server import build_executor
    from .stage3_dagger import (
        LLMCorrector, RolloutCollector, Task, build_dagger_examples,
    )
    from ..data.schema import save_examples

    tasks = [Task.from_dict(row) for row in read_jsonl(args.tasks)]
    if not tasks:
        print(f"no tasks in {args.tasks}")
        return 1

    harness = HarnessConfig.load(args.harness_config) if args.harness_config else HarnessConfig()
    executor = build_executor(args.policy, harness, args.backend, use_llm_verifier=not args.no_verifier)
    collector = RolloutCollector(executor.loop, output_dir=args.output or "runs/stage3/rollouts")
    records = collector.collect(tasks, rounds=args.rounds)

    failures = [r for r in records if r.failed or r.is_near_miss]
    print(f"collected {len(records)} rollouts; {len(failures)} failed or nearly did")

    corrections = []
    if not args.no_corrections and failures:
        corrector = LLMCorrector()
        for record in failures:
            corrections.extend(corrector.correct(record))
        print(f"produced {len(corrections)} corrections")

    examples = build_dagger_examples(records, corrections)
    save_examples(args.out, examples)
    print(f"wrote {len(examples)} stage-3 examples to {args.out}")
    return 0


def cmd_eval(args) -> int:
    from ..model.policy import GuiPolicy
    from .eval import evaluate_all

    policy = GuiPolicy.load(args.policy)
    loop = None
    if args.live:
        from ..harness.server import build_executor

        loop = build_executor(args.policy, HarnessConfig(), args.backend).loop
    print(json.dumps(evaluate_all(policy, args.eval_sets, loop), indent=2))
    return 0


def cmd_benchmark(args) -> int:
    from ..model.policy import GuiPolicy
    from .benchmark_latency import benchmark_policy

    policy = GuiPolicy.load(args.policy, device=args.device or "cpu")
    report = benchmark_policy(
        policy, n_samples=args.samples, target_hz=args.target_hz
    ).summary()
    print(json.dumps(report, indent=2))
    if not report.get("meets_target"):
        print(
            f"\nwarning: p95 {report.get('p95_ms')}ms exceeds the "
            f"{report.get('budget_ms')}ms budget for {report.get('target_hz')}Hz.\n"
            "Before optimising, check correctness first (plan 7.2 step 7). Then, in "
            "order of usual impact: quantise the decoder (load_in_4bit), switch the "
            "projector to 'perceiver' to cut the image prefix, then shrink the ViT."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gui-agent-train", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def train_args(p):
        p.add_argument("--data", required=True)
        p.add_argument("--val", default=None)
        p.add_argument("--pool", default="data_pool")
        p.add_argument("--raw", default=None)
        p.add_argument("--output", default=None)
        p.add_argument("--policy-config", default=None)
        p.add_argument("--train-config", default=None)
        p.add_argument("--epochs", type=int, default=1)
        p.add_argument("--max-steps", type=int, default=-1)
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--grad-accum", type=int, default=4)
        p.add_argument("--lr", type=float, default=2e-4)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--workers", type=int, default=4)
        p.add_argument("--device", default=None)
        return p

    train_args(sub.add_parser("stage1", help="movement pretraining")).set_defaults(func=cmd_stage1)

    p = train_args(sub.add_parser("stage2", help="instruction fine-tuning"))
    p.add_argument("--init", default=None, help="stage-1 checkpoint to continue from")
    p.set_defaults(func=cmd_stage2)

    p = train_args(sub.add_parser("stage3", help="closed-loop correction"))
    p.add_argument("--init", required=True, help="stage-2 checkpoint to correct")
    p.add_argument("--mix", default=None, help="stage-2 dataset to mix back in")
    p.set_defaults(func=cmd_stage3)

    p = sub.add_parser("collect", help="collect closed-loop rollouts and corrections")
    p.add_argument("--policy", required=True)
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", default="data/stage3.jsonl")
    p.add_argument("--output", default=None, help="rollout output directory")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--backend", default="auto")
    p.add_argument("--harness-config", default=None)
    p.add_argument("--no-verifier", action="store_true")
    p.add_argument("--no-corrections", action="store_true")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("eval", help="run the evaluation suites")
    p.add_argument("--policy", required=True)
    p.add_argument("--eval-sets", default="eval_sets")
    p.add_argument("--live", action="store_true", help="also run end-to-end tasks")
    p.add_argument("--backend", default="auto")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("benchmark", help="measure control-tick latency")
    p.add_argument("--policy", required=True)
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--target-hz", type=float, default=15.0)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_benchmark)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
