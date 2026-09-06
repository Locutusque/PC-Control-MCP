"""Stage 3: closed-loop correction (plan sections 4.4 / 5).

Pure imitation learning drifts under closed-loop rollout.  The policy only ever
saw expert states in training, so its first small mistake puts it somewhere the
dataset never covered, and the errors compound.  DAgger fixes this by training
on the states the *policy* visits rather than the ones a human visited.

The loop:

1. run the current policy live through the harness on a task set;
2. keep the rollouts that failed or nearly did;
3. have a judge say what should have happened at each divergence;
4. fold those corrections into the dataset and retrain.

This is where GUI agents go from working in demos to being reliable, and it is
ongoing rather than a milestone with an end date -- so this module is built to
be run repeatedly, with each round's corrections accumulating in the pool.

Escalation calibration
----------------------
Failed rollouts also supply the ESCALATE signal, in both directions (plan 5 and
8).  A rollout that ground on until it timed out should have handed back
earlier: its last ticks become positive escalate examples.  A rollout that
succeeded should not have escalated anywhere: every one of its ticks is
negative evidence.  Training on only the first direction produces a policy that
escalates constantly, which defeats the point of the system.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..actions import Action, ActionCodec, ActionType
from ..config import PolicyConfig, TrainConfig
from ..data.schema import ExampleKind, TrainingExample, load_examples, save_examples
from ..harness.loop import ControlLoop, RolloutTrace, SubtaskResult, SubtaskStatus
from .common import Trainer, build_dataloader, resolve_device, set_seed

log = logging.getLogger(__name__)

__all__ = [
    "Task", "RolloutRecord", "RolloutCollector", "Correction", "LLMCorrector",
    "build_dagger_examples", "escalation_labels", "run_stage3",
]

DEFAULT_MODEL = "claude-opus-5"

# How many ticks before a failure are treated as "should have escalated".  A
# policy rarely becomes stuck in one step; it degrades over a few.
_ESCALATE_WINDOW = 5


@dataclass(frozen=True)
class Task:
    """One entry in the closed-loop task set."""

    task_id: str
    instruction: str
    form_data: dict = field(default_factory=dict)
    success_criteria: str = ""
    timeout_s: float = 60.0
    setup: Callable[[], None] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            task_id=d["task_id"],
            instruction=d["instruction"],
            form_data=d.get("form_data", {}),
            success_criteria=d.get("success_criteria", ""),
            timeout_s=float(d.get("timeout_s", 60.0)),
        )


@dataclass
class RolloutRecord:
    """A rollout plus the frames needed to judge it."""

    task: Task
    result: SubtaskResult
    frame_paths: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.result.status is not SubtaskStatus.DONE

    @property
    def is_near_miss(self) -> bool:
        """Succeeded, but with signs of trouble along the way.

        Near-misses are worth correcting: they are where the policy is one bad
        frame from failing, and waiting for an outright failure discards the
        cheapest signal available.
        """
        trace = self.result.trace
        if trace is None or self.failed:
            return False
        blocked = sum(1 for t in trace.ticks if t.blocked_reason)
        stalled = sum(1 for t in trace.ticks if not t.frame_changed)
        return blocked > 0 or stalled >= 3

    def to_dict(self) -> dict:
        return {
            "task_id": self.task.task_id,
            "instruction": self.task.instruction,
            "status": self.result.status.value,
            "stop_reason": self.result.stop_reason,
            "n_ticks": self.result.n_ticks,
            "frame_paths": self.frame_paths,
            "trace": self.result.trace.to_dict() if self.result.trace else None,
        }


class RolloutCollector:
    """Runs the current policy over a task set and saves what happened."""

    def __init__(
        self,
        loop: ControlLoop,
        output_dir: str | Path = "runs/stage3/rollouts",
        save_frames: bool = True,
        max_frames_per_rollout: int = 200,
    ) -> None:
        self.loop = loop
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_frames = save_frames
        self.max_frames = max_frames_per_rollout

    def collect(self, tasks: Sequence[Task], rounds: int = 1) -> list[RolloutRecord]:
        records: list[RolloutRecord] = []
        for round_index in range(rounds):
            for task in tasks:
                if task.setup is not None:
                    task.setup()
                record = self._run_one(task, round_index)
                records.append(record)
                log.info(
                    "[round %d] %s -> %s (%d ticks)",
                    round_index, task.task_id, record.result.status.value,
                    record.result.n_ticks,
                )
        self._write_manifest(records)
        return records

    def _run_one(self, task: Task, round_index: int) -> RolloutRecord:
        frames: list = []
        original_capture = self.loop.capture

        def capturing():
            frame = original_capture()
            if self.save_frames and len(frames) < self.max_frames:
                frames.append(frame)
            return frame

        self.loop.capture = capturing
        try:
            result = self.loop.run(
                task.instruction, task.form_data, task.timeout_s, task.success_criteria
            )
        finally:
            self.loop.capture = original_capture

        paths: list[str] = []
        if self.save_frames and result.trace is not None:
            paths = self._save_frames(result.trace.rollout_id, frames)
        return RolloutRecord(task, result, paths)

    def _save_frames(self, rollout_id: str, frames: list) -> list[str]:
        directory = self.output_dir / rollout_id
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []
        try:
            from PIL import Image
            import numpy as np

            for index, frame in enumerate(frames):
                array = np.asarray(frame)
                if array.ndim == 3 and array.shape[2] == 4:
                    array = array[:, :, :3]
                path = directory / f"{index:05d}.png"
                Image.fromarray(array.astype("uint8")).save(path)
                paths.append(str(path))
        except Exception as exc:  # pragma: no cover
            log.warning("could not save rollout frames: %s", exc)
        return paths

    def _write_manifest(self, records: Sequence[RolloutRecord]) -> None:
        path = self.output_dir / "rollouts.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
        log.info("wrote %d rollouts to %s", len(records), path)


# --------------------------------------------------------------------------
# Correction
# --------------------------------------------------------------------------


@dataclass
class Correction:
    """What should have happened at one tick."""

    rollout_id: str
    tick_index: int
    corrected_atoms: list[str]
    reason: str = ""
    kind: ExampleKind = ExampleKind.CONTROL


class LLMCorrector:
    """Asks a vision model what the policy should have done instead.

    A human is better at this and should review anything the judge is unsure
    of; the point of the automated pass is to make the volume tractable, not to
    remove the human.  Corrections are written to disk in a reviewable form for
    exactly that reason.
    """

    def __init__(self, codec: ActionCodec | None = None, model: str = DEFAULT_MODEL,
                 client=None) -> None:
        self.codec = codec or ActionCodec()
        self.model = model
        self._client = client

    def _get_client(self):
        if self._client is None:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic()
        return self._client

    def correct(self, record: RolloutRecord) -> list[Correction]:
        trace = record.result.trace
        if trace is None or not trace.ticks:
            return []

        # Focus on where it went wrong: blocked actions, stalled frames, and
        # the run-up to the failure. Asking about every tick of a 300-tick
        # rollout is mostly asking about ticks that were fine.
        suspects = [
            t.index for t in trace.ticks
            if t.blocked_reason or not t.frame_changed
        ]
        suspects += [t.index for t in trace.ticks[-_ESCALATE_WINDOW:]]
        suspects = sorted(set(suspects))
        if not suspects:
            return []

        prompt = _correction_prompt(record, suspects)
        images = _load_images(record.frame_paths, suspects)
        try:
            data = self._ask(prompt, images)
        except Exception as exc:
            log.warning("correction failed for %s: %s", trace.rollout_id, exc)
            return []

        corrections: list[Correction] = []
        for entry in data.get("corrections", []):
            atoms = entry.get("atoms") or []
            if not self._valid(atoms):
                log.debug("discarding invalid correction %s", atoms)
                continue
            corrections.append(
                Correction(
                    rollout_id=trace.rollout_id,
                    tick_index=int(entry["tick_index"]),
                    corrected_atoms=list(atoms),
                    reason=str(entry.get("reason", "")),
                    kind=(
                        ExampleKind.ESCALATE
                        if atoms == ["<ESCALATE>"]
                        else ExampleKind.CONTROL
                    ),
                )
            )
        return corrections

    def _valid(self, atoms: Sequence[str]) -> bool:
        """Only well-formed actions become training targets."""
        try:
            self.codec.decode(list(atoms))
            return True
        except Exception:
            return False

    def _ask(self, prompt: str, images: Sequence[bytes]) -> dict:
        import base64

        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64", "media_type": "image/png",
                    "data": base64.standard_b64encode(png).decode(),
                },
            }
            for png in images
        ]
        content.append({"type": "text", "text": prompt})

        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=4096,
            system=_CORRECTION_SYSTEM,
            messages=[{"role": "user", "content": content}],
            output_config={"format": {"type": "json_schema", "schema": _CORRECTION_SCHEMA}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("correction request was declined")
        return json.loads(next(b.text for b in response.content if b.type == "text"))


_CORRECTION_SYSTEM = """\
You review recordings of a GUI control policy that failed a task, and say what \
it should have done instead at specific moments.

You are given the task, the sequence of actions the policy took, why the run \
ended, and screenshots from the ticks in question.

For each tick you are asked about, either give the correct action as a list of \
action tokens, or say nothing about that tick if the action taken was fine.

Two rules that matter more than the rest:
- If the policy could not have recovered on its own at that point -- it was \
stuck, blocked, or the screen was not what the task expected -- the correct \
action is ["<ESCALATE>"]. Handing control back is a correct action, not a \
failure.
- Only correct a tick if you can see the right action in the screenshot. A \
guessed coordinate is worse than no correction, because it becomes training \
data.
"""

_CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "corrections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tick_index": {"type": "integer"},
                    "atoms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Action tokens, e.g. ['<MOVE_CLICK>','<ROW_4>','<COL_9>','<OFFSET_3>'].",
                    },
                    "reason": {"type": "string"},
                },
                "required": ["tick_index", "atoms", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["corrections"],
    "additionalProperties": False,
}


def _correction_prompt(record: RolloutRecord, suspects: Sequence[int]) -> str:
    trace = record.result.trace
    lines = [
        f"Task: {record.task.instruction}",
        f"Success criteria: {record.task.success_criteria or '(none given)'}",
        f"Outcome: {record.result.status.value} - {record.result.stop_reason}",
        "",
        "Actions taken:",
    ]
    for tick in trace.ticks:
        marker = " <-- review this tick" if tick.index in suspects else ""
        blocked = f" [blocked: {tick.blocked_reason}]" if tick.blocked_reason else ""
        stalled = "" if tick.frame_changed else " [screen did not change]"
        lines.append(f"  {tick.index:3d} {tick.action_summary}{blocked}{stalled}{marker}")
    lines.append("")
    lines.append(f"Review these ticks: {list(suspects)}")
    return "\n".join(lines)


def _load_images(paths: Sequence[str], indices: Sequence[int], limit: int = 6) -> list[bytes]:
    out: list[bytes] = []
    for index in list(indices)[:limit]:
        if 0 <= index < len(paths):
            try:
                out.append(Path(paths[index]).read_bytes())
            except OSError:
                continue
    return out


# --------------------------------------------------------------------------
# Example construction
# --------------------------------------------------------------------------


def escalation_labels(record: RolloutRecord) -> list[tuple[int, bool]]:
    """``(tick_index, should_have_escalated)`` for every tick in a rollout.

    Both directions, deliberately.  Only training the positive case produces a
    policy that escalates constantly -- which is the failure plan section 8
    warns "defeats the purpose of the whole system".
    """
    trace = record.result.trace
    if trace is None:
        return []
    status = record.result.status
    n = len(trace.ticks)

    if status is SubtaskStatus.DONE:
        # It finished. No tick should have handed back.
        return [(t.index, False) for t in trace.ticks]
    if status is SubtaskStatus.ESCALATED and trace.stop_reason == "policy escalated":
        # It escalated on its own; the last tick was right to, earlier ones
        # were right not to.
        return [(t.index, t.index == n - 1) for t in trace.ticks]
    # Timed out, got stuck, or was escalated by a stop condition: it should
    # have handed back before it wasted the remaining budget.
    cutoff = max(0, n - _ESCALATE_WINDOW)
    return [(t.index, t.index >= cutoff) for t in trace.ticks]


def build_dagger_examples(
    records: Sequence[RolloutRecord],
    corrections: Sequence[Correction],
    include_escalation: bool = True,
    include_successes: bool = True,
) -> list[TrainingExample]:
    """Turn rollouts and corrections into training examples.

    Successful rollouts are included by default: they are on-policy states with
    known-good actions, which is exactly the distribution DAgger is trying to
    cover, and they supply the negative half of the escalation signal.
    """
    by_rollout: dict[str, dict[int, Correction]] = {}
    for correction in corrections:
        by_rollout.setdefault(correction.rollout_id, {})[correction.tick_index] = correction

    examples: list[TrainingExample] = []
    for record in records:
        trace = record.result.trace
        if trace is None:
            continue
        if not include_successes and not record.failed and not record.is_near_miss:
            continue

        fixes = by_rollout.get(trace.rollout_id, {})
        escalate = dict(escalation_labels(record)) if include_escalation else {}
        history: list[list[str]] = []

        for tick in trace.ticks:
            correction = fixes.get(tick.index)
            if correction is not None:
                target = correction.corrected_atoms
                kind = correction.kind
                corrected_from = tick.atoms
            elif escalate.get(tick.index) and record.failed:
                # No explicit correction, but the rollout was past saving here.
                target = ["<ESCALATE>"]
                kind = ExampleKind.ESCALATE
                corrected_from = tick.atoms
            elif record.failed:
                # An uncorrected tick of a failed rollout is not evidence of
                # anything: training on it would clone the behaviour that
                # failed.
                history.append(tick.atoms)
                continue
            else:
                target = tick.atoms
                kind = ExampleKind.CONTROL
                corrected_from = None

            examples.append(
                TrainingExample(
                    segment_id=trace.rollout_id,
                    frame_index=tick.index,
                    target=list(target),
                    kind=kind,
                    instruction=trace.instruction,
                    form_data=dict(trace.form_data),
                    history=[list(h) for h in history[-8:]],
                    app_context=tick.app_context,
                    rollout_id=trace.rollout_id,
                    corrected_from=list(corrected_from) if corrected_from else None,
                )
            )
            history.append(tick.atoms)

    log.info(
        "built %d stage-3 examples from %d rollouts (%d corrections)",
        len(examples), len(records), len(corrections),
    )
    return examples


def run_stage3(
    dagger_path: str | Path,
    base_dataset_path: str | Path | None = None,
    policy_config: PolicyConfig | None = None,
    train_config: TrainConfig | None = None,
    pool_root: str | None = "data_pool",
    raw_root: str | None = None,
    init_from: str | None = None,
    device: str | None = None,
):
    """Retrain on corrections, mixed with the stage-2 data.

    The stage-2 data is mixed back in on purpose: training only on corrections
    is a fast route to forgetting everything that already worked.
    """
    from ..model.policy import GuiPolicy

    config = train_config or TrainConfig(stage="stage3", output_dir="runs/stage3")
    set_seed(config.seed)

    examples = list(load_examples(dagger_path))
    if base_dataset_path:
        base = list(load_examples(base_dataset_path))
        log.info("mixing %d correction examples with %d stage-2 examples", len(examples), len(base))
        examples = examples + base
    if not examples:
        raise SystemExit("no stage-3 examples; collect rollouts first")

    for example in examples:
        if example.kind is ExampleKind.ESCALATE:
            example.loss_weight = config.escalate_loss_weight
        elif example.kind is ExampleKind.FIELD_FILL:
            example.loss_weight = config.copy_loss_weight

    start = init_from or config.resume_from
    if not start:
        raise SystemExit(
            "stage 3 corrects an existing policy; pass init_from=<stage2 checkpoint>"
        )
    policy = GuiPolicy.load(start)

    trainer = Trainer(
        policy, config,
        build_dataloader(examples, policy, config, pool_root, raw_root),
        device=resolve_device(device),
    )
    return trainer.train()
