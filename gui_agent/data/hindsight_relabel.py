"""Hindsight instruction relabelling (plan section 4.3).

Stage 2 needs a goal per trajectory, but asking for one up front adds friction
to every recording session and biases what gets recorded.  Instead the
trajectory is reviewed *after* it completes and an LLM writes the instruction it
would satisfy.

Two modes:

* :meth:`HindsightRelabeler.relabel` -- one segment at a time, for interactive
  use and for spot-checking prompt changes.
* :meth:`HindsightRelabeler.relabel_batch` -- the Batches API, which is the
  right tool for a backlog of thousands of segments: it is asynchronous anyway
  and costs half as much.

The relabeller also reports a confidence, and
:func:`evaluate_against_human_labels` scores it against the sparse in-the-moment
labels from the capture hotkey.  Without that check there is no way to know
whether hindsight labels describe the trajectory or merely sound plausible, and
plan section 4.1.5 puts those human labels there precisely so this comparison
is possible.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..actions import ActionCodec
from ..capture.schema import SegmentMeta, write_jsonl
from .encode import TimedAction, TrajectoryEncoder
from .sources import SegmentSource, iter_segments

log = logging.getLogger(__name__)

__all__ = [
    "RelabelResult", "HindsightRelabeler", "AnthropicRelabelClient",
    "evaluate_against_human_labels",
]

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You label recordings of a person using a computer.

You are given one trajectory: the sequence of GUI actions the person performed, \
the applications involved, and a few screenshots sampled across the recording. \
Write the single instruction that, if handed to an assistant controlling the \
same computer, would have produced this trajectory.

Rules:
- Write it as an instruction to perform, not a description of what happened. \
"Search the docs for the refund policy", not "The user searched the docs".
- Be specific about the target of the task, and only about what the trajectory \
actually shows. Do not invent motivation or context you cannot see.
- One sentence. No preamble.
- Set confidence below 0.5 when the trajectory is ambiguous, aimless, or shows \
several unrelated activities. A low-confidence label is far more useful than a \
confident guess: low-confidence segments are dropped from instruction \
fine-tuning rather than training the model on a goal that was never there.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "instruction": {
            "type": "string",
            "description": "The instruction this trajectory would satisfy, one sentence.",
        },
        "confidence": {
            "type": "number",
            "description": "0.0-1.0. Below 0.5 means the trajectory has no clear single goal.",
        },
        "is_multi_task": {
            "type": "boolean",
            "description": "True if the recording covers several unrelated tasks.",
        },
    },
    "required": ["instruction", "confidence", "is_multi_task"],
    "additionalProperties": False,
}


@dataclass
class RelabelResult:
    segment_id: str
    instruction: str
    confidence: float
    is_multi_task: bool = False
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and bool(self.instruction) and not self.is_multi_task


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def render_trajectory(
    actions: Sequence[TimedAction],
    codec: ActionCodec,
    max_actions: int = 120,
) -> str:
    """A compact textual rendering of the trajectory for the prompt."""
    lines: list[str] = []
    for ta in actions[:max_actions]:
        stamp = f"{ta.timestamp - actions[0].timestamp:6.1f}s"
        app = f" [{ta.app_context}]" if ta.app_context else ""
        lines.append(f"{stamp}{app} {ta.action.summary()}")
    if len(actions) > max_actions:
        lines.append(f"... {len(actions) - max_actions} further actions omitted")
    return "\n".join(lines)


def sample_frame_indices(actions: Sequence[TimedAction], n: int = 4) -> list[int]:
    """Evenly spaced frames, always including the first and last."""
    if not actions:
        return []
    indices = sorted({ta.frame_index for ta in actions})
    if len(indices) <= n:
        return indices
    step = (len(indices) - 1) / (n - 1)
    return [indices[min(int(round(i * step)), len(indices) - 1)] for i in range(n)]


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class AnthropicRelabelClient:
    """Anthropic-backed relabelling, using structured outputs for the schema."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024) -> None:
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "the anthropic package is required for hindsight relabelling "
                "(pip install anthropic), or pass your own client callable"
            ) from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def build_content(self, text: str, images: Sequence[bytes] = ()) -> list[dict]:
        content: list[dict] = []
        for png in images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(png).decode(),
                    },
                }
            )
        content.append({"type": "text", "text": text})
        return content

    def request_params(self, content: list[dict]) -> dict:
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        }

    def __call__(self, text: str, images: Sequence[bytes] = ()) -> dict:
        params = self.request_params(self.build_content(text, images))
        try:
            response = self.client.messages.create(**params)
        except self._anthropic.RateLimitError as exc:
            raise RuntimeError(f"rate limited: {exc}") from exc
        except self._anthropic.APIStatusError as exc:
            raise RuntimeError(f"relabel request failed ({exc.status_code}): {exc.message}") from exc
        if response.stop_reason == "refusal":
            raise RuntimeError("relabel request was declined by the safety classifier")
        payload = next(b.text for b in response.content if b.type == "text")
        return json.loads(payload)


# --------------------------------------------------------------------------
# Relabeller
# --------------------------------------------------------------------------


class HindsightRelabeler:
    """Assigns an instruction to each captured trajectory."""

    def __init__(
        self,
        client: Callable[..., dict] | None = None,
        codec: ActionCodec | None = None,
        encoder: TrajectoryEncoder | None = None,
        n_frames: int = 4,
        frame_loader=None,
    ) -> None:
        self.client = client
        self.codec = codec or ActionCodec()
        self.encoder = encoder or TrajectoryEncoder(self.codec)
        self.n_frames = n_frames
        self.frame_loader = frame_loader

    def _client(self) -> Callable[..., dict]:
        if self.client is None:
            self.client = AnthropicRelabelClient()
        return self.client

    def build_prompt(self, source: SegmentSource) -> tuple[str, list[bytes], list[TimedAction]]:
        actions = self.encoder.encode_segment(source.read_records())
        meta = source.meta
        header = [
            f"Application: {meta.app_context or 'unknown'}",
            f"Duration: {meta.duration_s:.1f}s over {meta.n_frames} frames",
            f"Recording ended because: {meta.cut_reason or 'unknown'}",
        ]
        text = (
            "\n".join(header)
            + "\n\nActions:\n"
            + (render_trajectory(actions, self.codec) or "(no actions recorded)")
        )
        images = self._load_frames(source, actions)
        return text, images, actions

    def _load_frames(self, source: SegmentSource, actions: Sequence[TimedAction]) -> list[bytes]:
        if self.frame_loader is None or not actions:
            return []
        out: list[bytes] = []
        for index in sample_frame_indices(actions, self.n_frames):
            try:
                out.append(_to_png(self.frame_loader.load(source, index)))
            except Exception as exc:
                log.debug("could not load frame %d of %s: %s", index, source.segment_id, exc)
        return out

    def relabel(self, source: SegmentSource) -> RelabelResult:
        text, images, actions = self.build_prompt(source)
        if not actions:
            return RelabelResult(source.segment_id, "", 0.0, error="no actions in segment")
        try:
            data = self._client()(text, images)
        except Exception as exc:
            log.warning("relabel failed for %s: %s", source.segment_id, exc)
            return RelabelResult(source.segment_id, "", 0.0, error=str(exc))
        return RelabelResult(
            segment_id=source.segment_id,
            instruction=str(data.get("instruction", "")).strip(),
            confidence=float(data.get("confidence", 0.0)),
            is_multi_task=bool(data.get("is_multi_task", False)),
        )

    def relabel_batch(
        self, sources: Sequence[SegmentSource], poll_interval_s: float = 30.0
    ) -> list[RelabelResult]:
        """Relabel a backlog through the Batches API.

        Relabelling is not latency-sensitive and runs over the whole archive at
        once, which is exactly what the batch endpoint is for -- half price, and
        no client-side rate-limit handling.
        """
        client = self._client()
        if not isinstance(client, AnthropicRelabelClient):
            log.info("client does not support batching; relabelling sequentially")
            return [self.relabel(s) for s in sources]

        import time

        import anthropic  # type: ignore
        from anthropic.types.messages.batch_create_params import Request  # type: ignore

        requests, prepared = [], {}
        for source in sources:
            text, images, actions = self.build_prompt(source)
            if not actions:
                prepared[source.segment_id] = RelabelResult(
                    source.segment_id, "", 0.0, error="no actions in segment"
                )
                continue
            requests.append(
                Request(
                    custom_id=source.segment_id,
                    params=anthropic.types.MessageCreateParamsNonStreaming(
                        **client.request_params(client.build_content(text, images))
                    ),
                )
            )
        if not requests:
            return list(prepared.values())

        batch = client.client.messages.batches.create(requests=requests)
        log.info("submitted batch %s with %d segments", batch.id, len(requests))
        while True:
            batch = client.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            time.sleep(poll_interval_s)

        # Results come back in arbitrary order; key by custom_id, never position.
        for entry in client.client.messages.batches.results(batch.id):
            sid = entry.custom_id
            if entry.result.type != "succeeded":
                prepared[sid] = RelabelResult(sid, "", 0.0, error=entry.result.type)
                continue
            try:
                payload = next(b.text for b in entry.result.message.content if b.type == "text")
                data = json.loads(payload)
                prepared[sid] = RelabelResult(
                    sid, str(data.get("instruction", "")).strip(),
                    float(data.get("confidence", 0.0)),
                    bool(data.get("is_multi_task", False)),
                )
            except Exception as exc:
                prepared[sid] = RelabelResult(sid, "", 0.0, error=str(exc))

        return [prepared.get(s.segment_id, RelabelResult(s.segment_id, "", 0.0, error="missing"))
                for s in sources]

    def relabel_pool(
        self,
        pool_root: str | Path,
        raw_root: str | Path | None = None,
        overwrite: bool = False,
        use_batch: bool = True,
    ) -> list[RelabelResult]:
        """Relabel every segment in the pool and write the labels back."""
        pool_root = Path(pool_root)
        sources = [
            s for s in iter_segments(pool_root, raw_root)
            if overwrite or not s.meta.hindsight_instruction
        ]
        if not sources:
            log.info("no segments need relabelling")
            return []

        results = (
            self.relabel_batch(sources) if use_batch else [self.relabel(s) for s in sources]
        )
        for source, result in zip(sources, results):
            if result.error:
                continue
            source.meta.hindsight_instruction = result.instruction
            source.meta.hindsight_confidence = result.confidence
            write_jsonl(
                pool_root / "segments" / f"{source.segment_id}.json", [source.meta]
            )
        usable = sum(1 for r in results if r.usable)
        log.info("relabelled %d segments (%d usable)", len(results), usable)
        return results


def _to_png(array) -> bytes:
    import io

    from PIL import Image  # type: ignore

    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------
# Calibration against human labels
# --------------------------------------------------------------------------


def evaluate_against_human_labels(
    results: Iterable[RelabelResult],
    segments: Iterable[SegmentMeta],
    judge: Callable[[str, str], bool] | None = None,
) -> dict:
    """Score hindsight labels against the sparse in-the-moment ones.

    Plan section 4.1.5 suggests labelling ~10% of sessions by hand so the
    hindsight labels have something to be checked against.  This is that check:
    the agreement rate, split by whether the relabeller was confident, tells you
    whether its confidence means anything -- and if it does not, the
    ``min_hindsight_confidence`` filter in stage 2 is doing nothing.
    """
    by_id = {s.segment_id: s for s in segments}
    judge = judge or _token_overlap_judge
    total = agree = confident = confident_agree = 0

    for result in results:
        meta = by_id.get(result.segment_id)
        if meta is None or not meta.user_label or result.error:
            continue
        total += 1
        ok = judge(meta.user_label, result.instruction)
        agree += bool(ok)
        if result.confidence >= 0.5:
            confident += 1
            confident_agree += bool(ok)

    return {
        "n_compared": total,
        "agreement": round(agree / total, 4) if total else None,
        "n_confident": confident,
        "confident_agreement": round(confident_agree / confident, 4) if confident else None,
        # If these two are equal the confidence score carries no information and
        # filtering on it is only shrinking the dataset.
        "confidence_is_informative": (
            None if not (total and confident)
            else (confident_agree / confident) > (agree / total) + 0.05
        ),
    }


def _token_overlap_judge(human: str, hindsight: str) -> bool:
    """Cheap default: content-word overlap.

    A real evaluation should use an LLM judge; this exists so the calibration
    check can run offline without an API key.
    """
    stop = {"the", "a", "an", "to", "in", "on", "for", "of", "and", "my", "then", "with"}
    ha = {w for w in _words(human) if w not in stop}
    hb = {w for w in _words(hindsight) if w not in stop}
    if not ha or not hb:
        return False
    return len(ha & hb) / len(ha) >= 0.5


def _words(text: str) -> list[str]:
    return [w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if w]
