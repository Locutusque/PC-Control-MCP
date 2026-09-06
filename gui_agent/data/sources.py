"""Locating captured segments and pulling frames out of them.

The capture pool stores one video plus one JSONL record file per segment.
Datasets address a frame as ``(segment_id, frame_index)``; this module resolves
that to pixels, decoding only the frames actually asked for rather than whole
segments.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..capture.schema import FrameRecord, SegmentMeta, read_frames, read_jsonl

log = logging.getLogger(__name__)

__all__ = ["SegmentSource", "FrameLoader", "iter_segments"]


@dataclass
class SegmentSource:
    """A promoted segment plus the paths needed to read it back."""

    meta: SegmentMeta
    root: Path

    @property
    def segment_id(self) -> str:
        return self.meta.segment_id

    @property
    def video(self) -> Path | None:
        return self.root / self.meta.video_path if self.meta.video_path else None

    @property
    def records(self) -> Path | None:
        return self.root / self.meta.records_path if self.meta.records_path else None

    def read_records(self) -> Iterator[FrameRecord]:
        path = self.records
        if path is None or not path.exists():
            log.warning("segment %s has no readable records", self.segment_id)
            return iter(())
        return read_frames(path)

    @property
    def instruction(self) -> str:
        """Best available goal label: in-the-moment beats hindsight."""
        return self.meta.user_label or self.meta.hindsight_instruction or ""


def iter_segments(
    pool_root: str | Path,
    raw_root: str | Path | None = None,
    require_promoted: bool = True,
) -> Iterator[SegmentSource]:
    """Yield every segment in the training pool.

    ``require_promoted`` is on by default: a segment that has not cleared the
    redaction sweep must never reach a dataset builder, and defaulting the
    other way would make that a one-flag mistake.
    """
    pool_root = Path(pool_root)
    seg_dir = pool_root / "segments"
    if not seg_dir.exists():
        log.warning("no promoted segments under %s", seg_dir)
        return

    for path in sorted(seg_dir.glob("*.json")):
        for row in read_jsonl(path):
            meta = SegmentMeta.from_dict(row)
            if require_promoted and not meta.is_promotable:
                log.debug("skipping %s (redaction_status=%s)", meta.segment_id, meta.redaction_status)
                continue
            root = Path(raw_root) / meta.session_id if raw_root else pool_root
            yield SegmentSource(meta, root)


class FrameLoader:
    """Decodes frames on demand, with a small cache of recent segments.

    Training shuffles examples, which would otherwise mean re-decoding a
    segment once per example drawn from it.  Group examples by segment in the
    sampler if throughput matters; the cache only softens the worst case.
    """

    def __init__(self, width: int, height: int, cache_segments: int = 4) -> None:
        self.width, self.height = width, height
        self._decode = lru_cache(maxsize=cache_segments)(self._decode_uncached)

    def _decode_uncached(self, video_path: str):
        from ..capture.screen import read_segment_frames

        return read_segment_frames(video_path, self.width, self.height)

    def load(self, source: SegmentSource, frame_index: int):
        """One frame as an ``(H, W, 3)`` uint8 array."""
        video = source.video
        if video is None or not video.exists():
            raise FileNotFoundError(f"segment {source.segment_id} has no video at {video}")
        frames = self._decode(str(video))
        if not 0 <= frame_index < len(frames):
            raise IndexError(
                f"frame {frame_index} out of range for {source.segment_id} "
                f"({len(frames)} frames decoded)"
            )
        return frames[frame_index]

    def clear(self) -> None:
        self._decode.cache_clear()
