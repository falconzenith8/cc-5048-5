"""Bounded decoded-frame cache.

Only decoded frames that may be presented soon are kept.  The cache is bounded
both by frame count and by decoded bytes; when the budget is exhausted frames
furthest from the play head are evicted, and an eviction underrun is reported.
Frames belonging to a superseded playback region (after a seek or a loop
wrap) are dropped on the floor instead of being presented.
"""

from __future__ import annotations

from dataclasses import dataclass

from .frame import Frame


@dataclass
class CacheStats:
    frames: int = 0
    decoded_bytes: int = 0
    peak_bytes: int = 0
    evictions: int = 0
    dropped_stale: int = 0


class BoundedFrameCache:
    def __init__(self, max_bytes: int = 64 * 1024 * 1024, max_frames: int = 48):
        self.max_bytes = max_bytes
        self.max_frames = max_frames
        self._frames: dict[int, Frame] = {}
        self.stats = CacheStats()

    def insert(self, frame_index: int, frame: Frame, epoch: int,
               frame_epoch: int, play_index: int) -> bool:
        """Return True if accepted, False if the frame is stale and dropped."""
        if frame_epoch != epoch:
            self.stats.dropped_stale += 1
            return False
        # Anything behind the play head can never be presented again.
        if frame_index < play_index:
            self.stats.dropped_stale += 1
            return False
        self._frames[frame_index] = frame
        self.stats.decoded_bytes += frame.decoded_bytes
        self._enforce_bounds(play_index)
        self.stats.frames = len(self._frames)
        self.stats.peak_bytes = max(self.stats.peak_bytes,
                                    self.stats.decoded_bytes)
        return True

    def get(self, frame_index: int) -> Frame | None:
        return self._frames.get(frame_index)

    def has(self, frame_index: int) -> bool:
        return frame_index in self._frames

    def earliest(self) -> int | None:
        return next(iter(self._frames)) if self._frames else None

    def _enforce_bounds(self, play_index: int) -> None:
        # First discard frames behind the play head.
        for idx in [i for i in self._frames if i < play_index]:
            self._evict(idx)
        # Then respect the hard bounds, evicting furthest from the head first.
        while (len(self._frames) > self.max_frames
               or self.stats.decoded_bytes > self.max_bytes):
            victim = max(self._frames)
            if victim == play_index and len(self._frames) == 1:
                break
            self._evict(victim)
            self.stats.evictions += 1

    def _evict(self, idx: int) -> None:
        frame = self._frames.pop(idx)
        self.stats.decoded_bytes -= frame.decoded_bytes

    def reset(self) -> None:
        self._frames.clear()
        self.stats = CacheStats()

    def items(self):
        return self._frames.items()
