"""Audio-clock-driven frame scheduler with two underrun policies.

Each tick the scheduler maps the authoritative audio position onto a frame
index and asks the bounded cache for it.  If the frame has not arrived yet the
policy decides what to do:

* ``hold``   - keep presenting the previous pose until the target catches up;
* ``skip``   - jump forward to the closest frame already available.

Only frames inside the current playback region are ever handed to the
renderer.  Seeking or looping bumps the region epoch, which invalidates every
in-flight frame, so stale frames from an earlier region can never be shown.

Metrics recorded for comparison:

* ``max_av_offset_s``  - largest absolute gap between audio time and the pts
  of the frame actually shown;
* ``cache_peak_bytes`` - high-water mark of decoded memory (the hard cap is
  enforced by BoundedFrameCache);
* underrun / skip / eviction counters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .cache import BoundedFrameCache
from .frame import Frame
from .streaming import ReadDecodeService


class UnderrunPolicy(str, Enum):
    HOLD = "hold"
    SKIP = "skip"


@dataclass
class SchedulerMetrics:
    ticks: int = 0
    underruns: int = 0
    skips: int = 0
    holds: int = 0
    late_frames_presented: int = 0
    max_av_offset_s: float = 0.0
    sum_av_offset_s: float = 0.0
    last_presented_index: int | None = None
    last_presented_pts: float | None = None
    region_epoch: int = 0

    @property
    def mean_av_offset_s(self) -> float:
        return self.sum_av_offset_s / self.ticks if self.ticks else 0.0


@dataclass
class PresentedFrame:
    frame: Frame
    index: int
    av_offset_s: float
    underrun: bool


class Scheduler:
    def __init__(self, timestamps, cache: BoundedFrameCache,
                 service: ReadDecodeService,
                 policy: UnderrunPolicy = UnderrunPolicy.HOLD):
        self.ts = timestamps
        self.cache = cache
        self.service = service
        self.policy = policy
        self.metrics = SchedulerMetrics()
        self._held: PresentedFrame | None = None

    # -- region management -------------------------------------------------
    def seek(self, position: float) -> None:
        idx = int(max(0, min(len(self.ts) - 1,
                             (self.ts <= position).sum() - 1)))
        self.metrics.region_epoch = self.service.reset_for_region(idx)
        # Drop the held pose: a HOLD after seek must not flash a frame from
        # the superseded region before the new region has decoded anything.
        self._held = None
        self.metrics.last_presented_index = None
        self.metrics.last_presented_pts = None

    # -- per-tick selection ------------------------------------------------
    def target_index(self, position: float) -> int:
        idx = int((self.ts <= position).sum() - 1)
        return max(0, min(idx, len(self.ts) - 1))

    def tick(self, position: float) -> PresentedFrame | None:
        target = self.target_index(position)
        self.service.set_play_index(target)
        m = self.metrics
        m.ticks += 1
        frame = self.cache.get(target)

        if frame is None:
            m.underruns += 1
            if self.policy is UnderrunPolicy.SKIP:
                return self._skip_policy(target, position)
            return self._hold_policy(target, position)

        # Frame available on time.  Note late heavy frames that finally land.
        if self._held is not None and target > self._held.index:
            m.late_frames_presented += 1
        presented = self._present(target, frame, position, underrun=False)
        self._held = presented
        return presented

    def _hold_policy(self, target: int, position: float) -> PresentedFrame | None:
        m = self.metrics
        if self._held is None:
            # Cold start with nothing decoded yet: render nothing this tick.
            m.holds += 1
            return None
        m.holds += 1
        held = self._held
        offset = abs(position - held.frame.pts)
        self._record_offset(offset)
        return PresentedFrame(held.frame, held.index, offset, underrun=True)

    def _skip_policy(self, target: int, position: float) -> PresentedFrame | None:
        m = self.metrics
        # Closest available frame at or after the target (never go backwards).
        available = sorted(i for (i, f) in self.cache.items() if i >= target)
        if not available:
            if self._held is not None:
                return self._hold_policy(target, position)
            m.skips += 1
            return None
        jump = available[0]
        if jump != target:
            m.skips += 1
        frame = self.cache.get(jump)
        presented = self._present(jump, frame, position, underrun=True)
        self._held = presented
        return presented

    def _present(self, index: int, frame: Frame, position: float,
                 underrun: bool) -> PresentedFrame:
        offset = abs(position - frame.pts)
        self._record_offset(offset)
        m = self.metrics
        m.last_presented_index = index
        m.last_presented_pts = frame.pts
        return PresentedFrame(frame, index, offset, underrun)

    def _record_offset(self, offset: float) -> None:
        m = self.metrics
        m.max_av_offset_s = max(m.max_av_offset_s, offset)
        m.sum_av_offset_s += offset

    # -- reporting ---------------------------------------------------------
    def memory_cap_bytes(self) -> int:
        return self.cache.max_bytes

    def report(self) -> dict:
        m = self.metrics
        return {
            "policy": self.policy.value,
            "ticks": m.ticks,
            "underruns": m.underruns,
            "holds": m.holds,
            "skips": m.skips,
            "late_frames_presented": m.late_frames_presented,
            "max_av_offset_s": round(m.max_av_offset_s, 4),
            "mean_av_offset_s": round(m.mean_av_offset_s, 4),
            "cache_peak_bytes": self.cache.stats.peak_bytes,
            "cache_cap_bytes": self.memory_cap_bytes(),
            "cache_evictions": self.cache.stats.evictions,
            "stale_frames_dropped": self.cache.stats.dropped_stale,
        }
