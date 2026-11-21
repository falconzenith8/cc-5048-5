"""Read / decode pipeline with injected network disorder.

Frames are read ahead of the play head, but each compressed frame is held in
a delay queue before decoding to emulate a lossy network.  Heavy frames are
deliberately held longer than the small frame right after them, so frame 43
(the 4.2k-point capture) always arrives *after* frame 44 even though it was
requested first.  A user-controlled read-rate throttle multiplies every delay.

Everything runs on a worker thread; the hand-off to the bounded cache is the
only cross-thread state and is guarded by the cache's epoch checks.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .cache import BoundedFrameCache
from .frame import FrameFile, decode_frame


@dataclass(order=True)
class _Pending:
    ready_at: float
    frame_index: int = field(compare=False)
    epoch: int = field(compare=False)
    blob: bytes = field(compare=False, default=b"", repr=False)


class OutOfOrderNetwork:
    """Deterministic per-frame latency model (seconds)."""

    def __init__(self, base_delay: float = 0.18, jitter: float = 0.10,
                 heavy_frames: tuple[int, ...] = (42, 43),
                 heavy_extra: float = 0.55, seed: int = 99):
        self.base_delay = base_delay
        self.jitter = jitter
        self.heavy_frames = set(heavy_frames)
        self.heavy_extra = heavy_extra
        self._rng = np.random.default_rng(seed)

    def delay_for(self, frame_index: int) -> float:
        d = self.base_delay + float(self._rng.random()) * self.jitter
        if frame_index in self.heavy_frames:
            # Heavy frame 43 (~4.2k pts) lands well after small frame 44.
            d += self.heavy_extra
        return max(0.0, d)


class ReadDecodeService:
    def __init__(self, frame_file: FrameFile, cache: BoundedFrameCache,
                 network: OutOfOrderNetwork, readahead_frames: int = 24,
                 clock=time.monotonic, sleep=time.sleep):
        self.ff = frame_file
        self.cache = cache
        self.network = network
        self.readahead = readahead_frames
        self._clock = clock
        self._sleep = sleep
        self._pending: list[_Pending] = []
        self._requested: set[int] = set()
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.epoch = 0
        self.play_index = 0
        self.read_rate = 1.0          # UI slider multiplier on latency
        self.paused = False
        self.bytes_read = 0
        self.bytes_decoded = 0
        self.frames_decoded = 0

    # -- epoch / transport -------------------------------------------------
    def reset_for_region(self, play_index: int) -> int:
        """Seek / loop wrap: invalidate every in-flight and cached frame."""
        with self._lock:
            self.epoch += 1
            self.play_index = play_index
            self._pending.clear()
            self._requested.clear()
            self.cache.reset()
            epoch = self.epoch
        self._wake.set()
        return epoch

    def set_play_index(self, play_index: int) -> None:
        with self._lock:
            if play_index != self.play_index:
                self.play_index = play_index
        self._wake.set()

    def set_read_rate(self, rate: float) -> None:
        self.read_rate = max(0.05, float(rate))

    # -- worker ------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="read-decode", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        n = len(self.ff)
        while not self._stop.is_set():
            now = self._clock()
            with self._lock:
                epoch = self.epoch
                head = self.play_index
                rate = self.read_rate
                # Decode anything whose injected delay has elapsed.
                due = []
                while self._pending and self._pending[0].ready_at <= now:
                    due.append(heapq.heappop(self._pending))
                # Issue reads up to the readahead window.
                horizon = min(n, head + self.readahead + 1)
                to_request = [i for i in range(head, horizon)
                              if i not in self._requested]
                for i in to_request:
                    self._requested.add(i)

            for item in due:
                if item.epoch != epoch:
                    continue
                frame, _ = decode_frame(item.blob, 0)
                self.bytes_decoded += frame.decoded_bytes
                self.frames_decoded += 1
                self.cache.insert(item.frame_index, frame, epoch, item.epoch,
                                   head)

            for i in to_request:
                try:
                    _entry, blob = self.ff.read_compressed(i)
                except OSError:
                    continue
                self.bytes_read += len(blob)
                ready = now + self.network.delay_for(i) * rate
                with self._lock:
                    heapq.heappush(self._pending,
                                   _Pending(ready, i, epoch, blob))

            # Wake shortly before the next pending frame becomes due.
            timeout = 0.05
            with self._lock:
                if self._pending:
                    timeout = max(0.005,
                                  min(0.05, self._pending[0].ready_at
                                      - self._clock()))
            self._wake.wait(timeout)
            self._wake.clear()

    def inflight(self) -> int:
        with self._lock:
            return len(self._pending)
