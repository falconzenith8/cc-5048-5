"""Bounded cache, streaming epochs and underrun policy semantics."""

import time

import numpy as np

from volumeview.cache import BoundedFrameCache
from volumeview.frame import Frame, FrameFile
from volumeview.scheduler import Scheduler, UnderrunPolicy
from volumeview.streaming import OutOfOrderNetwork, ReadDecodeService


def _fake_frame(pts: float, n: int = 50) -> Frame:
    rng = np.random.default_rng(int(pts * 1000) & 0xFFFF)
    return Frame(pts=pts, origin=(0.0, 0.0, 0.0),
                 points=rng.normal(size=(n, 3)).astype(np.float32),
                 colors=rng.integers(0, 255, (n, 3), np.uint8))


def test_cache_hard_memory_bound():
    cache = BoundedFrameCache(max_bytes=20_000, max_frames=100)
    for i in range(20):
        cache.insert(i, _fake_frame(i * 0.1, n=80), epoch=0,
                     frame_epoch=0, play_index=0)
    assert cache.stats.decoded_bytes <= 20_000
    assert cache.stats.evictions > 0
    assert cache.stats.peak_bytes <= 20_000


def test_cache_drops_stale_epoch_frames():
    cache = BoundedFrameCache(max_bytes=10 ** 9)
    accepted = cache.insert(5, _fake_frame(0.5), epoch=1, frame_epoch=0,
                            play_index=0)
    assert accepted is False
    assert cache.stats.dropped_stale == 1
    assert cache.get(5) is None


def test_hold_policy_repeats_previous_pose():
    cache = BoundedFrameCache(max_bytes=10 ** 9)
    timestamps = np.arange(10, dtype=np.float64) * 0.5
    svc = _manual_service(timestamps, cache)
    sched = Scheduler(timestamps, cache, svc, UnderrunPolicy.HOLD)

    cache.insert(0, _fake_frame(0.0), epoch=0, frame_epoch=0, play_index=0)
    first = sched.tick(0.0)
    assert first.index == 0
    # Frame 1 missing: hold must return the same frame again, flagged.
    held = sched.tick(0.55)
    assert held.underrun is True
    assert held.index == 0
    assert sched.metrics.underruns >= 1
    assert sched.metrics.max_av_offset_s >= 0.5


def test_skip_policy_jumps_to_next_available():
    cache = BoundedFrameCache(max_bytes=10 ** 9)
    timestamps = np.arange(10, dtype=np.float64) * 0.5
    svc = _manual_service(timestamps, cache)
    sched = Scheduler(timestamps, cache, svc, UnderrunPolicy.SKIP)

    cache.insert(0, _fake_frame(0.0), epoch=0, frame_epoch=0, play_index=0)
    cache.insert(2, _fake_frame(1.0), epoch=0, frame_epoch=0, play_index=0)
    sched.tick(0.0)
    jumped = sched.tick(0.55)  # target 1 absent, 2 is the nearest ahead
    assert jumped.index == 2
    assert jumped.underrun is True
    assert sched.metrics.skips >= 1


def test_seek_invalidates_region_and_caps_offset():
    cache = BoundedFrameCache(max_bytes=10 ** 9)
    timestamps = np.arange(20, dtype=np.float64) * 0.5
    svc = _manual_service(timestamps, cache)
    sched = Scheduler(timestamps, cache, svc, UnderrunPolicy.HOLD)
    cache.insert(0, _fake_frame(0.0), epoch=0, frame_epoch=0, play_index=0)
    sched.tick(0.0)
    assert sched._held is not None
    sched.seek(7.0)
    # Held pose from the old region must be cleared, so the first tick in the
    # new region with an empty cache renders nothing rather than a stale frame.
    assert sched._held is None
    assert sched.tick(7.0) is None
    # Old epoch frame must be rejected when it eventually "arrives".
    accepted = cache.insert(0, _fake_frame(0.0),
                            epoch=sched.metrics.region_epoch,
                            frame_epoch=0, play_index=14)
    assert accepted is False
    assert svc.epoch == sched.metrics.region_epoch


def test_repeated_seeks_keep_epochs_aligned():
    cache = BoundedFrameCache(max_bytes=10 ** 9)
    timestamps = np.arange(30, dtype=np.float64) * 0.5
    svc = _manual_service(timestamps, cache)
    sched = Scheduler(timestamps, cache, svc, UnderrunPolicy.HOLD)
    for pos in (2.0, 9.5, 0.5, 12.0):
        sched.seek(pos)
        assert sched.metrics.region_epoch == svc.epoch
        # A frame tagged with any earlier epoch is stale.
        stale_epoch = svc.epoch - 1
        accepted = cache.insert(0, _fake_frame(0.0), epoch=svc.epoch,
                                frame_epoch=stale_epoch, play_index=0)
        assert accepted is False
    assert svc.epoch == 4


class _ManualService(ReadDecodeService):
    def __init__(self, timestamps, cache):
        self.epoch = 0
        self.play_index = 0
        self.cache = cache

    def reset_for_region(self, play_index):
        self.epoch += 1
        self.play_index = play_index
        self.cache.reset()
        return self.epoch

    def set_play_index(self, idx):
        self.play_index = idx


def _manual_service(timestamps, cache):
    return _ManualService(timestamps, cache)


def test_worker_service_delivers_frames(assets):
    with FrameFile(assets["vpc"]) as ff:
        cache = BoundedFrameCache(max_bytes=10 ** 9)
        svc = ReadDecodeService(ff, cache, OutOfOrderNetwork(base_delay=0.02,
                                                             jitter=0.02,
                                                             heavy_extra=0.1),
                                readahead_frames=12)
        svc.start()
        try:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and not cache.has(6):
                time.sleep(0.02)
            assert cache.has(6)
        finally:
            svc.stop()
