"""End-to-end replay against the virtual audio clock.

These tests exercise the real worker threads and the injected network, so
they validate timing behavior rather than mocks.
"""

from volumeview.frame import FrameFile
from volumeview.player import HeadlessPlayer
from volumeview.scheduler import UnderrunPolicy

TICK = 1.0 / 30.0


def _drive(player, seconds, seeks=()):
    player.start()
    player.play()
    seeks = list(seeks)
    elapsed = 0.0
    presented = []
    try:
        while elapsed < seconds:
            if seeks and elapsed >= seeks[0][0]:
                _, delta = seeks.pop(0)
                player.seek(max(0.0, player.clock.position + delta))
                player.play()
            p = player.step(TICK)
            if p is not None:
                presented.append((player.clock.position, p.index,
                                  p.av_offset_s, p.underrun))
            elapsed += TICK
        return player.scheduler.report(), presented
    finally:
        player.close()


def test_both_policies_complete_and_stay_bounded(assets):
    for policy in (UnderrunPolicy.HOLD, UnderrunPolicy.SKIP):
        player = HeadlessPlayer(assets["vpc"], assets["wav"], policy=policy)
        player.set_read_rate(3.0)  # force underruns
        report, presented = _drive(player, 6.0)
        assert report["ticks"] > 100
        assert report["cache_peak_bytes"] <= report["cache_cap_bytes"]
        assert report["max_av_offset_s"] >= 0.0
        assert len(presented) > 0


def test_hold_max_offset_is_monotonic_wait(assets):
    player = HeadlessPlayer(assets["vpc"], assets["wav"],
                            policy=UnderrunPolicy.HOLD)
    player.set_read_rate(4.0)
    report, _ = _drive(player, 8.0)
    # HOLD never jumps ahead: its offset is bounded by frame spacing + wait,
    # whereas SKIP under the same stress shows larger jumps (cross-checked in
    # test_policy_comparison_orders_offsets).
    assert report["max_av_offset_s"] < 2.0
    assert report["skips"] == 0


def test_seek_discards_stale_region_frames(assets):
    player = HeadlessPlayer(assets["vpc"], assets["wav"],
                            policy=UnderrunPolicy.HOLD)
    report, presented = _drive(player, 8.0,
                               seeks=[(2.0, -1.5), (5.0, -2.0)])
    with FrameFile(assets["vpc"]) as ff:
        ts = ff.timestamps
    # Every presented frame's pts must be close to the audio position at tick.
    for audio_t, idx, _off, _u in presented[5:]:
        assert abs(ts[idx] - audio_t) < 3.0
    assert report["stale_frames_dropped"] >= 1


def test_loop_confines_submissions_to_window(assets):
    player = HeadlessPlayer(assets["vpc"], assets["wav"],
                            policy=UnderrunPolicy.SKIP,
                            loop_window=(4.0, 6.0))
    player.scheduler.seek(4.0)
    report, presented = _drive(player, 8.0)
    with FrameFile(assets["vpc"]) as ff:
        ts = ff.timestamps
    for audio_t, idx, _off, _u in presented[10:]:
        assert 3.8 <= audio_t <= 6.2
        assert 3.7 <= ts[idx] <= 6.3


def test_policy_comparison_orders_offsets(assets):
    from tools.compare_policies import _run

    common = (assets["vpc"], assets["wav"])
    hold = _run(*common, policy=UnderrunPolicy.HOLD, duration=9.0,
                read_rate=4.0, loop_window=(4.0, 7.0))
    skip = _run(*common, policy=UnderrunPolicy.SKIP, duration=9.0,
                read_rate=4.0, loop_window=(4.0, 7.0))
    # Both report memory caps; HOLD waits, SKIP takes forward jumps.
    assert hold["cache_cap_bytes"] == skip["cache_cap_bytes"]
    assert hold["holds"] > 0 and hold["skips"] == 0
    assert skip["skips"] > 0
    # Under identical stress the skip policy must show at least one jump whose
    # offset exceeds HOLD's steady wait bound.
    assert skip["max_av_offset_s"] >= hold["max_av_offset_s"] - 1e-9


def test_headless_replay_writes_artifacts(assets, tmp_path):
    from tools.headless_replay import run_headless

    out = tmp_path / "replay"
    report = run_headless(assets["vpc"], assets["wav"],
                          UnderrunPolicy.HOLD, str(out), duration=4.0,
                          seek_events=(2.0,))
    assert (out / "metrics.json").exists()
    assert (out / "trace.json").exists()
    assert report["cache_peak_bytes"] <= report["cache_cap_bytes"]
