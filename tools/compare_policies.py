"""Run both underrun policies against the same recording and compare them.

A deliberately slow read rate guarantees underruns around the heavy frames.
Each policy gets an identical setup (same container, same network seed, same
seek/loop script); only the underrun handling differs.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volumeview.player import HeadlessPlayer
from volumeview.scheduler import UnderrunPolicy

TICK = 1.0 / 30.0


def _run(vpc_path: str, wav_path: str, policy: UnderrunPolicy,
         duration: float, read_rate: float,
         loop_window: tuple[float, float]) -> dict:
    player = HeadlessPlayer(vpc_path, wav_path, policy=policy,
                            loop_window=loop_window)
    player.set_read_rate(read_rate)
    player.start()
    player.play()
    seek_at = {2.5, 6.0}
    elapsed = 0.0
    try:
        while elapsed < duration:
            if any(abs(elapsed - s) < TICK for s in seek_at):
                player.seek(max(0.0, player.clock.position - 2.0))
                player.play()
            player.step(TICK)
            elapsed += TICK
        return player.scheduler.report()
    finally:
        player.close()


def compare(vpc_path: str, wav_path: str, duration: float = 10.0,
            read_rate: float = 3.5,
            loop_window: tuple[float, float] = (4.0, 7.0),
            as_json: bool = False) -> dict:
    common = dict(duration=duration, read_rate=read_rate,
                  loop_window=list(loop_window))
    hold = _run(vpc_path, wav_path, UnderrunPolicy.HOLD, duration,
                read_rate, loop_window)
    skip = _run(vpc_path, wav_path, UnderrunPolicy.SKIP, duration,
                read_rate, loop_window)
    result = {"setup": common, "hold": hold, "skip": skip}
    out_path = os.path.join("recordings", "policy_comparison.json")
    os.makedirs("recordings", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    if as_json:
        print(json.dumps(result, indent=2))
    else:
        _print_table(result, out_path)
    return result


def _print_table(result: dict, out_path: str) -> None:
    hold, skip = result["hold"], result["skip"]
    rows = [
        ("policy", hold["policy"], skip["policy"]),
        ("underrun ticks", hold["underruns"], skip["underruns"]),
        ("holds / skips", f"{hold['holds']} / {hold['skips']}",
         f"{skip['holds']} / {skip['skips']}"),
        ("max A/V offset (s)", hold["max_av_offset_s"],
         skip["max_av_offset_s"]),
        ("mean A/V offset (s)", hold["mean_av_offset_s"],
         skip["mean_av_offset_s"]),
        ("cache peak (MiB)",
         round(hold["cache_peak_bytes"] / 1048576, 2),
         round(skip["cache_peak_bytes"] / 1048576, 2)),
        ("cache cap (MiB)",
         hold["cache_cap_bytes"] // 1048576,
         skip["cache_cap_bytes"] // 1048576),
        ("stale frames dropped", hold["stale_frames_dropped"],
         skip["stale_frames_dropped"]),
    ]
    print(f"{'metric':24s} {'HOLD':>14s} {'SKIP':>14s}")
    print("-" * 56)
    for name, a, b in rows:
        print(f"{name:24s} {str(a):>14s} {str(b):>14s}")
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vpc", default="assets/scene.vpc")
    ap.add_argument("--wav", default="assets/drums.wav")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--read-rate", type=float, default=3.5)
    ap.add_argument("--json", action="store_true")
    ns = ap.parse_args()
    compare(ns.vpc, ns.wav, duration=ns.duration,
            read_rate=ns.read_rate, as_json=ns.json)
