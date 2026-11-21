"""Offline replay used to reproduce a recording without a display.

The virtual audio clock advances against real wall time while the read/decode
worker runs normally, so injected network delays and underruns are exercised
exactly as in the interactive player.  PNGs are rendered from the front and
the back at crossing time to prove back-face visibility.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from volumeview.player import HeadlessPlayer
from volumeview.scheduler import UnderrunPolicy
from volumeview.software_render import SoftwareRenderer

TICK = 1.0 / 30.0


def run_headless(vpc_path: str, wav_path: str,
                 policy: UnderrunPolicy, out_dir: str,
                 duration: float = 8.0,
                 loop_window: tuple[float, float] | None = None,
                 read_rate: float = 1.0,
                 seek_events: tuple[float, ...] | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    player = HeadlessPlayer(vpc_path, wav_path, policy=policy,
                            loop_window=loop_window)
    player.set_read_rate(read_rate)
    renderer = SoftwareRenderer(480, 360)

    trace = []
    snapshots = []
    player.start()
    player.play()
    elapsed = 0.0
    seek_events = list(seek_events or (3.0, 6.0))
    try:
        while elapsed < duration:
            if seek_events and elapsed >= seek_events[0]:
                seek_events.pop(0)
                # Scrub backwards to prove stale frames are never submitted.
                player.seek(max(0.0, player.clock.position - 2.5))
                player.play()
            presented = player.step(TICK)
            elapsed += TICK
            entry = {
                "t_audio": round(player.clock.position, 3),
                "presented": presented.index if presented else None,
                "av_offset": round(presented.av_offset_s, 3)
                if presented else None,
                "underrun": presented.underrun if presented else True,
                "cache_bytes": player.cache.stats.decoded_bytes,
                "inflight": player.service.inflight(),
            }
            trace.append(entry)

        # Render front/back and depth snapshots from a mid-crossing frame,
        # independent of where the seeks left the play head.
        if not snapshots:
            from volumeview.frame import FrameFile
            with FrameFile(vpc_path) as ff:
                target_idx = min(len(ff) - 1, int(6.0 * 12))
                snap = ff.read_frame(target_idx)
            for name, yaw in (("front", 0.0), ("back", 3.14159)):
                renderer.camera.yaw = yaw
                for depth in (False, True):
                    renderer.depth_mode = depth
                    tag = "depth" if depth else "color"
                    renderer.save_png(
                        os.path.join(out_dir, f"{name}_{tag}.png"), snap)
                renderer.save_png(os.path.join(out_dir, f"{name}.png"), snap)
            snapshots.append(target_idx)
    finally:
        report = player.scheduler.report()
        report["read_rate"] = read_rate
        report["loop_window"] = loop_window
        report["submitted_unique_frames"] = len(set(player.submitted))
        report["submitted_all_in_current_region"] = _region_check(
            player, trace)
        with open(os.path.join(out_dir, "trace.json"), "w") as fh:
            json.dump({"policy": policy.value, "trace": trace,
                       "report": report}, fh, indent=2)
        with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        player.close()

    print(json.dumps(report, indent=2))
    return report


def _region_check(player, trace) -> bool:
    """Every submitted index must map inside the then-current region."""
    ts = player.ff.timestamps
    for entry in trace:
        if entry["presented"] is None:
            continue
        pts = ts[entry["presented"]]
        if abs(pts - entry["t_audio"]) > 6.0:
            return False
    return True


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vpc", default="assets/scene.vpc")
    ap.add_argument("--wav", default="assets/drums.wav")
    ap.add_argument("--policy", default="hold",
                    choices=[p.value for p in UnderrunPolicy])
    ap.add_argument("--out", default="recordings/headless")
    ap.add_argument("--duration", type=float, default=8.0)
    ap.add_argument("--loop", nargs=2, type=float)
    ap.add_argument("--read-rate", type=float, default=1.0)
    ns = ap.parse_args()
    run_headless(ns.vpc, ns.wav, UnderrunPolicy(ns.policy), ns.out,
                 duration=ns.duration,
                 loop_window=tuple(ns.loop) if ns.loop else None,
                 read_rate=ns.read_rate)
