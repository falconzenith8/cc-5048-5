"""Generate every shipped asset from a fixed seed.

Outputs:
  assets/scene.vpc (+ .idx)   volumetric frames, variable count/order
  assets/drums.wav            100 bpm reference drum track
  assets/beatmap.json         beat times and screen-crossing annotation
  assets/turn.vpc (+ turn_gt.npz)   turn-around capture with depth truth
  assets/swap.vpc (+ swap_gt.npz)   two actors swapping positions
  assets/swap_project.json    open tween-layer sample (region + marker)
  recordings/replay_plan.json the injected out-of-order delivery plan
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from volumeview.audio import (BPM, DURATION, SAMPLE_RATE, build_beatmap,
                              synthesize_drums, write_beatmap, write_wav)
from volumeview.frame import FrameWriter
from volumeview.scene import make_frame
from volumeview.streaming import OutOfOrderNetwork
from volumeview.synth import SwapScene, TurnScene

FPS = 12.0


def _write_scene(path: str, frames: dict[int, object]) -> None:
    with FrameWriter(path) as writer:
        for i in sorted(frames):
            writer.write(frames[i])


def generate(assets_dir: str, recordings_dir: str) -> dict:
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(recordings_dir, exist_ok=True)

    beats = build_beatmap(DURATION, BPM)
    beat_a, beat_b = beats[8], beats[12]
    write_beatmap(os.path.join(assets_dir, "beatmap.json"), beats, BPM,
                  crossing=(beat_a, beat_b))
    samples = synthesize_drums(beats, DURATION, SAMPLE_RATE)
    wav_path = os.path.join(assets_dir, "drums.wav")
    write_wav(wav_path, samples, SAMPLE_RATE)

    n_frames = int(DURATION * FPS)
    vpc_path = os.path.join(assets_dir, "scene.vpc")
    point_counts = []
    with FrameWriter(vpc_path) as writer:
        for i in range(n_frames):
            pts = round(i / FPS, 6)
            frame = make_frame(pts, i, beat_a, beat_b)
            point_counts.append(frame.npoints)
            writer.write(frame)

    network = OutOfOrderNetwork()
    plan = [{
        "frame": i,
        "pts": round(i / FPS, 6),
        "injected_delay_s": round(network.delay_for(i), 4),
        "npoints": point_counts[i],
    } for i in range(n_frames)]
    # Explicit narrative check: heavy frame 43 must be delivered after 44.
    plan_path = os.path.join(recordings_dir, "replay_plan.json")
    with open(plan_path, "w", encoding="utf-8") as fh:
        json.dump({
            "fps": FPS,
            "bpm": BPM,
            "heavy_frames": list(network.heavy_frames),
            "frames": plan,
            "ordering_guarantee": (
                "frame 43 ready_at > frame 44 ready_at despite being requested "
                "first"),
        }, fh, indent=2)

    summary = {
        "vpc": vpc_path,
        "wav": wav_path,
        "frames": n_frames,
        "min_points": int(np.min(point_counts)),
        "max_points": int(np.max(point_counts)),
        "crossing_beats": [beat_a, beat_b],
        "replay_plan": plan_path,
    }

    # Tween-layer scenes: turn-around with depth truth, and the two-actor
    # position swap with its open (unrepaired) project sample.
    turn = TurnScene()
    turn_vpc = os.path.join(assets_dir, "turn.vpc")
    _write_scene(turn_vpc, turn.frames())
    turn_gt = os.path.join(assets_dir, "turn_gt.npz")
    turn.save_ground_truth(turn_gt)
    summary["turn"] = {"vpc": turn_vpc, "gt": turn_gt,
                       "frames": turn.n,
                       "contact": turn.contact_event()}

    swap = SwapScene()
    swap_vpc = os.path.join(assets_dir, "swap.vpc")
    _write_scene(swap_vpc, swap.frames())
    swap_gt = os.path.join(assets_dir, "swap_gt.npz")
    swap.save_ground_truth(swap_gt)
    from tools.swap_sample import build_open_sample
    swap_project = os.path.join(assets_dir, "swap_project.json")
    build_open_sample(swap_vpc, swap_project)
    summary["swap"] = {"vpc": swap_vpc, "gt": swap_gt,
                       "frames": swap.n, "project": swap_project}

    with open(os.path.join(recordings_dir, "asset_manifest.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    out = generate("assets", "recordings")
    print(json.dumps(out, indent=2))
