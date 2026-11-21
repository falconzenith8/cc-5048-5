"""Generate every shipped asset from a fixed seed.

Outputs:
  assets/scene.vpc (+ .idx)   volumetric frames, variable count/order
  assets/drums.wav            100 bpm reference drum track
  assets/beatmap.json         beat times and screen-crossing annotation
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

FPS = 12.0


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
    with open(os.path.join(recordings_dir, "asset_manifest.json"),
              "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    out = generate("assets", "recordings")
    print(json.dumps(out, indent=2))
