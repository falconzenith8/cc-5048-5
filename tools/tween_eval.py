"""Evaluate the tween layer on the turn scene with depth ground truth.

Drops two frames from the capture, fills the gap with the interpolation
layer, then measures - from the original capture view and from a side
view:

* silhouette error: IoU between the rendered interpolation mask and the
  ground-truth depth mask;
* missing surface (漏面): fraction of back-projected ground-truth depth
  samples that have no interpolated point nearby;
* audio sync error: the drum-contact time detected on the timeline frames
  vs. the ground-truth contact time.

Finally saves the project, reloads it and scrubs backwards to confirm
every frame keeps its provenance (original vs. interpolated stand-in).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from volumeview.edits import TimelineView
from volumeview.frame import FrameFile
from volumeview.software_render import SoftwareRenderer
from volumeview.synth import DEPTH_H, DEPTH_W, make_camera
from volumeview.tween import OneSided

DROP = (10, 11)
MISS_RADIUS = 0.10


def _load_frames(vpc_path: str) -> dict[int, object]:
    with FrameFile(vpc_path) as ff:
        return {i: ff.read_frame(i) for i in range(len(ff))}


def _backproject(depth: np.ndarray, camera, width: int,
                 height: int) -> np.ndarray:
    vp = camera.view_projection(width / height)
    inv = np.linalg.inv(vp)
    ys, xs = np.nonzero(np.isfinite(depth))
    z = depth[ys, xs]
    ndc = np.column_stack([
        (xs + 0.5) / width * 2.0 - 1.0,
        1.0 - (ys + 0.5) / height * 2.0,
        z, np.ones_like(z)]).astype(np.float64)
    world = (inv @ ndc.T).T
    world = world[:, :3] / world[:, 3:4]
    return world.astype(np.float32)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return inter / union if union else 1.0


def _missing_fraction(gt_world: np.ndarray, points: np.ndarray,
                      radius: float) -> float:
    if gt_world.shape[0] == 0:
        return 0.0
    if points.shape[0] == 0:
        return 1.0
    missing = 0
    for lo in range(0, gt_world.shape[0], 512):
        part = gt_world[lo:lo + 512]
        d = np.sqrt(((part[:, None, :] - points[None, :, :]) ** 2)
                    .sum(axis=2)).min(axis=1)
        missing += int(np.count_nonzero(d > radius))
    return missing / gt_world.shape[0]


def _contact_time(frames_world: dict[int, np.ndarray], pos, threshold,
                  fps: float) -> float | None:
    times, weights = [], []
    for idx in sorted(frames_world):
        pts = frames_world[idx]
        if pts.shape[0] == 0:
            continue
        d = float(np.sqrt(((pts - pos) ** 2).sum(axis=1)).min())
        if d < threshold:
            times.append(idx / fps)
            weights.append(threshold - d)
    if not times:
        return None
    return float(np.average(times, weights=weights))


def run_eval(vpc_path: str, gt_path: str, out_dir: str,
             drop: tuple[int, ...] = DROP,
             onesided: OneSided = OneSided.KEEP) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    gt = np.load(gt_path)
    fps = float(gt["fps"])
    contact = json.loads(str(gt["contact"]))
    cams = {"front": json.loads(str(gt["front_cam"])),
            "side": json.loads(str(gt["side_cam"]))}
    depth_gt = {"front": gt["depth_front"], "side": gt["depth_side"]}

    frames = _load_frames(vpc_path)
    timeline_frames = {i: f for i, f in frames.items() if i not in drop}
    tl = TimelineView(timeline_frames, fps)
    region = tl.add_interp_region(min(drop) - 1, max(drop) + 1,
                                  onesided=onesided)

    renderer = SoftwareRenderer(DEPTH_W, DEPTH_H)
    renderer.show_screen = False
    metrics: dict = {"drop": list(drop), "region": region,
                     "frames": {}}
    for k in drop:
        tf = tl.frame_at(k)
        world = tf.frame.world_points
        entry = {"provenance": tf.provenance["source"],
                 "npoints": int(tf.frame.npoints),
                 "bilateral_points": tf.provenance["bilateral_points"],
                 "onesided_points": tf.provenance["onesided_points"],
                 "missing_points": tf.provenance["missing_points"]}
        for view in ("front", "side"):
            renderer.camera = make_camera(cams[view])
            img = renderer.render(tf.frame)
            mask = img.sum(axis=2) > 0
            gt_mask = np.isfinite(depth_gt[view][k])
            iou = _mask_iou(mask, gt_mask)
            gt_world = _backproject(depth_gt[view][k],
                                    make_camera(cams[view]),
                                    DEPTH_W, DEPTH_H)
            miss = _missing_fraction(gt_world[::2], world, MISS_RADIUS)
            entry[view] = {"silhouette_iou": round(iou, 4),
                           "missing_surface": round(miss, 4)}
            if k == drop[0]:
                from PIL import Image
                Image.fromarray(img).save(
                    os.path.join(out_dir, f"interp_{view}_f{k}.png"))
                Image.fromarray((gt_mask * 255).astype(np.uint8)).save(
                    os.path.join(out_dir, f"gt_{view}_f{k}.png"))
        metrics["frames"][k] = entry

    # Audio sync: contact time detected on the presented timeline frames
    # (originals + interpolated stand-ins) vs. ground truth.
    presented = {i: tl.frame_at(i).frame.world_points
                 for i in sorted(frames) if tl.frame_at(i) is not None}
    t_detected = _contact_time(presented, np.asarray(contact["position"]),
                               contact["threshold"] + 0.03, fps)
    metrics["audio_sync"] = {
        "gt_contact_time": contact["time"],
        "detected_contact_time": t_detected,
        "sync_error_s": (round(abs(t_detected - contact["time"]), 4)
                         if t_detected is not None else None),
    }

    # Save, reload, scrub backwards: provenance must survive.
    project = os.path.join(out_dir, "project.json")
    tl.save(project)
    reloaded = TimelineView.load(project, dict(timeline_frames))
    provenance = {}
    for idx in sorted(frames, reverse=True):   # reverse scrub
        tf = reloaded.frame_at(idx)
        provenance[str(idx)] = tf.provenance["source"] if tf else "gap"
    metrics["reverse_scrub_provenance"] = provenance
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(provenance, fh, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vpc", default="assets/turn.vpc")
    ap.add_argument("--gt", default="assets/turn_gt.npz")
    ap.add_argument("--out", default="recordings/tween_eval")
    ns = ap.parse_args()
    run_eval(ns.vpc, ns.gt, ns.out)
