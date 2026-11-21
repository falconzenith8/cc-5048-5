"""Open sample: two look-alike actors swap positions; the matcher links
the wrong pair.

Workflow demonstrated here (and verified by the tests):

1. build the tween region across the crossing - the greedy matcher
   connects actor A's start to actor B's end (their colours are nearly
   identical), so the interpolated frames grow a brief spurious surface
   between the two lanes, visible from the side view;
2. a contact marker placed on the interpolated surface follows the
   polluted identity: exported contact audio lands away from actor A;
3. the editor approves the static pillar group, vetoes the wrong
   correspondence group and recomputes - approved regions and the marker
   history survive, the spurious surface disappears;
4. the editor pins the correct A->A pair, the marker is resumed onto A's
   surface, and a final export puts the contact sound back on actor A.
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
from volumeview.synth import SIDE_CAM, SWAP_FPS, make_camera

DROP = (11, 12)
REGION = (10, 13)
GAP_Z = 0.14          # |z| below this is empty space between the lanes


def _load_frames(vpc_path: str) -> dict[int, object]:
    with FrameFile(vpc_path) as ff:
        return {i: ff.read_frame(i) for i in range(len(ff))}


def _actor_of(centroid: np.ndarray, centroids: np.ndarray,
              frame_index: int) -> int:
    d = np.linalg.norm(centroids[frame_index] - centroid, axis=1)
    return int(np.argmin(d))


def _cross_entry(tl: TimelineView, region_id: int,
                 centroids: np.ndarray) -> tuple[str, str]:
    """The table entry that links two different actors."""
    region = tl.regions[region_id]
    patches_a = {p.key: p for p in tl._patches_of(region.frame_a)}
    patches_b = {p.key: p for p in tl._patches_of(region.frame_b)}
    for e in region.table.active():
        pa, pb = patches_a[e.a], patches_b[e.b]
        if _actor_of(pa.centroid, centroids, region.frame_a) != \
                _actor_of(pb.centroid, centroids, region.frame_b):
            return e.a, e.b
    raise RuntimeError("no cross-actor entry found")


def _spurious_count(tl: TimelineView, frames: tuple[int, ...]) -> int:
    total = 0
    for idx in frames:
        tf = tl.frame_at(idx)
        world = tf.frame.world_points
        total += int(np.count_nonzero(np.abs(world[:, 2]) < GAP_Z))
    return total


def _render_side(tl: TimelineView, frame_index: int, path: str) -> None:
    renderer = SoftwareRenderer(320, 240)
    renderer.show_screen = False
    renderer.camera = make_camera(SIDE_CAM)
    renderer.save_png(path, tl.frame_at(frame_index).frame)


def build_open_sample(vpc_path: str, project_out: str) -> str:
    """The sample as shipped: region built, marker placed, nothing fixed."""
    frames = _load_frames(vpc_path)
    tl = TimelineView({i: f for i, f in frames.items() if i not in DROP},
                      SWAP_FPS)
    region = tl.add_interp_region(*REGION)
    mid = _place_marker_on_cross_blend(tl, region)
    tl.save(project_out)
    return mid


def _place_marker_on_cross_blend(tl: TimelineView, region_id: int) -> str:
    """Contact marker on the interpolated surface at the second gap frame."""
    frame_index = DROP[-1]
    tf = tl.frame_at(frame_index)
    # The interpolated blend patch (not a held one-sided cloud).
    for blend in tf.provenance["blends"]:
        if blend["mode"] == "bilateral":
            sid = tl.registry.surface_of(blend["key"])
            if sid is not None:
                return tl.place_marker(frame_index / tl.fps, surface_id=sid)
    raise RuntimeError("no interpolated blend to mark")


def run_swap(vpc_path: str, gt_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    gt = np.load(gt_path)
    centroids = gt["centroids"]
    frames = _load_frames(vpc_path)
    tl = TimelineView({i: f for i, f in frames.items() if i not in DROP},
                      SWAP_FPS)
    region = tl.add_interp_region(*REGION)
    cross = _cross_entry(tl, region, centroids)
    mid = _place_marker_on_cross_blend(tl, region)

    metrics: dict = {"region": list(REGION), "drop": list(DROP),
                     "cross_entry": list(cross), "marker": mid}

    # 1. naive interpolation: spurious surface between the lanes.
    metrics["spurious_points_before"] = _spurious_count(tl, DROP)
    _render_side(tl, DROP[0], os.path.join(out_dir, "side_before.png"))

    # 2. export with the polluted identity: contact audio leaves actor A.
    bad_events = os.path.join(out_dir, "events_before.json")
    bad_wav = os.path.join(out_dir, "contacts_before.wav")
    bad = tl.export(os.path.join(out_dir, "before.vpc"), bad_events, bad_wav)
    metrics["export_before"] = {
        "events": bad["events"], "skipped": bad["skipped"]}

    # 3. editor repair: approve the pillar group, veto the cross group.
    region_obj = tl.regions[region]
    pillar_group = None
    for gid, entries in region_obj.table.groups().items():
        if any((e.a, e.b) == cross for e in entries):
            cross_group = gid
        else:
            pillar_group = gid
    approved = tl.approve_group(region, pillar_group)
    tl.veto_group(region, cross_group)
    tl.recompute_region(region)
    metrics["approved_pairs"] = approved
    metrics["spurious_points_after_veto"] = _spurious_count(tl, DROP)
    _render_side(tl, DROP[0], os.path.join(out_dir, "side_after_veto.png"))
    metrics["marker_state_after_veto"] = tl.markers[mid].state

    # 4. pin the correct actor link, resume the marker onto actor A.
    patches_b = tl._patches_of(REGION[1])
    a_end = min(patches_b, key=lambda p: float(np.linalg.norm(
        p.centroid - centroids[REGION[1]][0])))
    patches_a = tl._patches_of(REGION[0])
    a_start = min(patches_a, key=lambda p: float(np.linalg.norm(
        p.centroid - centroids[REGION[0]][0])))
    tl.approve_pair(region, a_start.key, a_end.key)
    tl.recompute_region(region)
    a_surface = tl.registry.surface_of(a_end.key)
    tl.resume_marker(mid, a_surface)

    good_events = os.path.join(out_dir, "events_after.json")
    good_wav = os.path.join(out_dir, "contacts_after.wav")
    good = tl.export(os.path.join(out_dir, "after.vpc"), good_events,
                     good_wav)
    metrics["export_after"] = {
        "events": good["events"], "skipped": good["skipped"]}
    metrics["spurious_points_final"] = _spurious_count(tl, DROP)
    _render_side(tl, DROP[0], os.path.join(out_dir, "side_final.png"))

    tl.save(os.path.join(out_dir, "project.json"))
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--vpc", default="assets/swap.vpc")
    ap.add_argument("--gt", default="assets/swap_gt.npz")
    ap.add_argument("--out", default="recordings/swap_sample")
    ns = ap.parse_args()
    run_swap(ns.vpc, ns.gt, ns.out)
