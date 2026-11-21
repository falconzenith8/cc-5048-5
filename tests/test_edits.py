"""Timeline view: undoable tween layer, surface-bound markers, export."""

import json
import wave

import numpy as np

from volumeview.edits import TimelineView
from volumeview.tween import OneSided

DROP = (10, 11)
SWAP_DROP = (11, 12)


def _turn_timeline(turn):
    frames = {i: f for i, f in turn["frames"].items() if i not in DROP}
    tl = TimelineView(frames, turn["scene"].fps)
    tl.add_interp_region(9, 12)
    return tl


def _swap_timeline(swap):
    frames = {i: f for i, f in swap["frames"].items() if i not in SWAP_DROP}
    tl = TimelineView(frames, swap["scene"].fps)
    rid = tl.add_interp_region(10, 13)
    return tl, rid


def _cross_entry(tl, rid):
    table = tl.regions[rid].table
    return min(table.active(), key=lambda e: e.confidence)


def test_undo_redo_restores_layer_state(turn):
    frames = {i: f for i, f in turn["frames"].items() if i not in DROP}
    tl = TimelineView(frames, turn["scene"].fps)
    initial = tl._memento()

    rid = tl.add_interp_region(9, 12)
    tl.set_onesided(rid, OneSided.GAP)
    table = tl.regions[rid].table
    entry = table.active()[0]
    tl.delete_entry(rid, entry.a, entry.b)
    tl.lock_patch(rid, "r1f10s0")
    assert tl.undo_labels

    for _ in range(4):
        tl.undo()
    now = tl._memento()
    assert now["regions"] == initial["regions"]
    assert now["markers"] == initial["markers"]
    assert now["registry"] == initial["registry"]

    for _ in range(4):
        tl.redo()
    redone = tl._memento()
    assert redone["regions"][rid]["locks"] == ["r1f10s0"]
    assert redone["regions"][rid]["onesided"] == "gap"


def test_marker_references_surface_identity(turn):
    tl = _turn_timeline(turn)
    blend = next(b for b in tl.frame_at(10).provenance["blends"]
                 if b["mode"] == "bilateral")
    sid = tl.registry.surface_of(blend["key"])
    mid = tl.place_marker(10 / turn["scene"].fps, surface_id=sid)
    marker = tl.markers[mid]
    # The marker is bound to the surface id and a patch anchor, not to a
    # raw position.
    assert marker.surface_id == sid
    assert marker.anchor in tl.registry.members(sid)
    pos = tl.marker_position(mid)
    assert pos is not None and pos.shape == (3,)


def test_delete_entry_export_keeps_contact_off_other_actor(swap, tmp_path):
    tl, rid = _swap_timeline(swap)
    centroids = swap["gt"]["centroids"]
    cross = _cross_entry(tl, rid)

    # Contact marker on the interpolated surface at the second gap frame.
    blend = next(b for b in tl.frame_at(12).provenance["blends"]
                 if b["mode"] == "bilateral")
    sid = tl.registry.surface_of(blend["key"])
    mid = tl.place_marker(12 / swap["scene"].fps, surface_id=sid)

    out = tmp_path
    before = tl.export(str(out / "before.vpc"), str(out / "before.json"),
                       str(out / "before.wav"))
    # Hazard first: with the wrong link in place, the contact follows the
    # polluted identity and lands far from actor A (x ~ 0.73).
    assert len(before["events"]) == 1
    assert abs(before["events"][0]["position"][0]
               - float(centroids[12][0][0])) > 0.5

    # Delete the wrong row, export again: the marker's identity is broken,
    # so the contact is suspended - never re-attached to the other actor.
    tl.delete_entry(rid, cross.a, cross.b)
    after = tl.export(str(out / "after.vpc"), str(out / "after.json"),
                      str(out / "after.wav"))
    assert after["events"] == []
    assert any(s["id"] == mid for s in after["skipped"])
    # And the contact track is silent at the marker's time.
    with wave.open(str(out / "after.wav"), "rb") as wav:
        n = wav.getnframes()
        raw = wav.readframes(n)
    samples = np.frombuffer(raw, dtype="<i2").reshape(n, 2)
    t = int((12 / swap["scene"].fps) * 44100)
    window = samples[t:t + 4410]
    assert np.abs(window).max() == 0


def test_veto_group_recompute_preserves_approvals_and_markers(swap):
    tl, rid = _swap_timeline(swap)
    cross = _cross_entry(tl, rid)
    blend = next(b for b in tl.frame_at(12).provenance["blends"]
                 if b["mode"] == "bilateral")
    sid = tl.registry.surface_of(blend["key"])
    mid = tl.place_marker(12 / swap["scene"].fps, surface_id=sid)

    table = tl.regions[rid].table
    groups = table.groups()
    cross_group = next(g for g, es in groups.items()
                       if any((e.a, e.b) == (cross.a, cross.b) for e in es))
    pillar_group = next(g for g in groups if g != cross_group)
    tl.approve_group(rid, pillar_group)
    tl.veto_group(rid, cross_group)
    tl.recompute_region(rid)

    entries = tl.regions[rid].table.entries
    # The approved pillar pair is pinned as approved; the vetoed pair stays
    # vetoed; the historical marker is preserved (suspended, with history).
    assert any(e.state == "approved" and (e.a, e.b) != (cross.a, cross.b)
               for e in entries)
    assert any(e.state == "vetoed" and (e.a, e.b) == (cross.a, cross.b)
               for e in entries)
    marker = tl.markers[mid]
    assert marker.state == "suspended"
    assert any(h["event"] == "placed" for h in marker.history)

    # The spurious bridge between the lanes is gone from the side view.
    for k in SWAP_DROP:
        world = tl.frame_at(k).frame.world_points
        assert int(np.count_nonzero(np.abs(world[:, 2]) < 0.14)) == 0


def test_save_load_reverse_scrub_provenance(turn, tmp_path):
    tl = _turn_timeline(turn)
    # Frame 10 arrives late and is spliced; frame 11 stays interpolated.
    tl.on_late_frame(10, turn["frames"][10])
    blend = next(b for b in tl.frame_at(11).provenance["blends"]
                 if b["mode"] == "bilateral")
    sid = tl.registry.surface_of(blend["key"])
    tl.place_marker(11 / turn["scene"].fps, surface_id=sid)

    project = tmp_path / "project.json"
    tl.save(str(project))
    reloaded = TimelineView.load(str(project), dict(tl.frames))

    expected = {}
    for idx in sorted(turn["frames"]):
        if idx == 10:
            expected[idx] = "mixed"
        elif idx == 11:
            expected[idx] = "interp"
        else:
            expected[idx] = "original"
    for idx in sorted(turn["frames"], reverse=True):   # reverse scrub
        tf = reloaded.frame_at(idx)
        assert tf is not None
        assert tf.provenance["source"] == expected[idx], idx
    # Markers and their history survive the round trip.
    assert len(reloaded.markers) == 1
    loaded = next(iter(reloaded.markers.values()))
    assert loaded.history and loaded.state == "active"


def test_turn_eval_metrics(turn, tmp_path):
    """Silhouette, missing-surface and audio-sync error from both views."""
    from tools.tween_eval import run_eval
    from volumeview.frame import FrameWriter

    vpc = tmp_path / "turn.vpc"
    with FrameWriter(str(vpc)) as writer:
        for i in sorted(turn["frames"]):
            writer.write(turn["frames"][i])
    gt = tmp_path / "turn_gt.npz"
    turn["scene"].save_ground_truth(str(gt))

    metrics = run_eval(str(vpc), str(gt), str(tmp_path / "eval"))
    for frame_metrics in metrics["frames"].values():
        for view in ("front", "side"):
            assert frame_metrics[view]["silhouette_iou"] > 0.7
            assert frame_metrics[view]["missing_surface"] < 0.2
    assert metrics["audio_sync"]["sync_error_s"] < 1.0 / turn["scene"].fps
    provenance = metrics["reverse_scrub_provenance"]
    assert provenance["10"] == "interp" and provenance["11"] == "interp"
    assert provenance["9"] == "original" and provenance["12"] == "original"


def test_swap_sample_tool(swap, tmp_path):
    from tools.swap_sample import run_swap
    from volumeview.frame import FrameWriter

    vpc = tmp_path / "swap.vpc"
    with FrameWriter(str(vpc)) as writer:
        for i in sorted(swap["frames"]):
            writer.write(swap["frames"][i])
    gt = tmp_path / "swap_gt.npz"
    swap["scene"].save_ground_truth(str(gt))

    metrics = run_swap(str(vpc), str(gt), str(tmp_path / "swap_out"))
    # The wrong connection briefly grows a spurious surface between the
    # lanes; vetoing the group removes it; the final export puts the
    # contact back on actor A (x ~ 0.73).
    assert metrics["spurious_points_before"] > 200
    assert metrics["spurious_points_after_veto"] == 0
    assert metrics["spurious_points_final"] == 0
    before = metrics["export_before"]["events"][0]
    after = metrics["export_after"]["events"][0]
    assert before["position"][0] < 0.0
    assert after["position"][0] > 0.5
