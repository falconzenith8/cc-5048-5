"""Tween engine: segmentation, correspondence, one-sided policy, reconcile."""

import numpy as np

from volumeview.edits import TimelineView
from volumeview.frame import Frame
from volumeview.surfaces import compute_correspondence, segment_frame
from volumeview.tween import OneSided

DROP = (10, 11)


def _timeline_without(frames, drop):
    return {i: f for i, f in frames.items() if i not in drop}


def test_segmentation_is_deterministic(turn):
    frame = turn["frames"][8]
    a = segment_frame(frame, 8)
    b = segment_frame(frame, 8)
    assert [p.key for p in a] == [p.key for p in b]
    for pa, pb in zip(a, b):
        np.testing.assert_array_equal(pa.indices, pb.indices)
        np.testing.assert_allclose(pa.centroid, pb.centroid)


def test_capture_counts_and_visible_surface_change(turn):
    """The arm occludes and re-reveals the torso: point counts and the
    visible surface both change across the sweep."""
    from volumeview.synth import FRONT_CAM, make_camera, visible_indices

    frames = turn["frames"]
    counts = [frames[f].npoints for f in sorted(frames)]
    assert max(counts) - min(counts) > 0.05 * min(counts)
    # The set of visible surface points turns over substantially between
    # the pre-occlusion and mid-sweep frames (same local surface indexing).
    cam = make_camera(FRONT_CAM)
    vis = {}
    for f in (6, 11):
        world, _ = turn["scene"].dense_world(f)
        vis[f] = set(visible_indices(world, cam).tolist())
    jaccard = len(vis[6] & vis[11]) / len(vis[6] | vis[11])
    assert jaccard < 0.85


def test_bidirectional_blend_where_correspondence_suffices(turn):
    tl = TimelineView(_timeline_without(turn["frames"], DROP), turn["scene"].fps)
    tl.add_interp_region(9, 12)
    tf = tl.frame_at(10)
    prov = tf.provenance
    assert prov["source"] == "interp"
    # The torso/head bulk is tracked well enough for a bidirectional blend.
    assert prov["bilateral_points"] > 2000
    # Point count interpolates between the key frames' captures.
    ka, kb = turn["frames"][9].npoints, turn["frames"][12].npoints
    assert min(ka, kb) * 0.8 < tf.frame.npoints < max(ka, kb) * 1.3


def test_onesided_keep_vs_gap(turn):
    base = _timeline_without(turn["frames"], DROP)
    keep = TimelineView(dict(base), turn["scene"].fps)
    keep.add_interp_region(9, 12, onesided=OneSided.KEEP)
    gap = TimelineView(dict(base), turn["scene"].fps)
    gap.add_interp_region(9, 12, onesided=OneSided.GAP)

    keep_prov = keep.frame_at(10).provenance
    gap_prov = gap.frame_at(10).provenance
    # Same geometry was analysed: what KEEP carries as single-sided
    # evidence, GAP reports as missing (缺测).
    assert keep_prov["onesided_points"] > 0
    assert keep_prov["missing_points"] == 0
    assert gap_prov["onesided_points"] == 0
    assert gap_prov["missing_points"] == keep_prov["onesided_points"]
    assert gap.frame_at(10).frame.npoints < keep.frame_at(10).frame.npoints


def test_correspondence_groups_for_veto(swap):
    tl = TimelineView(_timeline_without(swap["frames"], (11, 12)),
                      swap["scene"].fps)
    rid = tl.add_interp_region(10, 13)
    table = tl.regions[rid].table
    groups = table.groups()
    # One group wrongly connects the two actors, one tracks the pillar.
    assert len(groups) == 2
    cross = [e for g in groups.values() for e in g
             if e.confidence < 0.9]
    assert len(cross) == 1
    assert cross[0].state == "proposed"


def test_late_frame_replaces_and_migrates_marker(turn):
    tl = TimelineView(_timeline_without(turn["frames"], DROP), turn["scene"].fps)
    tl.add_interp_region(9, 12)
    blend = next(b for b in tl.frame_at(10).provenance["blends"]
                 if b["mode"] == "bilateral")
    sid = tl.registry.surface_of(blend["key"])
    mid = tl.place_marker(10 / turn["scene"].fps, surface_id=sid)

    reports = tl.on_late_frame(10, turn["frames"][10])
    (report,) = reports
    # The reconstructor emitted the replacement table first ...
    assert report.replacements
    assert any(r.trustworthy for r in report.replacements)
    # ... every stand-in patch was superseded by the original ...
    assert report.kept_interp == []
    assert tl.frame_at(10).provenance["recovered"] is True
    # ... and the marker moved to the still-trustworthy new surface.
    marker = tl.markers[mid]
    assert marker.state == "active"
    assert any(h["event"] == "migrated_to_original"
               for h in marker.history)


def test_untrustworthy_late_frame_suspends_association(turn):
    tl = TimelineView(_timeline_without(turn["frames"], DROP), turn["scene"].fps)
    tl.add_interp_region(9, 12)
    blend = next(b for b in tl.frame_at(11).provenance["blends"]
                 if b["mode"] == "bilateral")
    sid = tl.registry.surface_of(blend["key"])
    mid = tl.place_marker(11 / turn["scene"].fps, surface_id=sid)

    rng = np.random.default_rng(0)
    garbage = Frame(pts=turn["frames"][11].pts, origin=(5.0, 5.0, 5.0),
                    points=rng.normal(size=(900, 3)).astype(np.float32),
                    colors=rng.integers(0, 255, (900, 3), np.uint8))
    (report,) = tl.on_late_frame(11, garbage)
    # Nothing matched: stand-ins are kept, the association is suspended.
    assert report.replacements == []
    assert len(report.kept_interp) > 0
    assert tl.markers[mid].state == "suspended"
    assert any(h.get("reason") == "association_suspended"
               for h in tl.markers[mid].history)


def test_locked_revision_conflicts_with_late_original(turn):
    tl = TimelineView(_timeline_without(turn["frames"], DROP), turn["scene"].fps)
    rid = tl.add_interp_region(9, 12)
    blend = next(b for b in tl.frame_at(10).provenance["blends"]
                 if b["mode"] == "bilateral")
    tl.lock_patch(rid, blend["key"])

    (report,) = tl.on_late_frame(10, turn["frames"][10])
    # The locked revision collides with the arrival: conflict, and the
    # locked patch is kept instead of auto-replaced.
    assert any(c["interp"] == blend["key"] and c["reason"] == "locked_revision"
               for c in report.conflicts)
    assert blend["key"] in report.kept_interp
    prov = tl.frame_at(10).provenance
    assert prov["source"] == "mixed"
    assert prov["conflicts"]
