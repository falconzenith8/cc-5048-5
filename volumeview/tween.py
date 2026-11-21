"""Interpolation engine for the tween (补帧) layer, and late-frame reconcile.

Given two key frames and a correspondence table, :func:`interpolate_frame`
synthesises the missing in-between capture.  Per matched patch pair the
engine first measures *geometric coverage* (which points of either side
have a counterpart within ``coverage_radius`` after centroid alignment):

* regions with sufficient mutual coverage are blended **bidirectionally** -
  both endpoint clouds are resampled along their principal axis and lerped;
* points visible on only one side are **single-sided evidence**: with
  ``OneSided.KEEP`` they are carried into the gap as-is, with
  ``OneSided.GAP`` they are omitted and reported as missing (缺测).

When a late original frame arrives, :func:`reconcile_late_frame` matches
the interpolated patches against the real capture and emits the
*replacement table*: which interp patches map to which original patches and
whether each mapping is trustworthy.  Two gates apply: a patch is replaced
whenever the geometry confirms the match (and the revision is not manually
locked), while the stricter *trustworthy* flag decides which identities are
linked for marker migration.  Locked revisions are kept and reported as
conflicts the editor must resolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .frame import Frame
from .surfaces import (CorrespondenceTable, MatchEntry, Patch, compute_correspondence,
                       interp_key)


class OneSided(str, Enum):
    KEEP = "keep"      # carry single-sided evidence into the gap
    GAP = "gap"        # show the region as missing (缺测)


@dataclass
class Blend:
    """Provenance of one slice of an interpolated frame."""

    key: str                  # interp patch key ("" for missing slices)
    mode: str                 # "bilateral" | "one_sided" | "missing"
    source: tuple[str, ...]   # (entry_a, entry_b) or (patch_key,)
    start: int                # slice into the frame's point arrays
    end: int

    @property
    def npoints(self) -> int:
        return self.end - self.start


@dataclass
class InterpResult:
    frame: Frame
    patches: list[Patch]      # interp patches (bilateral + kept one-sided)
    blends: list[Blend]
    bilateral_points: int
    onesided_points: int
    missing_points: int


def _pca_order(points: np.ndarray) -> np.ndarray:
    """Deterministic 1-D ordering of a cloud along its principal axis."""
    centered = points - points.mean(axis=0, keepdims=True)
    if centered.shape[0] < 3:
        return np.argsort(np.lexsort(centered.T[::-1]))
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    pivot = int(np.argmax(np.abs(axis)))
    if axis[pivot] < 0:
        axis = -axis
    proj = centered @ axis
    secondary = centered @ vt[1] if vt.shape[0] > 1 else np.zeros_like(proj)
    return np.argsort(np.lexsort((secondary, proj)))


def _resample(points: np.ndarray, colors: np.ndarray, order: np.ndarray,
              m: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.clip(np.round(np.linspace(0, len(order) - 1, m)).astype(int),
                  0, len(order) - 1)
    sel = order[idx]
    return points[sel], colors[sel]


def _min_distances(a: np.ndarray, b: np.ndarray,
                   chunk: int = 512) -> np.ndarray:
    out = np.empty(a.shape[0], dtype=np.float32)
    for lo in range(0, a.shape[0], chunk):
        part = a[lo:lo + chunk]
        d = np.sqrt(((part[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
        out[lo:lo + chunk] = d.min(axis=1)
    return out


def _blend_pair(pa: Patch, pb: Patch, world_a: np.ndarray,
                colors_a: np.ndarray, world_b: np.ndarray,
                colors_b: np.ndarray, t: float, coverage_radius: float,
                min_mutual: float, onesided: OneSided,
                slot_key: str) -> tuple[list[np.ndarray], list[np.ndarray],
                                        list[Blend], int, int]:
    """Blend one matched patch pair; returns point/color blocks, blend
    provenance and (onesided, missing) point counts."""
    A = world_a[pa.indices.astype(np.int64)]
    B = world_b[pb.indices.astype(np.int64)]
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    ac, bc = A - ca, B - cb
    da = _min_distances(ac, bc)
    db = _min_distances(bc, ac)
    cov_a = da < coverage_radius
    cov_b = db < coverage_radius
    mutual = (cov_a.sum() + cov_b.sum()) / max(1, len(A) + len(B))

    points_out: list[np.ndarray] = []
    colors_out: list[np.ndarray] = []
    blends: list[Blend] = []
    n_one = 0
    n_miss = 0
    cursor = 0

    def emit(pts: np.ndarray, col: np.ndarray, key: str, mode: str,
             source: tuple[str, ...]) -> None:
        nonlocal cursor
        if pts.shape[0] == 0:
            return
        points_out.append(pts)
        colors_out.append(col)
        blends.append(Blend(key=key, mode=mode, source=source,
                            start=cursor, end=cursor + pts.shape[0]))
        cursor += pts.shape[0]

    pair = (pa.key, pb.key)
    if mutual >= min_mutual and cov_a.any() and cov_b.any():
        # Sufficient geometric correspondence: bidirectional blend.
        m = max(4, int(round(cov_a.sum() * (1 - t) + cov_b.sum() * t)))
        a_pts, a_col = _resample(A[cov_a], colors_a[pa.indices.astype(
            np.int64)][cov_a], _pca_order(A[cov_a]), m)
        b_pts, b_col = _resample(B[cov_b], colors_b[pb.indices.astype(
            np.int64)][cov_b], _pca_order(B[cov_b]), m)
        pts = (a_pts * (1 - t) + b_pts * t).astype(np.float32)
        col = np.clip(np.round(a_col.astype(np.float32) * (1 - t)
                               + b_col.astype(np.float32) * t),
                      0, 255).astype(np.uint8)
        emit(pts, col, slot_key, "bilateral", pair)
        lone_a, lone_b = A[~cov_a], B[~cov_b]
        lone_ca = colors_a[pa.indices.astype(np.int64)][~cov_a]
        lone_cb = colors_b[pb.indices.astype(np.int64)][~cov_b]
    else:
        # Too little mutual geometry: the whole pair is single-sided.
        lone_a, lone_b = A, B
        lone_ca = colors_a[pa.indices.astype(np.int64)]
        lone_cb = colors_b[pb.indices.astype(np.int64)]

    for suffix, pts, col, src in (("~a", lone_a, lone_ca, (pa.key,)),
                                  ("~b", lone_b, lone_cb, (pb.key,))):
        if pts.shape[0] == 0:
            continue
        if onesided is OneSided.KEEP:
            emit(pts.astype(np.float32), col, slot_key + suffix,
                 "one_sided", src)
            n_one += pts.shape[0]
        else:
            blends.append(Blend(key="", mode="missing", source=src,
                                start=cursor, end=cursor))
            n_miss += pts.shape[0]
    return points_out, colors_out, blends, n_one, n_miss


def interpolate_frame(frame_a: Frame, patches_a: list[Patch],
                      frame_b: Frame, patches_b: list[Patch],
                      table: CorrespondenceTable, t: float, *,
                      region_id: int, frame_index: int,
                      onesided: OneSided = OneSided.KEEP,
                      coverage_radius: float = 0.12,
                      min_mutual: float = 0.35) -> InterpResult:
    """Synthesise the frame at fraction ``t`` between two key frames."""
    world_a, world_b = frame_a.world_points, frame_b.world_points
    by_key_a = {p.key: p for p in patches_a}
    by_key_b = {p.key: p for p in patches_b}
    matched_a: set[str] = set()
    matched_b: set[str] = set()

    blocks_p: list[np.ndarray] = []
    blocks_c: list[np.ndarray] = []
    blends: list[Blend] = []
    n_one = n_miss = 0
    cursor = 0

    def shift(new_blends: list[Blend]) -> None:
        nonlocal cursor
        base = cursor
        for b in new_blends:
            b.start += base
            b.end += base
            if b.mode != "missing":
                cursor = max(cursor, b.end)
        blends.extend(new_blends)

    for slot, entry in enumerate(table.active()):
        pa, pb = by_key_a.get(entry.a), by_key_b.get(entry.b)
        if pa is None or pb is None:
            continue
        matched_a.add(entry.a)
        matched_b.add(entry.b)
        key = interp_key(region_id, frame_index, slot)
        pts, cols, bl, one, miss = _blend_pair(
            pa, pb, world_a, frame_a.colors, world_b, frame_b.colors,
            t, coverage_radius, min_mutual, onesided, key)
        blocks_p.extend(pts)
        blocks_c.extend(cols)
        shift(bl)
        n_one += one
        n_miss += miss

    # Unmatched patches are purely single-sided evidence.
    for key, patch, world, colors in (
            [(p.key, p, world_a, frame_a.colors) for p in patches_a
             if p.key not in matched_a]
            + [(p.key, p, world_b, frame_b.colors) for p in patches_b
               if p.key not in matched_b]):
        pts = world[patch.indices.astype(np.int64)]
        col = colors[patch.indices.astype(np.int64)]
        if onesided is OneSided.KEEP:
            blocks_p.append(pts.astype(np.float32))
            blocks_c.append(col)
            blends.append(Blend(key=key, mode="one_sided", source=(key,),
                                start=cursor, end=cursor + pts.shape[0]))
            cursor += pts.shape[0]
            n_one += pts.shape[0]
        else:
            blends.append(Blend(key="", mode="missing", source=(key,),
                                start=cursor, end=cursor))
            n_miss += pts.shape[0]

    if blocks_p:
        points = np.concatenate(blocks_p, axis=0).astype(np.float32)
        colors = np.concatenate(blocks_c, axis=0).astype(np.uint8)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)

    origin = (np.asarray(frame_a.origin, dtype=np.float32) * (1 - t)
              + np.asarray(frame_b.origin, dtype=np.float32) * t)
    pts_time = float(frame_a.pts) * (1 - t) + float(frame_b.pts) * t
    frame = Frame(pts=pts_time, origin=tuple(float(v) for v in origin),
                  points=np.ascontiguousarray(points - origin),
                  colors=np.ascontiguousarray(colors))

    patches: list[Patch] = []
    for slot, blend in enumerate(blends):
        if blend.mode == "missing":
            continue
        idx = np.arange(blend.start, blend.end, dtype=np.int32)
        wpts = points[blend.start:blend.end]
        key = blend.key if blend.key else interp_key(region_id,
                                                     frame_index, slot)
        patches.append(Patch(key=key, frame_index=frame_index,
                             label=slot, indices=idx,
                             centroid=wpts.mean(axis=0).astype(np.float32),
                             color=colors[blend.start:blend.end]
                             .mean(axis=0).astype(np.float32)))
    n_bil = sum(b.npoints for b in blends if b.mode == "bilateral")
    return InterpResult(frame=frame, patches=patches, blends=blends,
                        bilateral_points=n_bil, onesided_points=n_one,
                        missing_points=n_miss)


# --------------------------------------------------------------------------
# Late original frames
# --------------------------------------------------------------------------

@dataclass
class ReplacementEntry:
    """One row of the replacement table emitted on a late arrival."""

    interp_key: str
    late_key: str
    confidence: float
    trustworthy: bool

    def to_dict(self) -> dict:
        return {"interp": self.interp_key, "late": self.late_key,
                "confidence": round(float(self.confidence), 4),
                "trustworthy": self.trustworthy}

    @staticmethod
    def from_dict(d: dict) -> "ReplacementEntry":
        return ReplacementEntry(d["interp"], d["late"],
                                float(d["confidence"]),
                                bool(d["trustworthy"]))


@dataclass
class ReconcileReport:
    frame_index: int
    replacements: list[ReplacementEntry]
    replaced: list[str]            # interp patch keys swapped for originals
    kept_interp: list[str]         # interp patches kept (untrusted / locked)
    conflicts: list[dict]          # locked revisions colliding with arrivals
    unmatched_late: list[str]      # new original patches without a counterpart

    def to_dict(self) -> dict:
        return {"frame_index": self.frame_index,
                "replacements": [r.to_dict() for r in self.replacements],
                "replaced": list(self.replaced),
                "kept_interp": list(self.kept_interp),
                "conflicts": [dict(c) for c in self.conflicts],
                "unmatched_late": list(self.unmatched_late)}

    @staticmethod
    def from_dict(d: dict) -> "ReconcileReport":
        return ReconcileReport(
            frame_index=int(d["frame_index"]),
            replacements=[ReplacementEntry.from_dict(r)
                          for r in d.get("replacements", [])],
            replaced=list(d.get("replaced", [])),
            kept_interp=list(d.get("kept_interp", [])),
            conflicts=[dict(c) for c in d.get("conflicts", [])],
            unmatched_late=list(d.get("unmatched_late", [])))

    @property
    def trustworthy_keys(self) -> set[str]:
        return {r.interp_key for r in self.replacements if r.trustworthy}


def match_replacement(interp_patches: list[Patch], late_patches: list[Patch],
                      max_dist: float = 0.6) -> list[tuple[Patch, Patch, float]]:
    """Many-to-one matching used for replacements: every interp patch may
    name its closest original patch (interp patches are fragments; several
    of them can legitimately be superseded by one original surface)."""
    out = []
    for ip in interp_patches:
        best, best_d = None, max_dist
        for lp in late_patches:
            d = float(np.linalg.norm(ip.centroid - lp.centroid))
            if d < best_d:
                best, best_d = lp, d
        if best is not None:
            out.append((ip, best, best_d))
    return out


def reconcile_late_frame(interp: InterpResult, late_frame: Frame,
                         late_patches: list[Patch], *,
                         frame_index: int,
                         locked: frozenset[str] | set[str] = frozenset(),
                         trust: float = 0.55,
                         max_dist: float = 0.6
                         ) -> tuple[Frame, list[Patch], ReconcileReport]:
    """Splice a late original frame into the interpolated gap.

    Returns the mixed frame, its patches and the replacement table.  Locked
    interp revisions are never replaced: their collision with the arriving
    original is reported as a conflict for the editor to resolve.
    """
    matches = match_replacement(interp.patches, late_patches, max_dist)
    by_interp: dict[str, tuple[Patch, float]] = {}
    for ip, lp, d in matches:
        by_interp[ip.key] = (lp, d)

    replacements: list[ReplacementEntry] = []
    replaced: list[str] = []
    kept: list[str] = []
    conflicts: list[dict] = []
    used_late: set[str] = set()

    for ip in interp.patches:
        hit = by_interp.get(ip.key)
        if hit is None:
            kept.append(ip.key)
            continue
        lp, dist = hit
        conf = float(np.clip(1.0 - dist / max_dist, 0.0, 1.0))
        trustworthy = conf >= trust
        replacements.append(ReplacementEntry(ip.key, lp.key, conf,
                                             trustworthy))
        if ip.key in locked:
            # Manually locked revision collides with the arriving original:
            # keep the revision, flag the conflict for the editor.
            conflicts.append({"interp": ip.key, "late": lp.key,
                              "reason": "locked_revision",
                              "confidence": round(conf, 4)})
            kept.append(ip.key)
            continue
        # Geometry confirms the patch: the original supersedes the stand-in.
        # Only trustworthy matches also link identities (marker migration).
        replaced.append(ip.key)
        used_late.add(lp.key)

    late_by_key = {p.key: p for p in late_patches}
    unmatched_late = [p.key for p in late_patches
                      if p.key not in used_late
                      and p.key not in {c["late"] for c in conflicts}]

    # Assemble the mixed frame: original evidence for replaced/new patches,
    # interpolated points only where they were kept.
    world_late = late_frame.world_points
    blocks_p: list[np.ndarray] = []
    blocks_c: list[np.ndarray] = []
    mixed_patches: list[Patch] = []
    cursor = 0

    def take(patch: Patch, world: np.ndarray, colors: np.ndarray,
             key: str, label: int) -> None:
        nonlocal cursor
        idx64 = patch.indices.astype(np.int64)
        pts = world[idx64].astype(np.float32)
        col = colors[idx64]
        n = pts.shape[0]
        blocks_p.append(pts)
        blocks_c.append(col)
        mixed_patches.append(Patch(
            key=key, frame_index=frame_index, label=label,
            indices=np.arange(cursor, cursor + n, dtype=np.int32),
            centroid=pts.mean(axis=0).astype(np.float32),
            color=col.mean(axis=0).astype(np.float32)))
        cursor += n

    label = 0
    for key in sorted(used_late | set(unmatched_late)):
        take(late_by_key[key], world_late, late_frame.colors, key, label)
        label += 1
    interp_by_key = {p.key: p for p in interp.patches}
    world_interp = interp.frame.world_points
    for key in sorted(kept):
        take(interp_by_key[key], world_interp, interp.frame.colors,
             key, label)
        label += 1

    if blocks_p:
        points = np.concatenate(blocks_p, axis=0)
        colors = np.concatenate(blocks_c, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.uint8)
    origin = np.asarray(late_frame.origin, dtype=np.float32)
    mixed = Frame(pts=late_frame.pts,
                  origin=tuple(float(v) for v in origin),
                  points=np.ascontiguousarray(points - origin),
                  colors=np.ascontiguousarray(colors))
    # Rebase mixed-patch indices: they were built against world arrays, and
    # the concatenation order matches the block order, so indices stay valid.
    report = ReconcileReport(frame_index=frame_index,
                             replacements=replacements, replaced=replaced,
                             kept_interp=kept, conflicts=conflicts,
                             unmatched_late=unmatched_late)
    return mixed, mixed_patches, report
