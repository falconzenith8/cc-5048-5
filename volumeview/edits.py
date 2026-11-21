"""Timeline view model: the undoable tween (补帧) layer and the edit track.

``TimelineView`` is the headless model behind the player's timeline view.
It owns three things:

* the **tween layer**: interpolation regions that fill gaps between
  available original captures.  Every edit to the layer (create region,
  veto / approve a correspondence group, delete a table entry, lock a
  patch revision, switch single-sided policy) is an undoable command;
* the **edit track**: contact markers that reference *surface identity*
  (a surface id from the registry), never a raw position, so they follow
  the surface through re-interpolation and late original frames - and are
  suspended instead of being re-bound when the identity breaks;
* **provenance**: every frame the view hands out reports whether it is an
  original capture, an interpolated stand-in, or a mixed frame spliced by
  a late arrival.  Provenance survives save / load and reverse scrubbing.

Late original frames are reconciled by :meth:`TimelineView.on_late_frame`:
the reconstructor emits the replacement table, the track migrates markers
whose new surface is still trustworthy, and suspends the association
otherwise.  Locked revisions are never auto-replaced; their collisions are
reported as conflicts.

Export resolves markers strictly through the registry: a marker whose
surface cannot be found (or is ambiguous) in its frame is skipped and
logged - contact audio is never re-attached to a different surface.
"""

from __future__ import annotations

import json
import wave
from dataclasses import dataclass, field

import numpy as np

from .frame import Frame, FrameWriter
from .surfaces import (APPROVED, CorrespondenceTable, MatchEntry, Patch,
                       SurfaceRegistry, VETOED, compute_correspondence,
                       original_key, segment_frame)
from .tween import (InterpResult, OneSided, ReconcileReport,
                    interpolate_frame, reconcile_late_frame)

AUDIO_RATE = 44100


# --------------------------------------------------------------------------
# Edit-track markers
# --------------------------------------------------------------------------

@dataclass
class ContactMarker:
    """A contact event bound to a surface identity."""

    id: str
    time: float
    surface_id: str
    anchor: str                 # patch key the marker was placed on
    state: str = "active"       # active | suspended
    history: list[dict] = field(default_factory=list)

    def log(self, event: str, **detail) -> None:
        self.history.append({"event": event, **detail})

    def to_dict(self) -> dict:
        return {"id": self.id, "time": self.time,
                "surface_id": self.surface_id, "anchor": self.anchor,
                "state": self.state, "history": list(self.history)}

    @staticmethod
    def from_dict(d: dict) -> "ContactMarker":
        return ContactMarker(id=d["id"], time=float(d["time"]),
                             surface_id=d["surface_id"], anchor=d["anchor"],
                             state=d.get("state", "active"),
                             history=[dict(h) for h in d.get("history", [])])


@dataclass
class InterpRegion:
    id: int
    frame_a: int
    frame_b: int
    onesided: OneSided
    table: CorrespondenceTable
    approved: set[tuple[str, str]] = field(default_factory=set)
    vetoed: set[tuple[str, str]] = field(default_factory=set)
    locks: set[str] = field(default_factory=set)
    version: int = 0

    def covers(self, frame_index: int) -> bool:
        return self.frame_a < frame_index < self.frame_b


@dataclass
class TimelineFrame:
    frame: Frame
    provenance: dict


@dataclass
class _Command:
    label: str
    before: dict
    after: dict


class TimelineView:
    """Headless model of the player's timeline view."""

    def __init__(self, frames: dict[int, Frame], fps: float):
        self.frames: dict[int, Frame] = dict(frames)
        self.fps = float(fps)
        self.regions: dict[int, InterpRegion] = {}
        self.markers: dict[str, ContactMarker] = {}
        self.registry = SurfaceRegistry()
        self.replaced: dict[int, dict] = {}   # frame_index -> mixed record
        self._undo: list[_Command] = []
        self._redo: list[_Command] = []
        self._patches: dict[int, list[Patch]] = {}
        self._interp: dict[tuple[int, int, int], InterpResult] = {}
        self._next_region = 1
        self._next_marker = 1
        self._rebuild_registry()

    # ------------------------------------------------------------------
    # frame access / provenance
    # ------------------------------------------------------------------
    def _patches_of(self, frame_index: int) -> list[Patch]:
        if frame_index not in self._patches:
            self._patches[frame_index] = segment_frame(
                self.frames[frame_index], frame_index)
        return self._patches[frame_index]

    def _region_of(self, frame_index: int) -> InterpRegion | None:
        for r in self.regions.values():
            if r.covers(frame_index):
                return r
        return None

    def _interp_for(self, region: InterpRegion,
                    frame_index: int) -> InterpResult:
        key = (region.id, frame_index, region.version)
        if key not in self._interp:
            t = ((frame_index - region.frame_a)
                 / (region.frame_b - region.frame_a))
            self._interp[key] = interpolate_frame(
                self.frames[region.frame_a], self._patches_of(region.frame_a),
                self.frames[region.frame_b], self._patches_of(region.frame_b),
                region.table, t, region_id=region.id,
                frame_index=frame_index, onesided=region.onesided)
        return self._interp[key]

    def _frame_patches(self, frame_index: int) -> list[Patch]:
        if frame_index in self.replaced:
            return self.replaced[frame_index]["patches"]
        if frame_index in self.frames:
            return self._patches_of(frame_index)
        region = self._region_of(frame_index)
        if region is not None:
            return self._interp_for(region, frame_index).patches
        return []

    def frame_at(self, frame_index: int) -> TimelineFrame | None:
        """Frame plus provenance; valid for any scrub direction."""
        if frame_index in self.replaced:
            rec = self.replaced[frame_index]
            report: ReconcileReport = rec["report"]
            return TimelineFrame(rec["frame"], {
                "source": "mixed",
                "frame_index": frame_index,
                "recovered": not report.kept_interp and not report.conflicts,
                "replaced_patches": list(report.replaced),
                "kept_interp_patches": list(report.kept_interp),
                "conflicts": [dict(c) for c in report.conflicts],
            })
        if frame_index in self.frames:
            return TimelineFrame(self.frames[frame_index], {
                "source": "original", "frame_index": frame_index})
        region = self._region_of(frame_index)
        if region is not None:
            res = self._interp_for(region, frame_index)
            t = ((frame_index - region.frame_a)
                 / (region.frame_b - region.frame_a))
            return TimelineFrame(res.frame, {
                "source": "interp",
                "region": region.id,
                "frame_index": frame_index,
                "key_a": region.frame_a, "key_b": region.frame_b,
                "t": round(t, 4),
                "onesided": region.onesided.value,
                "blends": [{"key": b.key, "mode": b.mode,
                            "source": list(b.source)}
                           for b in res.blends],
                "bilateral_points": res.bilateral_points,
                "onesided_points": res.onesided_points,
                "missing_points": res.missing_points,
            })
        return None

    # ------------------------------------------------------------------
    # registry / marker migration
    # ------------------------------------------------------------------
    def _rebuild_registry(self) -> None:
        all_keys: set[str] = set()
        unions: list[tuple[str, str]] = []
        for idx in sorted(self.frames):
            for p in self._patches_of(idx):
                all_keys.add(p.key)
        for region in self.regions.values():
            for e in region.table.active():
                unions.append((e.a, e.b))
            for idx in range(region.frame_a + 1, region.frame_b):
                res = self._interp_for(region, idx)
                for patch, blend in zip(res.patches,
                                        [b for b in res.blends
                                         if b.mode != "missing"]):
                    all_keys.add(patch.key)
                    unions.append((patch.key, blend.source[0]))
        for idx, rec in self.replaced.items():
            for p in rec["patches"]:
                all_keys.add(p.key)
            for rep in rec["report"].replacements:
                if rep.trustworthy and rep.interp_key in {
                        k for k in rec["report"].replaced}:
                    unions.append((rep.interp_key, rep.late_key))
        self.registry.rebuild(all_keys, unions)
        self._migrate_markers()

    def _migrate_markers(self) -> None:
        for m in self.markers.values():
            sid = self.registry.surface_of(m.anchor)
            if sid is None:
                self._suspend(m, "anchor_vanished")
            elif sid != m.surface_id:
                # The anchor's component no longer carries the marker's
                # surface id: identity could not be confirmed, so the
                # association is suspended rather than re-bound.
                self._suspend(m, "surface_rebuilt")

    @staticmethod
    def _suspend(marker: ContactMarker, reason: str) -> None:
        if marker.state != "suspended":
            marker.state = "suspended"
            marker.log("suspended", reason=reason)

    # ------------------------------------------------------------------
    # undoable edit commands
    # ------------------------------------------------------------------
    def _memento(self) -> dict:
        return {
            "regions": {rid: {
                "frame_a": r.frame_a, "frame_b": r.frame_b,
                "onesided": r.onesided.value, "version": r.version,
                "table": r.table.to_dict(),
                "approved": sorted(r.approved), "vetoed": sorted(r.vetoed),
                "locks": sorted(r.locks),
            } for rid, r in self.regions.items()},
            "markers": {mid: m.to_dict() for mid, m in self.markers.items()},
            "registry": self.registry.to_dict(),
            "replaced": {str(i): rec["report"].to_dict()
                         for i, rec in self.replaced.items()},
            "counters": [self._next_region, self._next_marker],
        }

    def _restore(self, mem: dict) -> None:
        self.regions = {}
        for rid, d in mem["regions"].items():
            region = InterpRegion(
                id=int(rid), frame_a=d["frame_a"], frame_b=d["frame_b"],
                onesided=OneSided(d["onesided"]),
                table=CorrespondenceTable.from_dict(d["table"]),
                approved={tuple(p) for p in d["approved"]},
                vetoed={tuple(p) for p in d["vetoed"]},
                locks=set(d["locks"]), version=d["version"])
            self.regions[region.id] = region
        self.markers = {mid: ContactMarker.from_dict(m)
                        for mid, m in mem["markers"].items()}
        self.registry = SurfaceRegistry.from_dict(mem["registry"])
        self._next_region, self._next_marker = mem["counters"]
        self._interp.clear()
        # Rebuild mixed frames from their reports (deterministic) without
        # re-firing marker side effects: marker states come from the memento.
        self.replaced = {}
        for idx_s, report_d in mem["replaced"].items():
            self._reconcile_silent(int(idx_s), ReconcileReport.from_dict(report_d))

    def _edit(self, label: str, mutate) -> None:
        before = self._memento()
        mutate()
        after = self._memento()
        self._undo.append(_Command(label, before, after))
        self._redo.clear()

    def undo(self) -> str | None:
        if not self._undo:
            return None
        cmd = self._undo.pop()
        self._restore(cmd.before)
        self._redo.append(cmd)
        return cmd.label

    def redo(self) -> str | None:
        if not self._redo:
            return None
        cmd = self._redo.pop()
        self._restore(cmd.after)
        self._undo.append(cmd)
        return cmd.label

    @property
    def undo_labels(self) -> list[str]:
        return [c.label for c in self._undo]

    # ------------------------------------------------------------------
    # tween-layer edits
    # ------------------------------------------------------------------
    def add_interp_region(self, frame_a: int, frame_b: int,
                          onesided: OneSided = OneSided.KEEP) -> int:
        if frame_b <= frame_a + 1:
            raise ValueError("a region must span at least one missing frame")
        for idx in (frame_a, frame_b):
            if idx not in self.frames:
                raise ValueError(f"key frame {idx} is not available")
        for r in self.regions.values():
            if r.frame_a < frame_b and frame_a < r.frame_b:
                raise ValueError("regions may not overlap")

        def mutate():
            rid = self._next_region
            self._next_region += 1
            table = compute_correspondence(self._patches_of(frame_a),
                                           self._patches_of(frame_b))
            self.regions[rid] = InterpRegion(id=rid, frame_a=frame_a,
                                             frame_b=frame_b,
                                             onesided=onesided, table=table)
            self._rebuild_registry()

        self._edit(f"add-region f{frame_a}-f{frame_b}", mutate)
        return self._next_region - 1

    def veto_group(self, region_id: int, group: int) -> list[str]:
        region = self.regions[region_id]
        hit: list[str] = []

        def mutate():
            for e in region.table.veto_group(group):
                region.vetoed.add((e.a, e.b))
                hit.append(f"{e.a}<->{e.b}")
            region.version += 1
            self._interp.clear()
            self._rebuild_registry()

        self._edit(f"veto-group r{region_id} g{group}", mutate)
        return hit

    def approve_group(self, region_id: int, group: int) -> list[str]:
        region = self.regions[region_id]
        hit: list[str] = []

        def mutate():
            for e in region.table.approve_group(group):
                region.approved.add((e.a, e.b))
                hit.append(f"{e.a}<->{e.b}")
            self._rebuild_registry()

        self._edit(f"approve-group r{region_id} g{group}", mutate)
        return hit

    def approve_pair(self, region_id: int, a: str, b: str) -> None:
        """Pin one correspondence by hand (e.g. the correct actor link the
        matcher missed).  Pinned pairs survive recomputation."""
        region = self.regions[region_id]

        def mutate():
            entry = region.table.find(a, b)
            if entry is None:
                entry = MatchEntry(a=a, b=b, confidence=1.0,
                                   state=APPROVED, origin="manual")
                region.table.entries.append(entry)
            else:
                entry.state = APPROVED
            region.approved.add((a, b))
            region.table.regroup()
            region.version += 1
            self._interp.clear()
            self._rebuild_registry()

        self._edit(f"approve-pair r{region_id} {a}<->{b}", mutate)

    def delete_entry(self, region_id: int, a: str, b: str) -> None:
        """Delete one row of the correspondence table.  The surface registry
        is rebuilt immediately, so markers bound to the broken identity are
        suspended rather than re-attached to a neighbouring surface."""
        region = self.regions[region_id]

        def mutate():
            region.table.delete_entry(a, b)
            region.table.regroup()
            region.version += 1
            self._interp.clear()
            self._rebuild_registry()

        self._edit(f"delete-entry r{region_id} {a}<->{b}", mutate)

    def lock_patch(self, region_id: int, patch_key: str) -> None:
        """Manually lock an interpolated revision: late originals must not
        auto-replace it, and a collision is reported as a conflict."""
        region = self.regions[region_id]

        def mutate():
            region.locks.add(patch_key)

        self._edit(f"lock-patch r{region_id} {patch_key}", mutate)

    def set_onesided(self, region_id: int, policy: OneSided) -> None:
        region = self.regions[region_id]

        def mutate():
            region.onesided = policy
            region.version += 1
            self._interp.clear()
            self._rebuild_registry()

        self._edit(f"onesided r{region_id} {policy.value}", mutate)

    def recompute_region(self, region_id: int) -> None:
        """Recompute a region's table.  Approved pairs are pinned, vetoed
        pairs stay excluded; markers and their history are preserved."""
        region = self.regions[region_id]

        def mutate():
            table = compute_correspondence(self._patches_of(region.frame_a),
                                           self._patches_of(region.frame_b))
            for e in table.entries:
                if (e.a, e.b) in region.vetoed:
                    e.state = VETOED
                elif (e.a, e.b) in region.approved:
                    e.state = APPROVED
            for a, b in sorted(region.approved):
                if table.find(a, b) is None:
                    table.entries.append(MatchEntry(
                        a=a, b=b, confidence=1.0, state=APPROVED,
                        origin="manual"))
            table.regroup()
            region.table = table
            region.version += 1
            self._interp.clear()
            self._rebuild_registry()

        self._edit(f"recompute r{region_id}", mutate)

    # ------------------------------------------------------------------
    # edit track
    # ------------------------------------------------------------------
    def place_marker(self, time: float, *, surface_id: str | None = None,
                     patch_key: str | None = None,
                     near: tuple[float, float, float] | None = None) -> str:
        """Place a contact marker on a surface (or a specific patch)."""
        frame_index = int(round(time * self.fps))
        if patch_key is not None:
            surface_id = self.registry.surface_of(patch_key)
            if surface_id is None:
                raise ValueError(f"patch {patch_key} has no surface")
            anchor = patch_key
        else:
            if surface_id is None:
                raise ValueError("surface_id or patch_key required")
            members = self.registry.members(surface_id)
            here = [k for k in self._frame_patch_keys(frame_index)
                    if k in members]
            if not here:
                raise ValueError(
                    f"surface {surface_id} has no patch in frame {frame_index}")
            anchor = self._pick_patch(frame_index, here, near)

        holder: dict[str, str] = {}

        def mutate():
            mid = f"M{self._next_marker}"
            self._next_marker += 1
            marker = ContactMarker(id=mid, time=float(time),
                                   surface_id=surface_id, anchor=anchor)
            marker.log("placed", frame=frame_index, surface=surface_id)
            self.markers[mid] = marker
            holder["mid"] = mid

        self._edit(f"place-marker @{time:.3f}s", mutate)
        return holder["mid"]

    def _frame_patch_keys(self, frame_index: int) -> list[str]:
        return [p.key for p in self._frame_patches(frame_index)]

    def _pick_patch(self, frame_index: int, keys: list[str],
                    near) -> str:
        if len(keys) == 1 or near is None:
            return sorted(keys)[0]
        patches = {p.key: p for p in self._frame_patches(frame_index)}
        target = np.asarray(near, dtype=np.float32)
        return min(keys, key=lambda k: float(
            np.linalg.norm(patches[k].centroid - target)))

    def resume_marker(self, marker_id: str, surface_id: str) -> None:
        """Re-bind a suspended marker to a surface the editor picked."""
        def mutate():
            m = self.markers[marker_id]
            frame_index = int(round(m.time * self.fps))
            members = self.registry.members(surface_id)
            here = [k for k in self._frame_patch_keys(frame_index)
                    if k in members]
            if not here:
                raise ValueError(
                    f"surface {surface_id} has no patch near frame "
                    f"{frame_index}")
            m.anchor = sorted(here)[0]
            m.surface_id = surface_id
            m.state = "active"
            m.log("resumed", surface=surface_id)

        self._edit(f"resume-marker {marker_id}", mutate)

    def _surface_position(self, surface_id: str,
                          frame_index: int) -> np.ndarray | None:
        """Point-weighted centroid of the surface's patches in a frame.

        A surface legitimately appears as several fragments in one frame
        (a bidirectional blend plus single-sided slivers of one tracked
        link), so all member patches contribute.  ``None`` when the
        surface has no patch in the frame at all.
        """
        members = self.registry.members(surface_id)
        patches = {p.key: p for p in self._frame_patches(frame_index)}
        here = [p for k, p in patches.items() if k in members]
        if not here:
            return None
        total = sum(p.npoints for p in here)
        acc = np.zeros(3, dtype=np.float64)
        for p in here:
            acc += p.centroid.astype(np.float64) * p.npoints
        return (acc / total).astype(np.float32)

    def marker_position(self, marker_id: str,
                        frame_index: int | None = None) -> np.ndarray | None:
        m = self.markers[marker_id]
        if frame_index is None:
            frame_index = int(round(m.time * self.fps))
        return self._surface_position(m.surface_id, frame_index)

    # ------------------------------------------------------------------
    # late original frames
    # ------------------------------------------------------------------
    def on_late_frame(self, frame_index: int,
                      frame: Frame) -> list[ReconcileReport]:
        """A late original arrived: reconcile every region covering it.

        The reconstructor emits the replacement table first; the track then
        migrates markers whose new surface is trustworthy and suspends the
        association otherwise.  Locked revisions are kept and reported as
        conflicts."""
        self.frames[frame_index] = frame
        self._patches.pop(frame_index, None)
        late_patches = self._patches_of(frame_index)
        reports: list[ReconcileReport] = []
        for region in list(self.regions.values()):
            if not region.covers(frame_index):
                continue
            interp = self._interp_for(region, frame_index)
            mixed, mixed_patches, report = reconcile_late_frame(
                interp, frame, late_patches, frame_index=frame_index,
                locked=region.locks)
            self.replaced[frame_index] = {
                "frame": mixed, "patches": mixed_patches,
                "report": report, "region": region.id}
            reports.append(report)
        self._rebuild_registry()
        # Marker migration rule: move only when the new surface is trusted.
        for report in reports:
            trusted = report.trustworthy_keys
            prefix = f"r{self.replaced[frame_index]['region']}" \
                     f"f{frame_index}s"
            for m in self.markers.values():
                if not m.anchor.startswith(prefix):
                    continue
                if m.anchor in trusted:
                    m.log("migrated_to_original", frame=frame_index,
                          surface=m.surface_id)
                else:
                    self._suspend(m, "association_suspended")
        return reports

    def _reconcile_silent(self, frame_index: int,
                          report: ReconcileReport) -> None:
        """Recompute a mixed frame from its report (load / undo restore)."""
        region = self._region_of(frame_index)
        if region is None or frame_index not in self.frames:
            return
        interp = self._interp_for(region, frame_index)
        mixed, mixed_patches, _ = reconcile_late_frame(
            interp, self.frames[frame_index],
            self._patches_of(frame_index), frame_index=frame_index,
            locked=region.locks)
        self.replaced[frame_index] = {
            "frame": mixed, "patches": mixed_patches,
            "report": report, "region": region.id}

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        mem = self._memento()
        payload = {
            "version": 1,
            "fps": self.fps,
            "frames": sorted(self.frames),
            "regions": mem["regions"],
            "markers": [m.to_dict() for m in self.markers.values()],
            "registry": mem["registry"],
            "replaced": mem["replaced"],
            "counters": mem["counters"],
            "undo_labels": self.undo_labels,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str, frames: dict[int, Frame]) -> "TimelineView":
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        # Only the frames recorded at save time had arrived then; anything
        # else in the caller's dict is a late frame and must enter through
        # on_late_frame so reconcile runs and markers migrate safely.
        known = set(payload["frames"])
        frames = {i: f for i, f in frames.items() if i in known}
        view = cls(frames, float(payload["fps"]))
        mem = {
            "regions": {int(rid): r for rid, r in
                        payload["regions"].items()},
            "markers": {m["id"]: m for m in payload["markers"]},
            "registry": payload["registry"],
            "replaced": payload["replaced"],
            "counters": payload["counters"],
        }
        view._restore(mem)
        return view

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def export(self, out_vpc: str, events_path: str,
               wav_path: str | None = None) -> dict:
        """Bake the timeline to a .vpc plus contact audio events.

        Markers resolve strictly through the surface registry: a marker
        whose surface is absent or ambiguous in its frame is skipped and
        logged, never re-attached to another surface.
        """
        last = max(max(self.frames, default=-1),
                   max((r.frame_b for r in self.regions.values()),
                       default=-1))
        skipped_frames: list[int] = []
        with FrameWriter(out_vpc) as writer:
            for idx in range(last + 1):
                tf = self.frame_at(idx)
                if tf is None:
                    skipped_frames.append(idx)
                    continue
                writer.write(tf.frame)

        events: list[dict] = []
        skipped: list[dict] = []
        for m in self.markers.values():
            if m.state != "active":
                skipped.append({"id": m.id, "reason": "suspended"})
                continue
            frame_index = int(round(m.time * self.fps))
            pos = self._surface_position(m.surface_id, frame_index)
            if pos is None:
                skipped.append({"id": m.id, "reason": "surface_absent"})
                continue
            events.append({"id": m.id, "time": m.time,
                           "surface_id": m.surface_id,
                           "position": [round(float(v), 4) for v in pos]})

        with open(events_path, "w", encoding="utf-8") as fh:
            json.dump({"events": events, "skipped": skipped,
                       "skipped_frames": skipped_frames}, fh, indent=2)
        if wav_path is not None:
            self._render_contact_wav(wav_path, events,
                                     duration=(last + 1) / self.fps + 0.5)
        return {"events": events, "skipped": skipped,
                "skipped_frames": skipped_frames}

    @staticmethod
    def _render_contact_wav(path: str, events: list[dict],
                            duration: float) -> None:
        """Stereo contact track: each event is a short click panned by the
        resolved surface position (left/right actor separation)."""
        n = int(duration * AUDIO_RATE)
        mix = np.zeros((n, 2), dtype=np.float32)
        click_len = int(0.045 * AUDIO_RATE)
        tt = np.arange(click_len) / AUDIO_RATE
        click = (np.sin(2 * np.pi * 880.0 * tt)
                 * np.exp(-tt * 60.0)).astype(np.float32)
        for ev in events:
            start = int(ev["time"] * AUDIO_RATE)
            end = min(start + click_len, n)
            if end <= start:
                continue
            pan = float(np.clip(ev["position"][0] / 1.5, -1.0, 1.0))
            angle = (pan + 1.0) * np.pi / 4.0
            gains = np.array([np.cos(angle), np.sin(angle)],
                             dtype=np.float32)
            mix[start:end] += click[:end - start, None] * gains
        peak = float(np.max(np.abs(mix))) or 1.0
        pcm = (np.clip(mix / peak * 0.9, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(path, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(AUDIO_RATE)
            wav.writeframes(pcm.tobytes())
