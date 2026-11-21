"""Surface identity: patch segmentation, correspondence tables, registry.

The interpolation layer never refers to raw points: it reasons about
*patches* (connected surface fragments of one capture) and *surfaces*
(persistent identities that span frames).  A contact marker on the edit
track references a surface id, so its meaning survives re-interpolation,
late original frames and project reloads.

Patch keys are plain strings so every structure here is JSON serialisable:

* original captures  ``f{frame_index}:{label}``
* interpolated slots ``r{region_id}f{frame_index}s{slot}``

A :class:`CorrespondenceTable` links patches of two key frames.  Entries
carry a state (``proposed`` / ``approved`` / ``vetoed`` / ``deleted``) and
a group id; groups are the connected components of the match graph, which
is the granularity at which an editor vetoes a wrong connection (e.g. two
look-alike actors swapped by the matcher).

The :class:`SurfaceRegistry` unions linked patches into surfaces.  On
rebuild it hands surface ids back out by Jaccard overlap with the previous
assignment: a component that still overlaps its previous incarnation by at
least ``trust`` keeps the id (the surface is "still trustworthy"), otherwise
a fresh id is issued and markers bound to the old id are left suspended by
the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Entry states
PROPOSED = "proposed"
APPROVED = "approved"
VETOED = "vetoed"
DELETED = "deleted"
ACTIVE_STATES = (PROPOSED, APPROVED)


def original_key(frame_index: int, label: int) -> str:
    return f"f{frame_index}:{label}"


def interp_key(region_id: int, frame_index: int, slot: int) -> str:
    return f"r{region_id}f{frame_index}s{slot}"


@dataclass(frozen=True)
class Patch:
    """A connected surface fragment of one frame."""

    key: str
    frame_index: int
    label: int
    indices: np.ndarray      # int32 point indices into the frame arrays
    centroid: np.ndarray     # float32 (3,), world space
    color: np.ndarray        # float32 (3,), mean RGB

    @property
    def npoints(self) -> int:
        return int(self.indices.shape[0])


def segment_frame(frame, frame_index: int, voxel: float = 0.12,
                  min_points: int = 24) -> list[Patch]:
    """Deterministic connected-components segmentation over a voxel grid.

    The point order of a capture is meaningless (frames arrive shuffled), so
    voxels are visited in sorted order and components are labelled by their
    smallest voxel key: re-segmenting the same frame always yields the same
    patch keys, which is what makes saved projects reloadable.
    """
    world = np.asarray(frame.world_points, dtype=np.float32)
    n = world.shape[0]
    if n == 0:
        return []
    cells = np.floor(world / float(voxel)).astype(np.int32)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for idx in range(n):
        key = (int(cells[idx, 0]), int(cells[idx, 1]), int(cells[idx, 2]))
        buckets.setdefault(key, []).append(idx)

    visited: set[tuple[int, int, int]] = set()
    components: list[list[int]] = []
    comp_keys: list[tuple[int, int, int]] = []
    neighbour_offsets = sorted(
        (dx, dy, dz)
        for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0))
    for start in sorted(buckets):
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        members: list[int] = []
        min_key = start
        while stack:
            cell = stack.pop()
            members.extend(buckets[cell])
            if cell < min_key:
                min_key = cell
            for dx, dy, dz in neighbour_offsets:
                nb = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                if nb in buckets and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components.append(members)
        comp_keys.append(min_key)

    order = sorted(range(len(components)), key=lambda i: comp_keys[i])
    components = [components[i] for i in order]

    # Fold tiny fragments into their nearest sizeable component so noise
    # specks do not become surfaces of their own.
    big = [i for i, c in enumerate(components) if len(c) >= min_points]
    if big:
        centroids = [world[np.asarray(c, dtype=np.int64)].mean(axis=0)
                     for c in components]
        for i, comp in enumerate(components):
            if i in big:
                continue
            dists = [float(np.linalg.norm(centroids[i] - centroids[j]))
                     for j in big]
            components[big[int(np.argmin(dists))]].extend(comp)
        components = [components[i] for i in big]

    patches = []
    for label, comp in enumerate(components):
        idx = np.asarray(sorted(comp), dtype=np.int32)
        pts = world[idx.astype(np.int64)]
        patches.append(Patch(
            key=original_key(frame_index, label),
            frame_index=frame_index,
            label=label,
            indices=idx,
            centroid=pts.mean(axis=0).astype(np.float32),
            color=frame.colors[idx.astype(np.int64)].mean(axis=0)
                  .astype(np.float32),
        ))
    return patches


@dataclass
class MatchEntry:
    """One row of a correspondence table: patch ``a`` ↔ patch ``b``."""

    a: str
    b: str
    confidence: float
    group: int = -1
    state: str = PROPOSED
    origin: str = "auto"       # auto | manual (editor-pinned)

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b,
                "confidence": round(float(self.confidence), 4),
                "group": self.group, "state": self.state,
                "origin": self.origin}

    @staticmethod
    def from_dict(d: dict) -> "MatchEntry":
        return MatchEntry(a=d["a"], b=d["b"],
                          confidence=float(d["confidence"]),
                          group=int(d.get("group", -1)),
                          state=d.get("state", PROPOSED),
                          origin=d.get("origin", "auto"))


class CorrespondenceTable:
    """The replacement/correspondence table emitted between two frames."""

    def __init__(self, entries: list[MatchEntry] | None = None):
        self.entries: list[MatchEntry] = list(entries or [])

    def active(self) -> list[MatchEntry]:
        return [e for e in self.entries if e.state in ACTIVE_STATES]

    def find(self, a: str, b: str) -> MatchEntry | None:
        for e in self.entries:
            if e.a == a and e.b == b:
                return e
        return None

    def groups(self) -> dict[int, list[MatchEntry]]:
        out: dict[int, list[MatchEntry]] = {}
        for e in self.active():
            out.setdefault(e.group, []).append(e)
        return out

    def veto_group(self, group: int) -> list[MatchEntry]:
        hit = [e for e in self.entries
               if e.group == group and e.state in ACTIVE_STATES]
        for e in hit:
            e.state = VETOED
        return hit

    def approve_group(self, group: int) -> list[MatchEntry]:
        hit = [e for e in self.entries if e.group == group
               and e.state == PROPOSED]
        for e in hit:
            e.state = APPROVED
        return hit

    def delete_entry(self, a: str, b: str) -> MatchEntry:
        e = self.find(a, b)
        if e is None:
            raise KeyError(f"no correspondence entry {a} <-> {b}")
        e.state = DELETED
        return e

    def regroup(self) -> None:
        """Recompute group ids over the active entries (connected components
        of the bipartite match graph)."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            parent[find(x)] = find(y)

        for e in self.active():
            union(e.a, e.b)
        roots = sorted({find(k) for k in parent})
        gid = {root: i for i, root in enumerate(roots)}
        for e in self.entries:
            e.group = gid[find(e.a)] if e.state in ACTIVE_STATES else -1

    def to_dict(self) -> dict:
        return {"entries": [e.to_dict() for e in self.entries]}

    @staticmethod
    def from_dict(d: dict) -> "CorrespondenceTable":
        return CorrespondenceTable([MatchEntry.from_dict(e)
                                    for e in d.get("entries", [])])


def compute_correspondence(patches_a: list[Patch], patches_b: list[Patch],
                           max_dist: float = 0.8,
                           color_weight: float = 0.001) -> CorrespondenceTable:
    """Greedy one-to-one patch matching between two frames.

    Spatial distance gates the match; mean colour only breaks near ties.
    With look-alike actors this deliberately admits the classic failure of
    swapping two people that cross at close range - the exact case the
    editor then repairs by vetoing the group.
    """
    candidates: list[tuple[float, int, int]] = []
    for i, pa in enumerate(patches_a):
        for j, pb in enumerate(patches_b):
            dist = float(np.linalg.norm(pa.centroid - pb.centroid))
            if dist > max_dist:
                continue
            cdist = float(np.linalg.norm(pa.color - pb.color))
            cost = dist + color_weight * cdist
            candidates.append((cost, i, j))
    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    entries: list[MatchEntry] = []
    gate = max_dist + color_weight * 441.7  # 441.7 = max RGB L2 distance
    for cost, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        conf = float(np.clip(1.0 - cost / gate, 0.0, 1.0))
        entries.append(MatchEntry(a=patches_a[i].key, b=patches_b[j].key,
                                  confidence=conf))
    entries.sort(key=lambda e: (e.a, e.b))
    table = CorrespondenceTable(entries)
    table.regroup()
    return table


class SurfaceRegistry:
    """Persistent surface ids over union-find components of patch keys."""

    def __init__(self, trust: float = 0.5):
        self.trust = float(trust)
        self.surfaces: dict[str, frozenset[str]] = {}
        self.counter = 0

    def surface_of(self, key: str) -> str | None:
        for sid, members in self.surfaces.items():
            if key in members:
                return sid
        return None

    def members(self, sid: str) -> frozenset[str]:
        return self.surfaces.get(sid, frozenset())

    def rebuild(self, all_keys: set[str],
                unions: list[tuple[str, str]]) -> dict:
        """Recompute components and hand out ids by trusted inheritance."""
        parent = {k: k for k in all_keys}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x, y in unions:
            if x in parent and y in parent:
                parent[find(x)] = find(y)

        comps: dict[str, set[str]] = {}
        for k in all_keys:
            comps.setdefault(find(k), set()).add(k)
        components = sorted(comps.values(), key=lambda c: min(c))

        # Greedy best-first inheritance by Jaccard overlap.
        scored: list[tuple[float, int, str]] = []
        for ci, comp in enumerate(components):
            for sid, members in self.surfaces.items():
                inter = len(comp & members)
                if not inter:
                    continue
                j = inter / len(comp | members)
                scored.append((j, ci, sid))
        scored.sort(key=lambda t: (-t[0], min(components[t[1]]), t[2]))
        assigned: dict[int, str] = {}
        used_sids: set[str] = set()
        for j, ci, sid in scored:
            if j < self.trust or ci in assigned or sid in used_sids:
                continue
            assigned[ci] = sid
            used_sids.add(sid)

        previous = self.surfaces
        new_surfaces: dict[str, frozenset[str]] = {}
        for ci, comp in enumerate(components):
            sid = assigned.get(ci)
            if sid is None:
                sid = f"S{self.counter}"
                self.counter += 1
            new_surfaces[sid] = frozenset(comp)
        self.surfaces = new_surfaces
        return {
            "inherited": sorted(used_sids),
            "created": sorted(s for s in new_surfaces if s not in used_sids),
            "vanished": sorted(s for s in previous if s not in new_surfaces),
        }

    def to_dict(self) -> dict:
        return {"trust": self.trust, "counter": self.counter,
                "surfaces": {sid: sorted(members)
                             for sid, members in self.surfaces.items()}}

    @staticmethod
    def from_dict(d: dict) -> "SurfaceRegistry":
        reg = SurfaceRegistry(trust=float(d.get("trust", 0.5)))
        reg.counter = int(d.get("counter", 0))
        reg.surfaces = {sid: frozenset(members)
                        for sid, members in d.get("surfaces", {}).items()}
        return reg
