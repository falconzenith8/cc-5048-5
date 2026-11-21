"""Synthetic captures with ground truth for the tween layer.

Two deterministic scenes:

* :class:`TurnScene` - one actor turns 180 degrees in place while an arm
  sweeps across the torso, so the visible surface and the captured point
  count change frame to frame (occlusion and re-reveal).  Ships with dense
  surface ground truth, per-frame depth maps from the capture view and a
  side view, and a drum-contact event whose time is known exactly.
* :class:`SwapScene` - two look-alike actors in parallel lanes swap
  positions; a static pillar anchors the background.  Geometry-only
  correspondence wrongly connects the two actors across the crossing,
  which is the case the editor repairs by vetoing one correspondence
  group.  Ground truth actor labels are stored alongside.

Both scenes are fully seeded; rebuilding them yields identical bytes.
"""

from __future__ import annotations

import json

import numpy as np

from .camera import OrbitCamera
from .frame import Frame

TURN_FRAMES = 21
TURN_FPS = 12.0
SWAP_FRAMES = 17
SWAP_FPS = 12.0
DEPTH_W, DEPTH_H = 192, 144

FRONT_CAM = {"distance": 4.0, "yaw": 0.0, "pitch": 0.05,
             "target": (0.0, 0.25, 0.0)}
SIDE_CAM = {"distance": 4.0, "yaw": float(np.pi / 2), "pitch": 0.05,
            "target": (0.0, 0.25, 0.0)}


def make_camera(spec: dict) -> OrbitCamera:
    return OrbitCamera(distance=spec["distance"], yaw=spec["yaw"],
                       pitch=spec["pitch"], target=spec["target"])


def project_points(world: np.ndarray, camera: OrbitCamera,
                   width: int, height: int):
    """Pinhole projection shared with the software renderer."""
    vp = camera.view_projection(width / height)
    n = world.shape[0]
    homog = np.concatenate(
        [world, np.ones((n, 1), dtype=np.float32)], axis=1) @ vp.T
    w = homog[:, 3]
    valid = np.abs(w) > 1e-6
    ndc = homog[:, :3] / w[:, None]
    sx = ((ndc[:, 0] + 1.0) * 0.5 * width).astype(np.int32)
    sy = ((1.0 - ndc[:, 1]) * 0.5 * height).astype(np.int32)
    return sx, sy, ndc[:, 2].astype(np.float32), valid


def depth_map(world: np.ndarray, camera: OrbitCamera, width: int = DEPTH_W,
              height: int = DEPTH_H, splat: int = 1) -> np.ndarray:
    """NDC-depth ground truth: nearest projected point per pixel (inf if
    empty).  ``splat`` grows each point to fill sampling gaps."""
    depth = np.full((height, width), np.inf, dtype=np.float32)
    sx, sy, nd, valid = project_points(world, camera, width, height)
    for ox in range(-splat, splat + 1):
        for oy in range(-splat, splat + 1):
            px, py = sx + ox, sy + oy
            ok = valid & (px >= 0) & (px < width) & (py >= 0) & (py < height)
            idx = np.nonzero(ok)[0]
            if idx.size == 0:
                continue
            x, y, z = px[idx], py[idx], nd[idx]
            cur = depth[y, x]
            better = z < cur
            depth[y[better], x[better]] = z[better]
    return depth


def visible_indices(world: np.ndarray, camera: OrbitCamera,
                    width: int = DEPTH_W, height: int = DEPTH_H,
                    slack: float = 0.03) -> np.ndarray:
    """Indices of points on the camera-visible surface shell (metric)."""
    sx, sy, _nd, valid = project_points(world, camera, width, height)
    eye_dist = np.linalg.norm(world - camera.eye(), axis=1)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    ok = valid & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
    idx = np.nonzero(ok)[0]
    cur = zbuf[sy[idx], sx[idx]]
    better = eye_dist[idx] < cur
    zbuf[sy[idx][better], sx[idx][better]] = eye_dist[idx][better]
    front = zbuf[sy[idx], sx[idx]]
    return idx[eye_dist[idx] <= front + slack]


def _sample_sphere(rng, center, radius, n):
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return np.asarray(center, np.float32) + radius * dirs.astype(np.float32)


def _sample_capsule(rng, a, b, radius, n):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    axis = b - a
    length = float(np.linalg.norm(axis))
    axis /= length
    ref = np.array([0.0, 0.0, 1.0], np.float32)
    if abs(axis[2]) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], np.float32)
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    cyl_frac = length / (length + 2 * radius)
    n_cyl = int(n * cyl_frac)
    n_cap = (n - n_cyl) // 2
    t = rng.random(n_cyl).astype(np.float32)
    theta = (rng.random(n_cyl) * 2 * np.pi).astype(np.float32)
    ring = (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    cyl = a + t[:, None] * (b - a) + radius * ring
    caps = np.concatenate([_sample_sphere(rng, a, radius, n_cap),
                           _sample_sphere(rng, b, radius,
                                          n - n_cyl - n_cap)], axis=0)
    return np.concatenate([cyl, caps], axis=0).astype(np.float32)


class TurnScene:
    """Actor turning in place with an arm sweep, plus depth ground truth."""

    def __init__(self, n_frames: int = TURN_FRAMES, fps: float = TURN_FPS,
                 seed: int = 20261121):
        self.n = n_frames
        self.fps = fps
        self.seed = seed
        self.contact_threshold = 0.12

    # -- actor geometry -------------------------------------------------
    def _theta(self, f: int) -> float:
        return float(np.pi * f / (self.n - 1))

    def _psi(self, f: float) -> float:
        return float(-0.9 + 1.8 * f / (self.n - 1))

    def hand_tip_local(self, f: float) -> np.ndarray:
        shoulder = np.array([0.30, 0.45, 0.02], np.float32)
        psi = self._psi(f)
        return shoulder + 0.58 * np.array(
            [np.sin(psi), -0.15, np.cos(psi)], np.float32)

    def _actor_local(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + 5000)
        parts = [
            (_sample_capsule(rng, (0.0, -0.45, 0.0), (0.0, 0.55, 0.0),
                             0.24, 2400), (205, 175, 150)),   # torso
            (_sample_sphere(rng, (0.0, 0.82, 0.0), 0.16, 800),
             (220, 195, 175)),                                  # head
            (_sample_capsule(rng, (0.30, 0.45, 0.02),
                             self.hand_tip_local(f), 0.07, 1000),
             (190, 155, 135)),                                  # sweeping arm
        ]
        clouds, colors = [], []
        for pts, rgb in parts:
            clouds.append(pts)
            jitter = rng.integers(-12, 13, size=(pts.shape[0], 1))
            col = np.clip(np.asarray(rgb, np.int16) + jitter, 0, 255)
            colors.append(col.astype(np.uint8))
        return (np.concatenate(clouds, axis=0).astype(np.float32),
                np.concatenate(colors, axis=0))

    def dense_world(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        local, colors = self._actor_local(f)
        th = self._theta(f)
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                       np.float32)
        return (local @ rot.T).astype(np.float32), colors

    # -- captures ---------------------------------------------------------
    def captured_frame(self, f: int) -> Frame:
        world, colors = self.dense_world(f)
        cam = make_camera(FRONT_CAM)
        vis = visible_indices(world, cam)
        rng = np.random.default_rng(self.seed + f)
        noise = rng.normal(scale=0.004, size=(vis.size, 3)).astype(np.float32)
        pts = world[vis] + noise
        col = colors[vis]
        perm = rng.permutation(vis.size)
        pts, col = np.ascontiguousarray(pts[perm]), np.ascontiguousarray(
            col[perm])
        origin = pts.mean(axis=0).astype(np.float32)
        return Frame(pts=round(f / self.fps, 6),
                     origin=tuple(float(v) for v in origin),
                     points=np.ascontiguousarray(pts - origin), colors=col)

    def frames(self) -> dict[int, Frame]:
        return {f: self.captured_frame(f) for f in range(self.n)}

    # -- ground truth -------------------------------------------------------
    def contact_event(self) -> dict:
        """Drum position and exact contact time of the sweeping hand."""
        pos = (self.dense_world(10)[0].mean(axis=0) * 0
               + (self._rotate(self.hand_tip_local(10), 10)
                  + self._rotate(self.hand_tip_local(11), 11)) * 0.5)
        pos = pos.astype(np.float32)
        times, weights = [], []
        for f in range(self.n):
            dense, _ = self.dense_world(f)
            d = float(np.sqrt(((dense - pos) ** 2).sum(axis=1)).min())
            if d < self.contact_threshold:
                times.append(f / self.fps)
                weights.append(self.contact_threshold - d)
        t = float(np.average(times, weights=weights)) if times else None
        return {"position": [float(v) for v in pos], "time": t,
                "threshold": self.contact_threshold}

    def _rotate(self, local: np.ndarray, f: int) -> np.ndarray:
        th = self._theta(f)
        c, s = np.cos(th), np.sin(th)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
                       np.float32)
        return local @ rot.T

    def ground_truth(self) -> dict:
        front = make_camera(FRONT_CAM)
        side = make_camera(SIDE_CAM)
        dense_all = np.zeros((self.n, 4200, 3), np.float32)
        colors_all = np.zeros((self.n, 4200, 3), np.uint8)
        depth_front = np.zeros((self.n, DEPTH_H, DEPTH_W), np.float32)
        depth_side = np.zeros((self.n, DEPTH_H, DEPTH_W), np.float32)
        for f in range(self.n):
            world, colors = self.dense_world(f)
            dense_all[f] = world
            colors_all[f] = colors
            depth_front[f] = depth_map(world, front)
            depth_side[f] = depth_map(world, side)
        return {
            "dense": dense_all, "dense_colors": colors_all,
            "depth_front": depth_front, "depth_side": depth_side,
            "contact": self.contact_event(),
            "front_cam": dict(FRONT_CAM), "side_cam": dict(SIDE_CAM),
            "fps": self.fps,
        }

    def save_ground_truth(self, path: str) -> None:
        gt = self.ground_truth()
        np.savez_compressed(
            path, dense=gt["dense"], dense_colors=gt["dense_colors"],
            depth_front=gt["depth_front"], depth_side=gt["depth_side"],
            contact=json.dumps(gt["contact"]),
            front_cam=json.dumps(gt["front_cam"]),
            side_cam=json.dumps(gt["side_cam"]), fps=gt["fps"])


class SwapScene:
    """Two look-alike actors swapping positions in parallel lanes."""

    LANE_Z = 0.35
    PILLAR_X = 0.2
    PILLAR_Z = 0.95

    def __init__(self, n_frames: int = SWAP_FRAMES, fps: float = SWAP_FPS,
                 seed: int = 20261122):
        self.n = n_frames
        self.fps = fps
        self.seed = seed

    def actor_x(self, actor: int, f: int) -> float:
        # Solved so the matcher fails across region (10, 13): A's start is
        # closest to B's end, so the greedy matcher links A->B.
        if actor == 0:
            return -0.4 + (f - 10) * (1.7 / 3.0)
        return 0.7 - (f - 10) * 0.3

    def actor_center(self, actor: int, f: int) -> np.ndarray:
        z = self.LANE_Z if actor == 0 else -self.LANE_Z
        return np.array([self.actor_x(actor, f), 0.1, z], np.float32)

    def _actor_cloud(self, actor: int, f: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + 100 * actor)
        body = _sample_capsule(rng, (0.0, -0.4, 0.0), (0.0, 0.5, 0.0),
                               0.2, 520)
        head = _sample_sphere(rng, (0.0, 0.75, 0.0), 0.14, 180)
        pts = np.concatenate([body, head], axis=0)
        base = np.array((200, 180, 170) if actor == 0 else (180, 185, 200),
                        np.int16)
        jitter = rng.integers(-8, 9, size=(pts.shape[0], 1))
        col = np.clip(base + jitter, 0, 255).astype(np.uint8)
        return pts.astype(np.float32) + self.actor_center(actor, f), col

    def _pillar(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + 900)
        n = 250
        pts = np.column_stack([
            rng.uniform(-0.12, 0.12, n),
            rng.uniform(-0.5, 1.1, n),
            rng.uniform(-0.12, 0.12, n)]).astype(np.float32)
        pts[:, 0] += self.PILLAR_X
        pts[:, 2] += self.PILLAR_Z
        col = np.clip(np.array((120, 200, 140), np.int16)
                      + rng.integers(-6, 7, size=(n, 1)), 0, 255)
        return pts, col.astype(np.uint8)

    def frame(self, f: int) -> tuple[Frame, np.ndarray]:
        clouds, colors, labels = [], [], []
        for actor in (0, 1):
            pts, col = self._actor_cloud(actor, f)
            clouds.append(pts)
            colors.append(col)
            labels.append(np.full(pts.shape[0], actor, np.uint8))
        pts, col = self._pillar()
        clouds.append(pts)
        colors.append(col)
        labels.append(np.full(pts.shape[0], 2, np.uint8))
        world = np.concatenate(clouds, axis=0)
        colour = np.concatenate(colors, axis=0)
        label = np.concatenate(labels, axis=0)
        rng = np.random.default_rng(self.seed + f)
        perm = rng.permutation(world.shape[0])
        world, colour, label = (world[perm], colour[perm], label[perm])
        frame = Frame(pts=round(f / self.fps, 6), origin=(0.0, 0.0, 0.0),
                      points=np.ascontiguousarray(world.astype(np.float32)),
                      colors=np.ascontiguousarray(colour))
        return frame, label

    def frames(self) -> dict[int, Frame]:
        return {f: self.frame(f)[0] for f in range(self.n)}

    def ground_truth(self) -> dict:
        labels = np.zeros((self.n, 1650), np.uint8)
        centroids = np.zeros((self.n, 3, 3), np.float32)
        for f in range(self.n):
            frame, lab = self.frame(f)
            labels[f] = lab
            for actor in (0, 1, 2):
                centroids[f, actor] = frame.points[lab == actor].mean(axis=0)
        return {"labels": labels, "centroids": centroids, "fps": self.fps}

    def save_ground_truth(self, path: str) -> None:
        gt = self.ground_truth()
        np.savez_compressed(path, labels=gt["labels"],
                            centroids=gt["centroids"], fps=gt["fps"])
