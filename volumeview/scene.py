"""Procedural volumetric capture of an actor walking through a paper screen.

The scene is fully deterministic (seeded).  Between two drum beats the actor
walks along the camera-facing depth axis and crosses a vertical screen plane
at z = 0.  Points in front of the screen are tinted warmly and points behind
it coolly, so the front/back depth split is visible head-on and reverses when
the camera orbits behind the actor.

Every frame has a different point count and a deliberately shuffled point
order: downstream code must never assume frame i and frame i+1 share
point identity or layout.
"""

from __future__ import annotations

import math

import numpy as np

from .frame import Frame

# Screen is an xy plane facing the default camera; actor crosses along z.
SCREEN_Z = 0.0
SCREEN_HALF_HEIGHT = 1.35
SCREEN_HALF_WIDTH = 1.05


def _actor_center(pts: float, beat_a: float, beat_b: float) -> np.ndarray:
    """Linear walk through the screen between two beats, with a bob."""
    span = max(beat_b - beat_a, 1e-6)
    u = np.clip((pts - beat_a) / span, 0.0, 1.0)
    z = 1.4 - 2.8 * u            # far (behind) -> near (in front)
    y = 0.05 * math.sin(u * math.pi * 4.0)
    x = 0.12 * math.sin(u * math.pi * 2.0)
    return np.array([x, y, z], dtype=np.float32)


def _varying_point_count(rng: np.random.Generator, frame_no: int) -> int:
    """Big deliberate spike: a few frames are much larger than their neighbors."""
    base = 900 + int(180 * math.sin(frame_no * 0.9))
    jitter = int(rng.integers(-90, 90))
    if frame_no in (42, 43):  # the heavy late-arriving capture
        base = 4200
    return max(350, base + jitter)


def make_frame(pts: float, frame_no: int, beat_a: float, beat_b: float,
               seed: int = 20261120) -> Frame:
    rng = np.random.default_rng(seed + frame_no)
    center = _actor_center(pts, beat_a, beat_b)
    n = _varying_point_count(rng, frame_no)

    # Volumetric body: capsule-ish cloud, columns sampled independently so
    # the silhouette is taller than it is wide.
    px = rng.normal(0.0, 0.26, n).astype(np.float32)
    py = rng.normal(0.0, 0.42, n).astype(np.float32)
    pz = rng.normal(0.0, 0.20, n).astype(np.float32)

    # Head and two swinging arms keep the silhouette human.
    head_mask = rng.random(n) < 0.18
    py[head_mask] += 0.62
    px[head_mask] *= 0.55
    pz[head_mask] *= 0.55

    arm_swing = 0.35 * math.sin((pts - beat_a) * 6.0)
    arm_mask = rng.random(n) < 0.16
    side = np.where(rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
    px[arm_mask] += side[arm_mask] * (0.34 + 0.05 * rng.normal(size=arm_mask.sum()))
    py[arm_mask] += arm_swing * side[arm_mask]

    # Step phase drives a slight lean as the actor passes the screen.
    u = np.clip((pts - beat_a) / max(beat_b - beat_a, 1e-6), 0.0, 1.0)
    pz += 0.12 * (u - 0.5)

    points = np.stack([px, py, pz], axis=1)

    # Per-point color depends on world depth relative to the screen, so the
    # front/back tint travels with the actor as it crosses the plane.
    # Tint is computed in absolute space before points become origin-relative.
    world_z = points[:, 2] + center[2]
    front = np.clip((SCREEN_Z - world_z) * 2.2 + 0.5, 0.0, 1.0)
    colors = np.empty((n, 3), dtype=np.uint8)
    # Cool blue behind the screen, warm amber in front of it.
    colors[:, 0] = (60 + 180 * front).astype(np.uint8)
    colors[:, 1] = (110 + 90 * front).astype(np.uint8)
    colors[:, 2] = (200 - 150 * front).astype(np.uint8)
    # Slight per-point luminance variation, then shuffle layout on purpose.
    lum = rng.integers(-18, 18, size=(n, 1))
    colors = np.clip(colors.astype(np.int16) + lum, 0, 255).astype(np.uint8)
    perm = rng.permutation(n)
    points = np.ascontiguousarray(points[perm])
    colors = np.ascontiguousarray(colors[perm])

    origin = (float(center[0]), float(center[1]), float(center[2]))
    # Store relative to origin so the moving rig origin is meaningful.
    rel = points - center
    return Frame(pts=pts, origin=origin,
                 points=np.ascontiguousarray(rel.astype(np.float32)),
                 colors=colors)
