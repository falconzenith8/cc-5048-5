"""Orbit camera shared by the OpenGL and software renderers."""

from __future__ import annotations

import math

import numpy as np


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


class OrbitCamera:
    """Spherical orbit around the screen crossing point.

    Yaw is unclamped so the user can swing fully behind the actor; pitch is
    limited to avoid gimbal flip.
    """

    def __init__(self, distance: float = 4.2, yaw: float = 0.0,
                 pitch: float = 0.12, target=(0.0, 0.0, 0.0)):
        self.distance = distance
        self.yaw = yaw
        self.pitch = pitch
        self.target = np.asarray(target, dtype=np.float32)

    def eye(self) -> np.ndarray:
        cp = math.cos(self.pitch)
        return self.target + np.array([
            self.distance * cp * math.sin(self.yaw),
            self.distance * math.sin(self.pitch),
            self.distance * cp * math.cos(self.yaw),
        ], dtype=np.float32)

    def view(self) -> np.ndarray:
        eye = self.eye()
        forward = self.target - eye
        forward /= np.linalg.norm(forward)
        # World up projected orthogonally; the fallback handles the rare
        # alignment where forward and world up become parallel.
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(world_up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = right
        m[1, :3] = up
        m[2, :3] = -forward
        m[:3, 3] = -m[:3, :3] @ eye
        return m

    def view_projection(self, aspect: float) -> np.ndarray:
        return perspective(50.0, aspect, 0.05, 50.0) @ self.view()

    def drag(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.008
        self.pitch = float(np.clip(self.pitch - dy * 0.008, -1.3, 1.3))

    def zoom(self, delta: float) -> None:
        self.distance = float(np.clip(self.distance * (1.0 + delta * 0.1),
                                      1.5, 12.0))
