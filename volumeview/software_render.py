"""Headless numpy software rasterizer.

This backend exists so the whole player is verifiable without a GPU or display
(CI, containers).  It implements the same visibility rule as the OpenGL
backend: a per-pixel 32-bit depth buffer, nearer fragments overwrite farther
ones.  When the camera swings behind the actor, points that were occluded from
the front are therefore correctly revealed.

Two view modes:

* ``color`` - per-point RGB;
* ``depth`` - grayscale eye-space depth, separately for the screen and the
  actor, making front/back depth easy to compare.
"""

from __future__ import annotations

import numpy as np

from .camera import OrbitCamera
from .scene import SCREEN_HALF_HEIGHT, SCREEN_HALF_WIDTH, SCREEN_Z


class SoftwareRenderer:
    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.camera = OrbitCamera()
        self.depth_mode = False
        self.point_size = 2
        self.show_screen = True

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = width, height

    def _project(self, world: np.ndarray, vp: np.ndarray):
        n = world.shape[0]
        homog = np.concatenate(
            [world, np.ones((n, 1), dtype=np.float32)], axis=1) @ vp.T
        w = homog[:, 3]
        valid = np.abs(w) > 1e-6
        ndc = homog[:, :3] / w[:, None]
        sx = ((ndc[:, 0] + 1.0) * 0.5 * self.width).astype(np.int32)
        sy = ((1.0 - ndc[:, 1]) * 0.5 * self.height).astype(np.int32)
        depth = ndc[:, 2]
        return sx, sy, depth, valid

    def _raster_points(self, framebuffer, zbuffer, world, colors,
                       vp, depth_shade: bool, eye_z: np.ndarray | None = None,
                       pixel_radius: int | None = None):
        if world.shape[0] == 0:
            return
        sx, sy, depth, valid = self._project(world, vp)
        r = self.point_size if pixel_radius is None else pixel_radius
        for ox in range(-r, r + 1):
            for oy in range(-r, r + 1):
                px = sx + ox
                py = sy + oy
                ok = valid & (px >= 0) & (px < self.width) & \
                     (py >= 0) & (py < self.height)
                idx = np.nonzero(ok)[0]
                if idx.size == 0:
                    continue
                if depth_shade:
                    z = eye_z[idx]
                    t = np.clip((z - 1.5) / 6.0, 0.0, 1.0)
                    shade = (235.0 * (1.0 - t)).astype(np.uint8)
                    rgb = np.repeat(shade[:, None], 3, axis=1)
                else:
                    rgb = colors[idx]
                self._blend(framebuffer, zbuffer, px[idx], py[idx],
                            depth[idx], rgb)

    @staticmethod
    def _blend(framebuffer, zbuffer, px, py, depth, rgb):
        # Vectorized z-test with deterministic tie handling.
        order = np.argsort(-depth)  # far to near so near wins last
        for i in order:
            x, y, z = int(px[i]), int(py[i]), float(depth[i])
            if z < zbuffer[y, x]:
                zbuffer[y, x] = z
                framebuffer[y, x] = rgb[i]

    def render(self, frame) -> np.ndarray:
        aspect = self.width / self.height
        vp = self.camera.view_projection(aspect)
        framebuffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        zbuffer = np.full((self.height, self.width), np.inf, dtype=np.float32)

        # Translucent paper screen: rasterize as dim quads per pixel column
        # with a fixed depth, drawn before the actor so actor points win the
        # z-test whenever they are in front of it.
        if self.show_screen:
            self._draw_screen(framebuffer, zbuffer, vp)

        if frame is not None:
            world = frame.world_points
            eye = world - self.camera.eye()
            eye_z = np.linalg.norm(eye, axis=1)
            self._raster_points(framebuffer, zbuffer, world, frame.colors,
                                vp, self.depth_mode, eye_z)
        return framebuffer

    def _draw_screen(self, framebuffer, zbuffer, vp):
        h, w = SCREEN_HALF_HEIGHT, SCREEN_HALF_WIDTH
        xs = np.linspace(-w, w, 60)
        ys = np.linspace(-h, h, 60)
        grid = np.array([[x, y, SCREEN_Z] for y in ys for x in xs],
                        dtype=np.float32)
        sx, sy, depth, valid = self._project(grid, vp)
        if self.depth_mode:
            eye = grid - self.camera.eye()
            eye_z = np.linalg.norm(eye, axis=1)
            t = np.clip((eye_z - 1.5) / 6.0, 0.0, 1.0)
            shade = (200.0 * (1.0 - t)).astype(np.uint8)
            paper = np.repeat(shade[:, None], 3, axis=1)
        else:
            paper = np.tile(np.array([70, 80, 95], dtype=np.uint8),
                            (grid.shape[0], 1))
        ok = valid & (sx >= 0) & (sx < self.width) & \
             (sy >= 0) & (sy < self.height)
        idx = np.nonzero(ok)[0]
        # Screen only writes where nothing nearer exists; a small z-bias lets
        # actor points win the test from either side of the plane.
        for i in idx:
            x, y, z = int(sx[i]), int(sy[i]), float(depth[i]) + 1e-3
            if z < zbuffer[y, x]:
                zbuffer[y, x] = z
                framebuffer[y, x] = paper[i]
        self._draw_screen_border(framebuffer, zbuffer, vp, h, w)

    def _draw_screen_border(self, framebuffer, zbuffer, vp, h, w):
        t = np.linspace(0.0, 1.0, 80)
        edges = [
            np.stack([-w + 2 * w * t, np.full_like(t, -h),
                      np.full_like(t, SCREEN_Z)], axis=1),
            np.stack([-w + 2 * w * t, np.full_like(t, h),
                      np.full_like(t, SCREEN_Z)], axis=1),
            np.stack([np.full_like(t, -w), -h + 2 * h * t,
                      np.full_like(t, SCREEN_Z)], axis=1),
            np.stack([np.full_like(t, w), -h + 2 * h * t,
                      np.full_like(t, SCREEN_Z)], axis=1),
        ]
        border = np.concatenate(edges, axis=0).astype(np.float32)
        sx, sy, depth, valid = self._project(border, vp)
        if self.depth_mode:
            color = np.array([110, 110, 110], dtype=np.uint8)
        else:
            color = np.array([150, 170, 200], dtype=np.uint8)
        ok = valid & (sx >= 0) & (sx < self.width) & \
             (sy >= 0) & (sy < self.height)
        for i in np.nonzero(ok)[0]:
            x, y, z = int(sx[i]), int(sy[i]), float(depth[i])
            if z < zbuffer[y, x]:
                zbuffer[y, x] = z
                framebuffer[y, x] = color

    def save_png(self, path: str, frame) -> None:
        from PIL import Image
        Image.fromarray(self.render(frame)).save(path)
