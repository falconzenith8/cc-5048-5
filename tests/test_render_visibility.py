"""Visibility semantics shared by the software and GL renderers.

The software rasterizer is used here because CI has no GPU; it implements the
same per-pixel depth test as the GL backend.  The key invariant: rotating to
the back must reveal points hidden from the front, i.e. the two images must
differ substantially rather than being mirrored duplicates of visible points.
"""

import numpy as np

from volumeview.frame import FrameFile
from volumeview.software_render import SoftwareRenderer


def _render(assets, frame_index, yaw, depth=False):
    with FrameFile(assets["vpc"]) as ff:
        frame = ff.read_frame(frame_index)
    r = SoftwareRenderer(320, 240)
    r.camera.yaw = yaw
    r.depth_mode = depth
    return r.render(frame)


def test_front_and_back_views_differ(assets):
    front = _render(assets, 72, 0.0)
    back = _render(assets, 72, np.pi)
    lit_front = np.count_nonzero(front.sum(axis=2))
    lit_back = np.count_nonzero(back.sum(axis=2))
    # Both views show the actor, but occlusion makes them genuinely different.
    assert lit_front > 100 and lit_back > 100
    diff = np.count_nonzero(np.any(front != back, axis=2))
    assert diff > 0.25 * max(lit_front, lit_back)


def test_depth_buffer_reveals_occluded_points_from_behind():
    # Unit-level proof of the exact z-test shared by both backends
    # (GL_LESS / strictly smaller NDC depth wins).  Near and far fragments
    # compete for one pixel in both draw orders; then the depth order is
    # reversed as an orbit behind the actor would do.
    red = np.array([[255, 0, 0]], np.uint8)
    green = np.array([[0, 255, 0]], np.uint8)
    px = py = np.array([32], np.int32)

    fb = np.zeros((64, 64, 3), np.uint8)
    zb = np.full((64, 64), np.inf, np.float32)
    SoftwareRenderer._blend(fb, zb, px, py, np.array([0.8]), green)
    SoftwareRenderer._blend(fb, zb, px, py, np.array([0.4]), red)
    assert tuple(fb[32, 32]) == (255, 0, 0)       # near red overwrites far
    assert zb[32, 32] == 0.4

    # Far fragment drawn later must be rejected by the stored near depth.
    SoftwareRenderer._blend(fb, zb, px, py, np.array([0.9]), green)
    assert tuple(fb[32, 32]) == (255, 0, 0)

    # Orbit behind: what was far is now near, so green wins the same test.
    fb2 = np.zeros((64, 64, 3), np.uint8)
    zb2 = np.full((64, 64), np.inf, np.float32)
    SoftwareRenderer._blend(fb2, zb2, px, py, np.array([0.8]), red)
    SoftwareRenderer._blend(fb2, zb2, px, py, np.array([0.4]), green)
    assert tuple(fb2[32, 32]) == (0, 255, 0)


def test_depth_mode_outputs_grayscale(assets):
    img = _render(assets, 72, 0.0, depth=True)
    lit = img[img.sum(axis=2) > 0]
    # Grayscale means R == G == B on lit pixels.
    assert np.all(lit[:, 0] == lit[:, 1]) and np.all(lit[:, 1] == lit[:, 2])
