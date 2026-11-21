"""Container, variable-layout and ordering guarantees."""

import json

import numpy as np

from volumeview.frame import FrameFile, FrameWriter, Frame, encode_frame, decode_frame
from volumeview.streaming import OutOfOrderNetwork


def test_container_roundtrip(tmp_path):
    path = tmp_path / "f.vpc"
    rng = np.random.default_rng(1)
    frames = []
    with FrameWriter(str(path)) as w:
        for i in range(5):
            n = 10 + i * 7
            f = Frame(pts=i * 0.5, origin=(float(i), 0.0, 0.0),
                      points=rng.normal(size=(n, 3)).astype(np.float32),
                      colors=rng.integers(0, 255, (n, 3), np.uint8))
            frames.append(f)
            w.write(f)
    with FrameFile(str(path)) as ff:
        assert len(ff) == 5
        for i, f in enumerate(frames):
            got = ff.read_frame(i)
            assert got.pts == f.pts
            assert got.npoints == f.npoints
            np.testing.assert_array_equal(got.points, f.points)
            np.testing.assert_array_equal(got.colors, f.colors)
            np.testing.assert_allclose(got.origin, f.origin, rtol=1e-6)


def test_variable_point_count_and_layout(assets):
    with FrameFile(assets["vpc"]) as ff:
        counts = [ff.index[i].npoints for i in range(len(ff))]
        assert min(counts) < 1000 < max(counts)
        # Heavy frames are genuinely larger than the small frame after them.
        assert ff.index[43].npoints > 3 * ff.index[44].npoints
        f43, f44 = ff.read_frame(43), ff.read_frame(44)
        assert f43.npoints != f44.npoints
        # Independently shuffled layouts: no implied point correspondence.
        assert f43.points.shape != f44.points.shape or not np.array_equal(
            f43.points[:50], f44.points[:50])


def test_heavy_frame_arrives_after_smaller_successor(assets):
    plan = json.load(open(assets["plan"]))
    net = OutOfOrderNetwork()
    d43, d44 = net.delay_for(43), net.delay_for(44)
    assert d43 > d44
    # Requested in index order at the same wall instant, 43 becomes ready last.
    assert d43 - d44 > 0.3
    # The shipped replay plan encodes the same guarantee.
    entries = {e["frame"]: e for e in plan["frames"]}
    assert entries[43]["injected_delay_s"] > entries[44]["injected_delay_s"]
    assert entries[43]["npoints"] > 3 * entries[44]["npoints"]


def test_encode_decode_in_memory():
    rng = np.random.default_rng(3)
    pts = rng.normal(size=(20, 3)).astype(np.float32)
    col = rng.integers(0, 255, (20, 3), np.uint8)
    blob = encode_frame(1.25, (0.1, 0.2, 0.3), pts, col)
    frame, consumed = decode_frame(blob)
    assert consumed == len(blob)
    np.testing.assert_array_equal(frame.points, pts)
    assert frame.pts == 1.25
