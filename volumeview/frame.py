"""On-disk volumetric frame container.

A .vpc file is a sequence of independently decodable frames.  Every frame
carries its own capture timestamp (pts, seconds), capture origin and a
variable number of points; both the point count and the point ordering may
change between frames.  Frame payloads are zlib compressed so the read /
decode pipeline has real work to schedule.

Container layout (little endian)::

    magic     b'VPC1'            4 bytes
    version   uint8              1
    flags     uint8              1
    reserved  uint16             2
    pts       float64            8   capture time in seconds
    origin    float32 x 3       12   capture origin in world space
    npoints   uint32             4
    payload   uint32             4   compressed byte count, then bytes

Compressed payload decompresses to::

    points    float32 (n * 3)    positions relative to origin
    colors    uint8  (n * 3)     per-point linear RGB
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

MAGIC = b"VPC1"
VERSION = 1
_HEADER = struct.Struct("<4sBBHd3fII")
HEADER_SIZE = _HEADER.size
IDX_MAGIC = b"VIDX"


@dataclass(frozen=True)
class Frame:
    """One decoded volumetric frame."""

    pts: float
    origin: tuple[float, float, float]
    points: np.ndarray  # float32 (n, 3), positions relative to origin
    colors: np.ndarray  # uint8 (n, 3)

    @property
    def npoints(self) -> int:
        return int(self.points.shape[0])

    @property
    def world_points(self) -> np.ndarray:
        return self.points + np.asarray(self.origin, dtype=np.float32)

    @property
    def decoded_bytes(self) -> int:
        return self.points.nbytes + self.colors.nbytes


@dataclass(frozen=True)
class IndexEntry:
    offset: int
    pts: float
    npoints: int
    payload_bytes: int


def encode_frame(pts: float, origin, points: np.ndarray, colors: np.ndarray) -> bytes:
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3)")
    if colors.shape != points.shape:
        raise ValueError("colors must match points shape (n, 3)")
    raw = points.tobytes() + colors.tobytes()
    payload = zlib.compress(raw, level=6)
    return _HEADER.pack(
        MAGIC, VERSION, 0, 0, float(pts),
        float(origin[0]), float(origin[1]), float(origin[2]),
        points.shape[0], len(payload),
    ) + payload


def decode_frame(buffer: bytes, offset: int = 0) -> tuple[Frame, int]:
    (magic, version, _flags, _reserved, pts,
     ox, oy, oz, npoints, payload_len) = _HEADER.unpack_from(buffer, offset)
    if magic != MAGIC:
        raise ValueError(f"bad frame magic at offset {offset}: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported container version {version}")
    start = offset + HEADER_SIZE
    raw = zlib.decompress(buffer[start:start + payload_len])
    pts_nbytes = npoints * 3 * 4
    points = np.frombuffer(raw[:pts_nbytes], dtype=np.float32).reshape(npoints, 3)
    colors = np.frombuffer(raw[pts_nbytes:pts_nbytes + npoints * 3],
                           dtype=np.uint8).reshape(npoints, 3)
    frame = Frame(pts, (ox, oy, oz), points.copy(), colors.copy())
    return frame, start + payload_len


class FrameFile:
    """Random access reader over a .vpc file plus its .vpc.idx sidecar."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        self.index: list[IndexEntry] = self._read_index(path + ".idx")

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "FrameFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _read_index(index_path: str) -> list[IndexEntry]:
        with open(index_path, "rb") as fh:
            data = fh.read()
        magic, count = struct.unpack_from("<4sI", data, 0)
        if magic != IDX_MAGIC:
            raise ValueError(f"bad index magic: {magic!r}")
        entry = struct.Struct("<QdII")
        out = []
        pos = 8
        for _ in range(count):
            offset, pts, npoints, payload_bytes = entry.unpack_from(data, pos)
            out.append(IndexEntry(offset, pts, npoints, payload_bytes))
            pos += entry.size
        return out

    @property
    def timestamps(self) -> np.ndarray:
        return np.fromiter((e.pts for e in self.index), dtype=np.float64)

    def read_frame(self, frame_index: int) -> Frame:
        entry = self.index[frame_index]
        self._fh.seek(entry.offset)
        header = self._fh.read(HEADER_SIZE)
        blob = header + self._fh.read(entry.payload_bytes)
        frame, _ = decode_frame(blob, 0)
        return frame

    def read_compressed(self, frame_index: int) -> tuple[IndexEntry, bytes]:
        """Read header + compressed payload without decoding (network step)."""
        entry = self.index[frame_index]
        self._fh.seek(entry.offset)
        return entry, self._fh.read(HEADER_SIZE + entry.payload_bytes)


def write_index(index_path: str, entries: list[IndexEntry]) -> None:
    entry = struct.Struct("<QdII")
    with open(index_path, "wb") as fh:
        fh.write(struct.pack("<4sI", IDX_MAGIC, len(entries)))
        for e in entries:
            fh.write(entry.pack(e.offset, e.pts, e.npoints, e.payload_bytes))


class FrameWriter:
    """Streams frames to a .vpc file and builds the sidecar index."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "wb")
        self.index: list[IndexEntry] = []

    def write(self, frame: Frame) -> None:
        blob = encode_frame(frame.pts, frame.origin, frame.points, frame.colors)
        offset = self._fh.tell()
        self._fh.write(blob)
        payload_len = len(blob) - HEADER_SIZE
        self.index.append(
            IndexEntry(offset, frame.pts, frame.npoints, payload_len))

    def close(self) -> None:
        self._fh.close()
        write_index(self.path + ".idx", self.index)

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
