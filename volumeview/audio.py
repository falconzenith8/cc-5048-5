"""Reference drum track, beat map and the audio playback clock.

The audio output position is the single authority for playback time.  When no
sound device is available (CI / headless containers) a virtual clock driven by
wall time keeps the same interface so the whole pipeline stays testable.
"""

from __future__ import annotations

import json
import threading
import time
import wave
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 44100
BPM = 100.0
DURATION = 19.2  # 32 beats at 100 bpm


def build_beatmap(duration: float = DURATION, bpm: float = BPM) -> list[float]:
    period = 60.0 / bpm
    n = int(round(duration / period))
    return [round(i * period, 6) for i in range(n + 1)]


def write_beatmap(path: str, beats: list[float], bpm: float = BPM,
                  crossing: tuple[float, float] | None = None) -> None:
    if crossing is None:
        crossing = (beats[8], beats[12])
    payload = {
        "bpm": bpm,
        "sample_rate": SAMPLE_RATE,
        "beats": beats,
        "screen_crossing": {
            "beat_a": crossing[0],
            "beat_b": crossing[1],
            "screen_z": 0.0,
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def read_beatmap(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def synthesize_drums(beats: list[float], duration: float = DURATION,
                     sr: int = SAMPLE_RATE) -> np.ndarray:
    n = int(duration * sr)
    buf = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(7)
    for i, bt in enumerate(beats):
        start = int(bt * sr)
        length = int(0.28 * sr)
        end = min(start + length, n)
        if start >= n:
            continue
        tt = np.arange(end - start) / sr
        accent = 1.0 if i % 4 == 0 else 0.7
        # Pitch-dropping sine gives a kick; a short noise burst gives a hat.
        freq = 110.0 * np.exp(-tt * 28.0) + 45.0
        kick = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-tt * 9.0)
        hat = rng.standard_normal(len(tt)) * np.exp(-tt * 45.0) * 0.12
        buf[start:end] += accent * (kick + hat).astype(np.float32)
    peak = float(np.max(np.abs(buf))) or 1.0
    return (buf / peak * 0.85).astype(np.float32)


def write_wav(path: str, samples: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm16.tobytes())


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav:
        sr = wav.getframerate()
        raw = wav.readframes(wav.getnframes())
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sr


@dataclass
class ClockStatus:
    position: float
    playing: bool
    device: str          # "sounddevice" or "virtual"
    underrun_count: int


class AudioClock:
    """Timeline position derived from the output stream's own clock.

    The PortAudio callback pulls samples by timeline position, so audible
    content and ``position`` can never drift apart.  Pausing outputs silence
    without stopping the stream; seeking only moves an anchor under a lock.
    """

    def __init__(self, wav_path: str,
                 loop_window: tuple[float, float] | None = None):
        self._samples, self.sr = read_wav(wav_path)
        self.loop_window = loop_window
        self._lock = threading.Lock()
        self._anchor_pos = 0.0
        self._anchor_wall = time.monotonic()
        self._playing = False
        self._underruns = 0
        self.device = "virtual"
        self._stream = None
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=self.sr, channels=1, dtype="float32",
                callback=self._callback, latency="low")
            self._stream.start()
            self.device = "sounddevice"
        except Exception:
            self._stream = None  # headless: virtual clock only

    def _callback(self, out, frames, _time_info, status) -> None:
        if status.output_underflow:
            self._underruns += 1
        with self._lock:
            playing = self._playing
            anchor_pos = self._anchor_pos
            anchor_wall = self._anchor_wall
        if not playing:
            out.fill(0)
            return
        now = time.monotonic()
        start = int((anchor_pos + (now - anchor_wall)) * self.sr)
        if self.loop_window is not None:
            a, b = self.loop_window
            span = b - a
            if span > 0:
                start = int(a + ((start / self.sr - a) % span) * self.sr)
        avail = min(frames, len(self._samples) - start)
        if avail <= 0:
            out.fill(0)
            return
        out[:avail, 0] = self._samples[start:start + avail]
        if avail < frames:
            out[avail:].fill(0)

    # -- transport ---------------------------------------------------------
    def play(self) -> None:
        with self._lock:
            if self._playing:
                return
            self._anchor_wall = time.monotonic()
            self._playing = True

    def pause(self) -> None:
        with self._lock:
            if not self._playing:
                return
            self._anchor_pos = self._position_locked(time.monotonic())
            self._playing = False

    def seek(self, position: float) -> None:
        position = max(0.0, min(float(position), self.duration - 1e-3))
        with self._lock:
            self._anchor_pos = position
            self._anchor_wall = time.monotonic()

    def _position_locked(self, now: float) -> float:
        pos = self._anchor_pos
        if self._playing:
            pos += now - self._anchor_wall
        if self.loop_window is not None:
            a, b = self.loop_window
            if b > a and pos >= b:
                pos = a + (pos - a) % (b - a)
                self._anchor_pos = pos
                self._anchor_wall = now
        return min(pos, self.duration)
    @property
    def duration(self) -> float:
        return len(self._samples) / self.sr

    @property
    def position(self) -> float:
        with self._lock:
            return self._position_locked(time.monotonic())

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def underrun_count(self) -> int:
        return self._underruns

    def status(self) -> ClockStatus:
        return ClockStatus(self.position, self._playing, self.device,
                           self._underruns)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
