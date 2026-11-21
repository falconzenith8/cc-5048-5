"""Player wiring: audio clock -> scheduler -> renderer.

Two front ends share the exact same pipeline:

* :class:`HeadlessPlayer` - advances an injectable clock by hand, used by the
  reproducibility tests and the offline comparison of underrun policies;
* :class:`GLPlayer`       - the interactive glfw window, used on machines with
  a display.

Only the frame selected by the scheduler for the *current* playback region is
ever submitted to a renderer; a region change (seek / loop wrap) bumps an
epoch and the streaming service drops every stale in-flight frame.
"""

from __future__ import annotations

import time

from .audio import AudioClock
from .cache import BoundedFrameCache
from .frame import FrameFile
from .scheduler import Scheduler, UnderrunPolicy
from .streaming import OutOfOrderNetwork, ReadDecodeService


class HeadlessPlayer:
    """Deterministic driver with an externally controlled virtual clock."""

    def __init__(self, vpc_path: str, wav_path: str,
                 policy: UnderrunPolicy = UnderrunPolicy.HOLD,
                 cache_bytes: int = 64 * 1024 * 1024,
                 loop_window: tuple[float, float] | None = None,
                 virtual: bool = True):
        self.ff = FrameFile(vpc_path)
        self.cache = BoundedFrameCache(max_bytes=cache_bytes)
        self.network = OutOfOrderNetwork()
        self.clock = AudioClock(wav_path, loop_window=loop_window)
        if loop_window is not None:
            self.clock.seek(loop_window[0])
        # Service uses the same monotonic source as the virtual clock so
        # injected delays elapse consistently during stepped playback.
        self.service = ReadDecodeService(
            self.ff, self.cache, self.network,
            clock=time.monotonic, sleep=lambda *_: None)
        self.scheduler = Scheduler(self.ff.timestamps, self.cache,
                                   self.service, policy=policy)
        self.submitted: list[int] = []
        self._last_epoch = self.service.epoch

    def start(self) -> None:
        self.service.start()

    def close(self) -> None:
        self.service.stop()
        self.ff.close()
        self.clock.close()

    # -- transport ---------------------------------------------------------
    def seek(self, position: float) -> None:
        self.clock.seek(position)
        self.scheduler.seek(position)

    def play(self) -> None:
        self.clock.play()

    def pause(self) -> None:
        self.clock.pause()

    def set_read_rate(self, rate: float) -> None:
        self.service.set_read_rate(rate)

    def set_loop(self, window: tuple[float, float] | None) -> None:
        self.clock.loop_window = window

    # -- stepping ----------------------------------------------------------
    def step(self, wall_seconds: float) -> object | None:
        """Burn wall time (worker thread runs in real time), then present."""
        time.sleep(wall_seconds)
        return self.poll()

    def poll(self):
        # Detect loop wraps performed inside the audio clock and treat them
        # exactly like an explicit seek: new region, old frames invalidated.
        if self.clock.loop_window is not None and self.clock.playing:
            a, b = self.clock.loop_window
            target = self.scheduler.target_index(self.clock.position)
            if target < self.service.play_index:
                self.scheduler.seek(self.clock.position)
        presented = self.scheduler.tick(self.clock.position)
        if presented is not None:
            self.submitted.append(presented.index)
        return presented


class GLPlayer:
    """Interactive windowed player."""

    def __init__(self, vpc_path: str, wav_path: str,
                 policy: UnderrunPolicy = UnderrunPolicy.HOLD,
                 cache_bytes: int = 64 * 1024 * 1024):
        from .gl_render import OpenGLRenderer

        self.ff = FrameFile(vpc_path)
        self.cache = BoundedFrameCache(max_bytes=cache_bytes)
        self.network = OutOfOrderNetwork()
        self.clock = AudioClock(wav_path)
        self.service = ReadDecodeService(self.ff, self.cache, self.network)
        self.scheduler = Scheduler(self.ff.timestamps, self.cache,
                                   self.service, policy=policy)
        self.renderer = OpenGLRenderer()
        self.renderer.on_key = self._on_key
        self._frame = None
        self._last_index = None
        self.loop_window: tuple[float, float] | None = None

    def _update_hud(self) -> None:
        import glfw
        m = self.scheduler.metrics
        mb = self.cache.stats.decoded_bytes / 1048576
        title = (
            f"volumeview | t={self.clock.position:6.2f}s "
            f"frame={self._last_index} "
            f"policy={self.scheduler.policy.value} "
            f"readrate={self.service.read_rate:.2f}x "
            f"cache={mb:.1f}MB "
            f"maxAV={m.max_av_offset_s:.3f}s "
            f"underrun={m.underruns} "
            f"depth={'on' if self.renderer.depth_mode else 'off'} "
            f"loop={'on' if self.loop_window else 'off'}")
        glfw.set_window_title(self.renderer._window, title)

    def _on_key(self, name: str) -> None:
        key = name.lower()
        if key == "space":
            self.clock.pause() if self.clock.playing else self.clock.play()
        elif key == "h":
            self.scheduler.policy = UnderrunPolicy.HOLD
        elif key == "j":
            self.scheduler.policy = UnderrunPolicy.SKIP
        elif key == "left":
            self.clock.seek(max(0.0, self.clock.position - 1.0))
            self.scheduler.seek(self.clock.position)
        elif key == "right":
            self.clock.seek(min(self.clock.duration - 0.01,
                                self.clock.position + 1.0))
            self.scheduler.seek(self.clock.position)
        elif key == "l":
            # Toggle a 4-second loop around the screen crossing.
            if self.loop_window is None:
                p = self.clock.position
                self.loop_window = (max(0.0, p - 2.0), p + 2.0)
                self.clock.loop_window = self.loop_window
            else:
                self.loop_window = None
                self.clock.loop_window = None
        elif key in ("+", "="):
            self.service.set_read_rate(self.service.read_rate / 1.5)
        elif key == "-":
            self.service.set_read_rate(self.service.read_rate * 1.5)

    def run(self) -> dict:
        if not self.renderer.init_window():
            raise RuntimeError(
                self.renderer.init_error
                or "no OpenGL context available; use the headless backend "
                   "(`python -m volumeview.cli headless`) or run on a machine "
                   "with a display")
        self.service.start()
        self.clock.play()
        try:
            while not self.renderer.should_close():
                self._handle_loop_wrap()
                presented = self.scheduler.tick(self.clock.position)
                if presented is not None:
                    self._frame = presented.frame
                    self._last_index = presented.index
                self.renderer.draw(self._frame)
                self._update_hud()
                self.renderer.swap()
                # ~60 fps polling; worker thread handles I/O concurrently.
                time.sleep(1.0 / 60.0)
        finally:
            report = self.scheduler.report()
            self.service.stop()
            self.ff.close()
            self.clock.close()
            self.renderer.terminate()
        return report

    def _handle_loop_wrap(self) -> None:
        if self.loop_window is None or not self.clock.playing:
            return
        target = self.scheduler.target_index(self.clock.position)
        if target < self.service.play_index:
            self.scheduler.seek(self.clock.position)
