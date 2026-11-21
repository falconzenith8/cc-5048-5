"""Command line entry point.

Subcommands:
  assets    generate the scene, frame container, drum track and beat map
  play      interactive OpenGL window (falls back with a clear message)
  headless  offline replay that renders PNGs and prints policy metrics
  compare   run both underrun policies on the same recording and compare
"""

from __future__ import annotations

import argparse
import sys

from .scheduler import UnderrunPolicy


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="volumeview")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_assets = sub.add_parser("assets", help="generate scene and media assets")
    p_assets.add_argument("--out", default="assets")
    p_assets.add_argument("--recordings", default="recordings")

    p_play = sub.add_parser("play", help="interactive OpenGL player")
    p_play.add_argument("--vpc", default="assets/scene.vpc")
    p_play.add_argument("--wav", default="assets/drums.wav")
    p_play.add_argument("--policy", choices=[p.value for p in UnderrunPolicy],
                        default=UnderrunPolicy.HOLD.value)
    p_play.add_argument("--cache-mb", type=int, default=64)

    p_head = sub.add_parser("headless", help="offline replay + PNG renders")
    p_head.add_argument("--vpc", default="assets/scene.vpc")
    p_head.add_argument("--wav", default="assets/drums.wav")
    p_head.add_argument("--policy", choices=[p.value for p in UnderrunPolicy],
                        default=UnderrunPolicy.HOLD.value)
    p_head.add_argument("--out", default="recordings/headless")
    p_head.add_argument("--duration", type=float, default=8.0)
    p_head.add_argument("--loop", nargs=2, type=float, metavar=("A", "B"))
    p_head.add_argument("--read-rate", type=float, default=1.0)

    p_cmp = sub.add_parser("compare", help="compare underrun policies")
    p_cmp.add_argument("--vpc", default="assets/scene.vpc")
    p_cmp.add_argument("--wav", default="assets/drums.wav")
    p_cmp.add_argument("--duration", type=float, default=10.0)
    p_cmp.add_argument("--read-rate", type=float, default=3.5)
    p_cmp.add_argument("--json", action="store_true",
                       help="emit machine-readable JSON")

    args = parser.parse_args(argv)

    if args.cmd == "assets":
        from tools.generate_assets import generate
        generate(args.out, args.recordings)
        return 0

    if args.cmd == "play":
        from .player import GLPlayer
        player = GLPlayer(args.vpc, args.wav,
                          policy=UnderrunPolicy(args.policy),
                          cache_bytes=args.cache_mb * 1024 * 1024)
        report = player.run()
        for key, value in report.items():
            print(f"{key:24s} {value}")
        return 0

    if args.cmd == "headless":
        from tools.headless_replay import run_headless
        run_headless(args.vpc, args.wav, UnderrunPolicy(args.policy),
                     args.out, duration=args.duration,
                     loop_window=tuple(args.loop) if args.loop else None,
                     read_rate=args.read_rate)
        return 0

    if args.cmd == "compare":
        from tools.compare_policies import compare
        compare(args.vpc, args.wav, duration=args.duration,
                read_rate=args.read_rate, as_json=args.json)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
