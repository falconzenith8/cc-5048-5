#!/usr/bin/env bash
# Reproduce every result shipped under recordings/ from a clean checkout.
# Runs fully headless: no display, GPU or audio device required.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "== 1/6 generate scene, drum track, beatmap, replay plan =="
"$PY" -m volumeview.cli assets

echo "== 2/6 headless replay, HOLD policy (seek + render front/back/depth) =="
"$PY" -m volumeview.cli headless --policy hold --duration 8 \
    --out recordings/hold_run

echo "== 3/6 headless replay, SKIP policy under the same conditions =="
"$PY" -m volumeview.cli headless --policy skip --duration 8 \
    --out recordings/skip_run --read-rate 3.5

echo "== 4/6 compare both underrun policies (slow read, seek + loop) =="
"$PY" -m volumeview.cli compare --duration 10 --read-rate 3.5

echo "== 5/6 tween layer vs depth ground truth (turn scene, 2 dropped frames) =="
"$PY" -m volumeview.cli tween-eval --out recordings/tween_eval

echo "== 6/6 two-actor swap sample: wrong link, veto, repair =="
"$PY" -m volumeview.cli swap-sample --out recordings/swap_sample

echo
echo "Artifacts:"
find recordings -maxdepth 2 -type f | sort
