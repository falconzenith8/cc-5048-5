#!/usr/bin/env bash
# Reproduce every result shipped under recordings/ from a clean checkout.
# Runs fully headless: no display, GPU or audio device required.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

echo "== 1/4 generate scene, drum track, beatmap, replay plan =="
"$PY" -m volumeview.cli assets

echo "== 2/4 headless replay, HOLD policy (seek + render front/back/depth) =="
"$PY" -m volumeview.cli headless --policy hold --duration 8 \
    --out recordings/hold_run

echo "== 3/4 headless replay, SKIP policy under the same conditions =="
"$PY" -m volumeview.cli headless --policy skip --duration 8 \
    --out recordings/skip_run --read-rate 3.5

echo "== 4/4 compare both underrun policies (slow read, seek + loop) =="
"$PY" -m volumeview.cli compare --duration 10 --read-rate 3.5

echo
echo "Artifacts:"
find recordings -maxdepth 2 -type f | sort
