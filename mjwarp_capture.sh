#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

exec uv run --isolated --no-project --python 3.14 \
  --with mujoco-warp==3.12.0 \
  ./mjwarp_capture.py "$@"
