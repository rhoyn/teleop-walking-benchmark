#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

exec uv run --isolated --no-project --python 3.14 \
  --with torch==2.14.0 \
  --with onnx==1.22.0 \
  --with onnxruntime==1.29.0 \
  ./export_onnx.py "$@"
