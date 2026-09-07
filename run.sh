#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

make

exec build/teleop-walking-benchmark "$@"
