#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

POLICIES="amo asap clobot clobot_with_arms falcon gr00t_wbc holosoma homie
          openwbt rl_gym rl_lab rl_mjlab robomimic run_residual"
PARALLEL=$(($(nproc) - 1))

make

for round in $(seq 0 9); do
  for policy in $POLICIES; do
    build/teleop-walking-benchmark --policy "$policy" \
      --seeds "$((round * 1000))-$((round * 1000 + 999))" --parallel "$PARALLEL"
  done
done
