#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RUNS=${RUNS:-1024}
ROUNDS=${ROUNDS:-10}
SONIC_DIV=${SONIC_DIV:-8}
MJOBS=${MJOBS:-2}
PJOBS=${PJOBS:-3}
NICE=${NICE:-10}
CPUS=1-$(($(nproc) - 1))
MTHREADS=${MTHREADS:-$((($(nproc) - 1) * 3 / MJOBS))}
export RUNS SONIC_DIV CPUS NICE

POLICIES=$(for d in policies/*/; do [ -f "$d/policy.cpp" ] && basename "$d"; done)
POLICIES="$POLICIES clobot_with_arms"

RUN='
  n=$RUNS
  [ "$0" = sonic ] && n=$((RUNS / SONIC_DIV))
  f=$((ROUND * n))
  out=$R.$0.$ENGINE
  if chrt --idle 0 nice -n "$NICE" taskset -c "$CPUS" build/teleop-walking-benchmark \
      --engine "$ENGINE" --policy "$0" --runids "$f-$((f + n - 1))" --threads "$TH" \
      --csv "results/$out.csv" >"progress/$out.log" 2>&1; then
    grep "^mujoco:" "progress/$out.log" | sed "s|^|$out |" >>benchmark.log
  else
    echo "FAILED $out" >&2
    { echo "FAILED $out"; cat "progress/$out.log"; } >>benchmark.log
  fi
  rm -f "progress/$out.log"'

chrt --idle 0 nice -n "$NICE" taskset -c "$CPUS" make
mkdir -p results progress
rm -f progress/*.log benchmark.log

while sleep 1; do
  printf '\033[H\033[2J'
  for f in progress/*.log; do
    [ -e "$f" ] || continue
    printf '%-34s %8s robot-sim-s/s\n' "$(basename "$f" .log)" \
      "$(grep -o '[0-9]\+ robot-sim-s/s' "$f" | tail -1 | cut -d' ' -f1)"
  done
done &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

for r in $(seq 0 $((ROUNDS - 1))); do
  export ROUND=$r R=$(printf 'r%02d' "$r")
  printf '%s\n' $POLICIES | ENGINE=mujoco TH=$MTHREADS xargs -P "$MJOBS" -n 1 bash -c "$RUN" &
  M=$!
  printf '%s\n' $POLICIES | ENGINE=physx TH=1 xargs -P "$PJOBS" -n 1 bash -c "$RUN" &
  wait "$M" "$!"
done
