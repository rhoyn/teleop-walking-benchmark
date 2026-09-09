#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RUNS=${RUNS:-1024}
ROUNDS=${ROUNDS:-8}
SONIC_DIV=${SONIC_DIV:-8}
MJOBS=${MJOBS:-3}
PJOBS=${PJOBS:-3}
NICE=${NICE:-10}
DELAY=${DELAY:-60}
CPUS=1-$(($(nproc) - 1))
MTHREADS=${MTHREADS:-$((($(nproc) - 1) * 3 / MJOBS))}
export RUNS SONIC_DIV CPUS NICE DELAY

POLICIES=$(for d in policies/*/; do [ -f "$d/policy.cpp" ] && basename "$d"; done |
  grep -vE '^(decoupled_wbc|gr00t_wbc)$')
POLICIES="$POLICIES clobot_with_arms handoff_with_arms"
for h in h074_p000 h070_p000 h066_p000 h066_p012; do
  POLICIES="$POLICIES gr00t_wbc_$h decoupled_wbc_$h"
done

RUN='
  sleep $((RANDOM % (DELAY + 1)))
  ROUND=$0
  P=$1
  R=$(printf "r%02d" "$ROUND")
  n=$RUNS
  [ "$P" = sonic ] && n=$((RUNS / SONIC_DIV))
  f=$((ROUND * n))
  out=$R.$P.$ENGINE
  if chrt --idle 0 nice -n "$NICE" taskset -c "$CPUS" build/teleop-walking-benchmark \
      --engine "$ENGINE" --policy "$P" --runids "$f-$((f + n - 1))" --threads "$TH" \
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

JOBS=$(for r in $(seq 0 $((ROUNDS - 1))); do for p in $POLICIES; do echo "$r $p"; done; done)

printf '%s\n' "$JOBS" | ENGINE=mujoco TH=$MTHREADS xargs -P "$MJOBS" -n 2 bash -c "$RUN" &
M=$!
printf '%s\n' "$JOBS" | ENGINE=physx TH=1 xargs -P "$PJOBS" -n 2 bash -c "$RUN" &
P=$!
wait "$M" "$P"
