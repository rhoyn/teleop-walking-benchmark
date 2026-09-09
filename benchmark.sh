#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

RUNS=${RUNS:-2048}
ROUNDS=${ROUNDS:-4}
SONIC_DIV=${SONIC_DIV:-8}
JOBS=${JOBS:-4}
NICE=${NICE:-10}
DELAY=${DELAY:-60}
CPUS=1-$(($(nproc) - 1))
NCORES=$(($(nproc) - 1))
CORESEQ=$(mktemp -t benchmark-core.XXXXXX)
echo 0 >"$CORESEQ"
NGPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
[ "$NGPUS" -ge 1 ] || NGPUS=1
GPUDIR=$(mktemp -d -t benchmark-gpu.XXXXXX)
GPUSLOTS=$(((JOBS + NGPUS - 1) / NGPUS))
export RUNS SONIC_DIV CPUS NICE DELAY NCORES CORESEQ NGPUS GPUDIR GPUSLOTS

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
  ENGINE=$2
  R=$(printf "r%02d" "$ROUND")
  n=$RUNS
  [ "$P" = sonic ] && n=$((RUNS / SONIC_DIV))
  f=$((ROUND * n))
  out=$R.$P.$ENGINE
  CORE=$(
    exec 9>"$CORESEQ.lock"
    flock 9
    i=$(cat "$CORESEQ")
    echo $(((i + 1) % NCORES)) >"$CORESEQ"
    echo $((1 + i))
  )
  GPU=
  while [ -z "$GPU" ]; do
    for s in $(seq 0 $((GPUSLOTS - 1))); do
      for g in $(seq 0 $((NGPUS - 1))); do
        exec 8>"$GPUDIR/$g.$s"
        if flock -n 8; then GPU=$g; break 2; fi
        exec 8>&-
      done
    done
    [ -z "$GPU" ] && sleep 1
  done
  if CUDA_VISIBLE_DEVICES="$GPU" chrt --idle 0 nice -n "$NICE" taskset -c "$CORE" build/teleop-walking-benchmark \
      --engine "$ENGINE" --policy "$P" --runids "$f-$((f + n - 1))" \
      --csv "results/$out.csv" >"progress/$out.log" 2>&1; then
    grep "^mujoco:" "progress/$out.log" | sed "s|^|$out |" >>benchmark.log
  else
    echo "FAILED $out" >&2
    { echo "FAILED $out"; cat "progress/$out.log"; } >>benchmark.log
  fi
  rm -f "progress/$out.log"'

chrt --idle 0 nice -n "$NICE" taskset -c "$CPUS" make
make capture MJWARP_NWORLD="$RUNS"
mkdir -p results progress
rm -f progress/*.log benchmark.log

CPUPREV=$(mktemp -t benchmark-cpu.XXXXXX)
grep '^cpu[0-9]' /proc/stat >"$CPUPREV"

while sleep 1; do
  printf '\033[H\033[2J'
  for f in progress/*.log; do
    [ -e "$f" ] || continue
    printf '%-34s %8s robot-sim-s/s\n' "$(basename "$f" .log)" \
      "$(grep -o '[0-9]\+ robot-sim-s/s' "$f" | tail -1 | cut -d' ' -f1)"
  done
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,power.max_limit \
    --format=csv,noheader,nounits 2>/dev/null |
    awk -F', *' '{printf "gpu%-2s %3s%%  %6s/%-6s MiB  %5.0f/%-5.0f W\n", $1, $2, $3, $4, $5, $6}'
  grep '^cpu[0-9]' /proc/stat >"$CPUPREV.new"
  awk 'NR==FNR { for (i = 2; i <= NF; i++) p[$1, i] = $i; next }
       { t = 0; for (i = 2; i <= NF; i++) t += $i - p[$1, i]
         idle = ($5 - p[$1, 5]) + ($6 - p[$1, 6])
         printf "%4.0f", (t > 0) ? 100 * (t - idle) / t : 0
         if (++n % 16 == 0) printf "\n" }
       END { if (n % 16) printf "\n" }' "$CPUPREV" "$CPUPREV.new"
  mv "$CPUPREV.new" "$CPUPREV"
done &
MON=$!
trap 'kill $MON 2>/dev/null || true; rm -rf "$CORESEQ" "$CORESEQ.lock" "$GPUDIR" "$CPUPREV" "$CPUPREV.new"' EXIT

for r in $(seq 0 $((ROUNDS - 1))); do
  for p in $POLICIES; do
    for e in mujoco physx; do echo "$r $p $e"; done
  done | xargs -P "$JOBS" -n 3 bash -c "$RUN"
done
