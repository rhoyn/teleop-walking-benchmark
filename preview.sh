#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

JOBS=${JOBS:-8}
RUNID=${RUNID:-0}
NICE=${NICE:-10}
CPUS=1-$(($(nproc) - 1))
WORK=${WORK:-build/preview-clips}
VIDEO=${VIDEO:-assets/preview.mp4}
IMAGE=${IMAGE:-assets/preview.jpg}
STILL=${STILL:-44.6}
FPS=${FPS:-60}
BIN=${BIN:-build/teleop-walking-benchmark}
DELAY=${DELAY:-20}
export RUNID NICE CPUS WORK BIN DELAY

COLS=10 ROWS=9 TW=128 TH=72 OW=1280 OH=720 BG=0x3D9356
FONT=assets/JetBrainsMono.ttf
ENGINES=${ENGINES:-"physx mujoco"}

POLICIES=${POLICIES:-$(build/table results/*.csv | sed 's/`<br>`/_/' |
  sed -n 's/^| ~*`\([a-z0-9_]*\)`.*/\1/p')}
TOTAL=$(($(echo $POLICIES | wc -w) * $(echo $ENGINES | wc -w)))
export TOTAL
[ "$TOTAL" -lt $((COLS * ROWS)) ] ||
  { echo "$TOTAL clips + label do not fit ${COLS}x${ROWS} tiles" >&2; exit 1; }

RUN='
  out=$WORK/$1/$0
  sleep $((RANDOM % (DELAY + 1)))
  nice -n "$NICE" taskset -c "$CPUS" "$BIN" --engine "$1" --policy "$0" \
    --runid "$RUNID" --record "$WORK/$1" --csv "$out.csv" \
    >"$out.log" 2>&1 &&
    echo "[$(ls "$WORK"/*/*.mp4 2>/dev/null | wc -l)/$TOTAL] $0 on $1" ||
    echo "FAILED $0 on $1" >&2'

nice -n "$NICE" taskset -c "$CPUS" make
nice -n "$NICE" taskset -c "$CPUS" make table
rm -rf "$WORK"
for e in $ENGINES; do mkdir -p "$WORK/$e"; done
echo "recording $TOTAL clips, $JOBS at a time"

for p in $POLICIES; do
  for e in $ENGINES; do echo "$p $e"; done
done | xargs -P "$JOBS" -n 2 bash -c "$RUN"

clips=() layout= scale= stack= n=0
for p in $POLICIES; do
  for e in $ENGINES; do
    clips+=(-i "$WORK/$e/$p.mp4")
    layout+="${layout:+|}$((n % COLS * TW))_$((n / COLS * TH))"
    scale+="[$n:v]scale=$TW:$TH:flags=lanczos[v$n];"
    stack+="[v$n]"
    n=$((n + 1))
  done
done
pad=$(((OH - ROWS * TH) / 2))
[ "$n" -gt 1 ] && stack+="xstack=inputs=$n:layout=$layout:fill=$BG,"

FILTER="$scale$stack
  pad=$OW:$OH:0:$pad:color=$BG,
  drawtext=fontfile=$FONT:text='r h o y n':fontcolor=white:fontsize=10
    :x=$((n % COLS * TW + 37)):y=$((n / COLS * TH + pad + 30)),
  format=yuv420p"

echo "combining $n clips"
nice -n "$NICE" taskset -c "$CPUS" ffmpeg -hide_banner -loglevel error -stats -y \
  "${clips[@]}" -filter_complex "$FILTER" \
  -c:v libx264 -preset slow -crf 18 -an "$WORK/montage.mp4"

echo "re-encoding into $VIDEO at $FPS fps"
nice -n "$NICE" taskset -c "$CPUS" ffmpeg -hide_banner -loglevel error -stats -y \
  -err_detect aggressive -fflags discardcorrupt -i "$WORK/montage.mp4" \
  -r "$FPS" -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -movflags +faststart -an "$VIDEO"

ffmpeg -hide_banner -loglevel error -y -ss "$STILL" -i "$VIDEO" \
  -frames:v 1 -q:v 2 "$IMAGE"
[ -s "$IMAGE" ] || { echo "no still at ${STILL}s of $VIDEO" >&2; exit 1; }

echo "wrote $VIDEO and $IMAGE"
