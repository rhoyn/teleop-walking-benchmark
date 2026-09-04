# teleop-walking-benchmark

Thirteen open-source Unitree G1 walking policies ported to one C++ interface and
scored against the same waypoint tour in MuJoCo. Twelve of the sixteen
checkpoints are committed under `policies/`; `./download_weights.sh` pulls the
other four.

The difficulty is deliberate. A benchmark every candidate completes ranks
nothing — it separates no one, it only shows the task was too easy. This one is
tuned to sit at the edge of what these policies can do: the best of them
finishes most seeds, none finishes all, and no row is a perfect score. The
punches are what make it a test of stability and disaster recovery rather than
of walking — a policy is judged on whether it can absorb a hit it never saw
coming and get back on the tour, not only on how neatly it tracks a target
while nothing is going wrong.

![The field at twenty-seven seconds in: thirteen tiles, one policy each and named beside it, most of the humanoids already down with the orange punch cylinder still resting where it hit them, a few still on their feet and tracking the next waypoint](assets/preview.jpg)

https://github.com/user-attachments/assets/161f3784-ec7e-44b6-9856-8ce661bb295d

*The whole field side by side, thirteen policies on the same tour — the closing
forty-two seconds of it, by which point several have already fallen. The same
clip is `assets/preview.mp4`.*

## Running it

```sh
./download_weights.sh                    # the four checkpoints not committed here
./export_onnx.py                         # robomimic's ONNX, converted from its .pt
make
./run.sh --policy gr00t_wbc              # one candidate, one tour
./run.sh --policy gr00t_wbc --seed 7     # a different tour and punch campaign
./run.sh --policy asap --realtime 1
./run.sh --policy amo --record videos    # videos/amo.mp4
```

`--policy` is required and takes one name: a run measures one candidate. The
whole field is a loop over invocations rather than a flag, because holding
thirteen checkpoints' engines and thirteen networks' CUDA kernels in one
process costs more memory than the runs themselves — one policy per process
bounds that by construction. Without `--realtime` the run is headless and as
fast as the machine allows.

`./export_onnx.py` is only needed for `robomimic`, whose checkpoint is fetched
rather than committed and so cannot have its conversion committed either. It
needs `torch` and `onnx`; the other twelve policies run from a plain clone.

### Campaigns

A seed is a whole task rather than a repetition, so a policy is only separated
from its rivals over many of them.

```sh
./run.sh --policy gr00t_wbc --runs 500 --parallel 16 --csv results/result.csv
./run.sh --policy amo --seeds 100-199 --parallel 16 --csv results/result.csv
```

`--runs N` walks seeds 0..N-1, `--seeds A-B` an explicit range. `--parallel W`
keeps W tours in flight, one core each, sharing one scene and one CUDA context;
the memory a sweep needs is set by W rather than by the size of the field.

`--csv FILE` appends one row per finished run — policy, seed, outcome,
survival, targets scored and the two mean errors — flushed as each run ends, so
an interrupted campaign keeps everything already finished. The header is
written only to an empty file, so a campaign resumes onto the same file. Those
seven columns are what the table below is computed from and nothing else; a run
stopped part way is recorded as `interrupted` and is not a measurement.

`--record DIR` writes `DIR/<policy>.mp4` at 50 fps, one frame per 0.02 s of
simulated time, so playback is exactly real time however fast the run computed.
The clip carries the policy name on the floor and nothing else; a punch draws a
red arrow at the contact point, sized by the force and held for 500 ms after the
hit.

## The 500-seed campaign

Seeds 0-499, all thirteen policies per seed: **6500 runs**, 160 robot-hours of
simulated walking, sixteen tours in flight on an RTX 4090 / Ryzen 7 7800X3D.
No run errored, and no run was interrupted.

Each seed draws its own 24-target tour and its own punch campaign, so a seed is
a whole task, not a repetition — the runs are deterministic and a repeated seed
reproduces bit-identically.

Every policy is handed the same stance. The crane ramps the robot to the shared
`DEFAULT_ANGLES` pose over three seconds and then lets go, rather than to a pose
each checkpoint chose for itself, so what is measured is how well a policy takes
over from a stance it was not necessarily trained around.

| `--policy` | completed | 95% CI | survived | median | worst | pos err | yaw err |
|---|---|---|---|---|---|---|---|
| `gr00t_wbc` | **374/500** | 71-78% | 120.0 s | 125.0 s | 60.7 s | 9.10 cm | 2.92° |
| `amo` | **330/500** | 62-70% | 118.1 s | 125.0 s | 60.8 s | 19.58 cm | 13.03° |
| `homie` | **292/500** | 54-63% | 116.3 s | 125.0 s | 65.8 s | 18.05 cm | 34.70° |
| `robomimic` | **242/500** | 44-53% | 113.7 s | 121.0 s | 46.0 s | 10.20 cm | 3.58° |
| `asap` | **175/500** | 31-39% | 109.2 s | 115.7 s | 46.0 s | 12.89 cm | 9.26° |
| `rl_mjlab` | **140/500** | 24-32% | 105.8 s | 110.6 s | 51.8 s | 13.17 cm | 3.31° |
| `holosoma` | **119/500** | 20-28% | 106.6 s | 110.6 s | 36.0 s | 15.13 cm | 3.13° |
| `run_residual` | **22/500** | 3-7% | 85.5 s | 86.2 s | 21.0 s | 516.15 cm | 3.28° |
| `rl_lab` | **13/500** | 2-4% | 86.5 s | 86.1 s | 24.1 s | 12.25 cm | 67.05° |
| `falcon` | **1/500** | 0-1% | 66.6 s | 66.0 s | 18.9 s | 29.63 cm | 3.25° |
| `openwbt` | **0/500** | 0-1% | 67.1 s | 66.2 s | 31.0 s | 8.80 cm | 34.39° |
| `rl_gym` | **0/500** | 0-1% | 51.4 s | 53.4 s | 6.8 s | 53.04 cm | 5.99° |
| `clobot` | **0/500** | 0-1% | 5.3 s | 5.3 s | 5.3 s | - | - |

**Four of these completed fewer than five tours** — `falcon` (1) and `openwbt`,
`rl_gym` and `clobot` (none) — so read their error columns with care. Those
figures rest on whichever fragments of a tour the policy reached before falling,
which is not the same measurement as the rows above them.

Survival is sim seconds to the fall, counting a completed tour as its full
125 s. Position and yaw errors are means over every target actually scored, so
a policy that falls early is judged only on the targets it reached — that
flatters the short-lived candidates rather than penalising them. Step timings
are deliberately absent: the sweeps shared the machine sixteen at a time, which
inflates them by an order of magnitude. Measure those solo.

**`gr00t_wbc` wins the criteria that matter** — fewest falls, longest survival,
lowest yaw error — and at five hundred seeds its interval clears `amo`'s with
nothing to spare, 70.8% against 70.0%. First place is not a sampling artefact,
but it is no longer a wide margin either. It does not hold the lowest mean
position error: `openwbt` edges it, 8.80 cm against 9.10 cm, on a row with zero
completions.

**Five hundred seeds separate what a hundred could not.** `robomimic` sat inside
`amo`'s and `homie`'s interval before and now sits clearly below both, fourth on
its own. What remains genuinely tied is `amo` with `homie`, `asap` with
`rl_mjlab` and `holosoma`, and the four rows at the bottom that never finish
anything — no sample size separates policies that all score zero.

**`openwbt` is accurate and cannot finish**: the best mean position error in the
field and zero completions in five hundred tries. Its 34.39° yaw error is where
it goes — capped at its published 0.3 m/s, it cannot hold heading on the harder
draws.

**`clobot` fails identically on every seed**: 5.3 s mean, median and worst, and
not one scored target in five hundred tours. No run differed from any other.

**Low error does not mean good.** `run_residual` cannot strafe, reverse or turn
in place — its command floor forces it forward at 0.5 m/s whenever it is off
target, so it can only approach on an arc — and 516 cm is a policy that cannot
do this task rather than a bad gait. `openwbt`'s first-place error is earned on
the fraction of each tour it survived. Read the completion column first.

**Nothing completes reliably.** The best policy in the field fails a quarter of
its tours, and the top four die around twenty seconds before the end of a 125 s
tour on average, where the punch ramp approaches its 600 N ceiling. Surviving
the end of the tour is the discriminator, not tracking error.

The runs behind this table are in [results/result.csv](results/result.csv), one
row per run.

## Inference

Every candidate runs on the GPU, all thirteen as ONNX compiled to TensorRT
plans, cached under `build/trt/` and keyed on the model's CRC32 plus the
TensorRT and CUDA versions, the GPU and the precision — a first run builds,
later runs load. Nothing links libtorch.

`amo`, `rl_gym` and `robomimic` shipped as TorchScript and were converted by
`./export_onnx.py`, which checks each export against the original before
writing it. `rl_gym` and `robomimic` carry an LSTM whose hidden and cell state
lived in buffers that `forward` wrote back into; ONNX has no in-place buffer
mutation and a plan is a pure function, so that state is lifted into explicit
inputs and outputs and the policy carries it from step to step. It is the same
recurrence, moved from the module into the caller.

Converting them changes no weight, but it does change the arithmetic. Float32
rounding of order 1e-6 per step, amplified across 125,000 physics steps of a
contact-rich tour, sends individual seeds down different trajectories — so a
converted policy's result on a given seed is not comparable to its result
before the conversion, even though its statistics over a campaign are.

Each tour in a sweep pins itself to one core, so concurrent tours spread across
the machine. Physics is the cost, not inference: a tour is 125,000 `mj_step`
calls against 6,250 policy steps, and inference is under 3% of a run.

## Policy weights

Sixteen checkpoints across the thirteen policies. Twelve are committed beside
the policy that loads them; `./download_weights.sh` pulls the other four from
their upstream repository and checks each file's sha256 as it lands.

A checkpoint's licence travels to anything derived from it, so the ONNX that
`./export_onnx.py` writes is shipped exactly where its source is. `amo` and
`rl_gym` are committed, so their conversions are committed too; `robomimic`'s
checkpoint is fetched rather than committed, so its conversion is generated
rather than committed. Converting a file to another format does not make it
redistributable.

Every checkpoint is third-party, as are the Unitree G1 description under
`assets/` and the bundled font. Their provenance and terms — which one is
non-commercial, which two declare nothing at all, and which is sim-only — are
recorded in [NOTICE](NOTICE). Read it before you build on a result.

## Adding a policy

One directory under `policies/`, and two lines in `main.cpp`. Implement
`ModelPolicy` — `init`, `step`, `name` — with the weights beside it, then add
the `#include` at the end of `main.cpp` and a branch in `make_policy`.

`step` receives the raw target error in the body frame and returns the motor
targets; deriving a velocity command from that error is the policy's own
business.
