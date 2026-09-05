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
./run.sh --policy gr00t_wbc --runs 500 --parallel 16
./run.sh --policy amo --seeds 100-199 --parallel 16
```

`--runs N` walks seeds 0..N-1, `--seeds A-B` an explicit range. `--parallel W`
keeps W tours in flight, one core each, sharing one scene and one CUDA context;
the memory a sweep needs is set by W rather than by the size of the field.

**Every run records itself.** Sweep or single seed, watched or headless, each
run appends to `results/result.csv` — created along with its directory if it is
not there — and `--csv FILE` only moves that somewhere else. A result that
reached nothing but a terminal is a result nobody can pool later, so there is no
flag to ask for the file and none to turn it off.

**One row is one run**, flushed as it ends so an interrupted campaign keeps
everything already finished. The header is written only to an empty file, and a
file whose header does not match is refused rather than appended to, so a
campaign resumes onto the same file but cannot interleave two schemas in one.

The row opens with the run itself — `policy`, `seed`, `outcome`, `survival_s`,
`targets`, `pos_err_cm`, `yaw_err_deg` — which is what the published table is
computed from and nothing else. The error columns are poolable across runs
because `targets` travels beside them as the weight, and a run stopped part way
is recorded as `interrupted` and is not a measurement.

What each leg of the tour cost follows as `s<i>_` columns, one block of ten per
target clock, twelve blocks in all. A leg is one target's clock: `s5_` is the
leg walked while target 5 was being asked for.

- **`s<i>_e_<group>_j`** is absolute mechanical work, the sum of `|τ·ω|·dt` over
  every 1 ms physics step of that leg, in joules. The absolute value is what
  makes it an actuator cost rather than a physics one — these motors burn
  current holding a limb against gravity whichever way it is moving, and a
  signed integral would let a leg swinging down pay back the leg that lifted it.
- **`s<i>_v_<group>_krads2`** is how much the group shook: the total variation
  of each joint's angular acceleration, the sum of `|α(t) − α(t−dt)|` over the
  leg, added across the group, in thousands of rad/s².

Both are summed at the 1 ms physics step rather than sampled at the 50 Hz
control rate, because a policy that buzzes its ankles at 200 Hz is invisible to
anything that only looks when the policy does. Every number is written to one
decimal place, which is all any of them is worth, and the unit is in the column
name.

The twenty-nine joints are binned five ways, splitting proximal from distal in
both limbs because that is where these joints genuinely differ — the proximal
ones carry the robot's weight and do nearly all the work, while the distal ones
are light, fast and where a policy's chatter shows up first:

| group | joints | n |
|---|---|---|
| `legs_upper` | hip pitch/roll/yaw, knee — both sides | 8 |
| `legs_lower` | ankle pitch/roll — both sides | 4 |
| `waist` | waist yaw/roll/pitch | 3 |
| `arms_upper` | shoulder pitch/roll/yaw, elbow — both sides | 8 |
| `arms_lower` | wrist roll/pitch/yaw — both sides | 6 |

A run that fell leaves the legs it never reached empty, and **nothing marks
which legs ran their full clock because nothing needs to.** Legs `0` to
`targets - 1` each had their whole five seconds; a block present at index
`targets` is the leg the run died in, running from `5 × targets` seconds to
`survival_s`. Both quantities are sums over the leg rather than rates, so that
last partial block covers less than a target clock and should be dropped before
pooling.

`--record DIR` writes `DIR/<policy>.mp4` at 50 fps, one frame per 0.02 s of
simulated time, so playback is exactly real time however fast the run computed.
The clip carries the policy name on the floor and nothing else; a punch draws a
red arrow at the contact point, sized by the force and held for 500 ms after the
hit.

## The 10,000-seed campaign

Seeds 0-9999, every policy over every seed: **140,000 runs**, 1568 robot-hours
of simulated walking, fifteen tours in flight on an RTX 4090 / Ryzen 7 7800X3D.
No run errored and no run was interrupted, so every row in `results/result.csv`
is a measurement.

Each seed draws its own 12-target tour and its own punch campaign, so a seed is
a whole task, not a repetition — the runs are deterministic and a repeated seed
reproduces bit-identically.

Every policy is handed the same stance. The crane ramps the robot to the shared
`DEFAULT_ANGLES` pose over three seconds and then lets go, rather than to a pose
each checkpoint chose for itself, so what is measured is how well a policy takes
over from a stance it was not necessarily trained around.

| `--policy` | completed | 95% CI | survived | median | worst | pos err | yaw err |
|---|---|---|---|---|---|---|---|
| `gr00t_wbc` | **7894/10000** | 78-80% | 56.7 s | 60.0 s | 5.8 s | 9.54 cm | 2.91° |
| `homie` | **7104/10000** | 70-72% | 55.6 s | 60.0 s | 5.8 s | 17.94 cm | 34.46° |
| `amo` | **6987/10000** | 69-71% | 54.5 s | 60.0 s | 0.8 s | 23.03 cm | 12.08° |
| `robomimic` | **6515/10000** | 64-66% | 54.9 s | 60.0 s | 0.9 s | 11.10 cm | 3.60° |
| `asap` | **5353/10000** | 53-55% | 52.5 s | 60.0 s | 0.8 s | 13.16 cm | 9.95° |
| `rl_mjlab` | **4592/10000** | 45-47% | 51.0 s | 56.3 s | 0.8 s | 14.74 cm | 3.42° |
| `holosoma` | **4169/10000** | 41-43% | 50.1 s | 55.7 s | 5.6 s | 18.07 cm | 3.14° |
| `run_residual` | **1435/10000** | 14-15% | 39.2 s | 40.9 s | 1.0 s | 432.32 cm | 3.24° |
| `rl_lab` | **731/10000** | 7-8% | 36.3 s | 36.3 s | 0.7 s | 13.75 cm | 65.47° |
| `falcon` | **226/10000** | 2-3% | 28.3 s | 26.6 s | 1.2 s | 33.64 cm | 4.07° |
| `rl_gym` | **114/10000** | 1-1% | 23.9 s | 22.1 s | 0.6 s | 57.89 cm | 6.22° |
| `openwbt` | **104/10000** | 1-1% | 26.1 s | 26.1 s | 0.8 s | 10.37 cm | 35.88° |
| `clobot` | **0/10000** | 0-0% | 2.2 s | 2.2 s | 0.7 s | - | - |
| ~~`clobot_with_arms`~~ | ~~**855/10000**~~ | ~~8-9%~~ | ~~33.1 s~~ | ~~31.1 s~~ | ~~0.8 s~~ | ~~33.81 cm~~ | ~~5.01°~~ |

**The bottom four complete under 3% of their tours** — `falcon`, `rl_gym`,
`openwbt` and `clobot`, which finishes none — so read their error columns with
care. Those figures rest on whichever fragments of a tour the policy reached
before falling, which is not the same measurement as the rows above them.

Survival is sim seconds to the fall **measured from the crane's release**, not
from the start of the simulation: the three seconds the crane spends ramping the
robot to the shared pose are seconds it could not fall in, and counting them
would hand every policy the same three-second floor and flatter the worst of
them most. The first target is demanded the moment the crane lets go, so a
completed tour is exactly 60 s — twelve targets on a 5 s clock, the last of them
closing on the minute. The punch campaign is drawn from the seed but timed from
that same release, so punch *i* lands a tenth of a second into leg *i* — far
enough in that the policy has taken the new demand and started to act on it, so
the punch disturbs the walking rather than the handover. Position and yaw errors are means over every target actually scored, so
a policy that falls early is judged only on the targets it reached — that
flatters the short-lived candidates rather than penalising them. Step timings
are deliberately absent: the sweeps shared the machine fifteen at a time, which
inflates them by an order of magnitude. Measure those solo.

**`gr00t_wbc` now wins every column.** Fewest falls, longest survival, lowest
position error and lowest yaw error — the first time one policy has taken all
four. It held the first three on the old tour but lost position error to
`openwbt`; at ten thousand seeds it takes that too, 9.54 cm against 10.37 cm,
and this time on a row that finishes four fifths of its tours rather than none.
Its interval clears second place by seven points, 78-80% against `homie`'s
70-72%. First place is not close.

**Ten thousand seeds leave almost nothing tied.** Every interval in the table is
two points wide or less, and only `homie` and `amo` overlap at all — 70-72%
against 69-71%, a single point of contact after seven thousand completions
apiece. `robomimic` is clearly fourth, `asap` clearly fifth. The sample size is
past the point of being the limiting factor: what separates these policies now
is the tour, not the statistics.

**`openwbt` is accurate and still cannot finish**: second-best mean position
error in the field and 104 completions in ten thousand tries. Its 35.88° yaw
error is where it goes — capped at its published 0.3 m/s, it cannot hold heading
on the harder draws.

**`clobot` fails on every seed, but no longer identically.** Not one scored
target in ten thousand tours, and a mean, median and worst of 2.2 s, 2.2 s and
0.7 s. On the old tour every run was the same 5.3 s to three significant
figures; now the first punch lands a tenth of a second after the release and
spreads the falls across 27 distinct times between 0.7 s and 3.3 s. The
punch changes when it goes over, not whether.

That figure is the tour's rule rather than the gait. `clobot` is the only
checkpoint here trained as a whole-body policy whose balance depends on its own
arm motion — its deploy config carries a per-joint action scale, `kp` and `kd`
for all 29 joints, wrists included. This tour holds the arms for every
candidate: whatever a policy leaves unowned sits at `DEFAULT_ANGLES` on the
shared gains, and `clobot` is driven on the same 15 lower-body joints as the
rest of the field. Take the upper body away and it topples in five seconds,
every time. Parking the arms at its own nominal pose rather than
`DEFAULT_ANGLES` recovers none of it — 5.9 s over 32 seeds — so what it wants is
the arm swing, not the arm pose.

In fairness to the checkpoint, the same weights are also wired up as
`--policy clobot_with_arms`, owning all 29 joints, and run over the same seeds
0-9999. It is struck through in the table because it does not rank: the table is
thirteen policies driving the lower body under one rule, and `clobot_with_arms`
is a harness configuration rather than a fourteenth checkpoint. Read its row
beside `clobot`'s rather than against the ones above it — giving the policy its
arms back is worth 855 tours against none, which would place it above `rl_lab`
rather than at the bottom of the field.

**Low error does not mean good.** `run_residual` cannot strafe, reverse or turn
in place — its command floor forces it forward at 0.5 m/s whenever it is off
target, so it can only approach on an arc — and 432 cm is a policy that cannot
do this task rather than a bad gait. `openwbt`'s near-first error is earned on
the fraction of each tour it survived. Read the completion column first.

**Nothing completes reliably.** The best policy in the field fails a fifth of
its tours, and when the top four do fall they fall with about fifteen seconds
left of a 60 s tour, where the punch ramp is closing on its 600 N ceiling.
Surviving the end of the tour is the discriminator, not tracking error.

**The opening punch is its own filter.** It lands a tenth of a second after the
crane lets go, before any policy has taken a step, and it ends 4.2% of `rl_gym`'s
runs and 2.8% of `openwbt`'s inside the first leg. The top four lose almost
nothing there — two runs in ten thousand for `robomimic` and `asap` — so it
separates the policies that cannot take a hit from a standing start rather than
punishing the field at random.

The runs behind this table are in [results/result.csv](results/result.csv), one
row per run, with `clobot_with_arms` in that file too — ten thousand rows that
no line of the table above draws on. Every run also carries what each leg of its
tour cost, joint group by joint group, in the `s<i>_` columns described above.

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
rounding of order 1e-6 per step, amplified across 60,000 physics steps of a
contact-rich tour, sends individual seeds down different trajectories — so a
converted policy's result on a given seed is not comparable to its result
before the conversion, even though its statistics over a campaign are.

Each tour in a sweep pins itself to one core, so concurrent tours spread across
the machine. Physics is the cost, not inference: a tour is 60,000 `mj_step`
calls against 3,000 policy steps, and inference is under 3% of a run.

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
