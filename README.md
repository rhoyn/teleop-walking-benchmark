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

**The arms are not the policy's to use.** This is a benchmark for a walking
policy you sit *underneath* while you drive the upper body yourself, so every
candidate is scored on the same 15 lower-body joints — twelve in the legs, three
in the waist — and the 14 arm joints are held at `DEFAULT_ANGLES` on shared
gains for the whole tour. Nothing a policy does can move them, and nothing it
does can borrow them.

That is harder than the alternative, and deliberately so. A humanoid's arms are
a real part of how it stays upright: swing them and you can shed momentum a fall
would otherwise keep. Take them away and a policy has to hold its balance on its
legs alone. But an operator teleoperating the robot is *using* those arms to do
a task, and the walking policy underneath cannot make plans that depend on them
— a controller that needs to windmill to stay standing is a controller that
falls the moment its operator reaches for a door handle. Holding the arms still
is not a handicap the benchmark imposes for difficulty's sake; it is the
condition the policy will actually run in. That it also makes the task harder is
the point rather than the cost. `clobot` is the extreme case and is discussed
below.

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

| `--policy` | completed | survived | pos err | yaw err |
|---:|---:|---:|---:|---:|
| `gr00t_wbc` | **78.9%** | 56.7 s | 9.5 cm | 2.9° |
| `homie` | **71.0%** | 55.6 s | 17.9 cm | 34.5° |
| `amo` | **69.9%** | 54.5 s | 23.0 cm | 12.1° |
| `robomimic` | **65.2%** | 54.9 s | 11.1 cm | 3.6° |
| `asap` | **53.5%** | 52.5 s | 13.2 cm | 9.9° |
| `rl_mjlab` | **45.9%** | 51.0 s | 14.7 cm | 3.4° |
| `holosoma` | **41.7%** | 50.1 s | 18.1 cm | 3.1° |
| `run_residual` | **14.4%** | 39.2 s | 432.3 cm | 3.2° |
| `rl_lab` | **7.3%** | 36.3 s | 13.7 cm | 65.5° |
| `falcon` | **2.3%** | 28.3 s | 33.6 cm | 4.1° |
| `rl_gym` | **1.1%** | 23.9 s | 57.9 cm | 6.2° |
| `openwbt` | **1.0%** | 26.1 s | 10.4 cm | 35.9° |
| `clobot` | **0.0%** | 2.2 s | - | - |
| ~~`clobot_with_arms`~~\*\* | ~~**8.6%**~~ | ~~33.1 s~~ | ~~33.8 cm~~ | ~~5.0°~~ |

\*\* Struck through because it does not rank. `clobot_with_arms` is the same
checkpoint as `clobot`, wired to own all 29 joints instead of the 15 every other
row drives — a harness configuration rather than a fourteenth policy. It sits
under `clobot` so the two can be read against each other, and not against the
rows above them.

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
`openwbt`; at ten thousand seeds it takes that too, 9.5 cm against 10.4 cm,
and this time on a row that finishes four fifths of its tours rather than none.
It clears second place by eight points of completion, 78.9% against `homie`'s
71.0%. First place is not close.

**Ten thousand seeds leave almost nothing tied.** No 95% Wilson interval in the
completion column is wider than two points, so every gap in it is real except
one: `homie` and `amo` finish 7104 and 6987 tours, 70.1-71.9% against
69.0-70.8%, close enough that their intervals still touch. Everything else separates cleanly — `robomimic` is
fourth, `asap` fifth, and the order below them holds. The sample size is past
the point of being the limiting factor: what separates these policies now is the
tour, not the statistics.

**`openwbt` is accurate and still cannot finish**: second-best mean position
error in the field and 104 completions in ten thousand tries. Its 35.9° yaw
error is where it goes — capped at its published 0.3 m/s, it cannot hold heading
on the harder draws.

**`clobot` fails on every seed, but no longer identically.** Not one scored
target in ten thousand tours, and a mean survival of 2.2 s against a worst of
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
0-9999. Giving the policy its arms back is worth 855 tours against none, which
would place it above `rl_lab` rather than at the bottom of the field.

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

### What the tour costs

Summing those columns across all five joint groups and accumulating them from
the release gives what a policy had spent by the time it reached a given leg.
Each column is then given as a percentage of its own mean, so **100% is what
that leg cost the average policy** and the figure says how a candidate compares
to the field rather than how many joules it drew. The mean is over the ranked
policies with a figure in that column; `clobot_with_arms` is scaled against the
same mean without contributing to it. Read at every second leg:

| `--policy` | leg 0 | leg 2 | leg 4 | leg 6 | leg 8 | leg 10 |
|---:|---:|---:|---:|---:|---:|---:|
| `gr00t_wbc` | 77% | 77% | 77% | 78% | 91% | 89% |
| `homie` | 105% | 106% | 105% | 105% | 122% | 119% |
| `amo` | 91% | 84% | 83% | 84% | 98% | 98% |
| `robomimic` | 66% | 67% | 68% | 70% | 82% | 80% |
| `asap` | 73% | 74% | 75% | 76% | 89% | 87% |
| `rl_mjlab` | 85% | 86% | 88% | 91% | 106% | 104% |
| `holosoma` | 77% | 80% | 81% | 81% | 93% | 89% |
| `run_residual` | 108% | 112% | 118% | 120% | 139% | 133% |
| `rl_lab` | 60% | 64% | 67% | 69% | 81% | - |
| `falcon` | 81% | 84% | 83% | 82% | - | - |
| `rl_gym` | 294% | 291% | 280% | 269% | - | - |
| `openwbt` | 82% | 76% | 75% | 74% | - | - |
| `clobot` | - | - | - | - | - | - |
| ~~`clobot_with_arms`~~\*\* | ~~84%~~ | ~~82%~~ | ~~81%~~ | ~~79%~~ | ~~89%~~ | - |

| `--policy` | leg 0 | leg 2 | leg 4 | leg 6 | leg 8 | leg 10 |
|---:|---:|---:|---:|---:|---:|---:|
| `gr00t_wbc` | 64% | 65% | 66% | 67% | 98% | 97% |
| `homie` | 84% | 86% | 86% | 86% | 125% | 122% |
| `amo` | 88% | 83% | 82% | 83% | 120% | 119% |
| `robomimic` | 61% | 63% | 64% | 65% | 94% | 93% |
| `asap` | 60% | 61% | 61% | 62% | 91% | 89% |
| `rl_mjlab` | 62% | 64% | 66% | 67% | 98% | 96% |
| `holosoma` | 39% | 44% | 46% | 47% | 70% | 69% |
| `run_residual` | 70% | 74% | 79% | 81% | 118% | 115% |
| `rl_lab` | 51% | 54% | 57% | 59% | 86% | - |
| `falcon` | 62% | 66% | 67% | 67% | - | - |
| `rl_gym` | 437% | 433% | 423% | 413% | - | - |
| `openwbt` | 120% | 107% | 105% | 104% | - | - |
| `clobot` | - | - | - | - | - | - |
| ~~`clobot_with_arms`~~\*\* | ~~51%~~ | ~~50%~~ | ~~50%~~ | ~~50%~~ | ~~73%~~ | - |

**A cell is blank unless at least 2000 runs got that far.** The mean is taken
only over runs whose leg *i* ran its full five-second clock, and a policy that
rarely reaches a leg has no honest average there. Fourteen of the eighty-four
cells fall under the floor: `clobot` never finishes a leg and has none at all;
`rl_gym`, `openwbt` and `falcon` run out after leg 6, having reached leg 8 only
897, 992 and 1472 times; and `rl_lab` and `clobot_with_arms` lose the last
column alone, at 1287 and 1364. `run_residual` keeps its at 2153, barely.

**Read these as survivor figures, and read them down rather than across.** Each
column averages a different subset of each policy's runs — everything that got
that far — so a row thins out as it moves right, and by leg 10 `gr00t_wbc` is
averaging 8416 runs against `run_residual`'s 2153. The columns say what a tour
cost the policies that were still walking, not what it costs on average.

Because each column is normalised against its own mean, and because the floor
changes which policies that mean is taken over, the jump nearly every row makes
between leg 6 and leg 8 is not the policies getting more expensive. It is
`rl_gym`, `openwbt` and `falcon` dropping out and taking their costs with them:
`rl_gym` alone sat at three to four times the field, and removing it pulls the
mean down and everyone else's percentage up. A row is only comparable within a
column. In absolute terms the field averaged 361.9 J and 704.4 krad/s² by the
end of leg 0, and 4872.1 J and 6358.6 krad/s² by the end of leg 10.

**`rl_gym` is the outlier that explains its row.** It draws 294% of the field's
energy in the first leg and shakes at 437%, against `gr00t_wbc`'s 77% and 64%.
It is not walking so much as vibrating along the tour, and it completes 1.1%.

**Neither cheap nor smooth means good.** `rl_lab` spends the least energy in
every column it appears in and completes 7.3%; `holosoma` shakes the least in
every column and completes 41.7%. `gr00t_wbc` wins the table above while sitting
mid-field on both — it is not the most efficient policy or the smoothest, it is
the one still standing. Cost separates *how* a policy walks; only the completion
column says whether it can.

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
