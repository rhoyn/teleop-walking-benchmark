# teleop-walking-benchmark

Blog post: [Stable walk](https://rhoyn.com/stable-walk?utm_source=github)

Twenty-eight open-source Unitree G1 walking policies ported to one C++ interface
and scored against the same waypoint tour in MuJoCo. Twenty-nine of the
thirty-three checkpoints are committed under `policies/`; `./download_weights.sh`
pulls three more and `./export_onnx.py` converts the last from a fetched
checkpoint.

The difficulty is deliberate. A benchmark every candidate completes ranks
nothing — it separates no one, it only shows the task was too easy. This one is
tuned to sit at the edge of what these policies can do: the best of them
finishes most seeds, none finishes all, and no row is a perfect score. The
punches are what make it a test of stability and disaster recovery rather than
of walking — a policy is judged on whether it can absorb a hit it never saw
coming and get back on the tour, not only on how neatly it tracks a target
while nothing is going wrong.

**The arms are not the policy's to use, and they will not hold still.** This is
a benchmark for a walking policy you sit *underneath* while you drive the upper
body yourself, so every candidate is scored on the same 15 lower-body joints —
twelve in the legs, three in the waist — while the 14 arm joints are driven on
shared gains by the tour itself. Nothing a policy does can move them, and
nothing it does can borrow them.

What the tour drives them with is a random walk. Every control step adds a
uniform draw to the arm targets it already holds, so the arms never settle: no
trajectory, no task, just a load that keeps moving. The two arms are posed as
mirror images of each other, which keeps the disturbance fore-and-aft rather
than a permanent list to one side, and a drawn pose is kept only if it leaves
both hands in front of the robot, below its head, inside the joint limits and
through neither the robot nor the floor. A pose that fails any of those is
thrown away and the draw taken again. Everything inside that is fair game: a
policy cannot learn where the hands will be, only that they are somewhere ahead
of the chest and moving. The walk is drawn from the run's seed like the
waypoints and the punches, so a seed still names one whole task and a repeated
seed still reproduces bit for bit.

That is harder than the alternative, and deliberately so. A humanoid's arms are
a real part of how it stays upright: swing them and you can shed momentum a fall
would otherwise keep. Take them away and a policy has to hold its balance on its
legs alone. But an operator teleoperating the robot is *using* those arms to do
a task, and the walking policy underneath cannot make plans that depend on them
— a controller that needs to windmill to stay standing is a controller that
falls the moment its operator reaches for a door handle. It is also not enough
to take the arms away and leave them parked: an arm being worked is a mass
moving around above the hips, and a policy that only ever met a still one has
not met the condition it will actually run in. That the moving arms also make
the task harder is the point rather than the cost. `clobot` is the extreme case
and is discussed below.

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

What each waypoint of the tour cost follows as `s<i>_` columns, one block of ten per
target clock, twelve blocks in all. A waypoint block spans one target's clock:
`s5_` covers the stretch walked while target 5 was being asked for.

- **`s<i>_e_<group>_j`** is absolute mechanical work, the sum of `|τ·ω|·dt` over
  every 1 ms physics step of that waypoint, in joules. The absolute value is what
  makes it an actuator cost rather than a physics one — these motors burn
  current holding a limb against gravity whichever way it is moving, and a
  signed integral would let a limb swinging down pay back the lift that raised
  it.
- **`s<i>_v_<group>_krads2`** is how much the group shook: the total variation
  of each joint's angular acceleration, the sum of `|α(t) − α(t−dt)|` over the
  waypoint, added across the group, in thousands of rad/s².

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

A run that fell leaves the waypoints it never reached empty, and **nothing marks
which waypoints ran their full clock because nothing needs to.** Waypoints `0` to
`targets - 1` each had their whole five seconds; a block present at index
`targets` is the waypoint the run died in, running from `5 × targets` seconds to
`survival_s`. Both quantities are sums over the waypoint rather than rates, so that
last partial block covers less than a target clock and should be dropped before
pooling.

`--record DIR` writes `DIR/<policy>.mp4` at 50 fps, one frame per 0.02 s of
simulated time, so playback is exactly real time however fast the run computed.
The clip carries the policy name on the floor and nothing else; a punch draws a
red arrow at the contact point, sized by the force and held for 500 ms after the
hit.

## The 50-seed campaign

Seeds 0-49, every policy over every seed: **1,450 runs**, 9.8 hours of simulated
walking, eight tours in flight on an RTX 4090 / Ryzen 7 7800X3D. No run errored
and no run was interrupted, so every row in `results/result.csv` is a
measurement. Every policy runs the same seed range into the same file, so the
whole table is one campaign.

**Fifty seeds is a small sample, and the table has to be read as one.** A 95%
Wilson interval on a completion figure in the middle of the field is about
twenty-five points wide — `decoupled_wbc`'s 74% spans 60.4-84.1 and
`gr00t_wbc`'s 64% spans 50.1-75.9 — so adjacent rows are not separated by
this campaign. The column that is solid is the difference between a policy that
finishes tours and one that finishes none.

Every number below was measured before the arms were set walking, with the 14
arm joints parked at `DEFAULT_ANGLES` for the whole tour. The campaign has not
been re-run since, so read the table as the ranking of the parked-arm tour; the
moving arms are a harder task and the rows will move.

Each seed draws its own 12-target tour and its own punch campaign, so a seed is
a whole task, not a repetition — the runs are deterministic and a repeated seed
reproduces bit-identically.

Every policy is handed the same stance. The crane ramps the robot to the shared
`DEFAULT_ANGLES` pose over three seconds and then lets go, rather than to a pose
each checkpoint chose for itself, so what is measured is how well a policy takes
over from a stance it was not necessarily trained around. The arms start
walking at the release and not before, from that same stance, so no candidate
is handed its robot mid-gesture.

| `--policy` | completed | survived | pos err | yaw err | tour<br>battery<br>energy<br>consumed | tour<br>vibrations |
|---:|---:|---:|---:|---:|---:|---:|
| `decoupled_wbc` | **74%** | 55 s | 8 cm | 4° | 5038 J | 10480 |
| `homie` | **64%** | 54 s | 17 cm | 35° | 6671 J | 12510 |
| `gr00t_wbc` | **64%** | 53 s | 11 cm | 3° | 5636 J | 11251 |
| `amo` | **62%** | 53 s | 27 cm | 14° | 6678 J | 12759 |
| `grove` | **58%** | 53 s | 17 cm | 3° | 8288 J | 12254 |
| `wbc_agile` | **52%** | 51 s | 14 cm | 4° | 5399 J | 9266 |
| `sonic` | **50%** | 50 s | 14 cm | 5° | 7143 J | 12621 |
| `asap` | **28%** | 41 s | 81 cm | 27° | 6523 J | 10813 |
| `wty_cpp` | **18%** | 38 s | 14 cm | 7° | - | - |
| `robomimic` | **16%** | 39 s | 100 cm | 4° | - | - |
| `run_residual` | **10%** | 37 s | 545 cm | 4° | - | - |
| `falcon` | **6%** | 29 s | 33 cm | 6° | - | - |
| `rl_lab` | **2%** | 22 s | 16 cm | 68° | - | - |
| `openwbt` | **0%** | 22 s | 38 cm | 40° | - | - |
| `handoff` | **0%** | 19 s | 86 cm | 5° | - | - |
| `rl_gym` | **0%** | 15 s | 48 cm | 6° | - | - |
| `rl_mjlab` | **0%** | 6 s | 154 cm | 20° | - | - |
| `zealot` | **0%** | 6 s | 285 cm | 97° | - | - |
| `holosoma` | **0%** | 5 s | 49 cm | 19° | - | - |
| `dm_march` | **0%** | 5 s | 166 cm | 99° | - | - |
| `legged_rl_lab` | **0%** | 4 s | 130 cm | 33° | - | - |
| `dm_agile` | **0%** | 3 s | 67 cm | 93° | - | - |
| `schoi` | **0%** | 3 s | 187 cm | 61° | - | - |
| `g1_gym` | **0%** | 2 s | 72 cm | 22° | - | - |
| `nanog1` | **0%** | 2 s | 113 cm | 3° | - | - |
| `clobot` | **0%** | 2 s | - | - | - | - |
| `stepdown` | **0%** | 2 s | - | - | - | - |
| `wcompton` | **0%** | 2 s | - | - | - | - |
| ~~`clobot_with_arms`~~\*\* | ~~**6%**~~ | ~~33 s~~ | ~~35 cm~~ | ~~5°~~ | - | - |

\*\* Struck through because it does not rank. `clobot_with_arms` is the same
checkpoint as `clobot`, wired to own all 29 joints instead of the 15 every other
row drives — a harness configuration rather than a fourteenth policy. It sits
under `clobot` so the two can be read against each other, and not against the
rows above them.

**Fifteen of the twenty-eight complete no tour at all**, and a further five
finish under a fifth of theirs, so read the error columns below the top eight
with care. Those figures rest on whichever fragments of a tour the policy
reached before falling, which is not the same measurement as the rows above
them: `nanog1`'s 3° is a heading it held for two seconds.

Only eight policies clear the observation floor on the cost columns and get a
figure there at all. The rest are blank, which is the floor doing its job
rather than data missing.

Survival is sim seconds to the fall **measured from the crane's release**, not
from the start of the simulation: the three seconds the crane spends ramping the
robot to the shared pose are seconds it could not fall in, and counting them
would hand every policy the same three-second floor and flatter the worst of
them most. The first target is demanded the moment the crane lets go, so a
completed tour is exactly 60 s — twelve targets on a 5 s clock, the last of them
closing on the minute. The punch campaign is drawn from the seed but timed from
that same release, so punch *i* lands a tenth of a second into waypoint *i* —
far enough in that the policy has taken the new demand and started to act on it,
so the punch disturbs the walking rather than the handover. Position and yaw
errors are means over every target actually scored, so a policy that falls early
is judged only on the targets it reached — that flatters the short-lived
candidates rather than penalising them. Step timings are deliberately absent:
the sweeps shared the machine fifteen at a time, which inflates them by an order
of magnitude. Measure those solo.

**`decoupled_wbc` leads the field, but fifty seeds cannot crown it.** It takes
completion at 74%, survival at 55 s and position error at 8 cm, and it is the
only policy to lead all three. Its interval nevertheless runs 60.4-84.1 and
overlaps every one of the six rows beneath it, so what this campaign supports is
that it belongs in the leading group, not that it is first. `homie` and
`gr00t_wbc` tie exactly on completion, 32 tours each, and are separated in the
table only by a second of survival.

**Fifty seeds leave almost everything tied.** No adjacent pair in the top eight
is separated by its Wilson intervals. Seven policies sit between 50% and 74% —
`decoupled_wbc`, `homie`, `gr00t_wbc`, `amo`, `grove`, `wbc_agile`, `sonic` —
and 24 to 26 points of interval is wider than the whole spread between them. The
gap the sample does support is the one below `asap`: the eight policies that
finish tours against the twenty that mostly or never do.

**Moving arms is the discriminator.** The tour now walks the arms and hands
under every policy, and it is a far harsher disturbance than the punches for
anything that was holding them still to stay upright. `rl_mjlab` and `holosoma`
finished eleven of twelve targets on the previous tour and now complete no tour
at all; `handoff` and `robomimic` lose most of theirs. The policies at the top
are the ones whose balance survives an upper body they do not control.

**`clobot` fails on every seed, but no longer identically.** Not one scored
target in fifty tours, and a mean survival of 1.9 s. On the old tour every run
was the same 5.3 s to three significant figures; now the first punch lands a
tenth of a second after the release and the arms move under it, spreading the
falls across 21 distinct times between 1.1 s and 3.3 s. The disturbances change
when it goes over, not whether.

That figure is the tour's rule rather than the gait. `clobot` is the only
checkpoint here trained as a whole-body policy whose balance depends on its own
arm motion — its deploy config carries a per-joint action scale, `kp` and `kd`
for all 29 joints, wrists included. This tour takes the arms away from every
candidate: whatever a policy leaves unowned is driven by the tour on shared
gains, and `clobot` is scored on the same 15 lower-body joints as the rest of
the field. Take the upper body away and it topples in five seconds,
every time. Parking the arms at its own nominal pose instead recovers none of
it — 5.9 s over 32 seeds — so what it wants is the arm swing, not the arm
pose.

In fairness to the checkpoint, the same weights are also wired up as
`--policy clobot_with_arms`, owning all 29 joints, and run over the same seeds
0-49. Giving the policy its arms back is worth 3 completed tours against none
and a mean survival of 33.0 s against 1.9 s — which now means owning the arms
the tour would otherwise be walking for it.

**Low error does not mean good.** `run_residual` cannot strafe, reverse or turn
in place — its command floor forces it forward at 0.5 m/s whenever it is off
target, so it can only approach on an arc — and 545 cm is a policy that cannot
do this task rather than a bad gait. `rl_lab`'s 16 cm and `wty_cpp`'s 14 cm are
earned on the fraction of each tour they survived. Read the completion column
first.

**Nothing completes reliably.** The best policy in the field fails a quarter of
its tours, twenty of the twenty-eight finish under a fifth of theirs, and
fifteen finish none. Surviving to the end of the tour is the discriminator, not
tracking error.

**The opening punch is its own filter.** It lands a tenth of a second after the
crane lets go, before any policy has taken a step, so it separates the policies
that cannot take a hit from a standing start rather than punishing the field at
random. At fifty seeds the per-policy share of runs it ends is too small a count
to quote; the ten-thousand-seed campaign put it at 4.2% of `rl_gym`'s runs and
2.8% of `openwbt`'s, against almost none for the leaders.

The runs behind this table are in [results/result.csv](results/result.csv), one
row per run, `clobot_with_arms` included — fifty rows that no ranked line of the
table draws on. Every run also carries what each waypoint of its tour cost,
joint group by joint group, in the `s<i>_` columns described above.

The table itself is computed by [table.cpp](table.cpp), which links nothing and
reads one campaign file at a time. `make table && ./build/table` reprints the
ranked rows, so the numbers above can be checked against the runs rather than
trusted:

```sh
make table && ./build/table \
  | diff - <(sed -n '/^| `--policy`/,/^$/p' README.md)
```

### What the tour costs

The last two columns come from the per-waypoint `s<i>_` columns, summed across
all five joint groups and over all twelve waypoints. They are averaged over
completed runs only — those are the only runs in which all twelve waypoints ran
their clock, and so the only ones whose totals mean the same thing — and are
blank unless at least a fifth of that policy's runs stand behind the average.
Eight policies clear that; the rest finish fewer than ten of their fifty tours,
and `clobot` finishes no waypoint at all. The floor is a share rather than a
count so that it means the same thing whatever the campaign size — at ten
thousand seeds it is the 2000 completed tours the previous campaign demanded.
These are survivor figures by construction: they say what a tour cost the runs
that completed it, not what it costs on average.

Every cost and error column is reported in whole units, which is as much
precision as a mean over fifty runs is worth reading. Energy itself is
mechanical work at the joints, `Σ|τ·ω|·dt`, which is a floor on what a battery
would actually deliver rather than the draw itself: it counts no drivetrain loss
and no current spent holding a limb still against gravity. Two policies an equal
distance apart on this column would be further apart on a real robot, not
closer.

**What the columns show is the price of a finish.** `homie` completes 64% to
`decoupled_wbc`'s 74% but pays 6671 J and 12510 for its tours against 5038 J
and 10480 — a third more energy and a fifth more shake for a worse result. This
time the leader is also the cheapest of the eight rows that carry figures, which
the previous campaign's winner was not; `wbc_agile` is the smoothest at 9266
against `decoupled_wbc`'s 10480 while completing 52%. The winner is simply the
one that converts what it spends into staying upright.

Every figure in these two columns is roughly half again what it was before the
tour began walking the arms — the arms are being driven through a random walk
now, and that work is counted.

**The rows with no figures are where the cost is most telling**, and the
per-waypoint columns in `results/result.csv` carry it even though this table
cannot. Over its opening waypoint `nanog1` spends less than any policy in the
field, 194 J, and completes none of its tours. `rl_gym` spends 1108 J there
against `gr00t_wbc`'s 309 and shakes 3270 against 809 — 3.6× the energy and 4.0×
the vibration — and also completes none. It is not walking so much as vibrating
along the tour. `schoi` is the extreme at 16155 J in that one waypoint, two
orders of magnitude above the cheapest, and falls at 3.1 s. Neither cheap nor
expensive predicts standing up; only the completion column does.

## Inference

Every candidate runs on the GPU, all twenty-eight as ONNX compiled to TensorRT
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

Thirty-three checkpoints across the twenty-eight policies. Twenty-nine are
committed beside the policy that loads them; `./download_weights.sh` pulls seven
files from their upstream repositories and checks each file's sha256 as it
lands.

A checkpoint's licence travels to anything derived from it, so the ONNX that
`./export_onnx.py` writes is shipped only where its source's terms allow.
`amo`'s and `rl_gym`'s conversions are committed because those checkpoints are
redistributable; the checkpoints themselves are fetched rather than committed,
which is a repository-size choice about PyTorch blobs and not a licence one.
`robomimic`'s checkpoint declares no licence at all, so its conversion is
generated on your machine rather than committed. Converting a file to another
format does not make it redistributable.

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
