# teleop-walking-benchmark

https://github.com/user-attachments/assets/26b80464-e7a7-4beb-b1a8-79c8715dfaa3

Twenty-nine open-source Unitree G1 walking policies behind one C++ interface,
scored on one tour in MuJoCo and PhysX.
[Blog post](https://rhoyn.com/stable-walk?utm_source=github)

## The task

Sixty seconds, twelve waypoints, 50 Hz control over a 2 ms step. A crane holds
a shared stance three seconds, then releases. Each segment lands a punch:
random joint and direction, 0.08 s, up to 600 N. The score is recovery, not
tracking. A policy owns the fifteen leg and waist joints; the fourteen arm
joints random-walk under the harness, never still, never borrowable for
balance. The best finishes most runs, none all.

## Unified benchmark interface

Every policy and every physics engine sits behind the same CUDA interface, so
the harness runs the whole field batched on the GPU instead of one process per
policy per run id. That is what makes 471,424 runs cheap enough to be a
benchmark rather than a demo. Both engines step on the GPU: PhysX natively,
MuJoCo through [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).

## Running

```sh
./download_weights.sh && ./export_onnx.sh && make && make capture
./run.sh --policy gr00t_wbc_h074_p000 --engine physx --runids 0-255
./benchmark.sh          # whole field, both engines
```

Batches write `results/r<NN>.<policy>.<engine>.csv`; `./build/table` renders
the table, problems land in `benchmark.log`. Policies are TensorRT plans cached
in `build/trt/`; inference is under 3% of a run.

MuJoCo Warp has no C API, so `make capture` records one substep as a CUDA graph
into `build/mjwarp/` and `--engine mujoco` replays it from C++ with no Python at
run time. It will not start without that capture, and the capture fixes the
fleet size — re-record when the model or `--runs` changes:
`make capture MJWARP_NWORLD=256`.

## Results

608,256 runs — four rounds of 2048 run ids, both engines, except that `sonic`
runs 256 a round. A run id names one whole task.

Both engines step on the GPU: MuJoCo through MuJoCo Warp, PhysX through its
GPU solver.

| `--policy` | completed<br>mujoco/physx | err<br>pos/yaw | walk<br>battery<br>energy<br>consumed | walk<br>vibrations |
|---:|---:|---:|---:|---:|
| `gr00t_wbc_h066_p012` | **87/78 %** | 36 cm / 8° | 5819 J | 721 |
| `gr00t_wbc_h066_p000` | **87/77 %** | 40 cm / 8° | 5837 J | 727 |
| `gr00t_wbc_h070_p000` | **83/80 %** | 19 cm / 5° | 5749 J | 735 |
| `gr00t_wbc_h074_p000` | **76/75 %** | 16 cm / 5° | 6090 J | 786 |
| `decoupled_wbc_h066_p000` | **78/72 %** | 18 cm / 6° | 7503 J | 1656 |
| `decoupled_wbc_h066_p012` | **78/71 %** | 17 cm / 6° | 7547 J | 1657 |
| `decoupled_wbc_h070_p000` | **75/69 %** | 15 cm / 6° | 7203 J | 1557 |
| `decoupled_wbc_h074_p000` | **72/65 %** | 15 cm / 7° | 7203 J | 1532 |
| `homie` | **68/67 %** | 21 cm / 40° | 7175 J | 1085 |
| `grove` | **62/73 %** | 24 cm / 7° | 9426 J | 1707 |
| `amo` | **65/72 %** | 36 cm / 16° | 7553 J | 996 |
| `sonic` | **55/26 %** | 39 cm / 10° | 10160 J | 1219 |
| `wbc_agile` | **41/34 %** | 39 cm / 10° | 5979 J | 1015 |
| `mimic_lite` | **20/30 %** | 84 cm / 12° | 11044 J | 1489 |
| `robomimic` | **18/17 %** | 183 cm / 16° | - | - |
| `run_residual` | **14/17 %** | 530 cm / 15° | - | - |
| `asap` | **13/16 %** | 123 cm / 34° | - | - |
| `falcon` | **4/6 %** | 52 cm / 18° | - | - |
| `openwbt` | **0/7 %** | 83 cm / 51° | - | - |
| `rl_gym` | **0/5 %** | 102 cm / 24° | - | - |
| `handoff` | **1/1 %** | 144 cm / 24° | - | - |
| `rl_lab` | **1/1 %** | 72 cm / 72° | - | - |
| `dm_agile` | **1/0 %** | 56 cm / 90° | - | - |
| `wty_cpp` | **1/0 %** | 38 cm / 26° | - | - |
| `bfm_zero` | **0/0 %** | 423 cm / 71° | - | - |
| `holosoma` | **0/0 %** | 84 cm / 50° | - | - |
| `zealot` | **0/0 %** | 311 cm / 87° | - | - |
| `rl_mjlab` | **0/0 %** | 239 cm / 64° | - | - |
| `dm_march` | **0/0 %** | 151 cm / 91° | - | - |
| `legged_rl_lab` | **0/0 %** | 222 cm / 83° | - | - |
| `schoi` | **0/0 %** | 173 cm / 88° | - | - |
| `g1_gym` | **0/0 %** | 108 cm / 89° | - | - |
| `nanog1` | **0/0 %** | 84 cm / 60° | - | - |
| `wcompton` | **0/0 %** | 136 cm / 65° | - | - |
| `clobot` | **0/0 %** | 131 cm / 78° | - | - |
| `stepdown` | **0/0 %** | 97 cm / 61° | - | - |
| ~~`handoff_with_arms`~~\*\* | ~~**48/50 %**~~ | ~~17 cm / 9°~~ | ~~6838 J~~ | ~~1227~~ |
| ~~`clobot_with_arms`~~\*\* | ~~**9/6 %**~~ | ~~39 cm / 20°~~ | - | - |

`gr00t_wbc` is HOMIE v2, not an independent policy family. Its author
developed it at GEAR as the successor to `homie`, with substantial
optimisations across the stack, so `gr00t_wbc` and `homie` are the same lineage
rather than separate entries. Confirmed by @Elgce in InternRobotics/OpenHomie#23.

\*\* Unranked, because it is the same policy given all 29 joints instead of 15.

### Why two policies appear four times each

`gr00t_wbc` and `decoupled_wbc` are each one checkpoint under four commanded
postures: `h` is torso height in metres, `p` torso pitch in radians. Every row
owns the same fifteen joints and takes the same commands as any other entry, so
every row is ranked. `gr00t_wbc_h074_p000` and `decoupled_wbc_h070_p000` are
the postures their authors call canonical.

Commanded height is the only thing that moves the score, and for `gr00t_wbc` it
buys completion by giving up tracking: 0.74 → 0.70 m is worth six points of
completion for four centimetres of position error, and 0.66 m another four
points for seventeen more centimetres. `decoupled_wbc` pays no such price —
0.70 → 0.66 m gains three points and its position error stays at 12-13 cm,
the lowest in the table. Commanded pitch does nothing for completion in either.

A wider sweep over `gr00t_wbc`'s walk/balance switch found it inert, so it is
not reported. Read every one of these rows against the error column as well as
the completion column.

The `decoupled_wbc` sweep was suggested by @chrisyrniu in
https://github.com/chrisyrniu/IsaacLab-Decoupled-WBC/issues/1#issuecomment-5596278961

### What a low score means

This benchmark scores a **walking policy that controls the lower body while the
upper body is driven externally**. The 15 leg and waist joints belong to the
policy; the 14 arm joints random-walk under the harness for the whole run.

Several policies score badly because they were trained to swing their arms to
keep balance, and that is not available to them here — the arms are assumed to
belong to a separate policy. **A low score is therefore not always a statement
that the policy is bad.** It often means the policy was never built for
lower-body balance under external upper-body control.

## MuJoCo vs Isaac Sim engine: PhysX

Each row is one policy run twice — the same exported weights, the same
waypoints, the same punches — stepped by two different physics engines. Nothing
about the policy changes between the two halves of `completed`; only the
contact model does.

They disagree because contact is solved differently: MuJoCo integrates soft
convex constraints, PhysX runs a TGS solver over rigid contact patches, and
neither is the real robot. A policy that scores in one engine has learned that
engine's contacts; one that scores in both survives a change of contact model
it never saw in training, which is the best proxy here for sim-to-real
transfer. Read the two numbers together, not separately.

Neither engine repeats a run id bit for bit: both step on the GPU and reduce
contact forces across threads in arbitrary order, so the same run id can pass
one round and fall the next. Compare aggregates across rounds, not individual
run ids.

## Weights

Third-party; terms in [NOTICE](NOTICE) — two non-commercial, nine silent, one
sim-only. Nineteen committed, the rest fetched or converted locally, since a
licence travels to its conversion. `wcompton` needs your own copy.
