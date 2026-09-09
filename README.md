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
benchmark rather than a demo — the one exception is MuJoCo, which is not GPU
friendly and steps on the CPU.

## Running

```sh
./download_weights.sh && ./export_onnx.py && make
./run.sh --policy gr00t_wbc --engine physx --runids 0-255
./benchmark.sh          # whole field, both engines
```

Batches write `results/r<NN>.<policy>.<engine>.csv`; `./build/table` renders
the table, problems land in `benchmark.log`. Policies are TensorRT plans cached
in `build/trt/`; inference is under 3% of a run.

## Results

538,880 runs — eight rounds of 1024 run ids, both engines, except that `sonic`
runs 128 a round and round 0 is short for `amo` and `asap`. A run id
names one whole task; MuJoCo repeats bit for bit, PhysX does not.

| `--policy` | completed<br>mujoco/physx | err<br>pos/yaw | walk<br>battery<br>energy<br>consumed | walk<br>vibrations |
|---:|---:|---:|---:|---:|
| `decoupled_wbc_h066_p012` | **77/84 %** | 12 cm / 6° | 5260 J | 1790 |
| `decoupled_wbc_h066_p000` | **78/83 %** | 13 cm / 6° | 5419 J | 1844 |
| `gr00t_wbc` | **77/78 %** | 15 cm / 5° | 6037 J | 1606 |
| `decoupled_wbc_h070_p000` | **75/80 %** | 13 cm / 6° | 5463 J | 1825 |
| `homie` | **69/73 %** | 21 cm / 39° | 7149 J | 1934 |
| `grove` | **62/77 %** | 21 cm / 6° | 8692 J | 2382 |
| `amo` | **64/73 %** | 36 cm / 16° | 7832 J | 1618 |
| `wbc_agile` | **40/38 %** | 35 cm / 10° | 5953 J | 1741 |
| `sonic` | **47/24 %** | 42 cm / 11° | 10947 J | 1814 |
| `mimic_lite` | **18/29 %** | 84 cm / 13° | 11719 J | 2216 |
| `robomimic` | **16/18 %** | 172 cm / 15° | - | - |
| `bfm_zero` | **22/12 %** | 157 cm / 44° | - | - |
| `run_residual` | **12/17 %** | 524 cm / 15° | - | - |
| `asap` | **12/16 %** | 123 cm / 34° | - | - |
| `falcon` | **3/6 %** | 51 cm / 17° | - | - |
| `openwbt` | **0/6 %** | 75 cm / 49° | - | - |
| `rl_gym` | **0/4 %** | 101 cm / 25° | - | - |
| `wty_cpp` | **2/1 %** | 39 cm / 27° | - | - |
| `handoff` | **0/1 %** | 144 cm / 24° | - | - |
| `dm_agile` | **1/1 %** | 55 cm / 89° | - | - |
| `rl_lab` | **0/0 %** | 78 cm / 73° | - | - |
| `zealot` | **0/0 %** | 352 cm / 87° | - | - |
| `holosoma` | **0/0 %** | 83 cm / 49° | - | - |
| `rl_mjlab` | **0/0 %** | 234 cm / 64° | - | - |
| `dm_march` | **0/0 %** | 149 cm / 90° | - | - |
| `legged_rl_lab` | **0/0 %** | 216 cm / 81° | - | - |
| `schoi` | **0/0 %** | 207 cm / 88° | - | - |
| `g1_gym` | **0/0 %** | 112 cm / 87° | - | - |
| `nanog1` | **0/0 %** | 82 cm / 61° | - | - |
| `wcompton` | **0/0 %** | 132 cm / 64° | - | - |
| `clobot` | **0/0 %** | 125 cm / 77° | - | - |
| `stepdown` | **0/0 %** | 95 cm / 63° | - | - |
| ~~`handoff_with_arms`~~\*\* | ~~**49/50 %**~~ | ~~17 cm / 9°~~ | ~~6920 J~~ | ~~1234~~ |
| ~~`clobot_with_arms`~~\*\* | ~~**9/5 %**~~ | ~~39 cm / 21°~~ | - | - |

`gr00t_wbc` is HOMIE v2, not an independent policy family. Its author
developed it at GEAR as the successor to `homie`, with substantial
optimisations across the stack, so `gr00t_wbc` and `homie` are the same lineage
rather than separate entries. Confirmed by @Elgce in InternRobotics/OpenHomie#23.

\*\* Unranked, because it is the same policy given all 29 joints instead of 15.

### Why `decoupled_wbc` appears three times

One policy and one checkpoint under three commanded postures: `h` is torso
height in metres, `p` torso pitch in radians. All three own the same fifteen
joints, so all three are ranked.

`h070_p000` is the posture its authors call canonical, and they suggested the
sweep. Height 0.70 → 0.66 m is worth **+2.5 points on MuJoCo, +3.8 on PhysX**,
90% intervals clear of it. Pitch adds nothing, yet `h066_p012` is the table's
lowest position error, energy and vibration.

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
convex constraints on the CPU, PhysX runs a TGS solver over rigid contact
patches on the GPU, and neither is the real robot. A policy that scores in one
engine has learned that engine's contacts; one that scores in both survives a
change of contact model it never saw in training, which is the best proxy here
for sim-to-real transfer. Read the two numbers together, not separately.

MuJoCo repeats a run id bit for bit. PhysX reduces contact forces across GPU
threads in arbitrary order, so the same run id can pass one round and fall the
next.

## Weights

Third-party; terms in [NOTICE](NOTICE) — two non-commercial, nine silent, one
sim-only. Nineteen committed, the rest fetched or converted locally, since a
licence travels to its conversion. `wcompton` needs your own copy.
