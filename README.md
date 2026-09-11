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

Batches write `results/<tier>.r<NN>.<policy>.<engine>.csv`; `./build/table`
renders the table, problems land in `benchmark.log`. Policies are TensorRT plans cached
in `build/trt/`; inference is under 3% of a run.

MuJoCo Warp has no C API, so `make capture` records one substep as a CUDA graph
into `build/mjwarp/` and `--engine mujoco` replays it from C++ with no Python at
run time. It will not start without that capture, and the capture fixes the
fleet size — re-record when the model or `--runs` changes:
`make capture MJWARP_NWORLD=256`.

## Results

140,800 runs — both engines, in rounds of 512 run ids. A run id names one
whole task. A policy earns depth by tier: tier A, the eleven that complete at
least 60 % averaged over the two engines, ran ten rounds, 5120 run ids; tier B
ran one, since a policy that falls most of the time needs no fourth decimal
place. `sonic` and `mimic_lite` are slow, so they run ten rounds of 64. The
`runs` column carries each policy's own count, so no row is read against a
total it never had.

Both engines step on the GPU: MuJoCo through MuJoCo Warp, PhysX through its
GPU solver.

| `--policy` | completed<br>mujoco/physx | runs | err<br>pos/yaw | walk<br>battery<br>energy<br>consumed | walk<br>vibrations |
|---:|---:|---:|---:|---:|---:|
| `gr00t_wbc_h066_p000` | **87/78 %** | 5120 | 40 cm / 8° | 5846 J | 729 |
| `gr00t_wbc_h066_p012` | **87/78 %** | 5120 | 36 cm / 8° | 5813 J | 721 |
| `gr00t_wbc_h070_p000` | **83/81 %** | 5120 | 19 cm / 5° | 5755 J | 735 |
| `gr00t_wbc_h074_p000` | **76/76 %** | 5120 | 15 cm / 5° | 6096 J | 787 |
| `decoupled_wbc_h066_p000` | **77/72 %** | 5120 | 18 cm / 6° | 7480 J | 1656 |
| `decoupled_wbc_h066_p012` | **78/71 %** | 5120 | 17 cm / 6° | 7557 J | 1658 |
| `decoupled_wbc_h070_p000` | **75/69 %** | 5120 | 15 cm / 6° | 7183 J | 1555 |
| `decoupled_wbc_h074_p000` | **72/67 %** | 5120 | 15 cm / 7° | 7209 J | 1534 |
| `grove` | **63/74 %** | 5120 | 24 cm / 7° | 9421 J | 1703 |
| `amo` | **65/71 %** | 5120 | 36 cm / 16° | 7574 J | 997 |
| `homie` | **68/67 %** | 5120 | 21 cm / 40° | 7183 J | 1084 |
| `sonic` | **57/28 %** | 640 | 37 cm / 10° | 10027 J | 1207 |
| `wbc_agile` | **40/31 %** | 512 | 44 cm / 11° | 6144 J | 1034 |
| `mimic_lite` | **25/27 %** | 640 | 81 cm / 12° | 11072 J | 1504 |
| `robomimic` | **17/17 %** | 512 | 173 cm / 15° | - | - |
| `asap` | **13/18 %** | 512 | 126 cm / 34° | - | - |
| `run_residual` | **13/16 %** | 512 | 533 cm / 15° | - | - |
| `falcon` | **3/2 %** | 512 | 53 cm / 19° | - | - |
| `openwbt` | **0/4 %** | 512 | 82 cm / 51° | - | - |
| `rl_gym` | **0/3 %** | 512 | 105 cm / 25° | - | - |
| `handoff` | **1/1 %** | 512 | 144 cm / 23° | - | - |
| `dm_agile` | **2/0 %** | 512 | 55 cm / 89° | - | - |
| `rl_lab` | **1/0 %** | 512 | 71 cm / 73° | - | - |
| `wty_cpp` | **0/0 %** | 512 | 50 cm / 27° | - | - |
| `bfm_zero` | **0/0 %** | 512 | 418 cm / 73° | - | - |
| `holosoma` | **0/0 %** | 512 | 85 cm / 49° | - | - |
| `zealot` | **0/0 %** | 512 | 315 cm / 87° | - | - |
| `rl_mjlab` | **0/0 %** | 512 | 243 cm / 65° | - | - |
| `dm_march` | **0/0 %** | 512 | 147 cm / 91° | - | - |
| `legged_rl_lab` | **0/0 %** | 512 | 224 cm / 84° | - | - |
| `schoi` | **0/0 %** | 512 | 173 cm / 88° | - | - |
| `g1_gym` | **0/0 %** | 512 | 109 cm / 89° | - | - |
| `nanog1` | **0/0 %** | 512 | 82 cm / 63° | - | - |
| `clobot` | **0/0 %** | 512 | 128 cm / 79° | - | - |
| `wcompton` | **0/0 %** | 512 | 134 cm / 67° | - | - |
| `stepdown` | **0/0 %** | 512 | 97 cm / 63° | - | - |
| ~~`handoff_with_arms`~~\*\* | ~~**48/51 %**~~ | ~~512~~ | ~~17 cm / 9°~~ | ~~6819 J~~ | ~~1229~~ |
| ~~`clobot_with_arms`~~\*\* | ~~**7/5 %**~~ | ~~512~~ | ~~41 cm / 21°~~ | - | - |

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
