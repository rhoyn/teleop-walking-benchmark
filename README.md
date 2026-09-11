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

144,384 runs — both engines, in rounds of 512 run ids. A run id names one
whole task. A policy earns depth by tier: tier A, the eleven that complete at
least 60 % averaged over the two engines, ran ten rounds, 5120 run ids; tier B
ran one, since a policy that falls most of the time needs no fourth decimal
place. The `runs` column carries each policy's own count, so no row is read
against a total it never had.

Both engines step on the GPU: MuJoCo through MuJoCo Warp, PhysX through its
GPU solver.

| `--policy` | mean<br>survival s | completed<br>mujoco/physx | runs | err<br>pos/yaw | walk<br>battery<br>energy<br>consumed | walk<br>vibrations |
|---:|---:|---:|---:|---:|---:|---:|
| `gr00t_wbc_h066_p012` | **84.1** | 79/74 % | 5120 | 30 cm / 8° | 8095 J | 1004 |
| `gr00t_wbc_h066_p000` | **84.1** | 80/72 % | 5120 | 34 cm / 8° | 8122 J | 1011 |
| `gr00t_wbc_h070_p000` | **83.6** | 73/75 % | 5120 | 17 cm / 5° | 8029 J | 1031 |
| `decoupled_wbc_h066_p000` | **82.8** | 74/69 % | 5120 | 15 cm / 6° | 9094 J | 2166 |
| `decoupled_wbc_h066_p012` | **82.7** | 75/68 % | 5120 | 15 cm / 6° | 9305 J | 2178 |
| `decoupled_wbc_h070_p000` | **82.2** | 72/65 % | 5120 | 12 cm / 7° | 8738 J | 2006 |
| `decoupled_wbc_h074_p000` | **81.1** | 69/62 % | 5120 | 12 cm / 7° | 8624 J | 1953 |
| `gr00t_wbc_h074_p000` | **81.0** | 63/67 % | 5120 | 14 cm / 5° | 8479 J | 1118 |
| `homie` | **78.6** | 55/58 % | 5120 | 17 cm / 34° | 9788 J | 1502 |
| `grove` | **77.7** | 49/63 % | 5120 | 21 cm / 7° | 12839 J | 2276 |
| `amo` | **77.7** | 54/62 % | 5120 | 30 cm / 17° | 10728 J | 1413 |
| `wbc_agile` | **69.1** | 34/27 % | 512 | 31 cm / 9° | 7828 J | 1338 |
| `sonic` | **61.6** | 36/15 % | 512 | 37 cm / 10° | 13788 J | 1704 |
| `mimic_lite` | **53.7** | 9/23 % | 512 | 84 cm / 12° | - | - |
| `run_residual` | **50.9** | 6/11 % | 512 | 585 cm / 13° | - | - |
| `asap` | **45.8** | 7/8 % | 512 | 135 cm / 33° | - | - |
| `robomimic` | **44.3** | 6/10 % | 512 | 190 cm / 15° | - | - |
| `falcon` | **39.0** | 2/4 % | 512 | 46 cm / 15° | - | - |
| `openwbt` | **35.7** | 0/6 % | 512 | 77 cm / 49° | - | - |
| `wty_cpp` | **28.3** | 0/4 % | 512 | 32 cm / 25° | - | - |
| `rl_lab` | **24.3** | 0/0 % | 512 | 63 cm / 75° | - | - |
| `handoff` | **22.0** | 0/0 % | 512 | 147 cm / 23° | - | - |
| `bfm_zero` | **11.7** | 0/0 % | 512 | 413 cm / 72° | - | - |
| `josabb`\* | **10.0** | 0/0 % | 512 | 111 cm / 59° | - | - |
| `huru`\* | **7.9** | 0/0 % | 512 | 147 cm / 60° | - | - |
| `sunny`\* | **7.5** | 0/0 % | 512 | 155 cm / 65° | - | - |
| `holosoma` | **6.2** | 0/0 % | 512 | 78 cm / 51° | - | - |
| `zealot` | **6.0** | 0/0 % | 512 | 322 cm / 89° | - | - |
| `rl_mjlab` | **5.4** | 0/0 % | 512 | 243 cm / 65° | - | - |
| `dm_march` | **5.3** | 0/0 % | 512 | 155 cm / 91° | - | - |
| `dm_agile` | **4.4** | 1/0 % | 512 | 55 cm / 88° | - | - |
| `legged_rl_lab` | **3.9** | 0/0 % | 512 | 220 cm / 82° | - | - |
| `mturan33`\* | **3.2** | 0/0 % | 512 | 213 cm / 90° | - | - |
| `schoi` | **3.1** | 0/0 % | 512 | 171 cm / 87° | - | - |
| `g1_gym` | **2.2** | 0/0 % | 512 | 112 cm / 86° | - | - |
| `nanog1` | **2.2** | 0/0 % | 512 | 86 cm / 59° | - | - |
| `wcompton` | **1.8** | 0/0 % | 512 | 133 cm / 69° | - | - |
| `clobot` | **1.8** | 0/0 % | 512 | 127 cm / 82° | - | - |
| `stepdown` | **1.6** | 0/0 % | 512 | 93 cm / 63° | - | - |
| `rl_gym` | **1.3** | 0/0 % | 512 | 105 cm / 77° | - | - |
| ~~`handoff_with_arms`~~\*\* | ~~**71.1**~~ | ~~34/44 %~~ | ~~512~~ | ~~15 cm / 8°~~ | ~~9303 J~~ | ~~1727~~ |
| ~~`clobot_with_arms`~~\*\* | ~~**41.0**~~ | ~~2/5 %~~ | ~~512~~ | ~~30 cm / 16°~~ | - | - |

`gr00t_wbc` is HOMIE v2, not an independent policy family. Its author
developed it at GEAR as the successor to `homie`, with substantial
optimisations across the stack, so `gr00t_wbc` and `homie` are the same lineage
rather than separate entries. Confirmed by @Elgce in InternRobotics/OpenHomie#23.

\*\* Unranked, because it is the same policy given all 29 joints instead of 15.

`mimic_lite` is MimicLite-ROA driven by the SONIC planner: the released ROA
tracker follows reference motion that the harness has `sonic`'s planner
generate from the same velocity command, since mimic-lite ships no planner of
its own.

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
