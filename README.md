# teleop-walking-benchmark

https://github.com/user-attachments/assets/a1d2430a-24ee-4409-8cc1-43f4147a1c67

Thirty-four open-source Unitree G1 walking policies behind one C++ interface,
scored on one tour in MuJoCo and PhysX.
[Blog post](https://rhoyn.com/stable-walk?utm_source=github)

## The task

Ninety seconds, eighteen waypoints, 50 Hz control over a 2 ms step. A crane
holds a shared stance three seconds, then releases. Each segment lands a punch:
random joint and direction, 0.08 s, up to 500 N — the ceiling ramps from a
third of that at the start to the full 500 N by the sixtieth second, and each
punch draws half to all of it. The score is recovery, not tracking. A policy
owns the fifteen leg and waist joints; the fourteen arm joints random-walk under
the harness, never still, never borrowable for balance. The best finishes most
runs, none all.

## Unified benchmark interface

Every policy and every physics engine sits behind the same CUDA interface, so
the harness runs the whole field batched on the GPU instead of one process per
policy per run id. That is what makes 144,384 runs cheap enough to be a
benchmark rather than a demo. Both engines step on the GPU: PhysX natively,
MuJoCo through [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).

## Running

```sh
./download_weights.sh && ./export_onnx.sh && make && make capture
./run.sh --policy gr00t_wbc_h074_p000 --engine physx --runids 0-255
./benchmark.sh          # whole field, both engines
```

Batches write `results/<tier>.r<NN>.<policy>.<engine>.csv`; `./build/table`
renders the table, problems land in `benchmark.log`. Policies are TensorRT
plans cached in `build/trt/`.

MuJoCo Warp has no C API, so `make capture` records one substep as a CUDA graph
into `build/mjwarp/` and `--engine mujoco` replays it from C++ with no Python at
run time. It will not start without that capture, and the capture fixes the
fleet size — re-record when the model or `--runs` changes:
`make capture MJWARP_NWORLD=256`.

## Results

144,384 runs — both engines, in rounds of 512 run ids. A run id names one
whole task. A policy earns depth by tier: tier A, the eleven rows that complete
at least 60 % averaged over the two engines, ran ten rounds, 5120 run ids; tier
B ran one, since a policy that falls most of the time needs no fourth decimal
place. The `runs` column carries each row's own count, so no row is read
against a total it never had.

Rows are ordered by mean survival over both engines. Errors are means over
targets reached; the walk energy and vibration columns are means over
completed runs, and an engine's cell is blank when fewer than a fifth of its
runs qualify. Both engines step on the GPU: MuJoCo through MuJoCo Warp, PhysX
through its GPU solver.

| `--policy` | mean<br>survival<br>sec<br>mujoco<br>physx | completed<br>percent<br>mujoco<br>physx | runs | pos<br>err<br>cm<br>mujoco<br>physx | yaw<br>err<br>deg<br>mujoco<br>physx | walk<br>energy<br>KJ<br>mujoco<br>physx | walk<br>vibrations<br>mujoco<br>physx |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `gr00t_wbc`<br>`h066_p012` | **84.9/83.4** | 79/74 | 5120 | 16/44 | 8/8 | 7.3/9.0 | 0.9/1.2 |
| `gr00t_wbc`<br>`h066_p000` | **85.2/82.9** | 80/72 | 5120 | 17/51 | 8/9 | 7.3/9.1 | 0.9/1.2 |
| `gr00t_wbc`<br>`h070_p000` | **83.3/83.8** | 73/75 | 5120 | 14/19 | 5/6 | 7.3/8.8 | 0.9/1.2 |
| `decoupled_wbc`<br>`h066_p000` | **83.8/81.9** | 74/69 | 5120 | 10/21 | 7/5 | 5.9/12.5 | 1.2/3.2 |
| `decoupled_wbc`<br>`h066_p012` | **83.9/81.5** | 75/68 | 5120 | 10/20 | 7/6 | 5.9/13.1 | 1.2/3.2 |
| `decoupled_wbc`<br>`h070_p000` | **83.3/81.0** | 72/65 | 5120 | 10/14 | 7/6 | 5.9/11.9 | 1.2/2.8 |
| `decoupled_wbc`<br>`h074_p000` | **82.3/79.9** | 69/62 | 5120 | 11/14 | 8/7 | 5.9/11.6 | 1.3/2.7 |
| `gr00t_wbc`<br>`h074_p000` | **80.2/81.8** | 63/67 | 5120 | 13/15 | 5/5 | 7.9/9.1 | 1.0/1.2 |
| `homie` | **78.2/79.0** | 55/58 | 5120 | 17/18 | 32/36 | 10.3/9.3 | 1.6/1.4 |
| `grove` | **74.9/80.5** | 49/63 | 5120 | 21/22 | 7/6 | 12.4/13.2 | 2.1/2.4 |
| `amo` | **75.9/79.5** | 54/62 | 5120 | 24/36 | 17/17 | 10.0/11.4 | 1.4/1.5 |
| `wbc_agile` | **71.2/67.1** | 34/27 | 512 | 23/39 | 8/10 | 7.7/8.0 | 1.3/1.4 |
| `sonic` | **69.2/54.0** | 36/15 | 512 | 29/44 | 9/12 | 11.2/- | 1.5/- |
| `mimic_lite` | **47.9/59.6** | 9/23 | 512 | 82/86 | 12/11 | -/17.4 | -/2.3 |
| `run_residual` | **50.7/51.1** | 6/11 | 512 | 581/589 | 14/13 | -/- | -/- |
| `asap` | **46.3/45.3** | 7/8 | 512 | 141/129 | 32/35 | -/- | -/- |
| `robomimic` | **44.7/43.8** | 6/10 | 512 | 195/185 | 14/16 | -/- | -/- |
| `falcon` | **37.7/40.2** | 2/4 | 512 | 42/49 | 14/16 | -/- | -/- |
| `openwbt` | **26.5/44.9** | 0/6 | 512 | 81/74 | 48/50 | -/- | -/- |
| `wty_cpp` | **23.0/33.7** | 0/4 | 512 | 31/32 | 24/25 | -/- | -/- |
| `rl_lab` | **25.5/23.2** | 0/0 | 512 | 57/68 | 74/76 | -/- | -/- |
| `handoff` | **21.0/22.9** | 0/0 | 512 | 144/149 | 23/24 | -/- | -/- |
| `bfm_zero` | **12.2/11.2** | 0/0 | 512 | 399/428 | 63/81 | -/- | -/- |
| `josabb`\* | **10.2/9.9** | 0/0 | 512 | 102/120 | 57/60 | -/- | -/- |
| `huru`\* | **7.8/8.1** | 0/0 | 512 | 149/145 | 59/60 | -/- | -/- |
| `sunny`\* | **7.2/7.8** | 0/0 | 512 | 150/161 | 63/67 | -/- | -/- |
| `holosoma` | **5.9/6.4** | 0/0 | 512 | 73/83 | 48/55 | -/- | -/- |
| `zealot` | **8.0/4.0** | 0/0 | 512 | 382/262 | 92/86 | -/- | -/- |
| `rl_mjlab` | **5.6/5.3** | 0/0 | 512 | 243/244 | 62/67 | -/- | -/- |
| `dm_march` | **6.3/4.3** | 0/0 | 512 | 147/162 | 91/91 | -/- | -/- |
| `dm_agile` | **5.6/3.2** | 1/0 | 512 | 58/52 | 88/88 | -/- | -/- |
| `legged_rl_lab` | **3.9/3.8** | 0/0 | 512 | 214/226 | 84/80 | -/- | -/- |
| `mturan33`\* | **2.7/3.8** | 0/0 | 512 | 211/214 | 89/91 | -/- | -/- |
| `schoi` | **3.6/2.6** | 0/0 | 512 | 200/142 | 87/87 | -/- | -/- |
| `g1_gym` | **2.4/1.9** | 0/0 | 512 | 112/111 | 87/86 | -/- | -/- |
| `nanog1` | **2.1/2.2** | 0/0 | 512 | 82/90 | 58/60 | -/- | -/- |
| `wcompton` | **2.0/1.6** | 0/0 | 512 | 127/138 | 63/76 | -/- | -/- |
| `clobot` | **1.8/1.8** | 0/0 | 512 | 129/124 | 83/81 | -/- | -/- |
| `stepdown` | **1.6/1.6** | 0/0 | 512 | 93/93 | 67/59 | -/- | -/- |
| `rl_gym` | **1.2/1.4** | 0/0 | 512 | 100/111 | 81/73 | -/- | -/- |
| ~~`handoff`<br>`with_arms`~~\*\* | ~~**69.4/72.8**~~ | ~~34/44~~ | ~~512~~ | ~~14/16~~ | ~~8/8~~ | ~~7.6/10.6~~ | ~~1.3/2.1~~ |
| ~~`clobot`<br>`with_arms`~~\*\* | ~~**43.8/38.2**~~ | ~~2/5~~ | ~~512~~ | ~~29/31~~ | ~~15/18~~ | ~~-/-~~ | ~~-/-~~ |

`gr00t_wbc` is HOMIE v2, not an independent policy family. Its author
developed it at GEAR as the successor to `homie`, with substantial
optimisations across the stack, so `gr00t_wbc` and `homie` are the same lineage
rather than separate entries. Confirmed by @Elgce in InternRobotics/OpenHomie#23.

\* Reads the base linear velocity straight from the simulator. No real robot
has that sensor, so the row is ranked but the policy is not deployable as
shipped.

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
buys completion by giving up tracking: 0.74 → 0.70 m is worth ten points of
completion in MuJoCo and eight in PhysX for one and four centimetres of
position error; 0.66 m is another seven points in MuJoCo for three more
centimetres, while in PhysX it gives three points back and the error more than
doubles, 19 → 44-51 cm. `decoupled_wbc` pays far less — 0.70 → 0.66 m gains
two to four points, and its position error holds at 10-11 cm in MuJoCo, the
lowest in the table, though PhysX drifts from 14 to 20-21 cm. Commanded pitch
does nothing for completion in either.

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

Third-party; terms in [NOTICE](NOTICE) — two non-commercial, eleven silent,
one sim-only. Nineteen weight files committed, the rest fetched or converted
locally, since a licence travels to its conversion.
