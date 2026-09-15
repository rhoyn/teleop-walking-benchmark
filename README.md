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
targets reached; the walk energy, vibration and stomp energy columns are means
over completed runs, and an engine's cell is blank when fewer than a fifth of
its runs qualify. Both engines step on the GPU: MuJoCo through MuJoCo Warp,
PhysX through its GPU solver.

Stomp energy is how hard a policy puts its feet down: the work a foot does
on the floor per touchdown, in joules — the ankle force integrated against
the foot's downward speed over each stance, every 2 ms substep, averaged over
the touchdowns of completed runs. Per-segment values are in each CSV as
`s<N>_stomp_j`.

| `--policy` | mean<br>survival<br>sec<br>mujoco<br>physx | completed<br>percent<br>mujoco<br>physx | runs | pos<br>err<br>cm<br>mujoco<br>physx | yaw<br>err<br>deg<br>mujoco<br>physx | walk<br>energy<br>KJ<br>mujoco<br>physx | walk<br>vibrations<br>mujoco<br>physx | stomp<br>energy<br>J<br>mujoco<br>physx |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gr00t_wbc`<br>`h066_p012` | **84.9/82.2** | 80/69 | 5120 | 16/49 | 7/8 | 7.3/9.6 | 0.9/1.2 | 2.0/0.9 |
| `gr00t_wbc`<br>`h066_p000` | **84.7/81.8** | 79/67 | 5120 | 17/57 | 8/9 | 7.3/9.8 | 0.9/1.3 | 1.9/0.9 |
| `gr00t_wbc`<br>`h070_p000` | **83.3/82.9** | 73/71 | 5120 | 14/21 | 5/6 | 7.3/9.5 | 0.9/1.2 | 2.3/1.5 |
| `decoupled_wbc`<br>`h066_p000` | **83.9/81.3** | 74/67 | 5120 | 10/22 | 7/5 | 5.9/13.2 | 1.2/3.3 | 2.5/1.3 |
| `decoupled_wbc`<br>`h066_p012` | **83.9/80.9** | 75/66 | 5120 | 10/20 | 7/6 | 5.9/13.6 | 1.2/3.3 | 2.5/1.5 |
| `decoupled_wbc`<br>`h070_p000` | **83.1/80.2** | 72/63 | 5120 | 10/15 | 7/6 | 5.9/12.5 | 1.2/3.0 | 2.5/1.7 |
| `decoupled_wbc`<br>`h074_p000` | **82.3/80.1** | 69/62 | 5120 | 11/14 | 8/7 | 6.0/12.0 | 1.3/2.8 | 2.5/2.3 |
| `gr00t_wbc`<br>`h074_p000` | **80.1/80.0** | 63/61 | 5120 | 13/16 | 5/6 | 7.8/9.8 | 1.0/1.3 | 2.9/2.3 |
| `homie` | **78.3/77.4** | 55/52 | 5120 | 17/19 | 32/37 | 10.4/10.2 | 1.6/1.5 | 3.1/3.1 |
| `wbc_agile_velocity` | **75.1/79.3** | 49/60 | 5120 | 22/23 | 7/6 | 12.5/14.5 | 2.1/2.6 | 5.0/4.8 |
| `amo` | **75.7/77.6** | 53/57 | 5120 | 24/39 | 18/17 | 10.0/12.2 | 1.4/1.5 | 2.7/2.8 |
| `wbc_agile` | **70.9/64.3** | 33/18 | 512 | 23/41 | 8/11 | 7.7/- | 1.3/- | 3.3/- |
| `sonic` | **68.5/52.9** | 36/11 | 512 | 30/44 | 9/11 | 11.5/- | 1.5/- | 3.4/- |
| `mimic_lite` | **48.1/59.5** | 9/21 | 512 | 76/89 | 12/12 | -/17.7 | -/2.3 | -/8.0 |
| `run_residual` | **51.7/48.1** | 8/5 | 512 | 589/586 | 13/14 | -/- | -/- | -/- |
| `asap` | **45.6/44.6** | 6/6 | 512 | 142/123 | 32/35 | -/- | -/- | -/- |
| `robomimic` | **45.1/42.5** | 7/7 | 512 | 193/179 | 15/17 | -/- | -/- | -/- |
| `falcon` | **36.8/35.8** | 1/1 | 512 | 42/50 | 14/18 | -/- | -/- | -/- |
| `wty_cpp` | **34.1/35.1** | 1/0 | 512 | 27/32 | 24/24 | -/- | -/- | -/- |
| `openwbt` | **26.8/40.6** | 0/0 | 512 | 82/73 | 48/50 | -/- | -/- | -/- |
| `rl_gym` | **19.7/37.7** | 0/0 | 512 | 120/78 | 28/16 | -/- | -/- | -/- |
| `rl_lab` | **25.4/22.1** | 0/0 | 512 | 57/71 | 73/74 | -/- | -/- | -/- |
| `handoff` | **21.3/22.8** | 0/0 | 512 | 148/149 | 24/23 | -/- | -/- | -/- |
| `bfm_zero` | **12.3/10.1** | 0/0 | 512 | 404/409 | 63/80 | -/- | -/- | -/- |
| `josabb`\* | **10.1/10.1** | 0/0 | 512 | 101/117 | 57/60 | -/- | -/- | -/- |
| `huru`\* | **7.8/8.0** | 0/0 | 512 | 150/145 | 58/63 | -/- | -/- | -/- |
| `sunny`\* | **7.3/7.9** | 0/0 | 512 | 152/164 | 63/67 | -/- | -/- | -/- |
| `zealot` | **8.2/4.0** | 0/0 | 512 | 389/257 | 92/84 | -/- | -/- | -/- |
| `holosoma` | **5.8/6.4** | 0/0 | 512 | 70/83 | 46/53 | -/- | -/- | -/- |
| `rl_mjlab` | **5.6/5.3** | 0/0 | 512 | 243/241 | 62/68 | -/- | -/- | -/- |
| `dm_march` | **6.4/4.0** | 0/0 | 512 | 147/162 | 92/90 | -/- | -/- | -/- |
| `dm_agile` | **5.2/3.4** | 1/0 | 512 | 57/52 | 88/87 | -/- | -/- | -/- |
| `legged_rl_lab` | **3.9/3.9** | 0/0 | 512 | 205/231 | 84/83 | -/- | -/- | -/- |
| `mturan33`\* | **2.8/4.3** | 0/0 | 512 | 223/212 | 87/88 | -/- | -/- | -/- |
| `schoi` | **3.8/2.2** | 0/0 | 512 | 205/142 | 90/89 | -/- | -/- | -/- |
| `g1_gym` | **2.5/1.9** | 0/0 | 512 | 112/111 | 87/86 | -/- | -/- | -/- |
| `nanog1` | **2.1/2.2** | 0/0 | 512 | 83/91 | 58/61 | -/- | -/- | -/- |
| `stepdown` | **2.1/1.9** | 0/0 | 512 | 75/74 | 52/60 | -/- | -/- | -/- |
| `clobot` | **1.8/1.8** | 0/0 | 512 | 129/125 | 84/81 | -/- | -/- | -/- |
| `wcompton` | **2.0/1.5** | 0/0 | 512 | 127/129 | 63/75 | -/- | -/- | -/- |
| ~~`handoff`<br>`with_arms`~~\*\* | ~~**70.1/69.7**~~ | ~~37/34~~ | ~~512~~ | ~~14/17~~ | ~~8/9~~ | ~~7.6/12.1~~ | ~~1.3/2.3~~ | ~~3.6/4.1~~ |
| ~~`clobot`<br>`with_arms`~~\*\* | ~~**43.6/36.3**~~ | ~~2/1~~ | ~~512~~ | ~~31/33~~ | ~~15/19~~ | ~~-/-~~ | ~~-/-~~ | ~~-/-~~ |

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
completion in either engine for one and five centimetres of position error;
0.66 m is another seven points in MuJoCo for two more centimetres, while in
PhysX it gives two to four points back and the error more than doubles,
21 → 49-57 cm. `decoupled_wbc` pays far less — 0.70 → 0.66 m gains two to four
points, and its position error holds at 10 cm in MuJoCo, the lowest in the
table, though PhysX drifts from 15 to 20-22 cm. Commanded pitch does nothing
for completion in either.

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
