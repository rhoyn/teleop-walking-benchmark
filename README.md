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
targets reached; the walk energy, vibration, touchdown and foot force columns
are means over completed runs, and an engine's cell is blank when fewer than a
fifth of its runs qualify. Both engines step on the GPU: MuJoCo through MuJoCo
Warp, PhysX through its GPU solver.

The last two columns say how hard a policy puts its feet down. Each foot is
watched every 2 ms substep through the force carried by its ankle joint; a
touchdown begins when that force rises above 30 N and ends when it drops below
10 N. `foot touchdown` is the downward speed of the foot on the substep before
contact, in m/s, averaged over every touchdown of every completed run; `peak
foot force` is the largest force one foot carries during that stance, in body
weights (35.1 kg, 344 N). A foot arriving at 0.5 m/s carries four times the
kinetic energy of one arriving at 0.25 m/s, and that energy goes into the
floor. The per-segment figures are in each CSV as `s<N>_steps`,
`s<N>_touchdown_mps` and `s<N>_peak_grf_bw`.

| `--policy` | mean<br>survival<br>sec<br>mujoco<br>physx | completed<br>percent<br>mujoco<br>physx | runs | pos<br>err<br>cm<br>mujoco<br>physx | yaw<br>err<br>deg<br>mujoco<br>physx | walk<br>energy<br>KJ<br>mujoco<br>physx | walk<br>vibrations<br>mujoco<br>physx | foot<br>touchdown<br>m/s<br>mujoco<br>physx | peak<br>foot<br>force<br>BW<br>mujoco<br>physx |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gr00t_wbc`<br>`h066_p012` | **85.1/82.2** | 80/68 | 5120 | 16/50 | 7/8 | 7.3/9.7 | 0.9/1.2 | 0.50/0.36 | 1.01/0.88 |
| `gr00t_wbc`<br>`h066_p000` | **85.0/82.0** | 80/68 | 5120 | 17/57 | 8/9 | 7.2/9.8 | 0.9/1.3 | 0.49/0.36 | 0.98/0.85 |
| `gr00t_wbc`<br>`h070_p000` | **83.2/82.9** | 73/70 | 5120 | 14/21 | 5/6 | 7.3/9.4 | 0.9/1.2 | 0.54/0.45 | 1.15/1.26 |
| `decoupled_wbc`<br>`h066_p000` | **83.9/81.3** | 74/67 | 5120 | 10/22 | 7/5 | 5.9/13.4 | 1.2/3.3 | 0.47/0.29 | 1.22/0.95 |
| `decoupled_wbc`<br>`h066_p012` | **84.0/80.8** | 75/66 | 5120 | 10/20 | 7/6 | 5.9/13.7 | 1.2/3.3 | 0.46/0.30 | 1.21/1.00 |
| `decoupled_wbc`<br>`h070_p000` | **83.3/80.3** | 72/64 | 5120 | 10/15 | 7/6 | 5.9/12.4 | 1.2/2.9 | 0.44/0.29 | 1.22/1.07 |
| `decoupled_wbc`<br>`h074_p000` | **82.3/79.7** | 69/61 | 5120 | 11/14 | 8/7 | 5.9/12.2 | 1.3/2.8 | 0.41/0.33 | 1.22/1.33 |
| `gr00t_wbc`<br>`h074_p000` | **80.2/80.0** | 63/61 | 5120 | 13/17 | 5/6 | 7.8/9.9 | 1.0/1.3 | 0.52/0.51 | 1.24/1.77 |
| `homie` | **78.4/77.5** | 55/52 | 5120 | 17/19 | 32/37 | 10.4/10.2 | 1.6/1.5 | 0.51/0.50 | 0.85/1.55 |
| `wbc_agile_velocity` | **75.1/79.1** | 49/60 | 5120 | 21/23 | 7/6 | 12.4/14.5 | 2.1/2.7 | 0.54/0.55 | 0.96/1.56 |
| `amo` | **75.8/77.9** | 53/57 | 5120 | 24/39 | 18/17 | 10.0/12.2 | 1.4/1.5 | 0.52/0.50 | 1.06/1.71 |
| `wbc_agile` | **72.1/64.5** | 37/21 | 512 | 22/42 | 8/11 | 7.7/9.7 | 1.3/1.7 | 0.52/0.51 | 1.22/1.86 |
| `sonic` | **68.6/52.7** | 37/13 | 512 | 30/45 | 9/11 | 11.5/- | 1.5/- | 0.25/- | 0.91/- |
| `mimic_lite` | **48.8/59.0** | 12/22 | 512 | 76/88 | 12/11 | -/18.0 | -/2.4 | -/0.52 | -/2.09 |
| `run_residual` | **51.7/48.1** | 7/5 | 512 | 581/586 | 13/14 | -/- | -/- | -/- | -/- |
| `asap` | **46.0/45.0** | 6/6 | 512 | 140/123 | 32/35 | -/- | -/- | -/- | -/- |
| `robomimic` | **44.8/43.3** | 6/7 | 512 | 193/179 | 15/16 | -/- | -/- | -/- | -/- |
| `falcon` | **36.9/35.4** | 1/0 | 512 | 42/50 | 15/18 | -/- | -/- | -/- | -/- |
| `openwbt` | **26.7/40.4** | 0/0 | 512 | 81/74 | 48/50 | -/- | -/- | -/- | -/- |
| `rl_gym` | **19.8/36.7** | 0/1 | 512 | 120/76 | 28/16 | -/- | -/- | -/- | -/- |
| `wty_cpp` | **23.5/30.2** | 0/0 | 512 | 32/31 | 25/24 | -/- | -/- | -/- | -/- |
| `rl_lab` | **25.3/22.2** | 0/0 | 512 | 56/69 | 73/73 | -/- | -/- | -/- | -/- |
| `handoff` | **21.1/22.5** | 0/0 | 512 | 145/149 | 23/24 | -/- | -/- | -/- | -/- |
| `bfm_zero` | **11.8/10.2** | 0/0 | 512 | 406/409 | 66/80 | -/- | -/- | -/- | -/- |
| `josabb`\* | **10.0/10.1** | 0/0 | 512 | 99/122 | 58/60 | -/- | -/- | -/- | -/- |
| `huru`\* | **7.7/8.2** | 0/0 | 512 | 150/146 | 57/63 | -/- | -/- | -/- | -/- |
| `sunny`\* | **7.2/7.6** | 0/0 | 512 | 151/159 | 63/67 | -/- | -/- | -/- | -/- |
| `holosoma` | **5.9/6.4** | 0/0 | 512 | 71/85 | 45/55 | -/- | -/- | -/- | -/- |
| `zealot` | **8.1/4.1** | 0/0 | 512 | 382/259 | 91/84 | -/- | -/- | -/- | -/- |
| `rl_mjlab` | **5.6/5.3** | 0/0 | 512 | 244/244 | 61/68 | -/- | -/- | -/- | -/- |
| `dm_march` | **6.3/4.2** | 0/0 | 512 | 147/163 | 91/90 | -/- | -/- | -/- | -/- |
| `dm_agile` | **5.4/3.4** | 1/0 | 512 | 58/52 | 88/88 | -/- | -/- | -/- | -/- |
| `legged_rl_lab` | **4.0/3.8** | 0/0 | 512 | 212/229 | 80/83 | -/- | -/- | -/- | -/- |
| `mturan33`\* | **2.7/4.3** | 0/0 | 512 | 216/218 | 89/92 | -/- | -/- | -/- | -/- |
| `schoi` | **3.6/2.4** | 0/0 | 512 | 203/143 | 88/90 | -/- | -/- | -/- | -/- |
| `g1_gym` | **2.5/1.9** | 0/0 | 512 | 110/108 | 86/89 | -/- | -/- | -/- | -/- |
| `nanog1` | **2.1/2.2** | 0/0 | 512 | 82/90 | 58/61 | -/- | -/- | -/- | -/- |
| `stepdown` | **2.1/1.9** | 0/0 | 512 | 75/74 | 52/61 | -/- | -/- | -/- | -/- |
| `wcompton` | **2.0/1.6** | 0/0 | 512 | 127/139 | 62/74 | -/- | -/- | -/- | -/- |
| `clobot` | **1.8/1.8** | 0/0 | 512 | 130/125 | 84/81 | -/- | -/- | -/- | -/- |
| ~~`handoff`<br>`with_arms`~~\*\* | ~~**69.9/70.8**~~ | ~~35/38~~ | ~~512~~ | ~~15/16~~ | ~~8/9~~ | ~~7.8/12.3~~ | ~~1.3/2.3~~ | ~~0.50/0.51~~ | ~~1.15/1.69~~ |
| ~~`clobot`<br>`with_arms`~~\*\* | ~~**44.0/35.8**~~ | ~~3/1~~ | ~~512~~ | ~~29/33~~ | ~~15/19~~ | ~~-/-~~ | ~~-/-~~ | ~~-/-~~ | ~~-/-~~ |

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
completion in MuJoCo and nine in PhysX for one and four centimetres of
position error; 0.66 m is another seven points in MuJoCo for two more
centimetres, while in PhysX it gives two points back and the error more than
doubles, 21 → 50-57 cm. `decoupled_wbc` pays far less — 0.70 → 0.66 m gains
two to three points, and its position error holds at 10 cm in MuJoCo, the
lowest in the table, though PhysX drifts from 15 to 20-22 cm. Commanded pitch
does nothing for completion in either.

Height also sets how hard the feet land. Lowering `gr00t_wbc` from 0.74 to
0.66 m takes its PhysX touchdown speed from 0.51 to 0.36 m/s and the peak foot
force from 1.77 to 0.85-0.88 body weights, the softest landing of any PhysX
row; in MuJoCo the same change barely moves touchdown speed (0.52 → 0.50 m/s)
but still drops the peak from 1.24 to 1.0. `decoupled_wbc` lands more slowly
than `gr00t_wbc` at every height (0.41-0.47 m/s in MuJoCo, 0.29-0.33 in PhysX)
yet hits the floor harder in MuJoCo, holding 1.2 body weights regardless of
height.

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

The contact model also shows in the foot force column: PhysX resolves a
landing as a rigid impulse, so its peaks run higher — 1.5-1.7 body weights for
`homie`, `amo` and `wbc_agile_velocity` against 0.9-1.1 in MuJoCo's soft
contacts, 2.1 for `mimic_lite` — while the touchdown speeds, which are set by
the policy rather than the solver, mostly agree to within a few hundredths.
Where they do not, PhysX is the one that lands softer: the `gr00t_wbc` 0.66
and 0.70 m rows and every `decoupled_wbc` row come down 0.1-0.2 m/s slower
there.

Neither engine repeats a run id bit for bit: both step on the GPU and reduce
contact forces across threads in arbitrary order, so the same run id can pass
one round and fall the next. Compare aggregates across rounds, not individual
run ids.

## Weights

Third-party; terms in [NOTICE](NOTICE) — two non-commercial, eleven silent,
one sim-only. Nineteen weight files committed, the rest fetched or converted
locally, since a licence travels to its conversion.
