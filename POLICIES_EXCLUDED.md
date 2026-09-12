# Candidates tried and not included

The harness feeds base linear velocity from the simulator now, to the rows the
README marks, so needing it is no longer a reason to exclude anything. Every
row below names the blocker that survives that.

`score` is the chance of turning the candidate into a new working row, out of
100. A 0 means there is no work to do: the weights are already in the table
under another name, or the robot is not a G1.

`josabb`, `sunny` and `zealot_v26` used to be on this list and are now in the
table.

| candidate | score | why not |
|---|---:|---|
| `kdh` | 40 | A 37-joint contract over arms and hands would narrow like `josabb`, but no repository by that name can be found any more. It also shipped no LICENSE, and its controller carried an NVIDIA proprietary header forbidding redistribution. |
| `yezzzye` | 40 | 123-dim observation over 37 joints including arms and hands, against our 29-DoF asset; the 37-wide action alone would have been fine. MIT and genuinely trained, so the width was the whole problem, but the repository is gone. |
| `snow` | 35 | A MuJoCo Playground joystick policy whose 53 extra dims are a snow-sensing rig never built. A flat no-snow constant would probably serve and the 29 actions are portable, but the checkpoint itself is lost. |
| `almi` | 30 | Dual-policy: `obs[76:93]` is the output of a separate 17-action upper-body net, which itself needs a 14-slot end-effector pose in an undocumented frame. One slot worth 25% RMS was never identified. No LICENSE. |
| `protomotions_gtp` | 30 | Tracker: 248 of 342 obs scalars, 72%, are a reference-motion window sampled 20/40/80/160 ms ahead, which the tour has no clip to supply. A clip generator would be needed. Its 29 action targets also cover arms and wrists. |
| `wbc_mjlab` | 30 | Tracker: 39 of 132 obs slots are read from a clip and the action is a residual on the clip pose (`action_mode: reference_residual`). With no clip there is no `q_ref`, so even the 15 owned entries have no defined meaning. |
| `zealot_v30` | 30 | The 79-dim generation of the same ladder: the 48-dim frame plus 13 upper-body joint targets and their velocities, all of which the harness already has. A contract change to port rather than a blocker. |
| `playground` | 25 | Ships no G1 weights at all: the repo holds the joystick environment and trains on demand. Its layout is the canonical one several other candidates here turn out to be variants of. Only a third-party run would give it a row. |
| `unity_rl` | 25 | Unity left-handed Y-up, joint order recorded nowhere but the scene file, and a low-pass plus double-integrator actuator rather than a position target. All three would have to be undone, not just one of them. |
| `pathon` | 20 | Ten of its 22 outputs are arm joints and none are waist, and it trained with the arms frozen for all 5,000 iterations while the tour random-walks them. Its three 22-blocks also disagree on joint order: arm-leg, leg-arm, arm-leg. |
| `rsamf_walk` | 20 | Tracker: "walk" names the clip it imitates. 152 of 188 obs slots are clip look-ahead with absolute pelvis height baked in, and its action normaliser decodes to the full 29-DoF limits, driving shoulders, elbows and six wrist DoF. |
| `dagger` | 15 | A DAgger navigation student distilled from a teacher, not a walker at all. Separately blocked on exteroception, which a flat ground constant may not adequately cover. |
| `from_w1` | 15 | 1821-dim, and its command channel is a reference motion (`v-teleop-extend-vr-max-nolinvel`) rather than a velocity command. Both G1 checkpoints share it and the project ships no velocity fallback, so a clip generator would be needed. |
| `mimickit` | 15 | Tracker across all four shipped G1 checkpoints, which share one 849-wide actor input: 612 of those scalars, 72%, are clip pose at t+dt, t+2dt and t+3dt. Absolute pelvis height at slot 0 and world-frame linvel at 7..9 are independently fatal. |
| `yahmp` | 15 | The repo defines a velocity-commanded task that would port cleanly, with every privileged term sitting in the critic group and never the actor, but no checkpoint for it was ever committed. All three that shipped are 1727-dim trackers. |
| `bfm` | 10 | 1150 privileged scalars: proprioceptive history, per-body state, a motion goal and a 21x21 heightmap. It is the Stage-1 privileged teacher meant to be distilled later, its own card says not directly deployable, and no student ever shipped. |
| `g1_cu` | 10 | Unsteerable: the command slots are compile-time constants, and `obs[96]` has mean +0.8 with variance 0, so nothing in the observation can be used to point it at a waypoint. |
| `gpc` | 10 | No G1 checkpoint exists. ProtoMotions ships a `gpc_prior` for `soma_bones`, an SMPL-format character rather than a G1, and its one G1 checkpoint is already covered here by `protomotions_gtp`. Worth revisiting if a G1 GPC is released. |
| `loco_mujoco` | 5 | The project publishes no trained weights at all, on GitHub, HuggingFace or PyPI, and not only for the G1: none for any of its other 15 robots either. There is simply nothing to port. |
| `motionbricks` | 5 | Not a controller but a kinematic motion generator: it emits MuJoCo `qpos` frames and never computes a torque, with zero `mj_step`, `ctrl` or actuator across 1515 lines. It also needs a one-hot clip index and a frame index. |
| `mturan33_dual` | 5 | The second `loco_actor` in a repository already represented in the table, and not a gait in any case: it drives the knees past zero. |
| `zealot_v24` | 5 | The rung below `zealot_v26` on the same ladder, sharing its 48-dim frame, and its own card states that it cannot turn. A second row would measure the ladder rather than a second policy. |
| `asap_pipeline` | 0 | An ASAP-lineage tracker blocked on `ref_motion_phase`, which the waypoint tour has no clip to supply. Its other weights are stock Unitree `motion.pt`, already shipped here as `rl_gym`. |
| `dm_g1_agile`, `dm_g1_march` | 0 | Their ONNX files are byte-identical to `policies/dm_agile/model.onnx` and `policies/dm_march/model.onnx`, both already in the table. Plan-only directories written by a second agent. |
| `grove` | 0 | Adyansh04/grove-g1 trains nothing: the one checkpoint it ships, `g1_controllers/policy/unitree_g1_velocity_e2e.onnx`, is NVIDIA's WBC-AGILE `Velocity-G1-History-v0` export vendored unchanged with NVIDIA's licence beside it, and it is byte-identical to `policies/wbc_agile_velocity/model.onnx`, already in the table. What grove-g1 adds is the ros2_control stack around the policy, which this harness does not run. That row was named `grove` after the repository it was fetched from until the identity of the weights was confirmed; it carries the same results under its new name. |
| `graystars` | 0 | Its `velocity_v0_exported_policy_raw.onnx` is byte-identical to `policies/rl_mjlab/model.onnx`, already in the table. |
| `leggedcontrol` | 0 | Not a G1 at all: every checkpoint in the repository targets a Go2 or a wheeled platform (`policy_go2w.onnx`). It fails task fit long before licensing would matter. |
| `polysim` | 0 | An ASAP-lineage tracker needing `ref_motion_phase`, and its `dec_loco_model_6600.onnx` is byte-identical to `policies/asap/model.onnx`, already in the table. |
| `rl_hnav` | 0 | A Nav2 and SLAM navigation stack that vendors someone else's walker: both of its G1 locomotion checkpoints are byte-identical to `policies/rl_gym/model.pt` and `policies/robomimic/model.pt`. |
| `yifeichen` | 0 | Four repositories, 78 weight files hashed. The G1 locomotion ones are byte-identical copies of `policies/robomimic/model.pt` and `policies/rl_gym/model.pt`; the remainder are ASAP trackers needing `ref_motion_phase`. |
| `zealot_walk_latest` | 0 | Byte-identical to `g1_v30_final50k`, so it is `zealot_v30` again under the author's moving name rather than a distinct checkpoint. |
