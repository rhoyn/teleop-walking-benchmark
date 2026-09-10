#!/usr/bin/env python3
import argparse
import os
import shutil
import struct

import mujoco
import numpy as np
import warp as wp

import mujoco_warp as mjw
from mujoco_warp._src.math import rot_vec_quat

JOINTS = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
GROUP = [0, 0, 0, 0, 1, 1] * 2 + [2, 2, 2] + [3, 3, 3, 3, 4, 4, 4] * 2
NM = len(JOINTS)
NG = 5
MAGIC = b"MJWARP01"


@wp.kernel
def k_servo(
    q_target: wp.array2d(dtype=wp.float32),
    kp: wp.array(dtype=wp.float32),
    kd: wp.array(dtype=wp.float32),
    frc_lo: wp.array(dtype=wp.float32),
    frc_hi: wp.array(dtype=wp.float32),
    qadr: wp.array(dtype=wp.int32),
    dadr: wp.array(dtype=wp.int32),
    act: wp.array(dtype=wp.int32),
    alive: wp.array(dtype=wp.int32),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
):
    w, j = wp.tid()
    a = act[j]
    if alive[w] == 0:
        ctrl[w, a] = 0.0
        return
    tau = kp[j] * (q_target[w, j] - qpos[w, qadr[j]]) - kd[j] * qvel[w, dadr[j]]
    ctrl[w, a] = wp.clamp(tau, frc_lo[j], frc_hi[j])


@wp.kernel
def k_punch_clear(xfrc: wp.array2d(dtype=wp.spatial_vectorf)):
    w, b = wp.tid()
    xfrc[w, b] = wp.spatial_vectorf()


@wp.kernel
def k_punch_apply(
    punch_joint: wp.array(dtype=wp.int32),
    punch_frame: wp.array(dtype=wp.int32),
    punch_force: wp.array2d(dtype=wp.float32),
    alive: wp.array(dtype=wp.int32),
    jnt_body: wp.array(dtype=wp.int32),
    jnt_frame: wp.array(dtype=wp.int32),
    jnt_anchor: wp.array(dtype=wp.vec3f),
    xquat: wp.array2d(dtype=wp.quatf),
    xpos: wp.array2d(dtype=wp.vec3f),
    xipos: wp.array2d(dtype=wp.vec3f),
    xfrc: wp.array2d(dtype=wp.spatial_vectorf),
):
    w = wp.tid()
    j = punch_joint[w]
    if j < 0 or alive[w] == 0:
        return
    b = jnt_body[j]
    fb = b
    if punch_frame[w] == 0:
        fb = jnt_frame[j]
    r = rot_vec_quat(jnt_anchor[j], xquat[w, b]) + xpos[w, b] - xipos[w, b]
    local = wp.vec3f(punch_force[w, 0], punch_force[w, 1], punch_force[w, 2])
    f = rot_vec_quat(local, xquat[w, fb])
    t = wp.cross(r, f)
    xfrc[w, b] = wp.spatial_vectorf(f[0], f[1], f[2], t[0], t[1], t[2])


@wp.kernel
def k_obs_joint(
    qadr: wp.array(dtype=wp.int32),
    dadr: wp.array(dtype=wp.int32),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    obs_q: wp.array2d(dtype=wp.float32),
    obs_dq: wp.array2d(dtype=wp.float32),
):
    w, j = wp.tid()
    obs_q[w, j] = qpos[w, qadr[j]]
    obs_dq[w, j] = qvel[w, dadr[j]]


@wp.kernel
def k_obs_base(
    s_quat: int,
    s_gyro: int,
    s_pvel: int,
    s_tvel: int,
    pelvis: int,
    foot0: int,
    foot1: int,
    sensordata: wp.array2d(dtype=wp.float32),
    xpos: wp.array2d(dtype=wp.vec3f),
    obs_quat: wp.array2d(dtype=wp.float32),
    obs_gyro: wp.array2d(dtype=wp.float32),
    obs_grav: wp.array2d(dtype=wp.float32),
    obs_pose: wp.array2d(dtype=wp.float32),
    obs_foot: wp.array2d(dtype=wp.float32),
    obs_speed: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    for k in range(4):
        obs_quat[w, k] = sensordata[w, s_quat + k]
    for k in range(3):
        obs_gyro[w, k] = sensordata[w, s_gyro + k]

    q = wp.quatf(
        sensordata[w, s_quat + 0],
        -sensordata[w, s_quat + 1],
        -sensordata[w, s_quat + 2],
        -sensordata[w, s_quat + 3],
    )
    g = rot_vec_quat(wp.vec3f(0.0, 0.0, -1.0), q)
    for k in range(3):
        obs_grav[w, k] = g[k]

    p = xpos[w, pelvis]
    obs_pose[w, 0] = -q[1]
    obs_pose[w, 1] = -q[2]
    obs_pose[w, 2] = -q[3]
    obs_pose[w, 3] = q[0]
    obs_pose[w, 4] = p[0]
    obs_pose[w, 5] = p[1]
    obs_pose[w, 6] = p[2]

    obs_foot[w, 0] = xpos[w, foot0][2]
    obs_foot[w, 1] = xpos[w, foot1][2]
    obs_speed[w, 0] = wp.length(
        wp.vec3f(
            sensordata[w, s_pvel + 0],
            sensordata[w, s_pvel + 1],
            sensordata[w, s_pvel + 2],
        )
    )
    obs_speed[w, 1] = wp.length(
        wp.vec3f(
            sensordata[w, s_tvel + 0],
            sensordata[w, s_tvel + 1],
            sensordata[w, s_tvel + 2],
        )
    )


def need(mjm, objtype, name):
    i = mujoco.mj_name2id(mjm, objtype, name)
    if i < 0:
        raise SystemExit(f"model has no {objtype} named {name!r}")
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="assets/g1_29dof.xml")
    ap.add_argument("--nworld", type=int, default=1024)
    ap.add_argument("--out", default="build/mjwarp/g1")
    ap.add_argument("--njmax", type=int, default=512)
    ap.add_argument("--nconmax", type=int, default=128)
    args = ap.parse_args()

    mjm = mujoco.MjModel.from_xml_path(args.xml)
    nworld = args.nworld
    m = mjw.put_model(mjm)
    m.opt.tolerance.assign(np.array([mjm.opt.tolerance], np.float32))
    d = mjw.make_data(mjm, nworld=nworld, njmax=args.njmax, nconmax=args.nconmax)

    J = mujoco.mjtObj.mjOBJ_JOINT
    A = mujoco.mjtObj.mjOBJ_ACTUATOR
    B = mujoco.mjtObj.mjOBJ_BODY
    S = mujoco.mjtObj.mjOBJ_SENSOR

    jid = [need(mjm, J, j) for j in JOINTS]
    qadr = np.array([mjm.jnt_qposadr[i] for i in jid], np.int32)
    dadr = np.array([mjm.jnt_dofadr[i] for i in jid], np.int32)
    act = np.array([need(mjm, A, j) for j in JOINTS], np.int32)
    jbody = np.array([mjm.jnt_bodyid[i] for i in jid], np.int32)
    jframe = np.array([mjm.body_parentid[mjm.jnt_bodyid[i]] for i in jid], np.int32)
    janchor = np.array([mjm.jnt_pos[i] for i in jid], np.float32)

    lo = np.full(NM, -np.inf, np.float32)
    hi = np.full(NM, np.inf, np.float32)
    for j, a in enumerate(act):
        if mjm.actuator_forcelimited[a]:
            lo[j], hi[j] = mjm.actuator_forcerange[a]

    pelvis = need(mjm, B, "pelvis")
    base_q = int(mjm.jnt_qposadr[mjm.body_jntadr[pelvis]])
    base_v = int(mjm.jnt_dofadr[mjm.body_jntadr[pelvis]])
    foot0 = need(mjm, B, "left_ankle_roll_link")
    foot1 = need(mjm, B, "right_ankle_roll_link")
    s_quat = int(mjm.sensor_adr[need(mjm, S, "imu_quat")])
    s_gyro = int(mjm.sensor_adr[need(mjm, S, "imu_gyro")])
    s_pvel = int(mjm.sensor_adr[need(mjm, S, "pelvis_vel")])
    s_tvel = int(mjm.sensor_adr[need(mjm, S, "torso_vel")])

    P = {
        "q_target": wp.zeros((nworld, NM), dtype=wp.float32),
        "kp": wp.zeros(NM, dtype=wp.float32),
        "kd": wp.zeros(NM, dtype=wp.float32),
        "alive": wp.ones(nworld, dtype=wp.int32),
        "punch_joint": wp.full(nworld, -1, dtype=wp.int32),
        "punch_frame": wp.zeros(nworld, dtype=wp.int32),
        "punch_force": wp.zeros((nworld, 3), dtype=wp.float32),
        "obs_q": wp.zeros((nworld, NM), dtype=wp.float32),
        "obs_dq": wp.zeros((nworld, NM), dtype=wp.float32),
        "obs_gyro": wp.zeros((nworld, 3), dtype=wp.float32),
        "obs_grav": wp.zeros((nworld, 3), dtype=wp.float32),
        "obs_quat": wp.zeros((nworld, 4), dtype=wp.float32),
        "obs_pose": wp.zeros((nworld, 7), dtype=wp.float32),
        "obs_foot": wp.zeros((nworld, 2), dtype=wp.float32),
        "obs_speed": wp.zeros((nworld, 2), dtype=wp.float32),
        "qpos": d.qpos,
        "qvel": d.qvel,
        "ctrl": d.ctrl,
        "qacc_warmstart": d.qacc_warmstart,
    }

    c_qadr = wp.array(qadr, dtype=wp.int32)
    c_dadr = wp.array(dadr, dtype=wp.int32)
    c_act = wp.array(act, dtype=wp.int32)
    c_jbody = wp.array(jbody, dtype=wp.int32)
    c_jframe = wp.array(jframe, dtype=wp.int32)
    c_janchor = wp.array(janchor, dtype=wp.vec3f)
    c_lo = wp.array(lo, dtype=wp.float32)
    c_hi = wp.array(hi, dtype=wp.float32)

    def substep():
        wp.launch(
            k_servo,
            dim=(nworld, NM),
            inputs=[
                P["q_target"],
                P["kp"],
                P["kd"],
                c_lo,
                c_hi,
                c_qadr,
                c_dadr,
                c_act,
                P["alive"],
                d.qpos,
                d.qvel,
            ],
            outputs=[d.ctrl],
        )
        wp.launch(k_punch_clear, dim=(nworld, mjm.nbody), outputs=[d.xfrc_applied])
        wp.launch(
            k_punch_apply,
            dim=nworld,
            inputs=[
                P["punch_joint"],
                P["punch_frame"],
                P["punch_force"],
                P["alive"],
                c_jbody,
                c_jframe,
                c_janchor,
                d.xquat,
                d.xpos,
                d.xipos,
            ],
            outputs=[d.xfrc_applied],
        )
        mjw.step(m, d)
        wp.launch(
            k_obs_joint,
            dim=(nworld, NM),
            inputs=[c_qadr, c_dadr, d.qpos, d.qvel],
            outputs=[P["obs_q"], P["obs_dq"]],
        )
        wp.launch(
            k_obs_base,
            dim=nworld,
            inputs=[
                s_quat,
                s_gyro,
                s_pvel,
                s_tvel,
                pelvis,
                foot0,
                foot1,
                d.sensordata,
                d.xpos,
            ],
            outputs=[
                P["obs_quat"],
                P["obs_gyro"],
                P["obs_grav"],
                P["obs_pose"],
                P["obs_foot"],
                P["obs_speed"],
            ],
        )

    substep()
    wp.synchronize()
    with wp.ScopedCapture(apic=True) as cap:
        substep()
    wp.synchronize()

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    wp.capture_save(cap.graph, args.out, inputs=P, outputs=P)

    lib = os.path.join(os.path.dirname(wp.__file__), "bin", "warp.so")
    dst = os.path.join(out_dir, "warp.so")
    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(lib):
        shutil.copyfile(lib, dst)
    shutil.copy2(
        os.path.join(os.path.dirname(wp.__file__), "bin", "warp.so"),
        os.path.join(os.path.dirname(args.out) or ".", "warp.so"),
    )

    with open(args.out + ".bin", "wb") as f:
        f.write(MAGIC)
        f.write(wp.__version__.encode().ljust(32, b"\0"))
        f.write(
            struct.pack("<8i", nworld, mjm.nq, mjm.nv, mjm.nu, NM, NG, base_q, base_v)
        )
        f.write(struct.pack("<f", float(mjm.opt.timestep)))
        for arr in (qadr, dadr, act, np.array(GROUP, np.int32)):
            f.write(arr.astype("<i4").tobytes())
        f.write(mjm.qpos0.astype("<f4").tobytes())


if __name__ == "__main__":
    main()
