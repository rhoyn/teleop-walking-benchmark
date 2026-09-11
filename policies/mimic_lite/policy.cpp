namespace mimic_lite {

constexpr int NUM_ACTIONS = 29;
constexpr int NUM_OWNED = 15;
constexpr int CMD_OBS = 304;
constexpr int POL_OBS = 535;
constexpr int PRIV_DIM = 1024;

constexpr int REF_STEPS = 8;
constexpr int HIST_STEPS = 7;
constexpr int HIST_RING = 17;
constexpr int PAST_RING = 9;
constexpr int ACT_STEPS = 3;

constexpr int CTX_FRAMES = 4;
constexpr int QPOS = 36;
constexpr int PLAN_FRAMES = 64;
constexpr float PLAN_HZ = 30.0f;
constexpr int REPLAN_TICKS = 5;
constexpr int LOOKAHEAD = 2;
constexpr int BLEND_FRAMES = 8;
constexpr float SEED_HEIGHT = 0.788740f;
constexpr int MODE_IDLE = 0;
constexpr int MODE_SLOW_WALK = 1;
constexpr int MODE_WALK = 2;
constexpr float WALK_SPEED = 0.8f;
constexpr float IDLE_SPEED = 0.12f;
constexpr float FACE_LEAD_S = 0.6f;
constexpr float PLAN_SEED = 1234.0f;
constexpr float CONTROL_DT = 0.02f;

constexpr int MAX_FRAMES = 128;

constexpr int FRAME = 3 + 4 + NUM_ACTIONS;
constexpr int F_POS = 0, F_QUAT = 3, F_Q = 7;

__device__ const int D_MJ_OF_IL[NUM_ACTIONS] = {0,  6,  12, 1,  7,  13, 2,  8,
                                                14, 3,  9,  15, 22, 4,  10, 16,
                                                23, 5,  11, 17, 24, 18, 25, 19,
                                                26, 20, 27, 21, 28};
__device__ const int D_IL_OF_MJ[NUM_ACTIONS] = {0,  3,  6,  9,  13, 17, 1,  4,
                                                7,  10, 14, 18, 2,  5,  8,  11,
                                                15, 19, 21, 23, 25, 27, 12, 16,
                                                20, 22, 24, 26, 28};

__device__ const int D_REF_STEP[REF_STEPS] = {-8, -4, -2, 0, 1, 2, 3, 4};
__device__ const int D_HIST_STEP[HIST_STEPS] = {0, 1, 2, 3, 4, 8, 16};

#define MIMIC_LITE_DEFAULTS                                                \
  -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f, -0.312f, 0.0f, 0.0f, 0.669f, \
      -0.363f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.0f, 0.6f, 0.0f, 0.0f, \
      0.0f, 0.2f, -0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f

__device__ const float D_DEFAULTS[NUM_ACTIONS] = {MIMIC_LITE_DEFAULTS};

__device__ const float D_SCALE_IL[NUM_ACTIONS] = {
    0.55f, 0.55f, 0.55f, 0.35f, 0.35f, 0.44f, 0.55f, 0.55f, 0.44f, 0.35f,
    0.35f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f,
    0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f
};

constexpr float KP_HIP = 99.09842777666111f;
constexpr float KP_YAW = 40.17923863450712f;
constexpr float KP_ANKLE = 28.50124619574858f;
constexpr float KD_HIP = 6.308801853496639f;
constexpr float KD_YAW = 2.557889775413375f;
constexpr float KD_ANKLE = 1.814445686584846f;

const float KPS[NUM_OWNED] = {
    KP_HIP,
    KP_HIP,
    KP_YAW,
    KP_HIP,
    KP_ANKLE,
    KP_ANKLE,
    KP_HIP,
    KP_HIP,
    KP_YAW,
    KP_HIP,
    KP_ANKLE,
    KP_ANKLE,
    KP_YAW,
    KP_ANKLE,
    KP_ANKLE
};
const float KDS[NUM_OWNED] = {
    KD_HIP,
    KD_HIP,
    KD_YAW,
    KD_HIP,
    KD_ANKLE,
    KD_ANKLE,
    KD_HIP,
    KD_HIP,
    KD_YAW,
    KD_HIP,
    KD_ANKLE,
    KD_ANKLE,
    KD_YAW,
    KD_ANKLE,
    KD_ANKLE
};

const policy_api::Limits LIMITS = {-0.6, 0.6, 0.4, 1.0, 0.6};

__device__ inline void q_mul(
    const float* a,
    const float* b,
    float* out
) {
  out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

__device__ inline float wrap_pi(float a) {
  while (a > float(M_PI)) a -= 2.0f * float(M_PI);
  while (a < -float(M_PI)) a += 2.0f * float(M_PI);
  return a;
}

__device__ inline float q_yaw(const float* q) {
  return atan2f(
      2.0f * (q[0] * q[3] + q[1] * q[2]),
      1.0f - 2.0f * (q[2] * q[2] + q[3] * q[3])
  );
}

__device__ inline void q_slerp(
    const float* a,
    const float* b,
    float t,
    float* out
) {
  const float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const float ad = fabsf(d);
  float s0, s1;
  if (ad >= 1.0f - 1e-6f) {
    s0 = 1.0f - t;
    s1 = t;
  } else {
    const float theta = acosf(ad);
    const float st = sinf(theta);
    s0 = sinf((1.0f - t) * theta) / st;
    s1 = sinf(t * theta) / st;
  }
  if (d < 0.0f) s1 = -s1;
  for (int k = 0; k < 4; ++k) out[k] = s0 * a[k] + s1 * b[k];
}

__device__ inline void q_norm(float* q) {
  const float n = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  const float s = n > 1e-12f ? 1.0f / n : 1.0f;
  for (int k = 0; k < 4; ++k) q[k] *= s;
}

__device__ inline void rot6_rows(
    const float* q,
    float* out
) {
  const float w = q[0], x = q[1], y = q[2], z = q[3];
  out[0] = 1.0f - 2.0f * (y * y + z * z);
  out[1] = 2.0f * (x * y - w * z);
  out[2] = 2.0f * (x * z + w * y);
  out[3] = 2.0f * (x * y + w * z);
  out[4] = 1.0f - 2.0f * (x * x + z * z);
  out[5] = 2.0f * (y * z - w * x);
}

__device__ inline const float* frame_at(
    const float* motion,
    int len,
    int f
) {
  const int last = len > 0 ? len - 1 : 0;
  const int i = f < 0 ? 0 : (f > last ? last : f);
  return motion + size_t(i) * FRAME;
}

__device__ inline int ring_slot(
    long long tick,
    int back,
    int size
) {
  long long s = (tick - back) % size;
  if (s < 0) s += size;
  return int(s);
}

__global__ void k_mimic_lite_drive(
    const float* __restrict__ cmd,
    const float* __restrict__ base_quat,
    const float* __restrict__ psiref,
    const float* __restrict__ yaw0,
    const int* __restrict__ len,
    float* __restrict__ mode,
    float* __restrict__ vel,
    float* __restrict__ move,
    float* __restrict__ face,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const bool first = len[env] == 0;
  const float here = q_yaw(base_quat + env * 4);
  const float psi = first ? here : psiref[env] + wrap_pi(here - yaw0[env]);
  const float* d = cmd + env * 3;

  const float speed = sqrtf(d[0] * d[0] + d[1] * d[1]);
  const float course = psi + atan2f(d[1], d[0]);
  const float face_yaw = psi + d[2] * FACE_LEAD_S;
  const bool stepping = speed > IDLE_SPEED;

  const int m = !stepping            ? MODE_IDLE
                : speed < WALK_SPEED ? MODE_SLOW_WALK
                                     : MODE_WALK;
  mode[env] = first ? float(MODE_IDLE) : float(m);
  vel[env] = (first || !stepping) ? 0.0f : speed;

  move[env * 3 + 0] = stepping ? cosf(course) : 0.0f;
  move[env * 3 + 1] = stepping ? sinf(course) : 0.0f;
  move[env * 3 + 2] = 0.0f;
  face[env * 3 + 0] = cosf(face_yaw);
  face[env * 3 + 1] = sinf(face_yaw);
  face[env * 3 + 2] = 0.0f;
}

__global__ void k_mimic_lite_context(
    const float* __restrict__ motion,
    const int* __restrict__ len,
    const int* __restrict__ cursor,
    const float* __restrict__ motor_q,
    float* __restrict__ ctx,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* out = ctx + size_t(env) * CTX_FRAMES * QPOS;
  for (int i = 0; i < CTX_FRAMES * QPOS; ++i) out[i] = 0.0f;

  if (len[env] == 0) {
    for (int n = 0; n < CTX_FRAMES; ++n) {
      out[n * QPOS + 2] = SEED_HEIGHT;
      out[n * QPOS + 3] = 1.0f;
      for (int j = 0; j < NUM_ACTIONS; ++j) {
        out[n * QPOS + 7 + j] = motor_q[env * POLICY_NUM_MOTOR + j];
      }
    }
    return;
  }

  const float* m = motion + size_t(env) * MAX_FRAMES * FRAME;
  const int from = cursor[env] + LOOKAHEAD;
  for (int n = 0; n < CTX_FRAMES; ++n) {
    const float t = float(from) * CONTROL_DT + float(n) / PLAN_HZ;
    const float f50 = t / CONTROL_DT;
    const int f0 = int(floorf(f50));
    const float w = f50 - floorf(f50);
    const float* a = frame_at(m, len[env], f0);
    const float* b = frame_at(m, len[env], f0 + 1);

    for (int k = 0; k < 3; ++k) {
      out[n * QPOS + k] = a[F_POS + k] * (1.0f - w) + b[F_POS + k] * w;
    }
    float q[4];
    q_slerp(a + F_QUAT, b + F_QUAT, w, q);
    for (int k = 0; k < 4; ++k) out[n * QPOS + 3 + k] = q[k];
    for (int j = 0; j < NUM_ACTIONS; ++j) {
      out[n * QPOS + 7 + D_MJ_OF_IL[j]] =
          a[F_Q + j] * (1.0f - w) + b[F_Q + j] * w;
    }
  }
}

__global__ void k_mimic_lite_apply(
    const float* __restrict__ base_quat,
    const float* __restrict__ qpos,
    const float* __restrict__ nframes,
    float* __restrict__ gen,
    const float* __restrict__ motion_in,
    float* __restrict__ motion_out,
    int* __restrict__ len,
    int* __restrict__ cursor,
    float* __restrict__ psiref,
    float* __restrict__ yaw0,
    float* __restrict__ yaw_off,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* raw = qpos + size_t(env) * PLAN_FRAMES * QPOS;
  float* g = gen + size_t(env) * MAX_FRAMES * FRAME;
  const float* mi = motion_in + size_t(env) * MAX_FRAMES * FRAME;
  float* mo = motion_out + size_t(env) * MAX_FRAMES * FRAME;
  const int old_len = len[env];
  const int cur = cursor[env];

  int n30 = int(lroundf(nframes[env]));
  n30 = n30 < 0 ? 0 : (n30 > PLAN_FRAMES ? PLAN_FRAMES : n30);

  int n50 = 0;
  if (n30 >= 2) {
    n50 = int(floor(double(n30) / double(PLAN_HZ) / double(CONTROL_DT)));
    if (n50 > MAX_FRAMES) n50 = MAX_FRAMES;
  }

  for (int f = 0; f < n50; ++f) {
    const float t = float(f) * CONTROL_DT * PLAN_HZ;
    int f0 = int(floorf(t));
    if (f0 > n30 - 1) f0 = n30 - 1;
    int f1 = f0 + 1 > n30 - 1 ? n30 - 1 : f0 + 1;
    const float w = t - floorf(t);
    const float* a = raw + size_t(f0) * QPOS;
    const float* b = raw + size_t(f1) * QPOS;
    float* o = g + size_t(f) * FRAME;
    for (int k = 0; k < 3; ++k) o[F_POS + k] = a[k] * (1.0f - w) + b[k] * w;
    float qa[4] = {a[3], a[4], a[5], a[6]};
    float qb[4] = {b[3], b[4], b[5], b[6]};
    q_slerp(qa, qb, w, o + F_QUAT);
    q_norm(o + F_QUAT);
    for (int j = 0; j < NUM_ACTIONS; ++j) {
      const int mj = D_MJ_OF_IL[j];
      o[F_Q + j] = a[7 + mj] * (1.0f - w) + b[7 + mj] * w;
    }
  }

  if (n50 == 0) {
    for (int f = 0; f < old_len; ++f) {
      for (int k = 0; k < FRAME; ++k) {
        mo[size_t(f) * FRAME + k] = mi[size_t(f) * FRAME + k];
      }
    }
    return;
  }

  if (old_len == 0) {
    for (int f = 0; f < n50; ++f) {
      for (int k = 0; k < FRAME; ++k) {
        mo[size_t(f) * FRAME + k] = g[size_t(f) * FRAME + k];
      }
    }
    len[env] = n50;
    cursor[env] = 0;

    psiref[env] = q_yaw(mo + F_QUAT);
    yaw0[env] = q_yaw(base_quat + env * 4);
    yaw_off[env] = psiref[env] - yaw0[env];
    return;
  }

  int length = LOOKAHEAD + n50;
  if (length > MAX_FRAMES) length = MAX_FRAMES;
  for (int f = 0; f < length; ++f) {
    const float* old = frame_at(mi, old_len, f + cur);
    int fi = f - LOOKAHEAD;
    fi = fi < 0 ? 0 : (fi > n50 - 1 ? n50 - 1 : fi);
    const float* fresh = g + size_t(fi) * FRAME;
    float w = float(f - LOOKAHEAD) / float(BLEND_FRAMES);
    w = w < 0.0f ? 0.0f : (w > 1.0f ? 1.0f : w);

    float* o = mo + size_t(f) * FRAME;
    for (int k = 0; k < 3; ++k) {
      o[F_POS + k] = old[F_POS + k] * (1.0f - w) + fresh[F_POS + k] * w;
    }
    q_slerp(old + F_QUAT, fresh + F_QUAT, w, o + F_QUAT);
    for (int j = 0; j < NUM_ACTIONS; ++j) {
      o[F_Q + j] = old[F_Q + j] * (1.0f - w) + fresh[F_Q + j] * w;
    }
  }
  len[env] = length;
  cursor[env] = 0;
}

__global__ void k_mimic_lite_past(
    const float* __restrict__ motion,
    const float* __restrict__ arm_pose,
    const int* __restrict__ len,
    const int* __restrict__ cursor,
    float* __restrict__ past,
    long long tick,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* m = motion + size_t(env) * MAX_FRAMES * FRAME;
  const float* now = frame_at(m, len[env], cursor[env]);
  float* ring = past + size_t(env) * PAST_RING * FRAME;

  const int first = tick == 0 ? 0 : ring_slot(tick, 0, PAST_RING);
  const int last = tick == 0 ? PAST_RING - 1 : first;
  for (int s = first; s <= last; ++s) {
    float* slot = ring + size_t(s) * FRAME;
    for (int k = 0; k < FRAME; ++k) slot[k] = now[k];
    for (int j = 0; j < NUM_ACTIONS; ++j) {
      const int mj = D_MJ_OF_IL[j];
      if (mj >= NUM_OWNED)
        slot[F_Q + j] = arm_pose[env * POLICY_NUM_MOTOR + mj];
    }
  }
}

__global__ void k_mimic_lite_state(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    float* __restrict__ hist_gyro,
    float* __restrict__ hist_grav,
    float* __restrict__ hist_q,
    float* __restrict__ hist_dq,
    long long tick,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const int first = tick == 0 ? 0 : ring_slot(tick, 0, HIST_RING);
  const int last = tick == 0 ? HIST_RING - 1 : first;

  for (int s = first; s <= last; ++s) {
    float* hg = hist_gyro + (size_t(env) * HIST_RING + s) * 3;
    float* hv = hist_grav + (size_t(env) * HIST_RING + s) * 3;
    for (int k = 0; k < 3; ++k) {
      hg[k] = gyro[env * 3 + k];
      hv[k] = gravity[env * 3 + k];
    }
    float* hq = hist_q + (size_t(env) * HIST_RING + s) * NUM_ACTIONS;
    float* hd = hist_dq + (size_t(env) * HIST_RING + s) * NUM_ACTIONS;
    for (int j = 0; j < NUM_ACTIONS; ++j) {
      const int mj = D_MJ_OF_IL[j];
      hq[j] = motor_q[env * POLICY_NUM_MOTOR + mj];
      hd[j] = motor_dq[env * POLICY_NUM_MOTOR + mj];
    }
  }
}

__global__ void k_mimic_lite_obs(
    const float* __restrict__ motion,
    const float* __restrict__ arm_pose,
    const int* __restrict__ len,
    const int* __restrict__ cursor,
    const float* __restrict__ past,
    const float* __restrict__ base_quat,
    const float* __restrict__ yaw_off,
    const float* __restrict__ hist_gyro,
    const float* __restrict__ hist_grav,
    const float* __restrict__ hist_q,
    const float* __restrict__ hist_dq,
    const float* __restrict__ prev_act,
    float* __restrict__ cmd_obs,
    float* __restrict__ pol_obs,
    long long tick,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* m = motion + size_t(env) * MAX_FRAMES * FRAME;
  const float* ring = past + size_t(env) * PAST_RING * FRAME;
  const float* here = frame_at(m, len[env], cursor[env]);

  const float yaw = q_yaw(here + F_QUAT);
  const float c = cosf(yaw), s = sinf(yaw);

  const float off[4] =
      {cosf(0.5f * yaw_off[env]), 0.0f, 0.0f, sinf(0.5f * yaw_off[env])};
  float aligned[4];
  q_mul(off, base_quat + env * 4, aligned);
  const float inv[4] = {aligned[0], -aligned[1], -aligned[2], -aligned[3]};

  float* o = cmd_obs + size_t(env) * CMD_OBS;
  for (int k = 0; k < REF_STEPS; ++k) {
    const int step = D_REF_STEP[k];
    const float* fr =
        step < 0 ? ring + size_t(ring_slot(tick, -step, PAST_RING)) * FRAME
                 : frame_at(m, len[env], cursor[env] + step);

    const float dx = fr[F_POS + 0] - here[F_POS + 0];
    const float dy = fr[F_POS + 1] - here[F_POS + 1];
    o[k * 3 + 0] = c * dx + s * dy;
    o[k * 3 + 1] = -s * dx + c * dy;
    o[k * 3 + 2] = fr[F_POS + 2];

    float rel[4];
    q_mul(inv, fr + F_QUAT, rel);
    rot6_rows(rel, o + 24 + k * 6);

    for (int j = 0; j < NUM_ACTIONS; ++j) {
      const int mj = D_MJ_OF_IL[j];
      o[72 + k * NUM_ACTIONS + j] = mj < NUM_OWNED || step < 0
                                        ? fr[F_Q + j]
                                        : arm_pose[env * POLICY_NUM_MOTOR + mj];
    }
  }

  float* p = pol_obs + size_t(env) * POL_OBS;
  for (int i = 0; i < HIST_STEPS; ++i) {
    const int slot = ring_slot(tick, D_HIST_STEP[i], HIST_RING);
    const float* hg = hist_gyro + (size_t(env) * HIST_RING + slot) * 3;
    const float* hv = hist_grav + (size_t(env) * HIST_RING + slot) * 3;
    for (int k = 0; k < 3; ++k) {
      p[i * 3 + k] = hg[k];
      p[21 + i * 3 + k] = hv[k];
    }
    const float* hq = hist_q + (size_t(env) * HIST_RING + slot) * NUM_ACTIONS;
    const float* hd = hist_dq + (size_t(env) * HIST_RING + slot) * NUM_ACTIONS;
    for (int j = 0; j < NUM_ACTIONS; ++j) {
      p[42 + i * NUM_ACTIONS + j] = hq[j];
      p[245 + i * NUM_ACTIONS + j] = hd[j];
    }
  }
  const float* pa = prev_act + size_t(env) * ACT_STEPS * NUM_ACTIONS;
  for (int i = 0; i < ACT_STEPS * NUM_ACTIONS; ++i) p[448 + i] = pa[i];
}

__global__ void k_mimic_lite_act(
    const float* __restrict__ action,
    const float* __restrict__ arm_pose,
    const int* __restrict__ len,
    float* __restrict__ prev_act,
    int* __restrict__ cursor,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = action + size_t(env) * NUM_ACTIONS;
  float* pa = prev_act + size_t(env) * ACT_STEPS * NUM_ACTIONS;
  for (int s = ACT_STEPS - 1; s > 0; --s) {
    for (int j = 0; j < NUM_ACTIONS; ++j) {
      pa[s * NUM_ACTIONS + j] = pa[(s - 1) * NUM_ACTIONS + j];
    }
  }
  for (int j = 0; j < NUM_ACTIONS; ++j) pa[j] = a[j];

  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  }
  for (int mj = 0; mj < NUM_OWNED; ++mj) {
    const int il = D_IL_OF_MJ[mj];
    q_target[env * POLICY_NUM_MOTOR + mj] =
        D_DEFAULTS[mj] + a[il] * D_SCALE_IL[il];
  }

  const int n = len[env];
  if (n > 0) {
    const int c = cursor[env] + 1;
    cursor[env] = c > n - 1 ? n - 1 : c;
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> planner, actor;

  float *d_ctx = nullptr, *d_mode = nullptr, *d_vel = nullptr;
  float *d_move = nullptr, *d_face = nullptr;
  float *d_qpos = nullptr, *d_nframes = nullptr;
  float *d_motion = nullptr, *d_motion2 = nullptr, *d_gen = nullptr;
  int *d_len = nullptr, *d_cursor = nullptr;
  float *d_psiref = nullptr, *d_yaw0 = nullptr, *d_yaw_off = nullptr;
  float* d_past = nullptr;
  float *d_hist_gyro = nullptr, *d_hist_grav = nullptr;
  float *d_hist_q = nullptr, *d_hist_dq = nullptr;
  float* d_prev = nullptr;
  float *d_cmd_obs = nullptr, *d_pol_obs = nullptr;
  float *d_act = nullptr, *d_priv = nullptr;

  float *d_seed = nullptr, *d_height = nullptr, *d_hst = nullptr;
  float *d_stp = nullptr, *d_sth = nullptr, *d_allow = nullptr;

  long long ticks = 0;
  int envs = 0;

  ~Policy() override {
    for (void* p : {(void*)d_ctx,     (void*)d_mode,      (void*)d_vel,
                    (void*)d_move,    (void*)d_face,      (void*)d_qpos,
                    (void*)d_nframes, (void*)d_motion,    (void*)d_motion2,
                    (void*)d_gen,     (void*)d_len,       (void*)d_cursor,
                    (void*)d_psiref,  (void*)d_yaw0,      (void*)d_yaw_off,
                    (void*)d_past,    (void*)d_hist_gyro, (void*)d_hist_grav,
                    (void*)d_hist_q,  (void*)d_hist_dq,   (void*)d_prev,
                    (void*)d_cmd_obs, (void*)d_pol_obs,   (void*)d_act,
                    (void*)d_priv,    (void*)d_seed,      (void*)d_height,
                    (void*)d_hst,     (void*)d_stp,       (void*)d_sth,
                    (void*)d_allow}) {
      if (p) cudaFree(p);
    }
  }

  static float* zeros(size_t n) {
    float* p = nullptr;
    cudaMalloc(&p, n * sizeof(float));
    cudaMemset(p, 0, n * sizeof(float));
    return p;
  }

  static float* filled(
      int n,
      float value
  ) {
    float* p = zeros(size_t(n));
    const std::vector<float> h(size_t(n), value);
    cudaMemcpy(p, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return p;
  }

  void init(int n) override {
    envs = n;

    planner = policy_api::engine_make(
        "policies/sonic/planner_sonic_rows.onnx",
        n,
        {{"context_mujoco_qpos", {-1, CTX_FRAMES, QPOS}},
         {"mode", {-1}},
         {"target_vel", {-1}},
         {"movement_direction", {-1, 3}},
         {"facing_direction", {-1, 3}},
         {"random_seed", {-1}},
         {"height", {-1}},
         {"has_specific_target", {-1, 1}},
         {"specific_target_positions", {-1, CTX_FRAMES, 3}},
         {"specific_target_headings", {-1, CTX_FRAMES}},
         {"allowed_pred_num_tokens", {-1, 11}}},
        {{"mujoco_qpos", {-1, PLAN_FRAMES, QPOS}}, {"num_pred_frames", {-1}}}
    );

    actor = policy_api::engine_make(
        "policies/mimic_lite/model.onnx",
        n,
        {{"command", {-1, CMD_OBS}}, {"policy", {-1, POL_OBS}}},
        {{"action", {-1, NUM_ACTIONS}}, {"priv_student", {-1, PRIV_DIM}}}
    );

    d_ctx = zeros(size_t(n) * CTX_FRAMES * QPOS);
    d_mode = zeros(size_t(n));
    d_vel = zeros(size_t(n));
    d_move = zeros(size_t(n) * 3);
    d_face = zeros(size_t(n) * 3);
    d_qpos = zeros(size_t(n) * PLAN_FRAMES * QPOS);
    d_nframes = zeros(size_t(n));
    d_motion = zeros(size_t(n) * MAX_FRAMES * FRAME);
    d_motion2 = zeros(size_t(n) * MAX_FRAMES * FRAME);
    d_gen = zeros(size_t(n) * MAX_FRAMES * FRAME);
    cudaMalloc(&d_len, size_t(n) * sizeof(int));
    cudaMemset(d_len, 0, size_t(n) * sizeof(int));
    cudaMalloc(&d_cursor, size_t(n) * sizeof(int));
    cudaMemset(d_cursor, 0, size_t(n) * sizeof(int));
    d_psiref = zeros(size_t(n));
    d_yaw0 = zeros(size_t(n));
    d_yaw_off = zeros(size_t(n));
    d_past = zeros(size_t(n) * PAST_RING * FRAME);
    d_hist_gyro = zeros(size_t(n) * HIST_RING * 3);
    d_hist_grav = zeros(size_t(n) * HIST_RING * 3);
    d_hist_q = zeros(size_t(n) * HIST_RING * NUM_ACTIONS);
    d_hist_dq = zeros(size_t(n) * HIST_RING * NUM_ACTIONS);
    d_prev = zeros(size_t(n) * ACT_STEPS * NUM_ACTIONS);
    d_cmd_obs = zeros(size_t(n) * CMD_OBS);
    d_pol_obs = zeros(size_t(n) * POL_OBS);
    d_act = zeros(size_t(n) * NUM_ACTIONS);
    d_priv = zeros(size_t(n) * PRIV_DIM);

    d_seed = filled(n, PLAN_SEED);
    d_height = filled(n, -1.0f);
    d_hst = zeros(size_t(n));
    d_stp = zeros(size_t(n) * CTX_FRAMES * 3);
    d_sth = zeros(size_t(n) * CTX_FRAMES);
    d_allow = zeros(size_t(n) * 11);
    std::vector<float> allow(size_t(n) * 11, 0.0f);
    for (int e = 0; e < n; ++e)
      for (int k = 0; k < 6; ++k) allow[size_t(e) * 11 + k] = 1.0f;
    cudaMemcpy(
        d_allow,
        allow.data(),
        allow.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );
  }

  void plan() {
    const float* in[11] = {
        d_ctx,
        d_mode,
        d_vel,
        d_move,
        d_face,
        d_seed,
        d_height,
        d_hst,
        d_stp,
        d_sth,
        d_allow
    };
    float* out[2] = {d_qpos, d_nframes};
    policy_api::engine_run(*planner, in, out, envs);
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;

    k_mimic_lite_drive<<<blocks, threads>>>(
        c.cmd,
        c.base_quat,
        d_psiref,
        d_yaw0,
        d_len,
        d_mode,
        d_vel,
        d_move,
        d_face,
        envs
    );

    if (ticks % REPLAN_TICKS == 0) {
      k_mimic_lite_context<<<blocks, threads>>>(
          d_motion,
          d_len,
          d_cursor,
          c.motor_q,
          d_ctx,
          envs
      );
      plan();
      k_mimic_lite_apply<<<blocks, threads>>>(
          c.base_quat,
          d_qpos,
          d_nframes,
          d_gen,
          d_motion,
          d_motion2,
          d_len,
          d_cursor,
          d_psiref,
          d_yaw0,
          d_yaw_off,
          envs
      );
      std::swap(d_motion, d_motion2);
    }

    k_mimic_lite_past<<<blocks, threads>>>(
        d_motion,
        c.arm_pose,
        d_len,
        d_cursor,
        d_past,
        ticks,
        envs
    );

    k_mimic_lite_state<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        d_hist_gyro,
        d_hist_grav,
        d_hist_q,
        d_hist_dq,
        ticks,
        envs
    );

    k_mimic_lite_obs<<<blocks, threads>>>(
        d_motion,
        c.arm_pose,
        d_len,
        d_cursor,
        d_past,
        c.base_quat,
        d_yaw_off,
        d_hist_gyro,
        d_hist_grav,
        d_hist_q,
        d_hist_dq,
        d_prev,
        d_cmd_obs,
        d_pol_obs,
        ticks,
        envs
    );

    const float* in[2] = {d_cmd_obs, d_pol_obs};
    float* out[2] = {d_act, d_priv};
    policy_api::engine_run(*actor, in, out, envs);

    k_mimic_lite_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_len,
        d_prev,
        d_cursor,
        c.q_target,
        envs
    );
    ++ticks;
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_OWNED; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "mimic_lite"; }
};

}
