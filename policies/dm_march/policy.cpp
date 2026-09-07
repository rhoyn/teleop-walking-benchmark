namespace dm_march {

constexpr int NUM_ACTIONS = 29;
constexpr int HISTORY = 5;
constexpr int NUM_TERMS = 7;
constexpr int GAIT_DIM = 2;
constexpr int NUM_OBS = HISTORY * (3 + 3 + 3 + 3 * NUM_ACTIONS + GAIT_DIM);
constexpr int OWNED = 15;

constexpr float GAIT_PERIOD = 0.8f;
constexpr float STEP_DT = 0.02f;
constexpr float ACTION_SCALE = 0.25f;
constexpr float TARGET_MIN = -1.0f, TARGET_MAX = 1.0f;
constexpr float ACTION_CLIP = 10.0f;
constexpr float ANG_VEL_SCALE = 0.2f;
constexpr float JOINT_VEL_SCALE = 0.05f;

constexpr int OFF_GYRO = 0;
constexpr int OFF_GRAV = OFF_GYRO + HISTORY * 3;
constexpr int OFF_CMD = OFF_GRAV + HISTORY * 3;
constexpr int OFF_QPOS = OFF_CMD + HISTORY * 3;
constexpr int OFF_QVEL = OFF_QPOS + HISTORY * NUM_ACTIONS;
constexpr int OFF_ACT = OFF_QVEL + HISTORY * NUM_ACTIONS;
constexpr int OFF_GAIT = OFF_ACT + HISTORY * NUM_ACTIONS;

#define DM_MARCH_DEFAULTS                                                      \
  -0.1f, -0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.3f, 0.3f,    \
      0.3f, -0.2f, -0.2f, 0.25f, -0.25f, 0.0f, 0.0f, 0.0f, 0.0f, 0.97f, 0.97f, \
      0.15f, -0.15f, 0.0f, 0.0f, 0.0f, 0.0f

#define DM_MARCH_ISAAC2MJ                                                   \
  0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, \
      18, 25, 19, 26, 20, 27, 21, 28

const float DEFAULTS[NUM_ACTIONS] = {DM_MARCH_DEFAULTS};
__device__ const float D_DEFAULTS[NUM_ACTIONS] = {DM_MARCH_DEFAULTS};
__device__ const int D_ISAAC2MJ[NUM_ACTIONS] = {DM_MARCH_ISAAC2MJ};

const float KPS[NUM_ACTIONS] = {100, 100, 100, 150, 40, 40, 100, 100, 100, 150,
                                40,  40,  200, 40,  40, 40, 40,  40,  40,  40,
                                40,  40,  40,  40,  40, 40, 40,  40,  40};
const float KDS[NUM_ACTIONS] = {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

const policy_api::Limits LIMITS = {-0.3, 0.7, 0.2, 0.0, 0.0};

__device__ inline void dm_march_push(
    float* base,
    int dim,
    const float* f,
    bool primed
) {
  if (primed) {
    for (int h = 0; h < HISTORY - 1; ++h) {
      for (int k = 0; k < dim; ++k) base[h * dim + k] = base[(h + 1) * dim + k];
    }
    for (int k = 0; k < dim; ++k) base[(HISTORY - 1) * dim + k] = f[k];
  } else {
    for (int h = 0; h < HISTORY; ++h) {
      for (int k = 0; k < dim; ++k) base[h * dim + k] = f[k];
    }
  }
}

__global__ void k_dm_march_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    int envs,
    int step_index,
    int primed
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* o = obs + env * NUM_OBS;
  const float* la = last_action + env * NUM_ACTIONS;

  float f[3];
  for (int k = 0; k < 3; ++k) f[k] = gyro[env * 3 + k] * ANG_VEL_SCALE;
  dm_march_push(o + OFF_GYRO, 3, f, primed);
  for (int k = 0; k < 3; ++k) f[k] = gravity[env * 3 + k];
  dm_march_push(o + OFF_GRAV, 3, f, primed);
  for (int k = 0; k < 3; ++k) f[k] = cmd[env * 3 + k];
  dm_march_push(o + OFF_CMD, 3, f, primed);

  float fq[NUM_ACTIONS], fv[NUM_ACTIONS], fa[NUM_ACTIONS];
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const int motor = D_ISAAC2MJ[i];

    if (motor >= OWNED) {
      fq[i] = la[i] * ACTION_SCALE;
      fv[i] = 0.0f;
    } else {
      fq[i] = motor_q[env * POLICY_NUM_MOTOR + motor] - D_DEFAULTS[i];
      fv[i] = motor_dq[env * POLICY_NUM_MOTOR + motor] * JOINT_VEL_SCALE;
    }
    fa[i] = la[i];
  }
  dm_march_push(o + OFF_QPOS, NUM_ACTIONS, fq, primed);
  dm_march_push(o + OFF_QVEL, NUM_ACTIONS, fv, primed);
  dm_march_push(o + OFF_ACT, NUM_ACTIONS, fa, primed);

  const float elapsed = float(step_index) * STEP_DT;
  const float phase = fmodf(elapsed, GAIT_PERIOD) / GAIT_PERIOD;
  float fg[GAIT_DIM];
  fg[0] = sinf(phase * 2.0f * float(M_PI));
  fg[1] = cosf(phase * 2.0f * float(M_PI));
  dm_march_push(o + OFF_GAIT, GAIT_DIM, fg, primed);
}

__global__ void k_dm_march_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + env * NUM_ACTIONS;
  float* la = last_action + env * NUM_ACTIONS;
  float* qt = q_target + env * POLICY_NUM_MOTOR;

  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    qt[j] = arm_pose[env * POLICY_NUM_MOTOR + j];

  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const float raw = a[i];
    const float clipped =
        isfinite(raw) ? fminf(fmaxf(raw, -ACTION_CLIP), ACTION_CLIP) : 0.0f;
    la[i] = clipped;
    const int motor = D_ISAAC2MJ[i];
    if (motor < OWNED) {
      const float target = D_DEFAULTS[i] + clipped * ACTION_SCALE;
      qt[motor] = fminf(fmaxf(target, TARGET_MIN), TARGET_MAX);
    }
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  int envs = 0;
  long step_index = 0;
  bool primed = false;

  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last})
      if (p) cudaFree(p);
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/dm_march/model.onnx",
        n,
        NUM_OBS,
        NUM_ACTIONS
    );
    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_ACTIONS * sizeof(float));
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_dm_march_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_obs,
        envs,
        int(step_index),
        primed ? 1 : 0
    );

    step_index += 1;
    primed = true;
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_dm_march_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return OWNED; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "dm_march"; }
};

}
