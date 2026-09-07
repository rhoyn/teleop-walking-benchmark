namespace rl_lab {

constexpr int NUM_ACTIONS = 29;
constexpr int HISTORY = 5;
constexpr int NUM_TERMS = 6;
constexpr int NUM_OBS = HISTORY * (3 + 3 + 3 + 3 * NUM_ACTIONS);
constexpr float ACTION_SCALE = 0.25f;
constexpr float ACTION_CLIP = 10.0f;
constexpr float ANG_VEL_SCALE = 0.2f;
constexpr float JOINT_VEL_SCALE = 0.05f;

constexpr int OWNED = 15;

__device__ const int D_TERM_DIM[NUM_TERMS] =
    {3, 3, 3, NUM_ACTIONS, NUM_ACTIONS, NUM_ACTIONS};
__device__ const int D_TERM_OFF[NUM_TERMS] = {
    0,
    HISTORY * 3,
    HISTORY * 6,
    HISTORY * 9,
    HISTORY * (9 + NUM_ACTIONS),
    HISTORY * (9 + 2 * NUM_ACTIONS)
};

__device__ const int D_ISAAC_TO_MUJOCO[NUM_ACTIONS] = {
    0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4, 10,
    16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
};

__device__ const float D_DEFAULT_ISAAC[NUM_ACTIONS] = {
    -0.1f, -0.1f, 0.0f,  0.0f,  0.0f,   0.0f,  0.0f,   0.0f, 0.0f, 0.3f,
    0.3f,  0.3f,  0.3f,  -0.2f, -0.2f,  0.25f, -0.25f, 0.0f, 0.0f, 0.0f,
    0.0f,  0.97f, 0.97f, 0.15f, -0.15f, 0.0f,  0.0f,   0.0f, 0.0f
};

const float KPS[NUM_ACTIONS] = {100, 100, 100, 150, 40,  40, 100, 100, 100, 150,
                                40,  40,  200, 200, 200, 40, 40,  40,  40,  40,
                                40,  40,  40,  40,  40,  40, 40,  40,  40};
const float KDS[NUM_ACTIONS] = {2,  2,  2,  4,  2,  2,  2,  2,  2,  4,
                                2,  2,  5,  5,  5,  10, 10, 10, 10, 10,
                                10, 10, 10, 10, 10, 10, 10, 10, 10};

const policy_api::Limits LIMITS =
    {-0.5, 1.0, 0.3, 0.2, 0.0, 0.10, 0.20, 0.05, 0.12, 2.0, 1.5};

__global__ void k_rl_lab_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    bool primed,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float frame[NUM_TERMS][NUM_ACTIONS];
  for (int k = 0; k < 3; ++k) {
    frame[0][k] = gyro[env * 3 + k] * ANG_VEL_SCALE;
    frame[1][k] = gravity[env * 3 + k];
    frame[2][k] = cmd[env * 3 + k];
  }
  const float* la = last_action + env * NUM_ACTIONS;
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const int motor = D_ISAAC_TO_MUJOCO[i];
    if (motor >= OWNED) {
      frame[3][i] = la[i] * ACTION_SCALE;
      frame[4][i] = 0.0f;
    } else {
      frame[3][i] =
          motor_q[env * POLICY_NUM_MOTOR + motor] - D_DEFAULT_ISAAC[i];
      frame[4][i] = motor_dq[env * POLICY_NUM_MOTOR + motor] * JOINT_VEL_SCALE;
    }
    frame[5][i] = la[i];
  }

  float* o = obs + env * NUM_OBS;
  for (int t = 0; t < NUM_TERMS; ++t) {
    const int dim = D_TERM_DIM[t];
    float* base = o + D_TERM_OFF[t];
    if (primed) {
      for (int i = 0; i < dim * (HISTORY - 1); ++i) base[i] = base[i + dim];
    } else {
      for (int h = 0; h < HISTORY - 1; ++h) {
        for (int i = 0; i < dim; ++i) base[h * dim + i] = frame[t][i];
      }
    }
    for (int i = 0; i < dim; ++i) base[(HISTORY - 1) * dim + i] = frame[t][i];
  }
}

__global__ void k_rl_lab_act(
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
  float* q = q_target + env * POLICY_NUM_MOTOR;

  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    q[j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const float v = a[i];
    const float clipped =
        isfinite(v) ? fminf(fmaxf(v, -ACTION_CLIP), ACTION_CLIP) : 0.0f;
    la[i] = clipped;
    const int motor = D_ISAAC_TO_MUJOCO[i];
    if (motor < OWNED) q[motor] = D_DEFAULT_ISAAC[i] + clipped * ACTION_SCALE;
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  int envs = 0;

  bool primed = false;

  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/rl_lab/model.onnx",
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
    k_rl_lab_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_obs,
        primed,
        envs
    );
    primed = true;
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_rl_lab_act<<<blocks, threads>>>(
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
  const char* name() const override { return "rl_lab"; }
};

}
