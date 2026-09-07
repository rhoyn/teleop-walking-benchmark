namespace decoupled_wbc {

constexpr int NUM_ACTIONS = 15;
constexpr int FRAME = 58;
constexpr int HISTORY = 2;
constexpr int NUM_OBS = FRAME * HISTORY;
constexpr float ACTION_SCALE = 0.25f;
constexpr float CMD_HEIGHT = 0.70f;
constexpr float CMD_BODY_ROLL = 0.0f, CMD_BODY_PITCH = 0.0f,
                CMD_BODY_YAW = 0.0f;

constexpr int WAIST_PITCH = 14;
constexpr float WAIST_PITCH_MIN = -0.60f, WAIST_PITCH_MAX = 0.60f;

__device__ const float D_DEFAULTS[NUM_ACTIONS] = {
    -0.20f,
    0.0f,
    0.0f,
    0.42f,
    -0.23f,
    0.0f,
    -0.20f,
    0.0f,
    0.0f,
    0.42f,
    -0.23f,
    0.0f,
    0.0f,
    0.0f,
    0.0f
};

const float KPS[NUM_ACTIONS] =
    {150, 150, 100, 150, 40, 40, 150, 150, 100, 150, 40, 40, 200, 150, 150};
const float KDS[NUM_ACTIONS] = {3, 3, 2, 4, 2, 2, 3, 3, 2, 4, 2, 2, 4, 4, 4};

const policy_api::Limits LIMITS = {-0.55, 0.55, 0.55, 1.57, 0.0};

__global__ void k_decoupled_wbc_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* o = obs + env * NUM_OBS;
  for (int i = 0; i < FRAME * (HISTORY - 1); ++i) o[i] = o[i + FRAME];

  float* f = o + FRAME * (HISTORY - 1);
  for (int k = 0; k < 3; ++k) {
    f[k] = gyro[env * 3 + k];
    f[3 + k] = gravity[env * 3 + k];
    f[6 + k] = cmd[env * 3 + k];
  }
  f[9] = CMD_HEIGHT;
  f[10] = CMD_BODY_ROLL;
  f[11] = CMD_BODY_PITCH;
  f[12] = CMD_BODY_YAW;

  for (int j = 0; j < NUM_ACTIONS; ++j) {
    f[13 + j] = motor_q[env * POLICY_NUM_MOTOR + j] - D_DEFAULTS[j];
    f[13 + NUM_ACTIONS + j] = motor_dq[env * POLICY_NUM_MOTOR + j];
    f[13 + 2 * NUM_ACTIONS + j] = last_action[env * NUM_ACTIONS + j];
  }
}

__global__ void k_decoupled_wbc_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + env * NUM_ACTIONS;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  }
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    last_action[env * NUM_ACTIONS + j] = a[j];
    q_target[env * POLICY_NUM_MOTOR + j] = D_DEFAULTS[j] + a[j] * ACTION_SCALE;
  }

  float* w = q_target + env * POLICY_NUM_MOTOR + WAIST_PITCH;
  *w = fminf(fmaxf(*w, WAIST_PITCH_MIN), WAIST_PITCH_MAX);
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  int envs = 0;

  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/decoupled_wbc/model.onnx",
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
    k_decoupled_wbc_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_obs,
        envs
    );
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_decoupled_wbc_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_ACTIONS; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "decoupled_wbc"; }
};

}
