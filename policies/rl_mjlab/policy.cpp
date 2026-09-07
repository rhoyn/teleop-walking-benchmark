namespace rl_mjlab {

constexpr int NUM_OBS = 98;
constexpr int NUM_ACTIONS = 29;
constexpr int FIRST_ARM_MOTOR = 15;
constexpr float CONTROL_DT = 0.02f;
constexpr float GAIT_PERIOD = 0.6f;
constexpr float GAIT_CMD_DEADBAND = 0.1f;

__device__ const float D_DEFAULT_POS[NUM_ACTIONS] = {
    -0.1f, 0.0f, 0.0f,  0.3f,   -0.2f, 0.0f,  -0.1f, 0.0f, 0.0f,  0.3f,
    -0.2f, 0.0f, 0.0f,  0.0f,   0.0f,  0.35f, 0.18f, 0.0f, 0.87f, 0.0f,
    0.0f,  0.0f, 0.35f, -0.18f, 0.0f,  0.87f, 0.0f,  0.0f, 0.0f
};

__device__ const float D_ACTION_SCALE[NUM_ACTIONS] = {
    0.55f, 0.35f, 0.55f, 0.35f, 0.44f, 0.44f, 0.55f, 0.35f, 0.55f, 0.35f,
    0.44f, 0.44f, 0.55f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f,
    0.07f, 0.07f, 0.44f, 0.44f, 0.44f, 0.44f, 0.44f, 0.07f, 0.07f
};

const float KPS[NUM_ACTIONS] = {40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,
                                40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,
                                40.2f, 28.5f, 28.5f, 14.3f, 14.3f, 14.3f,
                                14.3f, 14.3f, 16.8f, 16.8f, 14.3f, 14.3f,
                                14.3f, 14.3f, 14.3f, 16.8f, 16.8f};

const float KDS[NUM_ACTIONS] = {2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f, 2.6f, 6.3f,
                                2.6f, 6.3f, 1.8f, 1.8f, 2.6f, 1.8f, 1.8f, 0.9f,
                                0.9f, 0.9f, 0.9f, 0.9f, 1.1f, 1.1f, 0.9f, 0.9f,
                                0.9f, 0.9f, 0.9f, 1.1f, 1.1f};

const policy_api::Limits LIMITS = {-0.5, 1.0, 0.5, 1.0, 0.0};

__global__ void k_rl_mjlab_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    float phase,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* o = obs + env * NUM_OBS;
  const float* c = cmd + env * 3;
  for (int k = 0; k < 3; ++k) {
    o[k] = gyro[env * 3 + k];
    o[3 + k] = gravity[env * 3 + k];
    o[6 + k] = c[k];
  }
  const float norm = sqrtf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
  const bool gait_on = norm >= GAIT_CMD_DEADBAND;
  o[9] = gait_on ? sinf(phase * 2.0f * float(M_PI)) : 0.0f;
  o[10] = gait_on ? cosf(phase * 2.0f * float(M_PI)) : 0.0f;
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    o[11 + j] = motor_q[env * POLICY_NUM_MOTOR + j] - D_DEFAULT_POS[j];
    o[11 + NUM_ACTIONS + j] = motor_dq[env * POLICY_NUM_MOTOR + j];
    o[11 + 2 * NUM_ACTIONS + j] = last_action[env * NUM_ACTIONS + j];
  }
}

__global__ void k_rl_mjlab_act(
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

  for (int k = 0; k < NUM_ACTIONS; ++k)
    last_action[env * NUM_ACTIONS + k] = a[k];
  for (int j = 0; j < FIRST_ARM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] =
        D_DEFAULT_POS[j] + a[j] * D_ACTION_SCALE[j];
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  double gait_phase = 0.0;
  int envs = 0;

  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/rl_mjlab/model.onnx",
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
    gait_phase = std::fmod(gait_phase + CONTROL_DT / GAIT_PERIOD, 1.0);
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_rl_mjlab_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_obs,
        float(gait_phase),
        envs
    );
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_rl_mjlab_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return FIRST_ARM_MOTOR; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "rl_mjlab"; }
};

}
