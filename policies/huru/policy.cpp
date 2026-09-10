namespace huru {

constexpr int NUM_OBS = 99;
constexpr int NUM_ACTIONS = 29;
constexpr int FIRST_ARM_MOTOR = 15;

__device__ const float D_DEFAULT_POS[NUM_ACTIONS] = {
    -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f, -0.312f, 0.0f, 0.0f, 0.669f,
    -0.363f, 0.0f, 0.0f, 0.0f,   0.0f,    0.2f, 0.2f,    0.0f, 0.6f, 0.0f,
    0.0f,    0.0f, 0.2f, -0.2f,  0.0f,    0.6f, 0.0f,    0.0f, 0.0f
};

__device__ const float D_ACTION_SCALE[NUM_ACTIONS] = {
    0.548f, 0.351f, 0.548f, 0.351f, 0.439f, 0.439f, 0.548f, 0.351f,
    0.548f, 0.351f, 0.439f, 0.439f, 0.548f, 0.439f, 0.439f, 0.439f,
    0.439f, 0.439f, 0.439f, 0.439f, 0.075f, 0.075f, 0.439f, 0.439f,
    0.439f, 0.439f, 0.439f, 0.075f, 0.075f
};

const float KPS[NUM_ACTIONS] = {40.179f, 99.098f, 40.179f, 99.098f, 28.501f,
                                28.501f, 40.179f, 99.098f, 40.179f, 99.098f,
                                28.501f, 28.501f, 40.179f, 28.501f, 28.501f,
                                14.251f, 14.251f, 14.251f, 14.251f, 14.251f,
                                16.778f, 16.778f, 14.251f, 14.251f, 14.251f,
                                14.251f, 14.251f, 16.778f, 16.778f};

const float KDS[NUM_ACTIONS] = {2.558f, 6.309f, 2.558f, 6.309f, 1.814f, 1.814f,
                                2.558f, 6.309f, 2.558f, 6.309f, 1.814f, 1.814f,
                                2.558f, 1.814f, 1.814f, 0.907f, 0.907f, 0.907f,
                                0.907f, 0.907f, 1.068f, 1.068f, 0.907f, 0.907f,
                                0.907f, 0.907f, 0.907f, 1.068f, 1.068f};

const policy_api::Limits LIMITS = {-0.9, 0.9, 0.85, 0.5, 0.0};

__global__ void k_huru_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ lin_vel,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* o = obs + env * NUM_OBS;
  for (int k = 0; k < 3; ++k) {
    o[k] = lin_vel[env * 3 + k];
    o[3 + k] = gyro[env * 3 + k];
    o[6 + k] = gravity[env * 3 + k];
    o[96 + k] = cmd[env * 3 + k];
  }
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    o[9 + j] = motor_q[env * POLICY_NUM_MOTOR + j] - D_DEFAULT_POS[j];
    o[9 + NUM_ACTIONS + j] = motor_dq[env * POLICY_NUM_MOTOR + j];
    o[9 + 2 * NUM_ACTIONS + j] = last_action[env * NUM_ACTIONS + j];
  }
}

__global__ void k_huru_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + env * NUM_ACTIONS;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];

  for (int k = 0; k < NUM_ACTIONS; ++k)
    last_action[env * NUM_ACTIONS + k] = a[k];
  for (int j = 0; j < FIRST_ARM_MOTOR; ++j)
    q_target[env * POLICY_NUM_MOTOR + j] =
        D_DEFAULT_POS[j] + a[j] * D_ACTION_SCALE[j];
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  int envs = 0;

  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last})
      if (p) cudaFree(p);
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/huru/model.onnx",
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
    k_huru_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.base_lin_vel,
        c.cmd,
        d_last,
        d_obs,
        envs
    );
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_huru_act<<<blocks, threads>>>(
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
  const char* name() const override { return "huru"; }
};

}
