namespace sunny {

constexpr int NUM_OBS = 99;
constexpr int NUM_ACTIONS = 29;
constexpr int OWNED = 15;
constexpr float ACTION_SCALE = 0.5f;

#define SUNNY_MOTOR_OF_ISAAC                                                \
  0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, \
      18, 25, 19, 26, 20, 27, 21, 28

#define SUNNY_ISAAC_OF_MOTOR                                                \
  0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, \
      27, 12, 16, 20, 22, 24, 26, 28

__device__ const int D_MOTOR_OF_ISAAC[NUM_ACTIONS] = {SUNNY_MOTOR_OF_ISAAC};
__device__ const int D_ISAAC_OF_MOTOR[NUM_ACTIONS] = {SUNNY_ISAAC_OF_MOTOR};

#define SUNNY_DEFAULT_POS                                                     \
  -0.20f, 0.0f, 0.0f, 0.42f, -0.23f, 0.0f, -0.20f, 0.0f, 0.0f, 0.42f, -0.23f, \
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.16f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f,     \
      0.0f, -0.16f, 0.0f, 0.87f, 0.0f, 0.0f, 0.0f

__device__ const float D_DEFAULT_POS[NUM_ACTIONS] = {SUNNY_DEFAULT_POS};

const float KPS[OWNED] =
    {100, 100, 100, 200, 40, 40, 100, 100, 100, 200, 40, 40, 200, 200, 200};
const float KDS[OWNED] =
    {2.5f, 2.5f, 2.5f, 5, 2, 2, 2.5f, 2.5f, 2.5f, 5, 2, 2, 5, 5, 5};

const policy_api::Limits LIMITS = {-0.7, 0.7, 0.35, 0.5, 0.0};

__global__ void k_sunny_obs(
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
    o[9 + k] = cmd[env * 3 + k];
  }
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const int m = D_MOTOR_OF_ISAAC[i];
    o[12 + i] = motor_q[env * POLICY_NUM_MOTOR + m] - D_DEFAULT_POS[m];
    o[41 + i] = motor_dq[env * POLICY_NUM_MOTOR + m];
    o[70 + i] = last_action[env * NUM_ACTIONS + i];
  }
}

__global__ void k_sunny_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + env * NUM_ACTIONS;
  float* qt = q_target + env * POLICY_NUM_MOTOR;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    qt[j] = arm_pose[env * POLICY_NUM_MOTOR + j];

  for (int i = 0; i < NUM_ACTIONS; ++i)
    last_action[env * NUM_ACTIONS + i] = a[i];
  for (int j = 0; j < OWNED; ++j)
    qt[j] = D_DEFAULT_POS[j] + ACTION_SCALE * a[D_ISAAC_OF_MOTOR[j]];
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
        "policies/sunny/model.onnx",
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
    k_sunny_obs<<<blocks, threads>>>(
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
    k_sunny_act<<<blocks, threads>>>(
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
  const char* name() const override { return "sunny"; }
};

}
