namespace schoi {

constexpr int NUM_OBS = 47;
constexpr int NUM_ACTIONS = 12;
constexpr float ANG_VEL_SCALE = 0.5f;
constexpr float DOF_POS_SCALE = 1.0f;
constexpr float DOF_VEL_SCALE = 0.05f;
constexpr float ACTION_SCALE = 0.25f;
constexpr float CMD_SCALE_X = 1.0f, CMD_SCALE_Y = 1.0f, CMD_SCALE_YAW = 1.0f;
constexpr float GAIT_PERIOD = 0.8f;
constexpr float GAIT_CMD_DEADBAND = 0.1f;
constexpr float CONTROL_DT = 0.02f;

__device__ const int D_MUJOCO_FROM_ISAAC[NUM_ACTIONS] =
    {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};

__device__ const float D_DEFAULT_POS[NUM_ACTIONS] = {
    -0.10f,
    0.0f,
    0.0f,
    0.3f,
    -0.2f,
    0.0f,
    -0.10f,
    0.0f,
    0.0f,
    0.3f,
    -0.2f,
    0.0f
};

const float KPS[NUM_ACTIONS] = {
    100.0f,
    100.0f,
    100.0f,
    150.0f,
    40.0f,
    40.0f,
    100.0f,
    100.0f,
    100.0f,
    150.0f,
    40.0f,
    40.0f
};
const float KDS[NUM_ACTIONS] =
    {3.0f, 3.0f, 3.0f, 5.0f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f, 5.0f, 3.0f, 3.0f};

const policy_api::Limits LIMITS = {-1.0, 1.0, 0.5, 0.5, 0.0};

__global__ void k_schoi_obs(
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
  o[0] = gyro[env * 3 + 0] * ANG_VEL_SCALE;
  o[1] = gyro[env * 3 + 1] * ANG_VEL_SCALE;
  o[2] = gyro[env * 3 + 2] * ANG_VEL_SCALE;
  for (int k = 0; k < 3; ++k) o[3 + k] = gravity[env * 3 + k];
  o[6] = c[0] * CMD_SCALE_X;
  o[7] = c[1] * CMD_SCALE_Y;
  o[8] = c[2] * CMD_SCALE_YAW;

  for (int k = 0; k < NUM_ACTIONS; ++k) {
    const int j = D_MUJOCO_FROM_ISAAC[k];
    o[9 + k] = (motor_q[env * POLICY_NUM_MOTOR + j] - D_DEFAULT_POS[j]) *
               DOF_POS_SCALE;
    o[9 + NUM_ACTIONS + k] =
        motor_dq[env * POLICY_NUM_MOTOR + j] * DOF_VEL_SCALE;
    o[9 + 2 * NUM_ACTIONS + k] = last_action[env * NUM_ACTIONS + k];
  }

  const float norm = sqrtf(o[6] * o[6] + o[7] * o[7] + o[8] * o[8]);
  const bool gait_on = norm >= GAIT_CMD_DEADBAND;
  o[45] = gait_on ? sinf(phase * 2.0f * float(M_PI)) : 0.0f;
  o[46] = gait_on ? cosf(phase * 2.0f * float(M_PI)) : 0.0f;
}

__global__ void k_schoi_act(
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
  for (int k = 0; k < NUM_ACTIONS; ++k) {
    last_action[env * NUM_ACTIONS + k] = a[k];
    const int j = D_MUJOCO_FROM_ISAAC[k];
    q_target[env * POLICY_NUM_MOTOR + j] =
        D_DEFAULT_POS[j] + a[k] * ACTION_SCALE;
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
        "policies/schoi/model.onnx",
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
    k_schoi_obs<<<blocks, threads>>>(
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
    k_schoi_act<<<blocks, threads>>>(
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
  const char* name() const override { return "schoi"; }
};

}
