namespace wcompton {

constexpr int NUM_OBS = 74;
constexpr int NUM_ACTIONS = 21;
constexpr int OFF_ANG_VEL = 0;
constexpr int OFF_GRAVITY = 3;
constexpr int OFF_COMMAND = 6;
constexpr int OFF_PHASE = 9;
constexpr int OFF_JOINT_POS = 11;
constexpr int OFF_JOINT_VEL = 32;
constexpr int OFF_LAST_ACTION = 53;
constexpr int FIRST_ARM_MOTOR = 15;

constexpr int OWNED_MOTORS = 13;
constexpr float ACTION_SCALE = 0.5f;
constexpr float CONTROL_DT = 0.02f;
constexpr float GAIT_PERIOD = 0.8f;
constexpr float STAND_THRESHOLD = 0.1f;

#define WC_ISAAC_TO_MUJOCO \
  0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, 18, 25, 5, 11

#define WC_DEFAULT_ISAAC                                                       \
  -0.20f, -0.20f, 0.00f, 0.00f, 0.00f, 0.35f, 0.35f, 0.00f, 0.00f, 0.27f,      \
      -0.27f, 0.42f, 0.42f, 0.00f, 0.00f, -0.23f, -0.23f, 0.87f, 0.87f, 0.00f, \
      0.00f

const int ISAAC_TO_MUJOCO[NUM_ACTIONS] = {WC_ISAAC_TO_MUJOCO};
__device__ const int D_ISAAC_TO_MUJOCO[NUM_ACTIONS] = {WC_ISAAC_TO_MUJOCO};
__device__ const float D_DEFAULT_ISAAC[NUM_ACTIONS] = {WC_DEFAULT_ISAAC};

const float KPS_ISAAC[NUM_ACTIONS] = {
    40.179238f, 40.179238f, 28.501246f, 99.098428f, 99.098428f, 14.250623f,
    14.250623f, 40.179238f, 40.179238f, 14.250623f, 14.250623f, 99.098428f,
    99.098428f, 14.250623f, 14.250623f, 28.501246f, 28.501246f, 14.250623f,
    14.250623f, 28.501246f, 28.501246f
};

const float KDS_ISAAC[NUM_ACTIONS] = {
    2.557890f, 2.557890f, 1.814446f, 6.308802f, 6.308802f, 0.907223f,
    0.907223f, 2.557890f, 2.557890f, 0.907223f, 0.907223f, 6.308802f,
    6.308802f, 0.907223f, 0.907223f, 1.814446f, 1.814446f, 0.907223f,
    0.907223f, 1.814446f, 1.814446f
};

const std::array<float, POLICY_NUM_MOTOR> KPS = [] {
  std::array<float, POLICY_NUM_MOTOR> g{};
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    if (ISAAC_TO_MUJOCO[i] < FIRST_ARM_MOTOR)
      g[ISAAC_TO_MUJOCO[i]] = KPS_ISAAC[i];
  }
  return g;
}();
const std::array<float, POLICY_NUM_MOTOR> KDS = [] {
  std::array<float, POLICY_NUM_MOTOR> g{};
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    if (ISAAC_TO_MUJOCO[i] < FIRST_ARM_MOTOR)
      g[ISAAC_TO_MUJOCO[i]] = KDS_ISAAC[i];
  }
  return g;
}();

const policy_api::Limits LIMITS = {-0.5, 1.0, 0.25, 1.0, 0.0};

__global__ void k_wcompton_obs(
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
    o[OFF_ANG_VEL + k] = gyro[env * 3 + k];
    o[OFF_GRAVITY + k] = gravity[env * 3 + k];
    o[OFF_COMMAND + k] = c[k];
  }

  const float norm = sqrtf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
  const bool gait_on = norm >= STAND_THRESHOLD;
  const float angle = phase * 2.0f * float(M_PI);
  o[OFF_PHASE + 0] = gait_on ? sinf(angle) : 0.0f;
  o[OFF_PHASE + 1] = gait_on ? cosf(angle) : 0.0f;

  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const int m = D_ISAAC_TO_MUJOCO[i];
    o[OFF_JOINT_POS + i] =
        motor_q[env * POLICY_NUM_MOTOR + m] - D_DEFAULT_ISAAC[i];
    o[OFF_JOINT_VEL + i] = motor_dq[env * POLICY_NUM_MOTOR + m];
    o[OFF_LAST_ACTION + i] = last_action[env * NUM_ACTIONS + i];
  }
}

__global__ void k_wcompton_act(
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
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    last_action[env * NUM_ACTIONS + i] = a[i];
    const int m = D_ISAAC_TO_MUJOCO[i];
    if (m >= FIRST_ARM_MOTOR) continue;
    q_target[env * POLICY_NUM_MOTOR + m] =
        D_DEFAULT_ISAAC[i] + a[i] * ACTION_SCALE;
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
        "policies/wcompton/model.onnx",
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
    k_wcompton_obs<<<blocks, threads>>>(
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
    k_wcompton_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS.data(); }
  const float* kd() const override { return KDS.data(); }
  int owned() const override { return OWNED_MOTORS; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "wcompton"; }
};

}
