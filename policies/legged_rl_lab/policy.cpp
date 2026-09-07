namespace legged_rl_lab {

constexpr int NUM_OBS = 96;
constexpr int NUM_ACTIONS = 29;
constexpr int FIRST_ARM_MOTOR = 15;
constexpr float ANG_VEL_SCALE = 0.2f;
constexpr float DOF_POS_SCALE = 1.0f;
constexpr float DOF_VEL_SCALE = 0.05f;
constexpr float ACTION_SCALE = 0.25f;
constexpr float RAMP_S = 0.8f;
constexpr float CONTROL_DT = 0.02f;

#define LRL_MOTOR_OF_ISAAC                                                  \
  0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, \
      18, 25, 19, 26, 20, 27, 21, 28

#define LRL_ISAAC_OF_MOTOR                                                  \
  0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, \
      27, 12, 16, 20, 22, 24, 26, 28

#define LRL_DEFAULT_POS                                                        \
  -0.1f, -0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.3f, 0.3f,    \
      0.3f, -0.2f, -0.2f, 0.25f, -0.25f, 0.0f, 0.0f, 0.0f, 0.0f, 0.97f, 0.97f, \
      0.15f, -0.15f, 0.0f, 0.0f, 0.0f, 0.0f

const int MOTOR_OF_ISAAC[NUM_ACTIONS] = {LRL_MOTOR_OF_ISAAC};
__device__ const int D_MOTOR_OF_ISAAC[NUM_ACTIONS] = {LRL_MOTOR_OF_ISAAC};
const int ISAAC_OF_MOTOR[NUM_ACTIONS] = {LRL_ISAAC_OF_MOTOR};
__device__ const int D_ISAAC_OF_MOTOR[NUM_ACTIONS] = {LRL_ISAAC_OF_MOTOR};
__device__ const float D_DEFAULT_POS[NUM_ACTIONS] = {LRL_DEFAULT_POS};

const float KPS_ISAAC[NUM_ACTIONS] = {100.0f, 100.0f, 200.0f, 100.0f, 100.0f,
                                      40.0f,  100.0f, 100.0f, 40.0f,  150.0f,
                                      150.0f, 40.0f,  40.0f,  40.0f,  40.0f,
                                      40.0f,  40.0f,  40.0f,  40.0f,  40.0f,
                                      40.0f,  40.0f,  40.0f,  40.0f,  40.0f,
                                      40.0f,  40.0f,  40.0f,  40.0f};

const float KDS_ISAAC[NUM_ACTIONS] = {2.0f, 2.0f, 5.0f, 2.0f, 2.0f, 5.0f,
                                      2.0f, 2.0f, 5.0f, 4.0f, 4.0f, 1.0f,
                                      1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f,
                                      2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
                                      1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

const std::array<float, NUM_ACTIONS> KPS = [] {
  std::array<float, NUM_ACTIONS> g{};
  for (int m = 0; m < NUM_ACTIONS; ++m) g[m] = KPS_ISAAC[ISAAC_OF_MOTOR[m]];
  return g;
}();
const std::array<float, NUM_ACTIONS> KDS = [] {
  std::array<float, NUM_ACTIONS> g{};
  for (int m = 0; m < NUM_ACTIONS; ++m) g[m] = KDS_ISAAC[ISAAC_OF_MOTOR[m]];
  return g;
}();

const policy_api::Limits LIMITS = {0.0, 1.0, 0.5, 1.0, 0.0};

__global__ void k_legged_rl_lab_obs(
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
  for (int k = 0; k < 3; ++k) {
    o[k] = gyro[env * 3 + k] * ANG_VEL_SCALE;
    o[3 + k] = gravity[env * 3 + k];
    o[6 + k] = cmd[env * 3 + k];
  }
  for (int a = 0; a < NUM_ACTIONS; ++a) {
    const int m = D_MOTOR_OF_ISAAC[a];
    o[9 + a] = (motor_q[env * POLICY_NUM_MOTOR + m] - D_DEFAULT_POS[a]) *
               DOF_POS_SCALE;
    o[9 + NUM_ACTIONS + a] =
        motor_dq[env * POLICY_NUM_MOTOR + m] * DOF_VEL_SCALE;
    o[9 + 2 * NUM_ACTIONS + a] = last_action[env * NUM_ACTIONS + a];
  }
}

__global__ void k_legged_rl_lab_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    float ramp,
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
  for (int m = 0; m < FIRST_ARM_MOTOR; ++m) {
    const int i = D_ISAAC_OF_MOTOR[m];
    q_target[env * POLICY_NUM_MOTOR + m] =
        D_DEFAULT_POS[i] + ramp * (a[i] * ACTION_SCALE);
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  double control_time = 0.0;
  int envs = 0;

  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/legged_rl_lab/model.onnx",
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
    const float ramp = float(std::clamp(control_time / RAMP_S, 0.0, 1.0));
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_legged_rl_lab_obs<<<blocks, threads>>>(
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
    k_legged_rl_lab_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        ramp,
        envs
    );
    control_time += CONTROL_DT;
  }

  const float* kp() const override { return KPS.data(); }
  const float* kd() const override { return KDS.data(); }
  int owned() const override { return FIRST_ARM_MOTOR; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "legged_rl_lab"; }
};

}
