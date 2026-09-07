namespace gr00t_wbc {

constexpr int SINGLE_OBS = 86;
constexpr int HISTORY = 6;
constexpr int NUM_OBS = SINGLE_OBS * HISTORY;
constexpr int NUM_ACTIONS = 15;
constexpr float ACTION_SCALE = 0.25f;
constexpr float BALANCE_CMD_NORM = 0.05f;
constexpr float HEIGHT_CMD = 0.74f;
constexpr float ANG_VEL_SCALE = 0.5f;
constexpr float DOF_VEL_SCALE = 0.05f;
constexpr float CMD_SCALE_X = 2.0f, CMD_SCALE_Y = 2.0f, CMD_SCALE_YAW = 0.5f;

#define GR00T_WBC_DEFAULTS                                                    \
  -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, \
      0.0f, 0.0f, 0.0f

const float DEFAULTS[NUM_ACTIONS] = {GR00T_WBC_DEFAULTS};
__device__ const float D_DEFAULTS[NUM_ACTIONS] = {GR00T_WBC_DEFAULTS};
const float KPS[NUM_ACTIONS] =
    {150, 150, 150, 200, 40, 40, 150, 150, 150, 200, 40, 40, 250, 250, 250};
const float KDS[NUM_ACTIONS] = {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5};

const policy_api::Limits LIMITS = {-0.5, 0.5, 0.5, 1.0, 0.5};

__global__ void k_gr00t_wbc_obs(
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
  for (int i = 0; i < SINGLE_OBS * (HISTORY - 1); ++i) o[i] = o[i + SINGLE_OBS];

  float* f = o + SINGLE_OBS * (HISTORY - 1);
  const float* c = cmd + env * 3;
  f[0] = c[0] * CMD_SCALE_X;
  f[1] = c[1] * CMD_SCALE_Y;
  f[2] = c[2] * CMD_SCALE_YAW;
  f[3] = HEIGHT_CMD;
  f[4] = 0.0f;
  f[5] = 0.0f;
  f[6] = 0.0f;
  for (int k = 0; k < 3; ++k) f[7 + k] = gyro[env * 3 + k] * ANG_VEL_SCALE;
  for (int k = 0; k < 3; ++k) f[10 + k] = gravity[env * 3 + k];
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    const float def = j < NUM_ACTIONS ? D_DEFAULTS[j] : 0.0f;
    f[13 + j] = motor_q[env * POLICY_NUM_MOTOR + j] - def;
    f[13 + POLICY_NUM_MOTOR + j] =
        motor_dq[env * POLICY_NUM_MOTOR + j] * DOF_VEL_SCALE;
  }
  for (int k = 0; k < NUM_ACTIONS; ++k) {
    f[13 + 2 * POLICY_NUM_MOTOR + k] = last_action[env * NUM_ACTIONS + k];
  }
}

__global__ void k_gr00t_wbc_act(
    const float* __restrict__ act_walk,
    const float* __restrict__ act_balance,
    const float* __restrict__ cmd,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* c = cmd + env * 3;
  const float norm = sqrtf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
  const float* a =
      (norm > BALANCE_CMD_NORM ? act_walk : act_balance) + env * NUM_ACTIONS;

  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  }
  for (int k = 0; k < NUM_ACTIONS; ++k) {
    last_action[env * NUM_ACTIONS + k] = a[k];
    q_target[env * POLICY_NUM_MOTOR + k] = D_DEFAULTS[k] + a[k] * ACTION_SCALE;
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> walk, balance;
  float *d_obs = nullptr, *d_walk = nullptr, *d_balance = nullptr;
  float *d_last = nullptr, *d_defaults = nullptr;
  int envs = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_obs,
          (void*)d_walk,
          (void*)d_balance,
          (void*)d_last,
          (void*)d_defaults}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    walk = policy_api::engine_make(
        "policies/gr00t_wbc/model_walk.onnx",
        n,
        NUM_OBS,
        NUM_ACTIONS
    );
    balance = policy_api::engine_make(
        "policies/gr00t_wbc/model_balance.onnx",
        n,
        NUM_OBS,
        NUM_ACTIONS
    );
    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_walk, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_balance, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_defaults, NUM_ACTIONS * sizeof(float));
    cudaMemcpy(d_defaults, DEFAULTS, sizeof(DEFAULTS), cudaMemcpyHostToDevice);
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_gr00t_wbc_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_obs,
        envs
    );
    policy_api::engine_run(*walk, d_obs, d_walk, envs);
    policy_api::engine_run(*balance, d_obs, d_balance, envs);
    k_gr00t_wbc_act<<<blocks, threads>>>(
        d_walk,
        d_balance,
        c.cmd,
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
  const char* name() const override { return "gr00t_wbc"; }
};

}
