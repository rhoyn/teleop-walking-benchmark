namespace josabb {

constexpr int NUM_OBS = 159;
constexpr int NUM_ACTIONS = 41;
constexpr int NUM_JOINTS = 53;
constexpr int OWNED = 15;

constexpr int OBS_JOINT_POS = 9;
constexpr int OBS_JOINT_VEL = 62;
constexpr int OBS_ACTIONS = 115;
constexpr int OBS_COMMAND = 156;
constexpr int NUM_HAND = 24;

#define JOSABB_MOTOR_OF_JOINT                                                 \
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,   \
      21, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 22, 23, 24, 25, 26, \
      27, 28, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1

__device__ const int D_MOTOR_OF_JOINT[NUM_JOINTS] = {JOSABB_MOTOR_OF_JOINT};

#define JOSABB_HAND_SLOT                                                      \
  22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 41, 42, 43, 44, 45, 46, 47, \
      48, 49, 50, 51, 52

__device__ const int D_HAND_SLOT[NUM_HAND] = {JOSABB_HAND_SLOT};

__device__ const float D_HAND_POS_MEAN[NUM_HAND] = {
    0.1165f, 0.1147f, 0.1878f, 0.1886f, 0.1914f, 0.1554f, 0.1864f, 0.1512f,
    0.1887f, 0.1517f, 0.1823f, 0.1394f, 0.1156f, 0.1137f, 0.1843f, 0.1853f,
    0.1954f, 0.1558f, 0.1895f, 0.1536f, 0.1945f, 0.1568f, 0.1865f, 0.1426f
};

__device__ const float D_HAND_VEL_MEAN[NUM_HAND] = {
    0.0257f, 0.0257f, 0.0171f, 0.0131f, 0.0296f, 0.0152f, 0.0228f, 0.0129f,
    0.0249f, 0.0128f, 0.0238f, 0.0108f, 0.0137f, 0.0256f, 0.0170f, 0.0132f,
    0.0269f, 0.0129f, 0.0203f, 0.0106f, 0.0257f, 0.0132f, 0.0283f, 0.0114f
};

#define JOSABB_DEFAULT_MOTOR                                               \
  -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f, -0.312f, 0.0f, 0.0f, 0.669f, \
      -0.363f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.0f, 0.6f, 0.0f, 0.0f, \
      0.0f, 0.2f, -0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f

__device__ const float D_DEFAULT_MOTOR[POLICY_NUM_MOTOR] = {
    JOSABB_DEFAULT_MOTOR
};

__device__ const float D_ACTION_SCALE[OWNED] = {
    0.548f,
    0.351f,
    0.548f,
    0.351f,
    0.439f,
    0.439f,
    0.548f,
    0.351f,
    0.548f,
    0.351f,
    0.439f,
    0.439f,
    0.548f,
    0.439f,
    0.439f
};

const float KPS[OWNED] = {
    40.179f,
    99.098f,
    40.179f,
    99.098f,
    28.501f,
    28.501f,
    40.179f,
    99.098f,
    40.179f,
    99.098f,
    28.501f,
    28.501f,
    40.179f,
    28.501f,
    28.501f
};

const float KDS[OWNED] = {
    2.558f,
    6.309f,
    2.558f,
    6.309f,
    1.814f,
    1.814f,
    2.558f,
    6.309f,
    2.558f,
    6.309f,
    1.814f,
    1.814f,
    2.558f,
    1.814f,
    1.814f
};

const policy_api::Limits LIMITS = {-0.9, 0.9, 0.9, 0.5, 0.0};

__global__ void k_josabb_obs(
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
    o[OBS_COMMAND + k] = cmd[env * 3 + k];
  }

  for (int j = 0; j < NUM_JOINTS; ++j) {
    const int m = D_MOTOR_OF_JOINT[j];
    if (m < 0) continue;
    o[OBS_JOINT_POS + j] =
        motor_q[env * POLICY_NUM_MOTOR + m] - D_DEFAULT_MOTOR[m];
    o[OBS_JOINT_VEL + j] = motor_dq[env * POLICY_NUM_MOTOR + m];
  }
  for (int h = 0; h < NUM_HAND; ++h) {
    const int j = D_HAND_SLOT[h];
    o[OBS_JOINT_POS + j] = D_HAND_POS_MEAN[h];
    o[OBS_JOINT_VEL + j] = D_HAND_VEL_MEAN[h];
  }

  for (int i = 0; i < NUM_ACTIONS; ++i)
    o[OBS_ACTIONS + i] = last_action[env * NUM_ACTIONS + i];
}

__global__ void k_josabb_act(
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
    qt[j] = D_DEFAULT_MOTOR[j] + a[j] * D_ACTION_SCALE[j];
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
        "policies/josabb/model.onnx",
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
    k_josabb_obs<<<blocks, threads>>>(
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
    k_josabb_act<<<blocks, threads>>>(
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
  const char* name() const override { return "josabb"; }
};

}
