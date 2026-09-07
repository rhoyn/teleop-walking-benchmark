namespace wty_cpp {

constexpr int NUM_ACTIONS = 29;
constexpr int HISTORY = 5;
constexpr int FRAME = 3 + 3 + 3 + 3 * NUM_ACTIONS;
constexpr int NUM_OBS = HISTORY * FRAME;
constexpr float ACTION_SCALE = 0.25f;
constexpr float ACTION_GUARD = 10.0f;
constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float JOINT_VEL_SCALE = 0.05f;
constexpr float CMD_SCALE_X = 2.0f, CMD_SCALE_Y = 2.0f, CMD_SCALE_YAW = 0.25f;

constexpr int FIRST_ARM_MOTOR = 15;

__device__ const float D_DEFAULTS[NUM_ACTIONS] = {
    -0.1f, 0.0f, 0.0f, 0.3f,   -0.2f, 0.0f,  -0.1f,  0.0f, 0.0f,  0.3f,
    -0.2f, 0.0f, 0.0f, 0.0f,   0.0f,  0.0f,  0.25f,  0.0f, 0.97f, 0.15f,
    0.0f,  0.0f, 0.0f, -0.25f, 0.0f,  0.97f, -0.15f, 0.0f, 0.0f
};

const float KPS[NUM_ACTIONS] = {
    40.1792f, 99.0984f, 40.1792f, 99.0984f, 28.5012f, 28.5012f,
    40.1792f, 99.0984f, 40.1792f, 99.0984f, 28.5012f, 28.5012f,
    40.1792f, 28.5012f, 28.5012f, 50.0f,    50.0f,    50.0f,
    14.2506f, 14.2506f, 16.7783f, 16.7783f, 50.0f,    50.0f,
    50.0f,    14.2506f, 14.2506f, 16.7783f, 16.7783f
};
const float KDS[NUM_ACTIONS] = {2.5579f, 6.3088f, 2.5579f, 6.3088f, 1.8144f,
                                1.8144f, 2.5579f, 6.3088f, 2.5579f, 6.3088f,
                                1.8144f, 1.8144f, 2.5579f, 1.8144f, 1.8144f,
                                0.9072f, 0.9072f, 0.9072f, 0.9072f, 0.9072f,
                                1.0681f, 1.0681f, 0.9072f, 0.9072f, 0.9072f,
                                0.9072f, 0.9072f, 1.0681f, 1.0681f};

const policy_api::Limits LIMITS = {-0.5, 1.0, 0.4, 1.0, 0.0};

__global__ void k_wty_cpp_obs(
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

  float frame[FRAME];
  const float* c = cmd + env * 3;
  frame[0] = gyro[env * 3 + 0] * ANG_VEL_SCALE;
  frame[1] = gyro[env * 3 + 1] * ANG_VEL_SCALE;
  frame[2] = gyro[env * 3 + 2] * ANG_VEL_SCALE;
  for (int k = 0; k < 3; ++k) frame[3 + k] = gravity[env * 3 + k];
  frame[6] = c[0] * CMD_SCALE_X;
  frame[7] = c[1] * CMD_SCALE_Y;
  frame[8] = c[2] * CMD_SCALE_YAW;

  const float* la = last_action + env * NUM_ACTIONS;
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    if (j >= FIRST_ARM_MOTOR) {
      frame[9 + j] = la[j] * ACTION_SCALE;
      frame[9 + NUM_ACTIONS + j] = 0.0f;
    } else {
      frame[9 + j] = motor_q[env * POLICY_NUM_MOTOR + j] - D_DEFAULTS[j];
      frame[9 + NUM_ACTIONS + j] =
          motor_dq[env * POLICY_NUM_MOTOR + j] * JOINT_VEL_SCALE;
    }
    frame[9 + 2 * NUM_ACTIONS + j] = la[j];
  }

  float* o = obs + env * NUM_OBS;
  if (primed) {
    for (int i = 0; i < FRAME * (HISTORY - 1); ++i) o[i] = o[i + FRAME];
  } else {
    for (int h = 0; h < HISTORY - 1; ++h) {
      for (int i = 0; i < FRAME; ++i) o[h * FRAME + i] = frame[i];
    }
  }
  for (int i = 0; i < FRAME; ++i) o[(HISTORY - 1) * FRAME + i] = frame[i];
}

__global__ void k_wty_cpp_act(
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
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    const float v = a[j];
    la[j] = isfinite(v) ? fminf(fmaxf(v, -ACTION_GUARD), ACTION_GUARD) : 0.0f;
    if (j < FIRST_ARM_MOTOR) q[j] = D_DEFAULTS[j] + la[j] * ACTION_SCALE;
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
        "policies/wty_cpp/model.onnx",
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
    k_wty_cpp_obs<<<blocks, threads>>>(
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
    k_wty_cpp_act<<<blocks, threads>>>(
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
  const char* name() const override { return "wty_cpp"; }
};

}
