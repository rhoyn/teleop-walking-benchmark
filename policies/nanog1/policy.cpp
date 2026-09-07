namespace nanog1 {

constexpr int NUM_OBS = 98;
constexpr int NUM_ACTIONS = 29;
constexpr int NUM_LEG = 12;
constexpr int WAIST_END = 15;
constexpr int STATE_LAYERS = 3;
constexpr int STATE_WIDTH = 128;
constexpr float WAIST_KP = 75.0f;
constexpr float WAIST_KD = 2.0f;
constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float DOF_VEL_SCALE = 0.05f;
constexpr float ACTION_SCALE = 0.25f;
constexpr float ACTION_CLIP = 1.0f;
constexpr int PHASE_PERIOD = 40;

__device__ const float D_HOME[NUM_ACTIONS] = {
    -0.10f, 0.00f, 0.00f, 0.30f,  -0.20f, 0.00f, -0.10f, 0.00f, 0.00f, 0.30f,
    -0.20f, 0.00f, 0.00f, 0.00f,  0.00f,  0.20f, 0.20f,  0.00f, 1.28f, 0.00f,
    0.00f,  0.00f, 0.20f, -0.20f, 0.00f,  1.28f, 0.00f,  0.00f, 0.00f
};

__device__ const float D_CTRL_LO[NUM_LEG] = {
    -2.5307f,
    -0.5236f,
    -2.7576f,
    -0.087267f,
    -0.87267f,
    -0.2618f,
    -2.5307f,
    -0.5236f,
    -2.7576f,
    -0.087267f,
    -0.87267f,
    -0.2618f
};
__device__ const float D_CTRL_HI[NUM_LEG] = {
    2.8798f,
    2.9671f,
    2.7576f,
    2.8798f,
    0.5236f,
    0.2618f,
    2.8798f,
    2.9671f,
    2.7576f,
    2.8798f,
    0.5236f,
    0.2618f
};

const float KPS[WAIST_END] = {
    100,
    100,
    100,
    150,
    40,
    40,
    100,
    100,
    100,
    150,
    40,
    40,
    WAIST_KP,
    WAIST_KP,
    WAIST_KP
};
const float KDS[WAIST_END] = {
    4.0f,
    4.0f,
    4.0f,
    6.0f,
    3.0f,
    2.2f,
    4.0f,
    4.0f,
    4.0f,
    6.0f,
    3.0f,
    2.2f,
    WAIST_KD,
    WAIST_KD,
    WAIST_KD
};

const policy_api::Limits LIMITS = {-0.5, 0.8, 0.4, 1.0, 0.0};

__global__ void k_nanog1_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float sin_phase,
    float cos_phase,
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
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    o[9 + j] = motor_q[env * POLICY_NUM_MOTOR + j] - D_HOME[j];
    o[38 + j] = motor_dq[env * POLICY_NUM_MOTOR + j] * DOF_VEL_SCALE;
    o[67 + j] = last_action[env * NUM_ACTIONS + j];
  }
  o[96] = sin_phase;
  o[97] = cos_phase;
}

__global__ void k_nanog1_act(
    const float* __restrict__ action,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = action + env * NUM_ACTIONS;
  float* last = last_action + env * NUM_ACTIONS;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  }

  for (int j = NUM_LEG; j < NUM_ACTIONS; ++j) last[j] = 0.0f;
  for (int j = 0; j < NUM_LEG; ++j) {
    const float clipped = fminf(fmaxf(a[j], -ACTION_CLIP), ACTION_CLIP);
    last[j] = clipped;
    const float target = D_HOME[j] + clipped * ACTION_SCALE;
    q_target[env * POLICY_NUM_MOTOR + j] =
        fminf(fmaxf(target, D_CTRL_LO[j]), D_CTRL_HI[j]);
  }
  for (int j = NUM_LEG; j < WAIST_END; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = D_HOME[j];
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;

  float *d_state_in = nullptr, *d_state_out = nullptr;
  int64_t phase_counter = 0;
  int envs = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_obs,
          (void*)d_act,
          (void*)d_last,
          (void*)d_state_in,
          (void*)d_state_out}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;

    engine = policy_api::engine_make(
        "policies/nanog1/model.onnx",
        n,
        {{"obs", {-1, NUM_OBS}}, {"state_in", {STATE_LAYERS, -1, STATE_WIDTH}}},
        {{"action", {-1, NUM_ACTIONS}},
         {"state_out", {STATE_LAYERS, -1, STATE_WIDTH}}}
    );

    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_ACTIONS * sizeof(float));

    const size_t state = size_t(STATE_LAYERS) * n * STATE_WIDTH * sizeof(float);
    for (float** p : {&d_state_in, &d_state_out}) {
      cudaMalloc(p, state);
      cudaMemset(*p, 0, state);
    }
  }

  void step(const policy_api::Ctx& c) override {
    const double phase = double(phase_counter % PHASE_PERIOD) / PHASE_PERIOD;
    ++phase_counter;

    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_nanog1_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        float(std::sin(2.0 * M_PI * phase)),
        float(std::cos(2.0 * M_PI * phase)),
        d_obs,
        envs
    );
    const float* in[2] = {d_obs, d_state_in};
    float* out[2] = {d_act, d_state_out};
    policy_api::engine_run(*engine, in, out, envs);
    std::swap(d_state_in, d_state_out);
    k_nanog1_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return WAIST_END; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "nanog1"; }
};

}
