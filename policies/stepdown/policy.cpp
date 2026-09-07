namespace stepdown {

constexpr int NUM_LEG = 12;
constexpr int NUM_OBS = 47;
constexpr int HIDDEN = 64;
constexpr int WAIST_BEGIN = 12;
constexpr int WAIST_END = 15;
constexpr float WAIST_KP = 300.0f;
constexpr float WAIST_KD = 3.0f;
constexpr double CONTROL_DT = 0.02;
constexpr double GAIT_PERIOD = 0.8;
constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float DOF_VEL_SCALE = 0.05f;
constexpr float ACTION_SCALE = 0.25f;
constexpr float ACTION_CLIP = 1.0f;
constexpr float CMD_SCALE_X = 2.0f, CMD_SCALE_Y = 2.0f, CMD_SCALE_YAW = 0.25f;

#define STEPDOWN_DEFAULTS \
  -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f

__device__ const float D_DEFAULTS[NUM_LEG] = {STEPDOWN_DEFAULTS};

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
const float KDS[WAIST_END] =
    {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, WAIST_KD, WAIST_KD, WAIST_KD};

const policy_api::Limits LIMITS =
    {-0.4, 0.4, 0.3, 1.57, 0.0, 0.08, 0.15, 0.05, 0.10, 1.5, 2.0};

__global__ void k_stepdown_obs(
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
  }
  const float* c = cmd + env * 3;
  o[6] = c[0] * CMD_SCALE_X;
  o[7] = c[1] * CMD_SCALE_Y;
  o[8] = c[2] * CMD_SCALE_YAW;
  for (int j = 0; j < NUM_LEG; ++j) {
    o[9 + j] = motor_q[env * POLICY_NUM_MOTOR + j] - D_DEFAULTS[j];
    o[9 + NUM_LEG + j] = motor_dq[env * POLICY_NUM_MOTOR + j] * DOF_VEL_SCALE;
    o[9 + 2 * NUM_LEG + j] = last_action[env * NUM_LEG + j];
  }
  o[45] = sin_phase;
  o[46] = cos_phase;
}

__global__ void k_stepdown_act(
    const float* __restrict__ action,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = action + env * NUM_LEG;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  }
  for (int j = 0; j < NUM_LEG; ++j) {
    const float clipped = fminf(fmaxf(a[j], -ACTION_CLIP), ACTION_CLIP);
    last_action[env * NUM_LEG + j] = clipped;
    q_target[env * POLICY_NUM_MOTOR + j] =
        D_DEFAULTS[j] + clipped * ACTION_SCALE;
  }
  for (int j = WAIST_BEGIN; j < WAIST_END; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = 0.0f;
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  float *d_h_in = nullptr, *d_h_out = nullptr;
  float *d_c_in = nullptr, *d_c_out = nullptr;
  int64_t phase_counter = 0;
  int envs = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_obs,
          (void*)d_act,
          (void*)d_last,
          (void*)d_h_in,
          (void*)d_h_out,
          (void*)d_c_in,
          (void*)d_c_out}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/stepdown/model.onnx",
        n,
        {{"obs", {-1, NUM_OBS}},
         {"hidden_in", {1, -1, HIDDEN}},
         {"cell_in", {1, -1, HIDDEN}}},
        {{"action", {-1, NUM_LEG}},
         {"hidden_out", {1, -1, HIDDEN}},
         {"cell_out", {1, -1, HIDDEN}}}
    );

    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_LEG * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_LEG * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_LEG * sizeof(float));

    for (float** p : {&d_h_in, &d_h_out, &d_c_in, &d_c_out}) {
      cudaMalloc(p, size_t(n) * HIDDEN * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * HIDDEN * sizeof(float));
    }
  }

  void step(const policy_api::Ctx& c) override {
    const double phase =
        std::fmod(double(phase_counter) * CONTROL_DT, GAIT_PERIOD) /
        GAIT_PERIOD;
    ++phase_counter;

    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_stepdown_obs<<<blocks, threads>>>(
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
    const float* in[3] = {d_obs, d_h_in, d_c_in};
    float* out[3] = {d_act, d_h_out, d_c_out};
    policy_api::engine_run(*engine, in, out, envs);
    std::swap(d_h_in, d_h_out);
    std::swap(d_c_in, d_c_out);
    k_stepdown_act<<<blocks, threads>>>(
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
  const char* name() const override { return "stepdown"; }
};

}
