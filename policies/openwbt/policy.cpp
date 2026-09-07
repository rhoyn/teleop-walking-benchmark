namespace openwbt {

constexpr int NUM_ACTIONS = 12;
constexpr int NUM_OBS = 47;
constexpr int HIDDEN_DIM = 256;
constexpr int FIRST_WAIST = 12;
constexpr int FIRST_ARM = 15;
constexpr float WAIST_KP = 300.0f;
constexpr float WAIST_KD = 3.0f;
constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float DOF_VEL_SCALE = 0.05f;
constexpr float ACTION_SCALE = 0.25f;
constexpr float CMD_SCALE_X = 2.0f, CMD_SCALE_Y = 2.0f, CMD_SCALE_YAW = 0.25f;
constexpr float CLIP_OBS = 100.0f;
constexpr float CLIP_ACTIONS = 100.0f;

constexpr double CONTROL_DT = 0.02;
constexpr double GAIT_FREQ = 1.5;
constexpr double GAIT_PHASE_OFFSET = 0.5;
constexpr double GAIT_STANCE_RATIO = 0.6;
constexpr double GAIT_STANCE_MIDDLE = 0.3;

#define OPENWBT_DEFAULTS \
  -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f

__device__ const float D_DEFAULTS[NUM_ACTIONS] = {OPENWBT_DEFAULTS};

const float KPS[FIRST_ARM] = {
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
const float KDS[FIRST_ARM] =
    {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, WAIST_KD, WAIST_KD, WAIST_KD};

const policy_api::Limits LIMITS =
    {-0.3, 0.3, 0.3, 0.3, 0.0, 0.05, 0.10, 0.05, 0.10, 2.0, 2.0};

__device__ inline float clip_obs(float v) {
  return fminf(fmaxf(v, -CLIP_OBS), CLIP_OBS);
}

__global__ void k_openwbt_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    double* __restrict__ gait_index,
    float* __restrict__ obs,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* c = cmd + env * 3;

  const bool stance = c[0] == 0.0f && c[1] == 0.0f && c[2] == 0.0f;

  double g = fmod(gait_index[env] + CONTROL_DT * GAIT_FREQ, 1.0);
  double foot[2] = {fmod(g + GAIT_PHASE_OFFSET, 1.0), g};
  if (stance) {
    g = GAIT_STANCE_MIDDLE;
    foot[0] = GAIT_STANCE_MIDDLE;
    foot[1] = GAIT_STANCE_MIDDLE;
  }
  gait_index[env] = g;
  double clock[2];
  for (int i = 0; i < 2; ++i) {
    const double idx = foot[i];
    const double scaled =
        idx < GAIT_STANCE_RATIO
            ? 0.5 * idx / GAIT_STANCE_RATIO
            : 0.5 + 0.5 * (idx - GAIT_STANCE_RATIO) / (1.0 - GAIT_STANCE_RATIO);
    clock[i] = sin(2.0 * M_PI * scaled);
  }

  float* o = obs + env * NUM_OBS;
  o[0] = clip_obs(c[0] * CMD_SCALE_X);
  o[1] = clip_obs(c[1] * CMD_SCALE_Y);
  o[2] = clip_obs(c[2] * CMD_SCALE_YAW);
  for (int k = 0; k < 3; ++k) {
    o[3 + k] = clip_obs(gravity[env * 3 + k]);
    o[6 + k] = clip_obs(gyro[env * 3 + k] * ANG_VEL_SCALE);
  }
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    o[9 + j] = clip_obs(motor_q[env * POLICY_NUM_MOTOR + j] - D_DEFAULTS[j]);
    o[21 + j] = clip_obs(motor_dq[env * POLICY_NUM_MOTOR + j] * DOF_VEL_SCALE);
    o[33 + j] = clip_obs(last_action[env * NUM_ACTIONS + j]);
  }
  o[45] = clip_obs(float(clock[0]));
  o[46] = clip_obs(float(clock[1]));
}

__global__ void k_openwbt_act(
    const float* __restrict__ action,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = action + env * NUM_ACTIONS;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  }
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    const float clipped = fminf(fmaxf(a[j], -CLIP_ACTIONS), CLIP_ACTIONS);
    last_action[env * NUM_ACTIONS + j] = clipped;
    q_target[env * POLICY_NUM_MOTOR + j] =
        D_DEFAULTS[j] + clipped * ACTION_SCALE;
  }
  for (int j = FIRST_WAIST; j < FIRST_ARM; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = 0.0f;
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;

  float *d_h_in = nullptr, *d_h_out = nullptr;
  double* d_gait = nullptr;
  int envs = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_obs,
          (void*)d_act,
          (void*)d_last,
          (void*)d_h_in,
          (void*)d_h_out,
          (void*)d_gait}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/openwbt/model.onnx",
        n,
        {{"obs", {-1, NUM_OBS}}, {"input_hidden_states", {1, -1, HIDDEN_DIM}}},
        {{"action", {-1, NUM_ACTIONS}},
         {"output_hidden_states", {1, -1, HIDDEN_DIM}}}
    );

    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_ACTIONS * sizeof(float));

    for (float** p : {&d_h_in, &d_h_out}) {
      cudaMalloc(p, size_t(n) * HIDDEN_DIM * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * HIDDEN_DIM * sizeof(float));
    }

    cudaMalloc(&d_gait, size_t(n) * sizeof(double));
    std::vector<double> gait(size_t(n), GAIT_STANCE_MIDDLE);
    cudaMemcpy(
        d_gait,
        gait.data(),
        gait.size() * sizeof(double),
        cudaMemcpyHostToDevice
    );
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_openwbt_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_gait,
        d_obs,
        envs
    );
    const float* in[2] = {d_obs, d_h_in};
    float* out[2] = {d_act, d_h_out};
    policy_api::engine_run(*engine, in, out, envs);
    std::swap(d_h_in, d_h_out);
    k_openwbt_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return FIRST_ARM; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "openwbt"; }
};

}
