namespace clobot {

constexpr int NUM_ACTIONS = 29;
constexpr int HISTORY = 5;
constexpr int GAIT_DIM = 4;
constexpr int SINGLE_OBS = 3 + 3 + 3 + 29 + 29 + 29 + GAIT_DIM;
constexpr int NUM_OBS = SINGLE_OBS * HISTORY;

constexpr int OWNED_END = 15;
constexpr int WITH_ARMS_OWNED_END = POLICY_NUM_MOTOR;

constexpr float ANG_VEL_SCALE = 0.2f;
constexpr float DOF_VEL_SCALE = 0.05f;
constexpr float ACTION_CLIP = 5.0f;
constexpr float CONTROL_DT = 0.02f;
constexpr float GAIT_PERIOD = 1.0f;

constexpr int OFF_ANG_VEL = 0;
constexpr int OFF_GRAVITY = OFF_ANG_VEL + HISTORY * 3;
constexpr int OFF_CMD = OFF_GRAVITY + HISTORY * 3;
constexpr int OFF_JPOS = OFF_CMD + HISTORY * 3;
constexpr int OFF_JVEL = OFF_JPOS + HISTORY * 29;
constexpr int OFF_ACT = OFF_JVEL + HISTORY * 29;
constexpr int OFF_GAIT = OFF_ACT + HISTORY * 29;

#define CLOBOT_DEFAULT_ISAAC                                                 \
  -0.264f, -0.245f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.49f, 0.476f, \
      0.3f, 0.3f, -0.216f, -0.199f, 0.25f, -0.25f, 0.0f, 0.0f, 0.0f, 0.0f,   \
      0.97f, 0.97f, 0.15f, -0.15f, 0.0f, 0.0f, 0.0f, 0.0f

#define CLOBOT_ACTION_SCALE_ISAAC                                             \
  0.548f, 0.548f, 0.548f, 0.351f, 0.351f, 0.439f, 0.548f, 0.548f, 0.439f,     \
      0.351f, 0.351f, 0.439f, 0.439f, 0.439f, 0.439f, 0.439f, 0.439f, 0.439f, \
      0.439f, 0.439f, 0.439f, 0.439f, 0.439f, 0.439f, 0.439f, 0.0745f,        \
      0.0745f, 0.0745f, 0.0745f

#define CLOBOT_TO_MOTOR                                                     \
  0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, \
      18, 25, 19, 26, 20, 27, 21, 28

__device__ const float D_DEFAULT_ISAAC[NUM_ACTIONS] = {CLOBOT_DEFAULT_ISAAC};
__device__ const float D_ACTION_SCALE_ISAAC[NUM_ACTIONS] = {
    CLOBOT_ACTION_SCALE_ISAAC
};
__device__ const int D_TO_MOTOR[NUM_ACTIONS] = {CLOBOT_TO_MOTOR};

const float KPS[POLICY_NUM_MOTOR] = {40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,
                                     40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f,
                                     40.2f, 28.5f, 28.5f, 14.3f, 14.3f, 14.3f,
                                     14.3f, 14.3f, 16.8f, 16.8f, 14.3f, 14.3f,
                                     14.3f, 14.3f, 14.3f, 16.8f, 16.8f};
const float KDS[POLICY_NUM_MOTOR] = {2.56f,  6.31f,  2.56f,  6.31f,  1.81f,
                                     1.81f,  2.56f,  6.31f,  2.56f,  6.31f,
                                     1.81f,  1.81f,  2.56f,  1.81f,  1.81f,
                                     0.907f, 0.907f, 0.907f, 0.907f, 0.907f,
                                     1.07f,  1.07f,  0.907f, 0.907f, 0.907f,
                                     0.907f, 0.907f, 1.07f,  1.07f};

const policy_api::Limits LIMITS = {-0.5, 1.0, 0.17, 0.7, 0.0};

__global__ void k_clobot_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    float gait_seconds,
    int first,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* o = obs + size_t(env) * NUM_OBS;

  const float phase = fmodf(gait_seconds, GAIT_PERIOD) / GAIT_PERIOD;
  const float s = sinf(2.0f * float(M_PI) * phase);
  const float c = cosf(2.0f * float(M_PI) * phase);

  float frame[SINGLE_OBS];
  int at = 0;
  for (int k = 0; k < 3; ++k) frame[at++] = gyro[env * 3 + k] * ANG_VEL_SCALE;
  for (int k = 0; k < 3; ++k) frame[at++] = gravity[env * 3 + k];
  for (int k = 0; k < 3; ++k) frame[at++] = cmd[env * 3 + k];
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const int m = D_TO_MOTOR[i];
    frame[at++] =
        motor_q[size_t(env) * POLICY_NUM_MOTOR + m] - D_DEFAULT_ISAAC[i];
  }
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const int m = D_TO_MOTOR[i];
    frame[at++] = motor_dq[size_t(env) * POLICY_NUM_MOTOR + m] * DOF_VEL_SCALE;
  }
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    frame[at++] = last_action[size_t(env) * NUM_ACTIONS + i];
  }
  frame[at++] = s;
  frame[at++] = -s;
  frame[at++] = c;
  frame[at++] = -c;

  const int off[7] = {
      OFF_ANG_VEL,
      OFF_GRAVITY,
      OFF_CMD,
      OFF_JPOS,
      OFF_JVEL,
      OFF_ACT,
      OFF_GAIT
  };
  const int wid[7] = {3, 3, 3, 29, 29, 29, GAIT_DIM};
  int src = 0;
  for (int t = 0; t < 7; ++t) {
    float* b = o + off[t];
    const int w = wid[t];
    if (first) {
      for (int h = 0; h < HISTORY; ++h) {
        for (int k = 0; k < w; ++k) b[h * w + k] = frame[src + k];
      }
    } else {
      for (int i = 0; i < (HISTORY - 1) * w; ++i) b[i] = b[i + w];
      for (int k = 0; k < w; ++k) b[(HISTORY - 1) * w + k] = frame[src + k];
    }
    src += w;
  }
}

__global__ void k_clobot_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int owned_end,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + size_t(env) * NUM_ACTIONS;
  float* q = q_target + size_t(env) * POLICY_NUM_MOTOR;
  const float* hold = arm_pose + size_t(env) * POLICY_NUM_MOTOR;

  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) q[j] = hold[j];
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    last_action[size_t(env) * NUM_ACTIONS + i] = a[i];
    const float v = fminf(fmaxf(a[i], -ACTION_CLIP), ACTION_CLIP);
    const int m = D_TO_MOTOR[i];
    if (m < owned_end) q[m] = v * D_ACTION_SCALE_ISAAC[i] + D_DEFAULT_ISAAC[i];
  }
}

struct Base : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  int envs = 0;
  long long ticks = 0;
  int owned_end = OWNED_END;

  explicit Base(int end) : owned_end(end) {}

  ~Base() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/clobot/model.onnx",
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
    const float gait_seconds = float(ticks) * CONTROL_DT;
    k_clobot_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_obs,
        gait_seconds,
        ticks == 0 ? 1 : 0,
        envs
    );
    ++ticks;
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_clobot_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        owned_end,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return owned_end; }
  policy_api::Limits limits() const override { return LIMITS; }
};

struct Policy : Base {
  Policy() : Base(OWNED_END) {}
  const char* name() const override { return "clobot"; }
};

struct WithArmsPolicy : Base {
  WithArmsPolicy() : Base(WITH_ARMS_OWNED_END) {}
  const char* name() const override { return "clobot_with_arms"; }
};

}
