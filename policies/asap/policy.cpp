namespace asap {

constexpr int NUM_ACTIONS = 12;
constexpr int HISTORY_LEN = 4;
constexpr int FRAME = 100;
constexpr int NUM_OBS = FRAME + HISTORY_LEN * FRAME;
constexpr int NUM_OWNED = 15;
constexpr int NUM_UPPER = 17;
constexpr int FIRST_UPPER = 12;

constexpr float ACTION_SCALE = 0.25f;
constexpr float GAIT_PERIOD = 0.9f;
constexpr float DT = 0.02f;
constexpr float BASE_HEIGHT_CMD = 0.78f;
constexpr float BASE_HEIGHT_SCALE = 2.0f;
constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float DOF_VEL_SCALE = 0.05f;

constexpr float WALK_ENTER_POS_M = 0.08f;
constexpr float WALK_ENTER_YAW_RAD = 0.10f;
constexpr float WALK_EXIT_POS_M = 0.04f;
constexpr float WALK_EXIT_YAW_RAD = 0.05f;

constexpr int OFF_ACTIONS = 0;
constexpr int OFF_ANG_VEL = 12;
constexpr int OFF_CMD_YAW = 15;
constexpr int OFF_CMD_HEIGHT = 16;
constexpr int OFF_CMD_LIN = 17;
constexpr int OFF_CMD_STAND = 19;
constexpr int OFF_COS_PHASE = 20;
constexpr int OFF_DOF_POS = 21;
constexpr int OFF_DOF_VEL = 50;
constexpr int OFF_GRAVITY = 79;
constexpr int OFF_REF_UPPER = 82;
constexpr int OFF_SIN_PHASE = 99;
constexpr int HEAD = OFF_GRAVITY;

__device__ const int D_SEGMENTS[12][2] = {
    {0, 12},
    {12, 15},
    {15, 16},
    {16, 17},
    {17, 19},
    {19, 20},
    {20, 21},
    {21, 50},
    {50, 79},
    {79, 82},
    {82, 99},
    {99, 100}
};

#define ASAP_DEFAULTS                                                         \
  -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, \
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, \
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f

#define ASAP_Q_MIN                                                             \
  -2.5307f, -0.5236f, -2.7576f, -0.087267f, -0.87267f, -0.2618f, -2.5307f,     \
      -2.9671f, -2.7576f, -0.087267f, -0.87267f, -0.2618f, -2.618f, -0.52f,    \
      -0.52f, -3.0892f, -1.5882f, -2.618f, -1.0472f, -1.972222054f, -1.61443f, \
      -1.61443f, -3.0892f, -2.2515f, -2.618f, -1.0472f, -1.972222054f,         \
      -1.61443f, -1.61443f

#define ASAP_Q_MAX                                                         \
  2.8798f, 2.9671f, 2.7576f, 2.8798f, 0.5236f, 0.2618f, 2.8798f, 0.5236f,  \
      2.7576f, 2.8798f, 0.5236f, 0.2618f, 2.618f, 0.52f, 0.52f, 2.6704f,   \
      2.2515f, 2.618f, 2.0944f, 1.972222054f, 1.61443f, 1.61443f, 2.6704f, \
      1.5882f, 2.618f, 2.0944f, 1.972222054f, 1.61443f, 1.61443f

__device__ const float D_DEFAULTS[POLICY_NUM_MOTOR] = {ASAP_DEFAULTS};
__device__ const float D_Q_MIN[POLICY_NUM_MOTOR] = {ASAP_Q_MIN};
__device__ const float D_Q_MAX[POLICY_NUM_MOTOR] = {ASAP_Q_MAX};

const float KPS[NUM_OWNED] =
    {100, 100, 100, 200, 20, 20, 100, 100, 100, 200, 20, 20, 400, 400, 400};
const float KDS[NUM_OWNED] =
    {2.5f, 2.5f, 2.5f, 5, 0.2f, 0.1f, 2.5f, 2.5f, 2.5f, 5, 0.2f, 0.1f, 5, 5, 5};

const policy_api::Limits LIMITS = {-1.0, 1.0, 0.8, 0.8, 0.0};

__global__ void k_asap_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ task,
    const float* __restrict__ arm_pose,
    const float* __restrict__ last_action,
    unsigned char* __restrict__ walk_latch,
    float* __restrict__ history,
    float* __restrict__ obs,
    float control_time,
    int first,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* c = cmd + env * 3;

  const float dist = task[env * 4 + 0];
  const float yaw_err = fabsf(task[env * 4 + 1]);
  if (walk_latch[env] != 0) {
    if (dist < WALK_EXIT_POS_M && yaw_err < WALK_EXIT_YAW_RAD) {
      walk_latch[env] = 0;
    }
  } else if (dist > WALK_ENTER_POS_M || yaw_err > WALK_ENTER_YAW_RAD) {
    walk_latch[env] = 1;
  }
  const bool walking = walk_latch[env] != 0;
  const float stand = walking ? 1.0f : 0.0f;
  const float phase =
      walking ? fmodf(control_time, GAIT_PERIOD) / GAIT_PERIOD : 0.0f;

  float f[FRAME];
  for (int i = 0; i < FRAME; ++i) f[i] = 0.0f;
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    f[OFF_ACTIONS + i] = last_action[size_t(env) * NUM_ACTIONS + i];
  }
  for (int k = 0; k < 3; ++k) {
    f[OFF_ANG_VEL + k] = gyro[env * 3 + k] * ANG_VEL_SCALE;
    f[OFF_GRAVITY + k] = gravity[env * 3 + k];
  }
  f[OFF_CMD_YAW] = walking ? c[2] : 0.0f;
  f[OFF_CMD_HEIGHT] = BASE_HEIGHT_CMD * BASE_HEIGHT_SCALE;
  f[OFF_CMD_LIN + 0] = walking ? c[0] : 0.0f;
  f[OFF_CMD_LIN + 1] = walking ? c[1] : 0.0f;
  f[OFF_CMD_STAND] = stand;
  f[OFF_COS_PHASE] = cosf(2.0f * float(M_PI) * phase);
  f[OFF_SIN_PHASE] = sinf(2.0f * float(M_PI) * phase);
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    f[OFF_DOF_POS + j] =
        motor_q[size_t(env) * POLICY_NUM_MOTOR + j] - D_DEFAULTS[j];
    f[OFF_DOF_VEL + j] =
        motor_dq[size_t(env) * POLICY_NUM_MOTOR + j] * DOF_VEL_SCALE;
  }
  for (int i = 0; i < NUM_UPPER; ++i) {
    const int motor = FIRST_UPPER + i;

    const float target = motor < NUM_OWNED
                             ? D_DEFAULTS[motor]
                             : arm_pose[size_t(env) * POLICY_NUM_MOTOR + motor];
    f[OFF_REF_UPPER + i] = target - D_DEFAULTS[motor];
  }

  float* h = history + size_t(env) * HISTORY_LEN * FRAME;
  if (first) {
    for (int k = 0; k < HISTORY_LEN; ++k) {
      for (int i = 0; i < FRAME; ++i) h[k * FRAME + i] = f[i];
    }
  }

  float* o = obs + size_t(env) * NUM_OBS;
  int at = 0;
  for (int i = 0; i < HEAD; ++i) o[at++] = f[i];
  for (int s = 0; s < 12; ++s) {
    for (int k = 0; k < HISTORY_LEN; ++k) {
      const float* hf = h + k * FRAME;
      for (int i = D_SEGMENTS[s][0]; i < D_SEGMENTS[s][1]; ++i) o[at++] = hf[i];
    }
  }
  for (int i = HEAD; i < FRAME; ++i) o[at++] = f[i];

  for (int k = HISTORY_LEN - 1; k > 0; --k) {
    for (int i = 0; i < FRAME; ++i) h[k * FRAME + i] = h[(k - 1) * FRAME + i];
  }
  for (int i = 0; i < FRAME; ++i) h[i] = f[i];
}

__global__ void k_asap_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + size_t(env) * NUM_ACTIONS;
  float* q = q_target + size_t(env) * POLICY_NUM_MOTOR;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q[j] = arm_pose[size_t(env) * POLICY_NUM_MOTOR + j];
  }
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const float raw = fminf(fmaxf(a[i], -100.0f), 100.0f);
    last_action[size_t(env) * NUM_ACTIONS + i] = raw;
    q[i] = fminf(
        fmaxf(D_DEFAULTS[i] + raw * ACTION_SCALE, D_Q_MIN[i]),
        D_Q_MAX[i]
    );
  }

  for (int i = NUM_ACTIONS; i < NUM_OWNED; ++i) q[i] = D_DEFAULTS[i];
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr,
        *d_hist = nullptr;
  unsigned char* d_walk = nullptr;
  int envs = 0;
  long long ticks = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_obs,
          (void*)d_act,
          (void*)d_last,
          (void*)d_hist,
          (void*)d_walk}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/asap/model.onnx",
        n,
        NUM_OBS,
        NUM_ACTIONS
    );
    auto zeros = [n](float** p, size_t per) {
      cudaMalloc(p, size_t(n) * per * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * per * sizeof(float));
    };
    zeros(&d_obs, NUM_OBS);
    zeros(&d_act, NUM_ACTIONS);
    zeros(&d_last, NUM_ACTIONS);
    zeros(&d_hist, size_t(HISTORY_LEN) * FRAME);
    cudaMalloc(&d_walk, size_t(n));
    cudaMemset(d_walk, 0, size_t(n));
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;

    const float control_time = float(ticks) * DT;
    const int first = ticks == 0 ? 1 : 0;
    ++ticks;
    k_asap_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        c.task,
        c.arm_pose,
        d_last,
        d_walk,
        d_hist,
        d_obs,
        control_time,
        first,
        envs
    );
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_asap_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_OWNED; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "asap"; }
};

}
