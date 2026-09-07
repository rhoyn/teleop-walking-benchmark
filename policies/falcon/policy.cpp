namespace falcon {

constexpr int NUM_LOWER = 15;
constexpr int OBS_DIM = 102;
constexpr int NUM_UPPER = 14;

constexpr float ACTION_SCALE = 0.25f;
constexpr float GAIT_PERIOD = 0.9f;
constexpr double DT = 0.02;
constexpr double WARMUP_S = 1.5;
constexpr float SCALE_ANG_VEL = 0.25f;
constexpr float SCALE_DOF_VEL = 0.05f;
constexpr float ACTION_CLIP = 100.0f;

constexpr float VX_MIN = -0.6f, VX_MAX = 0.9f, VY_ABS = 0.5f;
constexpr float YAW_ABS = 0.8f, SPEED_NORM = 0.9f;

constexpr float CMD_KP_POS = 2.0f;

#define FALCON_DEFAULTS                                                       \
  -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, \
      0.0f, 0.0f, 0.0f

#define FALCON_POS_LOWER                                                    \
  -2.5307f, -0.5236f, -2.7576f, -0.087267f, -0.87267f, -0.2618f, -2.5307f,  \
      -2.9671f, -2.7576f, -0.087267f, -0.87267f, -0.2618f, -2.618f, -0.52f, \
      -0.52f

#define FALCON_POS_UPPER                                                  \
  2.8798f, 2.9671f, 2.7576f, 2.8798f, 0.5236f, 0.2618f, 2.8798f, 0.5236f, \
      2.7576f, 2.8798f, 0.5236f, 0.2618f, 2.618f, 0.52f, 0.52f

__device__ const float D_DEFAULTS[NUM_LOWER] = {FALCON_DEFAULTS};
__device__ const float D_POS_LOWER[NUM_LOWER] = {FALCON_POS_LOWER};
__device__ const float D_POS_UPPER[NUM_LOWER] = {FALCON_POS_UPPER};

const float KPS[NUM_LOWER] =
    {100, 100, 100, 200, 20, 20, 100, 100, 100, 200, 20, 20, 300, 300, 300};
const float KDS[NUM_LOWER] =
    {2.5f, 2.5f, 2.5f, 5, 0.2f, 0.1f, 2.5f, 2.5f, 2.5f, 5, 0.2f, 0.1f, 5, 5, 5};

const policy_api::Limits LIMITS =
    {-0.6, 0.9, 0.5, 0.8, 0.9, 0.10, 0.20, 0.05, 0.12, 2.0, 1.5};

__global__ void k_falcon_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ task,
    const float* __restrict__ last_action,
    float* __restrict__ phase_clock,
    float* __restrict__ obs,
    int past_warmup,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* c = cmd + size_t(env) * 3;
  float vx = c[0], vy = c[1];
  const float wz = c[2];

  const bool walking = (vx != 0.0f) || (vy != 0.0f) || (wz != 0.0f);

  const float speed = sqrtf(vx * vx + vy * vy);
  if (speed > 1e-9f) {
    const float dist = task[size_t(env) * 4 + 0];
    const float want = fminf(CMD_KP_POS * dist, SPEED_NORM);
    const float scale = want / speed;
    vx = fminf(fmaxf(vx * scale, VX_MIN), VX_MAX);
    vy = fminf(fmaxf(vy * scale, -VY_ABS), VY_ABS);
  }

  const bool stand = walking && past_warmup != 0;
  const float cmd_vx = stand ? vx : 0.0f;
  const float cmd_vy = stand ? vy : 0.0f;
  const float cmd_wz = stand ? wz : 0.0f;

  float clock = phase_clock[env];
  if (stand) clock += float(DT);
  phase_clock[env] = clock;
  const float phase = fmodf(clock, GAIT_PERIOD) / GAIT_PERIOD;

  float* o = obs + size_t(env) * OBS_DIM;
  int k = 0;
  for (int i = 0; i < NUM_LOWER; ++i) {
    o[k++] = last_action[size_t(env) * NUM_LOWER + i];
  }
  for (int i = 0; i < 3; ++i) {
    o[k++] = gyro[size_t(env) * 3 + i] * SCALE_ANG_VEL;
  }
  o[k++] = cmd_wz;
  o[k++] = cmd_vx;
  o[k++] = cmd_vy;
  o[k++] = stand ? 1.0f : 0.0f;
  for (int i = 0; i < 3; ++i) o[k++] = 0.0f;
  o[k++] = cosf(2.0f * float(M_PI) * phase);

  for (int i = 0; i < POLICY_NUM_MOTOR; ++i) {
    o[k++] = i >= NUM_LOWER
                 ? 0.0f
                 : motor_q[size_t(env) * POLICY_NUM_MOTOR + i] - D_DEFAULTS[i];
  }
  for (int i = 0; i < POLICY_NUM_MOTOR; ++i) {
    o[k++] = i >= NUM_LOWER
                 ? 0.0f
                 : motor_dq[size_t(env) * POLICY_NUM_MOTOR + i] * SCALE_DOF_VEL;
  }
  for (int i = 0; i < 3; ++i) o[k++] = gravity[size_t(env) * 3 + i];
  for (int i = 0; i < NUM_UPPER; ++i) o[k++] = 0.0f;
  o[k++] = sinf(2.0f * float(M_PI) * phase);
}

__global__ void k_falcon_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + size_t(env) * NUM_LOWER;
  float* q = q_target + size_t(env) * POLICY_NUM_MOTOR;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q[j] = arm_pose[size_t(env) * POLICY_NUM_MOTOR + j];
  }

  for (int i = 0; i < NUM_LOWER; ++i) {
    const float v = fminf(fmaxf(a[i], -ACTION_CLIP), ACTION_CLIP);
    last_action[size_t(env) * NUM_LOWER + i] = v;
    q[i] = fminf(
        fmaxf(D_DEFAULTS[i] + v * ACTION_SCALE, D_POS_LOWER[i]),
        D_POS_UPPER[i]
    );
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr,
        *d_phase = nullptr;
  int envs = 0;
  long long ticks = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_obs, (void*)d_act, (void*)d_last, (void*)d_phase}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/falcon/model.onnx",
        n,
        OBS_DIM,
        NUM_LOWER
    );
    cudaMalloc(&d_obs, size_t(n) * OBS_DIM * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * OBS_DIM * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_LOWER * sizeof(float));
    cudaMemset(d_act, 0, size_t(n) * NUM_LOWER * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_LOWER * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_LOWER * sizeof(float));
    cudaMalloc(&d_phase, size_t(n) * sizeof(float));
    cudaMemset(d_phase, 0, size_t(n) * sizeof(float));
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;

    const int past_warmup = double(ticks) * DT >= WARMUP_S ? 1 : 0;
    ++ticks;
    k_falcon_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        c.task,
        d_last,
        d_phase,
        d_obs,
        past_warmup,
        envs
    );
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_falcon_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_LOWER; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "falcon"; }
};

}
