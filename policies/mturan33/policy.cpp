namespace mturan33 {

constexpr int NUM_OBS = 57;
constexpr int NUM_ACTIONS = 12;
constexpr int OWNED = 15;

constexpr float ACTION_SCALE = 0.4f;
constexpr float OBS_CLIP = 10.0f;
constexpr float HEIGHT_CMD = 0.72f;
constexpr float GAIT_FREQUENCY = 1.5f;
constexpr float CONTROL_DT = 0.02f;

#define MTURAN33_MOTOR_OF_LEG 0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11

__device__ const int D_MOTOR_OF_LEG[NUM_ACTIONS] = {MTURAN33_MOTOR_OF_LEG};

#define MTURAN33_DEFAULT_LEG \
  -0.2f, -0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.4f, 0.4f, -0.2f, -0.2f, 0.0f, 0.0f

__device__ const float D_DEFAULT_LEG[NUM_ACTIONS] = {MTURAN33_DEFAULT_LEG};

const float KPS[OWNED] = {
    150.0f,
    150.0f,
    150.0f,
    150.0f,
    150.0f,
    150.0f,
    150.0f,
    150.0f,
    150.0f,
    150.0f,
    150.0f,
    150.0f,
    100.0f,
    100.0f,
    100.0f
};

const float KDS[OWNED] = {
    15.0f,
    15.0f,
    15.0f,
    15.0f,
    15.0f,
    15.0f,
    15.0f,
    15.0f,
    15.0f,
    15.0f,
    15.0f,
    15.0f,
    10.0f,
    10.0f,
    10.0f
};

const policy_api::Limits LIMITS = {0.0, 0.6, 0.13, 0.22, 0.0};

__device__ inline float clip10(float v) {
  return fminf(fmaxf(v, -OBS_CLIP), OBS_CLIP);
}

__global__ void k_mturan33_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ lin_vel,
    const float* __restrict__ base_quat,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    float phase,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* o = obs + env * NUM_OBS;
  for (int k = 0; k < 3; ++k) {
    o[k] = clip10(lin_vel[env * 3 + k]);
    o[3 + k] = clip10(gyro[env * 3 + k]);
    o[6 + k] = clip10(gravity[env * 3 + k]);
    o[34 + k] = clip10(cmd[env * 3 + k]);
  }

  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const int m = D_MOTOR_OF_LEG[i];
    o[9 + i] = clip10(motor_q[env * POLICY_NUM_MOTOR + m]);
    o[21 + i] = clip10(motor_dq[env * POLICY_NUM_MOTOR + m]);
    o[39 + i] = clip10(last_action[env * NUM_ACTIONS + i]);
  }

  o[33] = HEIGHT_CMD;
  o[37] = clip10(sinf(2.0f * float(M_PI) * phase));
  o[38] = clip10(cosf(2.0f * float(M_PI) * phase));

  o[51] = 0.0f;
  o[52] = 0.0f;
  o[53] = 0.0f;

  const float* q = base_quat + env * 4;
  const float x = q[0], y = q[1], z = q[2], w = q[3];
  o[54] = clip10(atan2f(2.0f * (w * x + y * z), 1.0f - 2.0f * (x * x + y * y)));
  o[55] = clip10(asinf(fminf(fmaxf(2.0f * (w * y - z * x), -1.0f), 1.0f)));
  o[56] = clip10(atan2f(2.0f * (w * z + x * y), 1.0f - 2.0f * (y * y + z * z)));
}

__global__ void k_mturan33_act(
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

  for (int i = 0; i < NUM_ACTIONS; ++i) {
    last_action[env * NUM_ACTIONS + i] = a[i];
    qt[D_MOTOR_OF_LEG[i]] = D_DEFAULT_LEG[i] + a[i] * ACTION_SCALE;
  }

  for (int j = NUM_ACTIONS; j < OWNED; ++j) qt[j] = 0.0f;
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  double gait_phase = 0.0;
  int envs = 0;

  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last})
      if (p) cudaFree(p);
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/mturan33/model.onnx",
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
    k_mturan33_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.base_lin_vel,
        c.base_quat,
        c.cmd,
        d_last,
        d_obs,
        float(gait_phase),
        envs
    );
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_mturan33_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
    gait_phase = std::fmod(gait_phase + GAIT_FREQUENCY * CONTROL_DT, 1.0);
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return OWNED; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "mturan33"; }
};

}
