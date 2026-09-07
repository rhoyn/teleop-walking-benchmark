namespace holosoma {

constexpr int NUM_ACTIONS = 29;
constexpr int NUM_OBS = 100;
constexpr float ACTION_SCALE = 0.25f;
constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float DOF_VEL_SCALE = 0.05f;

constexpr int OWNED = 15;

constexpr double CONTROL_DT = 0.02;
constexpr double GAIT_PERIOD_S = 1.0;
constexpr double STAND_EPS = 0.01;
constexpr double ARM_SOFT_START_S = 1.5;

__device__ const float D_DEFAULTS[NUM_ACTIONS] = {
    -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f, -0.312f, 0.0f, 0.0f, 0.669f,
    -0.363f, 0.0f, 0.0f, 0.0f,   0.0f,    0.2f, 0.2f,    0.0f, 0.6f, 0.0f,
    0.0f,    0.0f, 0.2f, -0.2f,  0.0f,    0.6f, 0.0f,    0.0f, 0.0f
};

const float KPS[NUM_ACTIONS] = {
    40.179238471f, 99.098427777f, 40.179238471f, 99.098427777f, 28.501246196f,
    28.501246196f, 40.179238471f, 99.098427777f, 40.179238471f, 99.098427777f,
    28.501246196f, 28.501246196f, 40.179238471f, 28.501246196f, 28.501246196f,
    14.250623098f, 14.250623098f, 14.250623098f, 14.250623098f, 14.250623098f,
    16.778327481f, 16.778327481f, 14.250623098f, 14.250623098f, 14.250623098f,
    14.250623098f, 14.250623098f, 16.778327481f, 16.778327481f
};
const float KDS[NUM_ACTIONS] = {
    2.557889765f, 6.308801854f, 2.557889765f, 6.308801854f, 1.814445687f,
    1.814445687f, 2.557889765f, 6.308801854f, 2.557889765f, 6.308801854f,
    1.814445687f, 1.814445687f, 2.557889765f, 1.814445687f, 1.814445687f,
    0.907222843f, 0.907222843f, 0.907222843f, 0.907222843f, 0.907222843f,
    1.068141502f, 1.068141502f, 0.907222843f, 0.907222843f, 0.907222843f,
    0.907222843f, 0.907222843f, 1.068141502f, 1.068141502f
};

const policy_api::Limits LIMITS = {-1.0, 1.0, 1.0, 1.0, 0.0};

__global__ void k_holosoma_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    double* __restrict__ phase,
    unsigned char* __restrict__ standing,
    float* __restrict__ obs,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* c = cmd + env * 3;
  double* ph = phase + env * 2;

  const double phase_dt = 2.0 * M_PI / (GAIT_PERIOD_S / CONTROL_DT);
  for (int k = 0; k < 2; ++k) {
    double p = fmod(ph[k] + phase_dt + M_PI, 2.0 * M_PI);
    if (p < 0.0) p += 2.0 * M_PI;
    ph[k] = p - M_PI;
  }
  const bool stand = hypotf(c[0], c[1]) < STAND_EPS && fabsf(c[2]) < STAND_EPS;
  if (stand) {
    ph[0] = M_PI;
    ph[1] = M_PI;
    standing[env] = 1;
  } else if (standing[env]) {
    ph[0] = 0.0;
    ph[1] = M_PI;
    standing[env] = 0;
  }

  float* o = obs + env * NUM_OBS;
  const float* la = last_action + env * NUM_ACTIONS;
  for (int i = 0; i < NUM_ACTIONS; ++i) o[i] = la[i];
  for (int k = 0; k < 3; ++k) o[29 + k] = gyro[env * 3 + k] * ANG_VEL_SCALE;
  o[32] = c[2];
  o[33] = c[0];
  o[34] = c[1];
  o[35] = float(cos(ph[0]));
  o[36] = float(cos(ph[1]));
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    o[37 + i] = motor_q[env * POLICY_NUM_MOTOR + i] - D_DEFAULTS[i];
    o[66 + i] = motor_dq[env * POLICY_NUM_MOTOR + i] * DOF_VEL_SCALE;
  }
  for (int k = 0; k < 3; ++k) o[95 + k] = gravity[env * 3 + k];
  o[98] = float(sin(ph[0]));
  o[99] = float(sin(ph[1]));
}

__global__ void k_holosoma_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    float ratio,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + env * NUM_ACTIONS;
  float* la = last_action + env * NUM_ACTIONS;
  float* q = q_target + env * POLICY_NUM_MOTOR;
  const float* arm = arm_pose + env * POLICY_NUM_MOTOR;

  for (int i = 0; i < NUM_ACTIONS; ++i) {
    la[i] = a[i];
    q[i] = D_DEFAULTS[i] + a[i] * ACTION_SCALE;
  }

  for (int i = OWNED; i < NUM_ACTIONS; ++i) {
    q[i] = (1.0f - ratio) * q[i] + ratio * arm[i];
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  double* d_phase = nullptr;
  unsigned char* d_standing = nullptr;
  int envs = 0;

  int steps = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_obs,
          (void*)d_act,
          (void*)d_last,
          (void*)d_phase,
          (void*)d_standing}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/holosoma/model.onnx",
        n,
        NUM_OBS,
        NUM_ACTIONS
    );
    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_standing, size_t(n));
    cudaMemset(d_standing, 0, size_t(n));

    cudaMalloc(&d_phase, size_t(n) * 2 * sizeof(double));
    std::vector<double> seed(size_t(n) * 2);
    for (int e = 0; e < n; ++e) {
      seed[size_t(e) * 2] = 0.0;
      seed[size_t(e) * 2 + 1] = M_PI;
    }
    cudaMemcpy(
        d_phase,
        seed.data(),
        seed.size() * sizeof(double),
        cudaMemcpyHostToDevice
    );
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_holosoma_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_phase,
        d_standing,
        d_obs,
        envs
    );
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    const double control_time = double(steps) * CONTROL_DT;
    const float ratio =
        float(std::clamp(control_time / ARM_SOFT_START_S, 0.0, 1.0));
    k_holosoma_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        ratio,
        envs
    );
    ++steps;
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return OWNED; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "holosoma"; }
};

}
