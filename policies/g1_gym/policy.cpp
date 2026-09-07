namespace g1_gym {

constexpr int NUM_OBS = 90;
constexpr int NUM_ACT = 27;
constexpr int HIDDEN = 64;
constexpr int DRIVEN = 13;
constexpr int WAIST_LOCK_BEGIN = 13;
constexpr int WAIST_LOCK_END = 15;
constexpr float WAIST_LOCK_KP = 300.0f;
constexpr float WAIST_LOCK_KD = 3.0f;
constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float DOF_VEL_SCALE = 0.05f;
constexpr float ACTION_SCALE = 0.25f;
constexpr float JOINT_MARGIN = 0.01f;
constexpr float CMD_SCALE_X = 2.0f, CMD_SCALE_Y = 2.0f, CMD_SCALE_YAW = 0.25f;

__host__ __device__ inline int to_bench(int j) {
  return j < DRIVEN ? j : j + 2;
}

#define G1_GYM_DEFAULTS                                                        \
  -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f,  \
      0.0f, 0.3f, 0.3f, 0.0f, 0.9f, 0.0f, 0.0f, 0.0f, 0.3f, -0.3f, 0.0f, 0.9f, \
      0.0f, 0.0f, 0.0f

__device__ const float D_DEFAULTS[NUM_ACT] = {G1_GYM_DEFAULTS};

#define G1_GYM_LOWER                                                       \
  -2.5307f, -0.5236f, -2.7576f, -0.087267f, -0.87267f, -0.2618f, -2.5307f, \
      -2.9671f, -2.7576f, -0.087267f, -0.87267f, -0.2618f, -2.618f
#define G1_GYM_UPPER                                                      \
  2.8798f, 2.9671f, 2.7576f, 2.8798f, 0.5236f, 0.2618f, 2.8798f, 0.5236f, \
      2.7576f, 2.8798f, 0.5236f, 0.2618f, 2.618f
__device__ const float D_LOWER[DRIVEN] = {G1_GYM_LOWER};
__device__ const float D_UPPER[DRIVEN] = {G1_GYM_UPPER};

const float KPS[WAIST_LOCK_END] = {
    100,
    100,
    100,
    200,
    20,
    20,
    100,
    100,
    100,
    200,
    20,
    20,
    100,
    WAIST_LOCK_KP,
    WAIST_LOCK_KP
};
const float KDS[WAIST_LOCK_END] = {
    2.5f,
    2.5f,
    2.5f,
    5.0f,
    0.2f,
    0.1f,
    2.5f,
    2.5f,
    2.5f,
    5.0f,
    0.2f,
    0.1f,
    4.0f,
    WAIST_LOCK_KD,
    WAIST_LOCK_KD
};

const policy_api::Limits LIMITS =
    {-1.0, 1.0, 0.5, 1.0, 0.0, 0.08, 0.15, 0.05, 0.10, 1.5, 2.0};

__global__ void k_g1_gym_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* o = obs + env * NUM_OBS;
  const float* c = cmd + env * 3;
  for (int k = 0; k < 3; ++k) {
    o[k] = gyro[env * 3 + k] * ANG_VEL_SCALE;
    o[3 + k] = gravity[env * 3 + k];
  }
  o[6] = c[0] * CMD_SCALE_X;
  o[7] = c[1] * CMD_SCALE_Y;
  o[8] = c[2] * CMD_SCALE_YAW;
  for (int j = 0; j < NUM_ACT; ++j) {
    const int b = to_bench(j);
    o[9 + j] = motor_q[env * POLICY_NUM_MOTOR + b] - D_DEFAULTS[j];
    o[9 + NUM_ACT + j] = motor_dq[env * POLICY_NUM_MOTOR + b] * DOF_VEL_SCALE;
    o[9 + 2 * NUM_ACT + j] = last_action[env * NUM_ACT + j];
  }
}

__global__ void k_g1_gym_act(
    const float* __restrict__ action,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = action + env * NUM_ACT;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  }

  for (int j = 0; j < NUM_ACT; ++j) last_action[env * NUM_ACT + j] = a[j];
  for (int j = 0; j < DRIVEN; ++j) {
    const float want = D_DEFAULTS[j] + a[j] * ACTION_SCALE;
    q_target[env * POLICY_NUM_MOTOR + to_bench(j)] = fminf(
        fmaxf(want, D_LOWER[j] + JOINT_MARGIN),
        D_UPPER[j] - JOINT_MARGIN
    );
  }
  for (int j = WAIST_LOCK_BEGIN; j < WAIST_LOCK_END; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = 0.0f;
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  float *d_h_in = nullptr, *d_h_out = nullptr;
  float *d_c_in = nullptr, *d_c_out = nullptr;
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
        "policies/g1_gym/model.onnx",
        n,
        {{"obs", {-1, NUM_OBS}},
         {"hidden_in", {1, -1, HIDDEN}},
         {"cell_in", {1, -1, HIDDEN}}},
        {{"action", {-1, NUM_ACT}},
         {"hidden_out", {1, -1, HIDDEN}},
         {"cell_out", {1, -1, HIDDEN}}}
    );

    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_ACT * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_ACT * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_ACT * sizeof(float));

    for (float** p : {&d_h_in, &d_h_out, &d_c_in, &d_c_out}) {
      cudaMalloc(p, size_t(n) * HIDDEN * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * HIDDEN * sizeof(float));
    }
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_g1_gym_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_obs,
        envs
    );
    const float* in[3] = {d_obs, d_h_in, d_c_in};
    float* out[3] = {d_act, d_h_out, d_c_out};
    policy_api::engine_run(*engine, in, out, envs);
    std::swap(d_h_in, d_h_out);
    std::swap(d_c_in, d_c_out);
    k_g1_gym_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return WAIST_LOCK_END; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "g1_gym"; }
};

}
