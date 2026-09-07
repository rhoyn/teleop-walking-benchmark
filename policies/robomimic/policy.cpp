namespace robomimic {

constexpr int NUM_ACTIONS = 29;
constexpr int NUM_OBS = 96;
constexpr int HIDDEN = 256;
constexpr int FIRST_ARM_MOTOR = 15;
constexpr float ACTION_SCALE = 0.25f;
constexpr float IO_CLIP = 100.0f;

#define ROBOMIMIC_JOINT_TO_MOTOR                                            \
  0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, \
      18, 25, 19, 26, 20, 27, 21, 28

#define ROBOMIMIC_DEFAULTS                                                     \
  -0.2f, -0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.42f, 0.42f, 0.35f, \
      0.35f, -0.23f, -0.23f, 0.18f, -0.18f, 0.0f, 0.0f, 0.0f, 0.0f, 0.87f,     \
      0.87f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f

const int JOINT_TO_MOTOR[NUM_ACTIONS] = {ROBOMIMIC_JOINT_TO_MOTOR};
__device__ const int D_JOINT_TO_MOTOR[NUM_ACTIONS] = {ROBOMIMIC_JOINT_TO_MOTOR};

__device__ const float D_DEFAULTS[NUM_ACTIONS] = {ROBOMIMIC_DEFAULTS};

const float KPS_JOINT[NUM_ACTIONS] = {200, 200, 200, 150, 150, 200, 150, 150,
                                      200, 200, 200, 100, 100, 20,  20,  100,
                                      100, 20,  20,  50,  50,  50,  50,  40,
                                      40,  40,  40,  40,  40};
const float KDS_JOINT[NUM_ACTIONS] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                      5, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                      2, 2, 2, 2, 2, 2, 2, 2, 2};

const std::array<float, FIRST_ARM_MOTOR> KPS = [] {
  std::array<float, FIRST_ARM_MOTOR> g{};
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    if (JOINT_TO_MOTOR[i] < FIRST_ARM_MOTOR)
      g[JOINT_TO_MOTOR[i]] = KPS_JOINT[i];
  }
  return g;
}();
const std::array<float, FIRST_ARM_MOTOR> KDS = [] {
  std::array<float, FIRST_ARM_MOTOR> g{};
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    if (JOINT_TO_MOTOR[i] < FIRST_ARM_MOTOR)
      g[JOINT_TO_MOTOR[i]] = KDS_JOINT[i];
  }
  return g;
}();

const policy_api::Limits LIMITS = {-0.4, 0.7, 0.4, 0.8, 0.4};

__device__ inline float clip_io(float v) {
  return fminf(fmaxf(v, -IO_CLIP), IO_CLIP);
}

__global__ void k_robomimic_obs(
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
  for (int k = 0; k < 3; ++k) {
    o[k] = clip_io(gyro[env * 3 + k]);
    o[3 + k] = clip_io(gravity[env * 3 + k]);
    o[6 + k] = clip_io(cmd[env * 3 + k]);
  }
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const int m = D_JOINT_TO_MOTOR[i];

    const bool is_arm = m >= FIRST_ARM_MOTOR;
    o[9 + i] =
        is_arm ? 0.0f
               : clip_io(motor_q[env * POLICY_NUM_MOTOR + m] - D_DEFAULTS[i]);
    o[9 + NUM_ACTIONS + i] =
        is_arm ? 0.0f : clip_io(motor_dq[env * POLICY_NUM_MOTOR + m]);
    o[9 + 2 * NUM_ACTIONS + i] = clip_io(last_action[env * NUM_ACTIONS + i]);
  }
}

__global__ void k_robomimic_act(
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
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const float clipped = clip_io(a[i]);

    last_action[env * NUM_ACTIONS + i] = clipped;
    const int m = D_JOINT_TO_MOTOR[i];
    if (m < FIRST_ARM_MOTOR) {
      q_target[env * POLICY_NUM_MOTOR + m] =
          D_DEFAULTS[i] + clipped * ACTION_SCALE;
    }
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
        "policies/robomimic/model.onnx",
        n,
        {{"obs", {-1, NUM_OBS}},
         {"hidden_in", {1, -1, HIDDEN}},
         {"cell_in", {1, -1, HIDDEN}}},
        {{"action", {-1, NUM_ACTIONS}},
         {"hidden_out", {1, -1, HIDDEN}},
         {"cell_out", {1, -1, HIDDEN}}}
    );

    const size_t obs_bytes = size_t(n) * NUM_OBS * sizeof(float);
    const size_t act_bytes = size_t(n) * NUM_ACTIONS * sizeof(float);
    const size_t state_bytes = size_t(n) * HIDDEN * sizeof(float);
    cudaMalloc(&d_obs, obs_bytes);
    cudaMemset(d_obs, 0, obs_bytes);
    cudaMalloc(&d_act, act_bytes);
    cudaMalloc(&d_last, act_bytes);
    cudaMemset(d_last, 0, act_bytes);

    for (float** p : {&d_h_in, &d_h_out, &d_c_in, &d_c_out}) {
      cudaMalloc(p, state_bytes);
      cudaMemset(*p, 0, state_bytes);
    }
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_robomimic_obs<<<blocks, threads>>>(
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
    k_robomimic_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS.data(); }
  const float* kd() const override { return KDS.data(); }
  int owned() const override { return FIRST_ARM_MOTOR; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "robomimic"; }
};

}
