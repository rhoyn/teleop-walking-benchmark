namespace run_residual {

constexpr int NUM_ACTIONS = 29;
constexpr int HISTORY = 5;
constexpr int NUM_OBS = HISTORY * (3 + 3 + 3 + 3 * NUM_ACTIONS);
constexpr int MOTION_DIM = 58;
constexpr int CMD_DIM = 3;
constexpr int HIDDEN_LAYERS = 2;
constexpr int HIDDEN_WIDTH = 256;
constexpr int FIRST_ARM = 15;

constexpr float ACTION_SCALE = 0.25f;
constexpr float AR_LEAK = 0.05f;
constexpr float CMG_CMD_EMA = 0.40f;
constexpr float CMG_SIGMA_CLAMP = 3.0f;
constexpr float CMG_OUTPUT_CLAMP = 3.14f;
constexpr float SPEED_FLOOR = 0.50f;
constexpr float VX_MAX = 1.5f;

constexpr float HOLD_POS_M = 0.20f;
constexpr float HOLD_YAW_RAD = 0.12f;

#define RUN_TERM_DIMS 3, 3, 3, NUM_ACTIONS, NUM_ACTIONS, NUM_ACTIONS
#define RUN_TERM_OFFS 0, 15, 30, 45, 190, 335

__device__ const int D_TERM_DIM[6] = {RUN_TERM_DIMS};
__device__ const int D_TERM_OFF[6] = {RUN_TERM_OFFS};

__device__ const float D_DEFAULT_ISAAC[NUM_ACTIONS] = {
    -0.1f, -0.1f, 0.0f,  0.0f,  0.0f,   0.0f,  0.0f,   0.0f, 0.0f, 0.3f,
    0.3f,  0.3f,  0.3f,  -0.2f, -0.2f,  0.25f, -0.25f, 0.0f, 0.0f, 0.0f,
    0.0f,  0.97f, 0.97f, 0.15f, -0.15f, 0.0f,  0.0f,   0.0f, 0.0f
};

__device__ const int D_MJ_OF_IL[NUM_ACTIONS] = {0,  6,  12, 1,  7,  13, 2,  8,
                                                14, 3,  9,  15, 22, 4,  10, 16,
                                                23, 5,  11, 17, 24, 18, 25, 19,
                                                26, 20, 27, 21, 28};
__device__ const int D_IL_OF_MJ[NUM_ACTIONS] = {0,  3,  6,  9,  13, 17, 1,  4,
                                                7,  10, 14, 18, 2,  5,  8,  11,
                                                15, 19, 21, 23, 25, 27, 12, 16,
                                                20, 22, 24, 26, 28};

__device__ const float D_MOTION_MEAN[MOTION_DIM] = {
    -0.138017774f,
    -0.0190545321f,
    0.0f,
    0.675016761f,
    -0.0547883585f,
    0.0f,
    -0.138017803f,
    0.0190545227f,
    0.0f,
    0.67501688f,
    -0.0547883362f,
    0.0f,
    -5.62028896e-11f,
    -3.69755875e-12f,
    0.0425066724f,
    0.2713103f,
    0.373977512f,
    -0.410876751f,
    0.636402845f,
    0.0f,
    0.0f,
    0.0f,
    0.27131018f,
    -0.373977512f,
    0.410876781f,
    0.636402309f,
    0.0f,
    0.0f,
    0.0f,
    0.0178835653f,
    6.16252141e-19f,
    0.0f,
    -0.000563064765f,
    0.0260602366f,
    0.0f,
    0.0178835429f,
    -6.16252296e-19f,
    0.0f,
    -0.000563072914f,
    0.0260602403f,
    0.0f,
    -1.01259494e-27f,
    -5.15739233e-26f,
    -0.0145212635f,
    0.0178965162f,
    -1.47961353e-18f,
    -1.43513841e-18f,
    -0.00474209245f,
    0.0f,
    0.0f,
    0.0f,
    0.01789652f,
    1.47961219e-18f,
    1.43514471e-18f,
    -0.00474209432f,
    0.0f,
    0.0f,
    0.0f
};
__device__ const float D_MOTION_STD[MOTION_DIM] = {
    0.306055456f, 0.104258813f, 0.100000001f, 0.311078489f, 0.231486648f,
    0.100000001f, 0.306055456f, 0.104258798f, 0.100000001f, 0.31107825f,
    0.231486648f, 0.100000001f, 0.190625548f, 0.100000001f, 0.100000001f,
    0.215748578f, 0.100000001f, 0.217310473f, 0.541980505f, 0.100000001f,
    0.100000001f, 0.100000001f, 0.215748414f, 0.100000001f, 0.217310473f,
    0.541980505f, 0.100000001f, 0.100000001f, 0.100000001f, 1.94409943f,
    0.100000001f, 0.100000001f, 3.34542727f,  2.35869026f,  0.100000001f,
    1.94409919f,  0.100000001f, 0.100000001f, 3.34542727f,  2.35869026f,
    0.100000001f, 0.100000001f, 0.100000001f, 0.390017211f, 1.29102993f,
    0.100000001f, 0.100000001f, 1.3150996f,   0.100000001f, 0.100000001f,
    0.100000001f, 1.29102993f,  0.100000001f, 0.100000001f, 1.3150996f,
    0.100000001f, 0.100000001f, 0.100000001f
};
__device__ const float D_CMD_MEAN[CMD_DIM] = {
    0.882105529f,
    -2.06150638e-10f,
    -6.58186339e-10f
};
__device__ const float D_CMD_STD[CMD_DIM] = {
    0.770421565f,
    0.196364716f,
    0.595767558f
};

const float KPS[NUM_ACTIONS] = {70.0f,  70.0f, 70.0f, 120.0f, 40.0f, 40.0f,
                                70.0f,  70.0f, 70.0f, 120.0f, 40.0f, 40.0f,
                                150.0f, 40.0f, 40.0f, 40.0f,  40.0f, 40.0f,
                                40.0f,  40.0f, 40.0f, 40.0f,  40.0f, 40.0f,
                                40.0f,  40.0f, 40.0f, 40.0f,  40.0f};
const float KDS[NUM_ACTIONS] = {2.5f, 4.0f, 2.5f, 5.2f, 2.0f, 2.0f, 2.5f, 4.0f,
                                2.5f, 5.2f, 2.0f, 2.0f, 5.0f, 5.0f, 5.0f, 2.0f,
                                2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                                2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

const policy_api::Limits LIMITS = {0.0, 1.5, 0.0, 1.0, 0.0};

__global__ void k_run_residual_cmg_in(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ cmd,
    const float* __restrict__ task,
    float* __restrict__ prev_motion,
    float* __restrict__ smoothed,
    float* __restrict__ drive,
    float* __restrict__ motion_in,
    float* __restrict__ cmd_in,
    int first,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* c = cmd + env * 3;
  float d[3] = {c[0], c[1], c[2]};
  const float dist = task[env * 4 + 0];
  const float yaw_err = task[env * 4 + 1];
  if (dist > HOLD_POS_M || fabsf(yaw_err) > HOLD_YAW_RAD) {
    d[0] = fminf(fmaxf(d[0], SPEED_FLOOR), VX_MAX);
    d[1] = 0.0f;
  }
  for (int k = 0; k < 3; ++k) drive[env * 3 + k] = d[k];

  float* pm = prev_motion + env * MOTION_DIM;
  if (first) {
    for (int m = 0; m < NUM_ACTIONS; ++m) {
      pm[m] = motor_q[env * POLICY_NUM_MOTOR + m];
      pm[NUM_ACTIONS + m] = motor_dq[env * POLICY_NUM_MOTOR + m];
    }
  }

  float* sm = smoothed + env * CMD_DIM;
  for (int i = 0; i < CMD_DIM; ++i) {
    sm[i] += CMG_CMD_EMA * (d[i] - sm[i]);
    cmd_in[env * CMD_DIM + i] = (sm[i] - D_CMD_MEAN[i]) / D_CMD_STD[i];
  }
  for (int i = 0; i < MOTION_DIM; ++i) {
    const float lo = D_MOTION_MEAN[i] - CMG_SIGMA_CLAMP * D_MOTION_STD[i];
    const float hi = D_MOTION_MEAN[i] + CMG_SIGMA_CLAMP * D_MOTION_STD[i];
    motion_in[env * MOTION_DIM + i] =
        (fminf(fmaxf(pm[i], lo), hi) - D_MOTION_MEAN[i]) / D_MOTION_STD[i];
  }
}

__global__ void k_run_residual_cmg_out(
    const float* __restrict__ motion_out,
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    float* __restrict__ prev_motion,
    float* __restrict__ qref,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* pm = prev_motion + env * MOTION_DIM;
  for (int i = 0; i < MOTION_DIM; ++i) {
    const float ref = fminf(
        fmaxf(
            motion_out[env * MOTION_DIM + i] * D_MOTION_STD[i] +
                D_MOTION_MEAN[i],
            -CMG_OUTPUT_CLAMP
        ),
        CMG_OUTPUT_CLAMP
    );
    const int motor = i % NUM_ACTIONS;
    const float measured = i < NUM_ACTIONS
                               ? motor_q[env * POLICY_NUM_MOTOR + motor]
                               : motor_dq[env * POLICY_NUM_MOTOR + motor];
    const float leak = motor < FIRST_ARM ? AR_LEAK : 0.0f;
    pm[i] = (1.0f - leak) * ref + leak * measured;
    if (i < NUM_ACTIONS) qref[env * NUM_ACTIONS + motor] = ref;
  }
}

__global__ void k_run_residual_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ drive,
    const float* __restrict__ qref,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    int first,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float frame[6][NUM_ACTIONS];
  for (int k = 0; k < 3; ++k) {
    frame[0][k] = gyro[env * 3 + k];
    frame[1][k] = gravity[env * 3 + k];
    frame[2][k] = drive[env * 3 + k];
  }
  const float* la = last_action + env * NUM_ACTIONS;
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const int motor = D_MJ_OF_IL[i];
    if (motor >= FIRST_ARM) {
      frame[3][i] = qref[env * NUM_ACTIONS + motor] + ACTION_SCALE * la[i] -
                    D_DEFAULT_ISAAC[i];
      frame[4][i] = 0.0f;
    } else {
      frame[3][i] =
          motor_q[env * POLICY_NUM_MOTOR + motor] - D_DEFAULT_ISAAC[i];
      frame[4][i] = motor_dq[env * POLICY_NUM_MOTOR + motor];
    }
    frame[5][i] = la[i];
  }

  float* o = obs + env * NUM_OBS;
  for (int t = 0; t < 6; ++t) {
    const int dim = D_TERM_DIM[t];
    float* base = o + D_TERM_OFF[t];
    if (first) {
      for (int h = 0; h < HISTORY; ++h) {
        for (int k = 0; k < dim; ++k) base[h * dim + k] = frame[t][k];
      }
    } else {
      for (int k = 0; k < (HISTORY - 1) * dim; ++k) base[k] = base[k + dim];
      for (int k = 0; k < dim; ++k) base[(HISTORY - 1) * dim + k] = frame[t][k];
    }
  }
}

__global__ void k_run_residual_act(
    const float* __restrict__ actions,
    const float* __restrict__ qref,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* la = last_action + env * NUM_ACTIONS;
  for (int i = 0; i < NUM_ACTIONS; ++i) la[i] = actions[env * NUM_ACTIONS + i];
  for (int m = 0; m < POLICY_NUM_MOTOR; ++m) {
    q_target[env * POLICY_NUM_MOTOR + m] = arm_pose[env * POLICY_NUM_MOTOR + m];
  }
  for (int m = 0; m < FIRST_ARM; ++m) {
    q_target[env * POLICY_NUM_MOTOR + m] =
        qref[env * NUM_ACTIONS + m] + ACTION_SCALE * la[D_IL_OF_MJ[m]];
  }
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> cmg, actor;
  float *d_prev_motion = nullptr, *d_smoothed = nullptr, *d_drive = nullptr;
  float *d_motion_in = nullptr, *d_cmd_in = nullptr, *d_motion_out = nullptr;
  float *d_qref = nullptr, *d_obs = nullptr, *d_last = nullptr,
        *d_actions = nullptr;
  float *d_h[2] = {nullptr, nullptr}, *d_c[2] = {nullptr, nullptr};
  int envs = 0;
  bool first = true;

  ~Policy() override {
    for (void* p :
         {(void*)d_prev_motion,
          (void*)d_smoothed,
          (void*)d_drive,
          (void*)d_motion_in,
          (void*)d_cmd_in,
          (void*)d_motion_out,
          (void*)d_qref,
          (void*)d_obs,
          (void*)d_last,
          (void*)d_actions,
          (void*)d_h[0],
          (void*)d_h[1],
          (void*)d_c[0],
          (void*)d_c[1]}) {
      if (p) cudaFree(p);
    }
  }

  static float* zeros(size_t n) {
    float* p = nullptr;
    cudaMalloc(&p, n * sizeof(float));
    cudaMemset(p, 0, n * sizeof(float));
    return p;
  }

  void init(int n) override {
    envs = n;
    cmg = policy_api::engine_make(
        "policies/run_residual/model_cmg.onnx",
        n,
        {{"prev_motion", {-1, MOTION_DIM}}, {"command", {-1, CMD_DIM}}},
        {{"motion", {-1, MOTION_DIM}}}
    );
    actor = policy_api::engine_make(
        "policies/run_residual/model_residual.onnx",
        n,
        {{"obs", {-1, NUM_OBS}},
         {"h_in", {HIDDEN_LAYERS, -1, HIDDEN_WIDTH}},
         {"c_in", {HIDDEN_LAYERS, -1, HIDDEN_WIDTH}}},
        {{"actions", {-1, NUM_ACTIONS}},
         {"h_out", {HIDDEN_LAYERS, -1, HIDDEN_WIDTH}},
         {"c_out", {HIDDEN_LAYERS, -1, HIDDEN_WIDTH}}}
    );

    const size_t rows = size_t(n);
    d_prev_motion = zeros(rows * MOTION_DIM);
    d_smoothed = zeros(rows * CMD_DIM);
    d_drive = zeros(rows * CMD_DIM);
    d_motion_in = zeros(rows * MOTION_DIM);
    d_cmd_in = zeros(rows * CMD_DIM);
    d_motion_out = zeros(rows * MOTION_DIM);
    d_qref = zeros(rows * NUM_ACTIONS);
    d_obs = zeros(rows * NUM_OBS);
    d_last = zeros(rows * NUM_ACTIONS);
    d_actions = zeros(rows * NUM_ACTIONS);
    for (int i = 0; i < 2; ++i) {
      d_h[i] = zeros(rows * HIDDEN_LAYERS * HIDDEN_WIDTH);
      d_c[i] = zeros(rows * HIDDEN_LAYERS * HIDDEN_WIDTH);
    }
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    const int seed = first ? 1 : 0;

    k_run_residual_cmg_in<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.cmd,
        c.task,
        d_prev_motion,
        d_smoothed,
        d_drive,
        d_motion_in,
        d_cmd_in,
        seed,
        envs
    );
    {
      const float* in[2] = {d_motion_in, d_cmd_in};
      float* out[1] = {d_motion_out};
      policy_api::engine_run(*cmg, in, out, envs);
    }
    k_run_residual_cmg_out<<<blocks, threads>>>(
        d_motion_out,
        c.motor_q,
        c.motor_dq,
        d_prev_motion,
        d_qref,
        envs
    );

    k_run_residual_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        d_drive,
        d_qref,
        d_last,
        d_obs,
        seed,
        envs
    );
    {
      const float* in[3] = {d_obs, d_h[0], d_c[0]};
      float* out[3] = {d_actions, d_h[1], d_c[1]};
      policy_api::engine_run(*actor, in, out, envs);
    }
    std::swap(d_h[0], d_h[1]);
    std::swap(d_c[0], d_c[1]);

    k_run_residual_act<<<blocks, threads>>>(
        d_actions,
        d_qref,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
    first = false;
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return FIRST_ARM; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "run_residual"; }
};

}
