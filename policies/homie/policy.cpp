namespace homie {

constexpr int NUM_ACTIONS = 12;
constexpr int OBS_DIM = 76;
constexpr int HISTORY = 6;
constexpr int NUM_OBS = HISTORY * OBS_DIM;
constexpr int NUM_DOFS = 27;
constexpr int OWNED = 15;

constexpr float ACTION_SCALE = 0.25f;
constexpr float ACTION_CLIP = 100.0f;
constexpr float CMD_SCALE_X = 2.0f, CMD_SCALE_Y = 2.0f;
constexpr float CMD_SCALE_YAW = 0.25f, CMD_SCALE_H = 1.0f;
constexpr float ANG_VEL_SCALE = 0.5f;
constexpr float DOF_VEL_SCALE = 0.05f;
constexpr float HEIGHT_CMD = 0.74f;

constexpr float VX_MIN = -0.4f, VX_MAX = 0.4f, VY_ABS = 0.4f;
constexpr float YAW_RATE_ABS = 0.8f, SPEED_NORM = 0.4f;
constexpr float POS_P = 0.8f, YAW_P = 1.2f;
constexpr float SHAPED_VX_MIN = -0.25f, SHAPED_VX_MAX = 0.4f,
                SHAPED_VY_ABS = 0.25f;
constexpr float FACE_FAR_M = 1.0f, FACE_NEAR_M = 0.35f;

#define HOMIE_DEFAULT_LEG \
  -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f

const float DEFAULT_LEG[NUM_ACTIONS] = {HOMIE_DEFAULT_LEG};
__device__ const float D_DEFAULT_LEG[NUM_ACTIONS] = {HOMIE_DEFAULT_LEG};

const float KPS[OWNED] =
    {150, 150, 150, 300, 40, 40, 150, 150, 150, 300, 40, 40, 300, 300, 300};
const float KDS[OWNED] = {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5};

const policy_api::Limits LIMITS =
    {VX_MIN, VX_MAX, VY_ABS, YAW_RATE_ABS, SPEED_NORM};

__global__ void k_homie_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ task,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    int envs,
    int primed
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float drive[3] = {cmd[env * 3 + 0], cmd[env * 3 + 1], cmd[env * 3 + 2]};
  const float dist = task[env * 4 + 0];
  const float yaw_err = task[env * 4 + 1];

  const bool translating = drive[0] != 0.0f || drive[1] != 0.0f;
  float bearing = 0.0f, face_w = 0.0f;
  if (translating) {
    bearing = atan2f(drive[1], drive[0]);
    face_w = fminf(
        fmaxf((dist - FACE_NEAR_M) / (FACE_FAR_M - FACE_NEAR_M), 0.0f),
        1.0f
    );
    const float speed = fminf(POS_P * dist, SPEED_NORM);
    drive[0] =
        fminf(fmaxf(speed * cosf(bearing), SHAPED_VX_MIN), SHAPED_VX_MAX);
    drive[1] =
        fminf(fmaxf(speed * sinf(bearing), -SHAPED_VY_ABS), SHAPED_VY_ABS);
  }
  if (face_w > 0.0f) {
    const float blended = remainderf(
        yaw_err + face_w * remainderf(bearing - yaw_err, 2.0f * float(M_PI)),
        2.0f * float(M_PI)
    );
    drive[2] = fminf(fmaxf(YAW_P * blended, -YAW_RATE_ABS), YAW_RATE_ABS);
  } else if (drive[2] != 0.0f) {
    drive[2] = fminf(fmaxf(YAW_P * yaw_err, -YAW_RATE_ABS), YAW_RATE_ABS);
  }

  float f[OBS_DIM];
  int k = 0;
  f[k++] = drive[0] * CMD_SCALE_X;
  f[k++] = drive[1] * CMD_SCALE_Y;
  f[k++] = drive[2] * CMD_SCALE_YAW;
  f[k++] = HEIGHT_CMD * CMD_SCALE_H;
  for (int j = 0; j < 3; ++j) f[k++] = gyro[env * 3 + j] * ANG_VEL_SCALE;
  for (int j = 0; j < 3; ++j) f[k++] = gravity[env * 3 + j];

  for (int i = 0; i < NUM_DOFS; ++i) {
    const bool arm = i >= 13;
    const float def = i < NUM_ACTIONS ? D_DEFAULT_LEG[i] : 0.0f;
    f[k++] = arm ? 0.0f : motor_q[env * POLICY_NUM_MOTOR + i] - def;
  }
  for (int i = 0; i < NUM_DOFS; ++i) {
    const bool arm = i >= 13;
    f[k++] = arm ? 0.0f : motor_dq[env * POLICY_NUM_MOTOR + i] * DOF_VEL_SCALE;
  }
  const float* la = last_action + env * NUM_ACTIONS;
  for (int i = 0; i < NUM_ACTIONS; ++i) f[k++] = la[i];

  float* o = obs + env * NUM_OBS;
  if (primed) {
    for (int i = 0; i < OBS_DIM * (HISTORY - 1); ++i) o[i] = o[i + OBS_DIM];
    for (int i = 0; i < OBS_DIM; ++i) o[OBS_DIM * (HISTORY - 1) + i] = f[i];
  } else {
    for (int h = 0; h < HISTORY; ++h) {
      for (int i = 0; i < OBS_DIM; ++i) o[h * OBS_DIM + i] = f[i];
    }
  }
}

__global__ void k_homie_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + env * NUM_ACTIONS;
  float* la = last_action + env * NUM_ACTIONS;
  float* qt = q_target + env * POLICY_NUM_MOTOR;

  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    qt[j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    la[i] = fminf(fmaxf(a[i], -ACTION_CLIP), ACTION_CLIP);
    qt[i] = ACTION_SCALE * la[i] + D_DEFAULT_LEG[i];
  }

  for (int j = NUM_ACTIONS; j < OWNED; ++j) qt[j] = 0.0f;
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  int envs = 0;
  bool primed = false;

  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last})
      if (p) cudaFree(p);
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        "policies/homie/model.onnx",
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
    k_homie_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        c.task,
        d_last,
        d_obs,
        envs,
        primed ? 1 : 0
    );
    primed = true;
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_homie_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return OWNED; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "homie"; }
};

}
