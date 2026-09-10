namespace zealot {

constexpr int NUM_ACTIONS = 12;
constexpr int FRAME_V28 = 53;
constexpr int FRAME_V26 = 48;
constexpr int FRAME_MAX = FRAME_V28;
constexpr int HISTORY = 5;

constexpr float CONTROL_DT = 0.02f;
constexpr float ACTION_SCALE = 0.5f;

constexpr float GAIT_PERIOD_SLOW = 0.8f;
constexpr float GAIT_PERIOD_FAST = 0.55f;
constexpr float GAIT_PERIOD_MIN = 0.40f;
constexpr float GAIT_SPEED_CAP = 0.8f;
constexpr float GAIT_STAND_SPEED = 0.1f;

constexpr float VX_MIN = -0.8f, VX_MAX = 0.8f, VY_ABS = 0.3f;
constexpr float YAW_RATE_ABS = 0.6f, SPEED_NORM = 0.8f;

constexpr float FACE_FAR_M = 1.5f, FACE_NEAR_M = 0.4f;
constexpr float WALK_SPEED = 0.5f, WALK_P = 1.5f, YAW_P = 1.2f;

constexpr int OFF_ACT_LAG2 = 0;
constexpr int OFF_CMD = 12;
constexpr int OFF_JPOS = 16;
constexpr int OFF_JVEL = 28;
constexpr int OFF_GRAVITY = 40;
constexpr int OFF_SIN_PHASE = 43;
constexpr int OFF_COS_PHASE = 44;
constexpr int OFF_GYRO = 45;

#define ZEALOT_DEFAULT_POS \
  -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f

#define ZEALOT_Q_MIN                                                       \
  -2.5307f, -0.5236f, -2.7576f, -0.087267f, -0.87267f, -0.2618f, -2.5307f, \
      -2.9671f, -2.7576f, -0.087267f, -0.87267f, -0.2618f

#define ZEALOT_Q_MAX                                                      \
  2.8798f, 2.9671f, 2.7576f, 2.8798f, 0.5236f, 0.2618f, 2.8798f, 0.5236f, \
      2.7576f, 2.8798f, 0.5236f, 0.2618f

__device__ const float D_DEFAULT_POS[NUM_ACTIONS] = {ZEALOT_DEFAULT_POS};
__device__ const float D_Q_MIN[NUM_ACTIONS] = {ZEALOT_Q_MIN};
__device__ const float D_Q_MAX[NUM_ACTIONS] = {ZEALOT_Q_MAX};

const float KPS[NUM_ACTIONS] =
    {100, 100, 100, 200, 40, 40, 100, 100, 100, 200, 40, 40};
const float KDS[NUM_ACTIONS] =
    {2.5f, 2.5f, 2.5f, 5, 2, 2, 2.5f, 2.5f, 2.5f, 5, 2, 2};

const policy_api::Limits LIMITS =
    {VX_MIN, VX_MAX, VY_ABS, YAW_RATE_ABS, SPEED_NORM};

__device__ inline float zealot_wrap(float a) {
  return a - 2.0f * float(M_PI) * rintf(a / (2.0f * float(M_PI)));
}

__device__ inline float zealot_gait_period(float cmd_speed) {
  const float t = (fminf(fabsf(cmd_speed), GAIT_SPEED_CAP) - 0.1f) / 0.4f;
  const float period =
      GAIT_PERIOD_SLOW + (GAIT_PERIOD_FAST - GAIT_PERIOD_SLOW) * fmaxf(t, 0.0f);
  return fmaxf(GAIT_PERIOD_MIN, period);
}

__device__ inline void zealot_shape(
    const float* c,
    const float* t,
    float out[3]
) {
  const float planar = sqrtf(c[0] * c[0] + c[1] * c[1]);
  const bool pos_reached = planar < 1e-9f;
  const bool yaw_reached = fabsf(c[2]) < 1e-9f;

  const float dist = t[0], yaw_err = t[1];

  out[0] = c[0];
  out[1] = c[1];
  out[2] = c[2];
  if (!(dist > 0.0f)) return;

  float bearing = 0.0f, face_w = 0.0f;
  if (!pos_reached) {
    bearing = atan2f(t[3], t[2]);
    face_w = fminf(
        fmaxf((dist - FACE_NEAR_M) / (FACE_FAR_M - FACE_NEAR_M), 0.0f),
        1.0f
    );

    const float gate = fmaxf(cosf(bearing), 0.0f);
    const float speed = gate * fminf(WALK_P * dist, WALK_SPEED);
    out[0] = speed;
    out[1] = fminf(fmaxf(speed * sinf(bearing), -VY_ABS), VY_ABS);
  }

  const float aim =
      face_w > 0.0f
          ? zealot_wrap(yaw_err + face_w * zealot_wrap(bearing - yaw_err))
          : (yaw_reached ? 0.0f : yaw_err);
  out[2] = fminf(fmaxf(YAW_P * aim, -YAW_RATE_ABS), YAW_RATE_ABS);
}

__global__ void k_zealot_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ task,
    const float* __restrict__ act_lag2,
    const float* __restrict__ phase,
    float* __restrict__ prev_q,
    float* __restrict__ drive,
    float* __restrict__ obs,
    int frame,
    int first,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float d[3];
  zealot_shape(cmd + env * 3, task + env * 4, d);
  for (int k = 0; k < 3; ++k) drive[env * 3 + k] = d[k];

  float f[FRAME_MAX];
  for (int i = 0; i < frame; ++i) f[i] = 0.0f;

  for (int i = 0; i < NUM_ACTIONS; ++i) {
    f[OFF_ACT_LAG2 + i] = act_lag2[size_t(env) * NUM_ACTIONS + i];
  }
  f[OFF_CMD + 0] = d[0];
  f[OFF_CMD + 1] = d[1];
  f[OFF_CMD + 2] = d[2];

  float* pq = prev_q + size_t(env) * NUM_ACTIONS;
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    const float q = motor_q[size_t(env) * POLICY_NUM_MOTOR + i];
    f[OFF_JPOS + i] = q - D_DEFAULT_POS[i];
    f[OFF_JVEL + i] = first ? 0.0f : (q - pq[i]) / CONTROL_DT;
    pq[i] = q;
  }

  const float ph = phase[env];
  for (int k = 0; k < 3; ++k) {
    f[OFF_GRAVITY + k] = gravity[env * 3 + k];
    f[OFF_GYRO + k] = gyro[env * 3 + k];
  }
  f[OFF_SIN_PHASE] = sinf(2.0f * float(M_PI) * ph);
  f[OFF_COS_PHASE] = cosf(2.0f * float(M_PI) * ph);

  float* o = obs + size_t(env) * size_t(frame) * HISTORY;
  if (first) {
    for (int h = 0; h < HISTORY; ++h) {
      for (int i = 0; i < frame; ++i) o[h * frame + i] = f[i];
    }
  } else {
    for (int i = 0; i < frame * (HISTORY - 1); ++i) o[i] = o[i + frame];
    for (int i = 0; i < frame; ++i) o[frame * (HISTORY - 1) + i] = f[i];
  }
}

__global__ void k_zealot_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    const float* __restrict__ drive,
    float* __restrict__ act_lag1,
    float* __restrict__ act_lag2,
    float* __restrict__ phase,
    float* __restrict__ q_target,
    int first,
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
    q[i] = fminf(
        fmaxf(D_DEFAULT_POS[i] + ACTION_SCALE * a[i], D_Q_MIN[i]),
        D_Q_MAX[i]
    );
  }

  const float* d = drive + env * 3;
  const float cmd_speed = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);

  if (!first && cmd_speed >= GAIT_STAND_SPEED) {
    phase[env] =
        fmodf(phase[env] + CONTROL_DT / zealot_gait_period(cmd_speed), 1.0f);
  }

  float* l1 = act_lag1 + size_t(env) * NUM_ACTIONS;
  float* l2 = act_lag2 + size_t(env) * NUM_ACTIONS;
  for (int i = 0; i < NUM_ACTIONS; ++i) {
    l2[i] = l1[i];
    l1[i] = a[i];
  }
}

struct Policy : policy_api::Policy {
  virtual int frame() const { return FRAME_V28; }
  virtual const char* onnx() const { return "policies/zealot/model.onnx"; }

  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_lag1 = nullptr,
        *d_lag2 = nullptr;
  float *d_prev_q = nullptr, *d_phase = nullptr, *d_drive = nullptr;
  int envs = 0;
  long long ticks = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_obs,
          (void*)d_act,
          (void*)d_lag1,
          (void*)d_lag2,
          (void*)d_prev_q,
          (void*)d_phase,
          (void*)d_drive}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    const int num_obs = frame() * HISTORY;
    engine = policy_api::engine_make(onnx(), n, num_obs, NUM_ACTIONS);
    auto zeros = [n](float** p, size_t per) {
      cudaMalloc(p, size_t(n) * per * sizeof(float));
      cudaMemset(*p, 0, size_t(n) * per * sizeof(float));
    };
    zeros(&d_obs, size_t(num_obs));
    zeros(&d_act, NUM_ACTIONS);
    zeros(&d_lag1, NUM_ACTIONS);
    zeros(&d_lag2, NUM_ACTIONS);
    zeros(&d_prev_q, NUM_ACTIONS);
    zeros(&d_phase, 1);
    zeros(&d_drive, 3);
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    const int first = ticks == 0 ? 1 : 0;
    ++ticks;
    k_zealot_obs<<<blocks, threads>>>(
        c.motor_q,
        c.gyro,
        c.gravity,
        c.cmd,
        c.task,
        d_lag2,
        d_phase,
        d_prev_q,
        d_drive,
        d_obs,
        frame(),
        first,
        envs
    );
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_zealot_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_drive,
        d_lag1,
        d_lag2,
        d_phase,
        c.q_target,
        first,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_ACTIONS; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "zealot"; }
};

struct V26Policy : Policy {
  int frame() const override { return FRAME_V26; }
  const char* onnx() const override { return "policies/zealot/model_v26.onnx"; }
  const char* name() const override { return "zealot_v26"; }
};

}
