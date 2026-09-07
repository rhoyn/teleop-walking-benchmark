namespace amo {

constexpr int NUM_DOF = 23;
constexpr int NUM_ACT = 15;
constexpr int NUM_ARM = 8;
constexpr int HISTORY = 10;
constexpr int EXTRA_HISTORY = 25;
constexpr int N_PROPRIO = 3 + 2 + 2 + NUM_DOF * 3 + 2 + NUM_ACT;
constexpr int N_PRIV = 3;
constexpr int N_DEMO = NUM_ARM + 3 + 3 + 3;
constexpr int NUM_OBS = N_PROPRIO + N_DEMO + N_PRIV + HISTORY * N_PROPRIO;
constexpr int NUM_EXTRA = EXTRA_HISTORY * N_PROPRIO;
constexpr int ADAPTER_IN = 4 + NUM_ARM;
constexpr int HIST_BASE = N_PROPRIO + N_DEMO + N_PRIV;

constexpr float ACTION_SCALE = 0.25f;
constexpr float ACTION_CLIP = 40.0f;
constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float DOF_VEL_SCALE = 0.05f;
constexpr float GAIT_FREQ = 1.3f;
constexpr float CONTROL_DT = 0.02f;
constexpr float TORSO_HEIGHT = 0.75f;
constexpr float TORSO_YAW = 0.0f;
constexpr float TORSO_PITCH = 0.0f;
constexpr float TORSO_ROLL = 0.0f;
constexpr float STAND_SPEED = 0.1f;
constexpr float MAX_VX = 0.5f;
constexpr float MAX_VY = 0.4f;

#define AMO_DEFAULTS                                                          \
  -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, \
      0.0f, 0.0f, 0.0f, 0.5f, 0.0f, 0.2f, 0.3f, 0.5f, 0.0f, -0.2f, 0.3f

#define AMO_DOF_TO_MOTOR                                                    \
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 22, 23, \
      24, 25

#define AMO_ADAPTER_IN_MEAN                                                 \
  0.6493941816443938f, 0.00835603437027632f, 0.5198069391833688f,           \
      -0.0028908478957657816f, -0.18861694226437198f, 0.41743932825218666f, \
      -0.29657144144767367f, 0.6869066831118564f, -0.20802372898486796f,    \
      -0.4032497266650522f, 0.2749623237878731f, 0.7164985165359848f

#define AMO_ADAPTER_IN_STD                                            \
  0.08622848682483915f, 0.905570059110108f, 0.5990661102951124f,      \
      0.4037710235670824f, 0.47366042735651465f, 0.4329358814650013f, \
      0.35596660246191353f, 0.56022898770754f, 0.49676887697830113f,  \
      0.4662358545315813f, 0.32249961752507345f, 0.5674630786438484f

#define AMO_ADAPTER_OUT_MEAN                                              \
  -0.8649367448034146f, -0.014551485616159491f, -0.014272786461742208f,   \
      1.0845364423583947f, -0.44747724731672117f, -0.015654007296027932f, \
      -0.8640702505124579f, 0.01381745330674642f, 0.009187855249187705f,  \
      1.0846158293741608f, -0.4484803223536554f, 0.013866172339457972f,   \
      0.005789055823254164f, 0.00010137134921915922f, 0.19727236245602361f

#define AMO_ADAPTER_OUT_STD                                            \
  0.3966769438866927f, 0.18091131746908673f, 0.1964029137827882f,      \
      0.5146759444113571f, 0.2880384543250629f, 0.07141340497610606f,  \
      0.39789043951513176f, 0.1814397591948851f, 0.19595788541221598f, \
      0.5149199082108712f, 0.28739879629613463f, 0.07155134830032818f, \
      0.7115220538732581f, 0.3607543644013931f, 0.32624274557512967f

__device__ const float D_DEFAULTS[NUM_DOF] = {AMO_DEFAULTS};
__device__ const int D_DOF_TO_MOTOR[NUM_DOF] = {AMO_DOF_TO_MOTOR};
__device__ const float D_AIN_MEAN[ADAPTER_IN] = {AMO_ADAPTER_IN_MEAN};
__device__ const float D_AIN_STD[ADAPTER_IN] = {AMO_ADAPTER_IN_STD};
__device__ const float D_AOUT_MEAN[NUM_ACT] = {AMO_ADAPTER_OUT_MEAN};
__device__ const float D_AOUT_STD[NUM_ACT] = {AMO_ADAPTER_OUT_STD};

const float KPS[NUM_ACT] =
    {150, 150, 150, 300, 80, 20, 150, 150, 150, 300, 80, 20, 400, 400, 400};
const float KDS[NUM_ACT] = {2, 2, 2, 4, 2, 1, 2, 2, 2, 4, 2, 1, 15, 15, 15};

const policy_api::Limits LIMITS =
    {-MAX_VX, MAX_VX, MAX_VY, 0.0, 0.0, 0.08, 0.15};

__global__ void k_amo_adapter_in(
    const float* __restrict__ motor_q,
    float* __restrict__ ain,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* a = ain + env * ADAPTER_IN;
  a[0] = TORSO_HEIGHT;
  a[1] = TORSO_YAW;
  a[2] = TORSO_PITCH;
  a[3] = TORSO_ROLL;
  for (int i = 0; i < NUM_ARM; ++i) {
    a[4 + i] = motor_q[env * POLICY_NUM_MOTOR + D_DOF_TO_MOTOR[NUM_ACT + i]];
  }
  for (int i = 0; i < ADAPTER_IN; ++i) {
    a[i] = (a[i] - D_AIN_MEAN[i]) / (D_AIN_STD[i] + 1e-8f);
  }
}

__global__ void k_amo_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ task,
    const float* __restrict__ adapter_raw,
    const float* __restrict__ last_action,
    const float* __restrict__ gait,
    const unsigned char* __restrict__ in_place,
    float* __restrict__ hist,
    float* __restrict__ obs,
    int envs,
    int primed
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* o = obs + env * NUM_OBS;
  float* h = hist + env * NUM_EXTRA;
  const float* mq = motor_q + env * POLICY_NUM_MOTOR;
  const float* mdq = motor_dq + env * POLICY_NUM_MOTOR;
  const float* c = cmd + env * 3;

  const float dist = task[env * 4 + 0];
  const float yaw_err = task[env * 4 + 1];
  const bool has_target = dist > 0.0f;
  const float dyaw = (in_place[env] != 0 || !has_target) ? 0.0f : -yaw_err;

  int k = 0;
  for (int i = 0; i < 3; ++i) o[k++] = gyro[env * 3 + i] * ANG_VEL_SCALE;

  const float gx = gravity[env * 3 + 0];
  const float gy = gravity[env * 3 + 1];
  const float gz = gravity[env * 3 + 2];
  o[k++] = atan2f(-gy, -gz);
  o[k++] = asinf(fminf(fmaxf(gx, -1.0f), 1.0f));
  o[k++] = sinf(dyaw);
  o[k++] = cosf(dyaw);
  for (int i = 0; i < NUM_DOF; ++i) {
    o[k++] = mq[D_DOF_TO_MOTOR[i]] - D_DEFAULTS[i];
  }
  for (int i = 0; i < NUM_DOF; ++i) {
    o[k++] = mdq[D_DOF_TO_MOTOR[i]] * DOF_VEL_SCALE;
  }
  for (int i = 0; i < NUM_DOF; ++i) o[k++] = last_action[env * NUM_DOF + i];
  o[k++] = sinf(gait[env * 2 + 0] * 2.0f * float(M_PI));
  o[k++] = sinf(gait[env * 2 + 1] * 2.0f * float(M_PI));
  for (int i = 0; i < NUM_ACT; ++i) {
    o[k++] = adapter_raw[env * NUM_ACT + i] * D_AOUT_STD[i] + D_AOUT_MEAN[i];
  }

  if (primed == 0) {
    for (int j = 0; j < EXTRA_HISTORY; ++j) {
      for (int i = 0; i < N_PROPRIO; ++i) h[j * N_PROPRIO + i] = o[i];
    }
  }

  for (int i = 0; i < NUM_ARM; ++i) {
    o[k++] = mq[D_DOF_TO_MOTOR[NUM_ACT + i]];
  }
  o[k++] = c[0];
  o[k++] = c[1];
  o[k++] = 0.0f;
  o[k++] = TORSO_YAW;
  o[k++] = TORSO_PITCH;
  o[k++] = TORSO_ROLL;
  o[k++] = TORSO_HEIGHT;
  o[k++] = TORSO_HEIGHT;
  o[k++] = TORSO_HEIGHT;
  for (int i = 0; i < N_PRIV; ++i) o[k++] = 0.0f;

  for (int j = 0; j < HISTORY; ++j) {
    const float* f = h + (EXTRA_HISTORY - HISTORY + j) * N_PROPRIO;
    for (int i = 0; i < N_PROPRIO; ++i) o[HIST_BASE + j * N_PROPRIO + i] = f[i];
  }

  for (int j = 0; j < EXTRA_HISTORY - 1; ++j) {
    for (int i = 0; i < N_PROPRIO; ++i) {
      h[j * N_PROPRIO + i] = h[(j + 1) * N_PROPRIO + i];
    }
  }
  for (int i = 0; i < N_PROPRIO; ++i) {
    h[(EXTRA_HISTORY - 1) * N_PROPRIO + i] = o[i];
  }
}

__global__ void k_amo_act(
    const float* __restrict__ raw,
    const float* __restrict__ motor_q,
    const float* __restrict__ cmd,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ gait,
    unsigned char* __restrict__ in_place,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = raw + env * NUM_ACT;
  const float* mq = motor_q + env * POLICY_NUM_MOTOR;
  float* la = last_action + env * NUM_DOF;

  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  }
  for (int i = 0; i < NUM_ACT; ++i) {
    const float act = fminf(fmaxf(a[i], -ACTION_CLIP), ACTION_CLIP);
    la[i] = act;
    q_target[env * POLICY_NUM_MOTOR + i] = D_DEFAULTS[i] + act * ACTION_SCALE;
  }
  for (int i = 0; i < NUM_ARM; ++i) {
    const int d = NUM_ACT + i;
    la[d] = (mq[D_DOF_TO_MOTOR[d]] - D_DEFAULTS[d]) / ACTION_SCALE;
  }

  const float* c = cmd + env * 3;
  const bool stand = sqrtf(c[0] * c[0] + c[1] * c[1]) < STAND_SPEED;
  in_place[env] = stand ? 1 : 0;

  float g0 = fmodf(gait[env * 2 + 0] + CONTROL_DT * GAIT_FREQ, 1.0f);
  float g1 = fmodf(gait[env * 2 + 1] + CONTROL_DT * GAIT_FREQ, 1.0f);
  const bool near0 = fabsf(g0 - 0.25f) < 0.05f;
  const bool near1 = fabsf(g1 - 0.25f) < 0.05f;
  if (stand && (near0 || near1)) {
    g0 = 0.25f;
    g1 = 0.25f;
  }
  if (!stand && near0 && near1) {
    g0 = 0.25f;
    g1 = 0.75f;
  }
  gait[env * 2 + 0] = g0;
  gait[env * 2 + 1] = g1;
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> adapter, walk;
  float *d_ain = nullptr, *d_araw = nullptr;
  float *d_obs = nullptr, *d_hist = nullptr, *d_act = nullptr;
  float *d_last = nullptr, *d_gait = nullptr;
  unsigned char* d_in_place = nullptr;
  int envs = 0;
  int primed = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_ain,
          (void*)d_araw,
          (void*)d_obs,
          (void*)d_hist,
          (void*)d_act,
          (void*)d_last,
          (void*)d_gait,
          (void*)d_in_place}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    adapter = policy_api::engine_make(
        "policies/amo/model_adapter.onnx",
        n,
        ADAPTER_IN,
        NUM_ACT
    );

    walk = policy_api::engine_make(
        "policies/amo/model.onnx",
        1,
        {{"obs_teacher", {1, NUM_OBS}}, {"extra_hist", {1, NUM_EXTRA}}},
        {{"output", {1, NUM_ACT}}}
    );

    cudaMalloc(&d_ain, size_t(n) * ADAPTER_IN * sizeof(float));
    cudaMalloc(&d_araw, size_t(n) * NUM_ACT * sizeof(float));
    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_hist, size_t(n) * NUM_EXTRA * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_ACT * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_DOF * sizeof(float));
    cudaMalloc(&d_gait, size_t(n) * 2 * sizeof(float));
    cudaMalloc(&d_in_place, size_t(n) * sizeof(unsigned char));

    cudaMemset(d_obs, 0, size_t(n) * NUM_OBS * sizeof(float));
    cudaMemset(d_hist, 0, size_t(n) * NUM_EXTRA * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_DOF * sizeof(float));

    cudaMemset(d_in_place, 1, size_t(n));
    std::vector<float> gait(size_t(n) * 2, 0.25f);
    cudaMemcpy(
        d_gait,
        gait.data(),
        gait.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;

    k_amo_adapter_in<<<blocks, threads>>>(c.motor_q, d_ain, envs);
    policy_api::engine_run(*adapter, d_ain, d_araw, envs);

    k_amo_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        c.task,
        d_araw,
        d_last,
        d_gait,
        d_in_place,
        d_hist,
        d_obs,
        envs,
        primed
    );
    primed = 1;

    for (int e = 0; e < envs; ++e) {
      const float* in[2] = {
          d_obs + size_t(e) * NUM_OBS,
          d_hist + size_t(e) * NUM_EXTRA
      };
      float* out[1] = {d_act + size_t(e) * NUM_ACT};
      policy_api::engine_run(*walk, in, out, 1);
    }

    k_amo_act<<<blocks, threads>>>(
        d_act,
        c.motor_q,
        c.cmd,
        c.arm_pose,
        d_last,
        d_gait,
        d_in_place,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_ACT; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "amo"; }
};

}
