namespace handoff {

constexpr int NUM_ACTIONS = 29;
constexpr int OWNED = 15;
constexpr int WITH_ARMS_OWNED = NUM_ACTIONS;
constexpr int FRAME_DIM = 106;
constexpr int HISTORY_LEN = 11;
constexpr int NUM_OBS = FRAME_DIM * (1 + HISTORY_LEN);

constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float JOINT_VEL_SCALE = 0.05f;
constexpr float GAIT_PERIOD = 1.0f;
constexpr float GAIT_OFFSET = 0.5f;
constexpr float STAND_VEL_THRESHOLD = 0.1f;
constexpr float HEIGHT_CMD = 0.78f;
constexpr float CONTROL_DT = 0.02f;

#define HANDOFF_DEFAULTS                                                   \
  -0.312f, 0.0f, 0.0f, 0.669f, -0.363f, 0.0f, -0.312f, 0.0f, 0.0f, 0.669f, \
      -0.363f, 0.0f, 0.0f, 0.0f, 0.0f, 0.2f, 0.2f, 0.0f, 0.6f, 0.0f, 0.0f, \
      0.0f, 0.2f, -0.2f, 0.0f, 0.6f, 0.0f, 0.0f, 0.0f

#define HANDOFF_ACT_SCALE                                                      \
  0.547546f, 0.350661f, 0.547546f, 0.350661f, 0.438577f, 0.438577f, 0.547546f, \
      0.350661f, 0.547546f, 0.350661f, 0.438577f, 0.438577f, 0.547546f,        \
      0.438577f, 0.438577f, 0.438577f, 0.438577f, 0.438577f, 0.438577f,        \
      0.438577f, 0.074501f, 0.074501f, 0.438577f, 0.438577f, 0.438577f,        \
      0.438577f, 0.438577f, 0.074501f, 0.074501f

#define HANDOFF_HAND_CMD \
  0.04f, 0.23044664f, -0.07842005f, 0.04f, -0.23043664f, -0.07842005f

const float DEFAULTS[NUM_ACTIONS] = {HANDOFF_DEFAULTS};
__device__ const float D_DEFAULTS[NUM_ACTIONS] = {HANDOFF_DEFAULTS};
__device__ const float D_ACT_SCALE[NUM_ACTIONS] = {HANDOFF_ACT_SCALE};
__device__ const float D_HAND_CMD[6] = {HANDOFF_HAND_CMD};

const float KPS[NUM_ACTIONS] = {
    40.1792f, 99.0984f, 40.1792f, 99.0984f, 28.5012f, 28.5012f,
    40.1792f, 99.0984f, 40.1792f, 99.0984f, 28.5012f, 28.5012f,
    40.1792f, 28.5012f, 28.5012f, 14.2506f, 14.2506f, 14.2506f,
    14.2506f, 14.2506f, 16.7783f, 16.7783f, 14.2506f, 14.2506f,
    14.2506f, 14.2506f, 14.2506f, 16.7783f, 16.7783f
};
const float KDS[NUM_ACTIONS] = {
    2.55789f, 6.30880f, 2.55789f, 6.30880f, 1.81445f, 1.81445f,
    2.55789f, 6.30880f, 2.55789f, 6.30880f, 1.81445f, 1.81445f,
    2.55789f, 1.81445f, 1.81445f, 0.90722f, 0.90722f, 0.90722f,
    0.90722f, 0.90722f, 1.06814f, 1.06814f, 0.90722f, 0.90722f,
    0.90722f, 0.90722f, 0.90722f, 1.06814f, 1.06814f
};

const policy_api::Limits LIMITS = {-1.0, 1.0, 1.0, 1.0, 0.0};

__device__ inline bool ankle_dof(int j) {
  return j == 4 || j == 5 || j == 10 || j == 11;
}

__global__ void k_handoff_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float* __restrict__ history,
    float* __restrict__ obs,
    int envs,
    float phase,
    int primed
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float f[FRAME_DIM];
  const float* c = cmd + env * 3;
  const float norm = sqrtf(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
  const bool stationary = norm < STAND_VEL_THRESHOLD;

  f[0] = stationary ? 0.0f : c[0];
  f[1] = stationary ? 0.0f : c[1];
  f[2] = HEIGHT_CMD;
  f[3] = stationary ? 0.0f : c[2];
  for (int k = 0; k < 6; ++k) f[4 + k] = D_HAND_CMD[k];

  if (stationary) {
    f[10] = 0.0f;
    f[11] = 1.0f;
    f[12] = 0.0f;
    f[13] = 1.0f;
  } else {
    const float left = phase;
    const float right = fmodf(phase + GAIT_OFFSET, 1.0f);
    f[10] = sinf(2.0f * float(M_PI) * left);
    f[11] = cosf(2.0f * float(M_PI) * left);
    f[12] = sinf(2.0f * float(M_PI) * right);
    f[13] = cosf(2.0f * float(M_PI) * right);
  }

  for (int k = 0; k < 3; ++k) f[14 + k] = gyro[env * 3 + k] * ANG_VEL_SCALE;

  const float gx = gravity[env * 3 + 0];
  const float gy = gravity[env * 3 + 1];
  const float gz = gravity[env * 3 + 2];
  f[17] = atan2f(-gy, -gz);
  f[18] = asinf(fminf(fmaxf(gx, -1.0f), 1.0f));

  const float* la = last_action + env * NUM_ACTIONS;
  for (int j = 0; j < NUM_ACTIONS; ++j) {
    f[19 + j] = motor_q[env * POLICY_NUM_MOTOR + j] - D_DEFAULTS[j];
    f[48 + j] = ankle_dof(j)
                    ? 0.0f
                    : motor_dq[env * POLICY_NUM_MOTOR + j] * JOINT_VEL_SCALE;
    f[77 + j] = la[j];
  }

  float* h = history + env * HISTORY_LEN * FRAME_DIM;
  if (primed) {
    for (int t = 0; t + 1 < HISTORY_LEN; ++t) {
      for (int k = 0; k < FRAME_DIM; ++k) {
        h[t * FRAME_DIM + k] = h[(t + 1) * FRAME_DIM + k];
      }
    }
    for (int k = 0; k < FRAME_DIM; ++k)
      h[(HISTORY_LEN - 1) * FRAME_DIM + k] = f[k];
  } else {
    for (int t = 0; t < HISTORY_LEN; ++t) {
      for (int k = 0; k < FRAME_DIM; ++k) h[t * FRAME_DIM + k] = f[k];
    }
  }

  float* o = obs + env * NUM_OBS;
  for (int k = 0; k < FRAME_DIM; ++k) o[k] = f[k];
  for (int t = 0; t < HISTORY_LEN; ++t) {
    for (int k = 0; k < FRAME_DIM; ++k)
      o[FRAME_DIM * (1 + t) + k] = h[t * FRAME_DIM + k];
  }
}

__global__ void k_handoff_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int owned_end,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + env * NUM_ACTIONS;
  float* la = last_action + env * NUM_ACTIONS;
  float* qt = q_target + env * POLICY_NUM_MOTOR;

  for (int j = 0; j < NUM_ACTIONS; ++j) la[j] = a[j];
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j)
    qt[j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  for (int j = 0; j < owned_end; ++j)
    qt[j] = D_DEFAULTS[j] + a[j] * D_ACT_SCALE[j];
}

const char* const MODEL_SRC = "policies/handoff/model.onnx";
const char* const MODEL_DYNAMIC = "build/trt/handoff_dynamic_batch.onnx";

std::string handoff_i64(long long v) {
  std::string s;
  const unsigned long long u = static_cast<unsigned long long>(v);
  for (int k = 0; k < 8; ++k) s.push_back(char((u >> (8 * k)) & 0xff));
  return s;
}

void handoff_open_shape(
    std::string& blob,
    const std::string& pat,
    const char* what
) {
  const size_t p = blob.find(pat);
  if (p == std::string::npos ||
      blob.find(pat, p + pat.size()) != std::string::npos) {
    throw std::runtime_error(
        "handoff: model.onnx has no single " + std::string(what) +
        " reshape target"
    );
  }
  const std::string lead = handoff_i64(-1);
  for (size_t k = 0; k < 8; ++k) blob[p + 2 + k] = lead[k];
}

void handoff_open_dim(
    std::string& blob,
    const std::string& name
) {
  std::string key;
  key.push_back(char(0x0a));
  key.push_back(char(name.size()));
  key += name;
  int hits = 0;
  for (size_t p = blob.find(key); p != std::string::npos;
       p = blob.find(key, p + key.size())) {
    const size_t q = p + key.size();
    if (q + 12 > blob.size()) continue;
    const unsigned char* b =
        reinterpret_cast<const unsigned char*>(blob.data()) + q;

    if (b[0] != 0x12 || b[2] != 0x0a || b[4] != 0x08 || b[6] != 0x12 ||
        b[8] != 0x0a || b[9] != 0x02 || b[10] != 0x08 || b[11] != 0x01) {
      continue;
    }
    blob[q + 10] = 0x1a;
    blob[q + 11] = 0x00;
    ++hits;
  }
  if (hits != 1) {
    throw std::runtime_error(
        "handoff: model.onnx does not declare '" + name + "' as one batched row"
    );
  }
}

std::string handoff_dynamic_model() {
  namespace fs = std::filesystem;
  if (fs::exists(MODEL_DYNAMIC) && fs::exists(MODEL_SRC) &&
      fs::last_write_time(MODEL_DYNAMIC) >= fs::last_write_time(MODEL_SRC)) {
    return MODEL_DYNAMIC;
  }

  std::ifstream in(MODEL_SRC, std::ios::binary);
  if (!in)
    throw std::runtime_error("handoff: cannot read " + std::string(MODEL_SRC));
  std::string blob(
      (std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>()
  );
  const size_t before = blob.size();

  const std::string tag_raw(1, char(0x4a));
  handoff_open_shape(
      blob,
      tag_raw + char(0x10) + handoff_i64(1) + handoff_i64(FRAME_DIM),
      "[1, 106]"
  );
  handoff_open_shape(
      blob,
      tag_raw + char(0x18) + handoff_i64(1) + handoff_i64(HISTORY_LEN) +
          handoff_i64(FRAME_DIM),
      "[1, 11, 106]"
  );
  handoff_open_dim(blob, "obs");
  handoff_open_dim(blob, "actions");

  if (blob.size() != before)
    throw std::runtime_error("handoff: the rewrite moved a byte");

  fs::create_directories("build/trt");
  const std::string tmp =
      std::string(MODEL_DYNAMIC) + "." +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()
      );
  {
    std::ofstream out(tmp, std::ios::binary);
    out.write(blob.data(), std::streamsize(blob.size()));
    if (!out) throw std::runtime_error("handoff: cannot write " + tmp);
  }
  fs::rename(tmp, MODEL_DYNAMIC);
  return MODEL_DYNAMIC;
}

struct Base : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr,
        *d_hist = nullptr;
  int envs = 0;
  long step_index = 0;
  bool primed = false;
  int owned_end = OWNED;

  explicit Base(int end) : owned_end(end) {}

  ~Base() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last, (void*)d_hist}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        handoff_dynamic_model(),
        n,
        NUM_OBS,
        NUM_ACTIONS
    );
    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_hist, size_t(n) * HISTORY_LEN * FRAME_DIM * sizeof(float));
    cudaMemset(d_hist, 0, size_t(n) * HISTORY_LEN * FRAME_DIM * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_ACTIONS * sizeof(float));
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    const float phase = float(
        fmod(double(step_index) * double(CONTROL_DT) / double(GAIT_PERIOD), 1.0)
    );
    k_handoff_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_hist,
        d_obs,
        envs,
        phase,
        primed ? 1 : 0
    );

    step_index += 1;
    primed = true;
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_handoff_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        owned_end,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return owned_end; }
  policy_api::Limits limits() const override { return LIMITS; }
};

struct Policy : Base {
  Policy() : Base(OWNED) {}
  const char* name() const override { return "handoff"; }
};

struct WithArmsPolicy : Base {
  WithArmsPolicy() : Base(WITH_ARMS_OWNED) {}
  const char* name() const override { return "handoff_with_arms"; }
};

}
