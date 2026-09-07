namespace bfm_zero {

constexpr int NUM_JOINTS = 29;
constexpr int NUM_OWNED = 15;
constexpr int HISTORY = 4;
constexpr int Z_DIM = 256;
constexpr int NUM_OBS = 465;
constexpr int NUM_INPUT = NUM_OBS + Z_DIM;
constexpr int NUM_LATENTS = 12;

constexpr int OFF_QPOS = 0;
constexpr int OFF_QVEL = 29;
constexpr int OFF_GRAV = 58;
constexpr int OFF_ANG = 61;
constexpr int OFF_LAST_ACTION = 64;
constexpr int OFF_HISTORY = 93;
constexpr int HIST_ACT = 0;
constexpr int HIST_ANG = 116;
constexpr int HIST_QPOS = 128;
constexpr int HIST_QVEL = 244;
constexpr int HIST_GRAV = 360;

constexpr float ANG_VEL_SCALE = 0.25f;
constexpr float ACTION_RESCALE = 5.0f;
constexpr float ACTION_CLIP = 5.0f;

constexpr int DWELL_STEPS = 3;
constexpr float YAW_WEIGHT = 0.2f;

struct LatentSpec {
  const char* name;
  float v[3];
};

const LatentSpec LATENTS[NUM_LATENTS] = {
    {"move-ego-0-0", {0.001f, -0.002f, -0.010f}},
    {"move-ego-0-0.3", {0.452f, 0.221f, 0.353f}},
    {"move-ego-0-0.7", {1.280f, -0.128f, 0.087f}},
    {"move-ego-90-0.3", {0.428f, -0.026f, -0.466f}},
    {"move-ego-90-0.7", {-0.217f, 1.617f, 0.058f}},
    {"move-ego-180-0.3", {-1.474f, 0.077f, 0.084f}},
    {"move-ego--90-0.3", {-0.380f, -1.076f, 0.005f}},
    {"move-ego--90-0.7", {-0.300f, -1.682f, -0.223f}},
    {"rotate-z-5-0.5", {0.077f, 0.137f, 4.091f}},
    {"rotate-z--5-0.5", {0.096f, -0.151f, -3.868f}},
    {"move-ego-low0.5-0-0", {0.053f, 0.157f, -0.319f}},
    {"move-ego-low0.6-0-0.7", {0.740f, -0.122f, 0.201f}}
};

#define BFM_ZERO_DEFAULTS                                                     \
  -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, -0.1f, 0.0f, 0.0f, 0.3f, -0.2f, 0.0f, \
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, \
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f

#define BFM_ZERO_ACTION_SCALE                                              \
  0.222001498914f, 0.22200157f, 0.54754699f, 0.35066156f, 0.43857802f,     \
      0.43857802f, 0.222001498914f, 0.22200157f, 0.54754699f, 0.35066156f, \
      0.43857802f, 0.43857802f, 0.54754699f, 0.43857802f, 0.43857802f,     \
      0.43857802f, 0.43857802f, 0.43857802f, 0.43857802f, 0.43857802f,     \
      0.07450086f, 0.07466888f, 0.43857802f, 0.43857802f, 0.43857802f,     \
      0.43857802f, 0.43857802f, 0.07450086f, 0.07450086f

__device__ const float D_DEFAULTS[NUM_JOINTS] = {BFM_ZERO_DEFAULTS};
__device__ const float D_ACTION_SCALE[NUM_JOINTS] = {BFM_ZERO_ACTION_SCALE};

const float KPS[NUM_JOINTS] = {
    99.09843f, 99.0984f,  40.1792f,  99.0984f,  28.5012f, 28.5012f,
    99.09843f, 99.0984f,  40.1792f,  99.0984f,  28.5012f, 28.5012f,
    40.1792f,  28.5012f,  28.5012f,  14.2506f,  14.2506f, 14.2506f,
    14.2506f,  14.25062f, 16.77833f, 16.77833f, 14.2506f, 14.2506f,
    14.2506f,  14.2506f,  14.25062f, 16.77833f, 16.77833f
};
const float KDS[NUM_JOINTS] = {6.3088f, 6.3088f, 2.5579f, 6.3088f, 1.8145f,
                               1.8145f, 6.3088f, 6.3088f, 2.5579f, 6.3088f,
                               1.8145f, 1.8145f, 2.5579f, 1.8145f, 1.8145f,
                               0.9072f, 0.9072f, 0.9072f, 0.9072f, 0.9072f,
                               1.0681f, 1.0681f, 0.9072f, 0.9072f, 0.9072f,
                               0.9072f, 0.9072f, 1.0681f, 1.0681f};

const policy_api::Limits LIMITS = {-0.7, 0.7, 0.7, 2.5, 0.7};

__device__ inline int latent_for_command(
    const float* __restrict__ latent_v,
    float vx,
    float vy,
    float wz
) {
  int best = 0;
  float best_cost = 3.0e38f;
  for (int k = 0; k < NUM_LATENTS; ++k) {
    const float ex = latent_v[k * 3 + 0] - vx;
    const float ey = latent_v[k * 3 + 1] - vy;
    const float ew = latent_v[k * 3 + 2] - wz;
    const float cost = ex * ex + ey * ey + YAW_WEIGHT * ew * ew;
    if (cost < best_cost) {
      best_cost = cost;
      best = k;
    }
  }
  return best;
}

__global__ void k_bfm_zero_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ latents,
    const float* __restrict__ latent_v,
    const float* __restrict__ last_action,
    float* __restrict__ hist,
    int* __restrict__ latent,
    int* __restrict__ pending,
    int* __restrict__ dwell,
    float* __restrict__ obs,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* c = cmd + env * 3;
  const int want = latent_for_command(latent_v, c[0], c[1], c[2]);
  if (want == latent[env] || want != pending[env]) {
    pending[env] = want;
    dwell[env] = 0;
  } else if (++dwell[env] >= DWELL_STEPS) {
    latent[env] = want;
    dwell[env] = 0;
  }

  float* o = obs + env * NUM_INPUT;
  const float* q = motor_q + env * NUM_JOINTS;
  const float* dq = motor_dq + env * NUM_JOINTS;
  const float* la = last_action + env * NUM_JOINTS;

  float* h_act = hist + env * (3 * HISTORY * NUM_JOINTS + 2 * HISTORY * 3);
  float* h_qpos = h_act + HISTORY * NUM_JOINTS;
  float* h_qvel = h_qpos + HISTORY * NUM_JOINTS;
  float* h_ang = h_qvel + HISTORY * NUM_JOINTS;
  float* h_grav = h_ang + HISTORY * 3;

  float qpos[NUM_JOINTS], qvel[NUM_JOINTS], grav[3], ang[3];
  for (int j = 0; j < NUM_JOINTS; ++j) {
    qpos[j] = q[j] - D_DEFAULTS[j];
    qvel[j] = dq[j];
  }
  for (int k = 0; k < 3; ++k) {
    grav[k] = gravity[env * 3 + k];
    ang[k] = gyro[env * 3 + k] * ANG_VEL_SCALE;
  }

  for (int j = 0; j < NUM_JOINTS; ++j) {
    o[OFF_QPOS + j] = qpos[j];
    o[OFF_QVEL + j] = qvel[j];
    o[OFF_LAST_ACTION + j] = la[j];
  }
  for (int k = 0; k < 3; ++k) {
    o[OFF_GRAV + k] = grav[k];
    o[OFF_ANG + k] = ang[k];
  }
  for (int i = 0; i < HISTORY * NUM_JOINTS; ++i) {
    o[OFF_HISTORY + HIST_ACT + i] = h_act[i];
    o[OFF_HISTORY + HIST_QPOS + i] = h_qpos[i];
    o[OFF_HISTORY + HIST_QVEL + i] = h_qvel[i];
  }
  for (int i = 0; i < HISTORY * 3; ++i) {
    o[OFF_HISTORY + HIST_ANG + i] = h_ang[i];
    o[OFF_HISTORY + HIST_GRAV + i] = h_grav[i];
  }
  const float* z = latents + latent[env] * Z_DIM;
  for (int i = 0; i < Z_DIM; ++i) o[NUM_OBS + i] = z[i];

  for (int i = HISTORY * NUM_JOINTS - NUM_JOINTS - 1; i >= 0; --i) {
    h_act[i + NUM_JOINTS] = h_act[i];
    h_qpos[i + NUM_JOINTS] = h_qpos[i];
    h_qvel[i + NUM_JOINTS] = h_qvel[i];
  }
  for (int i = HISTORY * 3 - 3 - 1; i >= 0; --i) {
    h_ang[i + 3] = h_ang[i];
    h_grav[i + 3] = h_grav[i];
  }
  for (int j = 0; j < NUM_JOINTS; ++j) {
    h_act[j] = la[j];
    h_qpos[j] = qpos[j];
    h_qvel[j] = qvel[j];
  }
  for (int k = 0; k < 3; ++k) {
    h_ang[k] = ang[k];
    h_grav[k] = grav[k];
  }
}

__global__ void k_bfm_zero_act(
    const float* __restrict__ action,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = action + env * NUM_JOINTS;
  float* la = last_action + env * NUM_JOINTS;
  for (int j = 0; j < NUM_JOINTS; ++j) {
    la[j] = fminf(fmaxf(ACTION_RESCALE * a[j], -ACTION_CLIP), ACTION_CLIP);
  }
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  }
  for (int j = 0; j < NUM_OWNED; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] =
        D_DEFAULTS[j] + la[j] * D_ACTION_SCALE[j];
  }
}

const char* const MODEL_SRC = "policies/bfm_zero/model.onnx";
const char* const MODEL_DYNAMIC = "build/trt/bfm_zero_dynamic_batch.onnx";

std::string bfm_zero_bytes(std::initializer_list<int> v) {
  std::string s;
  for (int b : v) s.push_back(char(b));
  return s;
}

std::string bfm_zero_dynamic_model() {
  namespace fs = std::filesystem;
  if (fs::exists(MODEL_DYNAMIC) && fs::exists(MODEL_SRC) &&
      fs::last_write_time(MODEL_DYNAMIC) >= fs::last_write_time(MODEL_SRC)) {
    return MODEL_DYNAMIC;
  }

  std::ifstream in(MODEL_SRC, std::ios::binary);
  if (!in)
    throw std::runtime_error("bfm_zero: cannot read " + std::string(MODEL_SRC));
  std::string blob(
      (std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>()
  );

  const std::string name = "actor_obs";
  const std::string key = bfm_zero_bytes({0x0a, int(name.size())}) + name;
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
        "bfm_zero: model.onnx does not declare 'actor_obs' as one batched row"
    );
  }

  fs::create_directories("build/trt");
  const std::string tmp =
      std::string(MODEL_DYNAMIC) + "." +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()
      );
  {
    std::ofstream out(tmp, std::ios::binary);
    out.write(blob.data(), std::streamsize(blob.size()));
    if (!out) throw std::runtime_error("bfm_zero: cannot write " + tmp);
  }
  fs::rename(tmp, MODEL_DYNAMIC);
  return MODEL_DYNAMIC;
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_obs = nullptr, *d_act = nullptr, *d_last = nullptr;
  float *d_hist = nullptr, *d_latents = nullptr, *d_latent_v = nullptr;
  int *d_latent = nullptr, *d_pending = nullptr, *d_dwell = nullptr;
  int envs = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_obs,
          (void*)d_act,
          (void*)d_last,
          (void*)d_hist,
          (void*)d_latents,
          (void*)d_latent_v,
          (void*)d_latent,
          (void*)d_pending,
          (void*)d_dwell}) {
      if (p) cudaFree(p);
    }
  }

  static std::vector<float> latents_load(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("bfm_zero: cannot open " + path);

    std::map<std::string, std::vector<float>> found;
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty()) continue;
      std::istringstream row(line);
      std::string name, cell;
      if (!std::getline(row, name, ',')) continue;
      if (name == "name") continue;
      std::vector<float> z;
      while (std::getline(row, cell, ',')) {
        const char* start = cell.c_str();
        char* end = nullptr;
        const float v = std::strtof(start, &end);
        if (end == start) {
          throw std::runtime_error("bfm_zero: " + path + " has a bad number");
        }
        if (!std::isfinite(v)) {
          throw std::runtime_error(
              "bfm_zero: the latent file holds a non-finite value"
          );
        }
        z.push_back(v);
      }
      if (z.size() != size_t(Z_DIM)) {
        throw std::runtime_error(
            "bfm_zero: latent '" + name + "' is " + std::to_string(z.size()) +
            " long, expected " + std::to_string(Z_DIM)
        );
      }
      found[name] = z;
    }

    std::vector<float> out;
    out.reserve(size_t(NUM_LATENTS) * Z_DIM);
    for (const LatentSpec& spec : LATENTS) {
      const auto it = found.find(spec.name);
      if (it == found.end()) {
        throw std::runtime_error(
            "bfm_zero: " + path + " has no latent named '" +
            std::string(spec.name) + "'"
        );
      }
      out.insert(out.end(), it->second.begin(), it->second.end());
    }
    return out;
  }

  void init(int n) override {
    envs = n;
    engine = policy_api::engine_make(
        bfm_zero_dynamic_model(),
        n,
        NUM_INPUT,
        NUM_JOINTS
    );
    const std::vector<float> latents =
        latents_load("policies/bfm_zero/latents.csv");

    const size_t hist_row = 3 * HISTORY * NUM_JOINTS + 2 * HISTORY * 3;
    cudaMalloc(&d_obs, size_t(n) * NUM_INPUT * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * NUM_INPUT * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_JOINTS * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_JOINTS * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_JOINTS * sizeof(float));
    cudaMalloc(&d_hist, size_t(n) * hist_row * sizeof(float));
    cudaMemset(d_hist, 0, size_t(n) * hist_row * sizeof(float));
    cudaMalloc(&d_latents, latents.size() * sizeof(float));
    cudaMemcpy(
        d_latents,
        latents.data(),
        latents.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );
    float host_v[NUM_LATENTS * 3];
    for (int k = 0; k < NUM_LATENTS; ++k) {
      for (int j = 0; j < 3; ++j) host_v[k * 3 + j] = LATENTS[k].v[j];
    }
    cudaMalloc(&d_latent_v, sizeof(host_v));
    cudaMemcpy(d_latent_v, host_v, sizeof(host_v), cudaMemcpyHostToDevice);
    cudaMalloc(&d_latent, size_t(n) * sizeof(int));
    cudaMemset(d_latent, 0, size_t(n) * sizeof(int));
    cudaMalloc(&d_pending, size_t(n) * sizeof(int));
    cudaMemset(d_pending, 0, size_t(n) * sizeof(int));
    cudaMalloc(&d_dwell, size_t(n) * sizeof(int));
    cudaMemset(d_dwell, 0, size_t(n) * sizeof(int));
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_bfm_zero_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_latents,
        d_latent_v,
        d_last,
        d_hist,
        d_latent,
        d_pending,
        d_dwell,
        d_obs,
        envs
    );
    policy_api::engine_run(*engine, d_obs, d_act, envs);
    k_bfm_zero_act<<<blocks, threads>>>(
        d_act,
        c.arm_pose,
        d_last,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return NUM_OWNED; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "bfm_zero"; }
};

}
