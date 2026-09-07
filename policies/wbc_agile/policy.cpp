namespace wbc_agile {

constexpr int HISTORY = 5;
constexpr int NUM_ACTIONS = 12;
constexpr int NUM_JOINTS = 29;
constexpr int OWNED = 15;
constexpr int NUM_OBS =
    HISTORY * (4 + 3 + 3 + NUM_JOINTS + NUM_JOINTS + NUM_ACTIONS);

constexpr int OFF_CMD = 0;
constexpr int OFF_GYRO = OFF_CMD + HISTORY * 4;
constexpr int OFF_GRAV = OFF_GYRO + HISTORY * 3;
constexpr int OFF_QPOS = OFF_GRAV + HISTORY * 3;
constexpr int OFF_QVEL = OFF_QPOS + HISTORY * NUM_JOINTS;
constexpr int OFF_ACT = OFF_QVEL + HISTORY * NUM_JOINTS;
static_assert(
    OFF_ACT + HISTORY * NUM_ACTIONS == NUM_OBS,
    "obs layout"
);

constexpr float HEIGHT_CMD = 0.72f;
constexpr float JOINT_VEL_SCALE = 0.1f;
constexpr float ACTION_HISTORY_CLIP = 10.0f;
constexpr float TARGET_CLIP = 6.0f;
constexpr float WAIST_POS = 0.0f;

constexpr int LAYERS = 4;
constexpr int WIDTHS[LAYERS + 1] = {NUM_OBS, 512, 256, 128, NUM_ACTIONS};
const char* const WEIGHT_NAMES[LAYERS] = {
    "_tensor_constant17",
    "_tensor_constant19",
    "_tensor_constant21",
    "_tensor_constant23"
};
const char* const BIAS_NAMES[LAYERS] = {
    "_tensor_constant18",
    "_tensor_constant20",
    "_tensor_constant22",
    "_tensor_constant24"
};

const char* const MODEL_PATH = "policies/wbc_agile/model.onnx";

#define WBC_AGILE_TO_BENCH                                                  \
  0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, \
      18, 25, 19, 26, 20, 27, 21, 28

#define WBC_AGILE_LEG_TO_BENCH 0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11

#define WBC_AGILE_STANCE                                                    \
  -0.1f, -0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.3f, 0.0f, \
      0.0f, -0.2f, -0.2f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,   \
      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f

#define WBC_AGILE_ACT_SCALE                                               \
  0.22f, 0.22f, 0.3475f, 0.3475f, 0.22f, 0.22f, 0.17375f, 0.17375f, 1.0f, \
      1.0f, 1.0f, 1.0f
#define WBC_AGILE_ACT_OFFSET \
  -0.1f, -0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.3f, -0.2f, -0.2f, 0.0f, 0.0f

__device__ const int D_TO_BENCH[NUM_JOINTS] = {WBC_AGILE_TO_BENCH};
__device__ const int D_LEG_TO_BENCH[NUM_ACTIONS] = {WBC_AGILE_LEG_TO_BENCH};
__device__ const float D_STANCE[NUM_JOINTS] = {WBC_AGILE_STANCE};
__device__ const float D_ACT_SCALE[NUM_ACTIONS] = {WBC_AGILE_ACT_SCALE};
__device__ const float D_ACT_OFFSET[NUM_ACTIONS] = {WBC_AGILE_ACT_OFFSET};

const float KPS[OWNED] =
    {100, 100, 100, 200, 20, 20, 100, 100, 100, 200, 20, 20, 300, 300, 300};
const float KDS[OWNED] = {
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
    5.0f,
    5.0f,
    5.0f
};

const policy_api::Limits LIMITS = {-0.5, 1.5, 0.5, 1.0, 0.0};

__global__ void k_wbc_agile_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    const float* __restrict__ last_action,
    float* __restrict__ obs,
    int primed,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  float* o = obs + size_t(env) * NUM_OBS;
  const float* c = cmd + env * 3;
  const float* g = gyro + env * 3;
  const float* v = gravity + env * 3;
  const float* q = motor_q + size_t(env) * POLICY_NUM_MOTOR;
  const float* dq = motor_dq + size_t(env) * POLICY_NUM_MOTOR;
  const float* a = last_action + size_t(env) * NUM_ACTIONS;

#define WBC_AGILE_ROLL(base, width)                     \
  {                                                     \
    float* blk = o + (base);                            \
    if (primed) {                                       \
      for (int i = 0; i < (width) * (HISTORY - 1); ++i) \
        blk[i] = blk[i + (width)];                      \
    }                                                   \
  }

  WBC_AGILE_ROLL(OFF_CMD, 4)
  WBC_AGILE_ROLL(OFF_GYRO, 3)
  WBC_AGILE_ROLL(OFF_GRAV, 3)
  WBC_AGILE_ROLL(OFF_QPOS, NUM_JOINTS)
  WBC_AGILE_ROLL(OFF_QVEL, NUM_JOINTS)
  WBC_AGILE_ROLL(OFF_ACT, NUM_ACTIONS)
#undef WBC_AGILE_ROLL

  const int first = primed ? (HISTORY - 1) : 0;
  for (int h = first; h < HISTORY; ++h) {
    float* f = o + OFF_CMD + h * 4;
    f[0] = c[0];
    f[1] = c[1];
    f[2] = c[2];
    f[3] = HEIGHT_CMD;

    float* w = o + OFF_GYRO + h * 3;
    for (int k = 0; k < 3; ++k) w[k] = g[k];

    float* p = o + OFF_GRAV + h * 3;
    for (int k = 0; k < 3; ++k) p[k] = v[k];

    float* jp = o + OFF_QPOS + h * NUM_JOINTS;
    float* jv = o + OFF_QVEL + h * NUM_JOINTS;
    for (int j = 0; j < NUM_JOINTS; ++j) {
      const int m = D_TO_BENCH[j];
      jp[j] = q[m] - D_STANCE[j];
      jv[j] = dq[m] * JOINT_VEL_SCALE;
    }

    float* ac = o + OFF_ACT + h * NUM_ACTIONS;
    for (int k = 0; k < NUM_ACTIONS; ++k) {
      ac[k] = fminf(fmaxf(a[k], -ACTION_HISTORY_CLIP), ACTION_HISTORY_CLIP);
    }
  }
}

__global__ void k_wbc_agile_act(
    const float* __restrict__ act,
    const float* __restrict__ arm_pose,
    float* __restrict__ last_action,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* a = act + size_t(env) * NUM_ACTIONS;
  float* out = q_target + size_t(env) * POLICY_NUM_MOTOR;
  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    out[j] = arm_pose[size_t(env) * POLICY_NUM_MOTOR + j];
  }
  for (int k = 0; k < NUM_ACTIONS; ++k) {
    last_action[size_t(env) * NUM_ACTIONS + k] = a[k];
    const float t = a[k] * D_ACT_SCALE[k] + D_ACT_OFFSET[k];
    out[D_LEG_TO_BENCH[k]] = fminf(fmaxf(t, -TARGET_CLIP), TARGET_CLIP);
  }
  for (int k = 0; k < 3; ++k) out[NUM_ACTIONS + k] = WAIST_POS;
}

class Logger : public nvinfer1::ILogger {
 public:
  void log(
      Severity severity,
      const char* msg
  ) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "trt: %s\n", msg);
  }
};

inline Logger& logger() {
  static Logger instance;
  return instance;
}

struct Wire {
  const unsigned char* p;
  const unsigned char* end;
};

inline unsigned long long wire_varint(Wire& w) {
  unsigned long long v = 0;
  int shift = 0;
  while (w.p < w.end) {
    const unsigned char b = *w.p++;
    v |= static_cast<unsigned long long>(b & 0x7f) << shift;
    if ((b & 0x80) == 0) break;
    shift += 7;
  }
  return v;
}

inline bool wire_next(
    Wire& w,
    int& field,
    int& kind,
    Wire& payload
) {
  if (w.p >= w.end) return false;
  const unsigned long long key = wire_varint(w);
  field = int(key >> 3);
  kind = int(key & 7);
  payload = Wire{w.p, w.p};
  switch (kind) {
    case 0:
      wire_varint(w);
      break;
    case 1:
      w.p += 8;
      break;
    case 2: {
      const unsigned long long n = wire_varint(w);
      payload = Wire{w.p, w.p + n};
      w.p += n;
      break;
    }
    case 5:
      w.p += 4;
      break;
    default:
      return false;
  }
  if (w.p > w.end) return false;
  return true;
}

std::map<
    std::string,
    std::vector<float>>
onnx_weights(
    const std::string& path,
    const std::vector<std::string>& wanted
) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("wbc_agile: cannot open " + path);
  const std::vector<char> buf(
      (std::istreambuf_iterator<char>(f)),
      std::istreambuf_iterator<char>()
  );
  const unsigned char* base =
      reinterpret_cast<const unsigned char*>(buf.data());

  Wire model{base, base + buf.size()};
  Wire graph{nullptr, nullptr};
  int field = 0, kind = 0;
  Wire payload{nullptr, nullptr};
  while (wire_next(model, field, kind, payload)) {
    if (field == 7 && kind == 2) {
      graph = payload;
      break;
    }
  }
  if (graph.p == nullptr)
    throw std::runtime_error("wbc_agile: no graph in " + path);

  std::map<std::string, std::vector<float>> out;
  while (wire_next(graph, field, kind, payload)) {
    if (field != 5 || kind != 2) continue;
    Wire tensor = payload;
    std::string name;
    const unsigned char* raw = nullptr;
    size_t raw_len = 0;
    int tf = 0, tk = 0;
    Wire tp{nullptr, nullptr};
    while (wire_next(tensor, tf, tk, tp)) {
      if (tf == 8 && tk == 2) {
        name.assign(reinterpret_cast<const char*>(tp.p), size_t(tp.end - tp.p));
      } else if (tf == 9 && tk == 2) {
        raw = tp.p;
        raw_len = size_t(tp.end - tp.p);
      }
    }
    if (raw == nullptr) continue;
    if (std::find(wanted.begin(), wanted.end(), name) == wanted.end()) continue;
    std::vector<float> v(raw_len / sizeof(float));
    std::memcpy(v.data(), raw, v.size() * sizeof(float));
    out[name] = std::move(v);
  }
  for (const std::string& n : wanted) {
    if (out.find(n) == out.end()) {
      throw std::runtime_error(
          "wbc_agile: " + path + " has no initializer '" + n + "'"
      );
    }
  }
  return out;
}

std::string plan_cache(int batch) {
  const std::filesystem::path p(MODEL_PATH);
  const std::uintmax_t size = std::filesystem::file_size(p);
  const auto stamp =
      std::filesystem::last_write_time(p).time_since_epoch().count();
  return "build/trt/wbc_agile_mlp_b" + std::to_string(batch) + "_" +
         std::to_string(size) + "_" +
         std::to_string(static_cast<long long>(stamp)) + "_trt" +
         std::to_string(NV_TENSORRT_MAJOR) + "." +
         std::to_string(NV_TENSORRT_MINOR) + ".plan";
}

std::vector<char> plan_build(int batch) {
  std::vector<std::string> wanted;
  for (int k = 0; k < LAYERS; ++k) {
    wanted.push_back(WEIGHT_NAMES[k]);
    wanted.push_back(BIAS_NAMES[k]);
  }
  const std::map<std::string, std::vector<float>> w =
      onnx_weights(MODEL_PATH, wanted);

  std::unique_ptr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger())
  );
  if (!builder) throw std::runtime_error("wbc_agile: cannot create a builder");
  std::unique_ptr<nvinfer1::INetworkDefinition> net(
      builder->createNetworkV2(0)
  );

  nvinfer1::ITensor* x = net->addInput(
      "obs",
      nvinfer1::DataType::kFLOAT,
      nvinfer1::Dims2{-1, NUM_OBS}
  );
  for (int k = 0; k < LAYERS; ++k) {
    const std::vector<float>& wk = w.at(WEIGHT_NAMES[k]);
    const std::vector<float>& bk = w.at(BIAS_NAMES[k]);
    const int in = WIDTHS[k], out = WIDTHS[k + 1];
    if (int(wk.size()) != in * out || int(bk.size()) != out) {
      throw std::runtime_error(
          "wbc_agile: layer weights are not the expected shape"
      );
    }
    nvinfer1::ITensor* wt = net->addConstant(
                                   nvinfer1::Dims2{out, in},
                                   nvinfer1::Weights{
                                       nvinfer1::DataType::kFLOAT,
                                       wk.data(),
                                       int64_t(wk.size())
                                   }
    )
                                ->getOutput(0);
    nvinfer1::ITensor* bt = net->addConstant(
                                   nvinfer1::Dims2{1, out},
                                   nvinfer1::Weights{
                                       nvinfer1::DataType::kFLOAT,
                                       bk.data(),
                                       int64_t(bk.size())
                                   }
    )
                                ->getOutput(0);
    x = net->addMatrixMultiply(
               *x,
               nvinfer1::MatrixOperation::kNONE,
               *wt,
               nvinfer1::MatrixOperation::kTRANSPOSE
    )
            ->getOutput(0);
    x = net->addElementWise(*x, *bt, nvinfer1::ElementWiseOperation::kSUM)
            ->getOutput(0);
    if (k + 1 < LAYERS) {
      nvinfer1::IActivationLayer* act =
          net->addActivation(*x, nvinfer1::ActivationType::kELU);
      act->setAlpha(1.0);
      x = act->getOutput(0);
    }
  }
  x->setName("act");
  net->markOutput(*x);

  std::unique_ptr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig()
  );
  nvinfer1::IOptimizationProfile* profile =
      builder->createOptimizationProfile();
  profile->setDimensions(
      "obs",
      nvinfer1::OptProfileSelector::kMIN,
      nvinfer1::Dims2{1, NUM_OBS}
  );
  profile->setDimensions(
      "obs",
      nvinfer1::OptProfileSelector::kOPT,
      nvinfer1::Dims2{batch, NUM_OBS}
  );
  profile->setDimensions(
      "obs",
      nvinfer1::OptProfileSelector::kMAX,
      nvinfer1::Dims2{batch, NUM_OBS}
  );
  config->addOptimizationProfile(profile);

  std::unique_ptr<nvinfer1::IHostMemory> plan(
      builder->buildSerializedNetwork(*net, *config)
  );
  if (!plan) throw std::runtime_error("wbc_agile: cannot build a plan");
  const char* p = static_cast<const char*>(plan->data());
  return std::vector<char>(p, p + plan->size());
}

std::shared_ptr<policy_api::Engine> engine_make(int batch) {
  const std::string cache = plan_cache(batch);
  std::vector<char> plan;
  if (std::ifstream f(cache, std::ios::binary); f) {
    plan.assign(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>()
    );
  } else {
    std::printf("trt: building a plan for %s at batch %d\n", MODEL_PATH, batch);
    plan = plan_build(batch);
    std::filesystem::create_directories("build/trt");
    std::ofstream out(cache, std::ios::binary);
    out.write(plan.data(), static_cast<std::streamsize>(plan.size()));
  }

  auto e = std::make_shared<policy_api::Engine>();
  e->max_batch = batch;
  e->obs_dim = NUM_OBS;
  e->inputs = {{"obs", {-1, NUM_OBS}}};
  e->outputs = {{"act", {-1, NUM_ACTIONS}}};
  e->input_name = "obs";
  e->output_name = "act";

  e->core = std::make_shared<policy_api::EngineCore>();
  e->core->runtime.reset(nvinfer1::createInferRuntime(logger()));
  e->core->engine.reset(
      e->core->runtime->deserializeCudaEngine(plan.data(), plan.size())
  );
  if (!e->core->engine) {
    throw std::runtime_error("wbc_agile: cannot deserialize the plan");
  }
  e->context.reset(e->core->engine->createExecutionContext());
  if (!e->context)
    throw std::runtime_error("wbc_agile: cannot create an execution context");
  return e;
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> net;
  float* d_obs = nullptr;
  float* d_act = nullptr;
  float* d_last = nullptr;
  int envs = 0;
  bool primed = false;

  ~Policy() override {
    for (void* p : {(void*)d_obs, (void*)d_act, (void*)d_last}) {
      if (p) cudaFree(p);
    }
  }

  void init(int n) override {
    envs = n;
    net = engine_make(n);
    cudaMalloc(&d_obs, size_t(n) * NUM_OBS * sizeof(float));
    cudaMemset(d_obs, 0, size_t(n) * NUM_OBS * sizeof(float));
    cudaMalloc(&d_act, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMemset(d_act, 0, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_last, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMemset(d_last, 0, size_t(n) * NUM_ACTIONS * sizeof(float));
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    k_wbc_agile_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_last,
        d_obs,
        primed ? 1 : 0,
        envs
    );
    primed = true;
    policy_api::engine_run(*net, d_obs, d_act, envs);
    k_wbc_agile_act<<<blocks, threads>>>(
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
  const char* name() const override { return "wbc_agile"; }
};

}
