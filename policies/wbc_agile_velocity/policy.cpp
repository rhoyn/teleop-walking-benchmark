namespace wbc_agile_velocity {

constexpr int NUM_JOINTS = 29;
constexpr int NUM_ACTIONS = 14;
constexpr int HISTORY = 5;

constexpr int WAIST_YAW = 12;
constexpr float WAIST_YAW_POS = 0.0f;

constexpr int OWNED = 15;

constexpr int HIST_VEC = HISTORY * 3;
constexpr int HIST_ACT = HISTORY * NUM_ACTIONS;

constexpr float ANG_VEL_SCALE = 0.2f;
constexpr float JOINT_VEL_SCALE = 0.05f;

#define WBC_AGILE_VEL_TO_MUJOCO_LIST                                        \
  0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, \
      18, 25, 19, 26, 20, 27, 21, 28

#define WBC_AGILE_VEL_ACTION_TO_MUJOCO_LIST \
  0, 6, 1, 7, 13, 2, 8, 14, 3, 9, 4, 10, 5, 11

#define WBC_AGILE_VEL_ACTION_TO_OBS_LIST \
  0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18

#define WBC_AGILE_VEL_DEFAULT_POS_LIST                                        \
  -0.1f, -0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.3f, 0.3f, -0.2f, -0.2f, \
      0.0f, 0.0f

__device__ const int D_TO_MUJOCO[NUM_JOINTS] = {WBC_AGILE_VEL_TO_MUJOCO_LIST};
__device__ const int D_ACTION_TO_MUJOCO[NUM_ACTIONS] = {
    WBC_AGILE_VEL_ACTION_TO_MUJOCO_LIST
};
__device__ const int D_ACTION_TO_OBS[NUM_ACTIONS] = {
    WBC_AGILE_VEL_ACTION_TO_OBS_LIST
};
__device__ const float D_DEFAULT_POS[NUM_ACTIONS] = {
    WBC_AGILE_VEL_DEFAULT_POS_LIST
};

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

const policy_api::Limits LIMITS = {-0.5, 0.5, 0.5, 1.0, 0.0};

constexpr int FEEDBACK = 7;
const char* const FB_IN[FEEDBACK] = {
    "last_actions",
    "base_ang_vel_history",
    "projected_gravity_history",
    "velocity_commands_history",
    "controlled_joint_pos_history",
    "controlled_joint_vel_history",
    "actions_history"
};
const char* const FB_OUT[FEEDBACK] = {
    "last_actions_out",
    "base_ang_vel_history_out",
    "projected_gravity_history_out",
    "velocity_commands_history_out",
    "controlled_joint_pos_history_out",
    "controlled_joint_vel_history_out",
    "actions_history_out"
};
const int FB_WIDTH[FEEDBACK] =
    {NUM_ACTIONS, HIST_VEC, HIST_VEC, HIST_VEC, HIST_ACT, HIST_ACT, HIST_ACT};

__global__ void k_wbc_agile_velocity_obs(
    const float* __restrict__ motor_q,
    const float* __restrict__ motor_dq,
    const float* __restrict__ gyro,
    const float* __restrict__ gravity,
    const float* __restrict__ cmd,
    float* __restrict__ quat,
    float* __restrict__ ang_vel,
    float* __restrict__ command,
    float* __restrict__ joint_pos,
    float* __restrict__ joint_vel,
    float* __restrict__ last_action,
    float* __restrict__ gyro_hist,
    float* __restrict__ grav_hist,
    float* __restrict__ cmd_hist,
    float* __restrict__ pos_hist,
    float* __restrict__ vel_hist,
    float* __restrict__ act_hist,
    int prime,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  const float* g = gravity + env * 3;

  float w = 1.0f - g[2], x = -g[1], y = g[0];
  const float n = sqrtf(w * w + x * x + y * y);
  if (n > 1e-6f) {
    w /= n;
    x /= n;
    y /= n;
  } else {
    w = 0.0f;
    x = 1.0f;
    y = 0.0f;
  }
  float* q = quat + env * 4;
  q[0] = w;
  q[1] = x;
  q[2] = y;
  q[3] = 0.0f;

  for (int k = 0; k < 3; ++k) {
    ang_vel[env * 3 + k] = gyro[env * 3 + k];
    command[env * 3 + k] = cmd[env * 3 + k];
  }
  float* jp = joint_pos + env * NUM_JOINTS;
  float* jv = joint_vel + env * NUM_JOINTS;
  for (int j = 0; j < NUM_JOINTS; ++j) {
    const int m = D_TO_MUJOCO[j];
    jp[j] = motor_q[env * POLICY_NUM_MOTOR + m];
    jv[j] = motor_dq[env * POLICY_NUM_MOTOR + m];
  }

  if (!prime) return;

  for (int h = 0; h < HISTORY; ++h) {
    for (int k = 0; k < 3; ++k) {
      gyro_hist[env * HIST_VEC + h * 3 + k] = gyro[env * 3 + k] * ANG_VEL_SCALE;
      grav_hist[env * HIST_VEC + h * 3 + k] = g[k];
      cmd_hist[env * HIST_VEC + h * 3 + k] = cmd[env * 3 + k];
    }
    for (int a = 0; a < NUM_ACTIONS; ++a) {
      const int j = D_ACTION_TO_OBS[a];
      const int at = env * HIST_ACT + h * NUM_ACTIONS + a;
      pos_hist[at] = jp[j] - D_DEFAULT_POS[a];
      vel_hist[at] = jv[j] * JOINT_VEL_SCALE;
      act_hist[at] = 0.0f;
    }
  }
  for (int a = 0; a < NUM_ACTIONS; ++a)
    last_action[env * NUM_ACTIONS + a] = 0.0f;
}

__global__ void k_wbc_agile_velocity_act(
    const float* __restrict__ action,
    const float* __restrict__ arm_pose,
    float* __restrict__ q_target,
    int envs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  for (int j = 0; j < POLICY_NUM_MOTOR; ++j) {
    q_target[env * POLICY_NUM_MOTOR + j] = arm_pose[env * POLICY_NUM_MOTOR + j];
  }
  for (int a = 0; a < NUM_ACTIONS; ++a) {
    q_target[env * POLICY_NUM_MOTOR + D_ACTION_TO_MUJOCO[a]] =
        action[env * NUM_ACTIONS + a];
  }
  q_target[env * POLICY_NUM_MOTOR + WAIST_YAW] = WAIST_YAW_POS;
}

const char* const MODEL_SRC = "policies/wbc_agile_velocity/model.onnx";
const char* const MODEL_DYNAMIC =
    "build/trt/wbc_agile_velocity_dynamic_batch.onnx";

const char* const IN_NAMES[] = {
    "root_link_quat_w",
    "root_ang_vel_b",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "last_actions",
    "base_ang_vel_history",
    "projected_gravity_history",
    "velocity_commands_history",
    "controlled_joint_pos_history",
    "controlled_joint_vel_history",
    "actions_history"
};
constexpr int NUM_IN = int(sizeof(IN_NAMES) / sizeof(IN_NAMES[0]));

std::string onnx_bytes(std::initializer_list<int> v) {
  std::string s;
  for (int b : v) s.push_back(char(b));
  return s;
}

int patch_all(
    std::string& blob,
    const std::string& pat,
    size_t at,
    char to
) {
  int hits = 0;
  for (size_t p = blob.find(pat); p != std::string::npos;
       p = blob.find(pat, p + pat.size())) {
    blob[p + at] = to;
    ++hits;
  }
  return hits;
}

std::string dynamic_model() {
  namespace fs = std::filesystem;
  if (fs::exists(MODEL_DYNAMIC) && fs::exists(MODEL_SRC) &&
      fs::last_write_time(MODEL_DYNAMIC) >= fs::last_write_time(MODEL_SRC)) {
    return MODEL_DYNAMIC;
  }

  std::ifstream in(MODEL_SRC, std::ios::binary);
  if (!in)
    throw std::runtime_error(
        "wbc_agile_velocity: cannot read " + std::string(MODEL_SRC)
    );
  std::string blob(
      (std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>()
  );

  const std::string shape = onnx_bytes(
      {0x4a,
       0x10,
       0x01,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0xff,
       0xff,
       0xff,
       0xff,
       0xff,
       0xff,
       0xff,
       0xff}
  );
  if (patch_all(blob, shape, 2, 0) != 1) {
    throw std::runtime_error(
        "wbc_agile_velocity: model.onnx has no single [1, -1] reshape target"
    );
  }

  const std::string allow = onnx_bytes({0x22, 0x07}) + "Reshape" +
                            onnx_bytes({0x2a, 0x10, 0x0a, 0x09}) + "allowzero" +
                            onnx_bytes({0x18, 0x01, 0xa0, 0x01, 0x02});
  if (patch_all(blob, allow, allow.size() - 4, 0) != 13) {
    throw std::runtime_error(
        "wbc_agile_velocity: model.onnx does not carry the 13 expected reshapes"
    );
  }

  for (int i = 0; i < NUM_IN; ++i) {
    const std::string name = IN_NAMES[i];
    const std::string key = onnx_bytes({0x0a, int(name.size())}) + name;
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
          "wbc_agile_velocity: model.onnx does not declare '" + name +
          "' as one batched row"
      );
    }
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
    if (!out)
      throw std::runtime_error("wbc_agile_velocity: cannot write " + tmp);
  }
  fs::rename(tmp, MODEL_DYNAMIC);
  return MODEL_DYNAMIC;
}

struct Policy : policy_api::Policy {
  std::shared_ptr<policy_api::Engine> engine;
  float *d_quat = nullptr, *d_ang_vel = nullptr, *d_cmd = nullptr;
  float *d_joint_pos = nullptr, *d_joint_vel = nullptr;
  float *d_action = nullptr, *d_gains = nullptr;

  float* d_fb[2][FEEDBACK] = {};
  int live = 0;
  bool primed = false;
  int envs = 0;

  ~Policy() override {
    for (void* p :
         {(void*)d_quat,
          (void*)d_ang_vel,
          (void*)d_cmd,
          (void*)d_joint_pos,
          (void*)d_joint_vel,
          (void*)d_action,
          (void*)d_gains}) {
      if (p) cudaFree(p);
    }
    for (int s = 0; s < 2; ++s) {
      for (int k = 0; k < FEEDBACK; ++k) {
        if (d_fb[s][k]) cudaFree(d_fb[s][k]);
      }
    }
  }

  void init(int n) override {
    envs = n;

    std::vector<policy_api::TensorSpec> inputs = {
        {"root_link_quat_w", {-1, 4}},
        {"root_ang_vel_b", {-1, 3}},
        {"velocity_commands", {-1, 3}},
        {"joint_pos", {-1, NUM_JOINTS}},
        {"joint_vel", {-1, NUM_JOINTS}}
    };
    std::vector<policy_api::TensorSpec> outputs = {
        {"action_joint_pos", {-1, NUM_ACTIONS}},

        {"action_joint_pos_kp_gains", {1, NUM_ACTIONS}},
        {"action_joint_pos_kd_gains", {1, NUM_ACTIONS}}
    };
    for (int k = 0; k < FEEDBACK; ++k) {
      const std::vector<int> shape =
          FB_WIDTH[k] == NUM_ACTIONS
              ? std::vector<int>{-1, NUM_ACTIONS}
              : std::vector<int>{-1, HISTORY, FB_WIDTH[k] / HISTORY};
      inputs.push_back({FB_IN[k], shape});
      outputs.push_back({FB_OUT[k], shape});
    }
    engine = policy_api::engine_make(dynamic_model(), n, inputs, outputs);

    cudaMalloc(&d_quat, size_t(n) * 4 * sizeof(float));
    cudaMalloc(&d_ang_vel, size_t(n) * 3 * sizeof(float));
    cudaMalloc(&d_cmd, size_t(n) * 3 * sizeof(float));
    cudaMalloc(&d_joint_pos, size_t(n) * NUM_JOINTS * sizeof(float));
    cudaMalloc(&d_joint_vel, size_t(n) * NUM_JOINTS * sizeof(float));
    cudaMalloc(&d_action, size_t(n) * NUM_ACTIONS * sizeof(float));
    cudaMalloc(&d_gains, size_t(2) * NUM_ACTIONS * sizeof(float));
    for (int s = 0; s < 2; ++s) {
      for (int k = 0; k < FEEDBACK; ++k) {
        const size_t bytes = size_t(n) * size_t(FB_WIDTH[k]) * sizeof(float);
        cudaMalloc(&d_fb[s][k], bytes);
        cudaMemset(d_fb[s][k], 0, bytes);
      }
    }
  }

  void step(const policy_api::Ctx& c) override {
    const int threads = 128, blocks = (envs + threads - 1) / threads;
    float** in_fb = d_fb[live];
    float** out_fb = d_fb[1 - live];

    k_wbc_agile_velocity_obs<<<blocks, threads>>>(
        c.motor_q,
        c.motor_dq,
        c.gyro,
        c.gravity,
        c.cmd,
        d_quat,
        d_ang_vel,
        d_cmd,
        d_joint_pos,
        d_joint_vel,
        in_fb[0],
        in_fb[1],
        in_fb[2],
        in_fb[3],
        in_fb[4],
        in_fb[5],
        in_fb[6],
        primed ? 0 : 1,
        envs
    );
    primed = true;

    const float* in[5 + FEEDBACK] =
        {d_quat, d_ang_vel, d_cmd, d_joint_pos, d_joint_vel};
    float* out[3 + FEEDBACK] = {d_action, d_gains, d_gains + NUM_ACTIONS};
    for (int k = 0; k < FEEDBACK; ++k) {
      in[5 + k] = in_fb[k];
      out[3 + k] = out_fb[k];
    }
    policy_api::engine_run(*engine, in, out, envs);
    live = 1 - live;

    k_wbc_agile_velocity_act<<<blocks, threads>>>(
        d_action,
        c.arm_pose,
        c.q_target,
        envs
    );
  }

  const float* kp() const override { return KPS; }
  const float* kd() const override { return KDS; }
  int owned() const override { return OWNED; }
  policy_api::Limits limits() const override { return LIMITS; }
  const char* name() const override { return "wbc_agile_velocity"; }
};

}
