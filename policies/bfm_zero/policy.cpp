namespace bfm_zero {

const std::string BFM_MODEL_PATH = "policies/bfm_zero/model.onnx";

const std::string BFM_LATENT_PATH = "policies/bfm_zero/latents.bin";

const int BFM_NUM_JOINTS = 29;

const int BFM_FIRST_ARM_MOTOR = 15;

const int BFM_HISTORY = 4;

const int BFM_Z_DIM = 256;

const int BFM_NUM_OBS = 465;

const int BFM_NUM_INPUT = BFM_NUM_OBS + BFM_Z_DIM;

const int BFM_NUM_LATENTS = 8;

const int BFM_OFF_QPOS = 0;

const int BFM_OFF_QVEL = 29;

const int BFM_OFF_GRAV = 58;

const int BFM_OFF_ANG = 61;

const int BFM_OFF_LAST_ACTION = 64;

const int BFM_OFF_HISTORY = 93;

const int BFM_HIST_ACT = 0;

const int BFM_HIST_ANG = 116;

const int BFM_HIST_QPOS = 128;

const int BFM_HIST_QVEL = 244;

const int BFM_HIST_GRAV = 360;

const double BFM_ANG_VEL_SCALE = 0.25;

const double BFM_ACTION_RESCALE = 5.0;

const double BFM_ACTION_CLIP = 5.0;

const double BFM_DEFAULT_POS[BFM_NUM_JOINTS] = {-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                                -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                                0.0,  0.0, 0.0, 0.0, 0.0,  0.0,
                                                0.0,  0.0, 0.0, 0.0, 0.0,  0.0,
                                                0.0,  0.0, 0.0, 0.0, 0.0};

const double BFM_ACTION_SCALE[BFM_NUM_JOINTS] = {
    0.222001498914, 0.22200157, 0.54754699, 0.35066156, 0.43857802, 0.43857802,
    0.222001498914, 0.22200157, 0.54754699, 0.35066156, 0.43857802, 0.43857802,
    0.54754699,     0.43857802, 0.43857802, 0.43857802, 0.43857802, 0.43857802,
    0.43857802,     0.43857802, 0.07450086, 0.07466888, 0.43857802, 0.43857802,
    0.43857802,     0.43857802, 0.43857802, 0.07450086, 0.07450086
};

const float BFM_KPS[BFM_NUM_JOINTS] = {
    99.09843f, 99.0984f,  40.1792f,  99.0984f,  28.5012f, 28.5012f,
    99.09843f, 99.0984f,  40.1792f,  99.0984f,  28.5012f, 28.5012f,
    40.1792f,  28.5012f,  28.5012f,  14.2506f,  14.2506f, 14.2506f,
    14.2506f,  14.25062f, 16.77833f, 16.77833f, 14.2506f, 14.2506f,
    14.2506f,  14.2506f,  14.25062f, 16.77833f, 16.77833f
};

const float BFM_KDS[BFM_NUM_JOINTS] = {
    6.3088f, 6.3088f, 2.5579f, 6.3088f, 1.8145f, 1.8145f, 6.3088f, 6.3088f,
    2.5579f, 6.3088f, 1.8145f, 1.8145f, 2.5579f, 1.8145f, 1.8145f, 0.9072f,
    0.9072f, 0.9072f, 0.9072f, 0.9072f, 1.0681f, 1.0681f, 0.9072f, 0.9072f,
    0.9072f, 0.9072f, 0.9072f, 1.0681f, 1.0681f
};

const int BFM_Z_STAND = 0;

const int BFM_Z_FORWARD_SLOW = 1;

const int BFM_Z_FORWARD_FAST = 2;

const int BFM_Z_LEFT_SLOW = 3;

const int BFM_Z_LEFT_FAST = 4;

const int BFM_Z_BACK_SLOW = 5;

const int BFM_Z_RIGHT_SLOW = 6;

const int BFM_Z_RIGHT_FAST = 7;

const double BFM_STAND_SPEED = 0.15;

const double BFM_FAST_SPEED = 0.5;

const int BFM_DWELL_STEPS = 10;

const double BFM_VX_MIN = -0.3;

const double BFM_VX_MAX = 0.7;

const double BFM_VY_ABS = 0.7;

const double BFM_YAW_RATE_ABS = 0.0;

const double BFM_SPEED_NORM = 0.7;

const Limits LIMITS = Limits{
    BFM_VX_MIN,
    BFM_VX_MAX,
    BFM_VY_ABS,
    BFM_YAW_RATE_ABS,
    BFM_SPEED_NORM
};

struct BfmZero {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::vector<float> latents_;
  std::array<float, BFM_NUM_JOINTS> last_action_;
  std::array<float, BFM_HISTORY * BFM_NUM_JOINTS> hist_act_;
  std::array<float, BFM_HISTORY * BFM_NUM_JOINTS> hist_qpos_;
  std::array<float, BFM_HISTORY * BFM_NUM_JOINTS> hist_qvel_;
  std::array<float, BFM_HISTORY * 3> hist_ang_;
  std::array<float, BFM_HISTORY * 3> hist_grav_;
  int latent_ = BFM_Z_STAND;
  int pending_ = BFM_Z_STAND;
  int dwell_ = 0;
};

std::vector<float> bfm_latents_load(const std::string& path) {
  const size_t want =
      static_cast<size_t>(BFM_NUM_LATENTS) * static_cast<size_t>(BFM_Z_DIM);
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("bfm_zero: cannot open latent file " + path);
  }
  const std::streamsize bytes = file.tellg();
  if (bytes != static_cast<std::streamsize>(want * sizeof(float))) {
    throw std::runtime_error(
        "bfm_zero: latent file " + path + " is " + std::to_string(bytes) +
        " bytes, expected " + std::to_string(want * sizeof(float))
    );
  }
  file.seekg(0, std::ios::beg);
  std::vector<float> latents(want);
  if (!file.read(
          reinterpret_cast<char*>(latents.data()),
          static_cast<std::streamsize>(want * sizeof(float))
      )) {
    throw std::runtime_error("bfm_zero: short read on latent file " + path);
  }
  for (size_t i = 0; i < want; ++i) {
    if (!std::isfinite(latents[i])) {
      throw std::runtime_error(
          "bfm_zero: latent file holds a non-finite value"
      );
    }
  }
  return latents;
}

std::shared_ptr<Engine> bfm_engine_init(const std::string& model_path) {
  std::cout << "bfm_zero: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "bfm_zero: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(BFM_NUM_INPUT),
      true,
      "bfm_zero"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(BFM_NUM_JOINTS),
      false,
      "bfm_zero"
  );
  std::cout << "bfm_zero: engine ready, input " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

int bfm_latent_for_command(const double drive[3]) {
  const double speed = std::sqrt(drive[0] * drive[0] + drive[1] * drive[1]);
  if (speed < BFM_STAND_SPEED) return BFM_Z_STAND;

  const bool fast = speed >= BFM_FAST_SPEED;
  const double heading = std::atan2(drive[1], drive[0]);
  const double quarter = M_PI / 4.0;

  if (heading > -quarter && heading <= quarter) {
    return fast ? BFM_Z_FORWARD_FAST : BFM_Z_FORWARD_SLOW;
  }
  if (heading > quarter && heading <= 3.0 * quarter) {
    return fast ? BFM_Z_LEFT_FAST : BFM_Z_LEFT_SLOW;
  }
  if (heading <= -quarter && heading > -3.0 * quarter) {
    return fast ? BFM_Z_RIGHT_FAST : BFM_Z_RIGHT_SLOW;
  }
  return BFM_Z_BACK_SLOW;
}

void bfm_latent_update(
    BfmZero& self,
    const double drive[3]
) {
  const int want = bfm_latent_for_command(drive);
  if (want == self.latent_) {
    self.pending_ = want;
    self.dwell_ = 0;
    return;
  }
  if (want != self.pending_) {
    self.pending_ = want;
    self.dwell_ = 0;
    return;
  }
  if (++self.dwell_ >= BFM_DWELL_STEPS) {
    self.latent_ = want;
    self.dwell_ = 0;
  }
}

template <typename Buffer>
void bfm_history_push(
    Buffer& buffer,
    const float* frame,
    int width
) {
  const int total = static_cast<int>(buffer.size());
  for (int i = total - width - 1; i >= 0; --i) {
    buffer[static_cast<size_t>(i + width)] = buffer[static_cast<size_t>(i)];
  }
  for (int i = 0; i < width; ++i) {
    buffer[static_cast<size_t>(i)] = frame[i];
  }
}

Output policy_step(
    BfmZero& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  bfm_latent_update(self, drive);

  Output out{};
  const RobotState& rs = *in.state;

  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  std::array<float, BFM_NUM_JOINTS> qpos{};
  std::array<float, BFM_NUM_JOINTS> qvel{};
  for (int j = 0; j < BFM_NUM_JOINTS; ++j) {
    qpos[static_cast<size_t>(j)] =
        static_cast<float>(rs.motor_q[j] - BFM_DEFAULT_POS[j]);
    qvel[static_cast<size_t>(j)] = static_cast<float>(rs.motor_dq[j]);
  }
  std::array<float, 3> grav{};
  std::array<float, 3> ang{};
  for (int k = 0; k < 3; ++k) {
    grav[static_cast<size_t>(k)] = static_cast<float>(gravity_dir[k]);
    ang[static_cast<size_t>(k)] =
        static_cast<float>(rs.imu_gyro[k] * BFM_ANG_VEL_SCALE);
  }

  std::array<float, BFM_NUM_INPUT> input{};
  for (int j = 0; j < BFM_NUM_JOINTS; ++j) {
    input[static_cast<size_t>(BFM_OFF_QPOS + j)] = qpos[static_cast<size_t>(j)];
    input[static_cast<size_t>(BFM_OFF_QVEL + j)] = qvel[static_cast<size_t>(j)];
    input[static_cast<size_t>(BFM_OFF_LAST_ACTION + j)] =
        self.last_action_[static_cast<size_t>(j)];
  }
  for (int k = 0; k < 3; ++k) {
    input[static_cast<size_t>(BFM_OFF_GRAV + k)] = grav[static_cast<size_t>(k)];
    input[static_cast<size_t>(BFM_OFF_ANG + k)] = ang[static_cast<size_t>(k)];
  }
  for (int i = 0; i < BFM_HISTORY * BFM_NUM_JOINTS; ++i) {
    input[static_cast<size_t>(BFM_OFF_HISTORY + BFM_HIST_ACT + i)] =
        self.hist_act_[static_cast<size_t>(i)];
    input[static_cast<size_t>(BFM_OFF_HISTORY + BFM_HIST_QPOS + i)] =
        self.hist_qpos_[static_cast<size_t>(i)];
    input[static_cast<size_t>(BFM_OFF_HISTORY + BFM_HIST_QVEL + i)] =
        self.hist_qvel_[static_cast<size_t>(i)];
  }
  for (int i = 0; i < BFM_HISTORY * 3; ++i) {
    input[static_cast<size_t>(BFM_OFF_HISTORY + BFM_HIST_ANG + i)] =
        self.hist_ang_[static_cast<size_t>(i)];
    input[static_cast<size_t>(BFM_OFF_HISTORY + BFM_HIST_GRAV + i)] =
        self.hist_grav_[static_cast<size_t>(i)];
  }
  const float* latent =
      self.latents_.data() + static_cast<size_t>(self.latent_) * BFM_Z_DIM;
  for (int i = 0; i < BFM_Z_DIM; ++i) {
    input[static_cast<size_t>(BFM_NUM_OBS + i)] = latent[i];
  }

  const std::vector<float> action =
      engine_run_single(*self.engine_, input.data(), input.size());

  bfm_history_push(self.hist_act_, self.last_action_.data(), BFM_NUM_JOINTS);
  bfm_history_push(self.hist_qpos_, qpos.data(), BFM_NUM_JOINTS);
  bfm_history_push(self.hist_qvel_, qvel.data(), BFM_NUM_JOINTS);
  bfm_history_push(self.hist_ang_, ang.data(), 3);
  bfm_history_push(self.hist_grav_, grav.data(), 3);

  for (int j = 0; j < BFM_NUM_JOINTS; ++j) {
    const double scaled = std::clamp(
        BFM_ACTION_RESCALE *
            static_cast<double>(action[static_cast<size_t>(j)]),
        -BFM_ACTION_CLIP,
        BFM_ACTION_CLIP
    );
    self.last_action_[static_cast<size_t>(j)] = static_cast<float>(scaled);
  }

  for (int i = 0; i < BFM_FIRST_ARM_MOTOR; ++i) {
    out.q_target[i] =
        BFM_DEFAULT_POS[i] +
        static_cast<double>(self.last_action_[static_cast<size_t>(i)]) *
            BFM_ACTION_SCALE[i];
    out.kp[i] = BFM_KPS[i];
    out.kd[i] = BFM_KDS[i];
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<BfmZero> policy_make() {
  const std::shared_ptr<BfmZero> self = std::make_shared<BfmZero>();

  self->last_action_.fill(0.0f);
  self->hist_act_.fill(0.0f);
  self->hist_qpos_.fill(0.0f);
  self->hist_qvel_.fill(0.0f);
  self->hist_ang_.fill(0.0f);
  self->hist_grav_.fill(0.0f);

  self->latents_ = bfm_latents_load(BFM_LATENT_PATH);
  self->engine_ = bfm_engine_init(BFM_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "bfm_zero"; }

 private:
  std::shared_ptr<BfmZero> state_;
};

}
