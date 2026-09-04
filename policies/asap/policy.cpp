namespace asap {

const int ASAP_NUM_ACTIONS = 12;

const int ASAP_HISTORY_LEN = 4;

const int ASAP_FRAME = 100;

const std::string MODEL_ASAP_PATH = "policies/asap/model.onnx";

const int ASAP_NUM_OBS = ASAP_FRAME + ASAP_HISTORY_LEN * ASAP_FRAME;

const int ASAP_NUM_OWNED = 15;

const double ASAP_DEFAULT_ANGLES[NUM_MOTOR] = {-0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                               -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
                                               0.0,  0.0, 0.0, 0.0, 0.0,  0.0,
                                               0.0,  0.0, 0.0, 0.0, 0.0,  0.0,
                                               0.0,  0.0, 0.0, 0.0, 0.0};

const int ASAP_NUM_UPPER = 17;

const int ASAP_FIRST_UPPER = 12;

const double ASAP_ACTION_SCALE = 0.25;

const double ASAP_GAIT_PERIOD = 0.9;

const double ASAP_BASE_HEIGHT_CMD = 0.78;

const double ASAP_ANG_VEL_SCALE = 0.25;

const double ASAP_DOF_VEL_SCALE = 0.05;

const double ASAP_BASE_HEIGHT_SCALE = 2.0;

const double ASAP_KPS[ASAP_NUM_OWNED] = {
    100.0,
    100.0,
    100.0,
    200.0,
    20.0,
    20.0,
    100.0,
    100.0,
    100.0,
    200.0,
    20.0,
    20.0,
    400.0,
    400.0,
    400.0
};

const double ASAP_KDS[ASAP_NUM_OWNED] =
    {2.5, 2.5, 2.5, 5.0, 0.2, 0.1, 2.5, 2.5, 2.5, 5.0, 0.2, 0.1, 5.0, 5.0, 5.0};

const double ASAP_Q_MIN[NUM_MOTOR] = {
    -2.5307, -0.5236,      -2.7576,      -0.087267, -0.87267, -0.2618,
    -2.5307, -2.9671,      -2.7576,      -0.087267, -0.87267, -0.2618,
    -2.618,  -0.52,        -0.52,        -3.0892,   -1.5882,  -2.618,
    -1.0472, -1.972222054, -1.61443,     -1.61443,  -3.0892,  -2.2515,
    -2.618,  -1.0472,      -1.972222054, -1.61443,  -1.61443
};

const double ASAP_Q_MAX[NUM_MOTOR] = {
    2.8798, 2.9671, 2.7576,      2.8798,      0.5236,  0.2618,  2.8798, 0.5236,
    2.7576, 2.8798, 0.5236,      0.2618,      2.618,   0.52,    0.52,   2.6704,
    2.2515, 2.618,  2.0944,      1.972222054, 1.61443, 1.61443, 2.6704, 1.5882,
    2.618,  2.0944, 1.972222054, 1.61443,     1.61443
};

const int OFF_ACTIONS = 0;

const int OFF_ANG_VEL = 12;

const int OFF_CMD_YAW = 15;

const int OFF_CMD_HEIGHT = 16;

const int OFF_CMD_LIN = 17;

const int OFF_CMD_STAND = 19;

const int OFF_COS_PHASE = 20;

const int OFF_DOF_POS = 21;

const int OFF_DOF_VEL = 50;

const int OFF_GRAVITY = 79;

const int OFF_REF_UPPER = 82;

const int OFF_SIN_PHASE = 99;

const int ASAP_SEGMENTS[][2] = {
    {0, 12},
    {12, 15},
    {15, 16},
    {16, 17},
    {17, 19},
    {19, 20},
    {20, 21},
    {21, 50},
    {50, 79},
    {79, 82},
    {82, 99},
    {99, 100}
};

const int ASAP_HEAD = OFF_GRAVITY;

const double ASAP_VX_MAX = 1.0;

const double ASAP_VY_MAX = 0.8;

const double ASAP_YAW_RATE_MAX = 0.8;

const double ASAP_WALK_ENTER_POS_M = 0.08;

const double ASAP_WALK_ENTER_YAW_RAD = 0.10;

const double ASAP_WALK_EXIT_POS_M = 0.04;

const double ASAP_WALK_EXIT_YAW_RAD = 0.05;

using Frame = std::array<double, ASAP_FRAME>;

const Limits LIMITS = Limits{
    .vx_min = -ASAP_VX_MAX,
    .vx_max = ASAP_VX_MAX,
    .vy_abs = ASAP_VY_MAX,
    .yaw_rate_abs = ASAP_YAW_RATE_MAX,
    .speed_norm = 0.0
};

struct Asap {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::array<Frame, ASAP_HISTORY_LEN> history_;
  bool history_primed_ = false;
  std::array<double, ASAP_NUM_ACTIONS> last_action_;
  bool walking_ = false;
  bool gate_fresh_ = false;
};

std::shared_ptr<Engine> asap_engine_init(const std::string& model_path) {
  std::cout << "asap: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "asap: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(ASAP_NUM_OBS),
      true,
      "asap"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(ASAP_NUM_ACTIONS),
      false,
      "asap"
  );
  std::cout << "asap: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

template <typename Obs>
std::vector<float> asap_engine_run(
    Engine& s,
    const Obs& obs
) {
  return engine_run_single(s, obs.data(), obs.size());
}

std::vector<float> asap_obs(
    const Asap& p,
    const Frame& now
) {
  std::vector<float> obs;
  obs.reserve(ASAP_NUM_OBS);
  for (int i = 0; i < ASAP_HEAD; ++i) {
    obs.push_back(static_cast<float>(now[i]));
  }
  for (const int (&seg)[2] : ASAP_SEGMENTS) {
    for (const Frame& f : p.history_) {
      for (int i = seg[0]; i < seg[1]; ++i) {
        obs.push_back(static_cast<float>(f[i]));
      }
    }
  }
  for (int i = ASAP_HEAD; i < ASAP_FRAME; ++i) {
    obs.push_back(static_cast<float>(now[i]));
  }
  if (obs.size() != static_cast<size_t>(ASAP_NUM_OBS)) {
    throw std::runtime_error("asap: observation assembly size");
  }
  return obs;
}

Output policy_step(
    Asap& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  [&] {
    if (self.walking_) {
      if (in.dist_m < ASAP_WALK_EXIT_POS_M &&
          std::fabs(in.yaw_err_rad) < ASAP_WALK_EXIT_YAW_RAD) {
        self.walking_ = false;
      }
    } else {
      if (in.dist_m > ASAP_WALK_ENTER_POS_M ||
          std::fabs(in.yaw_err_rad) > ASAP_WALK_ENTER_YAW_RAD) {
        self.walking_ = true;
      }
    }
    self.gate_fresh_ = true;
    if (!self.walking_) {
      drive[0] = 0.0;
      drive[1] = 0.0;
      drive[2] = 0.0;
    }
  }();
  Output out{};

  const RobotState& rs = *in.state;

  if (!self.gate_fresh_) self.walking_ = false;
  self.gate_fresh_ = false;

  const double stand = self.walking_ ? 1.0 : 0.0;
  const double phase =
      stand > 0.5
          ? std::fmod(in.control_time, ASAP_GAIT_PERIOD) / ASAP_GAIT_PERIOD
          : 0.0;

  Frame frame{};
  for (int i = 0; i < ASAP_NUM_ACTIONS; ++i) {
    frame[OFF_ACTIONS + i] = self.last_action_[i];
  }
  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int k = 0; k < 3; ++k) {
    frame[OFF_ANG_VEL + k] = rs.imu_gyro[k] * ASAP_ANG_VEL_SCALE;
    frame[OFF_GRAVITY + k] = gravity_dir[k];
  }
  frame[OFF_CMD_YAW] = drive[2];
  frame[OFF_CMD_HEIGHT] = ASAP_BASE_HEIGHT_CMD * ASAP_BASE_HEIGHT_SCALE;
  frame[OFF_CMD_LIN + 0] = drive[0];
  frame[OFF_CMD_LIN + 1] = drive[1];
  frame[OFF_CMD_STAND] = stand;
  frame[OFF_COS_PHASE] = std::cos(2.0 * M_PI * phase);
  frame[OFF_SIN_PHASE] = std::sin(2.0 * M_PI * phase);
  for (int j = 0; j < NUM_MOTOR; ++j) {
    frame[OFF_DOF_POS + j] = rs.motor_q[j] - ASAP_DEFAULT_ANGLES[j];
    frame[OFF_DOF_VEL + j] = rs.motor_dq[j] * ASAP_DOF_VEL_SCALE;
  }
  for (int i = 0; i < ASAP_NUM_UPPER; ++i) {
    const int motor = ASAP_FIRST_UPPER + i;
    const double target = motor < ASAP_NUM_OWNED ? ASAP_DEFAULT_ANGLES[motor]
                                                 : in.arm_targets[motor];
    frame[OFF_REF_UPPER + i] = target - ASAP_DEFAULT_ANGLES[motor];
  }

  if (!self.history_primed_) {
    for (Frame& f : self.history_) f = frame;
    self.history_primed_ = true;
  }

  const std::vector<float> obs = asap_obs(self, frame);
  for (int i = ASAP_HISTORY_LEN - 1; i > 0; --i)
    self.history_[i] = self.history_[i - 1];
  self.history_[0] = frame;

  const std::vector<float> action = asap_engine_run(*self.engine_, obs);

  for (int i = 0; i < ASAP_NUM_ACTIONS; ++i) {
    const double raw =
        std::clamp(static_cast<double>(action[i]), -100.0, 100.0);
    self.last_action_[i] = raw;
    out.q_target[i] = std::clamp(
        ASAP_DEFAULT_ANGLES[i] + raw * ASAP_ACTION_SCALE,
        ASAP_Q_MIN[i],
        ASAP_Q_MAX[i]
    );
  }
  for (int i = ASAP_NUM_ACTIONS; i < ASAP_NUM_OWNED; ++i) {
    out.q_target[i] = ASAP_DEFAULT_ANGLES[i];
  }
  for (int i = 0; i < ASAP_NUM_OWNED; ++i) {
    out.kp[i] = static_cast<float>(ASAP_KPS[i]);
    out.kd[i] = static_cast<float>(ASAP_KDS[i]);
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<Asap> policy_make() {
  const std::shared_ptr<Asap> self = std::make_shared<Asap>();

  for (Frame& f : self->history_) f.fill(0.0);
  self->history_primed_ = false;
  self->last_action_.fill(0.0);
  self->walking_ = false;
  self->gate_fresh_ = false;

  self->engine_ = asap_engine_init(MODEL_ASAP_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "asap"; }

 private:
  std::shared_ptr<Asap> state_;
};

}
