namespace handoff {

const std::string HANDOFF_MODEL_PATH = "policies/handoff/model.onnx";

const int HANDOFF_NUM_ACTIONS = 29;

const int HANDOFF_FIRST_ARM_MOTOR = 15;

const int HANDOFF_FRAME_DIM = 106;

const int HANDOFF_HISTORY_LEN = 11;

const int HANDOFF_NUM_OBS = HANDOFF_FRAME_DIM * (1 + HANDOFF_HISTORY_LEN);

const double HANDOFF_DEFAULT_POS[HANDOFF_NUM_ACTIONS] = {
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669,
    -0.363, 0.0, 0.0, 0.0,   0.0,    0.2, 0.2,    0.0, 0.6, 0.0,
    0.0,    0.0, 0.2, -0.2,  0.0,    0.6, 0.0,    0.0, 0.0
};

const double HANDOFF_ACTION_SCALE[HANDOFF_NUM_ACTIONS] = {
    0.547546, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577,
    0.547546, 0.350661, 0.547546, 0.350661, 0.438577, 0.438577,
    0.547546, 0.438577, 0.438577, 0.438577, 0.438577, 0.438577,
    0.438577, 0.438577, 0.074501, 0.074501, 0.438577, 0.438577,
    0.438577, 0.438577, 0.438577, 0.074501, 0.074501
};

const float HANDOFF_KPS[HANDOFF_NUM_ACTIONS] = {
    40.1792f, 99.0984f, 40.1792f, 99.0984f, 28.5012f, 28.5012f,
    40.1792f, 99.0984f, 40.1792f, 99.0984f, 28.5012f, 28.5012f,
    40.1792f, 28.5012f, 28.5012f, 14.2506f, 14.2506f, 14.2506f,
    14.2506f, 14.2506f, 16.7783f, 16.7783f, 14.2506f, 14.2506f,
    14.2506f, 14.2506f, 14.2506f, 16.7783f, 16.7783f
};

const float HANDOFF_KDS[HANDOFF_NUM_ACTIONS] = {
    2.55789f, 6.30880f, 2.55789f, 6.30880f, 1.81445f, 1.81445f,
    2.55789f, 6.30880f, 2.55789f, 6.30880f, 1.81445f, 1.81445f,
    2.55789f, 1.81445f, 1.81445f, 0.90722f, 0.90722f, 0.90722f,
    0.90722f, 0.90722f, 1.06814f, 1.06814f, 0.90722f, 0.90722f,
    0.90722f, 0.90722f, 0.90722f, 1.06814f, 1.06814f
};

const double HANDOFF_ANG_VEL_SCALE = 0.25;

const double HANDOFF_JOINT_VEL_SCALE = 0.05;

inline bool handoff_ankle_dof(int j) {
  return j == 4 || j == 5 || j == 10 || j == 11;
}

const double HANDOFF_GAIT_PERIOD = 1.0;

const double HANDOFF_GAIT_OFFSET = 0.5;

const double HANDOFF_STAND_VEL_THRESHOLD = 0.1;

const double HANDOFF_HEIGHT_CMD = 0.78;

const double HANDOFF_HAND_CMD[6] =
    {-0.08, 0.23044664, -0.09842005, -0.08, -0.23043664, -0.09842005};

const double HANDOFF_CONTROL_DT = 0.02;

const Limits LIMITS = Limits{-1.0, 1.0, 1.0, 1.0, 0.0};

struct Handoff {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::array<float, HANDOFF_NUM_ACTIONS> last_action_;

  double phase_ = 0.0;

  std::array<std::array<float, HANDOFF_FRAME_DIM>, HANDOFF_HISTORY_LEN>
      history_;

  bool history_ready_ = false;
};

std::shared_ptr<Engine> handoff_engine_init(const std::string& model_path) {
  std::cout << "handoff: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "handoff: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(HANDOFF_NUM_OBS),
      true,
      "handoff"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(HANDOFF_NUM_ACTIONS),
      false,
      "handoff"
  );
  std::cout << "handoff: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

void handoff_build_frame(
    const Handoff& self,
    const RobotState& rs,
    const double cmd[3],
    std::array<
        float,
        HANDOFF_FRAME_DIM>& frame
) {

  const double cmd_norm =
      std::sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1] + cmd[2] * cmd[2]);
  const bool stationary = cmd_norm < HANDOFF_STAND_VEL_THRESHOLD;

  frame[0] = stationary ? 0.0f : static_cast<float>(cmd[0]);
  frame[1] = stationary ? 0.0f : static_cast<float>(cmd[1]);
  frame[2] = static_cast<float>(HANDOFF_HEIGHT_CMD);
  frame[3] = stationary ? 0.0f : static_cast<float>(cmd[2]);
  for (int k = 0; k < 6; ++k) {
    frame[4 + k] = static_cast<float>(HANDOFF_HAND_CMD[k]);
  }

  if (stationary) {
    frame[10] = 0.0f;
    frame[11] = 1.0f;
    frame[12] = 0.0f;
    frame[13] = 1.0f;
  } else {
    const double left = self.phase_;
    const double right = std::fmod(self.phase_ + HANDOFF_GAIT_OFFSET, 1.0);
    frame[10] = static_cast<float>(std::sin(2.0 * M_PI * left));
    frame[11] = static_cast<float>(std::cos(2.0 * M_PI * left));
    frame[12] = static_cast<float>(std::sin(2.0 * M_PI * right));
    frame[13] = static_cast<float>(std::cos(2.0 * M_PI * right));
  }

  for (int k = 0; k < 3; ++k) {
    frame[14 + k] = static_cast<float>(rs.imu_gyro[k] * HANDOFF_ANG_VEL_SCALE);
  }

  const double w = rs.imu_quat[0];
  const double x = rs.imu_quat[1];
  const double y = rs.imu_quat[2];
  const double z = rs.imu_quat[3];
  const double roll =
      std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
  const double pitch = std::asin(std::clamp(2.0 * (w * y - z * x), -1.0, 1.0));
  frame[17] = static_cast<float>(roll);
  frame[18] = static_cast<float>(pitch);

  for (int j = 0; j < HANDOFF_NUM_ACTIONS; ++j) {
    frame[19 + j] = static_cast<float>(rs.motor_q[j] - HANDOFF_DEFAULT_POS[j]);
    frame[48 + j] =
        handoff_ankle_dof(j)
            ? 0.0f
            : static_cast<float>(rs.motor_dq[j] * HANDOFF_JOINT_VEL_SCALE);
    frame[77 + j] = self.last_action_[j];
  }
}

Output policy_step(
    Handoff& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  std::array<float, HANDOFF_FRAME_DIM> frame{};
  handoff_build_frame(self, rs, drive, frame);

  self.phase_ =
      std::fmod(self.phase_ + HANDOFF_CONTROL_DT / HANDOFF_GAIT_PERIOD, 1.0);

  if (!self.history_ready_) {
    self.history_.fill(frame);
    self.history_ready_ = true;
  } else {
    for (int t = 0; t + 1 < HANDOFF_HISTORY_LEN; ++t) {
      self.history_[t] = self.history_[t + 1];
    }
    self.history_[HANDOFF_HISTORY_LEN - 1] = frame;
  }

  std::array<float, HANDOFF_NUM_OBS> obs{};
  std::copy(frame.begin(), frame.end(), obs.begin());
  for (int t = 0; t < HANDOFF_HISTORY_LEN; ++t) {
    std::copy(
        self.history_[t].begin(),
        self.history_[t].end(),
        obs.begin() + HANDOFF_FRAME_DIM * (1 + t)
    );
  }

  const std::vector<float> action =
      engine_run_single(*self.engine_, obs.data(), obs.size());
  std::copy(action.begin(), action.end(), self.last_action_.begin());

  for (int i = 0; i < HANDOFF_FIRST_ARM_MOTOR; ++i) {
    out.q_target[i] = HANDOFF_DEFAULT_POS[i] +
                      static_cast<double>(action[i]) * HANDOFF_ACTION_SCALE[i];
    out.kp[i] = HANDOFF_KPS[i];
    out.kd[i] = HANDOFF_KDS[i];
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<Handoff> policy_make() {
  const std::shared_ptr<Handoff> self = std::make_shared<Handoff>();

  self->last_action_.fill(0.0f);
  for (std::array<float, HANDOFF_FRAME_DIM>& f : self->history_) f.fill(0.0f);

  self->engine_ = handoff_engine_init(HANDOFF_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "handoff"; }

 private:
  std::shared_ptr<Handoff> state_;
};

}
