namespace decoupled_wbc {

const int DWBC_NUM_ACTIONS = 15;

const int DWBC_NUM_OBS = 58;

const int DWBC_HISTORY = 2;

const int DWBC_NUM_INPUT = DWBC_NUM_OBS * DWBC_HISTORY;

const std::string DWBC_MODEL_PATH = "policies/decoupled_wbc/model.onnx";

const double DWBC_DEFAULT_POS[DWBC_NUM_ACTIONS] = {
    -0.20,
    0.0,
    0.0,
    0.42,
    -0.23,
    0.0,
    -0.20,
    0.0,
    0.0,
    0.42,
    -0.23,
    0.0,
    0.0,
    0.0,
    0.0
};

const float DWBC_KPS[DWBC_NUM_ACTIONS] = {
    150.0f,
    150.0f,
    100.0f,
    150.0f,
    40.0f,
    40.0f,
    150.0f,
    150.0f,
    100.0f,
    150.0f,
    40.0f,
    40.0f,
    200.0f,
    150.0f,
    150.0f
};

const float DWBC_KDS[DWBC_NUM_ACTIONS] = {
    3.0f,
    3.0f,
    2.0f,
    4.0f,
    2.0f,
    2.0f,
    3.0f,
    3.0f,
    2.0f,
    4.0f,
    2.0f,
    2.0f,
    4.0f,
    4.0f,
    4.0f
};

const double DWBC_ACTION_SCALE = 0.25;

const double DWBC_CMD_HEIGHT = 0.70;

const double DWBC_CMD_BODY_ROLL = 0.0;

const double DWBC_CMD_BODY_PITCH = 0.0;

const double DWBC_CMD_BODY_YAW = 0.0;

const int DWBC_WAIST_PITCH = 14;

const double DWBC_WAIST_PITCH_MIN = -0.60;

const double DWBC_WAIST_PITCH_MAX = 0.60;

const double DWBC_VX_ABS = 0.55;

const double DWBC_VY_ABS = 0.55;

const double DWBC_YAW_RATE_ABS = 1.57;

const Limits LIMITS =
    Limits{-DWBC_VX_ABS, DWBC_VX_ABS, DWBC_VY_ABS, DWBC_YAW_RATE_ABS, 0.0};

struct DecoupledWbc {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::array<float, DWBC_NUM_ACTIONS> last_action_;

  std::array<float, DWBC_NUM_INPUT> history_;
};

std::shared_ptr<Engine> dwbc_engine_init(const std::string& model_path) {
  std::cout << "decoupled_wbc: loading policy model " << model_path
            << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "decoupled_wbc: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(DWBC_NUM_INPUT),
      true,
      "decoupled_wbc"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(DWBC_NUM_ACTIONS),
      false,
      "decoupled_wbc"
  );
  std::cout << "decoupled_wbc: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

void dwbc_build_frame(
    const DecoupledWbc& self,
    const RobotState& rs,
    const double drive[3],
    float* frame
) {
  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  for (int k = 0; k < 3; ++k) {
    frame[k] = static_cast<float>(rs.imu_gyro[k]);
    frame[3 + k] = static_cast<float>(gravity_dir[k]);
    frame[6 + k] = static_cast<float>(drive[k]);
  }
  frame[9] = static_cast<float>(DWBC_CMD_HEIGHT);
  frame[10] = static_cast<float>(DWBC_CMD_BODY_ROLL);
  frame[11] = static_cast<float>(DWBC_CMD_BODY_PITCH);
  frame[12] = static_cast<float>(DWBC_CMD_BODY_YAW);

  for (int j = 0; j < DWBC_NUM_ACTIONS; ++j) {
    frame[13 + j] = static_cast<float>(rs.motor_q[j] - DWBC_DEFAULT_POS[j]);
    frame[13 + DWBC_NUM_ACTIONS + j] = static_cast<float>(rs.motor_dq[j]);
    frame[13 + 2 * DWBC_NUM_ACTIONS + j] = self.last_action_[j];
  }
}

Output policy_step(
    DecoupledWbc& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);

  const RobotState& rs = *in.state;

  std::copy(
      self.history_.begin() + DWBC_NUM_OBS,
      self.history_.end(),
      self.history_.begin()
  );
  dwbc_build_frame(
      self,
      rs,
      drive,
      self.history_.data() + (DWBC_HISTORY - 1) * DWBC_NUM_OBS
  );

  const std::vector<float> action = engine_run_single(
      *self.engine_,
      self.history_.data(),
      self.history_.size()
  );
  std::copy(action.begin(), action.end(), self.last_action_.begin());

  Output out{};
  for (int j = 0; j < DWBC_NUM_ACTIONS; ++j) {
    out.q_target[j] = DWBC_DEFAULT_POS[j] +
                      static_cast<double>(action[j]) * DWBC_ACTION_SCALE;
    out.kp[j] = DWBC_KPS[j];
    out.kd[j] = DWBC_KDS[j];
    out.owns[j] = true;
  }
  out.q_target[DWBC_WAIST_PITCH] = std::clamp(
      out.q_target[DWBC_WAIST_PITCH],
      DWBC_WAIST_PITCH_MIN,
      DWBC_WAIST_PITCH_MAX
  );
  return out;
}

std::shared_ptr<DecoupledWbc> policy_make() {
  const std::shared_ptr<DecoupledWbc> self = std::make_shared<DecoupledWbc>();

  self->last_action_.fill(0.0f);
  self->history_.fill(0.0f);

  self->engine_ = dwbc_engine_init(DWBC_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "decoupled_wbc"; }

 private:
  std::shared_ptr<DecoupledWbc> state_;
};

}
