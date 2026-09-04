namespace rl_mjlab {

const int MJLAB_NUM_ACTIONS = 29;

const std::string MJLAB_MODEL_PATH = "policies/rl_mjlab/model.onnx";

const int MJLAB_NUM_OBS = 98;

const int MJLAB_FIRST_ARM_MOTOR = 15;

const double MJLAB_DEFAULT_POS[MJLAB_NUM_ACTIONS] = {
    -0.1, 0.0, 0.0,  0.3,   -0.2, 0.0,  -0.1, 0.0, 0.0,  0.3,
    -0.2, 0.0, 0.0,  0.0,   0.0,  0.35, 0.18, 0.0, 0.87, 0.0,
    0.0,  0.0, 0.35, -0.18, 0.0,  0.87, 0.0,  0.0, 0.0
};

const double MJLAB_CONTROL_DT = 0.02;

const double MJLAB_ACTION_SCALE[MJLAB_NUM_ACTIONS] = {
    0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55, 0.35, 0.55, 0.35,
    0.44, 0.44, 0.55, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44,
    0.07, 0.07, 0.44, 0.44, 0.44, 0.44, 0.44, 0.07, 0.07
};

const float MJLAB_KPS[MJLAB_NUM_ACTIONS] = {
    40.2f, 99.1f, 40.2f, 99.1f, 28.5f, 28.5f, 40.2f, 99.1f, 40.2f, 99.1f,
    28.5f, 28.5f, 40.2f, 28.5f, 28.5f, 14.3f, 14.3f, 14.3f, 14.3f, 14.3f,
    16.8f, 16.8f, 14.3f, 14.3f, 14.3f, 14.3f, 14.3f, 16.8f, 16.8f
};

const float MJLAB_KDS[MJLAB_NUM_ACTIONS] = {2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
                                            2.6f, 6.3f, 2.6f, 6.3f, 1.8f, 1.8f,
                                            2.6f, 1.8f, 1.8f, 0.9f, 0.9f, 0.9f,
                                            0.9f, 0.9f, 1.1f, 1.1f, 0.9f, 0.9f,
                                            0.9f, 0.9f, 0.9f, 1.1f, 1.1f};

const double MJLAB_GAIT_PERIOD = 0.6;

const double MJLAB_GAIT_CMD_DEADBAND = 0.1;

const double MJLAB_VX_MIN = -0.5;

const double MJLAB_VX_MAX = 1.0;

const double MJLAB_VY_ABS = 0.5;

const double MJLAB_YAW_RATE_ABS = 1.0;

const Limits LIMITS =
    Limits{MJLAB_VX_MIN, MJLAB_VX_MAX, MJLAB_VY_ABS, MJLAB_YAW_RATE_ABS, 0.0};

struct RlMjlab {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::array<float, MJLAB_NUM_ACTIONS> last_action_;
  double gait_phase_ = 0.0;
};

std::shared_ptr<Engine> mjlab_engine_init(const std::string& model_path) {
  std::cout << "rl_mjlab: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "rl_mjlab: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(MJLAB_NUM_OBS),
      true,
      "rl_mjlab"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(MJLAB_NUM_ACTIONS),
      false,
      "rl_mjlab"
  );
  std::cout << "rl_mjlab: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

template <typename Obs>
std::vector<float> mjlab_engine_run(
    Engine& s,
    const Obs& obs
) {
  return engine_run_single(s, obs.data(), obs.size());
}

Output policy_step(
    RlMjlab& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  self.gait_phase_ =
      std::fmod(self.gait_phase_ + MJLAB_CONTROL_DT / MJLAB_GAIT_PERIOD, 1.0);
  const double cmd_norm = std::sqrt(
      drive[0] * drive[0] + drive[1] * drive[1] + drive[2] * drive[2]
  );
  const bool gait_on = cmd_norm >= MJLAB_GAIT_CMD_DEADBAND;

  std::array<float, MJLAB_NUM_OBS> obs{};
  for (int k = 0; k < 3; ++k) {
    obs[k] = static_cast<float>(rs.imu_gyro[k]);
    obs[3 + k] = static_cast<float>(gravity_dir[k]);
  }
  obs[6] = static_cast<float>(drive[0]);
  obs[7] = static_cast<float>(drive[1]);
  obs[8] = static_cast<float>(drive[2]);
  obs[9] = gait_on ? static_cast<float>(std::sin(self.gait_phase_ * 2.0 * M_PI))
                   : 0.0f;
  obs[10] = gait_on
                ? static_cast<float>(std::cos(self.gait_phase_ * 2.0 * M_PI))
                : 0.0f;
  for (int j = 0; j < MJLAB_NUM_ACTIONS; ++j) {
    obs[11 + j] = static_cast<float>(rs.motor_q[j] - MJLAB_DEFAULT_POS[j]);
    obs[11 + MJLAB_NUM_ACTIONS + j] = static_cast<float>(rs.motor_dq[j]);
    obs[11 + 2 * MJLAB_NUM_ACTIONS + j] = self.last_action_[j];
  }

  const std::vector<float> action = mjlab_engine_run(*self.engine_, obs);
  std::copy(action.begin(), action.end(), self.last_action_.begin());

  for (int i = 0; i < MJLAB_FIRST_ARM_MOTOR; ++i) {
    out.q_target[i] = MJLAB_DEFAULT_POS[i] +
                      static_cast<double>(action[i]) * MJLAB_ACTION_SCALE[i];
    out.kp[i] = MJLAB_KPS[i];
    out.kd[i] = MJLAB_KDS[i];
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<RlMjlab> policy_make() {
  const std::shared_ptr<RlMjlab> self = std::make_shared<RlMjlab>();

  self->last_action_.fill(0.0f);

  self->engine_ = mjlab_engine_init(MJLAB_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "rl_mjlab"; }

 private:
  std::shared_ptr<RlMjlab> state_;
};

}
