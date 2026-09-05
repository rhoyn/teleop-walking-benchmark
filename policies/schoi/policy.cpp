namespace schoi {

const int SCHOI_NUM_ACTIONS = 12;

const int SCHOI_NUM_OBS = 47;

const std::string SCHOI_MODEL_PATH = "policies/schoi/model.onnx";

const int SCHOI_MUJOCO_FROM_ISAAC[SCHOI_NUM_ACTIONS] = {
    0,
    6,
    1,
    7,
    2,
    8,
    3,
    9,
    4,
    10,
    5,
    11
};

const double SCHOI_DEFAULT_POS[SCHOI_NUM_ACTIONS] = {-0.10, 0.0, 0.0, 0.3,
                                                     -0.2,  0.0, -0.10, 0.0,
                                                     0.0,   0.3, -0.2,  0.0};

const float SCHOI_KPS[SCHOI_NUM_ACTIONS] = {100.0f, 100.0f, 100.0f, 150.0f,
                                            40.0f,  40.0f,  100.0f, 100.0f,
                                            100.0f, 150.0f, 40.0f,  40.0f};

const float SCHOI_KDS[SCHOI_NUM_ACTIONS] = {3.0f, 3.0f, 3.0f, 5.0f, 3.0f, 3.0f,
                                            3.0f, 3.0f, 3.0f, 5.0f, 3.0f, 3.0f};

const double SCHOI_ANG_VEL_SCALE = 0.5;

const double SCHOI_DOF_POS_SCALE = 1.0;

const double SCHOI_DOF_VEL_SCALE = 0.05;

const double SCHOI_ACTION_SCALE = 0.25;

const double SCHOI_CMD_SCALE[3] = {1.0, 1.0, 1.0};

const double SCHOI_GAIT_PERIOD = 0.8;

const double SCHOI_GAIT_CMD_DEADBAND = 0.1;

const double SCHOI_CONTROL_DT = 0.02;

const double SCHOI_VX_MAX = 1.0;

const double SCHOI_VX_MIN = -1.0;

const double SCHOI_VY_ABS = 0.5;

const double SCHOI_YAW_RATE_ABS = 0.5;

const Limits LIMITS =
    Limits{SCHOI_VX_MIN, SCHOI_VX_MAX, SCHOI_VY_ABS, SCHOI_YAW_RATE_ABS, 0.0};

struct Schoi {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;

  std::array<float, SCHOI_NUM_ACTIONS> last_action_;
  double gait_phase_ = 0.0;
};

std::shared_ptr<Engine> schoi_engine_init(const std::string& model_path) {
  std::cout << "schoi: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "schoi: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(SCHOI_NUM_OBS),
      true,
      "schoi"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(SCHOI_NUM_ACTIONS),
      false,
      "schoi"
  );
  std::cout << "schoi: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

Output policy_step(
    Schoi& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  self.gait_phase_ =
      std::fmod(self.gait_phase_ + SCHOI_CONTROL_DT / SCHOI_GAIT_PERIOD, 1.0);

  std::array<float, SCHOI_NUM_OBS> obs{};
  for (int k = 0; k < 3; ++k) {
    obs[k] = static_cast<float>(rs.imu_gyro[k] * SCHOI_ANG_VEL_SCALE);
    obs[3 + k] = static_cast<float>(gravity_dir[k]);
    obs[6 + k] = static_cast<float>(drive[k] * SCHOI_CMD_SCALE[k]);
  }

  for (int k = 0; k < SCHOI_NUM_ACTIONS; ++k) {
    const int j = SCHOI_MUJOCO_FROM_ISAAC[k];
    obs[9 + k] = static_cast<float>(
        (rs.motor_q[j] - SCHOI_DEFAULT_POS[j]) * SCHOI_DOF_POS_SCALE
    );
    obs[9 + SCHOI_NUM_ACTIONS + k] =
        static_cast<float>(rs.motor_dq[j] * SCHOI_DOF_VEL_SCALE);
    obs[9 + 2 * SCHOI_NUM_ACTIONS + k] = self.last_action_[k];
  }

  const double cmd_norm = std::sqrt(
      static_cast<double>(obs[6]) * obs[6] +
      static_cast<double>(obs[7]) * obs[7] + static_cast<double>(obs[8]) * obs[8]
  );
  if (cmd_norm >= SCHOI_GAIT_CMD_DEADBAND) {
    obs[45] = static_cast<float>(std::sin(self.gait_phase_ * 2.0 * M_PI));
    obs[46] = static_cast<float>(std::cos(self.gait_phase_ * 2.0 * M_PI));
  }

  const std::vector<float> action =
      engine_run_single(*self.engine_, obs.data(), obs.size());
  std::copy(action.begin(), action.end(), self.last_action_.begin());

  for (int k = 0; k < SCHOI_NUM_ACTIONS; ++k) {
    const int j = SCHOI_MUJOCO_FROM_ISAAC[k];
    out.q_target[j] = SCHOI_DEFAULT_POS[j] +
                      static_cast<double>(action[k]) * SCHOI_ACTION_SCALE;
    out.kp[j] = SCHOI_KPS[j];
    out.kd[j] = SCHOI_KDS[j];
    out.owns[j] = true;
  }
  return out;
}

std::shared_ptr<Schoi> policy_make() {
  const std::shared_ptr<Schoi> self = std::make_shared<Schoi>();

  self->last_action_.fill(0.0f);

  self->engine_ = schoi_engine_init(SCHOI_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "schoi"; }

 private:
  std::shared_ptr<Schoi> state_;
};

}
