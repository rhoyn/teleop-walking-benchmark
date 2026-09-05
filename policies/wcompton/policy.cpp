namespace wcompton {

const std::string WC_MODEL_PATH = "policies/wcompton/model.onnx";

const int WC_NUM_ACTIONS = 21;

const int WC_NUM_OBS = 74;

const int WC_OFF_ANG_VEL = 0;
const int WC_OFF_GRAVITY = 3;
const int WC_OFF_COMMAND = 6;
const int WC_OFF_PHASE = 9;
const int WC_OFF_JOINT_POS = 11;
const int WC_OFF_JOINT_VEL = 32;
const int WC_OFF_LAST_ACTION = 53;

const int WC_FIRST_ARM_MOTOR = 15;

const int WC_ISAAC_TO_MUJOCO[WC_NUM_ACTIONS] = {
    0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3, 9, 17, 24, 4, 10, 18, 25, 5, 11
};

const double WC_DEFAULT_ISAAC[WC_NUM_ACTIONS] = {
    -0.20, -0.20, 0.00, 0.00, 0.00,  0.35,  0.35, 0.00, 0.00, 0.27, -0.27,
    0.42,  0.42,  0.00, 0.00, -0.23, -0.23, 0.87, 0.87, 0.00, 0.00
};

const float WC_KPS_ISAAC[WC_NUM_ACTIONS] = {
    40.179238f, 40.179238f, 28.501246f, 99.098428f, 99.098428f, 14.250623f,
    14.250623f, 40.179238f, 40.179238f, 14.250623f, 14.250623f, 99.098428f,
    99.098428f, 14.250623f, 14.250623f, 28.501246f, 28.501246f, 14.250623f,
    14.250623f, 28.501246f, 28.501246f
};

const float WC_KDS_ISAAC[WC_NUM_ACTIONS] = {
    2.557890f, 2.557890f, 1.814446f, 6.308802f, 6.308802f, 0.907223f,
    0.907223f, 2.557890f, 2.557890f, 0.907223f, 0.907223f, 6.308802f,
    6.308802f, 0.907223f, 0.907223f, 1.814446f, 1.814446f, 0.907223f,
    0.907223f, 1.814446f, 1.814446f
};

const double WC_ACTION_SCALE = 0.5;

const double WC_CONTROL_DT = 0.02;

const double WC_GAIT_PERIOD = 0.8;

const double WC_STAND_THRESHOLD = 0.1;

const double WC_VX_MIN = -0.5;
const double WC_VX_MAX = 1.0;
const double WC_VY_ABS = 0.25;
const double WC_YAW_RATE_ABS = 1.0;

const Limits LIMITS =
    Limits{WC_VX_MIN, WC_VX_MAX, WC_VY_ABS, WC_YAW_RATE_ABS, 0.0};

struct Wcompton {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::array<float, WC_NUM_ACTIONS> last_action_;
  double gait_phase_ = 0.0;
};

std::shared_ptr<Engine> wc_engine_init(const std::string& model_path) {
  std::cout << "wcompton: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "wcompton: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(WC_NUM_OBS),
      true,
      "wcompton"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(WC_NUM_ACTIONS),
      false,
      "wcompton"
  );
  std::cout << "wcompton: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

template <typename Obs>
std::vector<float> wc_engine_run(
    Engine& s,
    const Obs& obs
) {
  return engine_run_single(s, obs.data(), obs.size());
}

std::array<
    float,
    WC_NUM_OBS>
wc_build_obs(
    const Wcompton& self,
    const RobotState& rs,
    const double cmd[3],
    bool gait_on
) {
  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  std::array<float, WC_NUM_OBS> obs{};
  for (int k = 0; k < 3; ++k) {
    obs[WC_OFF_ANG_VEL + k] = static_cast<float>(rs.imu_gyro[k]);
    obs[WC_OFF_GRAVITY + k] = static_cast<float>(gravity_dir[k]);
    obs[WC_OFF_COMMAND + k] = static_cast<float>(cmd[k]);
  }

  const double angle = self.gait_phase_ * 2.0 * M_PI;
  obs[WC_OFF_PHASE + 0] = gait_on ? static_cast<float>(std::sin(angle)) : 0.0f;
  obs[WC_OFF_PHASE + 1] = gait_on ? static_cast<float>(std::cos(angle)) : 0.0f;

  for (int i = 0; i < WC_NUM_ACTIONS; ++i) {
    const int motor = WC_ISAAC_TO_MUJOCO[i];
    obs[WC_OFF_JOINT_POS + i] =
        static_cast<float>(rs.motor_q[motor] - WC_DEFAULT_ISAAC[i]);
    obs[WC_OFF_JOINT_VEL + i] = static_cast<float>(rs.motor_dq[motor]);
    obs[WC_OFF_LAST_ACTION + i] = self.last_action_[i];
  }
  return obs;
}

Output policy_step(
    Wcompton& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);

  self.gait_phase_ =
      std::fmod(self.gait_phase_ + WC_CONTROL_DT / WC_GAIT_PERIOD, 1.0);
  const double cmd_norm = std::sqrt(
      drive[0] * drive[0] + drive[1] * drive[1] + drive[2] * drive[2]
  );
  const bool gait_on = cmd_norm >= WC_STAND_THRESHOLD;

  const std::array<float, WC_NUM_OBS> obs =
      wc_build_obs(self, *in.state, drive, gait_on);
  const std::vector<float> action = wc_engine_run(*self.engine_, obs);
  std::copy(action.begin(), action.end(), self.last_action_.begin());

  Output out{};
  for (int i = 0; i < WC_NUM_ACTIONS; ++i) {
    const int motor = WC_ISAAC_TO_MUJOCO[i];
    if (motor >= WC_FIRST_ARM_MOTOR) continue;
    out.q_target[motor] =
        WC_DEFAULT_ISAAC[i] + static_cast<double>(action[i]) * WC_ACTION_SCALE;
    out.kp[motor] = WC_KPS_ISAAC[i];
    out.kd[motor] = WC_KDS_ISAAC[i];
    out.owns[motor] = true;
  }
  return out;
}

std::shared_ptr<Wcompton> policy_make() {
  const std::shared_ptr<Wcompton> self = std::make_shared<Wcompton>();

  self->last_action_.fill(0.0f);

  self->engine_ = wc_engine_init(WC_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "wcompton"; }

 private:
  std::shared_ptr<Wcompton> state_;
};

}
