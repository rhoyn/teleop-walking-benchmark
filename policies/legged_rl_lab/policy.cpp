namespace legged_rl_lab {

const int LRL_NUM_ACTIONS = 29;

const int LRL_NUM_OBS = 96;

const int LRL_FIRST_ARM_MOTOR = 15;

const std::string LRL_MODEL_PATH = "policies/legged_rl_lab/model.onnx";

const int LRL_MOTOR_OF_ISAAC[LRL_NUM_ACTIONS] = {
    0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4, 10,
    16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
};

const int LRL_ISAAC_OF_MOTOR[LRL_NUM_ACTIONS] = {
    0,  3,  6,  9,  13, 17, 1,  4,  7,  10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
};

const double LRL_DEFAULT_POS[LRL_NUM_ACTIONS] = {
    -0.1,  -0.1,  0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.3,
    0.3,   0.3,   0.3,  -0.2, -0.2, 0.25, -0.25, 0.0, 0.0,  0.0,
    0.0,   0.97,  0.97, 0.15, -0.15, 0.0, 0.0,  0.0,  0.0
};

const float LRL_KPS[LRL_NUM_ACTIONS] = {
    100.0f, 100.0f, 200.0f, 100.0f, 100.0f, 40.0f, 100.0f, 100.0f, 40.0f,
    150.0f, 150.0f, 40.0f,  40.0f,  40.0f,  40.0f, 40.0f,  40.0f,  40.0f,
    40.0f,  40.0f,  40.0f,  40.0f,  40.0f,  40.0f, 40.0f,  40.0f,  40.0f,
    40.0f,  40.0f
};

const float LRL_KDS[LRL_NUM_ACTIONS] = {
    2.0f, 2.0f, 5.0f, 2.0f, 2.0f, 5.0f, 2.0f, 2.0f, 5.0f, 4.0f,
    4.0f, 1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

const double LRL_ANG_VEL_SCALE = 0.2;
const double LRL_DOF_POS_SCALE = 1.0;
const double LRL_DOF_VEL_SCALE = 0.05;

const double LRL_ACTION_SCALE = 0.25;

const double LRL_RAMP_S = 0.8;

const double LRL_VX_MIN = 0.0;

const double LRL_VX_MAX = 1.0;

const double LRL_VY_ABS = 0.5;

const double LRL_YAW_RATE_ABS = 1.0;

const Limits LIMITS =
    Limits{LRL_VX_MIN, LRL_VX_MAX, LRL_VY_ABS, LRL_YAW_RATE_ABS, 0.0};

struct LeggedRlLab {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::array<float, LRL_NUM_ACTIONS> last_action_;
};

std::shared_ptr<Engine> lrl_engine_init(const std::string& model_path) {
  std::cout << "legged_rl_lab: loading policy model " << model_path
            << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "legged_rl_lab: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(LRL_NUM_OBS),
      true,
      "legged_rl_lab"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(LRL_NUM_ACTIONS),
      false,
      "legged_rl_lab"
  );
  std::cout << "legged_rl_lab: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

void lrl_build_obs(
    const LeggedRlLab& self,
    const RobotState& rs,
    const double drive[3],
    std::array<float, LRL_NUM_OBS>& obs
) {
  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  for (int k = 0; k < 3; ++k) {
    obs[k] = static_cast<float>(rs.imu_gyro[k] * LRL_ANG_VEL_SCALE);
    obs[3 + k] = static_cast<float>(gravity_dir[k]);
    obs[6 + k] = static_cast<float>(drive[k]);
  }

  for (int a = 0; a < LRL_NUM_ACTIONS; ++a) {
    const int m = LRL_MOTOR_OF_ISAAC[a];
    obs[9 + a] = static_cast<float>(
        (rs.motor_q[m] - LRL_DEFAULT_POS[a]) * LRL_DOF_POS_SCALE
    );
    obs[9 + LRL_NUM_ACTIONS + a] =
        static_cast<float>(rs.motor_dq[m] * LRL_DOF_VEL_SCALE);
    obs[9 + 2 * LRL_NUM_ACTIONS + a] = self.last_action_[a];
  }
}

Output policy_step(
    LeggedRlLab& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);

  const RobotState& rs = *in.state;

  std::array<float, LRL_NUM_OBS> obs{};
  lrl_build_obs(self, rs, drive, obs);

  const std::vector<float> action =
      engine_run_single(*self.engine_, obs.data(), obs.size());
  std::copy(action.begin(), action.end(), self.last_action_.begin());

  const double ramp = std::clamp(in.control_time / LRL_RAMP_S, 0.0, 1.0);

  Output out{};

  for (int m = 0; m < LRL_FIRST_ARM_MOTOR; ++m) {
    const int a = LRL_ISAAC_OF_MOTOR[m];
    const double target =
        LRL_DEFAULT_POS[a] + static_cast<double>(action[a]) * LRL_ACTION_SCALE;
    out.q_target[m] = LRL_DEFAULT_POS[a] + ramp * (target - LRL_DEFAULT_POS[a]);
    out.kp[m] = LRL_KPS[a];
    out.kd[m] = LRL_KDS[a];
    out.owns[m] = true;
  }
  return out;
}

std::shared_ptr<LeggedRlLab> policy_make() {
  const std::shared_ptr<LeggedRlLab> self = std::make_shared<LeggedRlLab>();

  self->last_action_.fill(0.0f);

  self->engine_ = lrl_engine_init(LRL_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "legged_rl_lab"; }

 private:
  std::shared_ptr<LeggedRlLab> state_;
};

}
