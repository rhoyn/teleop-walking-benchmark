namespace gr00t_wbc {

const std::string MODEL_WALK_PATH = "policies/gr00t_wbc/model_walk.onnx";

const std::string MODEL_BALANCE_PATH = "policies/gr00t_wbc/model_balance.onnx";

const int WBC_NUM_ACTIONS = 15;

const int WBC_SINGLE_OBS = 86;

const int WBC_OBS_HISTORY = 6;

const int WBC_NUM_OBS = WBC_SINGLE_OBS * WBC_OBS_HISTORY;

const double WBC_DEFAULT_ANGLES[WBC_NUM_ACTIONS] = {
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    -0.1,
    0.0,
    0.0,
    0.3,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0
};

const double WBC_ACTION_SCALE = 0.25;

const double WBC_BALANCE_CMD_NORM = 0.05;

const double WBC_KPS[WBC_NUM_ACTIONS] = {
    150.0,
    150.0,
    150.0,
    200.0,
    40.0,
    40.0,
    150.0,
    150.0,
    150.0,
    200.0,
    40.0,
    40.0,
    250.0,
    250.0,
    250.0
};

const double WBC_KDS[WBC_NUM_ACTIONS] =
    {2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0, 2.0, 4.0, 2.0, 2.0, 5.0, 5.0, 5.0};

const double WBC_CMD_SCALE[3] = {2.0, 2.0, 0.5};

const double WBC_HEIGHT_CMD = 0.74;

const double WBC_ANG_VEL_SCALE = 0.5;

const double WBC_DOF_VEL_SCALE = 0.05;

const double WBC_MAX_SPEED = 0.5;

const double WBC_MAX_YAW_RATE = 1.0;

const Limits LIMITS = Limits{
    -WBC_MAX_SPEED,
    WBC_MAX_SPEED,
    WBC_MAX_SPEED,
    WBC_MAX_YAW_RATE,
    WBC_MAX_SPEED
};

struct Gr00tWbc {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> walk_;
  std::shared_ptr<Engine> balance_;
  std::vector<float> obs_;
  std::vector<float> last_action_;
};

std::shared_ptr<Engine> wbc_engine_init(const std::string& model_path) {
  std::cout << "gr00t_wbc: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "gr00t_wbc: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(WBC_NUM_OBS),
      true,
      "gr00t_wbc"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(WBC_NUM_ACTIONS),
      false,
      "gr00t_wbc"
  );
  std::cout << "gr00t_wbc: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

template <typename Obs>
std::vector<float> wbc_engine_run(
    Engine& s,
    const Obs& obs
) {
  return engine_run_single(s, obs.data(), obs.size());
}

std::array<
    float,
    WBC_SINGLE_OBS>
wbc_single_obs(
    const Gr00tWbc& p,
    const RobotState& rs,
    const double cmd[3]
) {
  std::array<float, WBC_SINGLE_OBS> obs{};
  obs[0] = static_cast<float>(cmd[0] * WBC_CMD_SCALE[0]);
  obs[1] = static_cast<float>(cmd[1] * WBC_CMD_SCALE[1]);
  obs[2] = static_cast<float>(cmd[2] * WBC_CMD_SCALE[2]);
  obs[3] = static_cast<float>(WBC_HEIGHT_CMD);
  obs[4] = 0.0f;
  obs[5] = 0.0f;
  obs[6] = 0.0f;
  for (int k = 0; k < 3; ++k) {
    obs[7 + k] = static_cast<float>(rs.imu_gyro[k] * WBC_ANG_VEL_SCALE);
  }
  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int k = 0; k < 3; ++k) {
    obs[10 + k] = static_cast<float>(gravity_dir[k]);
  }
  for (int j = 0; j < NUM_MOTOR; ++j) {
    const double def = j < WBC_NUM_ACTIONS ? WBC_DEFAULT_ANGLES[j] : 0.0;
    obs[13 + j] = static_cast<float>(rs.motor_q[j] - def);
    obs[13 + NUM_MOTOR + j] =
        static_cast<float>(rs.motor_dq[j] * WBC_DOF_VEL_SCALE);
  }
  for (int k = 0; k < WBC_NUM_ACTIONS; ++k) {
    obs[13 + 2 * NUM_MOTOR + k] = p.last_action_[k];
  }
  return obs;
}

Output policy_step(
    Gr00tWbc& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;
  const double cmd[3] = {drive[0], drive[1], drive[2]};

  const std::array<float, WBC_SINGLE_OBS> frame = wbc_single_obs(self, rs, cmd);
  std::copy(
      self.obs_.begin() + WBC_SINGLE_OBS,
      self.obs_.end(),
      self.obs_.begin()
  );
  std::copy(frame.begin(), frame.end(), self.obs_.end() - WBC_SINGLE_OBS);

  const double cmd_norm =
      std::sqrt(cmd[0] * cmd[0] + cmd[1] * cmd[1] + cmd[2] * cmd[2]);
  const bool use_walk = cmd_norm > WBC_BALANCE_CMD_NORM;
  self.last_action_ =
      wbc_engine_run(use_walk ? *self.walk_ : *self.balance_, self.obs_);

  for (int i = 0; i < WBC_NUM_ACTIONS; ++i) {
    out.q_target[i] =
        WBC_DEFAULT_ANGLES[i] +
        static_cast<double>(self.last_action_[i]) * WBC_ACTION_SCALE;
    out.kp[i] = static_cast<float>(WBC_KPS[i]);
    out.kd[i] = static_cast<float>(WBC_KDS[i]);
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<Gr00tWbc> policy_make() {
  const std::shared_ptr<Gr00tWbc> self = std::make_shared<Gr00tWbc>();
  self->obs_ = std::vector<float>(WBC_NUM_OBS, 0.0f);
  self->last_action_ = std::vector<float>(WBC_NUM_ACTIONS, 0.0f);

  self->walk_ = wbc_engine_init(MODEL_WALK_PATH);
  self->balance_ = wbc_engine_init(MODEL_BALANCE_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "gr00t_wbc"; }

 private:
  std::shared_ptr<Gr00tWbc> state_;
};

}
