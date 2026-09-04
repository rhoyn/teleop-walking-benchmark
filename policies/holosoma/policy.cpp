namespace holosoma {

const std::string MODEL_PATH = "policies/holosoma/model.onnx";

const int HOLO_NUM_ACTIONS = 29;

const int HOLO_NUM_OBS = 100;

const double HOLO_ACTION_SCALE = 0.25;

const int HOLO_FIRST_ARM_MOTOR = 15;

const double HOLO_ARM_SOFT_START_S = 1.5;

const double HOLO_DEFAULT_ANGLES[HOLO_NUM_ACTIONS] = {
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669,
    -0.363, 0.0, 0.0, 0.0,   0.0,    0.2, 0.2,    0.0, 0.6, 0.0,
    0.0,    0.0, 0.2, -0.2,  0.0,    0.6, 0.0,    0.0, 0.0
};

const double HOLO_KPS[HOLO_NUM_ACTIONS] = {
    40.179238471, 99.098427777, 40.179238471, 99.098427777, 28.501246196,
    28.501246196, 40.179238471, 99.098427777, 40.179238471, 99.098427777,
    28.501246196, 28.501246196, 40.179238471, 28.501246196, 28.501246196,
    14.250623098, 14.250623098, 14.250623098, 14.250623098, 14.250623098,
    16.778327481, 16.778327481, 14.250623098, 14.250623098, 14.250623098,
    14.250623098, 14.250623098, 16.778327481, 16.778327481
};

const double HOLO_KDS[HOLO_NUM_ACTIONS] = {
    2.557889765, 6.308801854, 2.557889765, 6.308801854, 1.814445687,
    1.814445687, 2.557889765, 6.308801854, 2.557889765, 6.308801854,
    1.814445687, 1.814445687, 2.557889765, 1.814445687, 1.814445687,
    0.907222843, 0.907222843, 0.907222843, 0.907222843, 0.907222843,
    1.068141502, 1.068141502, 0.907222843, 0.907222843, 0.907222843,
    0.907222843, 0.907222843, 1.068141502, 1.068141502
};

const double HOLO_ANG_VEL_SCALE = 0.25;

const double HOLO_DOF_VEL_SCALE = 0.05;

const double HOLO_CONTROL_DT = 0.02;

const double HOLO_GAIT_PERIOD_S = 1.0;

const double HOLO_STAND_EPS = 0.01;

const double HOLO_MAX_SPEED = 1.0;

const double HOLO_MAX_YAW_RATE = 1.0;

const Limits LIMITS = Limits{
    -HOLO_MAX_SPEED,
    HOLO_MAX_SPEED,
    HOLO_MAX_SPEED,
    HOLO_MAX_YAW_RATE,
    0.0
};

struct Holosoma {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::vector<float> last_action_;
  double phase_[2] = {0.0, M_PI};
  bool standing_ = false;
};

std::shared_ptr<Engine> holo_engine_init(const std::string& model_path) {
  std::cout << "holosoma: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "holosoma: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(HOLO_NUM_OBS),
      true,
      "holosoma"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(HOLO_NUM_ACTIONS),
      false,
      "holosoma"
  );
  std::cout << "holosoma: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

template <typename Obs>
std::vector<float> holo_engine_run(
    Engine& s,
    const Obs& obs
) {
  return engine_run_single(s, obs.data(), obs.size());
}

std::vector<float> holo_build_obs(
    const Holosoma& p,
    const RobotState& rs,
    const double cmd[3]
) {
  std::vector<float> obs;
  obs.reserve(HOLO_NUM_OBS);
  for (int i = 0; i < HOLO_NUM_ACTIONS; ++i) obs.push_back(p.last_action_[i]);
  for (int k = 0; k < 3; ++k) {
    obs.push_back(static_cast<float>(rs.imu_gyro[k] * HOLO_ANG_VEL_SCALE));
  }
  obs.push_back(static_cast<float>(cmd[2]));
  obs.push_back(static_cast<float>(cmd[0]));
  obs.push_back(static_cast<float>(cmd[1]));
  obs.push_back(static_cast<float>(std::cos(p.phase_[0])));
  obs.push_back(static_cast<float>(std::cos(p.phase_[1])));
  for (int i = 0; i < HOLO_NUM_ACTIONS; ++i) {
    obs.push_back(static_cast<float>(rs.motor_q[i] - HOLO_DEFAULT_ANGLES[i]));
  }
  for (int i = 0; i < HOLO_NUM_ACTIONS; ++i) {
    obs.push_back(static_cast<float>(rs.motor_dq[i] * HOLO_DOF_VEL_SCALE));
  }
  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int k = 0; k < 3; ++k) {
    obs.push_back(static_cast<float>(gravity_dir[k]));
  }
  obs.push_back(static_cast<float>(std::sin(p.phase_[0])));
  obs.push_back(static_cast<float>(std::sin(p.phase_[1])));
  if (obs.size() != static_cast<size_t>(HOLO_NUM_OBS)) {
    throw std::runtime_error("holosoma: observation assembled wrong");
  }
  return obs;
}

void holo_update_phase(
    Holosoma& self,
    const double cmd[3]
) {
  const double phase_dt = 2.0 * M_PI / (HOLO_GAIT_PERIOD_S / HOLO_CONTROL_DT);
  for (int k = 0; k < 2; ++k) {
    double p = std::fmod(self.phase_[k] + phase_dt + M_PI, 2.0 * M_PI);
    if (p < 0.0) p += 2.0 * M_PI;
    self.phase_[k] = p - M_PI;
  }
  const bool stand = std::hypot(cmd[0], cmd[1]) < HOLO_STAND_EPS &&
                     std::abs(cmd[2]) < HOLO_STAND_EPS;
  if (stand) {
    self.phase_[0] = M_PI;
    self.phase_[1] = M_PI;
    self.standing_ = true;
  } else if (self.standing_) {
    self.phase_[0] = 0.0;
    self.phase_[1] = M_PI;
    self.standing_ = false;
  }
}

Output policy_step(
    Holosoma& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;
  const double cmd[3] = {drive[0], drive[1], drive[2]};

  holo_update_phase(self, cmd);

  const std::vector<float> obs = holo_build_obs(self, rs, cmd);
  self.last_action_ = holo_engine_run(*self.engine_, obs);

  for (int i = 0; i < HOLO_NUM_ACTIONS; ++i) {
    out.q_target[i] =
        HOLO_DEFAULT_ANGLES[i] +
        static_cast<double>(self.last_action_[i]) * HOLO_ACTION_SCALE;
    out.kp[i] = static_cast<float>(HOLO_KPS[i]);
    out.kd[i] = static_cast<float>(HOLO_KDS[i]);
    out.owns[i] = true;
  }

  const double ratio =
      std::clamp(in.control_time / HOLO_ARM_SOFT_START_S, 0.0, 1.0);

  if (ratio >= 1.0) {
    for (int i = HOLO_FIRST_ARM_MOTOR; i < HOLO_NUM_ACTIONS; ++i) {
      out.owns[i] = false;
    }
    return out;
  }

  for (int i = HOLO_FIRST_ARM_MOTOR; i < HOLO_NUM_ACTIONS; ++i) {
    out.q_target[i] =
        (1.0 - ratio) * out.q_target[i] + ratio * in.arm_targets[i];
    out.kp[i] = KPS[i];
    out.kd[i] = KDS[i];
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<Holosoma> policy_make() {
  const std::shared_ptr<Holosoma> self = std::make_shared<Holosoma>();
  self->last_action_ = std::vector<float>(HOLO_NUM_ACTIONS, 0.0f);

  self->engine_ = holo_engine_init(MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "holosoma"; }

 private:
  std::shared_ptr<Holosoma> state_;
};

}
