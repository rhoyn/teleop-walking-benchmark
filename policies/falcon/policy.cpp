namespace falcon {

const std::string MODEL_PATH = "policies/falcon/model.onnx";

const int FALCON_NUM_LOWER = 15;

const int FALCON_OBS_DIM = 102;

const double FALCON_DEFAULT_ANGLES[FALCON_NUM_LOWER] = {
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

const double FALCON_KPS[FALCON_NUM_LOWER] = {
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
    300.0,
    300.0,
    300.0
};

const double FALCON_KDS[FALCON_NUM_LOWER] =
    {2.5, 2.5, 2.5, 5.0, 0.2, 0.1, 2.5, 2.5, 2.5, 5.0, 0.2, 0.1, 5.0, 5.0, 5.0};

const double FALCON_ACTION_SCALE = 0.25;

const double FALCON_GAIT_PERIOD = 0.9;

const double FALCON_DT = 0.02;

const double FALCON_WARMUP_S = 1.5;

const double FALCON_POS_LOWER[FALCON_NUM_LOWER] = {
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.5307,
    -2.9671,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.618,
    -0.52,
    -0.52
};

const double FALCON_POS_UPPER[FALCON_NUM_LOWER] = {
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.8798,
    0.5236,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.618,
    0.52,
    0.52
};

const int FALCON_NUM_UPPER = 14;

const bool FALCON_MASK_UPPER_OBS = true;

const double FALCON_SCALE_ANG_VEL = 0.25;

const double FALCON_SCALE_DOF_VEL = 0.05;

const double FALCON_VX_MAX = 0.9;

const double FALCON_VX_MIN = -0.6;

const double FALCON_VY_ABS = 0.5;

const double FALCON_YAW_ABS = 0.8;

const double FALCON_SPEED_NORM = 0.9;

const double FALCON_WALK_KP_POS = 2.0;

const double FALCON_CMD_KP_POS = 2.0;

const Limits LIMITS = [] {
  Limits l{
      FALCON_VX_MIN,
      FALCON_VX_MAX,
      FALCON_VY_ABS,
      FALCON_YAW_ABS,
      FALCON_SPEED_NORM
  };
  l.walk_kp_pos = FALCON_WALK_KP_POS;
  return l;
}();

struct Falcon {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::vector<float> obs_;
  std::vector<double> last_action_;
  double phase_clock_ = 0.0;
};

std::shared_ptr<Engine> falcon_engine_init(const std::string& model_path) {
  std::cout << "falcon: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "falcon: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(FALCON_OBS_DIM),
      true,
      "falcon"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(FALCON_NUM_LOWER),
      false,
      "falcon"
  );
  std::cout << "falcon: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

template <typename Obs>
std::vector<float> falcon_engine_run(
    Engine& s,
    const Obs& obs
) {
  return engine_run_single(s, obs.data(), obs.size());
}

void falcon_build_obs(
    Falcon& p,
    const RobotState& rs,
    double vx,
    double vy,
    double wz,
    double stand,
    double phase
) {
  int k = 0;

  for (int i = 0; i < FALCON_NUM_LOWER; ++i) {
    p.obs_[k++] = static_cast<float>(p.last_action_[i]);
  }

  for (int i = 0; i < 3; ++i) {
    p.obs_[k++] = static_cast<float>(rs.imu_gyro[i] * FALCON_SCALE_ANG_VEL);
  }

  p.obs_[k++] = static_cast<float>(wz);
  p.obs_[k++] = static_cast<float>(vx);
  p.obs_[k++] = static_cast<float>(vy);
  p.obs_[k++] = static_cast<float>(stand);

  for (int i = 0; i < 3; ++i) p.obs_[k++] = 0.0f;

  p.obs_[k++] = static_cast<float>(std::cos(2.0 * M_PI * phase));

  for (int i = 0; i < NUM_MOTOR; ++i) {
    const bool mask = FALCON_MASK_UPPER_OBS && i >= FALCON_NUM_LOWER;
    const double def = i < FALCON_NUM_LOWER ? FALCON_DEFAULT_ANGLES[i] : 0.0;
    p.obs_[k++] = mask ? 0.0f : static_cast<float>(rs.motor_q[i] - def);
  }

  for (int i = 0; i < NUM_MOTOR; ++i) {
    const bool mask = FALCON_MASK_UPPER_OBS && i >= FALCON_NUM_LOWER;
    p.obs_[k++] =
        mask ? 0.0f : static_cast<float>(rs.motor_dq[i] * FALCON_SCALE_DOF_VEL);
  }

  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int i = 0; i < 3; ++i) {
    p.obs_[k++] = static_cast<float>(gravity_dir[i]);
  }

  for (int i = 0; i < FALCON_NUM_UPPER; ++i) p.obs_[k++] = 0.0f;

  p.obs_[k++] = static_cast<float>(std::sin(2.0 * M_PI * phase));

  if (k != FALCON_OBS_DIM) {
    throw std::runtime_error("falcon: observation dimension");
  }
}

Output policy_step(
    Falcon& self,
    const Input& in
) {
  double drive[3];
  const bool walking = command_from_target(
      in,
      LIMITS,
      self.pos_reached_,
      self.yaw_reached_,
      drive
  );
  [&] {
    (void)in.yaw_err_rad;
    const double speed = std::hypot(drive[0], drive[1]);
    if (speed < 1e-9) return;
    const double want =
        std::min(FALCON_CMD_KP_POS * in.dist_m, FALCON_SPEED_NORM);
    const double scale = want / speed;
    drive[0] = std::clamp(drive[0] * scale, FALCON_VX_MIN, FALCON_VX_MAX);
    drive[1] = std::clamp(drive[1] * scale, -FALCON_VY_ABS, FALCON_VY_ABS);
  }();
  Output out{};

  const RobotState& rs = *in.state;

  const bool stand = walking && in.control_time >= FALCON_WARMUP_S;
  const double cmd_vx = stand ? drive[0] : 0.0;
  const double cmd_vy = stand ? drive[1] : 0.0;
  const double cmd_wz = stand ? drive[2] : 0.0;

  if (stand) self.phase_clock_ += FALCON_DT;
  const double phase =
      std::fmod(self.phase_clock_, FALCON_GAIT_PERIOD) / FALCON_GAIT_PERIOD;

  falcon_build_obs(self, rs, cmd_vx, cmd_vy, cmd_wz, stand ? 1.0 : 0.0, phase);

  const std::vector<float> action = falcon_engine_run(*self.engine_, self.obs_);

  for (int i = 0; i < FALCON_NUM_LOWER; ++i) {
    self.last_action_[i] =
        std::clamp(static_cast<double>(action[i]), -100.0, 100.0);
    out.q_target[i] = std::clamp(
        FALCON_DEFAULT_ANGLES[i] + self.last_action_[i] * FALCON_ACTION_SCALE,
        FALCON_POS_LOWER[i],
        FALCON_POS_UPPER[i]
    );
    out.kp[i] = static_cast<float>(FALCON_KPS[i]);
    out.kd[i] = static_cast<float>(FALCON_KDS[i]);
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<Falcon> policy_make() {
  const std::shared_ptr<Falcon> self = std::make_shared<Falcon>();
  self->obs_ = std::vector<float>(FALCON_OBS_DIM, 0.0f);
  self->last_action_ = std::vector<double>(FALCON_NUM_LOWER, 0.0);

  self->engine_ = falcon_engine_init(MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "falcon"; }

 private:
  std::shared_ptr<Falcon> state_;
};

}
