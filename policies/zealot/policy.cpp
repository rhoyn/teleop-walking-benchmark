namespace zealot {

const std::string MODEL_PATH = "policies/zealot/model.onnx";

const int ZEALOT_NUM_ACTIONS = 12;

const int ZEALOT_FRAME = 53;

const int ZEALOT_HISTORY = 5;

const int ZEALOT_NUM_OBS = ZEALOT_HISTORY * ZEALOT_FRAME;

const double ZEALOT_CONTROL_DT = 0.02;

const double ZEALOT_DEFAULT_POS[ZEALOT_NUM_ACTIONS] =
    {-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0};

const double ZEALOT_ACTION_SCALE = 0.5;

const double ZEALOT_Q_MIN[ZEALOT_NUM_ACTIONS] = {
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
    -0.2618
};

const double ZEALOT_Q_MAX[ZEALOT_NUM_ACTIONS] = {
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
    0.2618
};

const float ZEALOT_KP[ZEALOT_NUM_ACTIONS] = {
    100.0f,
    100.0f,
    100.0f,
    200.0f,
    40.0f,
    40.0f,
    100.0f,
    100.0f,
    100.0f,
    200.0f,
    40.0f,
    40.0f
};

const float ZEALOT_KD[ZEALOT_NUM_ACTIONS] =
    {2.5f, 2.5f, 2.5f, 5.0f, 2.0f, 2.0f, 2.5f, 2.5f, 2.5f, 5.0f, 2.0f, 2.0f};

const double ZEALOT_GAIT_PERIOD_SLOW = 0.8;

const double ZEALOT_GAIT_PERIOD_FAST = 0.55;

const double ZEALOT_GAIT_PERIOD_MIN = 0.40;

const double ZEALOT_GAIT_SPEED_CAP = 0.8;

const double ZEALOT_GAIT_STAND_SPEED = 0.1;

const double ZEALOT_VX_MIN = -0.8;

const double ZEALOT_VX_MAX = 0.8;

const double ZEALOT_VY_ABS = 0.3;

const double ZEALOT_YAW_RATE_ABS = 0.6;

const double ZEALOT_SPEED_NORM = 0.8;

const Limits LIMITS = Limits{
    ZEALOT_VX_MIN,
    ZEALOT_VX_MAX,
    ZEALOT_VY_ABS,
    ZEALOT_YAW_RATE_ABS,
    ZEALOT_SPEED_NORM
};

const double ZEALOT_FACE_FAR_M = 1.5;

const double ZEALOT_FACE_NEAR_M = 0.4;

const double ZEALOT_WALK_SPEED = 0.5;

const double ZEALOT_WALK_P = 1.5;

const double ZEALOT_YAW_P = 1.2;

struct Zealot {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;

  std::array<float, ZEALOT_NUM_OBS> obs_{};

  std::array<float, ZEALOT_NUM_ACTIONS> action_lag2_{};
  std::array<float, ZEALOT_NUM_ACTIONS> action_lag1_{};

  std::array<double, ZEALOT_NUM_ACTIONS> prev_q_{};

  double gait_phase_ = 0.0;

  int step_ = 0;
};

std::shared_ptr<Engine> zealot_engine_init(const std::string& model_path) {
  std::cout << "zealot: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "zealot: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(ZEALOT_NUM_OBS),
      true,
      "zealot"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(ZEALOT_NUM_ACTIONS),
      false,
      "zealot"
  );
  std::cout << "zealot: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

double zealot_gait_period(double cmd_speed) {
  const double t =
      (std::min(std::fabs(cmd_speed), ZEALOT_GAIT_SPEED_CAP) - 0.1) / 0.4;
  const double period =
      ZEALOT_GAIT_PERIOD_SLOW +
      (ZEALOT_GAIT_PERIOD_FAST - ZEALOT_GAIT_PERIOD_SLOW) * std::max(t, 0.0);
  return std::max(ZEALOT_GAIT_PERIOD_MIN, period);
}

void zealot_shape_command(
    const Zealot& self,
    const Input& in,
    double drive[3]
) {
  if (!in.has_target) return;

  double bearing = 0.0;
  double face_w = 0.0;
  if (!self.pos_reached_) {
    bearing = std::atan2(in.target_left_m, in.target_forward_m);
    face_w = std::clamp(
        (in.dist_m - ZEALOT_FACE_NEAR_M) /
            (ZEALOT_FACE_FAR_M - ZEALOT_FACE_NEAR_M),
        0.0,
        1.0
    );

    const double gate = std::max(std::cos(bearing), 0.0);
    const double speed =
        gate * std::min(ZEALOT_WALK_P * in.dist_m, ZEALOT_WALK_SPEED);
    drive[0] = speed;
    drive[1] =
        std::clamp(speed * std::sin(bearing), -ZEALOT_VY_ABS, ZEALOT_VY_ABS);
  }

  const double aim =
      face_w > 0.0
          ? std::remainder(
                in.yaw_err_rad +
                    face_w *
                        std::remainder(bearing - in.yaw_err_rad, 2.0 * M_PI),
                2.0 * M_PI
            )
          : (self.yaw_reached_ ? 0.0 : in.yaw_err_rad);
  drive[2] =
      std::clamp(ZEALOT_YAW_P * aim, -ZEALOT_YAW_RATE_ABS, ZEALOT_YAW_RATE_ABS);
}

void zealot_obs_frame(
    const Zealot& p,
    const RobotState& rs,
    const double drive[3],
    float frame[ZEALOT_FRAME]
) {
  std::fill(frame, frame + ZEALOT_FRAME, 0.0f);

  if (p.step_ >= 2) {
    std::copy(p.action_lag2_.begin(), p.action_lag2_.end(), frame);
  }

  frame[12] = static_cast<float>(drive[0]);
  frame[13] = static_cast<float>(drive[1]);
  frame[14] = static_cast<float>(drive[2]);

  for (int i = 0; i < ZEALOT_NUM_ACTIONS; ++i) {
    frame[16 + i] = static_cast<float>(rs.motor_q[i] - ZEALOT_DEFAULT_POS[i]);
    frame[28 + i] = p.step_ == 0
                        ? 0.0f
                        : static_cast<float>(
                              (rs.motor_q[i] - p.prev_q_[i]) / ZEALOT_CONTROL_DT
                          );
  }

  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int k = 0; k < 3; ++k) {
    frame[40 + k] = static_cast<float>(gravity_dir[k]);
    frame[45 + k] = static_cast<float>(rs.imu_gyro[k]);
  }

  frame[43] = static_cast<float>(std::sin(2.0 * M_PI * p.gait_phase_));
  frame[44] = static_cast<float>(std::cos(2.0 * M_PI * p.gait_phase_));
}

Output policy_step(
    Zealot& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  zealot_shape_command(self, in, drive);

  const RobotState& rs = *in.state;

  float frame[ZEALOT_FRAME];
  zealot_obs_frame(self, rs, drive, frame);

  if (self.step_ == 0) {
    for (int h = 0; h < ZEALOT_HISTORY; ++h) {
      std::copy(
          frame,
          frame + ZEALOT_FRAME,
          self.obs_.begin() + h * ZEALOT_FRAME
      );
    }
  } else {
    std::copy(
        self.obs_.begin() + ZEALOT_FRAME,
        self.obs_.end(),
        self.obs_.begin()
    );
    std::copy(frame, frame + ZEALOT_FRAME, self.obs_.end() - ZEALOT_FRAME);
  }

  const std::vector<float>& action =
      engine_run_single(*self.engine_, self.obs_.data(), self.obs_.size());

  Output out{};
  for (int i = 0; i < ZEALOT_NUM_ACTIONS; ++i) {
    out.q_target[i] = std::clamp(
        ZEALOT_DEFAULT_POS[i] +
            ZEALOT_ACTION_SCALE * static_cast<double>(action[i]),
        ZEALOT_Q_MIN[i],
        ZEALOT_Q_MAX[i]
    );
    out.kp[i] = ZEALOT_KP[i];
    out.kd[i] = ZEALOT_KD[i];
    out.owns[i] = true;
  }

  const double cmd_speed = std::sqrt(
      drive[0] * drive[0] + drive[1] * drive[1] + drive[2] * drive[2]
  );
  if (self.step_ >= 1 && cmd_speed >= ZEALOT_GAIT_STAND_SPEED) {
    self.gait_phase_ = std::fmod(
        self.gait_phase_ + ZEALOT_CONTROL_DT / zealot_gait_period(cmd_speed),
        1.0
    );
  }

  self.action_lag2_ = self.action_lag1_;
  std::copy(
      action.begin(),
      action.begin() + ZEALOT_NUM_ACTIONS,
      self.action_lag1_.begin()
  );
  for (int i = 0; i < ZEALOT_NUM_ACTIONS; ++i) self.prev_q_[i] = rs.motor_q[i];
  ++self.step_;

  return out;
}

std::shared_ptr<Zealot> policy_make() {
  const std::shared_ptr<Zealot> self = std::make_shared<Zealot>();
  self->engine_ = zealot_engine_init(MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "zealot"; }

 private:
  std::shared_ptr<Zealot> state_;
};

}
