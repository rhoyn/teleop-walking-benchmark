namespace stepdown {

const int STEPDOWN_NUM_LEG = 12;

const std::string STEPDOWN_MODEL_PATH = "policies/stepdown/model.onnx";

const int STEPDOWN_HIDDEN = 64;

const int STEPDOWN_NUM_OBS = 47;

const int STEPDOWN_WAIST_BEGIN = 12;

const int STEPDOWN_WAIST_END = 15;

const float STEPDOWN_WAIST_KP = 300.0f;

const float STEPDOWN_WAIST_KD = 3.0f;

const double STEPDOWN_DEFAULT_ANGLES[STEPDOWN_NUM_LEG] =
    {-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0};

const double STEPDOWN_CONTROL_DT = 0.02;

const float STEPDOWN_KPS[STEPDOWN_NUM_LEG] =
    {100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40};

const float STEPDOWN_KDS[STEPDOWN_NUM_LEG] =
    {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2};

const double STEPDOWN_ANG_VEL_SCALE = 0.25;

const double STEPDOWN_DOF_VEL_SCALE = 0.05;

const double STEPDOWN_ACTION_SCALE = 0.25;

const double STEPDOWN_CMD_SCALE[3] = {2.0, 2.0, 0.25};

const double STEPDOWN_GAIT_PERIOD = 0.8;

const float STEPDOWN_ACTION_CLIP = 1.0f;

const double STEPDOWN_MAX_CMD[3] = {0.4, 0.3, 1.57};

const double STEPDOWN_POS_REACHED_ENTER_M = 0.08;

const double STEPDOWN_POS_REACHED_EXIT_M = 0.15;

const double STEPDOWN_YAW_REACHED_ENTER_RAD = 0.05;

const double STEPDOWN_YAW_REACHED_EXIT_RAD = 0.10;

const double STEPDOWN_WALK_KP_POS = 1.5;

const double STEPDOWN_WALK_KP_YAW = 2.0;

const Limits LIMITS = [] {
  Limits l{};
  l.vx_min = -STEPDOWN_MAX_CMD[0];
  l.vx_max = STEPDOWN_MAX_CMD[0];
  l.vy_abs = STEPDOWN_MAX_CMD[1];
  l.yaw_rate_abs = STEPDOWN_MAX_CMD[2];
  l.speed_norm = 0.0;
  l.pos_reached_enter_m = STEPDOWN_POS_REACHED_ENTER_M;
  l.pos_reached_exit_m = STEPDOWN_POS_REACHED_EXIT_M;
  l.yaw_reached_enter_rad = STEPDOWN_YAW_REACHED_ENTER_RAD;
  l.yaw_reached_exit_rad = STEPDOWN_YAW_REACHED_EXIT_RAD;
  l.walk_kp_pos = STEPDOWN_WALK_KP_POS;
  l.walk_kp_yaw = STEPDOWN_WALK_KP_YAW;
  return l;
}();

struct Stepdown {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::array<float, STEPDOWN_NUM_LEG> last_action_;

  std::array<float, STEPDOWN_HIDDEN> hidden_{};
  std::array<float, STEPDOWN_HIDDEN> cell_{};
  int64_t phase_counter_ = 0;
};

Output policy_step(
    Stepdown& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  const double phase =
      std::fmod(
          static_cast<double>(self.phase_counter_) * STEPDOWN_CONTROL_DT,
          STEPDOWN_GAIT_PERIOD
      ) /
      STEPDOWN_GAIT_PERIOD;
  self.phase_counter_++;

  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  std::array<float, STEPDOWN_NUM_OBS> obs{};
  for (int k = 0; k < 3; ++k) {
    obs[k] = static_cast<float>(rs.imu_gyro[k] * STEPDOWN_ANG_VEL_SCALE);
    obs[3 + k] = static_cast<float>(gravity_dir[k]);
  }
  obs[6] = static_cast<float>(drive[0] * STEPDOWN_CMD_SCALE[0]);
  obs[7] = static_cast<float>(drive[1] * STEPDOWN_CMD_SCALE[1]);
  obs[8] = static_cast<float>(drive[2] * STEPDOWN_CMD_SCALE[2]);
  for (int j = 0; j < STEPDOWN_NUM_LEG; ++j) {
    obs[9 + j] = static_cast<float>(rs.motor_q[j] - STEPDOWN_DEFAULT_ANGLES[j]);
    obs[9 + STEPDOWN_NUM_LEG + j] =
        static_cast<float>(rs.motor_dq[j] * STEPDOWN_DOF_VEL_SCALE);
    obs[9 + 2 * STEPDOWN_NUM_LEG + j] = self.last_action_[j];
  }
  obs[45] = static_cast<float>(std::sin(2.0 * M_PI * phase));
  obs[46] = static_cast<float>(std::cos(2.0 * M_PI * phase));

  {
    Engine& e = *self.engine_;
    std::copy(obs.begin(), obs.end(), engine_input(e, "obs").data.begin());
    std::copy(
        self.hidden_.begin(),
        self.hidden_.end(),
        engine_input(e, "hidden_in").data.begin()
    );
    std::copy(
        self.cell_.begin(),
        self.cell_.end(),
        engine_input(e, "cell_in").data.begin()
    );
    engine_run(e);
    const std::vector<float>& action = engine_output(e, "action").data;
    for (int i = 0; i < STEPDOWN_NUM_LEG; ++i) {
      self.last_action_[i] =
          std::clamp(action[i], -STEPDOWN_ACTION_CLIP, STEPDOWN_ACTION_CLIP);
    }
    const std::vector<float>& h = engine_output(e, "hidden_out").data;
    const std::vector<float>& c = engine_output(e, "cell_out").data;
    std::copy(h.begin(), h.end(), self.hidden_.begin());
    std::copy(c.begin(), c.end(), self.cell_.begin());
  }

  for (int i = 0; i < STEPDOWN_NUM_LEG; ++i) {
    out.q_target[i] =
        STEPDOWN_DEFAULT_ANGLES[i] +
        static_cast<double>(self.last_action_[i]) * STEPDOWN_ACTION_SCALE;
    out.kp[i] = STEPDOWN_KPS[i];
    out.kd[i] = STEPDOWN_KDS[i];
    out.owns[i] = true;
  }
  for (int i = STEPDOWN_WAIST_BEGIN; i < STEPDOWN_WAIST_END; ++i) {
    out.q_target[i] = 0.0;
    out.kp[i] = STEPDOWN_WAIST_KP;
    out.kd[i] = STEPDOWN_WAIST_KD;
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<Stepdown> policy_make() {
  const std::shared_ptr<Stepdown> self = std::make_shared<Stepdown>();

  self->last_action_.fill(0.0f);
  self->hidden_.fill(0.0f);
  self->cell_.fill(0.0f);

  std::cout << "stepdown: loading policy model " << STEPDOWN_MODEL_PATH
            << std::endl;
  self->engine_ = engine_init(STEPDOWN_MODEL_PATH);
  Engine& e = *self->engine_;
  engine_expect(e, "obs", STEPDOWN_NUM_OBS, true, "stepdown");
  engine_expect(e, "hidden_in", STEPDOWN_HIDDEN, true, "stepdown");
  engine_expect(e, "cell_in", STEPDOWN_HIDDEN, true, "stepdown");
  engine_expect(e, "action", STEPDOWN_NUM_LEG, false, "stepdown");
  engine_expect(e, "hidden_out", STEPDOWN_HIDDEN, false, "stepdown");
  engine_expect(e, "cell_out", STEPDOWN_HIDDEN, false, "stepdown");
  std::cout << "stepdown: engine ready, obs " << STEPDOWN_NUM_OBS << " action "
            << STEPDOWN_NUM_LEG << " (lstm state carried by the caller)"
            << std::endl;
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "stepdown"; }

 private:
  std::shared_ptr<Stepdown> state_;
};

}
