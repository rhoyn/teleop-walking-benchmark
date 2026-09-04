namespace rl_gym {

const int RLGYM_NUM_LEG = 12;

const std::string RLGYM_MODEL_PATH = "policies/rl_gym/model.onnx";

const int RLGYM_HIDDEN = 64;

const int RLGYM_NUM_OBS = 47;

const int RLGYM_WAIST_BEGIN = 12;

const int RLGYM_WAIST_END = 15;

const double RLGYM_DEFAULT_ANGLES[RLGYM_NUM_LEG] =
    {-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0};

const float RLGYM_WAIST_KP = 300.0f;

const float RLGYM_WAIST_KD = 3.0f;

const double RLGYM_CONTROL_DT = 0.02;

const float RLGYM_KPS[RLGYM_NUM_LEG] =
    {100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40};

const float RLGYM_KDS[RLGYM_NUM_LEG] = {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2};

const double RLGYM_ANG_VEL_SCALE = 0.25;

const double RLGYM_DOF_VEL_SCALE = 0.05;

const double RLGYM_ACTION_SCALE = 0.25;

const double RLGYM_CMD_SCALE[3] = {2.0, 2.0, 0.25};

const double RLGYM_GAIT_PERIOD = 0.8;

const double RLGYM_MAX_CMD[3] = {0.8, 0.5, 1.57};

const double RLGYM_POS_REACHED_ENTER_M = 0.08;

const double RLGYM_POS_REACHED_EXIT_M = 0.15;

const double RLGYM_YAW_REACHED_ENTER_RAD = 0.05;

const double RLGYM_YAW_REACHED_EXIT_RAD = 0.10;

const double RLGYM_WALK_KP_POS = 1.5;

const double RLGYM_WALK_KP_YAW = 2.0;

const Limits LIMITS = [] {
  Limits l{};
  l.vx_min = -RLGYM_MAX_CMD[0];
  l.vx_max = RLGYM_MAX_CMD[0];
  l.vy_abs = RLGYM_MAX_CMD[1];
  l.yaw_rate_abs = RLGYM_MAX_CMD[2];
  l.speed_norm = 0.0;
  l.pos_reached_enter_m = RLGYM_POS_REACHED_ENTER_M;
  l.pos_reached_exit_m = RLGYM_POS_REACHED_EXIT_M;
  l.yaw_reached_enter_rad = RLGYM_YAW_REACHED_ENTER_RAD;
  l.yaw_reached_exit_rad = RLGYM_YAW_REACHED_EXIT_RAD;
  l.walk_kp_pos = RLGYM_WALK_KP_POS;
  l.walk_kp_yaw = RLGYM_WALK_KP_YAW;
  return l;
}();

struct RlGym {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::array<float, RLGYM_NUM_LEG> last_action_;

  std::array<float, RLGYM_HIDDEN> hidden_{};
  std::array<float, RLGYM_HIDDEN> cell_{};
  int64_t phase_counter_ = 0;
};

Output policy_step(
    RlGym& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  const double phase =
      std::fmod(
          static_cast<double>(self.phase_counter_) * RLGYM_CONTROL_DT,
          RLGYM_GAIT_PERIOD
      ) /
      RLGYM_GAIT_PERIOD;
  self.phase_counter_++;

  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  std::array<float, RLGYM_NUM_OBS> obs{};
  for (int k = 0; k < 3; ++k) {
    obs[k] = static_cast<float>(rs.imu_gyro[k] * RLGYM_ANG_VEL_SCALE);
    obs[3 + k] = static_cast<float>(gravity_dir[k]);
  }
  obs[6] = static_cast<float>(drive[0] * RLGYM_CMD_SCALE[0]);
  obs[7] = static_cast<float>(drive[1] * RLGYM_CMD_SCALE[1]);
  obs[8] = static_cast<float>(drive[2] * RLGYM_CMD_SCALE[2]);
  for (int j = 0; j < RLGYM_NUM_LEG; ++j) {
    obs[9 + j] = static_cast<float>(rs.motor_q[j] - RLGYM_DEFAULT_ANGLES[j]);
    obs[9 + RLGYM_NUM_LEG + j] =
        static_cast<float>(rs.motor_dq[j] * RLGYM_DOF_VEL_SCALE);
    obs[9 + 2 * RLGYM_NUM_LEG + j] = self.last_action_[j];
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
    std::copy(
        action.begin(),
        action.begin() + RLGYM_NUM_LEG,
        self.last_action_.begin()
    );
    const std::vector<float>& h = engine_output(e, "hidden_out").data;
    const std::vector<float>& c = engine_output(e, "cell_out").data;
    std::copy(h.begin(), h.end(), self.hidden_.begin());
    std::copy(c.begin(), c.end(), self.cell_.begin());
  }

  for (int i = 0; i < RLGYM_NUM_LEG; ++i) {
    out.q_target[i] =
        RLGYM_DEFAULT_ANGLES[i] +
        static_cast<double>(self.last_action_[i]) * RLGYM_ACTION_SCALE;
    out.kp[i] = RLGYM_KPS[i];
    out.kd[i] = RLGYM_KDS[i];
    out.owns[i] = true;
  }
  for (int i = RLGYM_WAIST_BEGIN; i < RLGYM_WAIST_END; ++i) {
    out.q_target[i] = 0.0;
    out.kp[i] = RLGYM_WAIST_KP;
    out.kd[i] = RLGYM_WAIST_KD;
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<RlGym> policy_make() {
  const std::shared_ptr<RlGym> self = std::make_shared<RlGym>();

  self->last_action_.fill(0.0f);
  self->hidden_.fill(0.0f);
  self->cell_.fill(0.0f);

  std::cout << "rl_gym: loading policy model " << RLGYM_MODEL_PATH << std::endl;
  self->engine_ = engine_init(RLGYM_MODEL_PATH);
  Engine& e = *self->engine_;
  engine_expect(e, "obs", RLGYM_NUM_OBS, true, "rl_gym");
  engine_expect(e, "hidden_in", RLGYM_HIDDEN, true, "rl_gym");
  engine_expect(e, "cell_in", RLGYM_HIDDEN, true, "rl_gym");
  engine_expect(e, "action", RLGYM_NUM_LEG, false, "rl_gym");
  engine_expect(e, "hidden_out", RLGYM_HIDDEN, false, "rl_gym");
  engine_expect(e, "cell_out", RLGYM_HIDDEN, false, "rl_gym");
  std::cout << "rl_gym: engine ready, obs " << RLGYM_NUM_OBS << " action "
            << RLGYM_NUM_LEG << " (lstm state carried by the caller)"
            << std::endl;
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "rl_gym"; }

 private:
  std::shared_ptr<RlGym> state_;
};

}
