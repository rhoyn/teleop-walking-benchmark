namespace openwbt {

const int OPENWBT_NUM_ACTIONS = 12;

const double OPENWBT_GAIT_STANCE_MIDDLE = 0.3;

const std::string OPENWBT_POLICY_PATH = "policies/openwbt/model.onnx";

const int OPENWBT_NUM_OBS = 47;

const int OPENWBT_HIDDEN_DIM = 256;

const int OPENWBT_FIRST_ARM = 15;

const double OPENWBT_DEFAULT_LEG[OPENWBT_NUM_ACTIONS] =
    {-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0};

const int OPENWBT_FIRST_WAIST = 12;

const float OPENWBT_LEG_KP[OPENWBT_NUM_ACTIONS] =
    {100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40};

const float OPENWBT_LEG_KD[OPENWBT_NUM_ACTIONS] =
    {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2};

const float OPENWBT_WAIST_KP = 300.0f;

const float OPENWBT_WAIST_KD = 3.0f;

const double OPENWBT_ANG_VEL_SCALE = 0.25;

const double OPENWBT_DOF_VEL_SCALE = 0.05;

const double OPENWBT_ACTION_SCALE = 0.25;

const double OPENWBT_CMD_SCALE[3] = {2.0, 2.0, 0.25};

const double OPENWBT_CLIP_ACTIONS = 100.0;

const double CONTROL_DT = 0.02;

const double OPENWBT_GAIT_FREQ = 1.5;

const double OPENWBT_GAIT_PHASE_OFFSET = 0.5;

const double OPENWBT_GAIT_STANCE_RATIO = 0.6;

const double OPENWBT_CLIP_OBS = 100.0;

const double OPENWBT_MAX_CMD = 0.3;

const double OPENWBT_CMD_CLIP = 0.1;

const double OPENWBT_WALK_KP_POS = 2.0;

const double OPENWBT_WALK_KP_YAW = 2.0;

struct GaitClock {
  double gait_index = OPENWBT_GAIT_STANCE_MIDDLE;
  double clock[2] = {0.0, 0.0};
};

const Limits LIMITS = [] {
  Limits l{};

  l.vx_min = -OPENWBT_MAX_CMD;
  l.vx_max = OPENWBT_MAX_CMD;
  l.vy_abs = OPENWBT_MAX_CMD;
  l.yaw_rate_abs = OPENWBT_MAX_CMD;

  l.speed_norm = 0.0;

  l.walk_kp_pos = OPENWBT_WALK_KP_POS;
  l.walk_kp_yaw = OPENWBT_WALK_KP_YAW;

  l.pos_reached_enter_m = OPENWBT_CMD_CLIP / OPENWBT_WALK_KP_POS;
  l.pos_reached_exit_m = 2.0 * OPENWBT_CMD_CLIP / OPENWBT_WALK_KP_POS;

  l.yaw_reached_enter_rad = OPENWBT_CMD_CLIP / OPENWBT_WALK_KP_YAW;
  l.yaw_reached_exit_rad = 2.0 * OPENWBT_CMD_CLIP / OPENWBT_WALK_KP_YAW;

  return l;
}();

struct OpenWbt {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  GaitClock gait_;
  std::array<float, OPENWBT_NUM_ACTIONS> last_action_{};
};

std::shared_ptr<Engine> openwbt_engine_init(const std::string& model_path) {
  std::cout << "openwbt: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  engine_expect(
      *loaded,
      "obs",
      static_cast<size_t>(OPENWBT_NUM_OBS),
      true,
      "openwbt"
  );
  engine_expect(
      *loaded,
      "input_hidden_states",
      static_cast<size_t>(OPENWBT_HIDDEN_DIM),
      true,
      "openwbt"
  );
  engine_expect(
      *loaded,
      "action",
      static_cast<size_t>(OPENWBT_NUM_ACTIONS),
      false,
      "openwbt"
  );
  engine_expect(
      *loaded,
      "output_hidden_states",
      static_cast<size_t>(OPENWBT_HIDDEN_DIM),
      false,
      "openwbt"
  );
  std::cout << "openwbt: engine ready, obs " << OPENWBT_NUM_OBS << " action "
            << OPENWBT_NUM_ACTIONS << " hidden " << OPENWBT_HIDDEN_DIM
            << std::endl;
  return loaded;
}

void gait_update(
    GaitClock& g,
    bool stop
) {
  g.gait_index = std::fmod(g.gait_index + CONTROL_DT * OPENWBT_GAIT_FREQ, 1.0);
  double foot[2];
  foot[0] = std::fmod(g.gait_index + OPENWBT_GAIT_PHASE_OFFSET, 1.0);
  foot[1] = g.gait_index;
  if (stop) {
    g.gait_index = OPENWBT_GAIT_STANCE_MIDDLE;
    foot[0] = OPENWBT_GAIT_STANCE_MIDDLE;
    foot[1] = OPENWBT_GAIT_STANCE_MIDDLE;
  }
  for (int i = 0; i < 2; ++i) {
    const double idx = foot[i];
    const double scaled = idx < OPENWBT_GAIT_STANCE_RATIO
                              ? 0.5 * idx / OPENWBT_GAIT_STANCE_RATIO
                              : 0.5 + 0.5 * (idx - OPENWBT_GAIT_STANCE_RATIO) /
                                          (1.0 - OPENWBT_GAIT_STANCE_RATIO);
    g.clock[i] = std::sin(2.0 * M_PI * scaled);
  }
}

void openwbt_engine_step(Engine& s) {
  engine_run(s);
  const std::vector<float>& next =
      engine_output(s, "output_hidden_states").data;
  std::vector<float>& hidden = engine_input(s, "input_hidden_states").data;
  std::copy(next.begin(), next.end(), hidden.begin());
}

float openwbt_clip_obs(double v) {
  return static_cast<float>(std::clamp(v, -OPENWBT_CLIP_OBS, OPENWBT_CLIP_OBS));
}

Output policy_step(
    OpenWbt& self,
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
  Output out{};

  if (!self.engine_) {
    throw std::runtime_error("openwbt: engine not ready");
  }
  std::vector<float>& obs = engine_input(*self.engine_, "obs").data;
  const RobotState& rs = *in.state;
  const double cmd[3] = {drive[0], drive[1], drive[2]};

  const bool stance =
      !walking || (cmd[0] == 0.0 && cmd[1] == 0.0 && cmd[2] == 0.0);
  gait_update(self.gait_, stance);

  for (int i = 0; i < 3; ++i) {
    obs[i] = openwbt_clip_obs(cmd[i] * OPENWBT_CMD_SCALE[i]);
  }
  const Eigen::Vector3d gravity =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int i = 0; i < 3; ++i) {
    obs[3 + i] = openwbt_clip_obs(gravity[i]);
    obs[6 + i] = openwbt_clip_obs(rs.imu_gyro[i] * OPENWBT_ANG_VEL_SCALE);
  }
  for (int i = 0; i < OPENWBT_NUM_ACTIONS; ++i) {
    obs[9 + i] = openwbt_clip_obs(rs.motor_q[i] - OPENWBT_DEFAULT_LEG[i]);
    obs[21 + i] = openwbt_clip_obs(rs.motor_dq[i] * OPENWBT_DOF_VEL_SCALE);
    obs[33 + i] = openwbt_clip_obs(self.last_action_[i]);
  }
  obs[45] = openwbt_clip_obs(self.gait_.clock[0]);
  obs[46] = openwbt_clip_obs(self.gait_.clock[1]);

  openwbt_engine_step(*self.engine_);
  const std::vector<float>& action =
      engine_output(*self.engine_, "action").data;

  for (int i = 0; i < OPENWBT_NUM_ACTIONS; ++i) {
    const double a = std::clamp(
        static_cast<double>(action[i]),
        -OPENWBT_CLIP_ACTIONS,
        OPENWBT_CLIP_ACTIONS
    );
    self.last_action_[i] = static_cast<float>(a);
    out.q_target[i] = OPENWBT_DEFAULT_LEG[i] + a * OPENWBT_ACTION_SCALE;
    out.kp[i] = OPENWBT_LEG_KP[i];
    out.kd[i] = OPENWBT_LEG_KD[i];
    out.owns[i] = true;
  }
  for (int i = OPENWBT_FIRST_WAIST; i < OPENWBT_FIRST_ARM; ++i) {
    out.q_target[i] = 0.0;
    out.kp[i] = OPENWBT_WAIST_KP;
    out.kd[i] = OPENWBT_WAIST_KD;
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<OpenWbt> policy_make() {
  const std::shared_ptr<OpenWbt> self = std::make_shared<OpenWbt>();

  self->engine_ = openwbt_engine_init(OPENWBT_POLICY_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "openwbt"; }

 private:
  std::shared_ptr<OpenWbt> state_;
};

}
