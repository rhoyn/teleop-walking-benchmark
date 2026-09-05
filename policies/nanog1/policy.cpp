namespace nanog1 {

const std::string NANOG1_MODEL_PATH = "policies/nanog1/model.onnx";

const int NANOG1_NUM_OBS = 98;

const int NANOG1_NUM_ACTIONS = 29;

const int NANOG1_NUM_LEG = 12;

const int NANOG1_STATE = 384;

const double NANOG1_HOME_ANGLES[NANOG1_NUM_ACTIONS] = {
    -0.10, 0.00,  0.00, 0.30, -0.20, 0.00,
    -0.10, 0.00,  0.00, 0.30, -0.20, 0.00,
    0.00,  0.00,  0.00,
    0.20,  0.20,  0.00, 1.28, 0.00,  0.00, 0.00,
    0.20,  -0.20, 0.00, 1.28, 0.00,  0.00, 0.00
};

const double NANOG1_CTRL_LO[NANOG1_NUM_LEG] = {
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618,
    -2.5307,
    -0.5236,
    -2.7576,
    -0.087267,
    -0.87267,
    -0.2618
};

const double NANOG1_CTRL_HI[NANOG1_NUM_LEG] = {
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618,
    2.8798,
    2.9671,
    2.7576,
    2.8798,
    0.5236,
    0.2618
};

const float NANOG1_KPS[NANOG1_NUM_LEG] =
    {100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40};

const float NANOG1_KDS[NANOG1_NUM_LEG] = {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2};

const float NANOG1_JOINT_DAMPING[6] = {2.0f, 2.0f, 2.0f, 2.0f, 1.0f, 0.2f};

const float NANOG1_WAIST_KP = 75.0f;

const float NANOG1_WAIST_KD = 2.0f;

const int NANOG1_WAIST_END = 15;

const double NANOG1_ANG_VEL_SCALE = 0.25;

const double NANOG1_DOF_VEL_SCALE = 0.05;

const double NANOG1_ACTION_SCALE = 0.25;

const double NANOG1_ACTION_CLIP = 1.0;

const int NANOG1_PHASE_PERIOD = 40;

const double NANOG1_VX_MIN = -0.5;

const double NANOG1_VX_MAX = 0.8;

const double NANOG1_VY_ABS = 0.4;

const double NANOG1_YAW_RATE_ABS = 1.0;

const Limits LIMITS = Limits{
    NANOG1_VX_MIN,
    NANOG1_VX_MAX,
    NANOG1_VY_ABS,
    NANOG1_YAW_RATE_ABS,
    0.0
};

struct NanoG1 {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;

  std::array<float, NANOG1_NUM_ACTIONS> last_action_{};

  std::array<float, NANOG1_STATE> carry_{};

  int64_t phase_counter_ = 0;
};

Output policy_step(
    NanoG1& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  const Eigen::Vector3d gravity =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  std::array<float, NANOG1_NUM_OBS> obs{};
  for (int k = 0; k < 3; ++k) {
    obs[k] = static_cast<float>(rs.imu_gyro[k] * NANOG1_ANG_VEL_SCALE);
    obs[3 + k] = static_cast<float>(gravity[k]);
    obs[6 + k] = static_cast<float>(drive[k]);
  }

  for (int j = 0; j < NANOG1_NUM_ACTIONS; ++j) {
    obs[9 + j] = static_cast<float>(rs.motor_q[j] - NANOG1_HOME_ANGLES[j]);
    obs[38 + j] = static_cast<float>(rs.motor_dq[j] * NANOG1_DOF_VEL_SCALE);
    obs[67 + j] = self.last_action_[j];
  }
  const double phase =
      static_cast<double>(self.phase_counter_ % NANOG1_PHASE_PERIOD) /
      static_cast<double>(NANOG1_PHASE_PERIOD);
  obs[96] = static_cast<float>(std::sin(2.0 * M_PI * phase));
  obs[97] = static_cast<float>(std::cos(2.0 * M_PI * phase));
  self.phase_counter_++;

  {
    Engine& e = *self.engine_;
    std::copy(obs.begin(), obs.end(), engine_input(e, "obs").data.begin());
    std::copy(
        self.carry_.begin(),
        self.carry_.end(),
        engine_input(e, "state_in").data.begin()
    );
    engine_run(e);
    const std::vector<float>& action = engine_output(e, "action").data;
    for (int i = 0; i < NANOG1_NUM_ACTIONS; ++i) {

      self.last_action_[i] = i < NANOG1_NUM_LEG
                                 ? std::clamp(
                                       action[i],
                                       static_cast<float>(-NANOG1_ACTION_CLIP),
                                       static_cast<float>(NANOG1_ACTION_CLIP)
                                   )
                                 : 0.0f;
    }
    const std::vector<float>& state = engine_output(e, "state_out").data;
    std::copy(state.begin(), state.end(), self.carry_.begin());
  }

  for (int i = 0; i < NANOG1_NUM_LEG; ++i) {
    out.q_target[i] = std::clamp(
        NANOG1_HOME_ANGLES[i] +
            static_cast<double>(self.last_action_[i]) * NANOG1_ACTION_SCALE,
        NANOG1_CTRL_LO[i],
        NANOG1_CTRL_HI[i]
    );
    out.kp[i] = NANOG1_KPS[i];
    out.kd[i] = NANOG1_KDS[i] + NANOG1_JOINT_DAMPING[i % 6];
    out.owns[i] = true;
  }

  for (int i = NANOG1_NUM_LEG; i < NANOG1_WAIST_END; ++i) {
    out.q_target[i] = NANOG1_HOME_ANGLES[i];
    out.kp[i] = NANOG1_WAIST_KP;
    out.kd[i] = NANOG1_WAIST_KD;
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<NanoG1> policy_make() {
  const std::shared_ptr<NanoG1> self = std::make_shared<NanoG1>();

  self->last_action_.fill(0.0f);
  self->carry_.fill(0.0f);

  std::cout << "nanog1: loading policy model " << NANOG1_MODEL_PATH
            << std::endl;
  self->engine_ = engine_init(NANOG1_MODEL_PATH);
  Engine& e = *self->engine_;
  engine_expect(e, "obs", NANOG1_NUM_OBS, true, "nanog1");
  engine_expect(e, "state_in", NANOG1_STATE, true, "nanog1");
  engine_expect(e, "action", NANOG1_NUM_ACTIONS, false, "nanog1");
  engine_expect(e, "state_out", NANOG1_STATE, false, "nanog1");
  std::cout << "nanog1: engine ready, obs " << NANOG1_NUM_OBS << " action "
            << NANOG1_NUM_ACTIONS << " (mingru carry held by the caller)"
            << std::endl;
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "nanog1"; }

 private:
  std::shared_ptr<NanoG1> state_;
};

}
