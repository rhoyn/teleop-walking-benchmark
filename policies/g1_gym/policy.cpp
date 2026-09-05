namespace g1_gym {

const int G1GYM_NUM_OBS = 90;

const int G1GYM_NUM_ACT = 27;

const int G1GYM_HIDDEN = 64;

const std::string G1GYM_MODEL_PATH = "policies/g1_gym/model.onnx";

inline int g1gym_to_bench(int policy_index) {
  return policy_index < 13 ? policy_index : policy_index + 2;
}

const double G1GYM_DEFAULT_ANGLES[G1GYM_NUM_ACT] = {
    -0.1, 0.0,  0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0,  0.0, 0.3, -0.2, 0.0,
    0.0,
    0.3,  0.3,  0.0, 0.9, 0.0,  0.0, 0.0,
    0.3,  -0.3, 0.0, 0.9, 0.0,  0.0, 0.0,
};

const float G1GYM_KPS[G1GYM_NUM_ACT] = {
    100, 100, 100, 200, 20, 20,
    100, 100, 100, 200, 20, 20,
    100,
    90,  60,  20,  60,  8,  8,  8,
    90,  60,  20,  60,  8,  8,  8,
};

const float G1GYM_KDS[G1GYM_NUM_ACT] = {
    2.5f, 2.5f, 2.5f, 5.0f, 0.2f, 0.1f,
    2.5f, 2.5f, 2.5f, 5.0f, 0.2f, 0.1f,
    4.0f,
    2.0f, 1.0f, 0.4f, 1.0f, 0.2f, 0.2f, 0.2f,
    2.0f, 1.0f, 0.4f, 1.0f, 0.2f, 0.2f, 0.2f,
};

const double G1GYM_JOINT_LOWER[13] = {
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
};

const double G1GYM_JOINT_UPPER[13] = {
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
};

const double G1GYM_JOINT_MARGIN = 0.01;

const int G1GYM_WAIST_LOCK_BEGIN = 13;

const int G1GYM_WAIST_LOCK_END = 15;

const float G1GYM_WAIST_LOCK_KP = 300.0f;

const float G1GYM_WAIST_LOCK_KD = 3.0f;

const double G1GYM_ANG_VEL_SCALE = 0.25;

const double G1GYM_DOF_VEL_SCALE = 0.05;

const double G1GYM_CMD_SCALE[3] = {2.0, 2.0, 0.25};

const double G1GYM_ACTION_SCALE = 0.25;

const double G1GYM_MAX_CMD[3] = {1.0, 0.5, 1.0};

const double G1GYM_POS_REACHED_ENTER_M = 0.08;

const double G1GYM_POS_REACHED_EXIT_M = 0.15;

const double G1GYM_YAW_REACHED_ENTER_RAD = 0.05;

const double G1GYM_YAW_REACHED_EXIT_RAD = 0.10;

const double G1GYM_WALK_KP_POS = 1.5;

const double G1GYM_WALK_KP_YAW = 2.0;

const Limits LIMITS = [] {
  Limits l{};
  l.vx_min = -G1GYM_MAX_CMD[0];
  l.vx_max = G1GYM_MAX_CMD[0];
  l.vy_abs = G1GYM_MAX_CMD[1];
  l.yaw_rate_abs = G1GYM_MAX_CMD[2];
  l.speed_norm = 0.0;
  l.pos_reached_enter_m = G1GYM_POS_REACHED_ENTER_M;
  l.pos_reached_exit_m = G1GYM_POS_REACHED_EXIT_M;
  l.yaw_reached_enter_rad = G1GYM_YAW_REACHED_ENTER_RAD;
  l.yaw_reached_exit_rad = G1GYM_YAW_REACHED_EXIT_RAD;
  l.walk_kp_pos = G1GYM_WALK_KP_POS;
  l.walk_kp_yaw = G1GYM_WALK_KP_YAW;
  return l;
}();

struct G1Gym {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;

  std::array<float, G1GYM_NUM_ACT> last_action_;

  std::array<float, G1GYM_HIDDEN> hidden_{};
  std::array<float, G1GYM_HIDDEN> cell_{};
};

Output policy_step(
    G1Gym& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  std::array<float, G1GYM_NUM_OBS> obs{};
  for (int k = 0; k < 3; ++k) {
    obs[k] = static_cast<float>(rs.imu_gyro[k] * G1GYM_ANG_VEL_SCALE);
    obs[3 + k] = static_cast<float>(gravity_dir[k]);
    obs[6 + k] = static_cast<float>(drive[k] * G1GYM_CMD_SCALE[k]);
  }
  for (int j = 0; j < G1GYM_NUM_ACT; ++j) {
    const int b = g1gym_to_bench(j);

    obs[9 + j] = static_cast<float>(rs.motor_q[b] - G1GYM_DEFAULT_ANGLES[j]);
    obs[9 + G1GYM_NUM_ACT + j] =
        static_cast<float>(rs.motor_dq[b] * G1GYM_DOF_VEL_SCALE);
    obs[9 + 2 * G1GYM_NUM_ACT + j] = self.last_action_[j];
  }

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
    std::copy(action.begin(), action.end(), self.last_action_.begin());
    const std::vector<float>& h = engine_output(e, "hidden_out").data;
    const std::vector<float>& c = engine_output(e, "cell_out").data;
    std::copy(h.begin(), h.end(), self.hidden_.begin());
    std::copy(c.begin(), c.end(), self.cell_.begin());
  }

  for (int j = 0; j <= 12; ++j) {
    const int b = g1gym_to_bench(j);
    out.q_target[b] = std::clamp(
        G1GYM_DEFAULT_ANGLES[j] +
            static_cast<double>(self.last_action_[j]) * G1GYM_ACTION_SCALE,
        G1GYM_JOINT_LOWER[j] + G1GYM_JOINT_MARGIN,
        G1GYM_JOINT_UPPER[j] - G1GYM_JOINT_MARGIN
    );
    out.kp[b] = G1GYM_KPS[j];
    out.kd[b] = G1GYM_KDS[j];
    out.owns[b] = true;
  }
  for (int i = G1GYM_WAIST_LOCK_BEGIN; i < G1GYM_WAIST_LOCK_END; ++i) {
    out.q_target[i] = 0.0;
    out.kp[i] = G1GYM_WAIST_LOCK_KP;
    out.kd[i] = G1GYM_WAIST_LOCK_KD;
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<G1Gym> policy_make() {
  const std::shared_ptr<G1Gym> self = std::make_shared<G1Gym>();

  self->last_action_.fill(0.0f);
  self->hidden_.fill(0.0f);
  self->cell_.fill(0.0f);

  std::cout << "g1_gym: loading policy model " << G1GYM_MODEL_PATH << std::endl;
  self->engine_ = engine_init(G1GYM_MODEL_PATH);
  Engine& e = *self->engine_;
  engine_expect(e, "obs", G1GYM_NUM_OBS, true, "g1_gym");
  engine_expect(e, "hidden_in", G1GYM_HIDDEN, true, "g1_gym");
  engine_expect(e, "cell_in", G1GYM_HIDDEN, true, "g1_gym");
  engine_expect(e, "action", G1GYM_NUM_ACT, false, "g1_gym");
  engine_expect(e, "hidden_out", G1GYM_HIDDEN, false, "g1_gym");
  engine_expect(e, "cell_out", G1GYM_HIDDEN, false, "g1_gym");
  std::cout << "g1_gym: engine ready, obs " << G1GYM_NUM_OBS << " action "
            << G1GYM_NUM_ACT << " (13 owned, lstm state carried by the caller)"
            << std::endl;
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "g1_gym"; }

 private:
  std::shared_ptr<G1Gym> state_;
};

}
