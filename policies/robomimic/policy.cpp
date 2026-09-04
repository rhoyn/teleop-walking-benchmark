namespace robomimic {

const int RM_NUM_ACTIONS = 29;

const int RM_NUM_OBS = 96;

const std::string MODEL_PATH = "policies/robomimic/model.onnx";

const int RM_HIDDEN = 256;

const int RM_JOINT_TO_MOTOR[RM_NUM_ACTIONS] = {0,  6,  12, 1,  7,  13, 2,  8,
                                               14, 3,  9,  15, 22, 4,  10, 16,
                                               23, 5,  11, 17, 24, 18, 25, 19,
                                               26, 20, 27, 21, 28};

const double RM_DEFAULT_ANGLES[RM_NUM_ACTIONS] = {
    -0.2, -0.2, 0.0,  0.0,   0.0,   0.0,  0.0,   0.0, 0.0, 0.42,
    0.42, 0.35, 0.35, -0.23, -0.23, 0.18, -0.18, 0.0, 0.0, 0.0,
    0.0,  0.87, 0.87, 0.0,   0.0,   0.0,  0.0,   0.0, 0.0
};

const double RM_ACTION_SCALE = 0.25;

const int RM_FIRST_ARM_MOTOR = 15;

const double RM_KPS[RM_NUM_ACTIONS] = {200, 200, 200, 150, 150, 200, 150, 150,
                                       200, 200, 200, 100, 100, 20,  20,  100,
                                       100, 20,  20,  50,  50,  50,  50,  40,
                                       40,  40,  40,  40,  40};

const double RM_KDS[RM_NUM_ACTIONS] = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                                       5, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                       2, 2, 2, 2, 2, 2, 2, 2, 2};

const double RM_IO_CLIP = 100.0;

const double RM_MAX_SPEED = 0.4;

const double RM_MAX_YAW_RATE = 0.8;

const double RM_VX_MIN = -0.4;

const double RM_VX_MAX = 0.7;

const double RM_VY_ABS = 0.4;

const Limits LIMITS =
    Limits{RM_VX_MIN, RM_VX_MAX, RM_VY_ABS, RM_MAX_YAW_RATE, RM_MAX_SPEED};

struct RoboMimic {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::array<double, RM_NUM_ACTIONS> last_action_;

  std::array<float, RM_HIDDEN> hidden_{};
  std::array<float, RM_HIDDEN> cell_{};
};

void rm_reset_recurrent_state(RoboMimic& p) {
  p.hidden_.fill(0.0f);
  p.cell_.fill(0.0f);
}

std::array<
    double,
    RM_NUM_ACTIONS>
rm_run(
    RoboMimic& p,
    const std::array<
        double,
        RM_NUM_OBS>& obs
) {
  if (!p.engine_) throw std::runtime_error("robomimic: model not loaded");
  Engine& e = *p.engine_;

  std::vector<float>& input = engine_input(e, "obs").data;
  for (int i = 0; i < RM_NUM_OBS; ++i) {
    input[i] = static_cast<float>(std::clamp(obs[i], -RM_IO_CLIP, RM_IO_CLIP));
  }
  std::copy(
      p.hidden_.begin(),
      p.hidden_.end(),
      engine_input(e, "hidden_in").data.begin()
  );
  std::copy(
      p.cell_.begin(),
      p.cell_.end(),
      engine_input(e, "cell_in").data.begin()
  );
  engine_run(e);

  const std::vector<float>& result = engine_output(e, "action").data;
  std::array<double, RM_NUM_ACTIONS> action{};
  for (int i = 0; i < RM_NUM_ACTIONS; ++i) {
    action[i] =
        std::clamp(static_cast<double>(result[i]), -RM_IO_CLIP, RM_IO_CLIP);
  }
  const std::vector<float>& h = engine_output(e, "hidden_out").data;
  const std::vector<float>& c = engine_output(e, "cell_out").data;
  std::copy(h.begin(), h.end(), p.hidden_.begin());
  std::copy(c.begin(), c.end(), p.cell_.begin());
  return action;
}

Output policy_step(
    RoboMimic& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  std::array<double, RM_NUM_OBS> obs{};
  const Eigen::Vector3d gravity =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  const double cmd[3] = {drive[0], drive[1], drive[2]};
  for (int k = 0; k < 3; ++k) {
    obs[k] = rs.imu_gyro[k];
    obs[3 + k] = gravity[k];
    obs[6 + k] = cmd[k];
  }
  for (int i = 0; i < RM_NUM_ACTIONS; ++i) {
    const int m = RM_JOINT_TO_MOTOR[i];
    const bool is_arm = m >= RM_FIRST_ARM_MOTOR;
    obs[9 + i] = is_arm ? 0.0 : rs.motor_q[m] - RM_DEFAULT_ANGLES[i];
    obs[9 + RM_NUM_ACTIONS + i] = is_arm ? 0.0 : rs.motor_dq[m];
    obs[9 + 2 * RM_NUM_ACTIONS + i] = self.last_action_[i];
  }

  const std::array<double, RM_NUM_ACTIONS> action = rm_run(self, obs);
  self.last_action_ = action;

  for (int i = 0; i < RM_NUM_ACTIONS; ++i) {
    const int m = RM_JOINT_TO_MOTOR[i];
    out.q_target[m] = action[i] * RM_ACTION_SCALE + RM_DEFAULT_ANGLES[i];
    out.kp[m] = static_cast<float>(RM_KPS[i]);
    out.kd[m] = static_cast<float>(RM_KDS[i]);
    out.owns[m] = m < RM_FIRST_ARM_MOTOR;
  }
  return out;
}

std::shared_ptr<RoboMimic> policy_make() {
  const std::shared_ptr<RoboMimic> self = std::make_shared<RoboMimic>();

  self->last_action_.fill(0.0);

  std::cout << "robomimic: loading policy model " << MODEL_PATH << std::endl;
  self->engine_ = engine_init(MODEL_PATH);
  Engine& e = *self->engine_;
  engine_expect(e, "obs", RM_NUM_OBS, true, "robomimic");
  engine_expect(e, "hidden_in", RM_HIDDEN, true, "robomimic");
  engine_expect(e, "cell_in", RM_HIDDEN, true, "robomimic");
  engine_expect(e, "action", RM_NUM_ACTIONS, false, "robomimic");
  engine_expect(e, "hidden_out", RM_HIDDEN, false, "robomimic");
  engine_expect(e, "cell_out", RM_HIDDEN, false, "robomimic");
  rm_reset_recurrent_state(*self);
  std::cout << "robomimic: engine ready, obs " << RM_NUM_OBS << " action "
            << RM_NUM_ACTIONS << " (lstm state carried by the caller)"
            << std::endl;
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "robomimic"; }

 private:
  std::shared_ptr<RoboMimic> state_;
};

}
