namespace wbc_agile {

const std::string AGILE_MODEL_PATH = "policies/wbc_agile/model.onnx";

const int AGILE_NUM_JOINTS = 29;

const int AGILE_NUM_ACTIONS = 12;

const int AGILE_HISTORY = 5;

const int AGILE_TO_MUJOCO[AGILE_NUM_JOINTS] = {
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28
};

const int AGILE_LEG_TO_MUJOCO[AGILE_NUM_ACTIONS] = {
    0,
    6,
    1,
    7,
    2,
    8,
    3,
    9,
    4,
    10,
    5,
    11
};

const double AGILE_DEFAULT_POS[AGILE_NUM_JOINTS] = {
    -0.1, -0.1, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.3,
    0.3,  0.0,  0.0, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0,  0.0,  0.0, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0
};

const int AGILE_WAIST_MUJOCO[3] = {12, 13, 14};

const float AGILE_WAIST_KP = 300.0f;

const float AGILE_WAIST_KD = 5.0f;

const double AGILE_WAIST_POS = 0.0;

const double AGILE_JOINT_VEL_SCALE = 0.1;

const double AGILE_HEIGHT_CMD = 0.72;

const Limits LIMITS = Limits{-0.5, 1.5, 0.5, 1.0, 0.0};

const char* const AGILE_IN_COMMAND = "base_velocity";

const char* const AGILE_IN_ANG_VEL = "robot_root_ang_vel_b";

const char* const AGILE_IN_QUAT = "robot_root_quat_w";

const char* const AGILE_IN_JOINT_POS = "robot_joint_pos";

const char* const AGILE_IN_JOINT_VEL = "robot_joint_vel";

const char* const AGILE_IN_LAST_ACTION = "last_action_in";

const char* const AGILE_OUT_JOINT_POS = "joint_pos";

const char* const AGILE_OUT_KP = "joint_pos_kp_gains";

const char* const AGILE_OUT_KD = "joint_pos_kd_gains";

struct AgileFeedback {
  const char* in;
  const char* out;
  size_t count;
};

const AgileFeedback AGILE_FEEDBACK[] = {
    {"h_policy_velocity_height_commands_in",
     "h_policy_velocity_height_commands_out",
     AGILE_HISTORY * 4},
    {"h_policy_base_ang_vel_in",
     "h_policy_base_ang_vel_out",
     AGILE_HISTORY * 3},
    {"h_policy_projected_gravity_in",
     "h_policy_projected_gravity_out",
     AGILE_HISTORY * 3},
    {"h_policy_joint_pos_in",
     "h_policy_joint_pos_out",
     AGILE_HISTORY * AGILE_NUM_JOINTS},
    {"h_policy_joint_vel_in",
     "h_policy_joint_vel_out",
     AGILE_HISTORY * AGILE_NUM_JOINTS},
    {"h_policy_actions_in",
     "h_policy_actions_out",
     AGILE_HISTORY * AGILE_NUM_ACTIONS},
    {AGILE_IN_LAST_ACTION, "last_action_out", AGILE_NUM_ACTIONS}
};

const int AGILE_NUM_FEEDBACK =
    static_cast<int>(sizeof(AGILE_FEEDBACK) / sizeof(AGILE_FEEDBACK[0]));

struct WbcAgile {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  bool primed_ = false;
};

std::shared_ptr<Engine> agile_engine_init(const std::string& model_path) {
  std::cout << "wbc_agile: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  Engine& e = *loaded;

  engine_expect(e, AGILE_IN_COMMAND, 4, true, "wbc_agile");
  engine_expect(e, AGILE_IN_ANG_VEL, 3, true, "wbc_agile");
  engine_expect(e, AGILE_IN_QUAT, 4, true, "wbc_agile");
  engine_expect(e, AGILE_IN_JOINT_POS, AGILE_NUM_JOINTS, true, "wbc_agile");
  engine_expect(e, AGILE_IN_JOINT_VEL, AGILE_NUM_JOINTS, true, "wbc_agile");
  engine_expect(e, AGILE_OUT_JOINT_POS, AGILE_NUM_ACTIONS, false, "wbc_agile");
  engine_expect(e, AGILE_OUT_KP, AGILE_NUM_ACTIONS, false, "wbc_agile");
  engine_expect(e, AGILE_OUT_KD, AGILE_NUM_ACTIONS, false, "wbc_agile");
  for (int i = 0; i < AGILE_NUM_FEEDBACK; ++i) {
    const AgileFeedback& f = AGILE_FEEDBACK[i];
    engine_expect(e, f.in, f.count, true, "wbc_agile");
    engine_expect(e, f.out, f.count, false, "wbc_agile");
  }

  std::cout << "wbc_agile: engine ready, " << e.inputs.size() << " inputs -> "
            << e.outputs.size() << " outputs" << std::endl;
  return loaded;
}

void agile_prime_history(
    Engine& e,
    const double command[4],
    const Eigen::Vector3d& ang_vel,
    const Eigen::Vector3d& gravity,
    const float* joint_pos,
    const float* joint_vel
) {
  std::vector<float>& cmd_hist =
      engine_input(e, "h_policy_velocity_height_commands_in").data;
  std::vector<float>& gyro_hist =
      engine_input(e, "h_policy_base_ang_vel_in").data;
  std::vector<float>& grav_hist =
      engine_input(e, "h_policy_projected_gravity_in").data;
  std::vector<float>& pos_hist = engine_input(e, "h_policy_joint_pos_in").data;
  std::vector<float>& vel_hist = engine_input(e, "h_policy_joint_vel_in").data;

  for (int h = 0; h < AGILE_HISTORY; ++h) {
    for (int k = 0; k < 4; ++k) {
      cmd_hist[h * 4 + k] = static_cast<float>(command[k]);
    }
    for (int k = 0; k < 3; ++k) {
      gyro_hist[h * 3 + k] = static_cast<float>(ang_vel[k]);
      grav_hist[h * 3 + k] = static_cast<float>(gravity[k]);
    }
    for (int j = 0; j < AGILE_NUM_JOINTS; ++j) {
      pos_hist[h * AGILE_NUM_JOINTS + j] =
          joint_pos[j] - static_cast<float>(AGILE_DEFAULT_POS[j]);
      vel_hist[h * AGILE_NUM_JOINTS + j] =
          joint_vel[j] * static_cast<float>(AGILE_JOINT_VEL_SCALE);
    }
  }
  std::fill(
      engine_input(e, "h_policy_actions_in").data.begin(),
      engine_input(e, "h_policy_actions_in").data.end(),
      0.0f
  );
  std::fill(
      engine_input(e, AGILE_IN_LAST_ACTION).data.begin(),
      engine_input(e, AGILE_IN_LAST_ACTION).data.end(),
      0.0f
  );
}

Output policy_step(
    WbcAgile& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  const double command[4] = {drive[0], drive[1], drive[2], AGILE_HEIGHT_CMD};

  const RobotState& rs = *in.state;
  Engine& e = *self.engine_;

  float joint_pos[AGILE_NUM_JOINTS];
  float joint_vel[AGILE_NUM_JOINTS];
  for (int j = 0; j < AGILE_NUM_JOINTS; ++j) {
    joint_pos[j] = static_cast<float>(rs.motor_q[AGILE_TO_MUJOCO[j]]);
    joint_vel[j] = static_cast<float>(rs.motor_dq[AGILE_TO_MUJOCO[j]]);
  }

  std::copy(
      joint_pos,
      joint_pos + AGILE_NUM_JOINTS,
      engine_input(e, AGILE_IN_JOINT_POS).data.begin()
  );
  std::copy(
      joint_vel,
      joint_vel + AGILE_NUM_JOINTS,
      engine_input(e, AGILE_IN_JOINT_VEL).data.begin()
  );

  std::vector<float>& cmd_in = engine_input(e, AGILE_IN_COMMAND).data;
  for (int k = 0; k < 4; ++k) cmd_in[k] = static_cast<float>(command[k]);

  std::vector<float>& gyro_in = engine_input(e, AGILE_IN_ANG_VEL).data;
  for (int k = 0; k < 3; ++k) {
    gyro_in[k] = static_cast<float>(rs.imu_gyro[k]);
  }

  std::vector<float>& quat_in = engine_input(e, AGILE_IN_QUAT).data;
  quat_in[0] = static_cast<float>(rs.imu_quat[1]);
  quat_in[1] = static_cast<float>(rs.imu_quat[2]);
  quat_in[2] = static_cast<float>(rs.imu_quat[3]);
  quat_in[3] = static_cast<float>(rs.imu_quat[0]);

  if (!self.primed_) {
    const Eigen::Vector3d gravity =
        quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
    agile_prime_history(e, command, rs.imu_gyro, gravity, joint_pos, joint_vel);
    self.primed_ = true;
  }

  engine_run(e);

  const std::vector<float>& target = engine_output(e, AGILE_OUT_JOINT_POS).data;
  const std::vector<float>& kp = engine_output(e, AGILE_OUT_KP).data;
  const std::vector<float>& kd = engine_output(e, AGILE_OUT_KD).data;

  Output out{};
  for (int a = 0; a < AGILE_NUM_ACTIONS; ++a) {
    const int m = AGILE_LEG_TO_MUJOCO[a];

    out.q_target[m] = static_cast<double>(target[a]);

    out.kp[m] = kp[a];
    out.kd[m] = kd[a];
    out.owns[m] = true;
  }

  for (int k = 0; k < 3; ++k) {
    const int m = AGILE_WAIST_MUJOCO[k];
    out.q_target[m] = AGILE_WAIST_POS;
    out.kp[m] = AGILE_WAIST_KP;
    out.kd[m] = AGILE_WAIST_KD;
    out.owns[m] = true;
  }

  for (int i = 0; i < AGILE_NUM_FEEDBACK; ++i) {
    const AgileFeedback& f = AGILE_FEEDBACK[i];
    const std::vector<float>& src = engine_output(e, f.out).data;
    std::copy(src.begin(), src.end(), engine_input(e, f.in).data.begin());
  }
  return out;
}

std::shared_ptr<WbcAgile> policy_make() {
  const std::shared_ptr<WbcAgile> self = std::make_shared<WbcAgile>();
  self->engine_ = agile_engine_init(AGILE_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "wbc_agile"; }

 private:
  std::shared_ptr<WbcAgile> state_;
};

}
