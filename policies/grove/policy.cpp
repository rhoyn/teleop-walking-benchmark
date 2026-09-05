namespace grove {

const std::string GROVE_MODEL_PATH = "policies/grove/model.onnx";

const int GROVE_NUM_JOINTS = 29;

const int GROVE_NUM_ACTIONS = 14;

const int GROVE_HISTORY = 5;

const int GROVE_TO_MUJOCO[GROVE_NUM_JOINTS] = {
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

const int GROVE_ACTION_TO_MUJOCO[GROVE_NUM_ACTIONS] = {
    0,
    6,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    4,
    10,
    5,
    11
};

const int GROVE_ACTION_TO_OBS[GROVE_NUM_ACTIONS] =
    {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 17, 18};

const double GROVE_DEFAULT_POS[GROVE_NUM_ACTIONS] =
    {-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, -0.2, -0.2, 0.0, 0.0};

const double GROVE_ANG_VEL_SCALE = 0.2;

const double GROVE_JOINT_VEL_SCALE = 0.05;

const int GROVE_WAIST_YAW_MUJOCO = 12;

const float GROVE_WAIST_YAW_KP = 300.0f;

const float GROVE_WAIST_YAW_KD = 5.0f;

const double GROVE_WAIST_YAW_POS = 0.0;

const Limits LIMITS = Limits{-0.5, 0.5, 0.5, 1.0, 0.0};

const char* const GROVE_IN_QUAT = "root_link_quat_w";

const char* const GROVE_IN_ANG_VEL = "root_ang_vel_b";

const char* const GROVE_IN_COMMAND = "velocity_commands";

const char* const GROVE_IN_JOINT_POS = "joint_pos";

const char* const GROVE_IN_JOINT_VEL = "joint_vel";

const char* const GROVE_IN_LAST_ACTION = "last_actions";

const char* const GROVE_IN_GYRO_HIST = "base_ang_vel_history";

const char* const GROVE_IN_GRAV_HIST = "projected_gravity_history";

const char* const GROVE_IN_CMD_HIST = "velocity_commands_history";

const char* const GROVE_IN_POS_HIST = "controlled_joint_pos_history";

const char* const GROVE_IN_VEL_HIST = "controlled_joint_vel_history";

const char* const GROVE_IN_ACT_HIST = "actions_history";

const char* const GROVE_OUT_JOINT_POS = "action_joint_pos";

const char* const GROVE_OUT_KP = "action_joint_pos_kp_gains";

const char* const GROVE_OUT_KD = "action_joint_pos_kd_gains";

struct GroveFeedback {
  const char* in;
  const char* out;
  size_t count;
};

const GroveFeedback GROVE_FEEDBACK[] = {
    {GROVE_IN_LAST_ACTION, "last_actions_out", GROVE_NUM_ACTIONS},
    {GROVE_IN_GYRO_HIST, "base_ang_vel_history_out", GROVE_HISTORY * 3},
    {GROVE_IN_GRAV_HIST, "projected_gravity_history_out", GROVE_HISTORY * 3},
    {GROVE_IN_CMD_HIST, "velocity_commands_history_out", GROVE_HISTORY * 3},
    {GROVE_IN_POS_HIST,
     "controlled_joint_pos_history_out",
     GROVE_HISTORY * GROVE_NUM_ACTIONS},
    {GROVE_IN_VEL_HIST,
     "controlled_joint_vel_history_out",
     GROVE_HISTORY * GROVE_NUM_ACTIONS},
    {GROVE_IN_ACT_HIST,
     "actions_history_out",
     GROVE_HISTORY * GROVE_NUM_ACTIONS}
};

const int GROVE_NUM_FEEDBACK =
    static_cast<int>(sizeof(GROVE_FEEDBACK) / sizeof(GROVE_FEEDBACK[0]));

struct Grove {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  bool primed_ = false;
};

std::shared_ptr<Engine> grove_engine_init(const std::string& model_path) {
  std::cout << "grove: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  Engine& e = *loaded;

  engine_expect(e, GROVE_IN_QUAT, 4, true, "grove");
  engine_expect(e, GROVE_IN_ANG_VEL, 3, true, "grove");
  engine_expect(e, GROVE_IN_COMMAND, 3, true, "grove");
  engine_expect(e, GROVE_IN_JOINT_POS, GROVE_NUM_JOINTS, true, "grove");
  engine_expect(e, GROVE_IN_JOINT_VEL, GROVE_NUM_JOINTS, true, "grove");
  engine_expect(e, GROVE_OUT_JOINT_POS, GROVE_NUM_ACTIONS, false, "grove");
  engine_expect(e, GROVE_OUT_KP, GROVE_NUM_ACTIONS, false, "grove");
  engine_expect(e, GROVE_OUT_KD, GROVE_NUM_ACTIONS, false, "grove");
  for (int i = 0; i < GROVE_NUM_FEEDBACK; ++i) {
    const GroveFeedback& f = GROVE_FEEDBACK[i];
    engine_expect(e, f.in, f.count, true, "grove");
    engine_expect(e, f.out, f.count, false, "grove");
  }

  std::cout << "grove: engine ready, " << e.inputs.size() << " inputs -> "
            << e.outputs.size() << " outputs" << std::endl;
  return loaded;
}

void grove_prime_history(
    Engine& e,
    const double command[3],
    const Eigen::Vector3d& ang_vel,
    const Eigen::Vector3d& gravity,
    const float* joint_pos,
    const float* joint_vel
) {
  std::vector<float>& gyro_hist = engine_input(e, GROVE_IN_GYRO_HIST).data;
  std::vector<float>& grav_hist = engine_input(e, GROVE_IN_GRAV_HIST).data;
  std::vector<float>& cmd_hist = engine_input(e, GROVE_IN_CMD_HIST).data;
  std::vector<float>& pos_hist = engine_input(e, GROVE_IN_POS_HIST).data;
  std::vector<float>& vel_hist = engine_input(e, GROVE_IN_VEL_HIST).data;

  for (int h = 0; h < GROVE_HISTORY; ++h) {
    for (int k = 0; k < 3; ++k) {
      gyro_hist[h * 3 + k] =
          static_cast<float>(ang_vel[k] * GROVE_ANG_VEL_SCALE);
      grav_hist[h * 3 + k] = static_cast<float>(gravity[k]);
      cmd_hist[h * 3 + k] = static_cast<float>(command[k]);
    }
    for (int a = 0; a < GROVE_NUM_ACTIONS; ++a) {

      const int j = GROVE_ACTION_TO_OBS[a];
      pos_hist[h * GROVE_NUM_ACTIONS + a] =
          joint_pos[j] - static_cast<float>(GROVE_DEFAULT_POS[a]);
      vel_hist[h * GROVE_NUM_ACTIONS + a] =
          joint_vel[j] * static_cast<float>(GROVE_JOINT_VEL_SCALE);
    }
  }
  std::fill(
      engine_input(e, GROVE_IN_ACT_HIST).data.begin(),
      engine_input(e, GROVE_IN_ACT_HIST).data.end(),
      0.0f
  );
  std::fill(
      engine_input(e, GROVE_IN_LAST_ACTION).data.begin(),
      engine_input(e, GROVE_IN_LAST_ACTION).data.end(),
      0.0f
  );
}

Output policy_step(
    Grove& self,
    const Input& in
) {
  double command[3];
  command_from_target(
      in,
      LIMITS,
      self.pos_reached_,
      self.yaw_reached_,
      command
  );

  const RobotState& rs = *in.state;
  Engine& e = *self.engine_;

  float joint_pos[GROVE_NUM_JOINTS];
  float joint_vel[GROVE_NUM_JOINTS];
  for (int j = 0; j < GROVE_NUM_JOINTS; ++j) {
    joint_pos[j] = static_cast<float>(rs.motor_q[GROVE_TO_MUJOCO[j]]);
    joint_vel[j] = static_cast<float>(rs.motor_dq[GROVE_TO_MUJOCO[j]]);
  }

  std::copy(
      joint_pos,
      joint_pos + GROVE_NUM_JOINTS,
      engine_input(e, GROVE_IN_JOINT_POS).data.begin()
  );
  std::copy(
      joint_vel,
      joint_vel + GROVE_NUM_JOINTS,
      engine_input(e, GROVE_IN_JOINT_VEL).data.begin()
  );

  std::vector<float>& cmd_in = engine_input(e, GROVE_IN_COMMAND).data;
  for (int k = 0; k < 3; ++k) cmd_in[k] = static_cast<float>(command[k]);

  std::vector<float>& gyro_in = engine_input(e, GROVE_IN_ANG_VEL).data;
  for (int k = 0; k < 3; ++k) gyro_in[k] = static_cast<float>(rs.imu_gyro[k]);

  std::vector<float>& quat_in = engine_input(e, GROVE_IN_QUAT).data;
  for (int k = 0; k < 4; ++k) quat_in[k] = static_cast<float>(rs.imu_quat[k]);

  if (!self.primed_) {
    const Eigen::Vector3d gravity =
        quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
    grove_prime_history(e, command, rs.imu_gyro, gravity, joint_pos, joint_vel);
    self.primed_ = true;
  }

  engine_run(e);

  const std::vector<float>& target = engine_output(e, GROVE_OUT_JOINT_POS).data;
  const std::vector<float>& kp = engine_output(e, GROVE_OUT_KP).data;
  const std::vector<float>& kd = engine_output(e, GROVE_OUT_KD).data;

  Output out{};
  for (int a = 0; a < GROVE_NUM_ACTIONS; ++a) {
    const int m = GROVE_ACTION_TO_MUJOCO[a];

    out.q_target[m] = static_cast<double>(target[a]);

    out.kp[m] = kp[a];
    out.kd[m] = kd[a];
    out.owns[m] = true;
  }

  out.q_target[GROVE_WAIST_YAW_MUJOCO] = GROVE_WAIST_YAW_POS;
  out.kp[GROVE_WAIST_YAW_MUJOCO] = GROVE_WAIST_YAW_KP;
  out.kd[GROVE_WAIST_YAW_MUJOCO] = GROVE_WAIST_YAW_KD;
  out.owns[GROVE_WAIST_YAW_MUJOCO] = true;

  for (int i = 0; i < GROVE_NUM_FEEDBACK; ++i) {
    const GroveFeedback& f = GROVE_FEEDBACK[i];
    const std::vector<float>& src = engine_output(e, f.out).data;
    std::copy(src.begin(), src.end(), engine_input(e, f.in).data.begin());
  }
  return out;
}

std::shared_ptr<Grove> policy_make() {
  const std::shared_ptr<Grove> self = std::make_shared<Grove>();
  self->engine_ = grove_engine_init(GROVE_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "grove"; }

 private:
  std::shared_ptr<Grove> state_;
};

}
