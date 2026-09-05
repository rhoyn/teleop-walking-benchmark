namespace wty_cpp {

const std::string WTY_MODEL_PATH = "policies/wty_cpp/model.onnx";

const int WTY_NUM_ACTIONS = 29;

const int WTY_HISTORY = 5;

const int WTY_NUM_TERMS = 6;

const int WTY_TERM_DIM[WTY_NUM_TERMS] = {3, 3, 3, WTY_NUM_ACTIONS,
                                         WTY_NUM_ACTIONS, WTY_NUM_ACTIONS};

const int WTY_FRAME = 3 + 3 + 3 + 3 * WTY_NUM_ACTIONS;

const int WTY_NUM_OBS = WTY_HISTORY * WTY_FRAME;

const int WTY_FIRST_ARM_MOTOR = 15;

const double WTY_DEFAULT_POS[WTY_NUM_ACTIONS] = {
    -0.1, 0.0, 0.0,  0.3,   -0.2, 0.0, -0.1, 0.0,  0.0, 0.3,
    -0.2, 0.0, 0.0,  0.0,   0.0,  0.0, 0.25, 0.0,  0.97, 0.15,
    0.0,  0.0, 0.0, -0.25,  0.0,  0.97, -0.15, 0.0, 0.0
};

const double WTY_ACTION_SCALE = 0.25;

const float WTY_ACTION_GUARD = 10.0f;

const float WTY_KPS[WTY_NUM_ACTIONS] = {
    40.1792f, 99.0984f, 40.1792f, 99.0984f, 28.5012f, 28.5012f, 40.1792f,
    99.0984f, 40.1792f, 99.0984f, 28.5012f, 28.5012f, 40.1792f, 28.5012f,
    28.5012f, 50.0f,    50.0f,    50.0f,    14.2506f, 14.2506f, 16.7783f,
    16.7783f, 50.0f,    50.0f,    50.0f,    14.2506f, 14.2506f, 16.7783f,
    16.7783f
};

const float WTY_KDS[WTY_NUM_ACTIONS] = {
    2.5579f, 6.3088f, 2.5579f, 6.3088f, 1.8144f, 1.8144f, 2.5579f, 6.3088f,
    2.5579f, 6.3088f, 1.8144f, 1.8144f, 2.5579f, 1.8144f, 1.8144f, 0.9072f,
    0.9072f, 0.9072f, 0.9072f, 0.9072f, 1.0681f, 1.0681f, 0.9072f, 0.9072f,
    0.9072f, 0.9072f, 0.9072f, 1.0681f, 1.0681f
};

const double WTY_ANG_VEL_SCALE = 0.25;

const double WTY_JOINT_VEL_SCALE = 0.05;

const double WTY_CMD_SCALE[3] = {2.0, 2.0, 0.25};

const double WTY_VX_MIN = -0.5;

const double WTY_VX_MAX = 1.0;

const double WTY_VY_ABS = 0.4;

const double WTY_YAW_RATE_ABS = 1.0;

const Limits LIMITS =
    Limits{WTY_VX_MIN, WTY_VX_MAX, WTY_VY_ABS, WTY_YAW_RATE_ABS, 0.0};

struct WtyCpp {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::vector<float> obs_;
  std::vector<float> last_action_;
  bool primed_ = false;
};

std::shared_ptr<Engine> wty_engine_init(const std::string& model_path) {
  std::cout << "wty_cpp: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 3) {
    throw std::runtime_error(
        "wty_cpp: expected one input and three outputs, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded, "obs", static_cast<size_t>(WTY_NUM_OBS), true, "wty_cpp"
  );
  engine_expect(
      *loaded, "actions", static_cast<size_t>(WTY_NUM_ACTIONS), false, "wty_cpp"
  );
  std::cout << "wty_cpp: engine ready, obs " << WTY_NUM_OBS << " action "
            << WTY_NUM_ACTIONS << " tensors "
            << engine_tensor_list(loaded->inputs) << " -> "
            << engine_tensor_list(loaded->outputs) << std::endl;
  return loaded;
}

void wty_obs_frame(
    const WtyCpp& p,
    const RobotState& rs,
    const double drive[3],
    float* frame
) {
  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));

  for (int k = 0; k < 3; ++k) {
    frame[k] = static_cast<float>(rs.imu_gyro[k] * WTY_ANG_VEL_SCALE);
    frame[3 + k] = static_cast<float>(gravity_dir[k]);
    frame[6 + k] = static_cast<float>(drive[k] * WTY_CMD_SCALE[k]);
  }

  float* joint_pos = frame + 9;
  float* joint_vel = joint_pos + WTY_NUM_ACTIONS;
  float* last_action = joint_vel + WTY_NUM_ACTIONS;
  for (int j = 0; j < WTY_NUM_ACTIONS; ++j) {
    if (j >= WTY_FIRST_ARM_MOTOR) {
      joint_pos[j] = static_cast<float>(p.last_action_[j] * WTY_ACTION_SCALE);
      joint_vel[j] = 0.0f;
    } else {
      joint_pos[j] = static_cast<float>(rs.motor_q[j] - WTY_DEFAULT_POS[j]);
      joint_vel[j] =
          static_cast<float>(rs.motor_dq[j] * WTY_JOINT_VEL_SCALE);
    }
    last_action[j] = p.last_action_[j];
  }
}

void wty_push_frame(
    WtyCpp& p,
    const float* frame
) {
  float* obs = p.obs_.data();
  if (p.primed_) {
    std::copy(obs + WTY_FRAME, obs + WTY_NUM_OBS, obs);
    std::copy(frame, frame + WTY_FRAME, obs + (WTY_HISTORY - 1) * WTY_FRAME);
  } else {
    for (int h = 0; h < WTY_HISTORY; ++h) {
      std::copy(frame, frame + WTY_FRAME, obs + h * WTY_FRAME);
    }
    p.primed_ = true;
  }
}

Output policy_step(
    WtyCpp& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  std::array<float, WTY_FRAME> frame{};
  wty_obs_frame(self, rs, drive, frame.data());
  wty_push_frame(self, frame.data());

  Engine& engine = *self.engine_;
  std::copy(
      self.obs_.begin(), self.obs_.end(), engine_input(engine, "obs").data.begin()
  );
  engine_run(engine);
  const std::vector<float>& action = engine_output(engine, "actions").data;

  for (int j = 0; j < WTY_NUM_ACTIONS; ++j) {
    self.last_action_[j] =
        std::isfinite(action[j])
            ? std::clamp(action[j], -WTY_ACTION_GUARD, WTY_ACTION_GUARD)
            : 0.0f;
  }

  for (int j = 0; j < WTY_NUM_ACTIONS; ++j) {
    out.q_target[j] = WTY_DEFAULT_POS[j] +
                      static_cast<double>(self.last_action_[j]) *
                          WTY_ACTION_SCALE;
    out.kp[j] = WTY_KPS[j];
    out.kd[j] = WTY_KDS[j];
    out.owns[j] = j < WTY_FIRST_ARM_MOTOR;
  }
  return out;
}

std::shared_ptr<WtyCpp> policy_make() {
  const std::shared_ptr<WtyCpp> self = std::make_shared<WtyCpp>();
  self->obs_ = std::vector<float>(WTY_NUM_OBS, 0.0f);
  self->last_action_ = std::vector<float>(WTY_NUM_ACTIONS, 0.0f);

  self->engine_ = wty_engine_init(WTY_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "wty_cpp"; }

 private:
  std::shared_ptr<WtyCpp> state_;
};

}
