namespace rl_lab {

const std::string MODEL_PATH = "policies/rl_lab/model.onnx";

const int RL_NUM_ACTIONS = 29;

const int RL_HISTORY = 5;

const int RL_NUM_OBS = RL_HISTORY * (3 + 3 + 3 + 3 * RL_NUM_ACTIONS);

const double RL_DEFAULT_ISAAC[RL_NUM_ACTIONS] = {
    -0.1, -0.1, 0.0,  0.0,  0.0,   0.0,  0.0,   0.0, 0.0, 0.3,
    0.3,  0.3,  0.3,  -0.2, -0.2,  0.25, -0.25, 0.0, 0.0, 0.0,
    0.0,  0.97, 0.97, 0.15, -0.15, 0.0,  0.0,   0.0, 0.0
};

const double RL_ACTION_SCALE = 0.25;

const float RL_KPS[RL_NUM_ACTIONS] = {100.0f, 100.0f, 100.0f, 150.0f, 40.0f,
                                      40.0f,  100.0f, 100.0f, 100.0f, 150.0f,
                                      40.0f,  40.0f,  200.0f, 200.0f, 200.0f,
                                      40.0f,  40.0f,  40.0f,  40.0f,  40.0f,
                                      40.0f,  40.0f,  40.0f,  40.0f,  40.0f,
                                      40.0f,  40.0f,  40.0f,  40.0f};

const float RL_KDS[RL_NUM_ACTIONS] = {2.0f,  2.0f,  2.0f,  4.0f,  2.0f,  2.0f,
                                      2.0f,  2.0f,  2.0f,  4.0f,  2.0f,  2.0f,
                                      5.0f,  5.0f,  5.0f,  10.0f, 10.0f, 10.0f,
                                      10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
                                      10.0f, 10.0f, 10.0f, 10.0f, 10.0f};

const float RL_ACTION_CLIP = 10.0f;

const int RL_TERM_DIM[6] =
    {3, 3, 3, RL_NUM_ACTIONS, RL_NUM_ACTIONS, RL_NUM_ACTIONS};

const double RL_ANG_VEL_SCALE = 0.2;

const double RL_JOINT_VEL_SCALE = 0.05;

const double RL_VX_MIN = -0.5;

const double RL_VX_MAX = 1.0;

const double RL_VY_ABS = 0.3;

const double RL_YAW_RATE_ABS = 0.2;

const Limits LIMITS = [] {
  Limits l{RL_VX_MIN, RL_VX_MAX, RL_VY_ABS, RL_YAW_RATE_ABS, 0.0};
  l.walk_kp_pos = 2.0;
  return l;
}();

struct RlLab {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::vector<float> obs_;
  std::vector<float> last_action_;
  int term_off_[6] = {0, 0, 0, 0, 0, 0};
  bool primed_ = false;
};

std::shared_ptr<Engine> rl_engine_init(const std::string& model_path) {
  std::cout << "rl_lab: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "rl_lab: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(RL_NUM_OBS),
      true,
      "rl_lab"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(RL_NUM_ACTIONS),
      false,
      "rl_lab"
  );
  std::cout << "rl_lab: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

template <typename Obs>
std::vector<float> rl_engine_run(
    Engine& s,
    const Obs& obs
) {
  return engine_run_single(s, obs.data(), obs.size());
}

void rl_prime_term(
    RlLab& p,
    int t,
    const float* frame
) {
  const int dim = RL_TERM_DIM[t];
  float* base = p.obs_.data() + p.term_off_[t];
  for (int h = 0; h < RL_HISTORY; ++h) {
    std::copy(frame, frame + dim, base + h * dim);
  }
}

void rl_push_term(
    RlLab& p,
    int t,
    const float* frame
) {
  const int dim = RL_TERM_DIM[t];
  float* base = p.obs_.data() + p.term_off_[t];
  std::copy(base + dim, base + RL_HISTORY * dim, base);
  std::copy(frame, frame + dim, base + (RL_HISTORY - 1) * dim);
}

void rl_obs_frame(
    const RlLab& p,
    const RobotState& rs,
    const Input&,
    float frame[6][RL_NUM_ACTIONS],
    const double drive[3]
) {
  for (int k = 0; k < 3; ++k) {
    frame[0][k] = static_cast<float>(rs.imu_gyro[k] * RL_ANG_VEL_SCALE);
  }
  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int k = 0; k < 3; ++k) {
    frame[1][k] = static_cast<float>(gravity_dir[k]);
  }
  frame[2][0] = static_cast<float>(drive[0]);
  frame[2][1] = static_cast<float>(drive[1]);
  frame[2][2] = static_cast<float>(drive[2]);

  for (int i = 0; i < RL_NUM_ACTIONS; ++i) {
    const int motor = MUJOCO_TO_ISAACLAB[i];
    if (motor >= 15) {
      frame[3][i] = static_cast<float>(p.last_action_[i] * RL_ACTION_SCALE);
      frame[4][i] = 0.0f;
    } else {
      frame[3][i] = static_cast<float>(rs.motor_q[motor] - RL_DEFAULT_ISAAC[i]);
      frame[4][i] = static_cast<float>(rs.motor_dq[motor] * RL_JOINT_VEL_SCALE);
    }
    frame[5][i] = p.last_action_[i];
  }
}

Output policy_step(
    RlLab& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  float frame[6][RL_NUM_ACTIONS];
  rl_obs_frame(self, rs, in, frame, drive);

  for (int t = 0; t < 6; ++t) {
    if (self.primed_) {
      rl_push_term(self, t, frame[t]);
    } else {
      rl_prime_term(self, t, frame[t]);
    }
  }
  self.primed_ = true;

  std::vector<float> action = rl_engine_run(*self.engine_, self.obs_);
  for (int i = 0; i < RL_NUM_ACTIONS; ++i) {
    action[i] = std::isfinite(action[i])
                    ? std::clamp(action[i], -RL_ACTION_CLIP, RL_ACTION_CLIP)
                    : 0.0f;
  }
  self.last_action_ = action;

  for (int i = 0; i < RL_NUM_ACTIONS; ++i) {
    const int motor = MUJOCO_TO_ISAACLAB[i];
    out.q_target[motor] =
        RL_DEFAULT_ISAAC[i] +
        static_cast<double>(self.last_action_[i]) * RL_ACTION_SCALE;
  }
  for (int m = 0; m < NUM_MOTOR; ++m) {
    out.kp[m] = RL_KPS[m];
    out.kd[m] = RL_KDS[m];
    out.owns[m] = m < 15;
  }
  return out;
}

std::shared_ptr<RlLab> policy_make() {
  const std::shared_ptr<RlLab> self = std::make_shared<RlLab>();
  self->obs_ = std::vector<float>(RL_NUM_OBS, 0.0f);
  self->last_action_ = std::vector<float>(RL_NUM_ACTIONS, 0.0f);

  int off = 0;
  for (int t = 0; t < 6; ++t) {
    self->term_off_[t] = off;
    off += RL_HISTORY * RL_TERM_DIM[t];
  }

  self->engine_ = rl_engine_init(MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "rl_lab"; }

 private:
  std::shared_ptr<RlLab> state_;
};

}
