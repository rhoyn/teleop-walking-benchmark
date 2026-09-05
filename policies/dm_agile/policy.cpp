namespace dm_agile {

const std::string MODEL_PATH = "policies/dm_agile/model.onnx";

const int DM_NUM_ACTIONS = 29;

const int DM_HISTORY = 5;

const int DM_NUM_TERMS = 6;

const int DM_TERM_DIM[DM_NUM_TERMS] =
    {3, 3, 3, DM_NUM_ACTIONS, DM_NUM_ACTIONS, DM_NUM_ACTIONS};

const int DM_NUM_OBS = DM_HISTORY * (3 + 3 + 3 + 3 * DM_NUM_ACTIONS);

const double DM_DEFAULT_ISAAC[DM_NUM_ACTIONS] = {
    -0.1, -0.1, 0.0,  0.0,  0.0,   0.0,  0.0,   0.0, 0.0, 0.3,
    0.3,  0.3,  0.3,  -0.2, -0.2,  0.25, -0.25, 0.0, 0.0, 0.0,
    0.0,  0.97, 0.97, 0.15, -0.15, 0.0,  0.0,   0.0, 0.0
};

const double DM_ACTION_SCALE = 0.25;

const float DM_ACTION_CLIP = 10.0f;

const float DM_KPS[DM_NUM_ACTIONS] = {
    100.0f, 100.0f, 100.0f, 150.0f, 40.0f, 40.0f, 100.0f, 100.0f,
    100.0f, 150.0f, 40.0f,  40.0f,  200.0f, 40.0f, 40.0f, 40.0f,
    40.0f,  40.0f,  40.0f,  40.0f,  40.0f,  40.0f, 40.0f, 40.0f,
    40.0f,  40.0f,  40.0f,  40.0f,  40.0f
};

const float DM_KDS[DM_NUM_ACTIONS] = {
    2.0f, 2.0f, 2.0f, 4.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 4.0f,
    2.0f, 2.0f, 5.0f, 5.0f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};

const double DM_ANG_VEL_SCALE = 0.2;

const double DM_JOINT_VEL_SCALE = 0.05;

const Eigen::Vector<int, NUM_MOTOR>& DM_ISAAC_TO_MUJOCO = MUJOCO_TO_ISAACLAB;

const Limits LIMITS = [] {
  Limits l{-0.8, 1.5, 0.5, 0.15, 0.0};
  return l;
}();

struct DmAgile {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::vector<float> obs_;
  std::vector<float> last_action_;
  int term_off_[DM_NUM_TERMS] = {0, 0, 0, 0, 0, 0};
  bool primed_ = false;
};

std::shared_ptr<Engine> dm_engine_init(const std::string& model_path) {
  std::cout << "dm_agile: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "dm_agile: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(DM_NUM_OBS),
      true,
      "dm_agile"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(DM_NUM_ACTIONS),
      false,
      "dm_agile"
  );
  std::cout << "dm_agile: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

void dm_prime_term(
    DmAgile& p,
    int t,
    const float* frame
) {
  const int dim = DM_TERM_DIM[t];
  float* base = p.obs_.data() + p.term_off_[t];
  for (int h = 0; h < DM_HISTORY; ++h) {
    std::copy(frame, frame + dim, base + h * dim);
  }
}

void dm_push_term(
    DmAgile& p,
    int t,
    const float* frame
) {
  const int dim = DM_TERM_DIM[t];
  float* base = p.obs_.data() + p.term_off_[t];
  std::copy(base + dim, base + DM_HISTORY * dim, base);
  std::copy(frame, frame + dim, base + (DM_HISTORY - 1) * dim);
}

void dm_obs_frame(
    const DmAgile& p,
    const RobotState& rs,
    float frame[DM_NUM_TERMS][DM_NUM_ACTIONS],
    const double drive[3]
) {
  for (int k = 0; k < 3; ++k) {
    frame[0][k] = static_cast<float>(rs.imu_gyro[k] * DM_ANG_VEL_SCALE);
  }
  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int k = 0; k < 3; ++k) {
    frame[1][k] = static_cast<float>(gravity_dir[k]);
  }
  frame[2][0] = static_cast<float>(drive[0]);
  frame[2][1] = static_cast<float>(drive[1]);
  frame[2][2] = static_cast<float>(drive[2]);

  for (int i = 0; i < DM_NUM_ACTIONS; ++i) {
    const int motor = DM_ISAAC_TO_MUJOCO[i];
    if (motor >= 15) {
      frame[3][i] = static_cast<float>(p.last_action_[i] * DM_ACTION_SCALE);
      frame[4][i] = 0.0f;
    } else {
      frame[3][i] = static_cast<float>(rs.motor_q[motor] - DM_DEFAULT_ISAAC[i]);
      frame[4][i] = static_cast<float>(rs.motor_dq[motor] * DM_JOINT_VEL_SCALE);
    }

    frame[5][i] = p.last_action_[i];
  }
}

Output policy_step(
    DmAgile& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  float frame[DM_NUM_TERMS][DM_NUM_ACTIONS];
  dm_obs_frame(self, rs, frame, drive);

  for (int t = 0; t < DM_NUM_TERMS; ++t) {
    if (self.primed_) {
      dm_push_term(self, t, frame[t]);
    } else {
      dm_prime_term(self, t, frame[t]);
    }
  }
  self.primed_ = true;

  std::vector<float> action =
      engine_run_single(*self.engine_, self.obs_.data(), self.obs_.size());
  for (int i = 0; i < DM_NUM_ACTIONS; ++i) {
    action[i] = std::isfinite(action[i])
                    ? std::clamp(action[i], -DM_ACTION_CLIP, DM_ACTION_CLIP)
                    : 0.0f;
  }
  self.last_action_ = action;

  for (int i = 0; i < DM_NUM_ACTIONS; ++i) {
    const int motor = DM_ISAAC_TO_MUJOCO[i];
    out.q_target[motor] =
        DM_DEFAULT_ISAAC[i] +
        static_cast<double>(self.last_action_[i]) * DM_ACTION_SCALE;
  }
  for (int m = 0; m < NUM_MOTOR; ++m) {
    out.kp[m] = DM_KPS[m];
    out.kd[m] = DM_KDS[m];
    out.owns[m] = m < 15;
  }
  return out;
}

std::shared_ptr<DmAgile> policy_make() {
  const std::shared_ptr<DmAgile> self = std::make_shared<DmAgile>();
  self->obs_ = std::vector<float>(DM_NUM_OBS, 0.0f);
  self->last_action_ = std::vector<float>(DM_NUM_ACTIONS, 0.0f);

  int off = 0;
  for (int t = 0; t < DM_NUM_TERMS; ++t) {
    self->term_off_[t] = off;
    off += DM_HISTORY * DM_TERM_DIM[t];
  }

  self->engine_ = dm_engine_init(MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "dm_agile"; }

 private:
  std::shared_ptr<DmAgile> state_;
};

}
