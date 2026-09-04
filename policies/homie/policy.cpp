namespace homie {

const std::string MODEL_PATH = "policies/homie/model.onnx";

const int HOMIE_NUM_ACTIONS = 12;

const int HOMIE_OBS_DIM = 76;

const int HOMIE_HISTORY = 6;

const int HOMIE_NUM_OBS = HOMIE_HISTORY * HOMIE_OBS_DIM;

const double HOMIE_DEFAULT_LEG[HOMIE_NUM_ACTIONS] =
    {-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0};

const double HOMIE_ACTION_SCALE = 0.25;

const double HOMIE_ACTION_CLIP = 100.0;

const float HOMIE_KP[15] =
    {150, 150, 150, 300, 40, 40, 150, 150, 150, 300, 40, 40, 300, 300, 300};

const float HOMIE_KD[15] = {2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5, 5, 5};

const int HOMIE_NUM_DOFS = 27;

const double HOMIE_CMD_SCALE[4] = {2.0, 2.0, 0.25, 1.0};

const double HOMIE_ANG_VEL_SCALE = 0.5;

const double HOMIE_DOF_VEL_SCALE = 0.05;

const double HOMIE_HEIGHT_CMD = 0.74;

const double HOMIE_VX_MIN = -0.4;

const double HOMIE_VX_MAX = 0.4;

const double HOMIE_VY_ABS = 0.4;

const double HOMIE_YAW_RATE_ABS = 0.8;

const double HOMIE_SPEED_NORM = 0.4;

const double HOMIE_POS_P = 0.8;

const double HOMIE_YAW_P = 1.2;

const double HOMIE_SHAPED_VX_MIN = -0.25;

const double HOMIE_SHAPED_VX_MAX = 0.4;

const double HOMIE_SHAPED_VY_ABS = 0.25;

const double HOMIE_FACE_FAR_M = 1.0;

const double HOMIE_FACE_NEAR_M = 0.35;

const Limits LIMITS = [] {
  Limits l{
      HOMIE_VX_MIN,
      HOMIE_VX_MAX,
      HOMIE_VY_ABS,
      HOMIE_YAW_RATE_ABS,
      HOMIE_SPEED_NORM
  };
  return l;
}();

struct Homie {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::vector<float> obs_;
  std::vector<float> last_action_;
  bool primed_ = false;
};

std::shared_ptr<Engine> homie_engine_init(const std::string& model_path) {
  std::cout << "homie: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "homie: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(HOMIE_NUM_OBS),
      true,
      "homie"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(HOMIE_NUM_ACTIONS),
      false,
      "homie"
  );
  std::cout << "homie: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

template <typename Obs>
std::vector<float> homie_engine_run(
    Engine& s,
    const Obs& obs
) {
  return engine_run_single(s, obs.data(), obs.size());
}

double homie_default_dof(int i) {
  return i < HOMIE_NUM_ACTIONS ? HOMIE_DEFAULT_LEG[i] : 0.0;
}

int homie_dof_to_hw(int i) { return i < 13 ? i : i + 2; }

void homie_obs_frame(
    const Homie& p,
    const RobotState& rs,
    const Input&,
    float frame[HOMIE_OBS_DIM],
    const double drive[3]
) {
  int k = 0;
  frame[k++] = static_cast<float>(drive[0] * HOMIE_CMD_SCALE[0]);
  frame[k++] = static_cast<float>(drive[1] * HOMIE_CMD_SCALE[1]);
  frame[k++] = static_cast<float>(drive[2] * HOMIE_CMD_SCALE[2]);
  frame[k++] = static_cast<float>(HOMIE_HEIGHT_CMD * HOMIE_CMD_SCALE[3]);

  for (int j = 0; j < 3; ++j) {
    frame[k++] = static_cast<float>(rs.imu_gyro[j] * HOMIE_ANG_VEL_SCALE);
  }

  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int j = 0; j < 3; ++j) {
    frame[k++] = static_cast<float>(gravity_dir[j]);
  }

  for (int i = 0; i < HOMIE_NUM_DOFS; ++i) {
    const bool arm = i >= 13;
    frame[k++] = arm ? 0.0f
                     : static_cast<float>(
                           rs.motor_q[homie_dof_to_hw(i)] - homie_default_dof(i)
                       );
  }
  for (int i = 0; i < HOMIE_NUM_DOFS; ++i) {
    const bool arm = i >= 13;
    frame[k++] = arm ? 0.0f
                     : static_cast<float>(
                           rs.motor_dq[homie_dof_to_hw(i)] * HOMIE_DOF_VEL_SCALE
                       );
  }
  for (int i = 0; i < HOMIE_NUM_ACTIONS; ++i) frame[k++] = p.last_action_[i];
}

Output policy_step(
    Homie& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  [&] {
    const bool translating = drive[0] != 0.0 || drive[1] != 0.0;

    double bearing = 0.0;
    double face_w = 0.0;
    if (translating) {
      bearing = std::atan2(drive[1], drive[0]);
      face_w = std::clamp(
          (in.dist_m - HOMIE_FACE_NEAR_M) /
              (HOMIE_FACE_FAR_M - HOMIE_FACE_NEAR_M),
          0.0,
          1.0
      );
      const double speed = std::min(HOMIE_POS_P * in.dist_m, HOMIE_SPEED_NORM);
      drive[0] = std::clamp(
          speed * std::cos(bearing),
          HOMIE_SHAPED_VX_MIN,
          HOMIE_SHAPED_VX_MAX
      );
      drive[1] = std::clamp(
          speed * std::sin(bearing),
          -HOMIE_SHAPED_VY_ABS,
          HOMIE_SHAPED_VY_ABS
      );
    }

    if (face_w > 0.0) {
      const double blended = std::remainder(
          in.yaw_err_rad +
              face_w * std::remainder(bearing - in.yaw_err_rad, 2.0 * M_PI),
          2.0 * M_PI
      );
      drive[2] = std::clamp(
          HOMIE_YAW_P * blended,
          -HOMIE_YAW_RATE_ABS,
          HOMIE_YAW_RATE_ABS
      );
    } else if (drive[2] != 0.0) {
      drive[2] = std::clamp(
          HOMIE_YAW_P * in.yaw_err_rad,
          -HOMIE_YAW_RATE_ABS,
          HOMIE_YAW_RATE_ABS
      );
    }
  }();
  Output out{};

  float frame[HOMIE_OBS_DIM];
  homie_obs_frame(self, *in.state, in, frame, drive);

  if (self.primed_) {
    std::copy(
        self.obs_.begin() + HOMIE_OBS_DIM,
        self.obs_.end(),
        self.obs_.begin()
    );
    std::copy(frame, frame + HOMIE_OBS_DIM, self.obs_.end() - HOMIE_OBS_DIM);
  } else {
    for (int h = 0; h < HOMIE_HISTORY; ++h) {
      std::copy(
          frame,
          frame + HOMIE_OBS_DIM,
          self.obs_.begin() + h * HOMIE_OBS_DIM
      );
    }
    self.primed_ = true;
  }

  const std::vector<float> action = homie_engine_run(*self.engine_, self.obs_);
  for (int i = 0; i < HOMIE_NUM_ACTIONS; ++i) {
    self.last_action_[i] = static_cast<float>(std::clamp(
        static_cast<double>(action[i]),
        -HOMIE_ACTION_CLIP,
        HOMIE_ACTION_CLIP
    ));
  }

  for (int i = 0; i < HOMIE_NUM_ACTIONS; ++i) {
    out.q_target[i] =
        HOMIE_ACTION_SCALE * self.last_action_[i] + HOMIE_DEFAULT_LEG[i];
  }
  for (int m = 12; m < 15; ++m) out.q_target[m] = 0.0;
  for (int m = 15; m < NUM_MOTOR; ++m) {
    out.q_target[m] = DEFAULT_ANGLES[m];
  }

  for (int m = 0; m < NUM_MOTOR; ++m) {
    out.kp[m] = m < 15 ? HOMIE_KP[m] : KPS[m];
    out.kd[m] = m < 15 ? HOMIE_KD[m] : KDS[m];
    out.owns[m] = m < 15;
  }
  return out;
}

std::shared_ptr<Homie> policy_make() {
  const std::shared_ptr<Homie> self = std::make_shared<Homie>();
  self->obs_ = std::vector<float>(HOMIE_NUM_OBS, 0.0f);
  self->last_action_ = std::vector<float>(HOMIE_NUM_ACTIONS, 0.0f);

  self->engine_ = homie_engine_init(MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "homie"; }

 private:
  std::shared_ptr<Homie> state_;
};

}
