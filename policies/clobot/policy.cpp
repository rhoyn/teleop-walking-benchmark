namespace clobot {

constexpr int CLOBOT_H_COUNT = 7;

const std::string CLOBOT_MODEL_PATH = "policies/clobot/model.onnx";

const int CLOBOT_NUM_ACTIONS = 29;

const int CLOBOT_HISTORY = 5;

const int CLOBOT_GAIT_DIM = 4;

const int CLOBOT_SINGLE_OBS = 3 + 3 + 3 + 29 + 29 + 29 + CLOBOT_GAIT_DIM;

const int CLOBOT_NUM_OBS = CLOBOT_SINGLE_OBS * CLOBOT_HISTORY;

const double CLOBOT_DEFAULT_ISAAC[29] = {
    -0.264, -0.245, 0.0,  0.0,    0.0,    0.0,  0.0,   0.0, 0.0, 0.49,
    0.476,  0.3,    0.3,  -0.216, -0.199, 0.25, -0.25, 0.0, 0.0, 0.0,
    0.0,    0.97,   0.97, 0.15,   -0.15,  0.0,  0.0,   0.0, 0.0
};

const double CLOBOT_ANG_VEL_SCALE = 0.2;

const double CLOBOT_DOF_VEL_SCALE = 0.05;

const double CLOBOT_ACTION_CLIP = 5.0;

const double CLOBOT_CONTROL_DT = 0.02;

const int CLOBOT_OWNED_END = 15;

const int CLOBOT_WITH_ARMS_OWNED_END = NUM_MOTOR;

const double CLOBOT_ACTION_SCALE_ISAAC[29] = {
    0.548, 0.548, 0.548, 0.351, 0.351, 0.439,  0.548,  0.548,  0.439, 0.351,
    0.351, 0.439, 0.439, 0.439, 0.439, 0.439,  0.439,  0.439,  0.439, 0.439,
    0.439, 0.439, 0.439, 0.439, 0.439, 0.0745, 0.0745, 0.0745, 0.0745
};

const double CLOBOT_KPS_SDK[29] = {40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
                                   40.2, 99.1, 40.2, 99.1, 28.5, 28.5,
                                   40.2, 28.5, 28.5, 14.3, 14.3, 14.3,
                                   14.3, 14.3, 16.8, 16.8, 14.3, 14.3,
                                   14.3, 14.3, 14.3, 16.8, 16.8};

const double CLOBOT_KDS_SDK[29] = {2.56,  6.31,  2.56,  6.31,  1.81,  1.81,
                                   2.56,  6.31,  2.56,  6.31,  1.81,  1.81,
                                   2.56,  1.81,  1.81,  0.907, 0.907, 0.907,
                                   0.907, 0.907, 1.07,  1.07,  0.907, 0.907,
                                   0.907, 0.907, 0.907, 1.07,  1.07};

constexpr int CLOBOT_H_ANG_VEL = 0;

constexpr int CLOBOT_H_GRAVITY = 1;

constexpr int CLOBOT_H_CMD = 2;

constexpr int CLOBOT_H_JPOS = 3;

constexpr int CLOBOT_H_JVEL = 4;

constexpr int CLOBOT_H_ACT = 5;

constexpr int CLOBOT_H_GAIT = 6;

const double CLOBOT_GAIT_PERIOD = 1.0;

const bool CLOBOT_GAIT_SIN_PAIR_FIRST = true;

const double CLOBOT_VX_MIN = -0.5;

const double CLOBOT_VX_MAX = 1.0;

const double CLOBOT_VY_ABS = 0.17;

const double CLOBOT_YAW_RATE_ABS = 0.7;

struct TermHistory {
  std::deque<std::vector<float>> buff;
  int len = 0;
};

const Limits LIMITS = Limits{
    CLOBOT_VX_MIN,
    CLOBOT_VX_MAX,
    CLOBOT_VY_ABS,
    CLOBOT_YAW_RATE_ABS,
    0.0
};

struct Clobot {
  int owned_end_ = CLOBOT_OWNED_END;
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> engine_;
  std::array<TermHistory, CLOBOT_H_COUNT> hist_;
  std::vector<float> last_action_;
  int64_t gait_ticks_ = 0;
};

std::shared_ptr<Engine> clobot_engine_init(const std::string& model_path) {
  std::cout << "clobot: loading policy model " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  if (loaded->inputs.size() != 1 || loaded->outputs.size() != 1) {
    throw std::runtime_error(
        "clobot: expected one input and one output, model has " +
        engine_tensor_list(loaded->inputs) + " -> " +
        engine_tensor_list(loaded->outputs)
    );
  }
  engine_expect(
      *loaded,
      loaded->inputs[0].name,
      static_cast<size_t>(CLOBOT_NUM_OBS),
      true,
      "clobot"
  );
  engine_expect(
      *loaded,
      loaded->outputs[0].name,
      static_cast<size_t>(CLOBOT_NUM_ACTIONS),
      false,
      "clobot"
  );
  std::cout << "clobot: engine ready, obs " << loaded->inputs[0].count
            << " action " << loaded->outputs[0].count << " tensors "
            << loaded->inputs[0].name << " -> " << loaded->outputs[0].name
            << std::endl;
  return loaded;
}

void term_clear(TermHistory& h) { h.buff.clear(); }

std::array<
    double,
    CLOBOT_GAIT_DIM>
clobot_gait_phase(double seconds) {
  const double phase =
      std::fmod(seconds, CLOBOT_GAIT_PERIOD) / CLOBOT_GAIT_PERIOD;
  const double s = std::sin(2.0 * M_PI * phase);
  const double c = std::cos(2.0 * M_PI * phase);
  if (CLOBOT_GAIT_SIN_PAIR_FIRST) {
    return {s, -s, c, -c};
  }
  return {c, -c, s, -s};
}

template <typename Obs>
std::vector<float> clobot_engine_run(
    Engine& s,
    const Obs& obs
) {
  return engine_run_single(s, obs.data(), obs.size());
}

void term_push(
    TermHistory& h,
    std::vector<float> v
) {
  if (h.buff.empty()) {
    for (int i = 0; i < h.len; ++i) h.buff.push_back(v);
    return;
  }
  h.buff.push_back(std::move(v));
  while (static_cast<int>(h.buff.size()) > h.len) h.buff.pop_front();
}

void term_append_to(
    const TermHistory& h,
    std::vector<float>& out
) {
  for (const std::vector<float>& e : h.buff) {
    out.insert(out.end(), e.begin(), e.end());
  }
}

Output policy_step(
    Clobot& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  const double gait_seconds =
      static_cast<double>(self.gait_ticks_) * CLOBOT_CONTROL_DT;
  self.gait_ticks_++;

  std::vector<float> f_ang_vel(3);
  std::vector<float> f_gravity(3);
  {
    const Eigen::Vector3d g =
        quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
    for (int k = 0; k < 3; ++k) {
      f_ang_vel[k] = static_cast<float>(rs.imu_gyro[k] * CLOBOT_ANG_VEL_SCALE);
      f_gravity[k] = static_cast<float>(g[k]);
    }
  }
  std::vector<float> f_cmd = {
      static_cast<float>(drive[0]),
      static_cast<float>(drive[1]),
      static_cast<float>(drive[2])
  };

  std::vector<float> f_jpos(CLOBOT_NUM_ACTIONS);
  std::vector<float> f_jvel(CLOBOT_NUM_ACTIONS);
  for (int i = 0; i < CLOBOT_NUM_ACTIONS; ++i) {
    const int sdk = MUJOCO_TO_ISAACLAB[i];
    f_jpos[i] = static_cast<float>(rs.motor_q[sdk] - CLOBOT_DEFAULT_ISAAC[i]);
    f_jvel[i] = static_cast<float>(rs.motor_dq[sdk] * CLOBOT_DOF_VEL_SCALE);
  }

  std::vector<float> f_act(self.last_action_.begin(), self.last_action_.end());

  std::vector<float> f_gait(CLOBOT_GAIT_DIM);
  {
    const std::array<double, CLOBOT_GAIT_DIM> ph =
        clobot_gait_phase(gait_seconds);
    for (int k = 0; k < CLOBOT_GAIT_DIM; ++k) {
      f_gait[k] = static_cast<float>(ph[k]);
    }
  }

  term_push(self.hist_[CLOBOT_H_ANG_VEL], std::move(f_ang_vel));
  term_push(self.hist_[CLOBOT_H_GRAVITY], std::move(f_gravity));
  term_push(self.hist_[CLOBOT_H_CMD], std::move(f_cmd));
  term_push(self.hist_[CLOBOT_H_JPOS], std::move(f_jpos));
  term_push(self.hist_[CLOBOT_H_JVEL], std::move(f_jvel));
  term_push(self.hist_[CLOBOT_H_ACT], std::move(f_act));
  term_push(self.hist_[CLOBOT_H_GAIT], std::move(f_gait));

  std::vector<float> obs;
  obs.reserve(CLOBOT_NUM_OBS);
  for (const TermHistory& h : self.hist_) term_append_to(h, obs);
  if (obs.size() != static_cast<size_t>(CLOBOT_NUM_OBS)) {
    throw std::runtime_error("clobot: bad observation size");
  }

  self.last_action_ = clobot_engine_run(*self.engine_, obs);

  for (int i = 0; i < CLOBOT_NUM_ACTIONS; ++i) {
    const double a = std::clamp(
        static_cast<double>(self.last_action_[i]),
        -CLOBOT_ACTION_CLIP,
        CLOBOT_ACTION_CLIP
    );
    const int sdk = MUJOCO_TO_ISAACLAB[i];
    out.q_target[sdk] =
        a * CLOBOT_ACTION_SCALE_ISAAC[i] + CLOBOT_DEFAULT_ISAAC[i];
    out.kp[sdk] = static_cast<float>(CLOBOT_KPS_SDK[sdk]);
    out.kd[sdk] = static_cast<float>(CLOBOT_KDS_SDK[sdk]);
    out.owns[sdk] = sdk < self.owned_end_;
  }
  return out;
}

std::shared_ptr<Clobot> policy_make(int owned_end) {
  const std::shared_ptr<Clobot> self = std::make_shared<Clobot>();
  self->owned_end_ = owned_end;
  self->last_action_ = std::vector<float>(CLOBOT_NUM_ACTIONS, 0.0f);

  for (TermHistory& h : self->hist_) h.len = CLOBOT_HISTORY;

  self->engine_ = clobot_engine_init(CLOBOT_MODEL_PATH);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(CLOBOT_OWNED_END); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "clobot"; }

 private:
  std::shared_ptr<Clobot> state_;
};

class WithArmsPolicy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(CLOBOT_WITH_ARMS_OWNED_END); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "clobot_with_arms"; }

 private:
  std::shared_ptr<Clobot> state_;
};

}
