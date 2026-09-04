namespace run_residual {

const int RUN_NUM_ACTIONS = 29;

const int RUN_HISTORY = 5;

const int RUN_NUM_OBS = RUN_HISTORY * (3 + 3 + 3 + 3 * RUN_NUM_ACTIONS);

const size_t RUN_MOTION_DIM = 58;

const size_t RUN_CMD_DIM = 3;

const size_t RUN_HIDDEN_DIM = 512;

const std::string RUN_CMG_PATH = "policies/run_residual/model_cmg.onnx";

const std::string RUN_POLICY_PATH = "policies/run_residual/model_residual.onnx";

const double RUN_ACTION_SCALE = 0.25;

const int RUN_FIRST_ARM = 15;

const float RUN_KPS[RUN_NUM_ACTIONS] = {
    70.0f, 70.0f, 70.0f,  120.0f, 40.0f, 40.0f, 70.0f, 70.0f, 70.0f, 120.0f,
    40.0f, 40.0f, 150.0f, 40.0f,  40.0f, 40.0f, 40.0f, 40.0f, 40.0f, 40.0f,
    40.0f, 40.0f, 40.0f,  40.0f,  40.0f, 40.0f, 40.0f, 40.0f, 40.0f
};

const float RUN_KDS[RUN_NUM_ACTIONS] = {2.5f, 4.0f, 2.5f, 5.2f, 2.0f, 2.0f,
                                        2.5f, 4.0f, 2.5f, 5.2f, 2.0f, 2.0f,
                                        5.0f, 5.0f, 5.0f, 2.0f, 2.0f, 2.0f,
                                        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
                                        2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

const int RUN_TERM_DIM[6] =
    {3, 3, 3, RUN_NUM_ACTIONS, RUN_NUM_ACTIONS, RUN_NUM_ACTIONS};

const double RUN_DEFAULT_ISAAC[RUN_NUM_ACTIONS] = {
    -0.1, -0.1, 0.0,  0.0,  0.0,   0.0,  0.0,   0.0, 0.0, 0.3,
    0.3,  0.3,  0.3,  -0.2, -0.2,  0.25, -0.25, 0.0, 0.0, 0.0,
    0.0,  0.97, 0.97, 0.15, -0.15, 0.0,  0.0,   0.0, 0.0
};

const double RUN_AR_LEAK = 0.05;

const double RUN_CMG_CMD_EMA = 0.40;

const double RUN_CMG_SIGMA_CLAMP = 3.0;

const double RUN_CMG_OUTPUT_CLAMP = 3.14;

const double RUN_CMG_MOTION_MEAN[RUN_MOTION_DIM] = {
    -0.138017774,
    -0.0190545321,
    0,
    0.675016761,
    -0.0547883585,
    0,
    -0.138017803,
    0.0190545227,
    0,
    0.67501688,
    -0.0547883362,
    0,
    -5.62028896e-11,
    -3.69755875e-12,
    0.0425066724,
    0.2713103,
    0.373977512,
    -0.410876751,
    0.636402845,
    0,
    0,
    0,
    0.27131018,
    -0.373977512,
    0.410876781,
    0.636402309,
    0,
    0,
    0,
    0.0178835653,
    6.16252141e-19,
    0,
    -0.000563064765,
    0.0260602366,
    0,
    0.0178835429,
    -6.16252296e-19,
    0,
    -0.000563072914,
    0.0260602403,
    0,
    -1.01259494e-27,
    -5.15739233e-26,
    -0.0145212635,
    0.0178965162,
    -1.47961353e-18,
    -1.43513841e-18,
    -0.00474209245,
    0,
    0,
    0,
    0.01789652,
    1.47961219e-18,
    1.43514471e-18,
    -0.00474209432,
    0,
    0,
    0
};

const double RUN_CMG_MOTION_STD[RUN_MOTION_DIM] = {
    0.306055456, 0.104258813, 0.100000001, 0.311078489, 0.231486648,
    0.100000001, 0.306055456, 0.104258798, 0.100000001, 0.31107825,
    0.231486648, 0.100000001, 0.190625548, 0.100000001, 0.100000001,
    0.215748578, 0.100000001, 0.217310473, 0.541980505, 0.100000001,
    0.100000001, 0.100000001, 0.215748414, 0.100000001, 0.217310473,
    0.541980505, 0.100000001, 0.100000001, 0.100000001, 1.94409943,
    0.100000001, 0.100000001, 3.34542727,  2.35869026,  0.100000001,
    1.94409919,  0.100000001, 0.100000001, 3.34542727,  2.35869026,
    0.100000001, 0.100000001, 0.100000001, 0.390017211, 1.29102993,
    0.100000001, 0.100000001, 1.3150996,   0.100000001, 0.100000001,
    0.100000001, 1.29102993,  0.100000001, 0.100000001, 1.3150996,
    0.100000001, 0.100000001, 0.100000001
};

const double RUN_CMG_CMD_MEAN[RUN_CMD_DIM] = {
    0.882105529,
    -2.06150638e-10,
    -6.58186339e-10
};

const double RUN_CMG_CMD_STD[RUN_CMD_DIM] = {
    0.770421565,
    0.196364716,
    0.595767558
};

const double RUN_VX_MIN = 0.0;

const double RUN_VX_MAX = 1.5;

const double RUN_VY_ABS = 0.0;

const double RUN_YAW_RATE_ABS = 1.0;

const double RUN_SPEED_FLOOR = 0.50;

const double RUN_HOLD_POS_M = 0.20;

const double RUN_HOLD_YAW_RAD = 0.12;

const Limits LIMITS =
    Limits{RUN_VX_MIN, RUN_VX_MAX, RUN_VY_ABS, RUN_YAW_RATE_ABS, 0.0};

struct RunResidual {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> cmg_;
  std::shared_ptr<Engine> actor_;
  std::vector<float> obs_;
  std::vector<float> last_action_;
  std::vector<float> lstm_h_;
  std::vector<float> lstm_c_;
  std::vector<double> prev_motion_;
  std::vector<double> smoothed_cmd_;
  std::vector<double> qref_;
  int term_off_[6] = {0, 0, 0, 0, 0, 0};
  bool cmg_started_ = false;
  bool primed_ = false;
};

struct RunTensor {
  const std::string name;
  const size_t elements;
};

std::shared_ptr<Engine> run_engine_init(
    const std::string& model_path,
    const std::vector<RunTensor>& inputs,
    const std::vector<RunTensor>& outputs
) {
  std::cout << "run_residual: loading " << model_path << std::endl;
  const std::shared_ptr<Engine> loaded = engine_init(model_path);
  for (const RunTensor& t : inputs) {
    engine_expect(*loaded, t.name, t.elements, true, "run_residual");
  }
  for (const RunTensor& t : outputs) {
    engine_expect(*loaded, t.name, t.elements, false, "run_residual");
  }
  std::cout << "run_residual: engine ready, "
            << engine_tensor_list(loaded->inputs) << " -> "
            << engine_tensor_list(loaded->outputs) << std::endl;
  return loaded;
}

void run_engine_step(
    Engine& s,
    const std::vector<std::pair<
        std::string,
        const std::vector<float>*>>& in,
    const std::vector<std::pair<
        std::string,
        std::vector<float>*>>& out
) {
  for (const std::pair<std::string, const std::vector<float>*>& p : in) {
    std::vector<float>& dst = engine_input(s, p.first).data;
    if (dst.size() != p.second->size()) {
      throw std::runtime_error("run_residual: bad size for " + p.first);
    }
    std::copy(p.second->begin(), p.second->end(), dst.begin());
  }
  engine_run(s);
  for (const std::pair<std::string, std::vector<float>*>& p : out) {
    const std::vector<float>& src = engine_output(s, p.first).data;
    if (src.size() != p.second->size()) {
      throw std::runtime_error("run_residual: bad size for " + p.first);
    }
    std::copy(src.begin(), src.end(), p.second->begin());
  }
}

void run_prime_term(
    RunResidual& p,
    int t,
    const float* frame
) {
  const int dim = RUN_TERM_DIM[t];
  float* base = p.obs_.data() + p.term_off_[t];
  for (int h = 0; h < RUN_HISTORY; ++h) {
    std::copy(frame, frame + dim, base + h * dim);
  }
}

void run_push_term(
    RunResidual& p,
    int t,
    const float* frame
) {
  const int dim = RUN_TERM_DIM[t];
  float* base = p.obs_.data() + p.term_off_[t];
  std::copy(base + dim, base + RUN_HISTORY * dim, base);
  std::copy(frame, frame + dim, base + (RUN_HISTORY - 1) * dim);
}

void run_obs_frame(
    const RunResidual& p,
    const RobotState& rs,
    const Input&,
    float frame[6][RUN_NUM_ACTIONS],
    const double drive[3]
) {
  for (int k = 0; k < 3; ++k) {
    frame[0][k] = static_cast<float>(rs.imu_gyro[k]);
  }
  const Eigen::Vector3d gravity_dir =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int k = 0; k < 3; ++k) {
    frame[1][k] = static_cast<float>(gravity_dir[k]);
  }
  frame[2][0] = static_cast<float>(drive[0]);
  frame[2][1] = static_cast<float>(drive[1]);
  frame[2][2] = static_cast<float>(drive[2]);

  for (int i = 0; i < RUN_NUM_ACTIONS; ++i) {
    const int motor = MUJOCO_TO_ISAACLAB[i];
    if (motor >= RUN_FIRST_ARM) {
      frame[3][i] = static_cast<float>(
          p.qref_[motor] +
          RUN_ACTION_SCALE * static_cast<double>(p.last_action_[i]) -
          RUN_DEFAULT_ISAAC[i]
      );
      frame[4][i] = 0.0f;
    } else {
      frame[3][i] =
          static_cast<float>(rs.motor_q[motor] - RUN_DEFAULT_ISAAC[i]);
      frame[4][i] = static_cast<float>(rs.motor_dq[motor]);
    }
    frame[5][i] = p.last_action_[i];
  }
}

void run_cmg_step(
    RunResidual& p,
    const RobotState& rs,
    const Input&,
    const double drive[3]
) {
  double measured[RUN_MOTION_DIM];
  for (int m = 0; m < RUN_NUM_ACTIONS; ++m) {
    measured[m] = rs.motor_q[m];
    measured[RUN_NUM_ACTIONS + m] = rs.motor_dq[m];
  }

  if (!p.cmg_started_) {
    std::copy(measured, measured + RUN_MOTION_DIM, p.prev_motion_.begin());
    p.cmg_started_ = true;
  }

  const double cmd_in_raw[RUN_CMD_DIM] = {drive[0], drive[1], drive[2]};
  for (size_t i = 0; i < RUN_CMD_DIM; ++i) {
    p.smoothed_cmd_[i] +=
        RUN_CMG_CMD_EMA * (cmd_in_raw[i] - p.smoothed_cmd_[i]);
  }

  std::vector<float> motion_in(RUN_MOTION_DIM, 0.0f);
  for (size_t i = 0; i < RUN_MOTION_DIM; ++i) {
    const double lo =
        RUN_CMG_MOTION_MEAN[i] - RUN_CMG_SIGMA_CLAMP * RUN_CMG_MOTION_STD[i];
    const double hi =
        RUN_CMG_MOTION_MEAN[i] + RUN_CMG_SIGMA_CLAMP * RUN_CMG_MOTION_STD[i];
    motion_in[i] = static_cast<float>(
        (std::clamp(p.prev_motion_[i], lo, hi) - RUN_CMG_MOTION_MEAN[i]) /
        RUN_CMG_MOTION_STD[i]
    );
  }
  std::vector<float> cmd_in(RUN_CMD_DIM, 0.0f);
  for (size_t i = 0; i < RUN_CMD_DIM; ++i) {
    cmd_in[i] = static_cast<float>(
        (p.smoothed_cmd_[i] - RUN_CMG_CMD_MEAN[i]) / RUN_CMG_CMD_STD[i]
    );
  }

  std::vector<float> motion_out(RUN_MOTION_DIM, 0.0f);
  run_engine_step(
      *p.cmg_,
      {{"prev_motion", &motion_in}, {"command", &cmd_in}},
      {{"motion", &motion_out}}
  );

  double motion_ref[RUN_MOTION_DIM];
  for (size_t i = 0; i < RUN_MOTION_DIM; ++i) {
    motion_ref[i] = std::clamp(
        static_cast<double>(motion_out[i]) * RUN_CMG_MOTION_STD[i] +
            RUN_CMG_MOTION_MEAN[i],
        -RUN_CMG_OUTPUT_CLAMP,
        RUN_CMG_OUTPUT_CLAMP
    );
  }

  for (size_t i = 0; i < RUN_MOTION_DIM; ++i) {
    const int motor = static_cast<int>(i % RUN_NUM_ACTIONS);
    const double leak = motor < RUN_FIRST_ARM ? RUN_AR_LEAK : 0.0;
    p.prev_motion_[i] = (1.0 - leak) * motion_ref[i] + leak * measured[i];
  }

  for (int m = 0; m < RUN_NUM_ACTIONS; ++m) p.qref_[m] = motion_ref[m];
}

Output policy_step(
    RunResidual& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  [&] {
    const bool correcting = in.dist_m > RUN_HOLD_POS_M ||
                            std::fabs(in.yaw_err_rad) > RUN_HOLD_YAW_RAD;
    if (!correcting) return;
    drive[0] = std::clamp(drive[0], RUN_SPEED_FLOOR, RUN_VX_MAX);
    drive[1] = 0.0;
  }();
  Output out{};

  const RobotState& rs = *in.state;

  run_cmg_step(self, rs, in, drive);

  float frame[6][RUN_NUM_ACTIONS];
  run_obs_frame(self, rs, in, frame, drive);
  for (int t = 0; t < 6; ++t) {
    if (self.primed_) {
      run_push_term(self, t, frame[t]);
    } else {
      run_prime_term(self, t, frame[t]);
    }
  }
  self.primed_ = true;

  std::vector<float> actions(RUN_NUM_ACTIONS, 0.0f);
  std::vector<float> h_out(RUN_HIDDEN_DIM, 0.0f);
  std::vector<float> c_out(RUN_HIDDEN_DIM, 0.0f);
  run_engine_step(
      *self.actor_,
      {{"obs", &self.obs_}, {"h_in", &self.lstm_h_}, {"c_in", &self.lstm_c_}},
      {{"actions", &actions}, {"h_out", &h_out}, {"c_out", &c_out}}
  );
  self.lstm_h_ = h_out;
  self.lstm_c_ = c_out;
  self.last_action_ = actions;

  for (int m = 0; m < NUM_MOTOR; ++m) {
    out.q_target[m] =
        self.qref_[m] +
        RUN_ACTION_SCALE *
            static_cast<double>(self.last_action_[ISAACLAB_TO_MUJOCO[m]]);
    out.kp[m] = RUN_KPS[m];
    out.kd[m] = RUN_KDS[m];
    out.owns[m] = m < RUN_FIRST_ARM;
  }
  return out;
}

std::shared_ptr<RunResidual> policy_make() {
  const std::shared_ptr<RunResidual> self = std::make_shared<RunResidual>();
  self->obs_ = std::vector<float>(RUN_NUM_OBS, 0.0f);
  self->last_action_ = std::vector<float>(RUN_NUM_ACTIONS, 0.0f);
  self->lstm_h_ = std::vector<float>(RUN_HIDDEN_DIM, 0.0f);
  self->lstm_c_ = std::vector<float>(RUN_HIDDEN_DIM, 0.0f);
  self->prev_motion_ = std::vector<double>(RUN_MOTION_DIM, 0.0);
  self->smoothed_cmd_ = std::vector<double>(RUN_CMD_DIM, 0.0);
  self->qref_ = std::vector<double>(RUN_NUM_ACTIONS, 0.0);

  int off = 0;
  for (int t = 0; t < 6; ++t) {
    self->term_off_[t] = off;
    off += RUN_HISTORY * RUN_TERM_DIM[t];
  }

  self->cmg_ = run_engine_init(
      RUN_CMG_PATH,
      {{"prev_motion", RUN_MOTION_DIM}, {"command", RUN_CMD_DIM}},
      {{"motion", RUN_MOTION_DIM}}
  );
  self->actor_ = run_engine_init(
      RUN_POLICY_PATH,
      {{"obs", static_cast<size_t>(RUN_NUM_OBS)},
       {"h_in", RUN_HIDDEN_DIM},
       {"c_in", RUN_HIDDEN_DIM}},
      {{"actions", static_cast<size_t>(RUN_NUM_ACTIONS)},
       {"h_out", RUN_HIDDEN_DIM},
       {"c_out", RUN_HIDDEN_DIM}}
  );
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "run_residual"; }

 private:
  std::shared_ptr<RunResidual> state_;
};

}
