namespace amo {

constexpr int AMO_NUM_DOF = 23;

constexpr int AMO_HISTORY_LEN = 10;

constexpr int AMO_EXTRA_HISTORY_LEN = 25;

constexpr int AMO_NUM_ACT = 15;

constexpr int AMO_N_PROPRIO = 3 + 2 + 2 + AMO_NUM_DOF * 3 + 2 + AMO_NUM_ACT;

const std::string AMO_POLICY_PATH = "policies/amo/model.onnx";

const std::string AMO_ADAPTER_PATH = "policies/amo/model_adapter.onnx";

constexpr int AMO_N_PRIV = 3;

constexpr int AMO_NUM_ARM = 8;

constexpr int AMO_N_DEMO = AMO_NUM_ARM + 3 + 3 + 3;

constexpr int AMO_NUM_OBS =
    AMO_N_PROPRIO + AMO_N_DEMO + AMO_N_PRIV + AMO_HISTORY_LEN * AMO_N_PROPRIO;

constexpr int AMO_NUM_EXTRA = AMO_EXTRA_HISTORY_LEN * AMO_N_PROPRIO;

constexpr int AMO_ADAPTER_IN = 4 + AMO_NUM_ARM;

const double AMO_DEFAULT_DOF_POS[AMO_NUM_DOF] = {
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3,  -0.2, 0.0,
    0.0,  0.0, 0.0, 0.5, 0.0,  0.2, 0.3,  0.5, 0.0, -0.2, 0.3
};

const float AMO_KPS[AMO_NUM_ACT] =
    {150, 150, 150, 300, 80, 20, 150, 150, 150, 300, 80, 20, 400, 400, 400};

const float AMO_KDS[AMO_NUM_ACT] =
    {2, 2, 2, 4, 2, 1, 2, 2, 2, 4, 2, 1, 15, 15, 15};

const int AMO_DOF_TO_MOTOR[AMO_NUM_DOF] = {0,  1,  2,  3,  4,  5,  6,  7,
                                           8,  9,  10, 11, 12, 13, 14, 15,
                                           16, 17, 18, 22, 23, 24, 25};

const double AMO_ACTION_SCALE = 0.25;

const double AMO_ACTION_CLIP = 40.0;

const double AMO_ANG_VEL_SCALE = 0.25;

const double AMO_DOF_VEL_SCALE = 0.05;

const double AMO_GAIT_FREQ = 1.3;

const double AMO_CONTROL_DT = 0.02;

const double AMO_TORSO_HEIGHT = 0.75;

const double AMO_TORSO_YAW = 0.0;

const double AMO_TORSO_PITCH = 0.0;

const double AMO_TORSO_ROLL = 0.0;

const double AMO_STAND_SPEED = 0.1;

const double AMO_ADAPTER_INPUT_MEAN[AMO_ADAPTER_IN] = {
    0.6493941816443938,
    0.00835603437027632,
    0.5198069391833688,
    -0.0028908478957657816,
    -0.18861694226437198,
    0.41743932825218666,
    -0.29657144144767367,
    0.6869066831118564,
    -0.20802372898486796,
    -0.4032497266650522,
    0.2749623237878731,
    0.7164985165359848
};

const double AMO_ADAPTER_INPUT_STD[AMO_ADAPTER_IN] = {
    0.08622848682483915,
    0.905570059110108,
    0.5990661102951124,
    0.4037710235670824,
    0.47366042735651465,
    0.4329358814650013,
    0.35596660246191353,
    0.56022898770754,
    0.49676887697830113,
    0.4662358545315813,
    0.32249961752507345,
    0.5674630786438484
};

const double AMO_ADAPTER_OUTPUT_MEAN[AMO_NUM_ACT] = {
    -0.8649367448034146,
    -0.014551485616159491,
    -0.014272786461742208,
    1.0845364423583947,
    -0.44747724731672117,
    -0.015654007296027932,
    -0.8640702505124579,
    0.01381745330674642,
    0.009187855249187705,
    1.0846158293741608,
    -0.4484803223536554,
    0.013866172339457972,
    0.005789055823254164,
    0.00010137134921915922,
    0.19727236245602361
};

const double AMO_ADAPTER_OUTPUT_STD[AMO_NUM_ACT] = {
    0.3966769438866927,
    0.18091131746908673,
    0.1964029137827882,
    0.5146759444113571,
    0.2880384543250629,
    0.07141340497610606,
    0.39789043951513176,
    0.1814397591948851,
    0.19595788541221598,
    0.5149199082108712,
    0.28739879629613463,
    0.07155134830032818,
    0.7115220538732581,
    0.3607543644013931,
    0.32624274557512967
};

const double AMO_MAX_VX = 0.5;

const double AMO_MAX_VY = 0.4;

using ProprioFrame = std::array<float, AMO_N_PROPRIO>;

const Limits LIMITS = Limits{
    .vx_min = -AMO_MAX_VX,
    .vx_max = AMO_MAX_VX,
    .vy_abs = AMO_MAX_VY,
    .yaw_rate_abs = 0.0,
    .speed_norm = 0.0,
    .pos_reached_enter_m = 0.08,
    .pos_reached_exit_m = 0.15
};

struct Amo {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;
  std::shared_ptr<Engine> adapter_;
  std::shared_ptr<Engine> policy_;
  std::deque<ProprioFrame> history_{AMO_HISTORY_LEN, ProprioFrame{}};
  std::deque<ProprioFrame> extra_history_{
      AMO_EXTRA_HISTORY_LEN,
      ProprioFrame{}
  };
  std::array<float, AMO_NUM_DOF> last_action_{};
  std::array<double, 2> gait_cycle_{0.25, 0.25};
  bool in_place_stand_ = true;
  bool primed_ = false;
};

Eigen::Vector3d amo_quat_to_euler(const Eigen::Vector4d& quat) {
  const double qw = quat[0];
  const double qx = quat[1];
  const double qy = quat[2];
  const double qz = quat[3];
  Eigen::Vector3d rpy;
  rpy[0] =
      std::atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy));
  const double sinp = 2.0 * (qw * qy - qz * qx);
  rpy[1] = std::fabs(sinp) >= 1.0 ? std::copysign(M_PI / 2.0, sinp)
                                  : std::asin(sinp);
  rpy[2] =
      std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
  return rpy;
}

Output policy_step(
    Amo& self,
    const Input& in
) {
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);
  Output out{};

  const RobotState& rs = *in.state;

  std::array<double, AMO_NUM_DOF> dof_pos{};
  std::array<double, AMO_NUM_DOF> dof_vel{};
  for (int i = 0; i < AMO_NUM_DOF; ++i) {
    dof_pos[i] = rs.motor_q[AMO_DOF_TO_MOTOR[i]];
    dof_vel[i] = rs.motor_dq[AMO_DOF_TO_MOTOR[i]];
  }

  std::array<float, AMO_NUM_ACT> adapter_out{};
  {
    std::array<float, AMO_ADAPTER_IN> ain{};
    ain[0] = static_cast<float>(AMO_TORSO_HEIGHT);
    ain[1] = static_cast<float>(AMO_TORSO_YAW);
    ain[2] = static_cast<float>(AMO_TORSO_PITCH);
    ain[3] = static_cast<float>(AMO_TORSO_ROLL);
    for (int i = 0; i < AMO_NUM_ARM; ++i) {
      ain[4 + i] = static_cast<float>(dof_pos[AMO_NUM_ACT + i]);
    }
    for (int i = 0; i < AMO_ADAPTER_IN; ++i) {
      ain[i] = static_cast<float>(
          (static_cast<double>(ain[i]) - AMO_ADAPTER_INPUT_MEAN[i]) /
          (AMO_ADAPTER_INPUT_STD[i] + 1e-8)
      );
    }
    const std::vector<float>& t =
        engine_run_single(*self.adapter_, ain.data(), ain.size());
    const float* o = t.data();
    for (int i = 0; i < AMO_NUM_ACT; ++i) {
      adapter_out[i] = static_cast<float>(
          static_cast<double>(o[i]) * AMO_ADAPTER_OUTPUT_STD[i] +
          AMO_ADAPTER_OUTPUT_MEAN[i]
      );
    }
  }

  const Eigen::Vector3d rpy = amo_quat_to_euler(rs.imu_quat);
  const double dyaw =
      (self.in_place_stand_ || !in.has_target) ? 0.0 : -in.yaw_err_rad;

  ProprioFrame frame{};
  int k = 0;
  for (int i = 0; i < 3; ++i) {
    frame[k++] = static_cast<float>(rs.imu_gyro[i] * AMO_ANG_VEL_SCALE);
  }
  frame[k++] = static_cast<float>(rpy[0]);
  frame[k++] = static_cast<float>(rpy[1]);
  frame[k++] = static_cast<float>(std::sin(dyaw));
  frame[k++] = static_cast<float>(std::cos(dyaw));
  for (int i = 0; i < AMO_NUM_DOF; ++i) {
    frame[k++] = static_cast<float>(dof_pos[i] - AMO_DEFAULT_DOF_POS[i]);
  }
  for (int i = 0; i < AMO_NUM_DOF; ++i) {
    frame[k++] = static_cast<float>(dof_vel[i] * AMO_DOF_VEL_SCALE);
  }
  for (int i = 0; i < AMO_NUM_DOF; ++i) frame[k++] = self.last_action_[i];
  frame[k++] = static_cast<float>(std::sin(self.gait_cycle_[0] * 2.0 * M_PI));
  frame[k++] = static_cast<float>(std::sin(self.gait_cycle_[1] * 2.0 * M_PI));
  for (int i = 0; i < AMO_NUM_ACT; ++i) frame[k++] = adapter_out[i];

  if (!self.primed_) {
    self.history_.assign(AMO_HISTORY_LEN, frame);
    self.extra_history_.assign(AMO_EXTRA_HISTORY_LEN, frame);
    self.primed_ = true;
  }

  std::array<float, AMO_NUM_OBS> obs{};
  int p = 0;
  for (int i = 0; i < AMO_N_PROPRIO; ++i) obs[p++] = frame[i];
  for (int i = 0; i < AMO_NUM_ARM; ++i) {
    obs[p++] = static_cast<float>(dof_pos[AMO_NUM_ACT + i]);
  }
  obs[p++] = static_cast<float>(drive[0]);
  obs[p++] = static_cast<float>(drive[1]);
  obs[p++] = 0.0f;
  obs[p++] = static_cast<float>(AMO_TORSO_YAW);
  obs[p++] = static_cast<float>(AMO_TORSO_PITCH);
  obs[p++] = static_cast<float>(AMO_TORSO_ROLL);
  obs[p++] = static_cast<float>(AMO_TORSO_HEIGHT);
  obs[p++] = static_cast<float>(AMO_TORSO_HEIGHT);
  obs[p++] = static_cast<float>(AMO_TORSO_HEIGHT);
  for (int i = 0; i < AMO_N_PRIV; ++i) obs[p++] = 0.0f;
  for (const ProprioFrame& f : self.history_) {
    for (int i = 0; i < AMO_N_PROPRIO; ++i) obs[p++] = f[i];
  }

  self.history_.pop_front();
  self.history_.push_back(frame);
  self.extra_history_.pop_front();
  self.extra_history_.push_back(frame);

  std::array<float, AMO_NUM_EXTRA> extra{};
  int e = 0;
  for (const ProprioFrame& f : self.extra_history_) {
    for (int i = 0; i < AMO_N_PROPRIO; ++i) extra[e++] = f[i];
  }

  std::array<float, AMO_NUM_ACT> action{};
  {
    Engine& e = *self.policy_;
    std::copy(
        obs.begin(),
        obs.end(),
        engine_input(e, "obs_teacher").data.begin()
    );
    std::copy(
        extra.begin(),
        extra.end(),
        engine_input(e, "extra_hist").data.begin()
    );
    engine_run(e);
    const float* a = engine_output(e, "output").data.data();
    for (int i = 0; i < AMO_NUM_ACT; ++i) {
      action[i] = static_cast<float>(std::clamp(
          static_cast<double>(a[i]),
          -AMO_ACTION_CLIP,
          AMO_ACTION_CLIP
      ));
    }
  }

  for (int i = 0; i < AMO_NUM_ACT; ++i) self.last_action_[i] = action[i];
  for (int i = 0; i < AMO_NUM_ARM; ++i) {
    const int d = AMO_NUM_ACT + i;
    self.last_action_[d] = static_cast<float>(
        (dof_pos[d] - AMO_DEFAULT_DOF_POS[d]) / AMO_ACTION_SCALE
    );
  }

  self.in_place_stand_ = std::hypot(drive[0], drive[1]) < AMO_STAND_SPEED;
  for (int i = 0; i < 2; ++i) {
    self.gait_cycle_[i] =
        std::fmod(self.gait_cycle_[i] + AMO_CONTROL_DT * AMO_GAIT_FREQ, 1.0);
  }
  const bool near0 = std::fabs(self.gait_cycle_[0] - 0.25) < 0.05;
  const bool near1 = std::fabs(self.gait_cycle_[1] - 0.25) < 0.05;
  if (self.in_place_stand_ && (near0 || near1)) self.gait_cycle_ = {0.25, 0.25};
  if (!self.in_place_stand_ && near0 && near1) self.gait_cycle_ = {0.25, 0.75};

  for (int i = 0; i < AMO_NUM_ACT; ++i) {
    out.q_target[i] = AMO_DEFAULT_DOF_POS[i] +
                      static_cast<double>(action[i]) * AMO_ACTION_SCALE;
    out.kp[i] = AMO_KPS[i];
    out.kd[i] = AMO_KDS[i];
    out.owns[i] = true;
  }
  return out;
}

std::shared_ptr<Amo> policy_make() {
  const std::shared_ptr<Amo> self = std::make_shared<Amo>();

  const auto load = [](const std::string& path) {
    std::cout << "amo: loading " << path << std::endl;
    return engine_init(path);
  };
  self->adapter_ = load(AMO_ADAPTER_PATH);
  self->policy_ = load(AMO_POLICY_PATH);

  engine_expect(*self->adapter_, "input", AMO_ADAPTER_IN, true, "amo");
  engine_expect(*self->adapter_, "output", AMO_NUM_ACT, false, "amo");
  engine_expect(*self->policy_, "obs_teacher", AMO_NUM_OBS, true, "amo");
  engine_expect(*self->policy_, "extra_hist", AMO_NUM_EXTRA, true, "amo");
  engine_expect(*self->policy_, "output", AMO_NUM_ACT, false, "amo");
  std::cout << "amo: obs " << AMO_NUM_OBS << " + hist " << AMO_NUM_EXTRA
            << ", actions " << AMO_NUM_ACT << " — ready" << std::endl;
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "amo"; }

 private:
  std::shared_ptr<Amo> state_;
};

}
