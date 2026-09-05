namespace sonic {

const std::string SONIC_PLANNER_PATH = "policies/sonic/planner_sonic_f32.onnx";

const std::string SONIC_ENCODER_PATH = "policies/sonic/model_encoder.onnx";

const std::string SONIC_DECODER_PATH = "policies/sonic/model_decoder.onnx";

const int SONIC_NUM_ACTIONS = 29;

const int SONIC_NUM_OWNED = 15;

const int SONIC_ENC_OBS = 1762;

const int SONIC_DEC_OBS = 994;

const int SONIC_TOKEN = 64;

const int SONIC_HIST = 10;

const int SONIC_REF_FRAMES = 10;

const int SONIC_REF_STEP = 5;

const int SONIC_CTX_FRAMES = 4;

const int SONIC_QPOS = 36;

const int SONIC_PLAN_FRAMES = 64;

const double SONIC_PLAN_HZ = 30.0;

const int SONIC_REPLAN_TICKS = 5;

const int SONIC_LOOKAHEAD = 2;

const int SONIC_BLEND_FRAMES = 8;

const double SONIC_SEED_HEIGHT = 0.788740;

const int SONIC_MODE_IDLE = 0;

const int SONIC_MODE_SLOW_WALK = 1;

const int SONIC_MODE_WALK = 2;

const double SONIC_WALK_SPEED = 0.8;

const double SONIC_IDLE_SPEED = 0.12;

const double SONIC_FACE_LEAD_S = 0.6;

const int SONIC_SEED = 1234;

const int SONIC_MJ_OF_IL[SONIC_NUM_ACTIONS] = {0,  6,  12, 1,  7,  13, 2,  8,
                                               14, 3,  9,  15, 22, 4,  10, 16,
                                               23, 5,  11, 17, 24, 18, 25, 19,
                                               26, 20, 27, 21, 28};

const int SONIC_IL_OF_MJ[SONIC_NUM_ACTIONS] = {0,  3,  6,  9,  13, 17, 1,  4,
                                               7,  10, 14, 18, 2,  5,  8,  11,
                                               15, 19, 21, 23, 25, 27, 12, 16,
                                               20, 22, 24, 26, 28};

const int SONIC_LOWER_IL[12] = {0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18};

const double SONIC_DEFAULT_ANGLES[SONIC_NUM_ACTIONS] = {
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0,  0.0, 0.669,
    -0.363, 0.0, 0.0, 0.0,   0.0,    0.2, 0.2,    0.0,  0.6, 0.0,
    0.0,    0.0, 0.2, -0.2,  0.0,    0.6, 0.0,    0.0,  0.0
};

const double SONIC_OMEGA = 2.0 * M_PI * 10.0;

const double SONIC_ZETA = 2.0;

const double SONIC_ARMATURE[4] = {0.003609725, 0.010177520, 0.025101925, 0.00425};

const double SONIC_EFFORT[4] = {25.0, 88.0, 139.0, 5.0};

const int SONIC_MOTOR[SONIC_NUM_ACTIONS] = {2, 2, 1, 2, 0, 0, 2, 2, 1, 2,
                                            0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                            3, 3, 0, 0, 0, 0, 0, 3, 3};

inline bool sonic_double_gain(int mj) {
  return mj == 4 || mj == 5 || mj == 10 || mj == 11 || mj == 13 || mj == 14;
}

inline double sonic_stiffness(int mj) {
  return SONIC_ARMATURE[SONIC_MOTOR[mj]] * SONIC_OMEGA * SONIC_OMEGA;
}

inline double sonic_damping(int mj) {
  return 2.0 * SONIC_ZETA * SONIC_ARMATURE[SONIC_MOTOR[mj]] * SONIC_OMEGA;
}

const double SONIC_MAX_VX = 0.6;

const double SONIC_MAX_VY = 0.4;

const double SONIC_MAX_YAW_RATE = 1.0;

const Limits LIMITS = [] {
  Limits l{};
  l.vx_min = -SONIC_MAX_VX;
  l.vx_max = SONIC_MAX_VX;
  l.vy_abs = SONIC_MAX_VY;
  l.yaw_rate_abs = SONIC_MAX_YAW_RATE;
  l.speed_norm = SONIC_MAX_VX;
  return l;
}();

struct SonicFrame {
  Eigen::Vector3d pos;
  Eigen::Quaterniond quat;
  std::array<double, SONIC_NUM_ACTIONS> q;
  std::array<double, SONIC_NUM_ACTIONS> dq;
};

struct Sonic {
  bool pos_reached_ = false;
  bool yaw_reached_ = false;

  std::shared_ptr<Engine> planner_;
  std::shared_ptr<Engine> encoder_;
  std::shared_ptr<Engine> decoder_;

  std::vector<SonicFrame> motion_;
  int cursor_ = 0;
  int ticks_ = 0;

  Eigen::Quaterniond to_world_ = Eigen::Quaterniond::Identity();

  std::array<std::array<double, SONIC_NUM_ACTIONS>, SONIC_HIST> hist_q_{};
  std::array<std::array<double, SONIC_NUM_ACTIONS>, SONIC_HIST> hist_dq_{};
  std::array<std::array<double, SONIC_NUM_ACTIONS>, SONIC_HIST> hist_act_{};
  std::array<std::array<double, 3>, SONIC_HIST> hist_gyro_{};
  std::array<std::array<double, 3>, SONIC_HIST> hist_grav_{};

  std::array<double, SONIC_NUM_ACTIONS> last_action_{};
};

inline double sonic_yaw_of(const Eigen::Quaterniond& q) {
  return std::atan2(
      2.0 * (q.w() * q.z() + q.x() * q.y()),
      1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z())
  );
}

inline Eigen::Quaterniond sonic_heading_of(const Eigen::Quaterniond& q) {
  return Eigen::Quaterniond(
      Eigen::AngleAxisd(sonic_yaw_of(q), Eigen::Vector3d::UnitZ())
  );
}

inline void sonic_ori6(
    const Eigen::Quaterniond& q,
    double out[6]
) {
  const Eigen::Matrix3d m = q.toRotationMatrix();
  out[0] = m(0, 0);
  out[1] = m(0, 1);
  out[2] = m(1, 0);
  out[3] = m(1, 1);
  out[4] = m(2, 0);
  out[5] = m(2, 1);
}

std::shared_ptr<Engine> sonic_engine_init(
    const std::string& path,
    const char* in_name,
    int in_count,
    const char* out_name,
    int out_count
) {
  std::cout << "sonic: loading " << path << std::endl;
  const std::shared_ptr<Engine> e = engine_init(path);
  engine_expect(*e, in_name, static_cast<size_t>(in_count), true, "sonic");
  engine_expect(*e, out_name, static_cast<size_t>(out_count), false, "sonic");
  return e;
}

int sonic_plan(
    Engine& e,
    const double ctx[SONIC_CTX_FRAMES][SONIC_QPOS],
    int mode,
    double speed,
    const double move[3],
    const double face[3],
    int seed,
    std::vector<std::array<double, SONIC_QPOS>>& out
) {
  std::vector<float>& ctx_in = engine_input(e, "context_mujoco_qpos").data;
  for (int f = 0; f < SONIC_CTX_FRAMES; ++f) {
    for (int j = 0; j < SONIC_QPOS; ++j) {
      ctx_in[f * SONIC_QPOS + j] = static_cast<float>(ctx[f][j]);
    }
  }
  engine_input(e, "mode").data[0] = static_cast<float>(mode);
  engine_input(e, "target_vel").data[0] = static_cast<float>(speed);
  engine_input(e, "random_seed").data[0] = static_cast<float>(seed);

  engine_input(e, "height").data[0] = -1.0f;
  for (int k = 0; k < 3; ++k) {
    engine_input(e, "movement_direction").data[k] = static_cast<float>(move[k]);
    engine_input(e, "facing_direction").data[k] = static_cast<float>(face[k]);
  }

  engine_input(e, "has_specific_target").data[0] = 0.0f;
  std::fill(
      engine_input(e, "specific_target_positions").data.begin(),
      engine_input(e, "specific_target_positions").data.end(),
      0.0f
  );
  std::fill(
      engine_input(e, "specific_target_headings").data.begin(),
      engine_input(e, "specific_target_headings").data.end(),
      0.0f
  );

  std::vector<float>& allowed = engine_input(e, "allowed_pred_num_tokens").data;
  std::fill(allowed.begin(), allowed.end(), 0.0f);
  for (size_t k = 0; k < 6 && k < allowed.size(); ++k) allowed[k] = 1.0f;

  engine_run(e);

  const std::vector<float>& qpos = engine_output(e, "mujoco_qpos").data;
  const int frames = std::clamp(
      static_cast<int>(std::lround(engine_output(e, "num_pred_frames").data[0])),
      0,
      SONIC_PLAN_FRAMES
  );
  out.resize(static_cast<size_t>(frames));
  for (int f = 0; f < frames; ++f) {
    for (int j = 0; j < SONIC_QPOS; ++j) {
      out[static_cast<size_t>(f)][static_cast<size_t>(j)] =
          static_cast<double>(qpos[f * SONIC_QPOS + j]);
    }
  }
  return frames;
}

void sonic_resample(
    const std::vector<std::array<double, SONIC_QPOS>>& raw,
    std::vector<SonicFrame>& out
) {
  const int n30 = static_cast<int>(raw.size());
  out.clear();
  if (n30 < 2) return;

  const int n50 = static_cast<int>(
      std::floor(static_cast<double>(n30) / SONIC_PLAN_HZ * (1.0 / PERIOD_S))
  );
  out.resize(static_cast<size_t>(n50));
  for (int f = 0; f < n50; ++f) {
    const double t = static_cast<double>(f) * PERIOD_S * SONIC_PLAN_HZ;
    const int f0 = std::min(static_cast<int>(std::floor(t)), n30 - 1);
    const int f1 = std::min(f0 + 1, n30 - 1);
    const double w = t - std::floor(t);

    const std::array<double, SONIC_QPOS>& a = raw[static_cast<size_t>(f0)];
    const std::array<double, SONIC_QPOS>& b = raw[static_cast<size_t>(f1)];
    SonicFrame& fr = out[static_cast<size_t>(f)];
    for (int k = 0; k < 3; ++k) fr.pos[k] = a[k] * (1.0 - w) + b[k] * w;
    fr.quat = Eigen::Quaterniond(a[3], a[4], a[5], a[6])
                  .slerp(w, Eigen::Quaterniond(b[3], b[4], b[5], b[6]));
    fr.quat.normalize();

    for (int j = 0; j < SONIC_NUM_ACTIONS; ++j) {
      const int mj = SONIC_MJ_OF_IL[j];
      fr.q[static_cast<size_t>(j)] =
          a[7 + mj] * (1.0 - w) + b[7 + mj] * w;
    }
  }
  for (int f = 0; f + 1 < n50; ++f) {
    for (int j = 0; j < SONIC_NUM_ACTIONS; ++j) {
      out[static_cast<size_t>(f)].dq[static_cast<size_t>(j)] =
          (out[static_cast<size_t>(f + 1)].q[static_cast<size_t>(j)] -
           out[static_cast<size_t>(f)].q[static_cast<size_t>(j)]) /
          PERIOD_S;
    }
  }
  if (n50 >= 2) out[static_cast<size_t>(n50 - 1)].dq = out[static_cast<size_t>(n50 - 2)].dq;
}

inline const SonicFrame& sonic_at(
    const Sonic& self,
    int f
) {
  const int last = static_cast<int>(self.motion_.size()) - 1;
  return self.motion_[static_cast<size_t>(std::clamp(f, 0, last))];
}

void sonic_context(
    const Sonic& self,
    int from,
    double ctx[SONIC_CTX_FRAMES][SONIC_QPOS]
) {
  for (int n = 0; n < SONIC_CTX_FRAMES; ++n) {
    const double t =
        static_cast<double>(from) * PERIOD_S + static_cast<double>(n) / SONIC_PLAN_HZ;
    const double f50 = t / PERIOD_S;
    const int f0 = static_cast<int>(std::floor(f50));
    const double w = f50 - std::floor(f50);
    const SonicFrame& a = sonic_at(self, f0);
    const SonicFrame& b = sonic_at(self, f0 + 1);

    for (int k = 0; k < 3; ++k) ctx[n][k] = a.pos[k] * (1.0 - w) + b.pos[k] * w;
    const Eigen::Quaterniond q = a.quat.slerp(w, b.quat);
    ctx[n][3] = q.w();
    ctx[n][4] = q.x();
    ctx[n][5] = q.y();
    ctx[n][6] = q.z();
    for (int j = 0; j < SONIC_NUM_ACTIONS; ++j) {
      ctx[n][7 + SONIC_MJ_OF_IL[j]] =
          a.q[static_cast<size_t>(j)] * (1.0 - w) + b.q[static_cast<size_t>(j)] * w;
    }
  }
}

void sonic_blend(
    Sonic& self,
    const std::vector<SonicFrame>& gen,
    int anchor
) {
  if (gen.empty()) return;
  if (self.motion_.empty()) {
    self.motion_ = gen;
    self.cursor_ = 0;
    return;
  }

  const int length = anchor + static_cast<int>(gen.size());
  std::vector<SonicFrame> merged(static_cast<size_t>(length));
  for (int f = 0; f < length; ++f) {
    const SonicFrame& old = sonic_at(self, f + self.cursor_);
    const SonicFrame& fresh =
        gen[static_cast<size_t>(std::clamp(f - anchor, 0, static_cast<int>(gen.size()) - 1))];
    const double w_new = std::clamp(
        static_cast<double>(f - anchor) / static_cast<double>(SONIC_BLEND_FRAMES),
        0.0,
        1.0
    );

    SonicFrame& out = merged[static_cast<size_t>(f)];
    out.pos = old.pos * (1.0 - w_new) + fresh.pos * w_new;
    out.quat = old.quat.slerp(w_new, fresh.quat);
    for (int j = 0; j < SONIC_NUM_ACTIONS; ++j) {
      const size_t s = static_cast<size_t>(j);
      out.q[s] = old.q[s] * (1.0 - w_new) + fresh.q[s] * w_new;
      out.dq[s] = old.dq[s] * (1.0 - w_new) + fresh.dq[s] * w_new;
    }
  }
  self.motion_ = std::move(merged);
  self.cursor_ = 0;
}

void sonic_replan(
    Sonic& self,
    int mode,
    double speed,
    const double move[3],
    const double face[3],
    bool first,
    const double* seed_pose
) {
  double ctx[SONIC_CTX_FRAMES][SONIC_QPOS] = {};
  if (first) {

    for (int n = 0; n < SONIC_CTX_FRAMES; ++n) {
      ctx[n][2] = SONIC_SEED_HEIGHT;
      ctx[n][3] = 1.0;
      for (int j = 0; j < SONIC_NUM_ACTIONS; ++j) ctx[n][7 + j] = seed_pose[j];
    }
  } else {
    sonic_context(self, self.cursor_ + SONIC_LOOKAHEAD, ctx);
  }

  std::vector<std::array<double, SONIC_QPOS>> raw;
  sonic_plan(*self.planner_, ctx, mode, speed, move, face, SONIC_SEED, raw);

  std::vector<SonicFrame> gen;
  sonic_resample(raw, gen);
  if (gen.empty()) return;
  sonic_blend(self, gen, first ? 0 : SONIC_LOOKAHEAD);
}

void sonic_encoder_obs(
    const Sonic& self,
    const Eigen::Quaterniond& base_quat,
    std::vector<float>& obs
) {
  std::fill(obs.begin(), obs.end(), 0.0f);

  obs[0] = 0.0f;

  int idx[SONIC_REF_FRAMES];
  for (int k = 0; k < SONIC_REF_FRAMES; ++k) {
    idx[k] = self.cursor_ + k * SONIC_REF_STEP;
  }

  for (int k = 0; k < SONIC_REF_FRAMES; ++k) {
    const SonicFrame& fr = sonic_at(self, idx[k]);
    for (int j = 0; j < SONIC_NUM_ACTIONS; ++j) {
      obs[4 + k * SONIC_NUM_ACTIONS + j] =
          static_cast<float>(fr.q[static_cast<size_t>(j)]);
      obs[294 + k * SONIC_NUM_ACTIONS + j] =
          static_cast<float>(fr.dq[static_cast<size_t>(j)]);
    }
  }

  for (int k = 0; k < SONIC_REF_FRAMES; ++k) {
    obs[584 + k] = static_cast<float>(sonic_at(self, idx[k]).pos[2]);
  }
  obs[594] = static_cast<float>(sonic_at(self, self.cursor_).pos[2]);

  const Eigen::Quaterniond base_inv = base_quat.conjugate();
  double ori[6];
  sonic_ori6(base_inv * (self.to_world_ * sonic_at(self, self.cursor_).quat), ori);
  for (int k = 0; k < 6; ++k) obs[595 + k] = static_cast<float>(ori[k]);
  for (int k = 0; k < SONIC_REF_FRAMES; ++k) {
    sonic_ori6(base_inv * (self.to_world_ * sonic_at(self, idx[k]).quat), ori);
    for (int c = 0; c < 6; ++c) obs[601 + k * 6 + c] = static_cast<float>(ori[c]);
  }

  for (int k = 0; k < SONIC_REF_FRAMES; ++k) {
    const SonicFrame& fr = sonic_at(self, idx[k]);
    for (int j = 0; j < 12; ++j) {
      const size_t s = static_cast<size_t>(SONIC_LOWER_IL[j]);
      obs[661 + k * 12 + j] = static_cast<float>(fr.q[s]);
      obs[781 + k * 12 + j] = static_cast<float>(fr.dq[s]);
    }
  }
}

void sonic_decoder_obs(
    const Sonic& self,
    const std::vector<float>& token,
    std::vector<float>& obs
) {
  for (int k = 0; k < SONIC_TOKEN; ++k) obs[k] = token[static_cast<size_t>(k)];

  for (int f = 0; f < SONIC_HIST; ++f) {
    for (int k = 0; k < 3; ++k) {
      obs[64 + f * 3 + k] = static_cast<float>(self.hist_gyro_[static_cast<size_t>(f)][static_cast<size_t>(k)]);
      obs[964 + f * 3 + k] = static_cast<float>(self.hist_grav_[static_cast<size_t>(f)][static_cast<size_t>(k)]);
    }
    for (int j = 0; j < SONIC_NUM_ACTIONS; ++j) {
      const size_t ff = static_cast<size_t>(f);
      const size_t jj = static_cast<size_t>(j);
      obs[94 + f * SONIC_NUM_ACTIONS + j] = static_cast<float>(self.hist_q_[ff][jj]);
      obs[384 + f * SONIC_NUM_ACTIONS + j] = static_cast<float>(self.hist_dq_[ff][jj]);
      obs[674 + f * SONIC_NUM_ACTIONS + j] = static_cast<float>(self.hist_act_[ff][jj]);
    }
  }
}

void sonic_push_history(
    Sonic& self,
    const RobotState& rs
) {
  for (int f = 0; f + 1 < SONIC_HIST; ++f) {
    const size_t a = static_cast<size_t>(f);
    const size_t b = static_cast<size_t>(f + 1);
    self.hist_q_[a] = self.hist_q_[b];
    self.hist_dq_[a] = self.hist_dq_[b];
    self.hist_act_[a] = self.hist_act_[b];
    self.hist_gyro_[a] = self.hist_gyro_[b];
    self.hist_grav_[a] = self.hist_grav_[b];
  }
  const size_t now = static_cast<size_t>(SONIC_HIST - 1);
  for (int j = 0; j < SONIC_NUM_ACTIONS; ++j) {
    const int mj = SONIC_MJ_OF_IL[j];
    self.hist_q_[now][static_cast<size_t>(j)] =
        rs.motor_q[mj] - SONIC_DEFAULT_ANGLES[mj];
    self.hist_dq_[now][static_cast<size_t>(j)] = rs.motor_dq[mj];
  }

  self.hist_act_[now] = self.last_action_;
  for (int k = 0; k < 3; ++k) self.hist_gyro_[now][static_cast<size_t>(k)] = rs.imu_gyro[k];
  const Eigen::Vector3d gravity =
      quat_rot_vec(quat_conj(rs.imu_quat), Eigen::Vector3d(0.0, 0.0, -1.0));
  for (int k = 0; k < 3; ++k) self.hist_grav_[now][static_cast<size_t>(k)] = gravity[k];
}

Output policy_step(
    Sonic& self,
    const Input& in
) {
  const RobotState& rs = *in.state;
  double drive[3];
  command_from_target(in, LIMITS, self.pos_reached_, self.yaw_reached_, drive);

  const Eigen::Quaterniond base(
      rs.imu_quat[0], rs.imu_quat[1], rs.imu_quat[2], rs.imu_quat[3]
  );

  const double psi = sonic_yaw_of(self.to_world_.conjugate() * base);
  const double speed = std::hypot(drive[0], drive[1]);
  const double course = psi + std::atan2(drive[1], drive[0]);
  const double face_yaw = psi + drive[2] * SONIC_FACE_LEAD_S;

  const bool stepping = speed > SONIC_IDLE_SPEED;
  const int mode = !stepping ? SONIC_MODE_IDLE
                   : speed < SONIC_WALK_SPEED ? SONIC_MODE_SLOW_WALK
                                              : SONIC_MODE_WALK;
  const double move[3] = {
      stepping ? std::cos(course) : 0.0, stepping ? std::sin(course) : 0.0, 0.0
  };
  const double face[3] = {std::cos(face_yaw), std::sin(face_yaw), 0.0};

  if (self.motion_.empty()) {
    double seed_pose[SONIC_NUM_ACTIONS];
    for (int j = 0; j < SONIC_NUM_ACTIONS; ++j) seed_pose[j] = rs.motor_q[j];
    sonic_replan(self, SONIC_MODE_IDLE, 0.0, move, face, true, seed_pose);

    self.to_world_ = sonic_heading_of(base) *
                     sonic_heading_of(sonic_at(self, 0).quat).conjugate();
  } else if (self.ticks_ % SONIC_REPLAN_TICKS == 0) {
    sonic_replan(self, mode, stepping ? speed : 0.0, move, face, false, nullptr);
  }

  sonic_push_history(self, rs);

  std::vector<float> enc_obs(SONIC_ENC_OBS);
  sonic_encoder_obs(self, base, enc_obs);
  const std::vector<float> token =
      engine_run_single(*self.encoder_, enc_obs.data(), enc_obs.size());

  std::vector<float> dec_obs(SONIC_DEC_OBS);
  sonic_decoder_obs(self, token, dec_obs);
  const std::vector<float>& action =
      engine_run_single(*self.decoder_, dec_obs.data(), dec_obs.size());

  for (int j = 0; j < SONIC_NUM_ACTIONS; ++j) {
    self.last_action_[static_cast<size_t>(j)] =
        static_cast<double>(action[static_cast<size_t>(j)]);
  }

  self.cursor_ =
      std::min(self.cursor_ + 1, static_cast<int>(self.motion_.size()) - 1);
  ++self.ticks_;

  Output out{};
  for (int mj = 0; mj < SONIC_NUM_OWNED; ++mj) {
    const double stiffness =
        sonic_stiffness(mj) * (sonic_double_gain(mj) ? 2.0 : 1.0);
    const double scale = 0.25 * SONIC_EFFORT[SONIC_MOTOR[mj]] / sonic_stiffness(mj);
    out.q_target[mj] =
        SONIC_DEFAULT_ANGLES[mj] + self.last_action_[static_cast<size_t>(SONIC_IL_OF_MJ[mj])] * scale;
    out.kp[mj] = static_cast<float>(stiffness);
    out.kd[mj] =
        static_cast<float>(sonic_damping(mj) * (sonic_double_gain(mj) ? 2.0 : 1.0));
    out.owns[mj] = true;
  }
  return out;
}

std::shared_ptr<Sonic> policy_make() {
  const std::shared_ptr<Sonic> self = std::make_shared<Sonic>();
  self->planner_ = engine_init(SONIC_PLANNER_PATH);
  engine_expect(
      *self->planner_,
      "context_mujoco_qpos",
      static_cast<size_t>(SONIC_CTX_FRAMES * SONIC_QPOS),
      true,
      "sonic"
  );
  engine_expect(
      *self->planner_,
      "mujoco_qpos",
      static_cast<size_t>(SONIC_PLAN_FRAMES * SONIC_QPOS),
      false,
      "sonic"
  );
  self->encoder_ = sonic_engine_init(
      SONIC_ENCODER_PATH, "obs_dict", SONIC_ENC_OBS, "encoded_tokens", SONIC_TOKEN
  );
  self->decoder_ = sonic_engine_init(
      SONIC_DECODER_PATH, "obs_dict", SONIC_DEC_OBS, "action", SONIC_NUM_ACTIONS
  );
  self->motion_.reserve(1500);
  return self;
}

class Policy : public ModelPolicy {
 public:
  void init() override { state_ = policy_make(); }

  Output step(const Input& in) override { return policy_step(*state_, in); }

  const char* name() const override { return "sonic"; }

 private:
  std::shared_ptr<Sonic> state_;
};

}
