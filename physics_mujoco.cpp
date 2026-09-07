#include "physics.h"

#include <cuda_runtime.h>
#include <unistd.h>

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <condition_variable>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace {

const char* const MJ_JOINTS[PHYS_NUM_MOTOR] = {
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
};

const int MJ_GROUP[PHYS_NUM_MOTOR] = {0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                                      1, 1, 2, 2, 2, 3, 3, 3, 3, 4,
                                      4, 4, 3, 3, 3, 3, 4, 4, 4};

std::string fleet_xml(
    const std::string& path,
    int count,
    double spacing
) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("mujoco: cannot read " + path);
  const std::string src(
      (std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>()
  );
  const size_t body = src.find("    <body name=\"pelvis\"");
  if (body == std::string::npos) {
    throw std::runtime_error("mujoco: " + path + " has no pelvis body");
  }
  const size_t end = src.find("  </worldbody>", body);
  if (end == std::string::npos) {
    throw std::runtime_error("mujoco: " + path + " has no closing worldbody");
  }
  std::ostringstream out;
  out << src.substr(0, body) << "    <replicate count=\"" << count
      << "\" offset=\"" << spacing << " 0 0\">\n"
      << src.substr(body, end - body) << "    </replicate>\n"
      << src.substr(end);
  std::string xml = out.str();

  const std::filesystem::path dir =
      std::filesystem::absolute(std::filesystem::path(path)).parent_path();
  const std::string mesh_rel = "meshdir=\"meshes\"";
  const size_t at = xml.find(mesh_rel);
  if (at != std::string::npos) {
    xml.replace(
        at,
        mesh_rel.size(),
        "meshdir=\"" + (dir / "meshes").string() + "\""
    );
  }
  return xml;
}

constexpr int SHARD_ROBOTS = 64;

std::string suffix_of(
    int index,
    int count
) {
  int width = 1;
  for (int cap = count; cap >= 10; cap /= 10) ++width;
  std::string s = std::to_string(index);
  while (static_cast<int>(s.size()) < width) s = "0" + s;
  return s;
}

struct Shard {
  mjModel* m = nullptr;
  mjData* d = nullptr;
  int first = 0;
  int envs = 0;
  std::vector<int> qadr;
  std::vector<int> dadr;
  std::vector<int> act;
  std::vector<int> base_qpos;
  std::vector<int> base_dof;
  std::vector<int> jnt_body;
  std::vector<double> jnt_anchor;
  std::vector<int> pelvis_body;
  std::vector<int> torso_body;
  std::vector<int> foot_body;
  std::vector<int> sensor_quat;
  std::vector<int> sensor_gyro;
  std::vector<double> frc_lo;
  std::vector<double> frc_hi;
  std::vector<double> home_x;
  std::vector<int> punched;
  int living = 0;
};

class MujocoPhysics : public Physics {
 public:
  MujocoPhysics(
      const mjcf::Model& model,
      const std::string& path,
      int envs,
      double spacing,
      int threads
  )
      : envs_(envs) {
    (void)model;
    const int per = std::max(1, std::min(SHARD_ROBOTS, envs));
    const int shards = (envs + per - 1) / per;
    const int workers = std::max(1, std::min(threads, shards));

    std::filesystem::create_directories("build");
    const std::string generated =
        "build/_fleet_mujoco_" + std::to_string(getpid()) + ".xml";

    int made = 0;
    mjModel* full = nullptr;
    for (int s = 0; s < shards && made < envs; ++s) {
      const int count = std::min(per, envs - made);
      Shard sh;
      if (count == per && full != nullptr) {
        sh.m = mj_copyModel(nullptr, full);
      } else {
        std::ofstream(generated) << fleet_xml(path, count, spacing);
        char err[1024] = {0};
        sh.m = mj_loadXML(generated.c_str(), nullptr, err, sizeof(err));
        if (sh.m == nullptr) {
          throw std::runtime_error(std::string("mujoco: ") + err);
        }
        if (count == per && full == nullptr) full = sh.m;
      }
      sh.d = mj_makeData(sh.m);
      sh.first = made;
      sh.envs = count;
      sh.living = count;
      resolve(sh, count);
      shards_.push_back(sh);
      made += count;
    }
    std::error_code ec;
    std::filesystem::remove(generated, ec);
    workers_wanted_ = workers;
    timestep_ = shards_.front().m->opt.timestep;

    host_q_.assign(size_t(envs) * PHYS_NUM_MOTOR, 0.0f);
    host_dq_.assign(size_t(envs) * PHYS_NUM_MOTOR, 0.0f);
    host_gyro_.assign(size_t(envs) * 3, 0.0f);
    host_grav_.assign(size_t(envs) * 3, 0.0f);
    host_quat_.assign(size_t(envs) * 4, 0.0f);
    host_pose_.assign(size_t(envs) * 7, 0.0f);
    host_foot_.assign(size_t(envs) * 2, 0.0f);
    host_speed_.assign(size_t(envs) * 2, 0.0f);
    host_target_.assign(size_t(envs) * PHYS_NUM_MOTOR, 0.0f);
    alive_.assign(size_t(envs), 1);
    energy_.assign(size_t(envs) * PHYS_GROUPS, 0.0f);
    vibration_.assign(size_t(envs) * PHYS_GROUPS, 0.0f);
    alpha_prev_.assign(size_t(envs) * PHYS_NUM_MOTOR, 0.0f);
    dq_prev_.assign(size_t(envs) * PHYS_NUM_MOTOR, 0.0f);
    punches_.assign(size_t(envs), PhysicsPunch{});

    const size_t n = size_t(envs);
    off_q_ = 0;
    off_dq_ = off_q_ + n * PHYS_NUM_MOTOR;
    off_gyro_ = off_dq_ + n * PHYS_NUM_MOTOR;
    off_grav_ = off_gyro_ + n * 3;
    off_quat_ = off_grav_ + n * 3;
    staging_.assign(off_quat_ + n * 4, 0.0f);
    device_alloc(&d_state_, staging_.size());
    device_alloc(&d_target_, n * PHYS_NUM_MOTOR);
  }

  ~MujocoPhysics() override {
    stop_workers();
    for (Shard& s : shards_) {
      if (s.d != nullptr) mj_deleteData(s.d);
      if (s.m != nullptr) mj_deleteModel(s.m);
    }
    for (float* p : {d_state_, d_target_}) {
      if (p != nullptr) cudaFree(p);
    }
  }

  int envs() const override { return envs_; }
  double timestep() const override { return timestep_; }

  void set_gains(
      const float* kp,
      const float* kd
  ) override {
    std::memcpy(kp_, kp, sizeof(kp_));
    std::memcpy(kd_, kd, sizeof(kd_));
  }

  void reset(
      const double* stance,
      double height
  ) override {
    for (Shard& s : shards_) {
      mj_resetData(s.m, s.d);
      for (int e = 0; e < s.envs; ++e) {
        for (int j = 0; j < PHYS_NUM_MOTOR; ++j) {
          s.d->qpos[s.qadr[size_t(e) * PHYS_NUM_MOTOR + size_t(j)]] = stance[j];
        }
        s.d->qpos[s.base_qpos[size_t(e)] + 2] = height;
      }
      mj_forward(s.m, s.d);
    }
    std::fill(alive_.begin(), alive_.end(), 1);
    for (Shard& s : shards_) s.living = s.envs;
    read();
  }

  void hold_base(double height) override {
    for (Shard& s : shards_) {
      for (int e = 0; e < s.envs; ++e) {
        const int q = s.base_qpos[size_t(e)];
        const int v = s.base_dof[size_t(e)];
        s.d->qpos[q + 0] = s.home_x[size_t(e)];
        s.d->qpos[q + 1] = 0.0;
        s.d->qpos[q + 2] = height;
        s.d->qpos[q + 3] = 1.0;
        s.d->qpos[q + 4] = 0.0;
        s.d->qpos[q + 5] = 0.0;
        s.d->qpos[q + 6] = 0.0;
        for (int k = 0; k < 6; ++k) s.d->qvel[v + k] = 0.0;
      }
    }
  }

  void read() override {
    for (Shard& s : shards_) {
      for (int e = 0; e < s.envs; ++e) {
        const int row = s.first + e;
        const size_t m = size_t(row) * PHYS_NUM_MOTOR;
        for (int j = 0; j < PHYS_NUM_MOTOR; ++j) {
          host_q_[m + size_t(j)] =
              float(s.d->qpos[s.qadr[size_t(e) * PHYS_NUM_MOTOR + size_t(j)]]);
          host_dq_[m + size_t(j)] =
              float(s.d->qvel[s.dadr[size_t(e) * PHYS_NUM_MOTOR + size_t(j)]]);
        }
        const int pb = s.pelvis_body[size_t(e)];
        const double* xp = s.d->xpos + 3 * pb;

        const double* xq = s.d->sensordata + s.sensor_quat[size_t(e)];
        const double* w = s.d->sensordata + s.sensor_gyro[size_t(e)];
        float* pose = host_pose_.data() + size_t(row) * 7;
        pose[0] = float(xq[1]);
        pose[1] = float(xq[2]);
        pose[2] = float(xq[3]);
        pose[3] = float(xq[0]);
        pose[4] = float(xp[0]);
        pose[5] = float(xp[1]);
        pose[6] = float(xp[2]);
        float* quat = host_quat_.data() + size_t(row) * 4;
        for (int k = 0; k < 4; ++k) quat[k] = float(xq[k]);

        float* gy = host_gyro_.data() + size_t(row) * 3;
        for (int k = 0; k < 3; ++k) gy[k] = float(w[k]);

        const double down[3] = {0.0, 0.0, -1.0};
        double g[3];
        world_to_body(xq, down, g);
        float* gv = host_grav_.data() + size_t(row) * 3;
        for (int k = 0; k < 3; ++k) gv[k] = float(g[k]);

        float* foot = host_foot_.data() + size_t(row) * 2;
        foot[0] = float(s.d->xpos[3 * s.foot_body[size_t(e) * 2 + 0] + 2]);
        foot[1] = float(s.d->xpos[3 * s.foot_body[size_t(e) * 2 + 1] + 2]);

        float* speed = host_speed_.data() + size_t(row) * 2;
        const int of[2] = {s.pelvis_body[size_t(e)], s.torso_body[size_t(e)]};
        for (int k = 0; k < 2; ++k) {
          double vel[6];
          mj_objectVelocity(s.m, s.d, mjOBJ_BODY, of[k], vel, 0);
          speed[k] = float(
              std::sqrt(vel[3] * vel[3] + vel[4] * vel[4] + vel[5] * vel[5])
          );
        }
      }
    }
    std::copy(host_q_.begin(), host_q_.end(), staging_.begin() + long(off_q_));
    std::copy(
        host_dq_.begin(),
        host_dq_.end(),
        staging_.begin() + long(off_dq_)
    );
    std::copy(
        host_gyro_.begin(),
        host_gyro_.end(),
        staging_.begin() + long(off_gyro_)
    );
    std::copy(
        host_grav_.begin(),
        host_grav_.end(),
        staging_.begin() + long(off_grav_)
    );
    std::copy(
        host_quat_.begin(),
        host_quat_.end(),
        staging_.begin() + long(off_quat_)
    );
    cudaMemcpy(
        d_state_,
        staging_.data(),
        staging_.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );
  }

  const float* motor_q() const override { return d_state_ + off_q_; }
  const float* motor_dq() const override { return d_state_ + off_dq_; }
  const float* gyro() const override { return d_state_ + off_gyro_; }
  const float* gravity() const override { return d_state_ + off_grav_; }
  const float* base_quat() const override { return d_state_ + off_quat_; }
  float* q_target() override { return d_target_; }

  const float* base_pose() const override { return host_pose_.data(); }
  const float* foot_height() const override { return host_foot_.data(); }
  const float* body_speed() const override { return host_speed_.data(); }

  void apply_targets() override {
    cudaMemcpy(
        host_target_.data(),
        d_target_,
        host_target_.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    );
  }

  void servo(Shard& s) {
    for (int e = 0; e < s.envs; ++e) {
      const int row = s.first + e;
      if (alive_[size_t(row)] == 0) {
        for (int j = 0; j < PHYS_NUM_MOTOR; ++j) {
          s.d->ctrl[s.act[size_t(e) * PHYS_NUM_MOTOR + size_t(j)]] = 0.0;
        }
        continue;
      }
      const size_t m = size_t(row) * PHYS_NUM_MOTOR;
      for (int j = 0; j < PHYS_NUM_MOTOR; ++j) {
        const size_t k = size_t(e) * PHYS_NUM_MOTOR + size_t(j);
        const double q = s.d->qpos[s.qadr[k]];
        const double dq = s.d->qvel[s.dadr[k]];
        const double want = host_target_[m + size_t(j)];
        const double tau = kp_[j] * (want - q) - kd_[j] * dq;
        s.d->ctrl[s.act[k]] = std::min(std::max(tau, s.frc_lo[k]), s.frc_hi[k]);
      }
    }
  }

  void set_punches(const PhysicsPunch* punches) override {
    std::copy(punches, punches + envs_, punches_.begin());
  }

  void step(int substeps) override {
    if (substeps < 1) return;
    if (shards_.size() == 1) {
      for (int k = 0; k < substeps; ++k) advance(shards_[0]);
      return;
    }
    start_workers();
    {
      std::unique_lock<std::mutex> lock(mu_);
      substeps_ = substeps;
      next_.store(0, std::memory_order_relaxed);
      pending_ = int(workers_.size());
      ++generation_;
      cv_start_.notify_all();
      cv_done_.wait(lock, [this] { return pending_ == 0; });
    }
  }

  void kill(int env) override {
    if (alive_[size_t(env)] == 0) return;
    alive_[size_t(env)] = 0;
    for (Shard& s : shards_) {
      if (env >= s.first && env < s.first + s.envs) {
        --s.living;
        break;
      }
    }
  }
  const unsigned char* alive() const override { return alive_.data(); }

  void meter(double dt) override {
    for (Shard& s : shards_) {
      for (int e = 0; e < s.envs; ++e) {
        const int row = s.first + e;
        if (alive_[size_t(row)] == 0) continue;
        for (int j = 0; j < PHYS_NUM_MOTOR; ++j) {
          const size_t k = size_t(e) * PHYS_NUM_MOTOR + size_t(j);
          const size_t m = size_t(row) * PHYS_NUM_MOTOR + size_t(j);
          const size_t g = size_t(row) * PHYS_GROUPS + size_t(MJ_GROUP[j]);
          const double dq = s.d->qvel[s.dadr[k]];
          const double tau = s.d->ctrl[s.act[k]];
          energy_[g] += float(std::fabs(tau * dq) * dt);
          const double alpha = (dq - dq_prev_[m]) / dt;
          vibration_[g] += float(std::fabs(alpha - alpha_prev_[m]));
          alpha_prev_[m] = float(alpha);
          dq_prev_[m] = float(dq);
        }
      }
    }
  }

  const float* energy() const override { return energy_.data(); }
  const float* vibration() const override { return vibration_.data(); }

 private:
  void advance(Shard& s) {
    if (s.living == 0) return;
    servo(s);
    apply_punch_forces(s);
    mj_step(s.m, s.d);
  }

  void start_workers() {
    if (!workers_.empty()) return;
    workers_.reserve(size_t(workers_wanted_));
    for (int w = 0; w < workers_wanted_; ++w) {
      workers_.emplace_back([this] {
        int seen = 0;
        for (;;) {
          {
            std::unique_lock<std::mutex> lock(mu_);
            cv_start_.wait(lock, [this, &seen] {
              return stop_ || generation_ != seen;
            });
            if (stop_) return;
            seen = generation_;
          }
          for (;;) {
            const size_t i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= shards_.size()) break;
            for (int k = 0; k < substeps_; ++k) advance(shards_[i]);
          }
          {
            const std::lock_guard<std::mutex> lock(mu_);
            if (--pending_ == 0) cv_done_.notify_one();
          }
        }
      });
    }
  }

  void stop_workers() {
    if (workers_.empty()) return;
    {
      const std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
      cv_start_.notify_all();
    }
    for (std::thread& t : workers_) t.join();
    workers_.clear();
  }

  static void device_alloc(
      float** p,
      size_t n
  ) {
    cudaMalloc(p, n * sizeof(float));
    cudaMemset(*p, 0, n * sizeof(float));
  }
  static void upload(
      const std::vector<float>& host,
      float* device
  ) {
    cudaMemcpy(
        device,
        host.data(),
        host.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );
  }
  static void world_to_body(
      const double* q,
      const double* v,
      double* out
  ) {
    const double w = q[0], x = -q[1], y = -q[2], z = -q[3];
    const double tx = 2.0 * (y * v[2] - z * v[1]);
    const double ty = 2.0 * (z * v[0] - x * v[2]);
    const double tz = 2.0 * (x * v[1] - y * v[0]);
    out[0] = v[0] + w * tx + (y * tz - z * ty);
    out[1] = v[1] + w * ty + (z * tx - x * tz);
    out[2] = v[2] + w * tz + (x * ty - y * tx);
  }

  void apply_punch_forces(Shard& s) {
    for (int e = 0; e < s.envs; ++e) {
      const int row = s.first + e;
      const PhysicsPunch& p = punches_[size_t(row)];
      const int prev = s.punched[size_t(e)];
      if (prev >= 0) {
        std::fill(
            s.d->xfrc_applied + 6 * prev,
            s.d->xfrc_applied + 6 * prev + 6,
            0.0
        );
        s.punched[size_t(e)] = -1;
      }
      if (p.joint < 0 || alive_[size_t(row)] == 0) continue;
      const int body = s.jnt_body[size_t(e) * PHYS_NUM_MOTOR + size_t(p.joint)];
      const double* anchor = s.jnt_anchor.data() +
                             (size_t(e) * PHYS_NUM_MOTOR + size_t(p.joint)) * 3;
      double world[3];
      const double* xq = s.d->xquat + 4 * body;
      const double* xp = s.d->xpos + 3 * body;
      body_to_world(xq, anchor, world);
      for (int k = 0; k < 3; ++k) world[k] += xp[k];
      const double* com = s.d->xipos + 3 * body;
      const double r[3] = {
          world[0] - com[0],
          world[1] - com[1],
          world[2] - com[2]
      };
      const double f[3] = {p.force[0], p.force[1], p.force[2]};
      double* dst = s.d->xfrc_applied + 6 * body;
      dst[0] = f[0];
      dst[1] = f[1];
      dst[2] = f[2];
      dst[3] = r[1] * f[2] - r[2] * f[1];
      dst[4] = r[2] * f[0] - r[0] * f[2];
      dst[5] = r[0] * f[1] - r[1] * f[0];
      s.punched[size_t(e)] = body;
    }
  }

  static void body_to_world(
      const double* q,
      const double* v,
      double* out
  ) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double tx = 2.0 * (y * v[2] - z * v[1]);
    const double ty = 2.0 * (z * v[0] - x * v[2]);
    const double tz = 2.0 * (x * v[1] - y * v[0]);
    out[0] = v[0] + w * tx + (y * tz - z * ty);
    out[1] = v[1] + w * ty + (z * tx - x * tz);
    out[2] = v[2] + w * tz + (x * ty - y * tx);
  }

  void resolve(
      Shard& s,
      int count
  ) {
    s.qadr.assign(size_t(count) * PHYS_NUM_MOTOR, 0);
    s.dadr.assign(size_t(count) * PHYS_NUM_MOTOR, 0);
    s.act.assign(size_t(count) * PHYS_NUM_MOTOR, 0);
    s.frc_lo.assign(size_t(count) * PHYS_NUM_MOTOR, 0.0);
    s.frc_hi.assign(size_t(count) * PHYS_NUM_MOTOR, 0.0);
    s.jnt_body.assign(size_t(count) * PHYS_NUM_MOTOR, 0);
    s.jnt_anchor.assign(size_t(count) * PHYS_NUM_MOTOR * 3, 0.0);
    s.base_qpos.assign(size_t(count), 0);
    s.base_dof.assign(size_t(count), 0);
    s.pelvis_body.assign(size_t(count), 0);
    s.torso_body.assign(size_t(count), 0);
    s.sensor_quat.assign(size_t(count), 0);
    s.sensor_gyro.assign(size_t(count), 0);
    s.foot_body.assign(size_t(count) * 2, 0);
    s.home_x.assign(size_t(count), 0.0);
    s.punched.assign(size_t(count), -1);

    for (int e = 0; e < count; ++e) {
      const std::string sfx = suffix_of(e, count);
      const int pelvis = need(s.m, mjOBJ_BODY, "pelvis" + sfx);
      s.pelvis_body[size_t(e)] = pelvis;
      s.torso_body[size_t(e)] = need(s.m, mjOBJ_BODY, "torso_link" + sfx);
      const int freejnt = s.m->body_jntadr[pelvis];
      s.base_qpos[size_t(e)] = s.m->jnt_qposadr[freejnt];
      s.base_dof[size_t(e)] = s.m->jnt_dofadr[freejnt];
      s.home_x[size_t(e)] = s.m->qpos0[s.base_qpos[size_t(e)]];
      s.sensor_quat[size_t(e)] =
          s.m->sensor_adr[need(s.m, mjOBJ_SENSOR, "imu_quat" + sfx)];
      s.sensor_gyro[size_t(e)] =
          s.m->sensor_adr[need(s.m, mjOBJ_SENSOR, "imu_gyro" + sfx)];
      s.foot_body[size_t(e) * 2 + 0] =
          need(s.m, mjOBJ_BODY, "left_ankle_roll_link" + sfx);
      s.foot_body[size_t(e) * 2 + 1] =
          need(s.m, mjOBJ_BODY, "right_ankle_roll_link" + sfx);
      for (int j = 0; j < PHYS_NUM_MOTOR; ++j) {
        const size_t k = size_t(e) * PHYS_NUM_MOTOR + size_t(j);
        const int jid = need(s.m, mjOBJ_JOINT, MJ_JOINTS[j] + sfx);
        s.qadr[k] = s.m->jnt_qposadr[jid];
        s.dadr[k] = s.m->jnt_dofadr[jid];
        s.act[k] = need(s.m, mjOBJ_ACTUATOR, MJ_JOINTS[j] + sfx);
        if (s.m->actuator_forcelimited[s.act[k]] != 0) {
          s.frc_lo[k] = s.m->actuator_forcerange[2 * s.act[k] + 0];
          s.frc_hi[k] = s.m->actuator_forcerange[2 * s.act[k] + 1];
        } else {
          s.frc_lo[k] = -std::numeric_limits<double>::infinity();
          s.frc_hi[k] = std::numeric_limits<double>::infinity();
        }
        s.jnt_body[k] = s.m->jnt_bodyid[jid];
        for (int c = 0; c < 3; ++c) {
          s.jnt_anchor[k * 3 + size_t(c)] = s.m->jnt_pos[3 * jid + c];
        }
      }
    }
  }

  static int need(
      const mjModel* m,
      mjtObj type,
      const std::string& name
  ) {
    const int id = mj_name2id(m, type, name.c_str());
    if (id < 0) throw std::runtime_error("mujoco: no such name " + name);
    return id;
  }

  int envs_ = 0;
  double timestep_ = 0.002;
  float kp_[PHYS_NUM_MOTOR] = {};
  float kd_[PHYS_NUM_MOTOR] = {};
  std::vector<Shard> shards_;
  std::vector<float> host_q_, host_dq_, host_gyro_, host_grav_, host_quat_;
  std::vector<float> host_pose_, host_foot_, host_target_, host_speed_;
  std::vector<float> energy_, vibration_, alpha_prev_, dq_prev_;
  std::vector<unsigned char> alive_;
  std::vector<PhysicsPunch> punches_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_start_;
  std::condition_variable cv_done_;
  int generation_ = 0;
  int pending_ = 0;
  int substeps_ = 1;
  int workers_wanted_ = 1;
  std::atomic<size_t> next_{0};
  bool stop_ = false;
  std::vector<float> staging_;
  size_t off_q_ = 0, off_dq_ = 0, off_gyro_ = 0, off_grav_ = 0, off_quat_ = 0;
  float *d_state_ = nullptr, *d_target_ = nullptr;
};

}

std::unique_ptr<Physics> physics_make_mujoco(
    const mjcf::Model& model,
    const std::string& model_path,
    int envs,
    double spacing,
    int threads
) {
  return std::make_unique<MujocoPhysics>(
      model,
      model_path,
      envs,
      spacing,
      threads
  );
}
