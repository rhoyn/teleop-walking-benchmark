
#include "physics.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char MJWARP_MAGIC[8] = {'M', 'J', 'W', 'A', 'R', 'P', '0', '1'};

void ck(
    cudaError_t e,
    const char* what
) {
  if (e != cudaSuccess) {
    throw std::runtime_error(
        std::string("mjwarp: ") + what + ": " + cudaGetErrorString(e)
    );
  }
}

struct Warp {
  void* lib = nullptr;
  int (*init)(const char*) = nullptr;
  void* (*primary_context)(int) = nullptr;
  void* (*load_graph)(
      void*,
      const char*,
      int
  ) = nullptr;
  void (*destroy_graph)(void*) = nullptr;
  void* (*get_param_ptr)(
      void*,
      const char*
  ) = nullptr;
  size_t (*get_param_size)(
      void*,
      const char*
  ) = nullptr;
  void* (*get_cuda_graph)(void*) = nullptr;
  void* (*get_cuda_graph_exec)(void*) = nullptr;
  const char* (*error_string)() = nullptr;

  template <typename F>
  void bind(
      F& fn,
      const char* name
  ) {
    fn = reinterpret_cast<F>(dlsym(lib, name));
    if (fn == nullptr) {
      throw std::runtime_error(
          std::string("mjwarp: warp.so has no symbol ") + name
      );
    }
  }

  void open(const std::string& path) {
    lib = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (lib == nullptr) {
      throw std::runtime_error(
          "mjwarp: cannot load " + path + ": " + dlerror() +
          "\n  set MJWARP_LIB to the warp.so that recorded the capture"
      );
    }
    bind(init, "wp_init");
    bind(primary_context, "wp_cuda_device_get_primary_context");
    bind(load_graph, "wp_apic_load_graph");
    bind(destroy_graph, "wp_apic_destroy_graph");
    bind(get_param_ptr, "wp_apic_get_param_ptr");
    bind(get_param_size, "wp_apic_get_param_size");
    bind(get_cuda_graph, "wp_apic_get_cuda_graph");
    bind(get_cuda_graph_exec, "wp_apic_get_cuda_graph_exec");
    bind(error_string, "wp_get_error_string");
  }
};

std::string warp_lib_path(const std::string& graph_path) {
  if (const char* env = std::getenv("MJWARP_LIB")) return env;
  const size_t slash = graph_path.find_last_of('/');
  const std::string dir = slash == std::string::npos
                              ? std::string(".")
                              : graph_path.substr(0, slash);
  return dir + "/warp.so";
}

struct Sidecar {
  std::string warp_version;
  int nworld = 0, nq = 0, nv = 0, nu = 0, motors = 0, groups = 0;
  int base_q = 0, base_v = 0;
  float timestep = 0.002f;
  std::vector<int> qadr, dadr, act, group;
  std::vector<float> qpos0;

  explicit Sidecar(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("mjwarp: cannot read " + path);

    char magic[8];
    in.read(magic, 8);
    if (std::memcmp(magic, MJWARP_MAGIC, 8) != 0) {
      throw std::runtime_error("mjwarp: " + path + " is not a capture sidecar");
    }
    char version[32];
    in.read(version, 32);
    warp_version.assign(version, strnlen(version, 32));

    int32_t head[8];
    in.read(reinterpret_cast<char*>(head), sizeof(head));
    nworld = head[0];
    nq = head[1];
    nv = head[2];
    nu = head[3];
    motors = head[4];
    groups = head[5];
    base_q = head[6];
    base_v = head[7];
    in.read(reinterpret_cast<char*>(&timestep), sizeof(timestep));

    if (motors != PHYS_NUM_MOTOR || groups != PHYS_GROUPS) {
      throw std::runtime_error("mjwarp: capture disagrees with physics.h");
    }
    auto ints = [&](std::vector<int>& v) {
      v.resize(size_t(motors));
      in.read(reinterpret_cast<char*>(v.data()), long(v.size() * sizeof(int)));
    };
    ints(qadr);
    ints(dadr);
    ints(act);
    ints(group);
    qpos0.resize(size_t(nq));
    in.read(
        reinterpret_cast<char*>(qpos0.data()),
        long(qpos0.size() * sizeof(float))
    );
    if (!in) throw std::runtime_error("mjwarp: " + path + " is truncated");
  }
};

__global__ void k_hold(
    float* qpos,
    float* qvel,
    int nworld,
    int nq,
    int nv,
    int base_q,
    int base_v,
    float height
) {
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nworld) return;
  float* q = qpos + size_t(w) * nq + base_q;
  float* v = qvel + size_t(w) * nv + base_v;
  q[0] = 0.0f;
  q[1] = 0.0f;
  q[2] = height;
  q[3] = 1.0f;
  q[4] = 0.0f;
  q[5] = 0.0f;
  q[6] = 0.0f;
  for (int k = 0; k < 6; ++k) v[k] = 0.0f;
}

__global__ void k_reset(
    float* qpos,
    float* qvel,
    float* warmstart,
    const float* qpos0,
    const float* stance,
    const int* qadr,
    int nworld,
    int nq,
    int nv,
    int motors,
    int base_q,
    float height
) {
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nworld) return;
  float* q = qpos + size_t(w) * nq;
  for (int i = 0; i < nq; ++i) q[i] = qpos0[i];
  for (int j = 0; j < motors; ++j) q[qadr[j]] = stance[j];
  q[base_q + 2] = height;
  float* v = qvel + size_t(w) * nv;
  for (int i = 0; i < nv; ++i) {
    v[i] = 0.0f;
    warmstart[size_t(w) * nv + i] = 0.0f;
  }
}

__global__ void k_base_lin_vel(
    const float* __restrict__ qvel,
    const float* __restrict__ quat,
    float* __restrict__ lin_vel,
    int nworld,
    int nv,
    int base_v
) {
  const int w = blockIdx.x * blockDim.x + threadIdx.x;
  if (w >= nworld) return;

  const float* v = qvel + size_t(w) * nv + base_v;
  const float* q = quat + size_t(w) * 4;
  const float qw = q[0], x = -q[1], y = -q[2], z = -q[3];
  const float tx = 2.0f * (y * v[2] - z * v[1]);
  const float ty = 2.0f * (z * v[0] - x * v[2]);
  const float tz = 2.0f * (x * v[1] - y * v[0]);
  float* o = lin_vel + size_t(w) * 3;
  o[0] = v[0] + qw * tx + (y * tz - z * ty);
  o[1] = v[1] + qw * ty + (z * tx - x * tz);
  o[2] = v[2] + qw * tz + (x * ty - y * tx);
}

__global__ void k_obs_joint(
    const float* qpos,
    const float* qvel,
    const int* qadr,
    const int* dadr,
    float* obs_q,
    float* obs_dq,
    int nworld,
    int nq,
    int nv,
    int motors
) {
  const int w = blockIdx.x;
  const int j = threadIdx.x;
  if (w >= nworld || j >= motors) return;
  const size_t m = size_t(w) * motors + j;
  obs_q[m] = qpos[size_t(w) * nq + qadr[j]];
  obs_dq[m] = qvel[size_t(w) * nv + dadr[j]];
}

__global__ void k_meter(
    const float* qvel,
    const float* ctrl,
    const int* dadr,
    const int* act,
    const int* group,
    const int* alive,
    float* dq_prev,
    float* alpha_prev,
    float* energy,
    float* vibration,
    float dt,
    int nworld,
    int nv,
    int nu,
    int motors,
    int groups
) {
  const int w = blockIdx.x;
  const int j = threadIdx.x;
  if (w >= nworld || j >= motors) return;
  if (alive[w] == 0) return;

  const size_t m = size_t(w) * motors + j;
  const float dq = qvel[size_t(w) * nv + dadr[j]];
  const float tau = ctrl[size_t(w) * nu + act[j]];
  const float alpha = (dq - dq_prev[m]) / dt;
  const size_t g = size_t(w) * groups + group[j];

  atomicAdd(energy + g, fabsf(tau * dq) * dt);
  atomicAdd(vibration + g, fabsf(alpha - alpha_prev[m]));
  alpha_prev[m] = alpha;
  dq_prev[m] = dq;
}

class MjWarpPhysics : public Physics {
 public:
  MjWarpPhysics(
      const std::string& graph_path,
      int envs
  )
      : side_(graph_path + ".bin") {
    if (envs > side_.nworld) {
      throw std::runtime_error(
          "mjwarp: capture holds " + std::to_string(side_.nworld) +
          " worlds but " + std::to_string(envs) +
          " envs were asked for; "
          "re-run mjwarp/capture.py --nworld " +
          std::to_string(envs)
      );
    }
    envs_ = envs;

    warp_.open(warp_lib_path(graph_path));
    if (warp_.init(side_.warp_version.c_str()) != 0) {
      throw std::runtime_error(
          std::string("mjwarp: wp_init failed: ") + warp_.error_string()
      );
    }

    void* ctx = warp_.primary_context(0);
    graph_ = warp_.load_graph(ctx, (graph_path + ".wrp").c_str(), 0);
    if (graph_ == nullptr) {
      throw std::runtime_error(
          "mjwarp: cannot load " + graph_path + ".wrp: " + warp_.error_string()
      );
    }

    warp_.get_cuda_graph(graph_);
    exec_ = static_cast<cudaGraphExec_t>(warp_.get_cuda_graph_exec(graph_));
    if (exec_ == nullptr) {
      throw std::runtime_error(
          std::string("mjwarp: no CUDA graph in capture: ") +
          warp_.error_string()
      );
    }

    p_qtarget_ =
        param<float>("q_target", size_t(side_.nworld) * PHYS_NUM_MOTOR);
    p_kp_ = param<float>("kp", PHYS_NUM_MOTOR);
    p_kd_ = param<float>("kd", PHYS_NUM_MOTOR);
    p_alive_ = param<int>("alive", size_t(side_.nworld));
    p_pjoint_ = param<int>("punch_joint", size_t(side_.nworld));
    p_pframe_ = param<int>("punch_frame", size_t(side_.nworld));
    p_pforce_ = param<float>("punch_force", size_t(side_.nworld) * 3);
    p_q_ = param<float>("obs_q", size_t(side_.nworld) * PHYS_NUM_MOTOR);
    p_dq_ = param<float>("obs_dq", size_t(side_.nworld) * PHYS_NUM_MOTOR);
    p_gyro_ = param<float>("obs_gyro", size_t(side_.nworld) * 3);
    p_grav_ = param<float>("obs_grav", size_t(side_.nworld) * 3);
    p_quat_ = param<float>("obs_quat", size_t(side_.nworld) * 4);
    p_pose_ = param<float>("obs_pose", size_t(side_.nworld) * 7);
    p_foot_ = param<float>("obs_foot", size_t(side_.nworld) * 2);
    p_speed_ = param<float>("obs_speed", size_t(side_.nworld) * 2);
    p_qpos_ = param<float>("qpos", size_t(side_.nworld) * side_.nq);
    p_qvel_ = param<float>("qvel", size_t(side_.nworld) * side_.nv);
    p_ctrl_ = param<float>("ctrl", size_t(side_.nworld) * side_.nu);
    p_warm_ = param<float>("qacc_warmstart", size_t(side_.nworld) * side_.nv);

    const size_t n = size_t(envs_);
    host_pose_.assign(n * 7, 0.0f);
    host_foot_.assign(n * 2, 0.0f);
    host_speed_.assign(n * 2, 0.0f);
    host_energy_.assign(n * PHYS_GROUPS, 0.0f);
    host_vibration_.assign(n * PHYS_GROUPS, 0.0f);
    alive_.assign(n, 1);
    host_pjoint_.assign(size_t(side_.nworld), -1);
    host_pframe_.assign(size_t(side_.nworld), 0);
    host_pforce_.assign(size_t(side_.nworld) * 3, 0.0f);
    host_alive_.assign(size_t(side_.nworld), 1);

    own(&d_qadr_, side_.qadr);
    own(&d_dadr_, side_.dadr);
    own(&d_act_, side_.act);
    own(&d_group_, side_.group);
    own(&d_qpos0_, side_.qpos0);
    zeros(&d_stance_, PHYS_NUM_MOTOR);
    zeros(&d_energy_, size_t(side_.nworld) * PHYS_GROUPS);
    zeros(&d_vibration_, size_t(side_.nworld) * PHYS_GROUPS);
    zeros(&d_linvel_, size_t(side_.nworld) * 3);
    zeros(&d_dqprev_, size_t(side_.nworld) * PHYS_NUM_MOTOR);
    zeros(&d_alphaprev_, size_t(side_.nworld) * PHYS_NUM_MOTOR);

    std::fprintf(
        stderr,
        "mjwarp: %s, %d of %d worlds, dt %g, warp %s\n",
        graph_path.c_str(),
        envs_,
        side_.nworld,
        side_.timestep,
        side_.warp_version.c_str()
    );
  }

  ~MjWarpPhysics() override {
    for (void* p :
         {(void*)d_qadr_,
          (void*)d_dadr_,
          (void*)d_act_,
          (void*)d_group_,
          (void*)d_qpos0_,
          (void*)d_stance_,
          (void*)d_energy_,
          (void*)d_vibration_,
          (void*)d_dqprev_,
          (void*)d_alphaprev_,
          (void*)d_linvel_}) {
      if (p != nullptr) cudaFree(p);
    }
    if (graph_ != nullptr) warp_.destroy_graph(graph_);
    if (warp_.lib != nullptr) dlclose(warp_.lib);
  }

  int envs() const override { return envs_; }
  double timestep() const override { return side_.timestep; }

  void set_gains(
      const float* kp,
      const float* kd
  ) override {
    up(p_kp_, kp, PHYS_NUM_MOTOR);
    up(p_kd_, kd, PHYS_NUM_MOTOR);
  }

  void reset(
      const double* stance,
      double height
  ) override {
    float row[PHYS_NUM_MOTOR];
    for (int j = 0; j < PHYS_NUM_MOTOR; ++j) row[j] = float(stance[j]);
    up(d_stance_, row, PHYS_NUM_MOTOR);

    const int blocks = (side_.nworld + 255) / 256;
    k_reset<<<blocks, 256>>>(
        p_qpos_,
        p_qvel_,
        p_warm_,
        d_qpos0_,
        d_stance_,
        d_qadr_,
        side_.nworld,
        side_.nq,
        side_.nv,
        PHYS_NUM_MOTOR,
        side_.base_q,
        float(height)
    );
    ck(cudaGetLastError(), "reset");

    std::fill(alive_.begin(), alive_.end(), 1);
    std::fill(host_alive_.begin(), host_alive_.end(), 1);
    up(p_alive_, host_alive_.data(), host_alive_.size());
    std::fill(host_pjoint_.begin(), host_pjoint_.end(), -1);
    up(p_pjoint_, host_pjoint_.data(), host_pjoint_.size());
    std::fill(host_pframe_.begin(), host_pframe_.end(), 0);
    up(p_pframe_, host_pframe_.data(), host_pframe_.size());
    ck(cudaMemset(p_pforce_, 0, host_pforce_.size() * sizeof(float)), "punch");
    ck(cudaMemset(
           d_energy_,
           0,
           size_t(side_.nworld) * PHYS_GROUPS * sizeof(float)
       ),
       "energy");
    ck(cudaMemset(
           d_vibration_,
           0,
           size_t(side_.nworld) * PHYS_GROUPS * sizeof(float)
       ),
       "vibration");
    ck(cudaMemset(
           d_dqprev_,
           0,
           size_t(side_.nworld) * PHYS_NUM_MOTOR * sizeof(float)
       ),
       "dq_prev");
    ck(cudaMemset(
           d_alphaprev_,
           0,
           size_t(side_.nworld) * PHYS_NUM_MOTOR * sizeof(float)
       ),
       "alpha_prev");

    hold_base(height);
    step(1);
    k_reset<<<blocks, 256>>>(
        p_qpos_,
        p_qvel_,
        p_warm_,
        d_qpos0_,
        d_stance_,
        d_qadr_,
        side_.nworld,
        side_.nq,
        side_.nv,
        PHYS_NUM_MOTOR,
        side_.base_q,
        float(height)
    );
    ck(cudaGetLastError(), "reset restore");
    k_obs_joint<<<side_.nworld, PHYS_NUM_MOTOR>>>(
        p_qpos_,
        p_qvel_,
        d_qadr_,
        d_dadr_,
        p_q_,
        p_dq_,
        side_.nworld,
        side_.nq,
        side_.nv,
        PHYS_NUM_MOTOR
    );
    ck(cudaGetLastError(), "reset obs");
    read();
  }

  void hold_base(double height) override {
    const int blocks = (side_.nworld + 255) / 256;
    k_hold<<<blocks, 256>>>(
        p_qpos_,
        p_qvel_,
        side_.nworld,
        side_.nq,
        side_.nv,
        side_.base_q,
        side_.base_v,
        float(height)
    );
    ck(cudaGetLastError(), "hold_base");
  }

  void read() override {
    const size_t n = size_t(envs_);
    const int blocks = (side_.nworld + 255) / 256;
    k_base_lin_vel<<<blocks, 256>>>(
        p_qvel_,
        p_quat_,
        d_linvel_,
        side_.nworld,
        side_.nv,
        side_.base_v
    );
    ck(cudaGetLastError(), "base_lin_vel");
    down(host_pose_.data(), p_pose_, n * 7);
    down(host_foot_.data(), p_foot_, n * 2);
    down(host_speed_.data(), p_speed_, n * 2);
    down(host_energy_.data(), d_energy_, n * PHYS_GROUPS);
    down(host_vibration_.data(), d_vibration_, n * PHYS_GROUPS);
    ck(cudaStreamSynchronize(nullptr), "read");
  }

  const float* motor_q() const override { return p_q_; }
  const float* motor_dq() const override { return p_dq_; }
  const float* gyro() const override { return p_gyro_; }
  const float* base_lin_vel() const override { return d_linvel_; }
  const float* gravity() const override { return p_grav_; }
  const float* base_quat() const override { return p_quat_; }
  float* q_target() override { return p_qtarget_; }

  const float* base_pose() const override { return host_pose_.data(); }
  const float* foot_height() const override { return host_foot_.data(); }
  const float* body_speed() const override { return host_speed_.data(); }

  void apply_targets() override {}

  void set_punches(const PhysicsPunch* punches) override {
    for (int e = 0; e < envs_; ++e) {
      host_pjoint_[size_t(e)] = punches[e].joint;
      host_pframe_[size_t(e)] = punches[e].frame_child;
      for (int k = 0; k < 3; ++k) {
        host_pforce_[size_t(e) * 3 + size_t(k)] = punches[e].force[k];
      }
    }
    up(p_pjoint_, host_pjoint_.data(), size_t(envs_));
    up(p_pframe_, host_pframe_.data(), size_t(envs_));
    up(p_pforce_, host_pforce_.data(), size_t(envs_) * 3);
  }

  void step(int substeps) override {
    if (substeps < 1) return;
    if (alive_dirty_) {
      up(p_alive_, host_alive_.data(), size_t(envs_));
      alive_dirty_ = false;
    }
    for (int k = 0; k < substeps; ++k) {
      ck(cudaGraphLaunch(exec_, nullptr), "graph launch");
    }
  }

  void kill(int env) override {
    if (alive_[size_t(env)] == 0) return;
    alive_[size_t(env)] = 0;
    host_alive_[size_t(env)] = 0;
    alive_dirty_ = true;
  }
  const unsigned char* alive() const override { return alive_.data(); }

  void meter(double dt) override {
    k_meter<<<side_.nworld, PHYS_NUM_MOTOR>>>(
        p_qvel_,
        p_ctrl_,
        d_dadr_,
        d_act_,
        d_group_,
        p_alive_,
        d_dqprev_,
        d_alphaprev_,
        d_energy_,
        d_vibration_,
        float(dt),
        side_.nworld,
        side_.nv,
        side_.nu,
        PHYS_NUM_MOTOR,
        PHYS_GROUPS
    );
    ck(cudaGetLastError(), "meter");
  }

  const float* energy() const override { return host_energy_.data(); }
  const float* vibration() const override { return host_vibration_.data(); }

 private:
  template <typename T>
  T* param(
      const char* name,
      size_t want
  ) {
    void* p = warp_.get_param_ptr(graph_, name);
    if (p == nullptr) {
      throw std::runtime_error(
          std::string("mjwarp: capture has no '") + name + "'"
      );
    }
    const size_t got = warp_.get_param_size(graph_, name);
    if (got != want * sizeof(T)) {
      throw std::runtime_error(
          std::string("mjwarp: '") + name + "' is " + std::to_string(got) +
          " bytes, expected " + std::to_string(want * sizeof(T))
      );
    }
    return static_cast<T*>(p);
  }

  template <typename T>
  static void up(
      T* dst,
      const T* src,
      size_t n
  ) {
    ck(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice), "upload");
  }
  template <typename T>
  static void down(
      T* dst,
      const T* src,
      size_t n
  ) {
    ck(cudaMemcpyAsync(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost),
       "download");
  }
  template <typename T>
  static void own(
      T** dst,
      const std::vector<T>& src
  ) {
    ck(cudaMalloc(dst, src.size() * sizeof(T)), "alloc");
    up(*dst, src.data(), src.size());
  }
  static void zeros(
      float** dst,
      size_t n
  ) {
    ck(cudaMalloc(dst, n * sizeof(float)), "alloc");
    ck(cudaMemset(*dst, 0, n * sizeof(float)), "zero");
  }

  Sidecar side_;
  Warp warp_;
  void* graph_ = nullptr;
  cudaGraphExec_t exec_ = nullptr;
  int envs_ = 0;
  bool alive_dirty_ = false;

  float *p_qtarget_, *p_kp_, *p_kd_, *p_pforce_;
  float *p_q_, *p_dq_, *p_gyro_, *p_grav_, *p_quat_;
  float *p_pose_, *p_foot_, *p_speed_;
  float *p_qpos_, *p_qvel_, *p_ctrl_, *p_warm_;
  int *p_alive_, *p_pjoint_, *p_pframe_;

  int *d_qadr_ = nullptr, *d_dadr_ = nullptr, *d_act_ = nullptr,
      *d_group_ = nullptr;
  float *d_qpos0_ = nullptr, *d_stance_ = nullptr;
  float *d_energy_ = nullptr, *d_vibration_ = nullptr;
  float *d_dqprev_ = nullptr, *d_alphaprev_ = nullptr;
  float* d_linvel_ = nullptr;

  std::vector<float> host_pose_, host_foot_, host_speed_;
  std::vector<float> host_energy_, host_vibration_, host_pforce_;
  std::vector<unsigned char> alive_;
  std::vector<int> host_pjoint_, host_pframe_, host_alive_;
};

}  // namespace

std::unique_ptr<Physics> physics_make_mjwarp(
    const std::string& graph_path,
    int envs
) {
  return std::make_unique<MjWarpPhysics>(graph_path, envs);
}
