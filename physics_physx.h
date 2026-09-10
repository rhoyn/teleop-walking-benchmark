#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include <PxPhysicsAPI.h>

#include "model.h"

namespace robot {

inline constexpr int NUM_MOTOR = 29;

struct Robot {
  physx::PxArticulationReducedCoordinate* art = nullptr;
  std::vector<physx::PxArticulationLink*> links;
  physx::PxVec3 origin{0.0f, 0.0f, 0.0f};
};

struct Fleet {
  physx::PxScene* scene = nullptr;
  std::vector<Robot> robots;

  std::vector<int> dof_of_motor;

  std::vector<int> link_of_body;

  int pelvis_link = 0;

  int max_links = 0;
  int max_dofs = 0;
};

struct Sim {
  physx::PxFoundation* foundation = nullptr;
  physx::PxPhysics* physics = nullptr;
  physx::PxCudaContextManager* cuda = nullptr;
  physx::PxDefaultCpuDispatcher* dispatcher = nullptr;
  physx::PxMaterial* material = nullptr;

  ~Sim();
};

void fleet_set_drives(
    Fleet& fleet,
    const mjcf::Model& model,
    const float* kp,
    const float* kd,
    const float* frclimit
);

void fleet_read_link_poses(
    Fleet& fleet,
    std::vector<physx::PxTransform>& out
);

double mesh_rbound(const std::string& path);

std::vector<physx::PxVec3> stl_vertices_of(const std::string& path);

std::unique_ptr<Sim> sim_make();

Fleet fleet_make(
    Sim& sim,
    const mjcf::Model& model,
    int count,
    double spacing
);

}

namespace world {

inline constexpr int NUM_GROUPS = 5;

struct World {
  robot::Fleet* fleet = nullptr;
  int envs = 0;
  int max_dofs = 0;
  int max_links = 0;
  double timestep = 0.002;
  double time = 0.0;

  void* d_indices = nullptr;

  float* d_q = nullptr;
  float* d_dq = nullptr;
  float* d_dq_prev = nullptr;
  float* d_alpha_prev = nullptr;
  float* d_force = nullptr;
  float* d_root_pose = nullptr;
  float* d_root_angvel = nullptr;
  float* d_root_linvel = nullptr;
  float* d_park = nullptr;
  float* d_zero3 = nullptr;
  float* d_link_pose = nullptr;
  float* d_link_vel = nullptr;
  float* d_link_force = nullptr;
  float* d_dof_target = nullptr;

  float* d_motor_q = nullptr;
  float* d_motor_dq = nullptr;
  float* d_imu_quat = nullptr;
  float* d_gravity = nullptr;
  float* d_gyro = nullptr;
  float* d_lin_vel = nullptr;

  float* d_q_target = nullptr;
  float* d_dq_target = nullptr;
  float* d_tau_ff = nullptr;

  float* d_kp = nullptr;
  float* d_kd = nullptr;
  float* d_frclimit = nullptr;
  float* d_jdamping = nullptr;
  int* d_dof_of_motor = nullptr;
  int* d_group_of_motor = nullptr;

  unsigned char* d_alive = nullptr;
  float* d_fell_at = nullptr;

  float* d_punch_force = nullptr;
  float* d_punch_torque = nullptr;
  int* d_punch_link = nullptr;
  float* d_link_torque = nullptr;

  float* d_energy = nullptr;
  float* d_vibration = nullptr;

  ~World();
};

World* world_make(
    robot::Fleet& fleet,
    const mjcf::Model& model,
    const float* kp,
    const float* kd,
    const float* frclimit,
    const float* jdamping,
    const int* group
);

void world_read(World& w);

void world_read_velocity(World& w);

void world_apply_pd(World& w);

void world_apply_targets(World& w);

void world_apply_punches(World& w);

void world_step(World& w);

void world_check_falls(
    World& w,
    double floor_z
);

void world_meter(
    World& w,
    double dt
);

void world_reset(
    World& w,
    const double* stance,
    double height
);

void world_hold_base(
    World& w,
    double height
);

void world_fetch(
    const float* device,
    float* host,
    size_t count
);

void world_upload(
    const float* host,
    float* device,
    size_t count
);

}
