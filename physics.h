#pragma once

#include <memory>
#include <string>

#include "model.h"

constexpr int PHYS_NUM_MOTOR = 29;
constexpr int PHYS_GROUPS = 5;

enum class PhysicsEngine { kMujoco, kPhysx };

struct PhysicsPunch {
  int joint = -1;
  float force[3] = {0.0f, 0.0f, 0.0f};
};

class Physics {
 public:
  virtual ~Physics() = default;

  virtual int envs() const = 0;
  virtual double timestep() const = 0;

  virtual void set_gains(
      const float* kp,
      const float* kd
  ) = 0;
  virtual void reset(
      const double* stance,
      double height
  ) = 0;
  virtual void hold_base(double height) = 0;

  virtual void read() = 0;

  virtual const float* motor_q() const = 0;
  virtual const float* motor_dq() const = 0;
  virtual const float* gyro() const = 0;
  virtual const float* gravity() const = 0;
  virtual const float* base_quat() const = 0;
  virtual float* q_target() = 0;

  virtual const float* base_pose() const = 0;
  virtual const float* foot_height() const = 0;
  virtual const float* body_speed() const = 0;

  virtual void apply_targets() = 0;
  virtual void set_punches(const PhysicsPunch* punches) = 0;
  virtual void step(int substeps) = 0;

  virtual void kill(int env) = 0;
  virtual const unsigned char* alive() const = 0;

  virtual void meter(double dt) = 0;
  virtual const float* energy() const = 0;
  virtual const float* vibration() const = 0;
};

std::unique_ptr<Physics> physics_make_physx(
    const mjcf::Model& model,
    int envs,
    double spacing
);

std::unique_ptr<Physics> physics_make_mjwarp(
    const std::string& graph_path,
    int envs
);
