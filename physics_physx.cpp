#include "physics_physx.h"

#include "physics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>

using namespace physx;

namespace robot {
namespace {

constexpr float MJ_CONTACT_STIFFNESS = 2770.1f * 0.95f / 0.05f;
constexpr float MJ_CONTACT_DAMPING = 458.8f;

PxDefaultAllocator g_alloc;
PxDefaultErrorCallback g_err;

PxQuat quat_of(const mjcf::Quat& q) {
  return PxQuat(
      static_cast<float>(q[1]),
      static_cast<float>(q[2]),
      static_cast<float>(q[3]),
      static_cast<float>(q[0])
  );
}

PxVec3 vec_of(const mjcf::Vec3& v) {
  return PxVec3(
      static_cast<float>(v[0]),
      static_cast<float>(v[1]),
      static_cast<float>(v[2])
  );
}

PxQuat rotation_to(const PxVec3& axis) {
  const PxVec3 to = axis.getNormalized();
  const PxVec3 from(1.0f, 0.0f, 0.0f);
  const float dot = from.dot(to);
  if (dot > 1.0f - 1e-6f) return PxQuat(PxIdentity);
  if (dot < -1.0f + 1e-6f) return PxQuat(PxPi, PxVec3(0.0f, 0.0f, 1.0f));
  const PxVec3 axis_of = from.cross(to).getNormalized();
  return PxQuat(std::acos(dot), axis_of);
}

std::vector<PxVec3> stl_vertices(const std::string& path) {
  const std::vector<mjcf::Vec3> raw = mjcf::stl_vertices(path);
  std::vector<PxVec3> out;
  out.reserve(raw.size());
  for (const mjcf::Vec3& v : raw) {
    out.push_back(PxVec3(float(v[0]), float(v[1]), float(v[2])));
  }
  return out;
}

PxConvexMesh* cook_hull(
    const std::string& path,
    int maxhullvert
) {
  const std::vector<PxVec3> verts = stl_vertices(path);

  PxConvexMeshDesc desc;
  desc.points.count = static_cast<PxU32>(verts.size());
  desc.points.stride = sizeof(PxVec3);
  desc.points.data = verts.data();
  desc.flags = PxConvexFlag::eCOMPUTE_CONVEX |
               PxConvexFlag::eDISABLE_MESH_VALIDATION |
               PxConvexFlag::eFAST_INERTIA_COMPUTATION;

  if (maxhullvert > 0) desc.vertexLimit = static_cast<PxU16>(maxhullvert);

  PxTolerancesScale scale;
  PxCookingParams params(scale);

  params.buildGPUData = true;
  PxConvexMesh* mesh = PxCreateConvexMesh(params, desc);
  if (mesh == nullptr)
    throw std::runtime_error("stl: cannot cook a hull for " + path);
  return mesh;
}

PxFilterFlags fleet_filter(
    PxFilterObjectAttributes a0,
    PxFilterData d0,
    PxFilterObjectAttributes a1,
    PxFilterData d1,
    PxPairFlags& pairFlags,
    const void*,
    PxU32
) {
  PX_UNUSED(a0);
  PX_UNUSED(a1);
  if (d0.word0 != 0 && d1.word0 != 0 && d0.word0 != d1.word0) {
    return PxFilterFlag::eSUPPRESS;
  }
  pairFlags = PxPairFlag::eCONTACT_DEFAULT;
  return PxFilterFlag::eDEFAULT;
}

void add_shape(
    PxPhysics& physics,
    PxArticulationLink& link,
    const mjcf::Geom& g,
    std::map<
        std::string,
        PxConvexMesh*>& hulls,
    const mjcf::Model& model,
    PxMaterial& material,
    PxU32 robot_id
) {
  PxTransform pose(vec_of(g.pos), quat_of(g.quat));
  PxShape* shape = nullptr;

  if (g.type == "mesh") {
    const auto file = model.mesh_file.find(g.mesh);
    if (file == model.mesh_file.end())
      throw std::runtime_error(
          "robot: geom names unknown mesh '" + g.mesh + "'"
      );
    auto hit = hulls.find(g.mesh);
    if (hit == hulls.end())
      hit = hulls
                .emplace(
                    g.mesh,
                    cook_hull(file->second.file, file->second.maxhullvert)
                )
                .first;
    shape =
        physics.createShape(PxConvexMeshGeometry(hit->second), material, true);
  } else if (g.type == "sphere") {
    if (g.size.empty())
      throw std::runtime_error("robot: sphere geom has no size");
    shape = physics.createShape(
        PxSphereGeometry(static_cast<float>(g.size[0])),
        material,
        true
    );
  } else if (g.type == "cylinder") {
    if (g.size.size() < 2)
      throw std::runtime_error(
          "robot: cylinder geom needs radius and half-length"
      );
    const float r = static_cast<float>(g.size[0]);
    const float h = static_cast<float>(g.size[1]);
    shape = physics.createShape(PxCapsuleGeometry(r, h), material, true);

    pose = pose * PxTransform(PxQuat(PxHalfPi, PxVec3(0.0f, 1.0f, 0.0f)));
  } else {
    throw std::runtime_error(
        "robot: geom type '" + g.type + "' is not implemented"
    );
  }

  shape->setLocalPose(pose);

  shape->setContactOffset(0.005f);
  shape->setRestOffset(0.0f);
  PxFilterData fd;
  fd.word0 = robot_id;
  shape->setSimulationFilterData(fd);
  link.attachShape(*shape);
  shape->release();
}

}

namespace {

void cuda_ok(
    cudaError_t e,
    const char* what
) {
  if (e != cudaSuccess)
    throw std::runtime_error(
        std::string("cuda: ") + what + ": " + cudaGetErrorString(e)
    );
}

}

void fleet_read_link_poses(
    Fleet& fleet,
    std::vector<PxTransform>& out
) {
  const size_t robots = fleet.robots.size();
  const size_t stride = static_cast<size_t>(fleet.max_links);
  out.resize(robots * stride);

  std::vector<PxArticulationGPUIndex> indices(robots);
  for (size_t r = 0; r < robots; ++r)
    indices[r] = fleet.robots[r].art->getGPUIndex();

  PxArticulationGPUIndex* d_idx = nullptr;
  PxTransform* d_pose = nullptr;
  cuda_ok(
      cudaMalloc(&d_idx, robots * sizeof(PxArticulationGPUIndex)),
      "malloc indices"
  );
  cuda_ok(
      cudaMalloc(&d_pose, out.size() * sizeof(PxTransform)),
      "malloc poses"
  );
  cuda_ok(
      cudaMemcpy(
          d_idx,
          indices.data(),
          robots * sizeof(PxArticulationGPUIndex),
          cudaMemcpyHostToDevice
      ),
      "copy indices"
  );

  const bool ok = fleet.scene->getDirectGPUAPI().getArticulationData(
      d_pose,
      d_idx,
      PxArticulationGPUAPIReadType::eLINK_GLOBAL_POSE,
      static_cast<PxU32>(robots)
  );
  if (!ok)
    throw std::runtime_error(
        "physx: getArticulationData(eLINK_GLOBAL_POSE) failed"
    );

  cuda_ok(
      cudaMemcpy(
          out.data(),
          d_pose,
          out.size() * sizeof(PxTransform),
          cudaMemcpyDeviceToHost
      ),
      "copy poses back"
  );
  cuda_ok(cudaFree(d_idx), "free indices");
  cuda_ok(cudaFree(d_pose), "free poses");
}

std::vector<PxVec3> stl_vertices_of(const std::string& path) {
  return stl_vertices(path);
}

double mesh_rbound(const std::string& path) { return mjcf::mesh_rbound(path); }

void fleet_set_drives(
    Fleet& fleet,
    const mjcf::Model& model,
    const float* kp,
    const float* kd,
    const float* frclimit
) {
  for (Robot& bot : fleet.robots) {
    int at = 0;
    for (size_t b = 0; b < model.bodies.size(); ++b) {
      for (const mjcf::Joint& j : model.bodies[b].joints) {
        if (j.is_free()) continue;
        PxArticulationJointReducedCoordinate* joint =
            bot.links[b]->getInboundJoint();
        PxArticulationDrive drive;
        drive.stiffness = kp[at];

        drive.damping = kd[at] + static_cast<float>(j.damping);
        drive.maxForce = frclimit[at] > 0.0f ? frclimit[at] : PX_MAX_F32;
        drive.driveType = PxArticulationDriveType::eFORCE;
        joint->setDriveParams(PxArticulationAxis::eTWIST, drive);
        ++at;
      }
    }
  }
}

Sim::~Sim() {
  if (material != nullptr) material->release();
  if (dispatcher != nullptr) dispatcher->release();
  if (physics != nullptr) physics->release();
  if (cuda != nullptr) cuda->release();
  if (foundation != nullptr) foundation->release();
}

std::unique_ptr<Sim> sim_make() {
  std::unique_ptr<Sim> sim = std::make_unique<Sim>();

  sim->foundation = PxCreateFoundation(PX_PHYSICS_VERSION, g_alloc, g_err);
  if (sim->foundation == nullptr)
    throw std::runtime_error("physx: cannot create the foundation");

  PxCudaContextManagerDesc cuda_desc;
  sim->cuda = PxCreateCudaContextManager(*sim->foundation, cuda_desc);
  if (sim->cuda == nullptr || !sim->cuda->contextIsValid()) {
    throw std::runtime_error("physx: no usable CUDA device");
  }

  sim->physics = PxCreatePhysics(
      PX_PHYSICS_VERSION,
      *sim->foundation,
      PxTolerancesScale(),
      true
  );
  if (sim->physics == nullptr)
    throw std::runtime_error("physx: cannot create the SDK");

  sim->dispatcher = PxDefaultCpuDispatcherCreate(0);

  sim->material =
      sim->physics->createMaterial(1.0f, 1.0f, -MJ_CONTACT_STIFFNESS);
  sim->material->setDamping(MJ_CONTACT_DAMPING);
  sim->material->setFlag(PxMaterialFlag::eCOMPLIANT_ACCELERATION_SPRING, true);
  return sim;
}

Fleet fleet_make(
    Sim& sim,
    const mjcf::Model& model,
    int count,
    double spacing
) {
  if (count <= 0)
    throw std::runtime_error("robot: a fleet needs at least one robot");

  PxSceneDesc desc(sim.physics->getTolerancesScale());
  desc.gravity = PxVec3(0.0f, 0.0f, -9.81f);
  desc.cpuDispatcher = sim.dispatcher;
  desc.filterShader = fleet_filter;

  const bool cpu_scene = getenv("CPUSCENE") != nullptr;
  if (!cpu_scene) {
    desc.cudaContextManager = sim.cuda;
    desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
    desc.flags |= PxSceneFlag::eENABLE_DIRECT_GPU_API;
    desc.broadPhaseType = PxBroadPhaseType::eGPU;
  }
  desc.solverType = PxSolverType::eTGS;

  desc.gpuMaxNumPartitions = 8;
  desc.gpuDynamicsConfig.maxRigidContactCount = 1u << 20;
  desc.gpuDynamicsConfig.maxRigidPatchCount = 1u << 18;

  Fleet fleet;
  fleet.scene = sim.physics->createScene(desc);
  if (fleet.scene == nullptr)
    throw std::runtime_error("physx: cannot create a GPU scene");
  if (!cpu_scene &&
      !fleet.scene->getFlags().isSet(PxSceneFlag::eENABLE_GPU_DYNAMICS)) {
    throw std::runtime_error("physx: the scene fell back to the CPU solver");
  }

  PxRigidStatic* floor = PxCreatePlane(
      *sim.physics,
      PxPlane(0.0f, 0.0f, 1.0f, 0.0f),
      *sim.material
  );
  {
    PxShape* shape = nullptr;
    floor->getShapes(&shape, 1);
    PxFilterData fd;
    fd.word0 = 0;
    shape->setSimulationFilterData(fd);
  }
  fleet.scene->addActor(*floor);

  std::vector<int> motor_body;
  for (size_t b = 0; b < model.bodies.size(); ++b) {
    for (const mjcf::Joint& j : model.bodies[b].joints) {
      if (!j.is_free()) motor_body.push_back(static_cast<int>(b));
    }
  }
  if (static_cast<int>(motor_body.size()) != NUM_MOTOR) {
    throw std::runtime_error(
        "robot: expected " + std::to_string(NUM_MOTOR) + " hinges, model has " +
        std::to_string(motor_body.size())
    );
  }

  std::map<std::string, PxConvexMesh*> hulls;
  const int side =
      static_cast<int>(std::ceil(std::sqrt(static_cast<double>(count))));

  for (int r = 0; r < count; ++r) {
    Robot bot;
    bot.origin = PxVec3(
        static_cast<float>((r % side) * spacing),
        static_cast<float>((r / side) * spacing),
        0.0f
    );

    bot.art = sim.physics->createArticulationReducedCoordinate();

    bot.art->setArticulationFlag(
        PxArticulationFlag::eDISABLE_SELF_COLLISION,
        true
    );
    bot.art->setSolverIterationCounts(8, 1);

    bot.art->setArticulationFlag(
        PxArticulationFlag::eDRIVE_LIMITS_ARE_FORCES,
        true
    );

    bot.links.assign(model.bodies.size(), nullptr);

    for (size_t b = 0; b < model.bodies.size(); ++b) {
      const mjcf::Body& body = model.bodies[b];

      PxTransform global(vec_of(body.pos), quat_of(body.quat));
      if (body.parent >= 0) {
        global = bot.links[static_cast<size_t>(body.parent)]->getGlobalPose() *
                 global;
      } else {
        global = PxTransform(global.p + bot.origin, global.q);
      }

      PxArticulationLink* parent =
          body.parent >= 0 ? bot.links[static_cast<size_t>(body.parent)]
                           : nullptr;
      PxArticulationLink* link = bot.art->createLink(parent, global);
      if (link == nullptr)
        throw std::runtime_error("robot: cannot create link " + body.name);
      bot.links[b] = link;

      for (const mjcf::Geom& g : body.geoms) {
        if (!g.collides) continue;
        add_shape(
            *sim.physics,
            *link,
            g,
            hulls,
            model,
            *sim.material,
            static_cast<PxU32>(r + 1)
        );
      }

      link->setMass(static_cast<float>(body.inertial.mass));
      link->setCMassLocalPose(
          PxTransform(vec_of(body.inertial.pos), quat_of(body.inertial.quat))
      );
      link->setMassSpaceInertiaTensor(vec_of(body.inertial.diaginertia));

      if (parent == nullptr) continue;

      if (body.joints.size() != 1) {
        throw std::runtime_error(
            "robot: body '" + body.name + "' has " +
            std::to_string(body.joints.size()) +
            " joints; exactly one hinge is expected"
        );
      }
      const mjcf::Joint& j = body.joints[0];

      PxArticulationJointReducedCoordinate* joint = link->getInboundJoint();

      joint->setJointType(PxArticulationJointType::eREVOLUTE_UNWRAPPED);

      const PxTransform child_pose(vec_of(j.pos), rotation_to(vec_of(j.axis)));
      const PxTransform body_pose(vec_of(body.pos), quat_of(body.quat));
      joint->setChildPose(child_pose);
      joint->setParentPose(body_pose * child_pose);

      const bool limited = j.limited;
      joint->setMotion(
          PxArticulationAxis::eTWIST,
          limited ? PxArticulationMotion::eLIMITED : PxArticulationMotion::eFREE
      );
      if (limited) {
        joint->setLimitParams(
            PxArticulationAxis::eTWIST,
            PxArticulationLimit(
                static_cast<float>(j.range[0]),
                static_cast<float>(j.range[1])
            )
        );
      }

      joint->setArmature(
          PxArticulationAxis::eTWIST,
          static_cast<float>(j.armature)
      );

      joint->setFrictionCoefficient(0.0f);
      joint->setFrictionParams(
          PxArticulationAxis::eTWIST,
          PxJointFrictionParams(
              static_cast<float>(j.frictionloss),
              static_cast<float>(j.frictionloss),
              0.0f
          )
      );
    }

    fleet.scene->addArticulation(*bot.art);
    fleet.robots.push_back(bot);
  }

  fleet.link_of_body.assign(model.bodies.size(), -1);
  for (size_t b = 0; b < model.bodies.size(); ++b) {
    fleet.link_of_body[b] =
        static_cast<int>(fleet.robots[0].links[b]->getLinkIndex());
  }
  fleet.pelvis_link = fleet.link_of_body[0];
  fleet.max_links = static_cast<int>(model.bodies.size());
  fleet.max_dofs = NUM_MOTOR;

  const PxU32 link_count = fleet.robots[0].art->getNbLinks();
  std::vector<PxArticulationLink*> by_index(link_count, nullptr);
  for (PxArticulationLink* link : fleet.robots[0].links) {
    const PxU32 at = link->getLinkIndex();
    if (at >= link_count) {
      throw std::runtime_error(
          "robot: a link indexes outside its articulation"
      );
    }
    by_index[at] = link;
  }
  std::vector<int> dof_start(link_count, -1);
  int dofs = 0;
  for (PxU32 i = 0; i < link_count; ++i) {
    if (by_index[i] == nullptr) {
      throw std::runtime_error(
          "robot: the articulation has a gap in its link indices"
      );
    }
    dof_start[i] = dofs;
    dofs += static_cast<int>(by_index[i]->getInboundJointDof());
  }

  fleet.dof_of_motor.assign(NUM_MOTOR, -1);
  for (int i = 0; i < NUM_MOTOR; ++i) {
    PxArticulationLink* link =
        fleet.robots[0]
            .links[static_cast<size_t>(motor_body[static_cast<size_t>(i)])];
    fleet.dof_of_motor[static_cast<size_t>(i)] =
        dof_start[link->getLinkIndex()];
  }

  std::vector<int> seen(static_cast<size_t>(dofs), 0);
  for (int d : fleet.dof_of_motor) {
    if (d < 0 || d >= dofs || seen[static_cast<size_t>(d)]++ != 0) {
      throw std::runtime_error(
          "robot: the joint-to-dof mapping is not one-to-one"
      );
    }
  }

  return fleet;
}

}

namespace {

constexpr int NUM_MOTOR = 29;

__device__ inline void quat_conj_rot(
    const float* q,
    float vx,
    float vy,
    float vz,
    float* out
) {
  const float x = -q[0], y = -q[1], z = -q[2], w = q[3];
  const float tx = 2.0f * (y * vz - z * vy);
  const float ty = 2.0f * (z * vx - x * vz);
  const float tz = 2.0f * (x * vy - y * vx);
  out[0] = vx + w * tx + (y * tz - z * ty);
  out[1] = vy + w * ty + (z * tx - x * tz);
  out[2] = vz + w * tz + (x * ty - y * tx);
}

}

__global__ void k_apply_pd(
    const float* __restrict__ q,
    const float* __restrict__ dq,
    const float* __restrict__ q_target,
    const float* __restrict__ dq_target,
    const float* __restrict__ tau_ff,
    const float* __restrict__ kp,
    const float* __restrict__ kd,
    const float* __restrict__ frclimit,
    const float* __restrict__ jdamping,
    const int* __restrict__ dof_of_motor,
    const unsigned char* __restrict__ alive,
    float* __restrict__ force,
    int envs,
    int max_dofs
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= envs * NUM_MOTOR) return;
  const int env = i / NUM_MOTOR;
  const int m = i - env * NUM_MOTOR;

  const int dof = dof_of_motor[m];
  const int slot = env * max_dofs + dof;

  if (alive[env] == 0) {
    force[slot] = 0.0f;
    return;
  }

  const int cmd = env * NUM_MOTOR + m;
  float tau = kp[m] * (q_target[cmd] - q[slot]) +
              kd[m] * (dq_target[cmd] - dq[slot]) + tau_ff[cmd];

  const float lim = frclimit[m];
  if (lim > 0.0f) tau = fminf(fmaxf(tau, -lim), lim);

  force[slot] = tau - jdamping[m] * dq[slot];
}

__global__ void k_read_sensors(
    const float* __restrict__ q,
    const float* __restrict__ dq,
    const float* __restrict__ root_pose,
    const float* __restrict__ root_angvel,
    const int* __restrict__ dof_of_motor,
    float* __restrict__ motor_q,
    float* __restrict__ motor_dq,
    float* __restrict__ imu_quat,
    float* __restrict__ gravity_body,
    float* __restrict__ gyro_body,
    int envs,
    int max_dofs
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  for (int m = 0; m < NUM_MOTOR; ++m) {
    const int slot = env * max_dofs + dof_of_motor[m];
    motor_q[env * NUM_MOTOR + m] = q[slot];
    motor_dq[env * NUM_MOTOR + m] = dq[slot];
  }

  const float* rp = root_pose + env * 7;

  imu_quat[env * 4 + 0] = rp[3];
  imu_quat[env * 4 + 1] = rp[0];
  imu_quat[env * 4 + 2] = rp[1];
  imu_quat[env * 4 + 3] = rp[2];

  quat_conj_rot(rp, 0.0f, 0.0f, -1.0f, gravity_body + env * 3);

  const float* wv = root_angvel + env * 3;
  quat_conj_rot(rp, wv[0], wv[1], wv[2], gyro_body + env * 3);
}

__global__ void k_check_falls(
    const float* __restrict__ link_pose,
    unsigned char* __restrict__ alive,
    float* __restrict__ fell_at,
    float now,
    float floor_z,
    int envs,
    int max_links,
    int pelvis_link
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs || alive[env] == 0) return;
  const float z = link_pose[(env * max_links + pelvis_link) * 7 + 6];
  if (z < floor_z) {
    alive[env] = 0;
    fell_at[env] = now;
  }
}

__global__ void k_apply_punches(
    const float* __restrict__ punch_force,
    const float* __restrict__ punch_torque,
    const int* __restrict__ punch_link,
    const unsigned char* __restrict__ alive,
    float* __restrict__ link_force,
    float* __restrict__ link_torque,
    int envs,
    int max_links
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs) return;

  for (int l = 0; l < max_links; ++l) {
    float* f = link_force + (env * max_links + l) * 3;
    float* t = link_torque + (env * max_links + l) * 3;
    f[0] = 0.0f;
    f[1] = 0.0f;
    f[2] = 0.0f;
    t[0] = 0.0f;
    t[1] = 0.0f;
    t[2] = 0.0f;
  }
  if (alive[env] == 0) return;

  const int l = punch_link[env];
  if (l < 0 || l >= max_links) return;

  float* f = link_force + (env * max_links + l) * 3;
  float* t = link_torque + (env * max_links + l) * 3;
  for (int k = 0; k < 3; ++k) {
    f[k] = punch_force[env * 3 + k];
    t[k] = punch_torque[env * 3 + k];
  }
}

__global__ void k_meter(
    const float* __restrict__ q,
    const float* __restrict__ dq,
    const float* __restrict__ dq_prev,
    const float* __restrict__ q_target,
    const float* __restrict__ kp,
    const float* __restrict__ kd,
    const float* __restrict__ frclimit,
    const int* __restrict__ dof_of_motor,
    const int* __restrict__ group_of_motor,
    const unsigned char* __restrict__ alive,
    float* __restrict__ alpha_prev,
    float* __restrict__ energy,
    float* __restrict__ vibration,
    float dt,
    int envs,
    int max_dofs,
    int groups
) {
  const int env = blockIdx.x * blockDim.x + threadIdx.x;
  if (env >= envs || alive[env] == 0) return;

  for (int m = 0; m < NUM_MOTOR; ++m) {
    const int slot = env * max_dofs + dof_of_motor[m];
    const int g = env * groups + group_of_motor[m];
    const int motor = env * NUM_MOTOR + m;
    const float v = dq[slot];

    float tau = kp[m] * (q_target[motor] - q[slot]) - kd[m] * v;
    const float cap = frclimit[m];
    if (cap > 0.0f) tau = fminf(fmaxf(tau, -cap), cap);
    energy[g] += fabsf(tau * v) * dt;

    const float alpha = (v - dq_prev[slot]) / dt;
    vibration[g] += fabsf(alpha - alpha_prev[slot]);
    alpha_prev[slot] = alpha;
  }
}

__global__ void k_scatter_targets(
    const float* __restrict__ q_target,
    const int* __restrict__ dof_of_motor,
    const unsigned char* __restrict__ alive,
    float* __restrict__ dof_target,
    int envs,
    int max_dofs
) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= envs * NUM_MOTOR) return;
  const int env = i / NUM_MOTOR;
  const int m = i - env * NUM_MOTOR;
  const int slot = env * max_dofs + dof_of_motor[m];

  dof_target[slot] = alive[env] != 0 ? q_target[env * NUM_MOTOR + m] : 0.0f;
}

namespace world {
namespace {

void cuda_ok(
    cudaError_t e,
    const char* what
) {
  if (e != cudaSuccess)
    throw std::runtime_error(
        std::string("cuda: ") + what + ": " + cudaGetErrorString(e)
    );
}

template <typename T>
T* device_zeros(size_t count) {
  T* p = nullptr;
  cuda_ok(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc");
  cuda_ok(cudaMemset(p, 0, count * sizeof(T)), "cudaMemset");
  return p;
}

template <typename T>
T* device_from(
    const T* host,
    size_t count
) {
  T* p = nullptr;
  cuda_ok(cudaMalloc(&p, count * sizeof(T)), "cudaMalloc");
  cuda_ok(
      cudaMemcpy(p, host, count * sizeof(T), cudaMemcpyHostToDevice),
      "cudaMemcpy H2D"
  );
  return p;
}

int blocks_for(
    int items,
    int threads
) {
  return (items + threads - 1) / threads;
}

void read_into(
    World& w,
    void* dst,
    PxArticulationGPUAPIReadType::Enum type,
    const char* what
) {
  const bool ok = w.fleet->scene->getDirectGPUAPI().getArticulationData(
      dst,
      static_cast<const PxArticulationGPUIndex*>(w.d_indices),
      type,
      static_cast<PxU32>(w.envs)
  );
  if (!ok)
    throw std::runtime_error(
        std::string("physx: getArticulationData(") + what + ") failed"
    );
}

constexpr float PARK_HEIGHT_M = 5000.0f;

void write_from(
    World& w,
    const void* src,
    PxArticulationGPUAPIWriteType::Enum type,
    const char* what
) {
  const bool ok = w.fleet->scene->getDirectGPUAPI().setArticulationData(
      src,
      static_cast<const PxArticulationGPUIndex*>(w.d_indices),
      type,
      static_cast<PxU32>(w.envs)
  );
  if (!ok)
    throw std::runtime_error(
        std::string("physx: setArticulationData(") + what + ") failed"
    );
}

}

World::~World() {
  for (void* p : {(void*)d_indices,      (void*)d_q,
                  (void*)d_dq,           (void*)d_dq_prev,
                  (void*)d_alpha_prev,   (void*)d_force,
                  (void*)d_root_pose,    (void*)d_park,
                  (void*)d_zero3,        (void*)d_root_angvel,
                  (void*)d_link_pose,    (void*)d_link_vel,
                  (void*)d_link_force,   (void*)d_dof_target,
                  (void*)d_motor_q,      (void*)d_motor_dq,
                  (void*)d_imu_quat,     (void*)d_gravity,
                  (void*)d_gyro,         (void*)d_q_target,
                  (void*)d_dq_target,    (void*)d_tau_ff,
                  (void*)d_kp,           (void*)d_kd,
                  (void*)d_frclimit,     (void*)d_jdamping,
                  (void*)d_dof_of_motor, (void*)d_group_of_motor,
                  (void*)d_alive,        (void*)d_fell_at,
                  (void*)d_punch_force,  (void*)d_punch_torque,
                  (void*)d_punch_link,   (void*)d_link_torque,
                  (void*)d_energy,       (void*)d_vibration}) {
    if (p != nullptr) cudaFree(p);
  }
}

World* world_make(
    robot::Fleet& fleet,
    const mjcf::Model& model,
    const float* kp,
    const float* kd,
    const float* frclimit,
    const float* jdamping,
    const int* group
) {
  World* w = new World();
  w->fleet = &fleet;
  w->envs = static_cast<int>(fleet.robots.size());
  w->timestep = model.timestep;

  fleet.scene->simulate(static_cast<PxReal>(w->timestep));
  fleet.scene->fetchResults(true);

  const PxArticulationGPUAPIMaxCounts caps =
      fleet.scene->getDirectGPUAPI().getArticulationGPUAPIMaxCounts();
  w->max_dofs = static_cast<int>(caps.maxDofs);
  w->max_links = static_cast<int>(caps.maxLinks);

  if (w->max_dofs <= 0 || w->max_links <= 0) {
    w->max_dofs = static_cast<int>(fleet.robots[0].art->getDofs());
    w->max_links = static_cast<int>(fleet.robots[0].art->getNbLinks());
  }
  if (w->max_dofs <= 0 || w->max_links <= 0) {
    throw std::runtime_error(
        "physx: the articulation reports no dofs or links"
    );
  }
  fleet.max_dofs = w->max_dofs;
  fleet.max_links = w->max_links;

  std::vector<PxArticulationGPUIndex> idx(static_cast<size_t>(w->envs));
  for (int r = 0; r < w->envs; ++r)
    idx[static_cast<size_t>(r)] =
        fleet.robots[static_cast<size_t>(r)].art->getGPUIndex();
  w->d_indices = device_from(idx.data(), idx.size());

  const size_t dofs =
      static_cast<size_t>(w->envs) * static_cast<size_t>(w->max_dofs);
  const size_t links =
      static_cast<size_t>(w->envs) * static_cast<size_t>(w->max_links);
  const size_t motors = static_cast<size_t>(w->envs) * NUM_MOTOR;

  w->d_q = device_zeros<float>(dofs);
  w->d_dq = device_zeros<float>(dofs);
  w->d_dq_prev = device_zeros<float>(dofs);
  w->d_alpha_prev = device_zeros<float>(dofs);
  w->d_force = device_zeros<float>(dofs);
  w->d_root_pose = device_zeros<float>(static_cast<size_t>(w->envs) * 7);
  w->d_park = device_zeros<float>(7);
  w->d_zero3 = device_zeros<float>(3);
  w->d_root_angvel = device_zeros<float>(static_cast<size_t>(w->envs) * 3);
  w->d_link_pose = device_zeros<float>(links * 7);
  w->d_link_vel = device_zeros<float>(links * 3);
  w->d_link_force = device_zeros<float>(links * 3);
  w->d_dof_target = device_zeros<float>(dofs);

  w->d_motor_q = device_zeros<float>(motors);
  w->d_motor_dq = device_zeros<float>(motors);
  w->d_imu_quat = device_zeros<float>(static_cast<size_t>(w->envs) * 4);
  w->d_gravity = device_zeros<float>(static_cast<size_t>(w->envs) * 3);
  w->d_gyro = device_zeros<float>(static_cast<size_t>(w->envs) * 3);

  w->d_q_target = device_zeros<float>(motors);
  w->d_dq_target = device_zeros<float>(motors);
  w->d_tau_ff = device_zeros<float>(motors);

  w->d_kp = device_from(kp, NUM_MOTOR);
  w->d_kd = device_from(kd, NUM_MOTOR);
  w->d_frclimit = device_from(frclimit, NUM_MOTOR);
  w->d_jdamping = device_from(jdamping, NUM_MOTOR);
  w->d_dof_of_motor =
      device_from(fleet.dof_of_motor.data(), fleet.dof_of_motor.size());
  w->d_group_of_motor = device_from(group, NUM_MOTOR);

  std::vector<unsigned char> alive(static_cast<size_t>(w->envs), 1);
  w->d_alive = device_from(alive.data(), alive.size());
  w->d_fell_at = device_zeros<float>(static_cast<size_t>(w->envs));

  w->d_punch_force = device_zeros<float>(static_cast<size_t>(w->envs) * 3);
  w->d_punch_torque = device_zeros<float>(static_cast<size_t>(w->envs) * 3);
  w->d_punch_link = device_zeros<int>(static_cast<size_t>(w->envs));
  w->d_link_torque = device_zeros<float>(links * 3);

  w->d_energy = device_zeros<float>(static_cast<size_t>(w->envs) * NUM_GROUPS);
  w->d_vibration =
      device_zeros<float>(static_cast<size_t>(w->envs) * NUM_GROUPS);

  return w;
}

void world_read(World& w) {
  cuda_ok(
      cudaMemcpy(
          w.d_dq_prev,
          w.d_dq,
          static_cast<size_t>(w.envs) * static_cast<size_t>(w.max_dofs) *
              sizeof(float),
          cudaMemcpyDeviceToDevice
      ),
      "keep dq"
  );

  read_into(
      w,
      w.d_q,
      PxArticulationGPUAPIReadType::eJOINT_POSITION,
      "joint position"
  );
  read_into(
      w,
      w.d_dq,
      PxArticulationGPUAPIReadType::eJOINT_VELOCITY,
      "joint velocity"
  );
  read_into(
      w,
      w.d_root_pose,
      PxArticulationGPUAPIReadType::eROOT_GLOBAL_POSE,
      "root pose"
  );
  read_into(
      w,
      w.d_root_angvel,
      PxArticulationGPUAPIReadType::eROOT_ANGULAR_VELOCITY,
      "root angular velocity"
  );
  read_into(
      w,
      w.d_link_pose,
      PxArticulationGPUAPIReadType::eLINK_GLOBAL_POSE,
      "link pose"
  );

  read_into(
      w,
      w.d_link_vel,
      PxArticulationGPUAPIReadType::eLINK_LINEAR_VELOCITY,
      "link linear velocity"
  );
  const int threads = 128;
  k_read_sensors<<<blocks_for(w.envs, threads), threads>>>(
      w.d_q,
      w.d_dq,
      w.d_root_pose,
      w.d_root_angvel,
      w.d_dof_of_motor,
      w.d_motor_q,
      w.d_motor_dq,
      w.d_imu_quat,
      w.d_gravity,
      w.d_gyro,
      w.envs,
      w.max_dofs
  );
  cuda_ok(cudaGetLastError(), "k_read_sensors");
}

void world_read_velocity(World& w) {
  cuda_ok(
      cudaMemcpy(
          w.d_dq_prev,
          w.d_dq,
          static_cast<size_t>(w.envs) * static_cast<size_t>(w.max_dofs) *
              sizeof(float),
          cudaMemcpyDeviceToDevice
      ),
      "keep dq"
  );
  read_into(
      w,
      w.d_dq,
      PxArticulationGPUAPIReadType::eJOINT_VELOCITY,
      "joint velocity"
  );
}

void world_apply_pd(World& w) {
  const int threads = 256;
  k_apply_pd<<<blocks_for(w.envs * NUM_MOTOR, threads), threads>>>(
      w.d_q,
      w.d_dq,
      w.d_q_target,
      w.d_dq_target,
      w.d_tau_ff,
      w.d_kp,
      w.d_kd,
      w.d_frclimit,
      w.d_jdamping,
      w.d_dof_of_motor,
      w.d_alive,
      w.d_force,
      w.envs,
      w.max_dofs
  );
  cuda_ok(cudaGetLastError(), "k_apply_pd");
  write_from(
      w,
      w.d_force,
      PxArticulationGPUAPIWriteType::eJOINT_FORCE,
      "joint force"
  );
}

void world_apply_targets(World& w) {
  const int threads = 256;
  k_scatter_targets<<<blocks_for(w.envs * NUM_MOTOR, threads), threads>>>(
      w.d_q_target,
      w.d_dof_of_motor,
      w.d_alive,
      w.d_dof_target,
      w.envs,
      w.max_dofs
  );
  cuda_ok(cudaGetLastError(), "k_scatter_targets");
  write_from(
      w,
      w.d_dof_target,
      PxArticulationGPUAPIWriteType::eJOINT_TARGET_POSITION,
      "joint target position"
  );
}

void world_apply_punches(World& w) {
  const int threads = 128;
  k_apply_punches<<<blocks_for(w.envs, threads), threads>>>(
      w.d_punch_force,
      w.d_punch_torque,
      w.d_punch_link,
      w.d_alive,
      w.d_link_force,
      w.d_link_torque,
      w.envs,
      w.max_links
  );
  cuda_ok(cudaGetLastError(), "k_apply_punches");
  write_from(
      w,
      w.d_link_force,
      PxArticulationGPUAPIWriteType::eLINK_FORCE,
      "link force"
  );
  write_from(
      w,
      w.d_link_torque,
      PxArticulationGPUAPIWriteType::eLINK_TORQUE,
      "link torque"
  );
}

void world_park(
    World& w,
    int env
) {
  const PxVec3 o = w.fleet->robots[static_cast<size_t>(env)].origin;
  const float pose[7] = {0.0f, 0.0f, 0.0f, 1.0f, o.x, o.y, PARK_HEIGHT_M};
  cuda_ok(
      cudaMemcpy(w.d_park, pose, sizeof(pose), cudaMemcpyHostToDevice),
      "park pose"
  );
  const PxArticulationGPUIndex* idx =
      static_cast<const PxArticulationGPUIndex*>(w.d_indices) + env;
  PxDirectGPUAPI& api = w.fleet->scene->getDirectGPUAPI();
  api.setArticulationData(
      w.d_park,
      idx,
      PxArticulationGPUAPIWriteType::eROOT_GLOBAL_POSE,
      1
  );
  api.setArticulationData(
      w.d_zero3,
      idx,
      PxArticulationGPUAPIWriteType::eROOT_LINEAR_VELOCITY,
      1
  );
  api.setArticulationData(
      w.d_zero3,
      idx,
      PxArticulationGPUAPIWriteType::eROOT_ANGULAR_VELOCITY,
      1
  );
  api.computeArticulationData(
      nullptr,
      idx,
      PxArticulationGPUAPIComputeType::eUPDATE_KINEMATIC,
      1
  );
}

void world_step(World& w) {
  w.fleet->scene->simulate(static_cast<PxReal>(w.timestep));
  w.fleet->scene->fetchResults(true);
  w.time += w.timestep;
}

void world_check_falls(
    World& w,
    double floor_z
) {
  const int threads = 128;
  k_check_falls<<<blocks_for(w.envs, threads), threads>>>(
      w.d_link_pose,
      w.d_alive,
      w.d_fell_at,
      static_cast<float>(w.time),
      static_cast<float>(floor_z),
      w.envs,
      w.max_links,
      w.fleet->pelvis_link
  );
  cuda_ok(cudaGetLastError(), "k_check_falls");
}

void world_meter(
    World& w,
    double dt
) {
  const int threads = 128;
  k_meter<<<blocks_for(w.envs, threads), threads>>>(
      w.d_q,
      w.d_dq,
      w.d_dq_prev,
      w.d_q_target,
      w.d_kp,
      w.d_kd,
      w.d_frclimit,
      w.d_dof_of_motor,
      w.d_group_of_motor,
      w.d_alive,
      w.d_alpha_prev,
      w.d_energy,
      w.d_vibration,
      static_cast<float>(dt),
      w.envs,
      w.max_dofs,
      NUM_GROUPS
  );
  cuda_ok(cudaGetLastError(), "k_meter");
}

void world_reset(
    World& w,
    const double* stance,
    double height
) {
  const size_t dofs =
      static_cast<size_t>(w.envs) * static_cast<size_t>(w.max_dofs);

  std::vector<float> q(dofs, 0.0f);
  for (int e = 0; e < w.envs; ++e) {
    for (int m = 0; m < NUM_MOTOR; ++m) {
      q[static_cast<size_t>(e) * static_cast<size_t>(w.max_dofs) +
        static_cast<size_t>(w.fleet->dof_of_motor[static_cast<size_t>(m)])] =
          static_cast<float>(stance[m]);
    }
  }
  cuda_ok(
      cudaMemcpy(w.d_q, q.data(), dofs * sizeof(float), cudaMemcpyHostToDevice),
      "reset q"
  );
  cuda_ok(cudaMemset(w.d_dq, 0, dofs * sizeof(float)), "reset dq");
  cuda_ok(cudaMemset(w.d_dq_prev, 0, dofs * sizeof(float)), "reset dq_prev");
  cuda_ok(
      cudaMemset(w.d_alpha_prev, 0, dofs * sizeof(float)),
      "reset alpha_prev"
  );

  std::vector<float> root(static_cast<size_t>(w.envs) * 7, 0.0f);
  for (int e = 0; e < w.envs; ++e) {
    const PxVec3 o = w.fleet->robots[static_cast<size_t>(e)].origin;
    float* r = root.data() + static_cast<size_t>(e) * 7;
    r[0] = 0.0f;
    r[1] = 0.0f;
    r[2] = 0.0f;
    r[3] = 1.0f;
    r[4] = o.x;
    r[5] = o.y;
    r[6] = static_cast<float>(height);
  }
  cuda_ok(
      cudaMemcpy(
          w.d_root_pose,
          root.data(),
          root.size() * sizeof(float),
          cudaMemcpyHostToDevice
      ),
      "reset root"
  );

  write_from(
      w,
      w.d_q,
      PxArticulationGPUAPIWriteType::eJOINT_POSITION,
      "joint position"
  );
  write_from(
      w,
      w.d_dq,
      PxArticulationGPUAPIWriteType::eJOINT_VELOCITY,
      "joint velocity"
  );
  write_from(
      w,
      w.d_root_pose,
      PxArticulationGPUAPIWriteType::eROOT_GLOBAL_POSE,
      "root pose"
  );
  cuda_ok(
      cudaMemset(
          w.d_root_angvel,
          0,
          static_cast<size_t>(w.envs) * 3 * sizeof(float)
      ),
      "reset angvel"
  );
  write_from(
      w,
      w.d_root_angvel,
      PxArticulationGPUAPIWriteType::eROOT_LINEAR_VELOCITY,
      "root linear velocity"
  );
  write_from(
      w,
      w.d_root_angvel,
      PxArticulationGPUAPIWriteType::eROOT_ANGULAR_VELOCITY,
      "root angular velocity"
  );

  w.fleet->scene->getDirectGPUAPI().computeArticulationData(
      nullptr,
      static_cast<const PxArticulationGPUIndex*>(w.d_indices),
      PxArticulationGPUAPIComputeType::eUPDATE_KINEMATIC,
      static_cast<PxU32>(w.envs)
  );

  std::vector<unsigned char> alive(static_cast<size_t>(w.envs), 1);
  cuda_ok(
      cudaMemcpy(w.d_alive, alive.data(), alive.size(), cudaMemcpyHostToDevice),
      "reset alive"
  );
  cuda_ok(
      cudaMemset(w.d_fell_at, 0, static_cast<size_t>(w.envs) * sizeof(float)),
      "reset fell_at"
  );
  cuda_ok(
      cudaMemset(
          w.d_energy,
          0,
          static_cast<size_t>(w.envs) * NUM_GROUPS * sizeof(float)
      ),
      "reset energy"
  );
  cuda_ok(
      cudaMemset(
          w.d_vibration,
          0,
          static_cast<size_t>(w.envs) * NUM_GROUPS * sizeof(float)
      ),
      "reset vibration"
  );
  w.time = 0.0;
}

void world_hold_base(
    World& w,
    double height
) {
  std::vector<float> root(static_cast<size_t>(w.envs) * 7, 0.0f);
  for (int e = 0; e < w.envs; ++e) {
    const PxVec3 o = w.fleet->robots[static_cast<size_t>(e)].origin;
    float* r = root.data() + static_cast<size_t>(e) * 7;
    r[0] = 0.0f;
    r[1] = 0.0f;
    r[2] = 0.0f;
    r[3] = 1.0f;
    r[4] = o.x;
    r[5] = o.y;
    r[6] = static_cast<float>(height);
  }
  cuda_ok(
      cudaMemcpy(
          w.d_root_pose,
          root.data(),
          root.size() * sizeof(float),
          cudaMemcpyHostToDevice
      ),
      "hold base"
  );
  write_from(
      w,
      w.d_root_pose,
      PxArticulationGPUAPIWriteType::eROOT_GLOBAL_POSE,
      "root pose"
  );
  cuda_ok(
      cudaMemset(
          w.d_root_angvel,
          0,
          static_cast<size_t>(w.envs) * 3 * sizeof(float)
      ),
      "hold angvel"
  );
  write_from(
      w,
      w.d_root_angvel,
      PxArticulationGPUAPIWriteType::eROOT_LINEAR_VELOCITY,
      "root linear velocity"
  );
  write_from(
      w,
      w.d_root_angvel,
      PxArticulationGPUAPIWriteType::eROOT_ANGULAR_VELOCITY,
      "root angular velocity"
  );

  w.fleet->scene->getDirectGPUAPI().computeArticulationData(
      nullptr,
      static_cast<const PxArticulationGPUIndex*>(w.d_indices),
      PxArticulationGPUAPIComputeType::eUPDATE_KINEMATIC,
      static_cast<PxU32>(w.envs)
  );
}

void world_fetch(
    const float* device,
    float* host,
    size_t count
) {
  cuda_ok(
      cudaMemcpy(host, device, count * sizeof(float), cudaMemcpyDeviceToHost),
      "fetch"
  );
}

void world_upload(
    const float* host,
    float* device,
    size_t count
) {
  cuda_ok(
      cudaMemcpy(device, host, count * sizeof(float), cudaMemcpyHostToDevice),
      "upload"
  );
}

}

namespace {

const int PX_GROUP[PHYS_NUM_MOTOR] = {0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                                      1, 1, 2, 2, 2, 3, 3, 3, 3, 4,
                                      4, 4, 3, 3, 3, 3, 4, 4, 4};

void px_quat_rot(
    const float* q,
    const double* v,
    double* out
) {
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  const double tx = 2.0 * (y * v[2] - z * v[1]);
  const double ty = 2.0 * (z * v[0] - x * v[2]);
  const double tz = 2.0 * (x * v[1] - y * v[0]);
  out[0] = v[0] + w * tx + (y * tz - z * ty);
  out[1] = v[1] + w * ty + (z * tx - x * tz);
  out[2] = v[2] + w * tz + (x * ty - y * tx);
}

class PhysxPhysics : public Physics {
 public:
  PhysxPhysics(
      const mjcf::Model& model,
      int envs,
      double spacing
  )
      : model_(model) {
    sim_ = robot::sim_make();
    fleet_ = robot::fleet_make(*sim_, model, envs, spacing);

    lim_.assign(PHYS_NUM_MOTOR, 0.0f);
    damp_.assign(PHYS_NUM_MOTOR, 0.0f);
    joint_body_.assign(PHYS_NUM_MOTOR, 0);
    joint_anchor_.assign(size_t(PHYS_NUM_MOTOR) * 3, 0.0);
    int at = 0;
    for (size_t b = 0; b < model.bodies.size(); ++b) {
      for (const mjcf::Joint& j : model.bodies[b].joints) {
        if (j.is_free()) continue;
        if (at < PHYS_NUM_MOTOR) {
          lim_[size_t(at)] =
              j.frc_limited ? float(std::min(-j.frcrange[0], j.frcrange[1]))
                            : 0.0f;
          damp_[size_t(at)] = float(j.damping);
          joint_body_[size_t(at)] = int(b);
          for (int k = 0; k < 3; ++k) {
            joint_anchor_[size_t(at) * 3 + size_t(k)] = j.pos[size_t(k)];
          }
        }
        ++at;
      }
    }

    std::vector<float> zero(PHYS_NUM_MOTOR, 0.0f);
    world_ = world::world_make(
        fleet_,
        model,
        zero.data(),
        zero.data(),
        lim_.data(),
        damp_.data(),
        PX_GROUP
    );

    const int n = world_->envs;
    host_pose_.assign(size_t(n) * 7, 0.0f);
    host_foot_.assign(size_t(n) * 2, 0.0f);
    host_speed_.assign(size_t(n) * 2, 0.0f);
    host_link_vel_.assign(size_t(n) * size_t(fleet_.max_links) * 3, 0.0f);
    host_link_.assign(size_t(n) * size_t(world_->max_links) * 7, 0.0f);
    alive_.assign(size_t(n), 1);
    energy_.assign(size_t(n) * PHYS_GROUPS, 0.0f);
    vibration_.assign(size_t(n) * PHYS_GROUPS, 0.0f);
    punch_force_.assign(size_t(n) * 3, 0.0f);
    punch_torque_.assign(size_t(n) * 3, 0.0f);
    punch_link_.assign(size_t(n), -1);
    pelvis_ = fleet_.pelvis_link;
    lfoot_ =
        fleet_.link_of_body[size_t(model.body_index("left_ankle_roll_link"))];
    rfoot_ =
        fleet_.link_of_body[size_t(model.body_index("right_ankle_roll_link"))];
    torso_ = fleet_.link_of_body[size_t(model.body_index("torso_link"))];
  }

  ~PhysxPhysics() override { delete world_; }

  int envs() const override { return world_->envs; }
  double timestep() const override { return world_->timestep; }

  void set_gains(
      const float* kp,
      const float* kd
  ) override {
    robot::fleet_set_drives(fleet_, model_, kp, kd, lim_.data());
    world::world_upload(kp, world_->d_kp, PHYS_NUM_MOTOR);
    world::world_upload(kd, world_->d_kd, PHYS_NUM_MOTOR);
  }

  void reset(
      const double* stance,
      double height
  ) override {
    world::world_reset(*world_, stance, height);
    std::fill(alive_.begin(), alive_.end(), 1);
  }

  void hold_base(double height) override {
    world::world_hold_base(*world_, height);
  }

  void read() override {
    world::world_read(*world_);
    world::world_fetch(
        world_->d_link_pose,
        host_link_.data(),
        host_link_.size()
    );
    world::world_fetch(
        world_->d_link_vel,
        host_link_vel_.data(),
        host_link_vel_.size()
    );
    const size_t stride = size_t(world_->max_links) * 7;
    for (int e = 0; e < world_->envs; ++e) {
      const float* base = host_link_.data() + size_t(e) * stride;
      std::copy(
          base + size_t(pelvis_) * 7,
          base + size_t(pelvis_) * 7 + 7,
          host_pose_.data() + size_t(e) * 7
      );
      host_foot_[size_t(e) * 2 + 0] = base[size_t(lfoot_) * 7 + 6];
      host_foot_[size_t(e) * 2 + 1] = base[size_t(rfoot_) * 7 + 6];
      const float* vel =
          host_link_vel_.data() + size_t(e) * size_t(world_->max_links) * 3;
      const int of[2] = {pelvis_, torso_};
      for (int k = 0; k < 2; ++k) {
        const float* v = vel + size_t(of[k]) * 3;
        host_speed_[size_t(e) * 2 + size_t(k)] =
            std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      }
    }
    cudaMemcpy(
        alive_.data(),
        world_->d_alive,
        alive_.size(),
        cudaMemcpyDeviceToHost
    );
  }

  const float* motor_q() const override { return world_->d_motor_q; }
  const float* motor_dq() const override { return world_->d_motor_dq; }
  const float* gyro() const override { return world_->d_gyro; }
  const float* gravity() const override { return world_->d_gravity; }
  const float* base_quat() const override { return world_->d_imu_quat; }
  float* q_target() override { return world_->d_q_target; }

  const float* base_pose() const override { return host_pose_.data(); }
  const float* foot_height() const override { return host_foot_.data(); }
  const float* body_speed() const override { return host_speed_.data(); }

  void apply_targets() override { world::world_apply_targets(*world_); }

  void set_punches(const PhysicsPunch* punches) override {
    const size_t stride = size_t(world_->max_links) * 7;
    for (int e = 0; e < world_->envs; ++e) {
      float* pf = punch_force_.data() + size_t(e) * 3;
      float* pt = punch_torque_.data() + size_t(e) * 3;
      punch_link_[size_t(e)] = -1;
      pf[0] = pf[1] = pf[2] = 0.0f;
      pt[0] = pt[1] = pt[2] = 0.0f;
      const PhysicsPunch& p = punches[e];
      if (p.joint < 0 || alive_[size_t(e)] == 0) continue;

      const int body = joint_body_[size_t(p.joint)];
      const int link = fleet_.link_of_body[size_t(body)];
      const float* L =
          host_link_.data() + size_t(e) * stride + size_t(link) * 7;
      const mjcf::Vec3& com = model_.bodies[size_t(body)].inertial.pos;
      const double local[3] = {
          joint_anchor_[size_t(p.joint) * 3 + 0] - com[0],
          joint_anchor_[size_t(p.joint) * 3 + 1] - com[1],
          joint_anchor_[size_t(p.joint) * 3 + 2] - com[2]
      };
      double arm[3];
      px_quat_rot(L, local, arm);
      const double f[3] = {p.force[0], p.force[1], p.force[2]};
      for (int k = 0; k < 3; ++k) pf[k] = float(f[k]);
      pt[0] = float(arm[1] * f[2] - arm[2] * f[1]);
      pt[1] = float(arm[2] * f[0] - arm[0] * f[2]);
      pt[2] = float(arm[0] * f[1] - arm[1] * f[0]);
      punch_link_[size_t(e)] = link;
    }
    world::world_upload(
        punch_force_.data(),
        world_->d_punch_force,
        punch_force_.size()
    );
    world::world_upload(
        punch_torque_.data(),
        world_->d_punch_torque,
        punch_torque_.size()
    );
    cudaMemcpy(
        world_->d_punch_link,
        punch_link_.data(),
        punch_link_.size() * sizeof(int),
        cudaMemcpyHostToDevice
    );
  }

  void step(int substeps) override {
    for (int k = 0; k < substeps; ++k) {
      world::world_apply_punches(*world_);
      world::world_step(*world_);
    }
  }

  void kill(int env) override {
    if (alive_[size_t(env)] == 0) return;
    alive_[size_t(env)] = 0;
    world::world_park(*world_, env);
    cudaMemcpy(
        world_->d_alive + env,
        alive_.data() + env,
        1,
        cudaMemcpyHostToDevice
    );
  }
  const unsigned char* alive() const override { return alive_.data(); }

  void meter(double dt) override { world::world_meter(*world_, dt); }

  const float* energy() const override {
    world::world_fetch(world_->d_energy, energy_.data(), energy_.size());
    return energy_.data();
  }
  const float* vibration() const override {
    world::world_fetch(
        world_->d_vibration,
        vibration_.data(),
        vibration_.size()
    );
    return vibration_.data();
  }

 private:
  const mjcf::Model& model_;
  std::unique_ptr<robot::Sim> sim_;
  robot::Fleet fleet_;
  world::World* world_ = nullptr;
  std::vector<float> lim_, damp_;
  std::vector<int> joint_body_;
  std::vector<double> joint_anchor_;
  std::vector<float> host_pose_, host_foot_, host_link_;
  std::vector<float> host_speed_, host_link_vel_;
  mutable std::vector<float> energy_, vibration_;
  std::vector<float> punch_force_, punch_torque_;
  std::vector<int> punch_link_;
  std::vector<unsigned char> alive_;
  int pelvis_ = 0, lfoot_ = 0, rfoot_ = 0, torso_ = 0;
};

}

std::unique_ptr<Physics> physics_make_physx(
    const mjcf::Model& model,
    int envs,
    double spacing
) {
  return std::make_unique<PhysxPhysics>(model, envs, spacing);
}
