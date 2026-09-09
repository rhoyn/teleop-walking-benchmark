#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <GLFW/glfw3.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <PxPhysicsAPI.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <freetype/freetype.h>
#include <mujoco/mujoco.h>

#include "model.h"
#include "physics.h"
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

constexpr uint32_t STREAM_GUARD = 1;
constexpr uint32_t STREAM_TOUR = 2;
constexpr uint32_t STREAM_PUNCH = 3;
constexpr uint32_t STREAM_ARM = 4;

std::mt19937 random_for(
    uint32_t id,
    uint32_t stream
) {
  std::seed_seq seq{id, stream};
  return std::mt19937(seq);
}

double random_unit(std::mt19937& r) {
  return double(r() >> 8) * (1.0 / 16777216.0);
}

double random_between(
    std::mt19937& r,
    double lo,
    double hi
) {
  return lo + (hi - lo) * random_unit(r);
}

uint32_t random_below(
    std::mt19937& r,
    uint32_t bound
) {
  const uint64_t span = 0x100000000ull / bound;
  const uint64_t limit = span * bound;
  uint64_t draw = 0;
  do {
    draw = r();
  } while (draw >= limit);
  return uint32_t(draw / span);
}

namespace policy_api {

inline constexpr int NUM_MOTOR = 29;

struct Limits {
  double vx_min, vx_max, vy_abs, yaw_rate_abs, speed_norm;

  double pos_reached_enter_m = 0.10;
  double pos_reached_exit_m = 0.20;
  double yaw_reached_enter_rad = 0.05;
  double yaw_reached_exit_rad = 0.12;

  double walk_kp_pos = 1.5;
  double walk_kp_yaw = 1.5;
};

struct Ctx {
  const float* motor_q;
  const float* motor_dq;
  const float* gyro;
  const float* gravity;
  const float* cmd;
  const float* task;

  const float* arm_pose;
  const float* base_quat;

  float* q_target;
};

struct Policy {
  virtual ~Policy() = default;
  virtual void init(int envs) = 0;
  virtual void step(const Ctx& c) = 0;
  virtual const float* kp() const = 0;
  virtual const float* kd() const = 0;
  virtual int owned() const = 0;
  virtual Limits limits() const = 0;
  virtual const char* name() const = 0;
};

struct TensorSpec {
  std::string name;
  std::vector<int> shape;
};

struct EngineCore {
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
};

struct Engine {
  std::shared_ptr<EngineCore> core;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  std::string input_name;
  std::string output_name;
  int max_batch = 0;
  int obs_dim = 0;
  std::vector<TensorSpec> inputs;
  std::vector<TensorSpec> outputs;

  std::vector<void*> spare;
  ~Engine() {
    context.reset();
    for (void* p : spare) cudaFree(p);
  }
};

std::shared_ptr<Engine> engine_make(
    const std::string& onnx_path,
    int batch,
    int obs_dim,
    int act_dim
);

std::shared_ptr<Engine> engine_make(
    const std::string& onnx_path,
    int batch,
    const std::vector<TensorSpec>& inputs,
    const std::vector<TensorSpec>& outputs
);

void engine_run(
    Engine& e,
    const float* const* d_in,
    float* const* d_out,
    int batch
);

void engine_run(
    Engine& e,
    const float* d_obs,
    float* d_act,
    int batch
);

}

constexpr int POLICY_NUM_MOTOR = policy_api::NUM_MOTOR;

std::unique_ptr<policy_api::Policy> make_policy(const std::string& name);

std::vector<std::string> policy_names();

namespace policy_api {
namespace {

class Logger : public nvinfer1::ILogger {
 public:
  void log(
      Severity severity,
      const char* msg
  ) noexcept override {
    if (severity <= Severity::kWARNING) std::fprintf(stderr, "trt: %s\n", msg);
  }
};

Logger& logger() {
  static Logger instance;
  return instance;
}

std::string cache_path(
    const std::string& onnx_path,
    int batch,
    int obs_dim
) {
  std::filesystem::path p(onnx_path);
  const std::uintmax_t size = std::filesystem::file_size(p);
  const auto stamp =
      std::filesystem::last_write_time(p).time_since_epoch().count();
  return "build/trt/" + p.stem().string() + "_b" + std::to_string(batch) +
         "_o" + std::to_string(obs_dim) + "_" + std::to_string(size) + "_" +
         std::to_string(static_cast<long long>(stamp)) + "_trt" +
         std::to_string(NV_TENSORRT_MAJOR) + "." +
         std::to_string(NV_TENSORRT_MINOR) + ".plan";
}

nvinfer1::Dims dims_of(
    const TensorSpec& t,
    int batch
) {
  nvinfer1::Dims d;
  d.nbDims = int(t.shape.size());
  for (size_t k = 0; k < t.shape.size(); ++k) {
    d.d[k] = t.shape[k] < 0 ? batch : t.shape[k];
  }
  return d;
}

std::vector<char> plan_build(
    const std::string& onnx_path,
    int batch,
    const std::vector<TensorSpec>& specs
) {
  std::unique_ptr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(logger())
  );
  if (!builder) throw std::runtime_error("trt: cannot create a builder");

  std::unique_ptr<nvinfer1::INetworkDefinition> net(
      builder->createNetworkV2(0)
  );
  std::unique_ptr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*net, logger())
  );
  if (!parser->parseFromFile(
          onnx_path.c_str(),
          static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)
      )) {
    throw std::runtime_error("trt: cannot parse " + onnx_path);
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig()
  );

  if (std::getenv("TRT_FP32")) config->clearFlag(nvinfer1::BuilderFlag::kTF32);

  const std::string timing_path = "build/trt/timing.cache";
  std::vector<char> timing_blob;
  if (std::ifstream tf(timing_path, std::ios::binary); tf) {
    timing_blob.assign(
        std::istreambuf_iterator<char>(tf),
        std::istreambuf_iterator<char>()
    );
  }
  std::unique_ptr<nvinfer1::ITimingCache> timing(
      config->createTimingCache(timing_blob.data(), timing_blob.size())
  );
  if (timing) config->setTimingCache(*timing, false);

  nvinfer1::IOptimizationProfile* profile =
      builder->createOptimizationProfile();

  bool any_fixed = false;
  for (size_t si = 0; si < specs.size(); ++si) {
    const TensorSpec& t = specs[si];

    nvinfer1::ITensor* in = nullptr;
    if (t.name.empty()) {
      in = net->getInput(int(si));
    } else {
      for (int i = 0; i < net->getNbInputs(); ++i) {
        if (t.name == net->getInput(i)->getName()) {
          in = net->getInput(i);
          break;
        }
      }
    }
    if (in == nullptr)
      throw std::runtime_error(
          "trt: " + onnx_path + " has no input '" + t.name + "'"
      );
    const std::string bound = in->getName();

    nvinfer1::Dims shape = in->getDimensions();
    bool fixed = false;
    for (size_t k = 0; k < t.shape.size(); ++k) {
      if (t.shape[k] < 0 && k < size_t(shape.nbDims) && shape.d[k] != -1)
        fixed = true;
    }
    if (fixed) {
      nvinfer1::Dims open = dims_of(t, -1);
      in->setDimensions(open);
      any_fixed = true;
    }
    profile->setDimensions(
        bound.c_str(),
        nvinfer1::OptProfileSelector::kMIN,
        dims_of(t, 1)
    );
    profile->setDimensions(
        bound.c_str(),
        nvinfer1::OptProfileSelector::kOPT,
        dims_of(t, batch)
    );
    profile->setDimensions(
        bound.c_str(),
        nvinfer1::OptProfileSelector::kMAX,
        dims_of(t, batch)
    );
  }

  if (any_fixed) {
    for (int li = 0; li < net->getNbLayers(); ++li) {
      nvinfer1::ILayer* layer = net->getLayer(li);
      if (layer->getType() != nvinfer1::LayerType::kSHUFFLE) continue;
      auto* shuffle = static_cast<nvinfer1::IShuffleLayer*>(layer);
      if (shuffle->getNbInputs() > 1 && shuffle->getInput(1) != nullptr)
        continue;
      nvinfer1::Dims r = shuffle->getReshapeDimensions();
      if (r.nbDims <= 0 || r.d[0] != 1) continue;
      bool wildcard = false;
      for (int k = 1; k < r.nbDims; ++k) wildcard = wildcard || r.d[k] < 0;
      if (wildcard) continue;
      r.d[0] = -1;
      shuffle->setReshapeDimensions(r);
    }
  }

  config->addOptimizationProfile(profile);

  std::unique_ptr<nvinfer1::IHostMemory> plan(
      builder->buildSerializedNetwork(*net, *config)
  );
  if (!plan)
    throw std::runtime_error("trt: cannot build a plan for " + onnx_path);

  if (const nvinfer1::ITimingCache* used = config->getTimingCache()) {
    std::unique_ptr<nvinfer1::IHostMemory> blob(used->serialize());
    if (blob) {
      std::filesystem::create_directories("build/trt");
      std::ofstream tf(timing_path, std::ios::binary);
      tf.write(
          static_cast<const char*>(blob->data()),
          static_cast<std::streamsize>(blob->size())
      );
    }
  }
  const char* p = static_cast<const char*>(plan->data());
  return std::vector<char>(p, p + plan->size());
}

}

void engine_run(
    Engine& e,
    const float* const* d_in,
    float* const* d_out,
    int batch
);

void engine_bind_spares(Engine& e) {
  auto named = [&](const char* nm) {
    for (const TensorSpec& t : e.inputs)
      if (t.name == nm) return true;
    for (const TensorSpec& t : e.outputs)
      if (t.name == nm) return true;
    return false;
  };
  for (const TensorSpec& t : e.inputs) {
    e.context->setInputShape(t.name.c_str(), dims_of(t, e.max_batch));
  }
  for (int i = 0; i < e.core->engine->getNbIOTensors(); ++i) {
    const char* nm = e.core->engine->getIOTensorName(i);
    if (named(nm)) continue;
    const nvinfer1::Dims d = e.context->getTensorShape(nm);
    size_t n = 1;
    for (int k = 0; k < d.nbDims; ++k)
      n *= size_t(d.d[k] < 0 ? e.max_batch : d.d[k]);
    float* p = nullptr;
    cudaMalloc(&p, n * sizeof(float));
    cudaMemset(p, 0, n * sizeof(float));
    e.spare.push_back(p);
    e.context->setTensorAddress(nm, p);
  }
}

void engine_assert_batched(
    Engine& e,
    const std::string& onnx_path
) {
  if (e.max_batch < 2) return;

  auto volume = [&](const TensorSpec& t) {
    size_t n = 1;
    for (int d : t.shape) n *= size_t(d < 0 ? e.max_batch : d);
    return n;
  };

  auto split = [&](const TensorSpec& t, size_t& outer, size_t& inner) {
    outer = 1;
    inner = 1;
    bool seen = false;
    for (int d : t.shape) {
      if (d < 0) {
        seen = true;
        continue;
      }
      (seen ? inner : outer) *= size_t(d);
    }
  };

  std::mt19937 rng = random_for(0, STREAM_GUARD);
  std::vector<void*> owned;
  std::vector<std::vector<float>> host_out(e.outputs.size());
  bool ok = false;

  auto bind = [&](const char* nm, const nvinfer1::Dims& d, bool input) {
    size_t n = 1;
    for (int k = 0; k < d.nbDims; ++k)
      n *= size_t(d.d[k] < 0 ? e.max_batch : d.d[k]);
    float* p = nullptr;
    cudaMalloc(&p, n * sizeof(float));

    owned.push_back(p);
    if (input) {
      std::vector<float> h(n);
      for (float& v : h) v = float(random_between(rng, -1.0, 1.0));
      cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    }
    e.context->setTensorAddress(nm, p);
    return p;
  };

  for (const TensorSpec& t : e.inputs) {
    e.context->setInputShape(t.name.c_str(), dims_of(t, e.max_batch));
  }
  std::map<std::string, float*> addr;
  for (const TensorSpec& t : e.inputs)
    addr[t.name] =
        bind(t.name.c_str(), e.context->getTensorShape(t.name.c_str()), true);
  for (const TensorSpec& t : e.outputs)
    addr[t.name] =
        bind(t.name.c_str(), e.context->getTensorShape(t.name.c_str()), false);

  if (!e.context->enqueueV3(nullptr)) {
    for (void* p : owned) cudaFree(p);
    throw std::runtime_error(
        "trt: " + onnx_path + " will not run a batch of noise"
    );
  }
  cudaDeviceSynchronize();

  for (size_t k = 0; k < e.outputs.size() && !ok; ++k) {
    host_out[k].resize(volume(e.outputs[k]));
    cudaMemcpy(
        host_out[k].data(),
        addr[e.outputs[k].name],
        host_out[k].size() * sizeof(float),
        cudaMemcpyDeviceToHost
    );
    size_t outer = 1, inner = 1;
    split(e.outputs[k], outer, inner);
    const size_t rows = size_t(e.max_batch);
    for (size_t o = 0; o < outer && !ok; ++o) {
      for (size_t i = 0; i < inner && !ok; ++i) {
        const float first = host_out[k][(o * rows + 0) * inner + i];
        for (size_t r = 1; r < rows; ++r) {
          if (host_out[k][(o * rows + r) * inner + i] != first) {
            ok = true;
            break;
          }
        }
      }
    }
  }

  for (void* p : owned) cudaFree(p);
  if (!ok) {
    throw std::runtime_error(
        "trt: " + onnx_path +
        " gives every row the same answer, so its batch axis is not a batch. "
        "The graph was exported at a literal batch of one and has a node that "
        "reads it as a value; it needs a re-export with a dynamic batch, or a "
        "policy that steps it a row at a time."
    );
  }
}

namespace {

std::mutex& plan_lock() {
  static std::mutex m;
  return m;
}

std::shared_ptr<EngineCore> core_for(
    const std::string& key,
    const std::vector<char>& plan,
    const std::string& onnx_path
) {
  static std::map<std::string, std::weak_ptr<EngineCore>> live;
  if (std::shared_ptr<EngineCore> held = live[key].lock()) return held;

  auto core = std::make_shared<EngineCore>();
  core->runtime.reset(nvinfer1::createInferRuntime(logger()));
  core->engine.reset(
      core->runtime->deserializeCudaEngine(plan.data(), plan.size())
  );
  if (!core->engine) {
    throw std::runtime_error(
        "trt: cannot deserialize the plan for " + onnx_path
    );
  }
  live[key] = core;
  return core;
}

}

std::shared_ptr<Engine> engine_make(
    const std::string& onnx_path,
    int batch,
    const std::vector<TensorSpec>& inputs,
    const std::vector<TensorSpec>& outputs
) {
  const std::lock_guard<std::mutex> hold(plan_lock());
  if (batch < 1)
    throw std::runtime_error("trt: a plan needs a batch of at least one");

  std::string key;
  for (const TensorSpec& t : inputs) {
    key += "_" + t.name;
    for (int d : t.shape) key += "x" + std::to_string(d);
  }
  const std::string cache =
      cache_path(onnx_path, batch, int(std::hash<std::string>{}(key) & 0xffff));

  std::vector<char> plan;
  if (std::ifstream f(cache, std::ios::binary); f) {
    plan.assign(
        std::istreambuf_iterator<char>(f),
        std::istreambuf_iterator<char>()
    );
  } else {
    std::printf(
        "trt: building a plan for %s at batch %d\n",
        onnx_path.c_str(),
        batch
    );
    plan = plan_build(onnx_path, batch, inputs);
    std::filesystem::create_directories("build/trt");
    std::ofstream out(cache, std::ios::binary);
    out.write(plan.data(), static_cast<std::streamsize>(plan.size()));
  }

  auto e = std::make_shared<Engine>();
  e->max_batch = batch;
  e->inputs = inputs;
  e->outputs = outputs;
  e->obs_dim = inputs.empty() ? 0 : inputs[0].shape.back();
  e->core = core_for(cache, plan, onnx_path);
  e->context.reset(e->core->engine->createExecutionContext());
  if (!e->context)
    throw std::runtime_error("trt: cannot create an execution context");

  size_t ni = 0, no = 0;
  for (int i = 0; i < e->core->engine->getNbIOTensors(); ++i) {
    const char* nm = e->core->engine->getIOTensorName(i);
    if (e->core->engine->getTensorIOMode(nm) ==
        nvinfer1::TensorIOMode::kINPUT) {
      if (ni < e->inputs.size() && e->inputs[ni].name.empty())
        e->inputs[ni].name = nm;
      ++ni;
    } else {
      if (no < e->outputs.size() && e->outputs[no].name.empty())
        e->outputs[no].name = nm;
      ++no;
    }
  }
  if (!e->inputs.empty()) e->input_name = e->inputs[0].name;
  if (!e->outputs.empty()) e->output_name = e->outputs[0].name;
  engine_bind_spares(*e);
  engine_assert_batched(*e, onnx_path);
  return e;
}

std::shared_ptr<Engine> engine_make(
    const std::string& onnx_path,
    int batch,
    int obs_dim,
    int act_dim
) {
  return engine_make(
      onnx_path,
      batch,
      {{"", {-1, obs_dim}}},
      {{"", {-1, act_dim}}}
  );
}

void engine_run(
    Engine& e,
    const float* const* d_in,
    float* const* d_out,
    int batch
) {
  if (batch > e.max_batch) {
    throw std::runtime_error("trt: batch exceeds what the plan was built for");
  }
  for (size_t k = 0; k < e.inputs.size(); ++k) {
    e.context->setInputShape(
        e.inputs[k].name.c_str(),
        dims_of(e.inputs[k], batch)
    );
    e.context->setTensorAddress(
        e.inputs[k].name.c_str(),
        const_cast<float*>(d_in[k])
    );
  }
  for (size_t k = 0; k < e.outputs.size(); ++k) {
    e.context->setTensorAddress(e.outputs[k].name.c_str(), d_out[k]);
  }
  if (!e.context->enqueueV3(nullptr))
    throw std::runtime_error("trt: enqueue failed");
}

void engine_run(
    Engine& e,
    const float* d_obs,
    float* d_act,
    int batch
) {
  const float* in[1] = {d_obs};
  float* out[1] = {d_act};
  engine_run(e, in, out, batch);
}

}

#include "policies/gr00t_wbc/policy.cpp"
#include "policies/amo/policy.cpp"
#include "policies/asap/policy.cpp"
#include "policies/bfm_zero/policy.cpp"
#include "policies/clobot/policy.cpp"
#include "policies/decoupled_wbc/policy.cpp"
#include "policies/dm_agile/policy.cpp"
#include "policies/dm_march/policy.cpp"
#include "policies/falcon/policy.cpp"
#include "policies/g1_gym/policy.cpp"
#include "policies/grove/policy.cpp"
#include "policies/handoff/policy.cpp"
#include "policies/holosoma/policy.cpp"
#include "policies/homie/policy.cpp"
#include "policies/legged_rl_lab/policy.cpp"
#include "policies/mimic_lite/policy.cpp"
#include "policies/nanog1/policy.cpp"
#include "policies/openwbt/policy.cpp"
#include "policies/rl_gym/policy.cpp"
#include "policies/rl_lab/policy.cpp"
#include "policies/rl_mjlab/policy.cpp"
#include "policies/robomimic/policy.cpp"
#include "policies/run_residual/policy.cpp"
#include "policies/schoi/policy.cpp"
#include "policies/sonic/policy.cpp"
#include "policies/stepdown/policy.cpp"
#include "policies/wbc_agile/policy.cpp"
#include "policies/wcompton/policy.cpp"
#include "policies/wty_cpp/policy.cpp"
#include "policies/zealot/policy.cpp"

namespace {

const char* const NAMES[] = {
    "amo",
    "asap",
    "bfm_zero",
    "clobot",
    "clobot_with_arms",
    "dm_agile",
    "dm_march",
    "falcon",
    "g1_gym",
    "grove",
    "handoff",
    "handoff_with_arms",
    "holosoma",
    "homie",
    "legged_rl_lab",
    "mimic_lite",
    "nanog1",
    "openwbt",
    "rl_gym",
    "rl_lab",
    "rl_mjlab",
    "robomimic",
    "run_residual",
    "schoi",
    "sonic",
    "stepdown",
    "wbc_agile",
    "wcompton",
    "wty_cpp",
    "zealot",
};

}

std::vector<std::string> policy_names() {
  std::vector<std::string> names(std::begin(NAMES), std::end(NAMES));
  const std::vector<std::string> gr00t = gr00t_wbc::names();
  names.insert(names.begin(), gr00t.begin(), gr00t.end());
  const std::vector<std::string> decoupled = decoupled_wbc::names();
  names.insert(names.end(), decoupled.begin(), decoupled.end());
  return names;
}

std::unique_ptr<policy_api::Policy> make_policy(const std::string& name) {
  if (auto gr00t = gr00t_wbc::make(name)) return gr00t;
  if (auto dec = decoupled_wbc::make(name)) return dec;
  if (name == "amo") return std::make_unique<amo::Policy>();
  if (name == "asap") return std::make_unique<asap::Policy>();
  if (name == "bfm_zero") return std::make_unique<bfm_zero::Policy>();
  if (name == "clobot") return std::make_unique<clobot::Policy>();
  if (name == "clobot_with_arms")
    return std::make_unique<clobot::WithArmsPolicy>();
  if (name == "dm_agile") return std::make_unique<dm_agile::Policy>();
  if (name == "dm_march") return std::make_unique<dm_march::Policy>();
  if (name == "falcon") return std::make_unique<falcon::Policy>();
  if (name == "g1_gym") return std::make_unique<g1_gym::Policy>();
  if (name == "grove") return std::make_unique<grove::Policy>();
  if (name == "handoff") return std::make_unique<handoff::Policy>();
  if (name == "handoff_with_arms")
    return std::make_unique<handoff::WithArmsPolicy>();
  if (name == "holosoma") return std::make_unique<holosoma::Policy>();
  if (name == "homie") return std::make_unique<homie::Policy>();
  if (name == "legged_rl_lab") return std::make_unique<legged_rl_lab::Policy>();
  if (name == "mimic_lite") return std::make_unique<mimic_lite::Policy>();
  if (name == "nanog1") return std::make_unique<nanog1::Policy>();
  if (name == "openwbt") return std::make_unique<openwbt::Policy>();
  if (name == "rl_gym") return std::make_unique<rl_gym::Policy>();
  if (name == "rl_lab") return std::make_unique<rl_lab::Policy>();
  if (name == "rl_mjlab") return std::make_unique<rl_mjlab::Policy>();
  if (name == "robomimic") return std::make_unique<robomimic::Policy>();
  if (name == "run_residual") return std::make_unique<run_residual::Policy>();
  if (name == "schoi") return std::make_unique<schoi::Policy>();
  if (name == "sonic") return std::make_unique<sonic::Policy>();
  if (name == "stepdown") return std::make_unique<stepdown::Policy>();
  if (name == "wbc_agile") return std::make_unique<wbc_agile::Policy>();
  if (name == "wcompton") return std::make_unique<wcompton::Policy>();
  if (name == "wty_cpp") return std::make_unique<wty_cpp::Policy>();
  if (name == "zealot") return std::make_unique<zealot::Policy>();

  std::string msg = "unknown policy '" + name + "'. Ported: ";
  for (size_t i = 0; i < std::size(NAMES); ++i) {
    if (i) msg += ", ";
    msg += NAMES[i];
  }
  throw std::runtime_error(msg);
}

namespace {

constexpr double WALK_S = 60.0;
constexpr int WAYPOINTS = 12;
constexpr double POINT_S = WALK_S / WAYPOINTS;
constexpr double CENTRE_X_M = 0.3;
constexpr double CENTRE_Y_M = 0.0;
constexpr double RADIUS_M = 1.0;
constexpr double INIT_DURATION_S = 3.0;
constexpr double FALL_PELVIS_Z = 0.20;
constexpr double PERIOD_S = 0.02;
constexpr double PROGRESS_S = 0.1;

constexpr int NUM_MOTOR = POLICY_NUM_MOTOR;

constexpr double A5020 = 0.003609725, A7520_14 = 0.010177520,
                 A7520_22 = 0.025101925, A4010 = 0.00425;
constexpr double WN = 10 * 2.0 * 3.1415926535, ZETA = 2.0;
const double ARMATURE[NUM_MOTOR] = {
    A7520_22, A7520_22,  A7520_14,  A7520_22, 2 * A5020, 2 * A5020,
    A7520_22, A7520_22,  A7520_14,  A7520_22, 2 * A5020, 2 * A5020,
    A7520_14, 2 * A5020, 2 * A5020, A5020,    A5020,     A5020,
    A5020,    A5020,     A4010,     A4010,    A5020,     A5020,
    A5020,    A5020,     A5020,     A4010,    A4010
};
const double STANCE[NUM_MOTOR] = {-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                                  -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                                  0.0,    0.0, 0.0, 0.2,   0.2,    0.0,
                                  0.6,    0.0, 0.0, 0.0,   0.2,    -0.2,
                                  0.0,    0.6, 0.0, 0.0,   0.0};

constexpr double FORCE_MAX_N = 600.0;
constexpr double FORCE_SCALE_MIN = 0.5;
constexpr double RAMP_FLOOR = 1.0 / 3.0;
constexpr double RAMP_S = 60.0;
constexpr double PUNCH_DELAY_S = 0.1;
constexpr double PUNCH_DURATION_S = 0.08;

constexpr double PUNCH_HOLD_S = 0.5;
constexpr double PUNCH_ARROW_SCALE = 2.0;
constexpr char FLOOR_FONT[] = "assets/JetBrainsMono.ttf";
constexpr char FLOOR_TEXTURE[] = "floor_label";
constexpr double FLOOR_EM_FRACTION = 0.072;
constexpr double FLOOR_TRACKING_EM = 0.18;
constexpr double FLOOR_LINE_EM = 1.35;
constexpr double FLOOR_MID_EM = 0.72;
constexpr int RGB_CHANNELS = 3;
constexpr int RECORD_FPS = 60;
constexpr int RECORD_WIDTH = 160;
constexpr int RECORD_HEIGHT = 90;

constexpr int ARM_LEFT_FIRST = 15, ARM_RIGHT_FIRST = 22, ARM_DOF = 7;
constexpr double ARM_STEP_RAD = 0.06;
constexpr int ARM_DRAWS = 8;

const double ARM_MIRROR[ARM_DOF] = {1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0};

struct Waypoint {
  double x, y, yaw;
};

struct Punch {
  double time = 0.0;
  int joint = 0;
  double dir[3] = {};
  double force_n = 0.0;
};

void direction(
    std::mt19937& rng,
    double out[3]
) {
  const double z = random_between(rng, -1.0, 1.0);
  const double phi = random_between(rng, 0.0, 2.0 * M_PI);
  const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
  out[0] = r * std::cos(phi);
  out[1] = r * std::sin(phi);
  out[2] = z;
}

struct Run {
  uint32_t run_id = 0;
  std::vector<Waypoint> tour;
  std::mt19937 arm_rng;
  double arm_left[ARM_DOF] = {};
  double arm_lo[ARM_DOF] = {};
  double arm_hi[ARM_DOF] = {};

  bool anchored = false;
  double ax = 0, ay = 0, ayaw = 0;
  int current_target = -1;
  bool pos_reached = false, yaw_reached = false;

  std::vector<Punch> punches;

  bool released_once = false;
  bool alive = true;
  double fell_at = 0.0;
  int scored = 0;
  double seg_energy[WAYPOINTS][PHYS_GROUPS] = {};
  double seg_vibration[WAYPOINTS][PHYS_GROUPS] = {};
  double taken_energy[PHYS_GROUPS] = {};
  double taken_vibration[PHYS_GROUPS] = {};
  double pos_err_sum = 0.0, yaw_err_sum = 0.0;
  double pelvis_speed = 0.0, head_speed = 0.0;
  double last_dist = 0.0, last_yaw_err = 0.0;
  bool have_last = false;
};

std::vector<Waypoint> tour_make(uint32_t run_id) {
  std::vector<Waypoint> tour;
  std::mt19937 rng = random_for(run_id, STREAM_TOUR);
  for (int i = 0; i < WAYPOINTS; ++i) {
    const double radius = RADIUS_M * std::sqrt(random_unit(rng));
    const double bearing = random_between(rng, 0.0, 2.0 * M_PI);
    const double yaw = random_between(rng, 0.0, 2.0 * M_PI);
    tour.push_back(
        Waypoint{
            CENTRE_X_M + radius * std::cos(bearing),
            CENTRE_Y_M + radius * std::sin(bearing),
            std::remainder(yaw, 2.0 * M_PI)
        }
    );
  }
  return tour;
}

std::vector<Punch> schedule_make(
    uint32_t run_id,
    double t_first,
    double period,
    int count
) {
  std::vector<Punch> out;
  if (count <= 0 || period <= 0.0) return out;
  std::mt19937 rng = random_for(run_id, STREAM_PUNCH);
  for (int i = 0; i < count; ++i) {
    Punch p;
    p.time = t_first + period * double(i);
    p.joint = int(random_below(rng, uint32_t(NUM_MOTOR)));
    direction(rng, p.dir);
    const double climbed = std::min((p.time - t_first) / RAMP_S, 1.0);
    const double ceiling =
        FORCE_MAX_N * (RAMP_FLOOR + (1.0 - RAMP_FLOOR) * climbed);
    p.force_n = ceiling * random_between(rng, FORCE_SCALE_MIN, 1.0);
    out.push_back(p);
  }
  if (std::getenv("PUNCHDUMP") != nullptr) {
    for (const Punch& q : out) {
      std::fprintf(
          stderr,
          "PUNCH t=%.4f joint=%d dir=%.5f,%.5f,%.5f f=%.3f\n",
          q.time,
          q.joint,
          q.dir[0],
          q.dir[1],
          q.dir[2],
          q.force_n
      );
    }
  }
  return out;
}

struct Tf {
  double q[4] = {1, 0, 0, 0};
  double p[3] = {0, 0, 0};
};

void qmul(
    const double* a,
    const double* b,
    double* o
) {
  o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

void qrot(
    const double* q,
    const double* v,
    double* o
) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double tx = 2.0 * (y * v[2] - z * v[1]);
  const double ty = 2.0 * (z * v[0] - x * v[2]);
  const double tz = 2.0 * (x * v[1] - y * v[0]);
  o[0] = v[0] + w * tx + (y * tz - z * ty);
  o[1] = v[1] + w * ty + (z * tx - x * tz);
  o[2] = v[2] + w * tz + (x * ty - y * tx);
}

Tf tf_mul(
    const Tf& a,
    const Tf& b
) {
  Tf o;
  qmul(a.q, b.q, o.q);
  double r[3];
  qrot(a.q, b.p, r);
  for (int k = 0; k < 3; ++k) o.p[k] = a.p[k] + r[k];
  return o;
}

Tf pose_in(
    const mjcf::Model& m,
    int anc,
    int body,
    const double* q_of_body
) {
  std::vector<int> chain;
  for (int b = body; b >= 0 && b != anc; b = m.bodies[size_t(b)].parent)
    chain.push_back(b);
  Tf x;
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    const mjcf::Body& bd = m.bodies[size_t(*it)];
    Tf step;
    for (int k = 0; k < 3; ++k) step.p[k] = bd.pos[k];
    for (int k = 0; k < 4; ++k) step.q[k] = bd.quat[k];
    x = tf_mul(x, step);
    for (const mjcf::Joint& j : bd.joints) {
      if (j.is_free()) continue;

      const double a = q_of_body[*it];
      const double n = std::sqrt(
          j.axis[0] * j.axis[0] + j.axis[1] * j.axis[1] + j.axis[2] * j.axis[2]
      );
      const double u[3] = {j.axis[0] / n, j.axis[1] / n, j.axis[2] / n};
      Tf rot;
      rot.q[0] = std::cos(a / 2);
      for (int k = 0; k < 3; ++k) rot.q[k + 1] = u[k] * std::sin(a / 2);
      Tf to, back;
      for (int k = 0; k < 3; ++k) {
        to.p[k] = j.pos[k];
        back.p[k] = -j.pos[k];
      }
      x = tf_mul(x, tf_mul(to, tf_mul(rot, back)));
    }
  }
  return x;
}

struct ArmBounds {
  int torso = -1;
  int hand[2] = {-1, -1};
  double corner[2][8][3] = {};
  double front_m = 0.0;
  double head_m = 0.0;
};

void mesh_corners(
    const std::string& path,
    double out[8][3]
) {
  double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
  for (const mjcf::Vec3& v : mjcf::stl_vertices(path)) {
    const double q[3] = {v[0], v[1], v[2]};
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], q[k]);
      hi[k] = std::max(hi[k], q[k]);
    }
  }
  for (int c = 0; c < 8; ++c) {
    out[c][0] = (c & 1) ? hi[0] : lo[0];
    out[c][1] = (c & 2) ? hi[1] : lo[1];
    out[c][2] = (c & 4) ? hi[2] : lo[2];
  }
}

void hand_box(
    const mjcf::Model& m,
    const ArmBounds& ab,
    const double* q_of_body,
    int side,
    double lo[3],
    double hi[3]
) {
  const Tf x = pose_in(m, ab.torso, ab.hand[side], q_of_body);
  for (int k = 0; k < 3; ++k) {
    lo[k] = 1e30;
    hi[k] = -1e30;
  }
  for (int c = 0; c < 8; ++c) {
    double r[3];
    qrot(x.q, ab.corner[side][c], r);
    for (int k = 0; k < 3; ++k) {
      const double v = x.p[k] + r[k];
      lo[k] = std::min(lo[k], v);
      hi[k] = std::max(hi[k], v);
    }
  }
}

ArmBounds arm_bounds_make(const mjcf::Model& m) {
  ArmBounds ab;
  ab.torso = m.body_index("torso_link");
  ab.hand[0] = m.body_index("left_wrist_yaw_link");
  ab.hand[1] = m.body_index("right_wrist_yaw_link");
  const char* mesh[2] = {"left_rubber_hand", "right_rubber_hand"};
  for (int side = 0; side < 2; ++side) {
    mesh_corners(m.mesh_file.at(mesh[side]).file, ab.corner[side]);
  }

  std::vector<double> q(m.bodies.size(), 0.0);
  {
    int at = 0;
    for (size_t b = 0; b < m.bodies.size(); ++b) {
      for (const mjcf::Joint& j : m.bodies[b].joints) {
        if (j.is_free()) continue;
        q[b] = STANCE[at++];
      }
    }
  }
  ab.front_m = 1e30;
  for (int side = 0; side < 2; ++side) {
    double lo[3], hi[3];
    hand_box(m, ab, q.data(), side, lo, hi);
    ab.front_m = std::min(ab.front_m, lo[0]);
  }

  double clo[8][3];
  mesh_corners(m.mesh_file.at("head_link").file, clo);
  const mjcf::Geom* hg = nullptr;
  for (const mjcf::Geom& g : m.bodies[size_t(ab.torso)].geoms) {
    if (g.mesh == "head_link") {
      hg = &g;
      break;
    }
  }
  ab.head_m = 1e30;
  for (int c = 0; c < 8; ++c) {
    const double z = (hg ? hg->pos[2] : 0.0) + clo[c][2];
    ab.head_m = std::min(ab.head_m, z);
  }
  return ab;
}

void paint_floor_text(
    mjtByte* rgb,
    int width,
    int height,
    const std::vector<std::string>& lines
) {
  if (lines.empty()) return;

  FT_Library library = nullptr;
  if (FT_Init_FreeType(&library) != 0) {
    throw std::runtime_error("floor: cannot initialise freetype");
  }
  FT_Face face = nullptr;
  if (FT_New_Face(library, FLOOR_FONT, 0, &face) != 0) {
    FT_Done_FreeType(library);
    throw std::runtime_error(std::string("floor: cannot read ") + FLOOR_FONT);
  }

  const double em_px = FLOOR_EM_FRACTION * width;
  const double tracking_px = FLOOR_TRACKING_EM * em_px;
  FT_Set_Pixel_Sizes(face, 0, static_cast<FT_UInt>(em_px));

  double lo_x = 0.0, hi_x = 0.0, lo_y = 0.0, hi_y = 0.0;

  auto measure = [&](const std::string& line) {
    double pen = 0.0;
    bool any = false;
    lo_x = hi_x = lo_y = hi_y = 0.0;
    for (const char c : line) {
      FT_Load_Char(face, static_cast<FT_ULong>(c), FT_LOAD_RENDER);
      const FT_GlyphSlot glyph = face->glyph;
      if (glyph->bitmap.width > 0 && glyph->bitmap.rows > 0) {
        const double gx = pen + glyph->bitmap_left;
        const double gy = -glyph->bitmap_top;
        if (!any) {
          lo_x = gx;
          hi_x = gx + glyph->bitmap.width;
          lo_y = gy;
          hi_y = gy + glyph->bitmap.rows;
          any = true;
        } else {
          lo_x = std::min(lo_x, gx);
          hi_x = std::max(hi_x, gx + glyph->bitmap.width);
          lo_y = std::min(lo_y, gy);
          hi_y = std::max(hi_y, gy + glyph->bitmap.rows);
        }
      }
      pen += (glyph->advance.x / 64.0) + tracking_px;
    }
  };

  std::vector<double> centre(lines.size(), 0.0);
  for (size_t i = 0; i < lines.size(); ++i) {
    measure(lines[i]);
    centre[i] = 0.5 * (lo_x + hi_x);
  }
  const double step = em_px * FLOOR_LINE_EM;
  const double top = -0.5 * step * double(lines.size() - 1);
  const double centre_x = 0.5 * width;
  const double centre_y = 0.5 * height;

  for (size_t i = 0; i < lines.size(); ++i) {
    const double lift = top + step * double(i) + 0.5 * em_px * FLOOR_MID_EM;
    double pen = 0.0;
    for (const char c : lines[i]) {
      FT_Load_Char(face, static_cast<FT_ULong>(c), FT_LOAD_RENDER);
      const FT_GlyphSlot glyph = face->glyph;
      const FT_Bitmap& bits = glyph->bitmap;
      for (unsigned int r = 0; r < bits.rows; ++r) {
        for (unsigned int q = 0; q < bits.width; ++q) {
          const unsigned char alpha =
              bits.buffer
                  [static_cast<int>(r) * bits.pitch + static_cast<int>(q)];
          if (alpha == 0) continue;
          const double sx = pen + glyph->bitmap_left + static_cast<int>(q);
          const double sy = -glyph->bitmap_top + static_cast<int>(r);
          const long tx = std::lround(centre_x + (sy + lift));
          const long ty = std::lround(centre_y - (sx - centre[i]));
          if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;
          mjtByte* pixel =
              rgb + (static_cast<size_t>(ty) * width + tx) * RGB_CHANNELS;
          const float wash = alpha / 255.0f;
          for (int channel = 0; channel < RGB_CHANNELS; ++channel) {
            const float ground = pixel[channel];
            pixel[channel] =
                static_cast<mjtByte>(ground + (255.0f - ground) * wash + 0.5f);
          }
        }
      }
      pen += (glyph->advance.x / 64.0) + tracking_px;
    }
  }

  FT_Done_Face(face);
  FT_Done_FreeType(library);
}

void paint_floor_texture(
    mjModel* m,
    const std::vector<std::string>& lines
) {
  const int texture = mj_name2id(m, mjOBJ_TEXTURE, FLOOR_TEXTURE);
  if (texture < 0) return;
  if (m->tex_nchannel[texture] != RGB_CHANNELS) {
    throw std::runtime_error("floor: the label texture is not RGB");
  }
  paint_floor_text(
      m->tex_data + m->tex_adr[texture],
      m->tex_width[texture],
      m->tex_height[texture],
      lines
  );
}

struct Preview {
  mjModel* m = nullptr;
  mjData* d = nullptr;
  GLFWwindow* window = nullptr;
  mjvCamera camera;
  mjvOption option;
  mjvScene scene;
  mjrContext context;
  std::vector<int> qpos_adr;
  std::vector<int> jnt_id;
  int base_qpos = 0;
  int target_mocap = -1;
  int heading_mocap = -1;

  ~Preview() {
    if (d) mj_deleteData(d);
    if (m) mj_deleteModel(m);
    if (window) glfwDestroyWindow(window);
  }
};

std::unique_ptr<Preview> preview_open(
    const mjcf::Model& model,
    const std::string& path,
    const std::vector<std::string>& label,
    bool visible
) {
  auto v = std::make_unique<Preview>();
  char err[1000] = "";
  v->m = mj_loadXML(path.c_str(), nullptr, err, sizeof(err));
  if (v->m == nullptr) throw std::runtime_error(std::string("preview: ") + err);
  v->d = mj_makeData(v->m);

  int at = 0;
  for (const mjcf::Body& b : model.bodies) {
    for (const mjcf::Joint& j : b.joints) {
      if (j.is_free()) {
        v->base_qpos =
            v->m->jnt_qposadr[mj_name2id(v->m, mjOBJ_JOINT, j.name.c_str())];
        continue;
      }
      const int id = mj_name2id(v->m, mjOBJ_JOINT, j.name.c_str());
      if (id < 0)
        throw std::runtime_error("preview: MuJoCo has no joint " + j.name);
      v->qpos_adr.push_back(v->m->jnt_qposadr[id]);
      v->jnt_id.push_back(id);
      ++at;
    }
  }
  const int tgt = mj_name2id(v->m, mjOBJ_BODY, "target");
  const int hdg = mj_name2id(v->m, mjOBJ_BODY, "target_heading");
  v->target_mocap = tgt >= 0 ? v->m->body_mocapid[tgt] : -1;
  v->heading_mocap = hdg >= 0 ? v->m->body_mocapid[hdg] : -1;

  paint_floor_texture(v->m, label);

  if (!glfwInit()) throw std::runtime_error("preview: glfwInit failed");

  glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
  v->window = glfwCreateWindow(
      1200,
      800,
      "teleop-walking-benchmark (PhysX)",
      nullptr,
      nullptr
  );
  if (v->window == nullptr) throw std::runtime_error("preview: no GLFW window");
  glfwMakeContextCurrent(v->window);
  glfwSwapInterval(0);
  mjv_defaultCamera(&v->camera);
  mjv_defaultOption(&v->option);
  mjv_defaultScene(&v->scene);
  mjr_defaultContext(&v->context);
  mjv_makeScene(v->m, &v->scene, 2000);
  v->m->vis.global.offwidth = RECORD_WIDTH;
  v->m->vis.global.offheight = RECORD_HEIGHT;
  mjr_makeContext(v->m, &v->context, mjFONTSCALE_150);

  const double design = -45.0 * M_PI / 180.0;
  const double ray = 2.0 / -std::sin(design) - 1.0;
  const double back = ray * std::cos(design);
  const double rise = ray * -std::sin(design) + 1.0;
  v->camera.type = mjCAMERA_FREE;
  v->camera.azimuth = 180.0;
  v->camera.elevation = -std::atan2(rise, back) * 180.0 / M_PI;
  v->camera.distance = std::hypot(back, rise);
  v->camera.lookat[0] = CENTRE_X_M;
  v->camera.lookat[1] = CENTRE_Y_M;
  v->camera.lookat[2] = 0.0;
  v->m->vis.global.fovy = 65.0;
  return v;
}

struct Recorder {
  FILE* pipe = nullptr;
  int width = RECORD_WIDTH;
  int height = RECORD_HEIGHT;
  std::vector<unsigned char> rgb;
  long frames = 0;
  std::string path;
  std::string partial;
};

std::unique_ptr<Recorder> recorder_open(const std::string& path) {
  auto r = std::make_unique<Recorder>();
  r->path = path;
  r->partial = path + ".tmp";
  r->rgb.assign(size_t(r->width) * size_t(r->height) * 3, 0);
  std::ostringstream cmd;
  cmd << "ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt rgb24"
      << " -s " << r->width << "x" << r->height << " -r " << RECORD_FPS
      << " -i - -vf vflip -c:v libx264 -preset veryfast -crf 20"
      << " -pix_fmt yuv420p -f mp4 '" << r->partial << "'";
  r->pipe = popen(cmd.str().c_str(), "w");
  if (r->pipe == nullptr)
    throw std::runtime_error("record: cannot start ffmpeg for " + path);
  return r;
}

void preview_punch_arrow(
    Preview& v,
    const Punch& p
) {
  if (v.scene.ngeom >= v.scene.maxgeom) return;
  if (p.joint < 0 || size_t(p.joint) >= v.jnt_id.size()) return;
  const double* to = v.d->xanchor + 3 * v.jnt_id[size_t(p.joint)];
  const double scale = p.force_n / FORCE_MAX_N;
  const double length = PUNCH_ARROW_SCALE * (0.15 + 0.45 * scale);
  double axis_z[3] = {p.dir[0], p.dir[1], p.dir[2]};
  const double axis_len = std::sqrt(
      axis_z[0] * axis_z[0] + axis_z[1] * axis_z[1] + axis_z[2] * axis_z[2]
  );
  if (axis_len < 1e-9) return;
  for (int k = 0; k < 3; ++k) axis_z[k] /= axis_len;

  const double seed[3] = {
      std::fabs(axis_z[2]) < 0.9 ? 0.0 : 1.0,
      0.0,
      std::fabs(axis_z[2]) < 0.9 ? 1.0 : 0.0
  };
  double axis_x[3] = {
      seed[1] * axis_z[2] - seed[2] * axis_z[1],
      seed[2] * axis_z[0] - seed[0] * axis_z[2],
      seed[0] * axis_z[1] - seed[1] * axis_z[0]
  };
  const double x_len = std::sqrt(
      axis_x[0] * axis_x[0] + axis_x[1] * axis_x[1] + axis_x[2] * axis_x[2]
  );
  for (int k = 0; k < 3; ++k) axis_x[k] /= x_len;
  const double axis_y[3] = {
      axis_z[1] * axis_x[2] - axis_z[2] * axis_x[1],
      axis_z[2] * axis_x[0] - axis_z[0] * axis_x[2],
      axis_z[0] * axis_x[1] - axis_z[1] * axis_x[0]
  };

  const double width = PUNCH_ARROW_SCALE * (0.008 + 0.022 * scale);
  const double size[3] = {width, width, length};
  const double mat[9] = {
      axis_x[0],
      axis_y[0],
      axis_z[0],
      axis_x[1],
      axis_y[1],
      axis_z[1],
      axis_x[2],
      axis_y[2],
      axis_z[2]
  };
  const float rgba[4] = {1.0f, 0.35f, 0.15f, 0.9f};

  const double centre[3] = {
      to[0] - axis_z[0] * 0.5 * length,
      to[1] - axis_z[1] * 0.5 * length,
      to[2] - axis_z[2] * 0.5 * length
  };

  mjvGeom* g = &v.scene.geoms[v.scene.ngeom];
  mjv_initGeom(g, mjGEOM_ARROW, size, centre, mat, rgba);
  ++v.scene.ngeom;
}

const Punch* punch_active(
    const std::vector<Punch>& punches,
    double t
) {
  for (const Punch& p : punches) {
    if (t >= p.time && t < p.time + PUNCH_DURATION_S + PUNCH_HOLD_S) return &p;
  }
  return nullptr;
}

void recorder_frame(
    Recorder& r,
    Preview& v,
    const Punch* active
) {
  const mjrRect viewport = {0, 0, r.width, r.height};
  mjv_updateScene(v.m, v.d, &v.option, nullptr, &v.camera, mjCAT_ALL, &v.scene);
  if (active != nullptr) preview_punch_arrow(v, *active);
  mjr_setBuffer(mjFB_OFFSCREEN, &v.context);
  mjr_render(viewport, &v.scene, &v.context);
  mjr_readPixels(r.rgb.data(), nullptr, viewport, &v.context);
  mjr_setBuffer(mjFB_WINDOW, &v.context);
  if (std::fwrite(r.rgb.data(), 1, r.rgb.size(), r.pipe) != r.rgb.size()) {
    throw std::runtime_error("record: ffmpeg stopped accepting frames");
  }
  ++r.frames;
}

void recorder_close(std::unique_ptr<Recorder> r) {
  if (!r) return;
  const bool encoded = r->pipe == nullptr || pclose(r->pipe) == 0;
  r->pipe = nullptr;
  if (!encoded) {
    std::fprintf(stderr, "record: ffmpeg failed for %s\n", r->path.c_str());
    return;
  }
  std::error_code ec;
  std::filesystem::rename(r->partial, r->path, ec);
  if (ec) {
    std::fprintf(
        stderr,
        "record: cannot name %s: %s\n",
        r->path.c_str(),
        ec.message().c_str()
    );
    return;
  }
  std::printf(
      "record: wrote %s (%ld frames, %.2f s at %d fps)\n",
      r->path.c_str(),
      r->frames,
      double(r->frames) / RECORD_FPS,
      RECORD_FPS
  );
}

void preview_sync(
    Preview& v,
    const float* root,
    const float* motor_q
) {
  v.d->qpos[v.base_qpos + 0] = root[4];
  v.d->qpos[v.base_qpos + 1] = root[5];
  v.d->qpos[v.base_qpos + 2] = root[6];
  v.d->qpos[v.base_qpos + 3] = root[3];
  v.d->qpos[v.base_qpos + 4] = root[0];
  v.d->qpos[v.base_qpos + 5] = root[1];
  v.d->qpos[v.base_qpos + 6] = root[2];
  for (size_t i = 0; i < v.qpos_adr.size(); ++i)
    v.d->qpos[v.qpos_adr[i]] = motor_q[i];
  mj_forward(v.m, v.d);
}

void preview_markers(
    Preview& v,
    bool has_target,
    double tx,
    double ty,
    double tyaw
) {
  const double away[3] = {0.0, 0.0, -5.0};
  if (v.target_mocap >= 0) {
    const double* p = has_target ? nullptr : away;
    v.d->mocap_pos[3 * v.target_mocap + 0] = p ? p[0] : tx;
    v.d->mocap_pos[3 * v.target_mocap + 1] = p ? p[1] : ty;
    v.d->mocap_pos[3 * v.target_mocap + 2] = p ? p[2] : 0.0;
  }
  if (v.heading_mocap >= 0) {
    v.d->mocap_pos[3 * v.heading_mocap + 0] = has_target ? tx : away[0];
    v.d->mocap_pos[3 * v.heading_mocap + 1] = has_target ? ty : away[1];
    v.d->mocap_pos[3 * v.heading_mocap + 2] = has_target ? 0.0 : away[2];
    v.d->mocap_quat[4 * v.heading_mocap + 0] = std::cos(tyaw / 2);
    v.d->mocap_quat[4 * v.heading_mocap + 1] = 0.0;
    v.d->mocap_quat[4 * v.heading_mocap + 2] = 0.0;
    v.d->mocap_quat[4 * v.heading_mocap + 3] = std::sin(tyaw / 2);
  }
}

bool preview_draw(
    Preview& v,
    const std::string& overlay,
    const Punch* active
) {
  mjrRect viewport = {0, 0, 0, 0};
  glfwGetFramebufferSize(v.window, &viewport.width, &viewport.height);
  mjv_updateScene(v.m, v.d, &v.option, nullptr, &v.camera, mjCAT_ALL, &v.scene);
  if (active != nullptr) preview_punch_arrow(v, *active);
  mjr_render(viewport, &v.scene, &v.context);
  mjr_overlay(
      mjFONT_NORMAL,
      mjGRID_TOPLEFT,
      viewport,
      overlay.c_str(),
      nullptr,
      &v.context
  );
  glfwSwapBuffers(v.window);
  glfwPollEvents();
  return !glfwWindowShouldClose(v.window);
}

double heading_of(const float* q) {
  const double x = q[0], y = q[1], z = q[2], w = q[3];
  return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

}

void close_segment(
    Run& r,
    int env,
    int segment,
    const float* energy,
    const float* vibration
) {
  if (segment < 0 || segment >= WAYPOINTS) return;
  for (int g = 0; g < PHYS_GROUPS; ++g) {
    const size_t at = size_t(env) * PHYS_GROUPS + size_t(g);
    r.seg_energy[segment][g] = energy[at] - r.taken_energy[g];
    r.seg_vibration[segment][g] = vibration[at] - r.taken_vibration[g];
    r.taken_energy[g] = energy[at];
    r.taken_vibration[g] = vibration[at];
  }
}

void upload_floats(
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

int run(
    int argc,
    char** argv
) {
  int envs = 256;
  uint32_t runid0 = 0;
  std::string csv = "results/result.csv";
  double walk_s = WALK_S;
  double init_s = INIT_DURATION_S;
  double realtime = 0.0;
  bool preview_on = false;
  std::string record_dir;
  std::string policy_name = "gr00t_wbc_h074_p000";
  PhysicsEngine engine = PhysicsEngine::kPhysx;
  int threads = int(std::thread::hardware_concurrency());
  if (threads < 1) threads = 1;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--runs") && i + 1 < argc)
      envs = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--runid") && i + 1 < argc)
      runid0 = uint32_t(std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc)
      csv = argv[++i];
    else if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc)
      walk_s = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--init") && i + 1 < argc)
      init_s = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc)
      threads = std::max(1, std::atoi(argv[++i]));
    else if (!std::strcmp(argv[i], "--engine") && i + 1 < argc) {
      const std::string want = argv[++i];
      if (want == "physx") {
        engine = PhysicsEngine::kPhysx;
      } else if (want == "mujoco") {
        engine = PhysicsEngine::kMujoco;
      } else {
        std::fprintf(stderr, "unknown --engine '%s'\n", want.c_str());
        return 2;
      }
    } else if (!std::strcmp(argv[i], "--realtime") && i + 1 < argc)
      realtime = std::atof(argv[++i]);
    else if (!std::strcmp(argv[i], "--preview"))
      preview_on = true;
    else if (!std::strcmp(argv[i], "--record") && i + 1 < argc)
      record_dir = argv[++i];
    else if (!std::strcmp(argv[i], "--policy") && i + 1 < argc)
      policy_name = argv[++i];
    else if (!std::strcmp(argv[i], "--runids") && i + 1 < argc) {
      const char* r = argv[++i];
      const char* dash = std::strchr(r, '-');
      runid0 = uint32_t(std::atoi(r));
      envs = dash ? std::atoi(dash + 1) - int(runid0) + 1 : 1;
      if (envs < 1) {
        std::fprintf(stderr, "--runids %s is empty\n", r);
        return 1;
      }
    } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      std::printf(
          "usage: teleop-walking-benchmark [options]\n"
          "  --engine NAME     mujoco or physx (default physx)\n"
          "  --threads N       mujoco fleet shards (default: cores)\n"
          "  --policy NAME     which candidate to drive (default "
          "gr00t_wbc_h074_p000)\n"
          "  --runs N          fleet size, one robot per run id (default 256)\n"
          "  --runid N         first run id (default 0)\n"
          "  --runids A-B      the same, as a closed range\n"
          "  --seconds S       walking time per run (default %g)\n"
          "  --init S          settling time before the policy is engaged\n"
          "  --csv PATH        where to write the per-run rows\n"
          "  --realtime R      pace the run at R times the wall clock\n"
          "  --preview         watch one robot in a window\n"
          "  --record DIR      write DIR/NAME.mp4 for one robot\n"
          "  --help            this\n",
          WALK_S
      );
      return 0;
    }
  }

  if (preview_on || !record_dir.empty()) {
    if (envs != 1)
      std::printf("preview: showing one robot (--runs forced to 1)\n");
    envs = 1;
    if (preview_on && realtime <= 0.0) realtime = 1.0;
  }
  const std::string model_path = "assets/g1_29dof.xml";
  const mjcf::Model model = mjcf::load(model_path);
  std::unique_ptr<Physics> phys =
      engine == PhysicsEngine::kPhysx
          ? physics_make_physx(model, envs, 4.0)
          : physics_make_mujoco(model, model_path, envs, 4.0, threads);

  std::unique_ptr<policy_api::Policy> pol = make_policy(policy_name);
  pol->init(envs);
  const policy_api::Limits LIM = pol->limits();

  float kp[NUM_MOTOR], kd[NUM_MOTOR];
  for (int i = 0; i < NUM_MOTOR; ++i) {
    kp[i] = float(ARMATURE[i] * WN * WN);
    kd[i] = float(2.0 * ZETA * ARMATURE[i] * WN);
  }

  float crane_kp[NUM_MOTOR], crane_kd[NUM_MOTOR];
  std::memcpy(crane_kp, kp, sizeof(kp));
  std::memcpy(crane_kd, kd, sizeof(kd));
  for (int i = 0; i < pol->owned(); ++i) {
    kp[i] = pol->kp()[i];
    kd[i] = pol->kd()[i];
  }

  phys->set_gains(crane_kp, crane_kd);
  bool gains_released = false;

  float *d_cmd, *d_arm_pose, *d_task, *d_base_quat;
  cudaMalloc(&d_cmd, size_t(envs) * 3 * sizeof(float));
  cudaMalloc(&d_task, size_t(envs) * 4 * sizeof(float));
  cudaMalloc(&d_base_quat, size_t(envs) * 4 * sizeof(float));
  cudaMalloc(&d_arm_pose, size_t(envs) * NUM_MOTOR * sizeof(float));

  const ArmBounds arm_bounds = arm_bounds_make(model);

  int arm_body[2][ARM_DOF] = {};
  {
    int at = 0;
    for (size_t b = 0; b < model.bodies.size(); ++b) {
      for (const mjcf::Joint& j : model.bodies[b].joints) {
        if (j.is_free()) continue;
        if (at >= ARM_LEFT_FIRST && at < ARM_LEFT_FIRST + ARM_DOF)
          arm_body[0][at - ARM_LEFT_FIRST] = int(b);
        if (at >= ARM_RIGHT_FIRST && at < ARM_RIGHT_FIRST + ARM_DOF)
          arm_body[1][at - ARM_RIGHT_FIRST] = int(b);
        ++at;
      }
    }
  }
  std::vector<double> q_of_body(model.bodies.size(), 0.0);
  std::vector<Run> runs(static_cast<size_t>(envs));
  for (int e = 0; e < envs; ++e) {
    Run& r = runs[size_t(e)];
    r.run_id = runid0 + uint32_t(e);
    r.tour = tour_make(r.run_id);
    r.arm_rng = random_for(r.run_id, STREAM_ARM);

    if (getenv("NOPUNCH") == nullptr)
      r.punches = schedule_make(r.run_id, PUNCH_DELAY_S, POINT_S, WAYPOINTS);
    for (int i = 0; i < ARM_DOF; ++i)
      r.arm_left[i] = STANCE[ARM_LEFT_FIRST + i];
  }

  for (int i = 0; i < ARM_DOF; ++i) {
    int found = 0;
    for (const mjcf::Body& b : model.bodies) {
      for (const mjcf::Joint& j : b.joints) {
        if (j.is_free()) continue;
        if (found == ARM_LEFT_FIRST + i) {
          for (Run& r : runs) {
            r.arm_lo[i] = j.range[0];
            r.arm_hi[i] = j.range[1];
          }
        }
        ++found;
      }
    }
  }

  std::unique_ptr<Preview> preview;
  std::unique_ptr<Recorder> recorder;
  if (preview_on || !record_dir.empty()) {
    const std::vector<std::string> label = {
        engine == PhysicsEngine::kMujoco ? "mujoco" : "physx",
        policy_name
    };
    preview = preview_open(model, "assets/g1_29dof.xml", label, preview_on);
  }
  if (!record_dir.empty()) {
    std::filesystem::create_directories(record_dir);
    recorder = recorder_open(record_dir + "/" + policy_name + ".mp4");
  }

  phys->reset(STANCE, 0.793);

  std::vector<PhysicsPunch> h_punch(static_cast<size_t>(envs));
  std::vector<float> h_cmd(size_t(envs) * 3, 0.0f);
  std::vector<float> h_task(size_t(envs) * 4, 0.0f);
  std::vector<float> h_quat(size_t(envs) * 4, 0.0f);
  std::vector<float> h_arm(size_t(envs) * NUM_MOTOR);

  double view_tx = 0.0, view_ty = 0.0, view_tyaw = 0.0;
  bool view_has = false;
  std::vector<float> h_mq(size_t(envs) * NUM_MOTOR);
  const double dt = phys->timestep();
  const int per_control = std::max(1, int(std::lround(PERIOD_S / dt)));
  const double total_s = init_s + walk_s;
  const int steps = int(total_s / dt);
  const auto t0 = std::chrono::steady_clock::now();
  double t_read = 0, t_host = 0, t_infer = 0, t_targets = 0, t_step = 0,
         t_other = 0;
  auto last_report = t0;
  double reported_at = 0.0;
  auto tick = [] { return std::chrono::steady_clock::now(); };
  auto secs = [](auto a, auto b) {
    return std::chrono::duration<double>(b - a).count();
  };

  for (int s = 0; s < steps; ++s) {
    const double now = double(s) * dt;
    const double elapsed = now - init_s;
    const bool control = (s % per_control) == 0;

    {
      auto a = tick();
      if (control) phys->read();
      t_read += secs(a, tick());
    }

    if (control) {
      auto host_a = tick();
      const float* pose = phys->base_pose();
      const unsigned char* live = phys->alive();
      const float* speed = phys->body_speed();
      const float* cost_e = phys->energy();
      const float* cost_v = phys->vibration();

      view_has = false;
      for (int e = 0; e < envs; ++e) {
        Run& r = runs[size_t(e)];
        r.alive = live[size_t(e)] != 0;
        if (r.alive) {
          r.pelvis_speed = speed[size_t(e) * 2 + 0];
          r.head_speed = speed[size_t(e) * 2 + 1];
        }

        const float* p = pose + size_t(e) * 7;
        const double wx = p[4], wy = p[5], wyaw = heading_of(p);

        const bool arms_walk = r.released_once;
        if (elapsed >= 0.0) r.released_once = true;
        if (arms_walk && r.alive) {
          const int draws = ARM_DRAWS;
          for (int draw = 0; draw < draws; ++draw) {
            double cand[ARM_DOF];
            for (int i = 0; i < ARM_DOF; ++i) {
              cand[i] = std::clamp(
                  r.arm_left[i] +
                      ARM_STEP_RAD * random_between(r.arm_rng, -1.0, 1.0),
                  r.arm_lo[i],
                  r.arm_hi[i]
              );
            }

            for (int i = 0; i < ARM_DOF; ++i) {
              q_of_body[size_t(arm_body[0][i])] = cand[i];
              q_of_body[size_t(arm_body[1][i])] = ARM_MIRROR[i] * cand[i];
            }
            bool ok = true;
            for (int side = 0; side < 2 && ok; ++side) {
              double blo[3], bhi[3];
              hand_box(model, arm_bounds, q_of_body.data(), side, blo, bhi);
              if (blo[0] < arm_bounds.front_m || bhi[2] > arm_bounds.head_m)
                ok = false;
            }
            if (!ok && draws > 1) continue;
            for (int i = 0; i < ARM_DOF; ++i) r.arm_left[i] = cand[i];
            break;
          }
        }
        float* arm = h_arm.data() + size_t(e) * NUM_MOTOR;
        for (int j = 0; j < NUM_MOTOR; ++j) arm[j] = float(STANCE[j]);
        for (int i = 0; i < ARM_DOF; ++i) {
          arm[ARM_LEFT_FIRST + i] = float(r.arm_left[i]);
          arm[ARM_RIGHT_FIRST + i] = float(ARM_MIRROR[i] * r.arm_left[i]);
        }

        h_punch[size_t(e)] = PhysicsPunch{};
        if (elapsed >= 0.0 && r.alive) {
          for (const Punch& pu : r.punches) {
            if (elapsed < pu.time || elapsed >= pu.time + PUNCH_DURATION_S)
              continue;
            h_punch[size_t(e)].joint = pu.joint;
            for (int k = 0; k < 3; ++k) {
              h_punch[size_t(e)].force[k] = float(pu.dir[k] * pu.force_n);
            }
            break;
          }
        }

        h_quat[size_t(e) * 4 + 0] = p[3];
        h_quat[size_t(e) * 4 + 1] = p[0];
        h_quat[size_t(e) * 4 + 2] = p[1];
        h_quat[size_t(e) * 4 + 3] = p[2];

        double cmd[3] = {0.0, 0.0, 0.0};
        if (elapsed >= 0.0 && r.alive) {
          if (!r.anchored) {
            r.anchored = true;
            r.ax = wx;
            r.ay = wy;
            r.ayaw = wyaw;
          }
          const int index = int(elapsed / POINT_S);
          if (index != r.current_target) {
            if (r.current_target >= 0 && r.have_last) {
              r.pos_err_sum += r.last_dist * 100.0;
              r.yaw_err_sum += std::fabs(r.last_yaw_err) * 180.0 / M_PI;
              ++r.scored;
              close_segment(r, e, r.current_target, cost_e, cost_v);
            }
            r.current_target = index;
            r.pos_reached = false;
            r.yaw_reached = false;
          }
          for (int k = 0; k < 4; ++k) h_task[size_t(e) * 4 + k] = 0.0f;
          if (index >= 0 && index < int(r.tour.size())) {
            const Waypoint& t = r.tour[size_t(index)];
            const double tx =
                r.ax + t.x * std::cos(r.ayaw) - t.y * std::sin(r.ayaw);
            const double ty =
                r.ay + t.x * std::sin(r.ayaw) + t.y * std::cos(r.ayaw);
            const double tyaw = r.ayaw + t.yaw;
            if (e == 0) {
              view_tx = tx;
              view_ty = ty;
              view_tyaw = tyaw;
              view_has = true;
            }
            const double dx = tx - wx, dy = ty - wy;
            const double dist = std::hypot(dx, dy);
            const double yaw_err = std::remainder(tyaw - wyaw, 2.0 * M_PI);
            r.last_dist = dist;
            r.last_yaw_err = yaw_err;
            r.have_last = true;

            h_task[size_t(e) * 4 + 0] = float(dist);
            h_task[size_t(e) * 4 + 1] = float(yaw_err);

            if (r.pos_reached) {
              if (dist > LIM.pos_reached_exit_m) r.pos_reached = false;
            } else {
              if (dist < LIM.pos_reached_enter_m) r.pos_reached = true;
            }
            if (r.yaw_reached) {
              if (std::fabs(yaw_err) > LIM.yaw_reached_exit_rad)
                r.yaw_reached = false;
            } else {
              if (std::fabs(yaw_err) < LIM.yaw_reached_enter_rad)
                r.yaw_reached = true;
            }

            const double cy = std::cos(wyaw), sy = std::sin(wyaw);
            h_task[size_t(e) * 4 + 2] = float(cy * dx + sy * dy);
            h_task[size_t(e) * 4 + 3] = float(-sy * dx + cy * dy);
            if (!r.pos_reached) {
              cmd[0] = LIM.walk_kp_pos * (cy * dx + sy * dy);
              cmd[1] = LIM.walk_kp_pos * (-sy * dx + cy * dy);
            }
            if (!r.yaw_reached) cmd[2] = LIM.walk_kp_yaw * yaw_err;

            const double norm = std::hypot(cmd[0], cmd[1]);
            if (LIM.speed_norm > 0.0 && norm > LIM.speed_norm) {
              cmd[0] *= LIM.speed_norm / norm;
              cmd[1] *= LIM.speed_norm / norm;
            }
            cmd[0] = std::clamp(cmd[0], LIM.vx_min, LIM.vx_max);
            cmd[1] = std::clamp(cmd[1], -LIM.vy_abs, LIM.vy_abs);
            cmd[2] = std::clamp(cmd[2], -LIM.yaw_rate_abs, LIM.yaw_rate_abs);
          }
        }
        for (int k = 0; k < 3; ++k) h_cmd[size_t(e) * 3 + k] = float(cmd[k]);
      }

      if (getenv("TRACE") != nullptr && elapsed >= 0.0 &&
          elapsed <= 1.0 + 1e-9) {
        static bool header = false;
        if (!header) {
          std::printf("t,px,py,pz,qw,qx,qy,qz");
          for (int i = 0; i < NUM_MOTOR; ++i) std::printf(",q%d", i);
          for (int i = 0; i < NUM_MOTOR; ++i) std::printf(",tq%d", i);
          std::printf(",lfz,rfz\n");
          header = true;
        }
        std::vector<float> mq(size_t(envs) * NUM_MOTOR),
            tq(size_t(envs) * NUM_MOTOR);
        cudaMemcpy(
            mq.data(),
            phys->motor_q(),
            mq.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        cudaMemcpy(
            tq.data(),
            phys->q_target(),
            tq.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        );
        const float* p0 = phys->base_pose();

        std::printf(
            "%.4f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
            elapsed,
            p0[4],
            p0[5],
            p0[6],
            p0[3],
            p0[0],
            p0[1],
            p0[2]
        );
        for (int i = 0; i < NUM_MOTOR; ++i) std::printf(",%.6f", mq[i]);
        for (int i = 0; i < NUM_MOTOR; ++i) std::printf(",%.6f", tq[i]);
        std::printf(
            ",%.6f,%.6f",
            phys->foot_height()[0],
            phys->foot_height()[1]
        );
        std::printf("\n");
      }
      phys->set_punches(h_punch.data());
      upload_floats(h_cmd, d_cmd);
      upload_floats(h_task, d_task);
      upload_floats(h_quat, d_base_quat);
      upload_floats(h_arm, d_arm_pose);
      t_host += secs(host_a, tick());

      auto inf_a = tick();
      if (elapsed >= 0.0 && !gains_released) {
        gains_released = true;
        phys->set_gains(kp, kd);
      }
      if (elapsed >= 0.0) {
        const policy_api::Ctx pctx{
            phys->motor_q(),
            phys->motor_dq(),
            phys->gyro(),
            phys->gravity(),
            d_cmd,
            d_task,
            d_arm_pose,
            d_base_quat,
            phys->q_target()
        };
        pol->step(pctx);
      } else {
        upload_floats(h_arm, phys->q_target());
      }
      t_infer += secs(inf_a, tick());
    }

    if (elapsed < 0.0) phys->hold_base(0.793);
    if (control) {
      auto a = tick();
      phys->apply_targets();
      t_targets += secs(a, tick());
    }
    {
      auto a = tick();
      phys->step(1);
      t_step += secs(a, tick());
    }

    if (preview && control) {
      cudaMemcpy(
          h_mq.data(),
          phys->motor_q(),
          h_mq.size() * sizeof(float),
          cudaMemcpyDeviceToHost
      );
      const bool showing = preview_on;
      preview_markers(*preview, view_has, view_tx, view_ty, view_tyaw);
      preview_sync(*preview, phys->base_pose(), h_mq.data());
      const Punch* active = elapsed >= 0.0 && runs[0].alive
                                ? punch_active(runs[0].punches, elapsed)
                                : nullptr;
      char hud[256];
      std::snprintf(
          hud,
          sizeof(hud),
          "t %.2f s\ntarget %d\ndist %.2f m\n%s",
          elapsed,
          runs[0].current_target,
          runs[0].have_last ? runs[0].last_dist : 0.0,
          runs[0].alive ? "" : "FELL"
      );
      if (recorder) recorder_frame(*recorder, *preview, active);
      if (showing && !preview_draw(*preview, hud, active)) break;
    }

    if (realtime > 0.0) {
      const double target_wall = (now + dt) / realtime;
      const double elapsed_wall = secs(t0, tick());
      if (target_wall > elapsed_wall) {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(target_wall - elapsed_wall)
        );
      }
    }

    if (control && elapsed >= 0.0) {
      const float* pose = phys->base_pose();
      const unsigned char* live = phys->alive();
      int standing = 0;
      for (int e = 0; e < envs; ++e) {
        if (live[size_t(e)] == 0) continue;
        if (pose[size_t(e) * 7 + 6] < FALL_PELVIS_Z) {
          phys->kill(e);
          runs[size_t(e)].fell_at = now;
          continue;
        }
        ++standing;
      }
      if (standing == 0) break;
    }

    if (control && elapsed >= 0.0) phys->meter(PERIOD_S);

    if (control && elapsed >= 0.0) {
      const int extra = std::min(per_control - 1, steps - 1 - s);
      if (extra > 0) {
        auto a = tick();
        phys->step(extra);
        t_step += secs(a, tick());
        s += extra;
      }
    }

    {
      const double done = double(s + 1) * dt;
      if (done - reported_at >= PROGRESS_S) {
        const double since = secs(last_report, tick());
        std::printf(
            "progress: sim %6.2f s of %6.2f   %7.0f robot-sim-s/s\n",
            done,
            total_s,
            since > 0.0 ? double(envs) * (done - reported_at) / since : 0.0
        );
        std::fflush(stdout);
        reported_at = done;
        last_report = tick();
      }
    }
  }

  const double wall =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  t_other = wall - t_read - t_host - t_infer - t_targets - t_step;
  std::printf(
      "timing: read %.1fs  host %.1fs  infer %.1fs  targets %.1fs  step %.1fs  "
      "other %.1fs  total %.1fs\n",
      t_read,
      t_host,
      t_infer,
      t_targets,
      t_step,
      t_other,
      wall
  );

  const unsigned char* h_alive = phys->alive();
  const float* final_e = phys->energy();
  const float* final_v = phys->vibration();
  for (int e = 0; e < envs; ++e) {
    Run& r = runs[size_t(e)];
    if (r.have_last && r.current_target >= 0) {
      r.pos_err_sum += r.last_dist * 100.0;
      r.yaw_err_sum += std::fabs(r.last_yaw_err) * 180.0 / M_PI;
      ++r.scored;
    }
    close_segment(r, e, r.current_target, final_e, final_v);
    r.alive = h_alive[size_t(e)] != 0;
  }

  std::filesystem::create_directories(std::filesystem::path(csv).parent_path());
  std::ofstream out(csv);

  static const char* const COST_GROUPS[PHYS_GROUPS] =
      {"legs_upper", "legs_lower", "waist", "arms_upper", "arms_lower"};
  out << "policy,runid,outcome,survival_s,targets,pos_err_cm,yaw_err_deg"
      << ",pelvis_speed_mps,head_speed_mps";
  for (int i = 0; i < WAYPOINTS; ++i) {
    for (const char* g : COST_GROUPS) out << ",s" << i << "_e_" << g << "_j";
    for (const char* g : COST_GROUPS) {
      out << ",s" << i << "_v_" << g << "_krads2";
    }
  }
  out << '\n';
  int complete = 0;
  double err_sum = 0.0;
  int err_n = 0;
  for (const Run& r : runs) {
    const bool done = r.alive;
    if (done) ++complete;
    const double surv = done ? walk_s : std::max(0.0, r.fell_at - init_s);
    out << policy_name << ',' << r.run_id << ',' << (done ? "complete" : "fell")
        << ',' << surv << ',' << r.scored << ','
        << (r.scored ? r.pos_err_sum / r.scored : 0.0) << ','
        << (r.scored ? r.yaw_err_sum / r.scored : 0.0) << ',' << r.pelvis_speed
        << ',' << r.head_speed;
    for (int i = 0; i < WAYPOINTS; ++i) {
      const bool reached = i < r.scored;
      for (int g = 0; g < PHYS_GROUPS; ++g) {
        out << ',';
        if (reached) out << r.seg_energy[i][g];
      }
      for (int g = 0; g < PHYS_GROUPS; ++g) {
        out << ',';
        if (reached) out << r.seg_vibration[i][g] * 1e-3;
      }
    }
    out << '\n';
    if (done && r.scored) {
      err_sum += r.pos_err_sum / r.scored;
      ++err_n;
    }
  }
  std::printf(
      "\n%d runs, %d complete (%.0f%%), %.1f s wall, %.1fx realtime\n",
      envs,
      complete,
      100.0 * complete / envs,
      wall,
      envs * total_s / wall
  );
  if (err_n)
    std::printf(
        "mean position error over finishers: %.2f cm\n",
        err_sum / err_n
    );
  recorder_close(std::move(recorder));
  std::printf("wrote %s\n", csv.c_str());
  return 0;
}

void mujoco_warning(const char* text) {
  std::fprintf(stderr, "mujoco: %s\n", text);
}

void mujoco_error(const char* text) {
  std::fprintf(stderr, "mujoco: %s\n", text);
  std::exit(1);
}

void reserve_system_core() {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) return;
  if (CPU_COUNT(&set) <= 1 || !CPU_ISSET(0, &set)) return;
  CPU_CLR(0, &set);
  sched_setaffinity(0, sizeof(set), &set);
}

int main(
    int argc,
    char** argv
) {
  reserve_system_core();
  mju_user_warning = mujoco_warning;
  mju_user_error = mujoco_error;
  return run(argc, argv);
}
