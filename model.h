#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace xml {

struct Node {
  std::string tag;
  std::map<std::string, std::string> attr;
  std::vector<std::unique_ptr<Node>> kids;

  bool has(const std::string& key) const { return attr.count(key) != 0; }

  const std::string& get(
      const std::string& key,
      const std::string& fallback
  ) const {
    const auto it = attr.find(key);
    return it == attr.end() ? fallback : it->second;
  }

  std::vector<const Node*> children(const std::string& want) const {
    std::vector<const Node*> found;
    for (const std::unique_ptr<Node>& k : kids) {
      if (k->tag == want) found.push_back(k.get());
    }
    return found;
  }

  const Node* child(const std::string& want) const {
    for (const std::unique_ptr<Node>& k : kids) {
      if (k->tag == want) return k.get();
    }
    return nullptr;
  }
};

std::unique_ptr<Node> parse(const std::string& text);

}

namespace mjcf {

using Vec3 = std::array<double, 3>;

using Quat = std::array<double, 4>;

struct Inertial {
  Vec3 pos{{0.0, 0.0, 0.0}};
  Quat quat{{1.0, 0.0, 0.0, 0.0}};
  double mass = 0.0;
  Vec3 diaginertia{{0.0, 0.0, 0.0}};
};

struct Joint {
  std::string name;
  std::string type = "hinge";
  Vec3 pos{{0.0, 0.0, 0.0}};
  Vec3 axis{{0.0, 0.0, 1.0}};
  bool limited = false;
  double range[2] = {0.0, 0.0};
  double armature = 0.0;
  double damping = 0.0;
  double frictionloss = 0.0;
  bool frc_limited = false;
  double frcrange[2] = {0.0, 0.0};

  bool is_free() const { return type == "free"; }
};

struct Geom {
  std::string name;
  std::string type = "sphere";
  std::string mesh;
  std::vector<double> size;
  Vec3 pos{{0.0, 0.0, 0.0}};
  Quat quat{{1.0, 0.0, 0.0, 0.0}};
  bool has_fromto = false;
  double fromto[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  bool collides = false;
};

struct Body {
  std::string name;
  int parent = -1;
  Vec3 pos{{0.0, 0.0, 0.0}};
  Quat quat{{1.0, 0.0, 0.0, 0.0}};
  Inertial inertial;
  std::vector<Joint> joints;
  std::vector<Geom> geoms;
  std::vector<int> kids;
};

struct Mesh {
  std::string file;
  int maxhullvert = 0;
};

struct Model {
  std::vector<Body> bodies;
  std::map<std::string, Mesh> mesh_file;
  std::vector<Geom> world_geoms;
  double timestep = 0.002;

  int body_index(const std::string& name) const {
    for (size_t i = 0; i < bodies.size(); ++i) {
      if (bodies[i].name == name) return static_cast<int>(i);
    }
    return -1;
  }
};

Model load(const std::string& path);

std::vector<Vec3> stl_vertices(const std::string& path);

double mesh_rbound(const std::string& path);

}
