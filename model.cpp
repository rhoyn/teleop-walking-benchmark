#include "model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <stdexcept>

namespace xml {
namespace {

struct Cursor {
  const std::string& s;
  size_t i = 0;

  bool done() const { return i >= s.size(); }
  char peek() const { return i < s.size() ? s[i] : '\0'; }

  bool looking_at(const char* lit) const {
    return s.compare(i, std::char_traits<char>::length(lit), lit) == 0;
  }

  void skip_space() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])) != 0)
      ++i;
  }

  [[noreturn]] void fail(const std::string& why) const {
    size_t line = 1;
    for (size_t k = 0; k < i && k < s.size(); ++k) {
      if (s[k] == '\n') ++line;
    }
    throw std::runtime_error(
        "xml: " + why + " at line " + std::to_string(line)
    );
  }
};

void skip_trivia(Cursor& c) {
  for (;;) {
    c.skip_space();
    if (!c.looking_at("<!--")) return;
    const size_t end = c.s.find("-->", c.i);
    if (end == std::string::npos) c.fail("unterminated comment");
    c.i = end + 3;
  }
}

std::string read_name(Cursor& c) {
  const size_t start = c.i;
  while (!c.done()) {
    const char ch = c.peek();
    const bool ok = std::isalnum(static_cast<unsigned char>(ch)) != 0 ||
                    ch == '_' || ch == '-' || ch == ':' || ch == '.';
    if (!ok) break;
    ++c.i;
  }
  if (c.i == start) c.fail("expected a name");
  return c.s.substr(start, c.i - start);
}

std::string read_value(Cursor& c) {
  const char quote = c.peek();
  if (quote != '"' && quote != '\'') c.fail("expected a quoted value");
  ++c.i;
  const size_t start = c.i;
  while (!c.done() && c.peek() != quote) {
    if (c.peek() == '&') c.fail("entity references are not supported");
    ++c.i;
  }
  if (c.done()) c.fail("unterminated value");
  const std::string value = c.s.substr(start, c.i - start);
  ++c.i;
  return value;
}

std::unique_ptr<Node> read_element(Cursor& c);

void read_children(
    Cursor& c,
    Node& parent
) {
  for (;;) {
    skip_trivia(c);
    if (c.done()) c.fail("unterminated <" + parent.tag + ">");
    if (!c.looking_at("<")) {
      while (!c.done() && c.peek() != '<') ++c.i;
      continue;
    }
    if (c.looking_at("</")) {
      c.i += 2;
      const std::string close = read_name(c);
      if (close != parent.tag)
        c.fail("</" + close + "> closes <" + parent.tag + ">");
      skip_trivia(c);
      if (c.peek() != '>') c.fail("expected '>'");
      ++c.i;
      return;
    }
    parent.kids.push_back(read_element(c));
  }
}

std::unique_ptr<Node> read_element(Cursor& c) {
  if (c.peek() != '<') c.fail("expected '<'");
  if (c.looking_at("<![CDATA[")) c.fail("CDATA is not supported");
  if (c.looking_at("<!")) c.fail("declarations are not supported");
  ++c.i;

  std::unique_ptr<Node> node = std::make_unique<Node>();
  node->tag = read_name(c);

  for (;;) {
    skip_trivia(c);
    if (c.done()) c.fail("unterminated <" + node->tag + ">");
    if (c.looking_at("/>")) {
      c.i += 2;
      return node;
    }
    if (c.peek() == '>') {
      ++c.i;
      read_children(c, *node);
      return node;
    }
    const std::string key = read_name(c);
    skip_trivia(c);
    if (c.peek() != '=') c.fail("expected '=' after " + key);
    ++c.i;
    skip_trivia(c);
    if (node->attr.count(key) != 0) c.fail("duplicate attribute " + key);
    node->attr[key] = read_value(c);
  }
}

}

std::unique_ptr<Node> parse(const std::string& text) {
  Cursor c{text};
  skip_trivia(c);

  if (c.looking_at("<?")) {
    const size_t end = text.find("?>", c.i);
    if (end == std::string::npos) c.fail("unterminated prolog");
    c.i = end + 2;
    skip_trivia(c);
  }
  std::unique_ptr<Node> root = read_element(c);
  skip_trivia(c);
  if (!c.done()) c.fail("trailing content after the root element");
  return root;
}

}

namespace mjcf {
namespace {

std::vector<double> numbers(const std::string& text) {
  std::vector<double> out;
  std::istringstream in(text);
  double v = 0.0;
  while (in >> v) out.push_back(v);
  return out;
}

std::vector<double> fixed(
    const xml::Node& n,
    const std::string& key,
    size_t want,
    const std::vector<double>& fallback
) {
  if (!n.has(key)) return fallback;
  const std::vector<double> got = numbers(n.get(key, ""));
  if (got.size() != want) {
    throw std::runtime_error(
        "mjcf: <" + n.tag + "> " + key + " wants " + std::to_string(want) +
        " numbers, has " + std::to_string(got.size())
    );
  }
  return got;
}

Vec3 vec3(
    const xml::Node& n,
    const std::string& key,
    Vec3 fallback
) {
  const std::vector<double> got =
      fixed(n, key, 3, {fallback[0], fallback[1], fallback[2]});
  return Vec3{{got[0], got[1], got[2]}};
}

Quat quat(
    const xml::Node& n,
    const std::string& key,
    Quat fallback
) {
  const std::vector<double> got =
      fixed(n, key, 4, {fallback[0], fallback[1], fallback[2], fallback[3]});
  return Quat{{got[0], got[1], got[2], got[3]}};
}

double number(
    const xml::Node& n,
    const std::string& key,
    double fallback
) {
  if (!n.has(key)) return fallback;
  return std::stod(n.get(key, ""));
}

bool collides(
    const xml::Node& n,
    int default_contype,
    int default_conaffinity
) {
  const int contype = static_cast<int>(number(n, "contype", default_contype));
  const int conaffinity =
      static_cast<int>(number(n, "conaffinity", default_conaffinity));
  return contype != 0 || conaffinity != 0;
}

Geom read_geom(
    const xml::Node& n,
    int d_contype,
    int d_conaffinity
) {
  Geom g;
  g.name = n.get("name", "");
  g.mesh = n.get("mesh", "");
  g.type = n.get("type", g.mesh.empty() ? "sphere" : "mesh");
  g.pos = vec3(n, "pos", Vec3{{0.0, 0.0, 0.0}});
  g.quat = quat(n, "quat", Quat{{1.0, 0.0, 0.0, 0.0}});
  g.size = n.has("size") ? numbers(n.get("size", "")) : std::vector<double>{};
  if (n.has("fromto")) {
    const std::vector<double> ft = fixed(n, "fromto", 6, {});
    g.has_fromto = true;
    for (int k = 0; k < 6; ++k) g.fromto[k] = ft[static_cast<size_t>(k)];
  }
  g.collides = collides(n, d_contype, d_conaffinity);
  return g;
}

Joint read_joint(const xml::Node& n) {
  Joint j;
  j.name = n.get("name", "");
  j.type = n.get("type", "hinge");
  if (j.type != "hinge" && j.type != "free") {
    throw std::runtime_error(
        "mjcf: joint '" + j.name + "' is type '" + j.type +
        "'; only hinge and free are implemented"
    );
  }
  j.pos = vec3(n, "pos", Vec3{{0.0, 0.0, 0.0}});
  j.axis = vec3(n, "axis", Vec3{{0.0, 0.0, 1.0}});
  j.armature = number(n, "armature", 0.0);
  j.damping = number(n, "damping", 0.0);
  j.frictionloss = number(n, "frictionloss", 0.0);
  if (n.has("range")) {
    const std::vector<double> r = fixed(n, "range", 2, {});
    j.range[0] = r[0];
    j.range[1] = r[1];

    j.limited = n.get("limited", "true") != "false";
  }
  if (n.has("actuatorfrcrange")) {
    const std::vector<double> r = fixed(n, "actuatorfrcrange", 2, {});
    j.frcrange[0] = r[0];
    j.frcrange[1] = r[1];
    j.frc_limited = n.get("actuatorfrclimited", "true") != "false";
  }
  return j;
}

void read_body(
    const xml::Node& n,
    int parent,
    Model& m,
    int d_contype,
    int d_conaffinity
) {
  Body b;
  b.name = n.get("name", "");
  b.parent = parent;
  b.pos = vec3(n, "pos", Vec3{{0.0, 0.0, 0.0}});
  b.quat = quat(n, "quat", Quat{{1.0, 0.0, 0.0, 0.0}});

  if (const xml::Node* in = n.child("inertial"); in != nullptr) {
    b.inertial.pos = vec3(*in, "pos", Vec3{{0.0, 0.0, 0.0}});
    b.inertial.quat = quat(*in, "quat", Quat{{1.0, 0.0, 0.0, 0.0}});
    b.inertial.mass = number(*in, "mass", 0.0);
    b.inertial.diaginertia = vec3(*in, "diaginertia", Vec3{{0.0, 0.0, 0.0}});
  } else {
    throw std::runtime_error(
        "mjcf: body '" + b.name +
        "' has no <inertial>; mass would have to be inferred from meshes"
    );
  }

  for (const xml::Node* j : n.children("joint"))
    b.joints.push_back(read_joint(*j));
  for (const xml::Node* g : n.children("geom"))
    b.geoms.push_back(read_geom(*g, d_contype, d_conaffinity));

  const int self = static_cast<int>(m.bodies.size());
  m.bodies.push_back(b);
  if (parent >= 0) m.bodies[static_cast<size_t>(parent)].kids.push_back(self);

  for (const xml::Node* k : n.children("body"))
    read_body(*k, self, m, d_contype, d_conaffinity);
}

}

Model load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("mjcf: cannot open " + path);
  std::ostringstream buf;
  buf << in.rdbuf();

  const std::unique_ptr<xml::Node> root = xml::parse(buf.str());
  if (root->tag != "mujoco")
    throw std::runtime_error(
        "mjcf: root element is <" + root->tag + ">, not <mujoco>"
    );

  Model m;

  std::string meshdir;
  if (const xml::Node* c = root->child("compiler"); c != nullptr) {
    if (c->get("angle", "degree") != "radian") {
      throw std::runtime_error(
          "mjcf: compiler angle is not 'radian'; this reader does no conversion"
      );
    }
    meshdir = c->get("meshdir", "");
  }
  if (const xml::Node* o = root->child("option"); o != nullptr) {
    m.timestep = number(*o, "timestep", m.timestep);
  }

  int d_contype = 1;
  int d_conaffinity = 1;
  if (const xml::Node* d = root->child("default"); d != nullptr) {
    if (const xml::Node* g = d->child("geom"); g != nullptr) {
      d_contype = static_cast<int>(number(*g, "contype", 1));
      d_conaffinity = static_cast<int>(number(*g, "conaffinity", 1));
    }
  }

  const std::filesystem::path base = std::filesystem::path(path).parent_path();
  for (const xml::Node* a : root->children("asset")) {
    for (const xml::Node* mesh : a->children("mesh")) {
      const std::string name = mesh->get("name", "");
      const std::string file = mesh->get("file", "");
      if (name.empty() || file.empty())
        throw std::runtime_error("mjcf: <mesh> needs both name and file");
      Mesh asset;
      asset.file = (base / meshdir / file).string();
      asset.maxhullvert = static_cast<int>(number(*mesh, "maxhullvert", 0.0));
      m.mesh_file[name] = asset;
    }
  }

  const std::vector<const xml::Node*> worlds = root->children("worldbody");
  if (worlds.empty()) throw std::runtime_error("mjcf: no <worldbody>");

  std::vector<const xml::Node*> roots;
  for (const xml::Node* world : worlds) {
    for (const xml::Node* g : world->children("geom")) {
      m.world_geoms.push_back(read_geom(*g, d_contype, d_conaffinity));
    }
    for (const xml::Node* b : world->children("body")) {
      if (b->get("mocap", "false") == "true") continue;
      roots.push_back(b);
    }
  }
  if (roots.size() != 1) {
    throw std::runtime_error(
        "mjcf: expected exactly one non-mocap root body, found " +
        std::to_string(roots.size())
    );
  }
  read_body(*roots[0], -1, m, d_contype, d_conaffinity);

  return m;
}

}

namespace mjcf {

std::vector<Vec3> stl_vertices(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("stl: cannot open " + path);
  std::vector<char> raw(
      (std::istreambuf_iterator<char>(in)),
      std::istreambuf_iterator<char>()
  );
  if (raw.size() < 84) {
    throw std::runtime_error("stl: " + path + " is too short to be binary STL");
  }
  uint32_t tris = 0;
  std::memcpy(&tris, raw.data() + 80, 4);
  if (raw.size() != 84 + static_cast<size_t>(tris) * 50) {
    throw std::runtime_error(
        "stl: " + path +
        " is not binary STL (size does not match its triangle count)"
    );
  }
  std::vector<Vec3> verts;
  verts.reserve(static_cast<size_t>(tris) * 3);
  for (uint32_t t = 0; t < tris; ++t) {
    const char* rec = raw.data() + 84 + static_cast<size_t>(t) * 50 + 12;
    for (int v = 0; v < 3; ++v) {
      float xyz[3];
      std::memcpy(xyz, rec + v * 12, 12);
      verts.push_back(Vec3{{xyz[0], xyz[1], xyz[2]}});
    }
  }
  return verts;
}

double mesh_rbound(const std::string& path) {
  double lo[3] = {1e30, 1e30, 1e30};
  double hi[3] = {-1e30, -1e30, -1e30};
  for (const Vec3& v : stl_vertices(path)) {
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], v[static_cast<size_t>(k)]);
      hi[k] = std::max(hi[k], v[static_cast<size_t>(k)]);
    }
  }
  double r2 = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double half = 0.5 * (hi[k] - lo[k]);
    r2 += half * half;
  }
  return std::sqrt(r2);
}

}
