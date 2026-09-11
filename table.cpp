#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace table {

inline constexpr std::array<const char*, 5> GROUPS =
    {"legs_upper", "legs_lower", "waist", "arms_upper", "arms_lower"};

inline constexpr int FLOOR_NUMERATOR = 1;
inline constexpr int FLOOR_DENOMINATOR = 5;

inline constexpr std::array<const char*, 2> UNRANKED = {
    "clobot_with_arms",
    "handoff_with_arms"
};

inline bool is_unranked(const std::string& name) {
  for (const char* u : UNRANKED) {
    if (name == u) return true;
  }
  return false;
}

constexpr const char* SIM_VELOCITY[] = {"huru", "josabb", "mturan33", "sunny"};

inline bool needs_sim_velocity(const std::string& name) {
  for (const char* v : SIM_VELOCITY) {
    if (name == v) return true;
  }
  return false;
}

struct Totals {
  long runs = 0;
  long completed = 0;
  double survival = 0.0;
  long runs_mj = 0;
  long done_mj = 0;
  long runs_px = 0;
  long done_px = 0;

  long scored = 0;
  long diverged = 0;
  double pos = 0.0;
  double yaw = 0.0;

  long finished = 0;
  double walk_e = 0.0;
  double walk_v = 0.0;
};

void split(
    const std::string& line,
    std::vector<std::string>& out
) {
  out.clear();
  size_t start = 0;
  while (true) {
    const size_t comma = line.find(',', start);
    if (comma == std::string::npos) {
      out.emplace_back(line, start);
      return;
    }
    out.emplace_back(line, start, comma - start);
    start = comma + 1;
  }
}

double number(const std::string& field) {
  if (field.empty()) return 0.0;
  double value = 0.0;
  const auto end = field.data() + field.size();
  const auto result = std::from_chars(field.data(), end, value);
  if (result.ec != std::errc() || result.ptr != end) {
    throw std::runtime_error("table: '" + field + "' is not a number");
  }
  return value;
}

std::string fixed(
    double value,
    int places
) {
  std::array<char, 64> buffer{};
  const auto result = std::to_chars(
      buffer.data(),
      buffer.data() + buffer.size(),
      value,
      std::chars_format::fixed
  );
  if (result.ec != std::errc()) {
    throw std::runtime_error("table: cannot format a figure");
  }
  std::string text(buffer.data(), result.ptr);

  const bool negative = !text.empty() && text.front() == '-';
  if (negative) text.erase(0, 1);
  if (text.find('.') == std::string::npos) text += '.';

  const size_t point = text.find('.');
  std::string digits = text.substr(0, point) + text.substr(point + 1);
  const size_t kept = point + static_cast<size_t>(places);

  if (digits.size() > kept) {
    const bool round_up = digits[kept] >= '5';
    digits.resize(kept);
    if (round_up) {
      size_t i = digits.size();
      while (i > 0) {
        if (digits[i - 1] != '9') {
          ++digits[i - 1];
          break;
        }
        digits[i - 1] = '0';
        --i;
      }
      if (i == 0) digits.insert(digits.begin(), '1');
    }
  } else {
    digits.append(kept - digits.size(), '0');
  }

  const size_t whole = digits.size() - static_cast<size_t>(places);
  std::string out = digits.substr(0, whole);
  if (out.empty()) out = "0";
  if (places > 0) out += "." + digits.substr(whole);
  return negative && out.find_first_not_of("0.") != std::string::npos
             ? "-" + out
             : out;
}

std::string cell(
    double sum,
    long count,
    long runs,
    int places,
    const std::string& unit
) {
  if (count * FLOOR_DENOMINATOR < runs * FLOOR_NUMERATOR) return "-";
  return fixed(sum / static_cast<double>(count), places) + unit;
}

std::map<
    std::string,
    size_t>
columns(const std::string& header) {
  std::vector<std::string> names;
  split(header, names);
  std::map<std::string, size_t> index;
  for (size_t i = 0; i < names.size(); ++i) index[names[i]] = i;
  return index;
}

size_t column(
    const std::map<
        std::string,
        size_t>& index,
    const std::string& name
) {
  const auto found = index.find(name);
  if (found == index.end()) {
    throw std::runtime_error("table: the campaign file has no '" + name + "'");
  }
  return found->second;
}

void accumulate(
    const std::string& path,
    std::map<
        std::string,
        Totals>& totals
) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("table: cannot open " + path);

  std::string line;
  if (!std::getline(in, line)) {
    throw std::runtime_error("table: " + path + " is empty");
  }
  const std::map<std::string, size_t> index = columns(line);

  const size_t policy_at = column(index, "policy");
  const size_t outcome_at = column(index, "outcome");
  const size_t survival_at = column(index, "survival_s");
  const size_t pos_at = column(index, "pos_err_cm");
  const size_t yaw_at = column(index, "yaw_err_deg");

  std::vector<size_t> energy;
  std::vector<size_t> vibration;
  size_t widest = 0;
  for (int seg = 0;; ++seg) {
    const std::string prefix = "s" + std::to_string(seg) + "_";
    const std::string first = prefix + "e_" + GROUPS.front() + "_j";
    if (index.find(first) == index.end()) break;
    for (const char* group : GROUPS) {
      energy.push_back(column(index, prefix + "e_" + group + "_j"));
      vibration.push_back(column(index, prefix + "v_" + group + "_krads2"));
      widest = std::max({widest, energy.back(), vibration.back()});
    }
  }
  if (energy.empty()) {
    throw std::runtime_error(
        "table: " + path + " has no 's0_e_" + GROUPS.front() + "_j'"
    );
  }

  const bool mujoco = path.find(".mujoco.") != std::string::npos;

  std::vector<std::string> field;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    split(line, field);
    if (field.size() <= widest) {
      throw std::runtime_error("table: short row in " + path);
    }

    Totals& t = totals[field[policy_at]];
    ++t.runs;
    t.survival += number(field[survival_at]);
    if (mujoco) {
      ++t.runs_mj;
      if (field[outcome_at] == "complete") ++t.done_mj;
    } else {
      ++t.runs_px;
      if (field[outcome_at] == "complete") ++t.done_px;
    }
    const bool complete = field[outcome_at] == "complete";
    if (complete) ++t.completed;

    const double pos = number(field[pos_at]);
    const double yaw = number(field[yaw_at]);
    const bool sane = std::isfinite(pos) && std::isfinite(yaw);
    if (!sane) ++t.diverged;
    const bool scored = !field[energy.front()].empty() && sane;
    if (scored) {
      ++t.scored;
      t.pos += pos;
      t.yaw += yaw;
    }

    if (complete && scored) {
      ++t.finished;
      for (const size_t at : energy) t.walk_e += number(field[at]);
      for (const size_t at : vibration) t.walk_v += number(field[at]);
    }
  }
}

double percent(
    long done,
    long runs
) {
  if (runs == 0) return 0.0;
  return std::lround(
             100.0 * static_cast<double>(done) / static_cast<double>(runs)
         ) *
         1.0;
}

double mean_survival(const Totals& t) {
  if (t.runs == 0) return 0.0;
  return t.survival / static_cast<double>(t.runs);
}

std::vector<std::string> ordered(
    const std::map<
        std::string,
        Totals>& totals
) {
  std::vector<std::string> names;
  for (const auto& [name, _] : totals) names.push_back(name);
  std::sort(
      names.begin(),
      names.end(),
      [&totals](const std::string& a, const std::string& b) {
        if (is_unranked(a) != is_unranked(b)) return is_unranked(b);
        const Totals& ta = totals.at(a);
        const Totals& tb = totals.at(b);

        const double sa = mean_survival(ta);
        const double sb = mean_survival(tb);
        if (sa != sb) return sa > sb;

        if ((ta.scored > 0) != (tb.scored > 0)) return ta.scored > 0;
        if (ta.scored > 0) {
          const long ea = std::lround(ta.pos / static_cast<double>(ta.scored));
          const long eb = std::lround(tb.pos / static_cast<double>(tb.scored));
          if (ea != eb) return ea < eb;
        }
        return a < b;
      }
  );
  return names;
}

std::string rate(
    long done,
    long runs
) {
  if (runs == 0) return "-";
  return fixed(
      100.0 * static_cast<double>(done) / static_cast<double>(runs),
      0
  );
}

std::string both(const Totals& t) {
  if (t.runs_mj == 0 && t.runs_px == 0) return "-";
  return rate(t.done_mj, t.runs_mj) + "/" + rate(t.done_px, t.runs_px) + " %";
}

std::string survival_cell(const Totals& t) {
  if (t.runs == 0) return "-";
  return fixed(mean_survival(t), 1);
}

std::string runs_cell(const Totals& t) {
  if (t.runs_mj == 0 && t.runs_px == 0) return "-";
  if (t.runs_mj == t.runs_px) return std::to_string(t.runs_mj);
  return std::to_string(t.runs_mj) + "/" + std::to_string(t.runs_px);
}

std::string row(
    const std::string& name,
    const Totals& t
) {
  const double weight = static_cast<double>(t.scored);
  const std::vector<std::string> cells = {
      survival_cell(t),
      both(t),
      runs_cell(t),
      t.scored > 0
          ? fixed(t.pos / weight, 0) + " cm / " + fixed(t.yaw / weight, 0) + "°"
          : "-",
      cell(t.walk_e, t.finished, t.runs, 0, " J"),
      cell(t.walk_v, t.finished, t.runs, 0, "")
  };

  const bool unranked = is_unranked(name);
  std::ostringstream out;
  const std::string mark = needs_sim_velocity(name) ? "\\*" : "";
  out << "| "
      << (unranked ? "~~`" + name + "`~~\\*\\*" : "`" + name + "`" + mark);
  for (size_t i = 0; i < cells.size(); ++i) {
    std::string text = i < 1 ? "**" + cells[i] + "**" : cells[i];
    if (unranked && cells[i] != "-") text = "~~" + text + "~~";
    out << " | " << text;
  }
  out << " |";
  return out.str();
}

}

int main(
    int argc,
    char** argv
) {
  if (argc < 2) {
    std::cerr << "usage: table CAMPAIGN.csv...\n"
              << "  one round of one policy on one engine per file, as in\n"
              << "  table results/*.csv\n";
    return 1;
  }
  try {
    std::map<std::string, table::Totals> totals;
    for (int i = 1; i < argc; ++i) table::accumulate(argv[i], totals);
    if (totals.empty()) throw std::runtime_error("table: no runs to pool");

    std::cout << "| `--policy` | mean<br>survival s "
                 "| completed<br>mujoco/physx | runs | err<br>pos/yaw "
                 "| walk<br>battery<br>energy<br>consumed "
                 "| walk<br>vibrations |\n"
              << "|---:|---:|---:|---:|---:|---:|---:|\n";
    for (const std::string& name : table::ordered(totals)) {
      std::cout << table::row(name, totals.at(name)) << '\n';
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
