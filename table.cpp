/**
 * @file table.cpp
 * @brief Recomputes the results table in README.md from results/result.csv.
 *
 * The table in the README is the only part of this repository that is
 * arithmetic over the campaign rather than a description of it, which makes
 * it the only part that can quietly stop matching the file it claims to
 * summarise. This program is the arithmetic, so the table can be checked
 * against the runs at any time instead of trusted:
 *
 *     make table && ./build/table | diff - <(sed -n '/^| `--policy`/,/^$/p' README.md)
 *
 * It links nothing. The campaign file is plain CSV and every figure in the
 * table is a mean over it, so the whole computation is a single pass with
 * standard library only, and it will keep working long after the MuJoCo and
 * TensorRT versions the harness was built against have moved on.
 */

#include <algorithm>
#include <array>
#include <charconv>
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

/**
 * How many targets a tour demands, and so how many waypoint blocks a row
 * carries.
 */
inline constexpr int WAYPOINTS = 12;

/**
 * The joint groups each waypoint block is broken into, in column order.
 *
 * Summed away immediately: the table reports what a policy spent across all
 * 29 joints, and the per-group split exists in the CSV for readers with a
 * narrower question than this table answers.
 */
inline constexpr std::array<const char*, 5> GROUPS =
    {"legs_upper", "legs_lower", "waist", "arms_upper", "arms_lower"};

/**
 * Smallest number of runs that may stand behind a printed mean.
 *
 * A policy that rarely reaches a waypoint has no honest average there, and
 * printing one invites a comparison the sample cannot support. Cells under
 * the floor are written as a dash instead.
 */
inline constexpr int FLOOR = 2000;

/**
 * The one row that is reported but not ranked.
 *
 * `clobot_with_arms` is the `clobot` checkpoint wired to own all 29 joints
 * rather than the 15 every other row drives, which makes it a harness
 * configuration rather than a fourteenth policy. It is forced to the bottom
 * of the table and struck through wherever it appears.
 */
inline constexpr const char* UNRANKED = "clobot_with_arms";

/**
 * Everything one policy's runs contribute, accumulated over the campaign.
 *
 * Kept as running sums rather than as stored rows because the campaign is
 * 140,000 lines and none of the figures in the table need a second pass.
 * The two counts are separate because the averages are taken over different
 * populations: every run for survival and the errors, and only the runs
 * that finished the tour for what a tour cost.
 */
struct Totals {
  long runs = 0;          ///< every run of this policy
  long completed = 0;     ///< runs whose outcome was `complete`
  double survival = 0.0;  ///< sum of survival, s

  long targets = 0;  ///< targets scored, the weight the errors carry
  double pos = 0.0;  ///< sum of position error weighted by targets, cm
  double yaw = 0.0;  ///< sum of heading error weighted by targets, deg

  long finished = 0;    ///< runs that finished all `WAYPOINTS` waypoints
  double tour_e = 0.0;  ///< sum of whole-tour energy, J
  double tour_v = 0.0;  ///< sum of whole-tour vibration, krad/s^2
};

/**
 * Split one CSV line on commas.
 *
 * No quoting is handled and none is needed: every field this file carries is
 * a bare number or a policy name, and the writer never emits a comma inside
 * one. A line that grows a quoted field would be a schema change, and the
 * header check would catch it before this ever saw it.
 *
 * @param[in] line   the line, without its newline
 * @param[out] out   reused between lines so the pass does not allocate
 * @exceptsafe basic
 */
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

/**
 * Read a field as a double, treating an empty field as zero.
 *
 * Empty is not missing data here: a run that fell leaves the waypoints it
 * never reached blank, and those rows are excluded from the relevant average
 * by their count rather than by their value.
 *
 * @param[in] field  the raw field
 * @returns its value, or 0.0 if it is empty
 * @throws std::runtime_error if the field is neither empty nor a number
 * @exceptsafe strong
 */
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

/**
 * Round a double to a fixed number of decimals, half away from zero.
 *
 * Done on the decimal text rather than on the double because the two
 * disagree exactly where a reader would notice. 6515 completions in 10000
 * runs is 65.15%, which a reader rounds to 65.2%, but the nearest double to
 * 65.15 is a hair below it and every binary rounding — including
 * `std::round(x * 10) / 10` — answers 65.1%. Formatting to the shortest text
 * that round-trips and then carrying the decimal by hand gives the answer
 * the arithmetic has, not the answer the representation has.
 *
 * @param[in] value   the figure to render, finite
 * @param[in] places  decimals to keep, zero or more
 * @returns the rounded figure as text, always with exactly `places` decimals
 * @throws std::runtime_error if the value cannot be formatted
 * @exceptsafe strong
 */
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

/**
 * Render one cell, or a dash when too few runs stand behind it.
 *
 * @param[in] sum     the running total
 * @param[in] count   how many runs contributed to it
 * @param[in] places  decimals to keep; the cost columns are whole units,
 *                    since a tenth of a joule or of a krad/s^2 over a
 *                    whole tour is noise
 * @param[in] unit    appended to the figure, empty for a bare number
 * @returns the cell text
 * @exceptsafe strong
 */
std::string cell(
    double sum,
    long count,
    int places,
    const std::string& unit
) {
  if (count < FLOOR) return "-";
  return fixed(sum / static_cast<double>(count), places) + unit;
}

/**
 * Map every column name in the header to its position.
 *
 * Looked up by name rather than by offset so that a column added to the
 * middle of the file cannot silently shift every figure in the table one
 * place to the left.
 *
 * @param[in] header  the first line of the campaign file
 * @returns name to index
 * @exceptsafe basic
 */
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

/**
 * Find one column, refusing to guess if it is not there.
 *
 * @param[in] index  the header map
 * @param[in] name   the column wanted
 * @returns its position
 * @throws std::runtime_error if the campaign file has no such column
 * @exceptsafe strong
 */
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

/**
 * Accumulate one campaign file into per-policy totals.
 *
 * One pass, one row at a time. A row is charged to the opening-waypoint
 * average if it scored a target at all, and to the whole-tour average only
 * if it completed: those are the only runs in which all twelve waypoints ran
 * their clock, and so the only ones whose totals mean the same thing.
 *
 * @param[in] path  the campaign file
 * @returns totals per policy name
 * @throws std::runtime_error if the file cannot be read, is empty, or is
 *         missing a column this table needs
 * @exceptsafe basic
 */
std::map<
    std::string,
    Totals>
accumulate(const std::string& path) {
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
  const size_t targets_at = column(index, "targets");
  const size_t pos_at = column(index, "pos_err_cm");
  const size_t yaw_at = column(index, "yaw_err_deg");

  std::vector<std::vector<size_t>> energy(WAYPOINTS);
  std::vector<std::vector<size_t>> vibration(WAYPOINTS);
  for (int i = 0; i < WAYPOINTS; ++i) {
    const std::string prefix = "s" + std::to_string(i) + "_";
    for (const char* group : GROUPS) {
      energy[i].push_back(column(index, prefix + "e_" + group + "_j"));
      vibration[i].push_back(column(index, prefix + "v_" + group + "_krads2"));
    }
  }

  std::map<std::string, Totals> totals;
  std::vector<std::string> field;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    split(line, field);
    if (field.size() <= vibration[WAYPOINTS - 1].back()) {
      throw std::runtime_error("table: short row in " + path);
    }

    Totals& t = totals[field[policy_at]];
    ++t.runs;
    t.survival += number(field[survival_at]);
    const long scored = static_cast<long>(number(field[targets_at]));
    const bool complete = field[outcome_at] == "complete";
    if (complete) ++t.completed;

    if (scored > 0) {
      t.targets += scored;
      t.pos += number(field[pos_at]) * static_cast<double>(scored);
      t.yaw += number(field[yaw_at]) * static_cast<double>(scored);
    }

    if (complete) {
      ++t.finished;
      for (int i = 0; i < WAYPOINTS; ++i) {
        for (const size_t at : energy[i]) t.tour_e += number(field[at]);
        for (const size_t at : vibration[i]) t.tour_v += number(field[at]);
      }
    }
  }
  return totals;
}

/**
 * Order the rows as the table prints them.
 *
 * Most completions first, because that is the column the table is read on.
 * `UNRANKED` is pulled to the bottom whatever it scored, since it is there
 * to be read against `clobot` rather than against the field.
 *
 * @param[in] totals  every policy in the campaign
 * @returns policy names in printing order
 * @exceptsafe basic
 */
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
        if ((a == UNRANKED) != (b == UNRANKED)) return b == UNRANKED;
        const long ca = totals.at(a).completed;
        const long cb = totals.at(b).completed;
        return ca != cb ? ca > cb : a < b;
      }
  );
  return names;
}

/**
 * Write one policy's row in the README's markdown.
 *
 * The unranked row is struck through cell by cell and carries the footnote
 * marker, which is what tells a reader it is not competing with the rows
 * above it. A dash is never struck: there is nothing there to strike.
 *
 * @param[in] name  the policy
 * @param[in] t     its totals
 * @returns the row, without its newline
 * @exceptsafe basic
 */
std::string row(
    const std::string& name,
    const Totals& t
) {
  const double runs = static_cast<double>(t.runs);
  const double weight = static_cast<double>(t.targets);
  const std::vector<std::string> cells = {
      fixed(100.0 * static_cast<double>(t.completed) / runs, 1) + "%",
      fixed(t.survival / runs, 1) + " s",
      t.targets > 0 ? fixed(t.pos / weight, 0) + " cm" : "-",
      t.targets > 0 ? fixed(t.yaw / weight, 0) + "°" : "-",
      cell(t.tour_e, t.finished, 0, " J"),
      cell(t.tour_v, t.finished, 0, "")
  };

  const bool unranked = name == UNRANKED;
  std::ostringstream out;
  out << "| " << (unranked ? "~~`" + name + "`~~\\*\\*" : "`" + name + "`");
  for (size_t i = 0; i < cells.size(); ++i) {
    std::string text = i == 0 ? "**" + cells[i] + "**" : cells[i];
    if (unranked && cells[i] != "-") text = "~~" + text + "~~";
    out << " | " << text;
  }
  out << " |";
  return out.str();
}

}

/**
 * Print the README's results table for one campaign file.
 *
 * @param[in] argc  argument count
 * @param[in] argv  optional path to the campaign file
 * @returns 0 on success, 1 if the campaign file could not be summarised
 * @exceptsafe basic
 */
int main(
    int argc,
    char** argv
) {
  const std::string path = argc > 1 ? argv[1] : "results/result.csv";
  try {
    const std::map<std::string, table::Totals> totals = table::accumulate(path);
    if (totals.empty()) {
      throw std::runtime_error("table: " + path + " holds no runs");
    }

    std::cout << "| `--policy` | completed | survived | pos err | yaw err "
                 "| tour<br>battery<br>energy<br>consumed "
                 "| tour<br>vibrations |\n"
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
