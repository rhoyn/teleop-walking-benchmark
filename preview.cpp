/**
 * @file preview.cpp
 * @brief Generates assets/preview.mp4 and assets/preview.jpg from the field.
 *
 * The two files at the top of the README are the only assets in this
 * repository that are a rendering of a campaign rather than a description
 * of one, which makes them the only ones that can quietly stop showing what
 * the harness now does. This program is how they are made, so they can be
 * rebuilt from the current code instead of trusted:
 *
 *     make && make preview && ./build/preview
 *
 * It renders nothing itself. Every frame comes from the harness's own
 * `--record`, one policy per invocation, and everything after that is
 * ffmpeg: thirteen clips of one seed scaled into a four by four grid, the
 * wordmark in the free cell, a still lifted out of the middle, and the
 * closing seconds encoded for the web. Keeping the compositing in ffmpeg
 * rather than in the harness is what lets the clips be recorded once and
 * recut as often as the framing needs it.
 *
 * A run ends the moment its policy falls, so the thirteen clips are
 * thirteen different lengths. Each tile is padded by cloning its last
 * frame, which is why the montage shows a fallen policy lying where it went
 * down for the rest of the tour rather than the grid going dark as the
 * field thins out.
 */

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <sys/wait.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace preview {

/**
 * The field, in the order the results table ranks it.
 *
 * The grid reads like the table: best completion first, filling left to
 * right and top to bottom after the wordmark. `clobot_with_arms` is absent
 * on purpose — it is the `clobot` checkpoint wired to own all 29 joints
 * rather than a fourteenth policy, and showing it beside the rows it does
 * not rank against is exactly the comparison the table refuses to invite.
 */
inline constexpr std::array<const char*, 13> POLICIES = {
    "gr00t_wbc",
    "homie",
    "amo",
    "robomimic",
    "asap",
    "rl_mjlab",
    "holosoma",
    "run_residual",
    "rl_lab",
    "falcon",
    "rl_gym",
    "openwbt",
    "clobot"
};

/**
 * Columns and rows of the grid.
 *
 * Four by four is the smallest arrangement that holds thirteen tiles and
 * the wordmark at 16:9 without letterboxing a single one of them: the
 * harness records at 1280x720, and a quarter of 1920x1080 is the same
 * shape.
 */
inline constexpr int COLS = 4;
inline constexpr int ROWS = 4;

/**
 * The montage's size in pixels, and therefore each tile's.
 */
inline constexpr int OUT_WIDTH = 1920;
inline constexpr int OUT_HEIGHT = 1080;
inline constexpr int TILE_WIDTH = OUT_WIDTH / COLS;
inline constexpr int TILE_HEIGHT = OUT_HEIGHT / ROWS;

/**
 * Which cell the wordmark occupies.
 *
 * The top left, so that the eye lands on it before the field. It costs one
 * of the sixteen cells, which is free: the field is thirteen.
 */
inline constexpr int LOGO_CELL = 0;

/**
 * Frame rate of the harness's recordings, in frames per second.
 *
 * Must match `RECORD_FPS` in main.cpp. The montage is assembled at this
 * rate and resampled once, at the end, so the tiles stay in step with each
 * other while they are being combined.
 */
inline constexpr int RECORD_FPS = 50;

/**
 * Frame rate the published clip is encoded at.
 *
 * The recording's own rate, so no frame is dropped or repeated on the way
 * out. Encoding a 50 fps source at 30 keeps three frames in five and drops
 * the rest on an uneven cadence, which reads as a stutter under exactly the
 * smooth walking this clip exists to show. `--fps` changes it for anyone
 * who needs a particular rate more than they need the motion.
 */
inline constexpr int OUT_FPS = RECORD_FPS;

/**
 * How the published clip is encoded.
 *
 * One pass, straight from the filtergraph: the grid is composed and encoded
 * in the same ffmpeg invocation, so the published file is a first
 * generation encode of the recordings rather than a re-encode of an
 * intermediate. `+faststart` moves the index to the front so the file
 * starts playing before it has finished downloading, and there is no audio
 * track at all.
 *
 * Deliberately not `-tune zerolatency`: that tune is for a stream being
 * watched as it is made, and it pays for the latency by giving up the
 * lookahead and the B-frames that a downloaded file has no reason to give
 * up. On this footage it costs visible quality at the same bitrate.
 */
inline constexpr const char* ENCODE =
    "-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p"
    " -movflags +faststart -an";

/**
 * Input flags every clip is read with.
 *
 * The clips are this program's own output, so nothing here should ever
 * trigger. They are the flags the rest of this repository's video is read
 * with, and a preview that stopped dead at one damaged frame would be worse
 * than one that skipped it.
 */
inline constexpr const char* INPUT_FLAGS =
    "-err_detect aggressive -fflags discardcorrupt";

/**
 * Length of a completed tour, in seconds.
 *
 * Twelve targets on a five second clock, and the length every tile is
 * padded to. A clip shorter than this is a policy that fell.
 */
inline constexpr double TOUR_S = 60.0;

/**
 * How much of the run the published clip keeps, in seconds, counting back
 * from the end.
 *
 * Zero for all of it, which is the default: the whole tour, the crane's
 * ramp included. A positive value keeps only that many closing seconds,
 * where the punch ramp is near its ceiling and most of the field is down.
 */
inline constexpr double CLIP_S = 0.0;

/**
 * The harness's clock, mirrored from main.cpp.
 *
 * The still is timed against a punch, and a punch is timed against the
 * crane's release, so this program has to know the same schedule the
 * harness runs. These are `INIT_DURATION_S`, `PERIOD_S`, `LEAD_IN_S` and
 * `PUNCH_DELAY_S` there, and `POINT_S` is `WALK_S / WAYPOINTS`. If one of
 * them moves in main.cpp and not here, the still lands between punches and
 * comes out with no orange in it, which is the failure showing itself
 * rather than hiding.
 */
inline constexpr double INIT_DURATION_S = 3.0;
inline constexpr double PERIOD_S = 0.02;
inline constexpr double LEAD_IN_S = 0.0;
inline constexpr double PUNCH_DELAY_S = 0.1;
inline constexpr double POINT_S = 5.0;

/**
 * How long the punch marker stays up after the hit, in seconds.
 *
 * Mirrored the same way, and used only to refuse a still timed after the
 * marker has gone.
 */
inline constexpr double PUNCH_HOLD_S = 0.5;

/**
 * Which punch the still is taken on, and how long after it.
 *
 * Punch 10 of the twelve lands at about 53 s, deep enough into the tour
 * that most of the field is already down and the ramp is near its ceiling,
 * and four tenths of a second later the cylinder is still resting where it
 * hit, with the robot further into whatever the blow did to it.
 * Every tile is struck at the same instant, so one moment catches the whole
 * field mid-punch; a still half a second later would catch none of them.
 * Written as a punch and an offset rather than as a number of seconds so it
 * does not have to be rederived whenever the clock moves.
 */
inline constexpr int STILL_PUNCH = 10;
inline constexpr double STILL_AFTER_S = 0.4;
static_assert(
    STILL_AFTER_S < PUNCH_HOLD_S,
    "the still would be taken after the punch marker has gone"
);

/**
 * The wordmark, and how it is set.
 *
 * Drawn with the vendored font rather than resolved from the system, for
 * the same reason the floor labels are: a preview built on one machine
 * should carry the same lettering as one built anywhere else. ffmpeg's
 * drawtext has no letter tracking, so the spacing is spelled into the
 * string.
 */
inline constexpr const char* LOGO_FONT = "assets/JetBrainsMono.ttf";
inline constexpr const char* LOGO_TEXT = "r h o y n";
inline constexpr int LOGO_SIZE_PX = 60;
inline constexpr int LOGO_X_PX = 80;
inline constexpr int LOGO_Y_PX = 95;

/**
 * Colour the cells no policy occupies are filled with.
 *
 * A fallback only. The arena renders lit, so the green in a frame is not
 * the green the scene declares, and a filler cell mixed from the XML would
 * show as a visible square. The colour is sampled from the footage instead
 * and this constant is what stands in when sampling fails.
 */
inline constexpr const char* BACKGROUND_FALLBACK = "0x3D9356";

/**
 * Everything the run can be pointed at, and the two files it writes.
 */
struct Options {
  int seed = 0;  ///< the tour every tile walks
  std::string binary = "build/teleop-walking-benchmark";
  std::string work = "build/preview-clips";  ///< where the clips are kept
  std::string video = "assets/preview.mp4";  ///< the published clip
  std::string image = "assets/preview.jpg";  ///< the published still
  double still_s = -1.0;                     ///< still time, or -1 to derive
  double clip_s = CLIP_S;                    ///< closing seconds kept
  int fps = OUT_FPS;                         ///< rate the clip is written at
  bool rerecord = false;                     ///< ignore clips already there
  std::string logo;                          ///< image to use as the wordmark
};

/**
 * Refuse a path that would end the quoting of a shell command.
 *
 * Every path here is handed to ffmpeg through a shell, single quoted the
 * way the harness quotes its own encoder command. A quote inside one would
 * not be a broken filename so much as an arbitrary command, so the paths
 * are checked once, here, rather than escaped at each of the four places
 * they are used.
 *
 * @param[in] path  the path to check
 * @param[in] what  what it names, for the message
 * @throws std::runtime_error if the path carries a single quote
 * @exceptsafe strong
 */
void quotable(
    const std::string& path,
    const std::string& what
) {
  if (path.find('\'') != std::string::npos) {
    throw std::runtime_error("preview: " + what + " cannot contain a quote");
  }
}

/**
 * Run a command and require it to succeed.
 *
 * Echoed before it runs, because a preview takes thirteen tours to build
 * and the thing worth knowing when one fails is which policy the harness
 * was on.
 *
 * @param[in] cmd  the command line
 * @throws std::runtime_error if the command cannot be started or exits
 *         non-zero
 * @exceptsafe basic
 */
void run(const std::string& cmd) {
  std::cout << "preview: " << cmd << std::endl;
  const int status = std::system(cmd.c_str());
  if (status != 0) {
    throw std::runtime_error("preview: command failed: " + cmd);
  }
}

/**
 * Run a recording and require only that it produced a clip.
 *
 * The harness reports the run itself through its exit status: 0 for a tour
 * that completed, 1 for one that did not, and 2 for a harness or checkpoint
 * error. Most of this field falls on most seeds, and a fall is what the
 * preview is for, so a status of 1 is a result rather than a failure here.
 * Only 2 stops the build — and the clip is checked for by the caller, since
 * a run that ended before the encoder started leaves the grid a cell short.
 *
 * @param[in] cmd  the command line
 * @throws std::runtime_error if the harness could not be started or failed
 * @exceptsafe basic
 */
void record_run(const std::string& cmd) {
  std::cout << "preview: " << cmd << std::endl;
  const int status = std::system(cmd.c_str());
  if (status == -1) {
    throw std::runtime_error("preview: cannot start the harness");
  }
  const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  if (code != 0 && code != 1) {
    throw std::runtime_error("preview: the harness failed: " + cmd);
  }
}

/**
 * Read one line of output from a command.
 *
 * @param[in] cmd  the command line
 * @returns its first line, without the newline, or an empty string
 * @exceptsafe basic
 */
std::string capture(const std::string& cmd) {
  FILE* pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) return {};
  std::string out;
  int c = 0;
  while ((c = std::fgetc(pipe)) != EOF && c != '\n') {
    out.push_back(static_cast<char>(c));
  }
  pclose(pipe);
  return out;
}

/**
 * How long a clip is, in seconds.
 *
 * Asked of the file rather than of the run that wrote it, so that clips
 * left over from an earlier seed are padded by what they actually contain.
 *
 * @param[in] path  an mp4
 * @returns its duration, or 0 if ffprobe could not say
 * @exceptsafe basic
 */
double duration_s(const std::string& path) {
  const std::string out = capture(
      "ffprobe -v error -show_entries format=duration"
      " -of default=nw=1:nk=1 '" +
      path + "'"
  );
  try {
    return out.empty() ? 0.0 : std::stod(out);
  } catch (const std::exception&) {
    return 0.0;
  }
}

/**
 * Sample the arena's rendered green from a clip.
 *
 * Takes the top left corner of the first frame, which is the craned stance:
 * the robot is at the centre of the shot and the corner is arena. Scaling
 * the patch to a single pixel averages away the codec's ringing, so the
 * filler cells match the tiles beside them rather than nearly matching
 * them.
 *
 * @param[in] path  a recorded clip
 * @returns the colour as `0xRRGGBB`, or the fallback if sampling fails
 * @exceptsafe basic
 */
std::string background(const std::string& path) {
  const std::string cmd = "ffmpeg -v error -i '" + path +
                          "' -vf crop=16:16:0:0,scale=1:1 -frames:v 1"
                          " -f rawvideo -pix_fmt rgb24 - 2>/dev/null";
  FILE* pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) return BACKGROUND_FALLBACK;
  unsigned char rgb[3] = {0, 0, 0};
  const size_t got = std::fread(rgb, 1, sizeof(rgb), pipe);
  pclose(pipe);
  if (got != sizeof(rgb)) return BACKGROUND_FALLBACK;
  char hex[16] = {};
  std::snprintf(hex, sizeof(hex), "0x%02X%02X%02X", rgb[0], rgb[1], rgb[2]);
  return hex;
}

/**
 * Record one tour per policy, unless the clips are already there.
 *
 * One invocation each, in sequence. The harness refuses `--record` under
 * `--parallel` and holds one GL context per process, so this is not a loop
 * that wants widening; it is thirteen tours, which is a minute of simulated
 * walking apiece.
 *
 * Every run records itself and there is no flag to stop it, so the runs are
 * pointed at a file beside the clips. The published campaign is a fixed set
 * of seeds and these thirteen runs repeat thirteen of them: appended to it
 * they would be counted twice by anything averaging the file.
 *
 * @param[in] opts  seed, harness and working directory
 * @returns the clips, in grid order
 * @throws std::runtime_error if the harness fails or writes nothing
 * @exceptsafe basic
 */
std::vector<std::string> record(const Options& opts) {
  std::error_code ec;
  std::filesystem::create_directories(opts.work, ec);
  if (ec) {
    throw std::runtime_error("preview: cannot create " + opts.work);
  }

  std::vector<std::string> clips;
  for (const char* policy : POLICIES) {
    const std::string path = opts.work + "/" + policy + ".mp4";
    const bool have =
        std::filesystem::exists(path) && std::filesystem::file_size(path) > 0;
    if (opts.rerecord || !have) {
      std::ostringstream cmd;
      const std::string log = opts.work + "/" + policy + ".log";
      cmd << "'" << opts.binary << "' --policy " << policy << " --seed "
          << opts.seed << " --record '" << opts.work << "' --csv '" << opts.work
          << "/preview.csv' > '" << log << "' 2>&1; status=$?; cat '" << log
          << "'; exit $status";
      record_run(cmd.str());
    } else {
      std::cout << "preview: keeping " << path << std::endl;
    }
    if (!std::filesystem::exists(path)) {
      throw std::runtime_error("preview: no clip written for " + path);
    }
    clips.push_back(path);
  }
  return clips;
}

/**
 * When the crane let go, in seconds of simulated time.
 *
 * Read from what the harness printed rather than assumed, because the
 * release is the first control step at or after `INIT_DURATION_S` and not
 * that constant itself: the loop's clock accumulates in steps of `PERIOD_S`,
 * so the step that clears three seconds is the one after the one that
 * should have. Every run releases at the same moment whatever policy is
 * being driven, so the first log that says so answers for all thirteen. The
 * fallback is that same step computed, which is what a set of clips
 * recorded before this program kept logs will use.
 *
 * @param[in] work  the working directory the logs are kept in
 * @returns the release time, in seconds
 * @exceptsafe basic
 */
double release_s(const std::string& work) {
  const std::string out = capture(
      "grep -h -m1 -o 'crane released at t=[0-9.]*' '" + work +
      "'/*.log 2>/dev/null | head -1"
  );
  const std::string mark = "t=";
  const size_t at = out.find(mark);
  if (at != std::string::npos) {
    try {
      return std::stod(out.substr(at + mark.size()));
    } catch (const std::exception&) {
    }
  }
  return INIT_DURATION_S + PERIOD_S;
}

/**
 * When a punch lands, in seconds of simulated time.
 *
 * The harness draws the campaign when the crane lets go, so punch `i` lands
 * `PUNCH_DELAY_S` into waypoint `i` — far enough in that the policy has
 * taken the new demand, and the same instant in every tile.
 *
 * @param[in] work   the working directory the logs are kept in
 * @param[in] index  which punch, counting from zero
 * @returns when it lands, in seconds
 * @exceptsafe basic
 */
double punch_s(
    const std::string& work,
    int index
) {
  return release_s(work) + LEAD_IN_S + PUNCH_DELAY_S +
         POINT_S * static_cast<double>(index);
}

/**
 * How long the run actually is, in seconds.
 *
 * Taken from the longest clip rather than from `TOUR_S`, because a run is
 * the crane's ramp and then the tour: a policy that walks the whole thing
 * records about three seconds more than the tour's own clock. Padding to
 * the clock instead would cut the finish off every tile, which is the part
 * worth watching.
 *
 * @param[in] clips  the recorded clips
 * @returns the length every tile is padded to
 * @throws std::runtime_error if no clip has a readable duration
 * @exceptsafe basic
 */
double full_length(const std::vector<std::string>& clips) {
  double longest = 0.0;
  for (const std::string& clip : clips) {
    longest = std::max(longest, duration_s(clip));
  }
  if (longest < TOUR_S) {
    throw std::runtime_error(
        "preview: no clip reaches the tour's own clock; the field is short"
    );
  }
  return longest;
}

/**
 * Build the filtergraph that lays the clips out.
 *
 * Every tile is scaled to a quarter of the frame, held on its last frame
 * until the longest run is up, and cut back to that length so a clip that
 * overran cannot stretch the montage. The tiles are laid onto a filled
 * background rather than stacked, because two of the sixteen cells hold
 * nothing and a background that is already the arena's green is what makes
 * those cells invisible.
 *
 * The graph ends in two outputs rather than one. `[v]` is the clip, cropped
 * to its closing seconds if that was asked for, and `[img]` is the single
 * frame the still is made from, taken before the crop so that `--still`
 * always means the same moment of the run whatever `--clip` is set to.
 *
 * @param[in] clips   the recorded clips, in grid order
 * @param[in] opts    wordmark and framing
 * @param[in] fill    background colour, as `0xRRGGBB`
 * @param[in] length  what every tile is padded to, in seconds
 * @returns the graph, ready to be single quoted into a command
 * @exceptsafe basic
 */
std::string graph(
    const std::vector<std::string>& clips,
    const Options& opts,
    const std::string& fill,
    double length
) {
  std::ostringstream g;
  g << "color=c=" << fill << ":s=" << OUT_WIDTH << "x" << OUT_HEIGHT
    << ":r=" << RECORD_FPS << ":d=" << length << "[bg]";

  for (size_t i = 0; i < clips.size(); ++i) {
    const double have = duration_s(clips[i]);
    const double pad = have < length ? length - have : 0.0;
    g << ";[" << i << ":v]scale=" << TILE_WIDTH << ":" << TILE_HEIGHT
      << ",tpad=stop_mode=clone:stop_duration=" << pad << ",trim=0:" << length
      << ",setpts=PTS-STARTPTS,fps=" << RECORD_FPS << "[t" << i << "]";
  }

  std::string base = "bg";
  for (size_t i = 0; i < clips.size(); ++i) {
    const int slot = static_cast<int>(i);
    const int cell = slot < LOGO_CELL ? slot : slot + 1;
    const int x = (cell % COLS) * TILE_WIDTH;
    const int y = (cell / COLS) * TILE_HEIGHT;
    const std::string out = "o" + std::to_string(i);
    g << ";[" << base << "][t" << i << "]overlay=x=" << x << ":y=" << y
      << ":shortest=0[" << out << "]";
    base = out;
  }

  const int logo_x = (LOGO_CELL % COLS) * TILE_WIDTH + LOGO_X_PX;
  const int logo_y = (LOGO_CELL / COLS) * TILE_HEIGHT + LOGO_Y_PX;
  if (opts.logo.empty()) {
    g << ";[" << base << "]drawtext=fontfile=" << LOGO_FONT
      << ":text=" << LOGO_TEXT << ":fontcolor=white:fontsize=" << LOGO_SIZE_PX
      << ":x=" << logo_x << ":y=" << logo_y;
  } else {
    g << ";[" << base << "][" << clips.size() << ":v]overlay=x=" << logo_x
      << ":y=" << logo_y;
  }
  g << ",format=yuv420p,split=2[whole][shot]";

  const double keep = opts.clip_s > 0.0 ? opts.clip_s : length;
  const double start = keep < length ? length - keep : 0.0;
  if (start > 0.0) {
    g << ";[whole]trim=start=" << start << ",setpts=PTS-STARTPTS[v]";
  } else {
    g << ";[whole]null[v]";
  }

  const double frame_s = 1.0 / RECORD_FPS;
  g << ";[shot]trim=start=" << opts.still_s
    << ":end=" << (opts.still_s + 2.0 * frame_s) << ",setpts=PTS-STARTPTS[img]";
  return g.str();
}

/**
 * Compose the grid and write both assets, in one pass.
 *
 * The clip is encoded straight out of the filtergraph rather than composed
 * to an intermediate and re-encoded, so nothing published here is a second
 * generation of anything, and the still is written by the same command from
 * the same frames. Two outputs, one decode of the thirteen clips.
 *
 * @param[in] clips  the recorded clips, in grid order
 * @param[in] opts   framing, wordmark and where the two files go
 * @throws std::runtime_error if ffmpeg fails
 * @exceptsafe basic
 */
void compose(
    const std::vector<std::string>& clips,
    const Options& opts
) {
  const std::string fill = background(clips.front());
  const double length = full_length(clips);

  Options timed = opts;
  if (timed.still_s < 0.0) {
    timed.still_s = punch_s(opts.work, STILL_PUNCH) + STILL_AFTER_S;
    std::cout << "preview: still at " << timed.still_s << " s, "
              << STILL_AFTER_S << " s after punch " << STILL_PUNCH << std::endl;
  }
  if (timed.still_s > length) {
    throw std::runtime_error("preview: the still is timed past the run");
  }

  std::ostringstream cmd;
  cmd << "ffmpeg -hide_banner -loglevel error -y";
  for (const std::string& clip : clips) {
    cmd << " " << INPUT_FLAGS << " -i '" << clip << "'";
  }
  if (!opts.logo.empty()) {
    cmd << " -loop 1 -framerate " << RECORD_FPS << " -i '" << opts.logo << "'";
  }
  cmd << " -filter_complex '" << graph(clips, timed, fill, length) << "'"
      << " -map '[v]' -r " << opts.fps << " " << ENCODE << " '" << opts.video
      << "'"
      << " -map '[img]' -frames:v 1 -q:v 2 '" << opts.image << "'";
  run(cmd.str());
}

/**
 * Print the command line synopsis.
 *
 * @exceptsafe basic
 */
void usage() {
  std::cout
      << "usage: preview [--seed N] [--work DIR] [--video FILE] "
         "[--image FILE]\n"
         "               [--still S] [--clip S] [--fps N] [--rerecord]\n"
         "               [--logo FILE]\n\n"
         "  --seed N      the tour every policy walks (default 0)\n"
         "  --work DIR    where clips and the montage are kept "
         "(default build/preview-clips)\n"
         "  --video FILE  the published clip (default assets/preview.mp4)\n"
         "  --image FILE  the published still (default assets/preview.jpg)\n"
         "  --still S     when the still is taken, in seconds of the run\n"
         "                (default: 0.4 s after punch 10, about 53.5 s)\n"
         "  --clip S      closing seconds published (default 0, the whole "
         "run)\n"
         "  --fps N       rate the clip is written at (default 50, the "
         "recording's)\n"
         "  --rerecord    record every policy again, ignoring kept clips\n"
         "  --logo FILE   an image for the free cell instead of the wordmark\n"
         "  --binary PATH the harness to record with\n";
}

/**
 * Read the command line.
 *
 * @param[in] argc  argument count
 * @param[in] argv  argument values
 * @returns what was asked for
 * @throws std::runtime_error on an unknown or incomplete flag
 * @exceptsafe basic
 */
Options parse(
    int argc,
    char** argv
) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto next = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(flag + ": expected " + what);
      }
      return argv[++i];
    };
    if (flag == "--seed") {
      opts.seed = std::stoi(next("a seed"));
    } else if (flag == "--work") {
      opts.work = next("a directory");
    } else if (flag == "--video") {
      opts.video = next("a file");
    } else if (flag == "--image") {
      opts.image = next("a file");
    } else if (flag == "--still") {
      opts.still_s = std::stod(next("a time in seconds"));
    } else if (flag == "--clip") {
      opts.clip_s = std::stod(next("a length in seconds"));
    } else if (flag == "--fps") {
      opts.fps = std::stoi(next("a frame rate"));
    } else if (flag == "--rerecord") {
      opts.rerecord = true;
    } else if (flag == "--logo") {
      opts.logo = next("an image");
    } else if (flag == "--binary") {
      opts.binary = next("a path");
    } else if (flag == "--help" || flag == "-h") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("preview: unknown flag " + flag);
    }
  }
  if (opts.still_s >= 0.0 && opts.still_s > TOUR_S + INIT_DURATION_S) {
    throw std::runtime_error("--still: must be within the run");
  }
  if (opts.clip_s < 0.0) {
    throw std::runtime_error("--clip: cannot be negative");
  }
  if (opts.fps < 1) {
    throw std::runtime_error("--fps: must be at least one");
  }
  quotable(opts.binary, "the harness path");
  quotable(opts.work, "the working directory");
  quotable(opts.video, "the clip path");
  quotable(opts.image, "the still path");
  quotable(opts.logo, "the logo path");
  return opts;
}

}  // namespace preview

/**
 * Record the field, compose the grid, and write both assets.
 *
 * @param[in] argc  argument count
 * @param[in] argv  argument values
 * @returns 0 on success, 1 if anything on the way failed
 * @exceptsafe no-throw
 */
int main(
    int argc,
    char** argv
) {
  try {
    const preview::Options opts = preview::parse(argc, argv);
    if (!std::filesystem::exists(opts.binary)) {
      throw std::runtime_error(
          "preview: " + opts.binary + " is not built; run make first"
      );
    }
    const std::vector<std::string> clips = preview::record(opts);
    preview::compose(clips, opts);
    std::cout << "preview: wrote " << opts.image << " and " << opts.video
              << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
