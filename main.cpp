#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <ft2build.h>
#include FT_FREETYPE_H

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <thread>
#include <stdexcept>
#include <string>
#include <vector>

namespace benchmark {

/**
 * Set when SIGINT arrives, polled by the simulation loop.
 *
 * A run owns the only copy of the world it is stepping, so the handler
 * cannot tear anything down itself: it flips this flag and the loop leaves
 * at the top of its next iteration with the report it has so far, which is
 * why an interrupted sweep still prints its table. Atomic because the
 * handler runs on whichever thread the signal lands on.
 */
inline std::atomic<bool> g_stop{false};

/**
 * Joints every policy in the field drives, in MuJoCo order.
 *
 * The G1 as sold with 29 degrees of freedom: twelve in the legs, three at
 * the waist and seven per arm. Every vector crossing the policy boundary is
 * this long, which is what lets one `Output` be read the same way whichever
 * checkpoint produced it.
 */
inline constexpr int NUM_MOTOR = 29;

/**
 * Where every run records itself, unless `--csv` names somewhere else.
 *
 * Relative to the working directory, like the scene the harness loads, and
 * `run.sh` changes to the project root before exec-ing the binary. Every
 * run appends here whether it was a sweep, a single seed or a watched
 * preview: a result that only ever reached a terminal is a result nobody
 * can pool later, and a run worth watching is a run worth keeping.
 */
inline const char* DEFAULT_CSV = "results/result.csv";

/**
 * Seed used when `--seed` is absent.
 *
 * One seed picks the tour and the punch campaign together, so two runs at
 * the same seed are the same task and reproduce bit for bit. The default
 * exists so that a bare `./run.sh` is that reproducible case rather than an
 * arbitrary one.
 */
inline constexpr uint32_t DEFAULT_SEED = 0u;

/**
 * Quiet time between the crane letting go and the first target, in seconds.
 *
 * A policy takes over from a stance it did not choose and most of the field
 * needs a step or two to settle. Scoring from the release would charge that
 * transient to the first waypoint, which says more about the handover than
 * about the policy.
 */
inline constexpr double LEAD_IN_S = 2.0;

/**
 * Radius of the disc the tour draws its targets from, in metres.
 *
 * One metre keeps every waypoint inside the floor mark and inside the
 * camera framing, so a recorded run never walks out of shot, and it keeps
 * the leg between two targets short enough that one target clock is a fair
 * budget for crossing it.
 */
inline constexpr double DEFAULT_RADIUS_M = 1.0;

/**
 * Length of the scored walk, in simulated seconds.
 *
 * Long enough that the punch campaign climbs from its floor to full force
 * and that a policy which drifts has time to show it; short enough that a
 * hundred seeds across the whole field still fits in a sitting.
 */
inline constexpr double WALK_S = 60.0;

/**
 * Number of targets a tour visits.
 *
 * Only ever used to derive `POINT_S`. The tour itself is built from the
 * walk length and the target clock, so a run fills its full minute
 * however this is set.
 */
inline constexpr int WAYPOINTS = 12;

/**
 * Time a target stays current before the next replaces it, in seconds.
 *
 * The clock never waits for the robot to arrive. Five seconds is about what
 * an unhurried policy needs to cross the disc, so a candidate that cannot
 * keep up is scored wherever the clock left it rather than being handed the
 * tour at its own pace.
 */
inline constexpr double POINT_S = WALK_S / WAYPOINTS;

/**
 * Centre of the target disc in the released frame, in metres.
 *
 * Targets are drawn in the robot's own frame at the moment the crane lets
 * go, not in world coordinates, so every policy walks the same shape
 * whatever heading it ended up holding. Offsetting the centre forward makes
 * the opening target a walk rather than a turn in place. `centre_check`
 * refuses to start a run whose `floor_mark` geom disagrees with this.
 */
inline constexpr double CENTRE_X_M = 0.3;

/**
 * Centre of the target disc in the released frame, in metres.
 *
 * Zero keeps the disc symmetric about the release heading, so a seed is as
 * likely to send the robot left as right and no policy is flattered by
 * turning better one way than the other.
 */
inline constexpr double CENTRE_Y_M = 0.0;

/**
 * Reflected inertia of a 5020 actuator, in kg m^2.
 *
 * The gains below are derived from the armature rather than tuned joint by
 * joint, so every joint closes at the same bandwidth and no policy is
 * helped by a limb that happens to track better than the rest.
 */
inline const double ARMATURE_5020 = 0.003609725;

/**
 * Natural frequency the joint gains are designed for, in rad/s.
 *
 * Ten hertz is quick enough that a joint substantially closes on a new
 * target inside one 20 ms control period, and slow enough that the
 * simulated actuator is not asked for torque a real one could not hold.
 * Shared by every joint, which is what makes each gain a pure function of
 * the inertia behind it.
 */
inline const double NATURAL_FREQ = 10 * 2.0 * 3.1415926535;

/**
 * Position gain for a joint driven by one 5020, in Nm/rad.
 *
 * kp = J w^2: the stiffness that gives a joint of this inertia the shared
 * natural frequency. Derived rather than tuned, so the number carries no
 * opinion about any particular policy.
 */
inline const double STIFFNESS_5020 =
    ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ;

/**
 * Reflected inertia of a 7520-14 actuator, in kg m^2.
 *
 * The 7520 appears in two variants whose gearing differs, and gearing
 * enters the inertia seen at the joint squared, so the two cannot share one
 * figure however similar the motors are.
 */
inline const double ARMATURE_7520_14 = 0.010177520;

/**
 * Position gain for a joint driven by one 7520-14, in Nm/rad.
 */
inline const double STIFFNESS_7520_14 =
    ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ;

/**
 * Reflected inertia of a 7520-22 actuator, in kg m^2.
 *
 * The heaviest joints on the robot, the hips and the knees, sit behind
 * these, and at seven times the 5020 figure they set the stiffest gains in
 * `KPS`.
 */
inline const double ARMATURE_7520_22 = 0.025101925;

/**
 * Position gain for a joint driven by one 7520-22, in Nm/rad.
 */
inline const double STIFFNESS_7520_22 =
    ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ;

/**
 * Reflected inertia of a 4010 actuator, in kg m^2.
 *
 * Drives the wrist pitch and yaw, the only joints light enough that their
 * gains barely matter to whether the robot stays upright. They are derived
 * the same way regardless, so that no joint on the robot is left untuned.
 */
inline const double ARMATURE_4010 = 0.00425;

/**
 * Position gain for a joint driven by one 4010, in Nm/rad.
 */
inline const double STIFFNESS_4010 =
    ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ;

/**
 * Position gain per joint, in MuJoCo order, in Nm/rad.
 *
 * Each entry is the stiffness of whatever drives that joint. Both ankles
 * and the waist roll and pitch carry twice a single-actuator figure because
 * a parallel linkage puts two motors behind one joint, and it is the pair's
 * stiffness the controller sees. Held as float because that is what
 * `MotorCommand` hands to the policies.
 */
inline const Eigen::Vector<float, NUM_MOTOR> KPS = {
    STIFFNESS_7520_22, STIFFNESS_7520_22,    STIFFNESS_7520_14,
    STIFFNESS_7520_22, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
    STIFFNESS_7520_22, STIFFNESS_7520_22,    STIFFNESS_7520_14,
    STIFFNESS_7520_22, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
    STIFFNESS_7520_14, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
    STIFFNESS_5020,    STIFFNESS_5020,       STIFFNESS_5020,
    STIFFNESS_5020,    STIFFNESS_5020,       STIFFNESS_4010,
    STIFFNESS_4010,    STIFFNESS_5020,       STIFFNESS_5020,
    STIFFNESS_5020,    STIFFNESS_5020,       STIFFNESS_5020,
    STIFFNESS_4010,    STIFFNESS_4010,
};

/**
 * Damping ratio the joint damping is designed for.
 *
 * Deliberately overdamped. With targets arriving only every 20 ms a
 * critically damped joint overshoots between updates, and overshoot at an
 * ankle is what turns a small tracking error into a fall.
 */
inline const double DAMPING_RATIO = 2;

/**
 * Velocity gain for a joint driven by one 5020, in Nm s/rad.
 *
 * kd = 2 z J w: the damping that pairs with `STIFFNESS_5020` at the shared
 * ratio and frequency.
 */
inline const double DAMPING_5020 =
    2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ;

/**
 * Velocity gain for a joint driven by one 7520-14, in Nm s/rad.
 */
inline const double DAMPING_7520_14 =
    2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ;

/**
 * Velocity gain for a joint driven by one 7520-22, in Nm s/rad.
 */
inline const double DAMPING_7520_22 =
    2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ;

/**
 * Velocity gain for a joint driven by one 4010, in Nm s/rad.
 */
inline const double DAMPING_4010 =
    2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ;

/**
 * Velocity gain per joint, in MuJoCo order, in Nm s/rad.
 *
 * Laid out exactly like `KPS`, doubled at the same parallel-linkage joints,
 * so a joint's stiffness and its damping are always read from the same
 * position in two tables that cannot drift apart.
 */
inline const Eigen::Vector<float, NUM_MOTOR> KDS = {
    DAMPING_7520_22, DAMPING_7520_22,    DAMPING_7520_14,
    DAMPING_7520_22, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
    DAMPING_7520_22, DAMPING_7520_22,    DAMPING_7520_14,
    DAMPING_7520_22, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
    DAMPING_7520_14, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
    DAMPING_5020,    DAMPING_5020,       DAMPING_5020,
    DAMPING_5020,    DAMPING_5020,       DAMPING_4010,
    DAMPING_4010,    DAMPING_5020,       DAMPING_5020,
    DAMPING_5020,    DAMPING_5020,       DAMPING_5020,
    DAMPING_4010,    DAMPING_4010,
};

/**
 * Stance every run starts from, in MuJoCo order, in radians.
 *
 * The crane ramps to this pose over `INIT_DURATION_S` and then lets go, and
 * any joint a policy declines to own is held here for the rest of the run.
 * It is deliberately not the pose any one checkpoint would have picked:
 * what the benchmark measures is how well a policy takes over a stance it
 * did not choose.
 */
inline const Eigen::Vector<double, NUM_MOTOR> DEFAULT_ANGLES = {
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669,
    -0.363, 0.0, 0.0, 0.0,   0.0,    0.2, 0.2,    0.0, 0.6, 0.0,
    0.0,    0.0, 0.2, -0.2,  0.0,    0.6, 0.0,    0.0, 0.0
};

/**
 * Control period, in seconds.
 *
 * Fifty hertz, the rate this field was trained and deployed at. The loop
 * holds it against simulated time rather than wall time, so a policy that
 * answers in 3 ms and one that takes 30 ms are shown exactly the same
 * world and are separated by their walking, not by their latency.
 */
inline constexpr double PERIOD_S = 0.02;

/**
 * Time the crane spends ramping to the stance before letting go, in
 * seconds.
 *
 * The base is pinned throughout, so this is not a balancing test: three
 * seconds is simply long enough for the joints to reach the stance under
 * their own PD without the ramp itself throwing the robot at release.
 */
inline constexpr double INIT_DURATION_S = 3.0;

/**
 * Pelvis height below which a run is called a fall, in metres.
 *
 * Standing puts the pelvis near 0.79 m, so a fifth of a metre is far below
 * any crouch a policy might intend and is reached only once the robot is
 * already on the floor. Tested only after release, so it never fires
 * against the pinned base.
 */
inline constexpr double FALL_PELVIS_Z = 0.20;

/**
 * Slack allowed past the expected run length before a timeout, in seconds.
 *
 * A tour ends when its clock runs out rather than when the robot arrives,
 * so the only way to exceed the expected length is for the loop itself to
 * stall. The margin is small on purpose: it is a backstop, not a budget.
 */
const double TIMEOUT_MARGIN_S = 5.0;

/**
 * Frame rate of a recorded run, in frames per second.
 *
 * Deliberately 1/`PERIOD_S`. One frame per control step means playback runs
 * at exactly real time however fast or slow the machine computed the run,
 * so two clips of two policies can be watched against each other.
 */
constexpr int RECORD_FPS = 50;

/**
 * Font the policy name on the floor is set in.
 *
 * Vendored under `assets/` rather than resolved from the system, so a clip
 * recorded on one machine carries the same lettering as one recorded
 * anywhere else.
 */
inline constexpr char FLOOR_FONT[] = "assets/JetBrainsMono.ttf";

/**
 * Ceiling on punch force at the end of the ramp, in newtons.
 *
 * Set so that a policy holding a good stance survives a hit at full
 * strength and one already off balance does not. The campaign is there to
 * separate the field, not to knock all of it over.
 */
inline constexpr double FORCE_MAX_N = 600.0;

/**
 * Smallest fraction of the current ceiling a punch may draw.
 *
 * Every punch lands somewhere in [FORCE_SCALE_MIN, 1] of the ceiling, so
 * even late in a run a policy cannot count on the next hit being the
 * hardest one it has already survived.
 */
inline constexpr double FORCE_SCALE_MIN = 0.5;

/**
 * Fraction of `FORCE_MAX_N` the punch ceiling starts at.
 *
 * The campaign opens at a third rather than at nothing, so the first
 * waypoints are already a disturbance a policy has to answer instead of a
 * stretch of quiet walking that separates nobody.
 */
inline constexpr double RAMP_FLOOR = 1.0 / 3.0;

/**
 * Time over which the punch ceiling ramps from `RAMP_FLOOR` to
 * `FORCE_MAX_N`, in seconds.
 *
 * Measured from the first punch and matched to the walk length, so the
 * campaign is gentler while a policy is still finding its feet and hardest
 * at the end. A run that reaches the last waypoint has faced the whole
 * range.
 */
inline constexpr double RAMP_S = 60.0;

/**
 * How long a punch pushes, in seconds.
 *
 * Four control periods: long enough that the policy sees the disturbance in
 * its own observations and has to answer it, short enough to read as an
 * impulse rather than as someone leaning on the robot.
 */
inline constexpr double DURATION_S = 0.08;

/**
 * How long a punch arrow stays drawn after the push ends, in seconds.
 *
 * The push is four frames at `RECORD_FPS`, easy to miss on playback.
 * Holding the arrow half a second costs nothing physical — it is drawn, not
 * applied — and makes the cause of a fall visible in the clip.
 */
inline constexpr double PUNCH_HOLD_S = 0.5;

/**
 * Multiplier on the drawn size of a punch arrow.
 *
 * Purely cosmetic. An arrow scaled to the true contact geometry is nearly
 * invisible beside the robot, so it is drawn at twice size to read at 720p.
 */
inline constexpr double PUNCH_ARROW_SCALE = 2.0;

/**
 * Scene texture the policy label is painted into.
 *
 * The scene ships this texture blank and the label is drawn into it at load
 * time from the policy name, so one XML serves the whole field and a clip
 * can never be mislabelled by a stale asset.
 */
inline constexpr char TEXTURE_NAME[] = "floor_label";

/**
 * Height of the floor lettering, as a fraction of the texture width.
 *
 * Expressed against the texture rather than in pixels so the label keeps
 * its proportions if the scene ever ships a larger one.
 */
inline constexpr double FLOOR_EM_FRACTION = 0.08;

/**
 * Extra space between floor glyphs, as a fraction of the em size.
 *
 * Letters this size on a surface seen at a shallow angle run into each
 * other; tracking them apart is what keeps a name like `run_residual`
 * readable in a recording.
 */
inline constexpr double FLOOR_TRACKING_EM = 0.18;

/**
 * Channels in the floor label texture.
 *
 * Named because the painter walks raw texture bytes: `paint_floor_texture`
 * refuses a texture that is not RGB rather than writing at the wrong
 * stride and corrupting the scene.
 */
inline constexpr int RGB_CHANNELS = 3;

/**
 * Joint names in the order every vector in this file is indexed by.
 *
 * This array is the ordering convention rather than a description of one:
 * `handles_make` resolves each name once against the model, and gains,
 * stances, observations and motor targets are all read at the same index
 * afterwards. Each actuator carries its joint's name, so one lookup covers
 * both, and a name the scene does not carry stops the run instead of
 * silently shifting every index after it.
 */
inline const char* JOINT_NAMES[NUM_MOTOR] = {
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint"
};

/**
 * How many bins the joints are reported in.
 *
 * Five rather than twenty-nine because the question a reader brings to this
 * file is which part of the robot a policy is spending itself on, not which
 * of three wrist axes. The per-joint detail exists for the length of a leg
 * and is summed away before it is written.
 */
inline constexpr int NUM_GROUPS = 5;

/**
 * The bins, in the order their columns appear.
 *
 * Split proximal from distal in both limbs, because that is where these
 * joints genuinely differ: the proximal ones carry the robot's weight and
 * do nearly all the work, while the distal ones are light, fast and where
 * a policy's chatter shows up first. Lumping an ankle in with a hip would
 * hide exactly the contrast the two columns exist to show.
 */
inline const char* GROUP_NAMES[NUM_GROUPS] = {
    "legs_upper",  ///< hips and knees, both sides
    "legs_lower",  ///< ankles, both sides
    "waist",       ///< the three waist axes
    "arms_upper",  ///< shoulders and elbows, both sides
    "arms_lower"   ///< wrists, both sides
};

/**
 * Which bin each joint falls in, parallel to `JOINT_NAMES`.
 */
inline constexpr int JOINT_GROUP[NUM_MOTOR] = {
    0, 0, 0, 0,  // left hip pitch, roll, yaw, knee
    1, 1,        // left ankle pitch, roll
    0, 0, 0, 0,  // right hip pitch, roll, yaw, knee
    1, 1,        // right ankle pitch, roll
    2, 2, 2,     // waist yaw, roll, pitch
    3, 3, 3, 3,  // left shoulder pitch, roll, yaw, elbow
    4, 4, 4,     // left wrist roll, pitch, yaw
    3, 3, 3, 3,  // right shoulder pitch, roll, yaw, elbow
    4, 4, 4      // right wrist roll, pitch, yaw
};

/**
 * Every joint lands in a bin that exists, and every bin has a joint.
 */
constexpr bool groups_are_sound() {
  int seen[NUM_GROUPS] = {};
  for (const int g : JOINT_GROUP) {
    if (g < 0 || g >= NUM_GROUPS) return false;
    ++seen[g];
  }
  for (const int n : seen) {
    if (n == 0) return false;
  }
  return true;
}
static_assert(
    groups_are_sound(),
    "every joint needs a bin and every bin a joint"
);

/**
 * Camera azimuth about the arena centre, in degrees.
 *
 * Fixed rather than following the robot. A static frame makes drift across
 * the disc plain, where a chase camera would hide exactly the error the
 * tour exists to measure.
 */
inline constexpr double CAMERA_AZIMUTH_DEG = 180.0;

/**
 * Elevation the framing is designed around, in degrees.
 *
 * Looking down at forty-five degrees shows the floor mark and the target
 * marker together with the robot's own posture. `viewer_open` turns this
 * and the height below into the distance and elevation the camera is
 * actually given.
 */
inline constexpr double CAMERA_ELEVATION_DEG = -45.0;

/**
 * Height the design ray is struck from, in metres.
 *
 * The framing is specified as a viewpoint two metres up looking down at
 * `CAMERA_ELEVATION_DEG`, which is easier to reason about than a distance
 * and an angle about a lookat point; `viewer_open` does the conversion.
 */
inline constexpr double CAMERA_HEIGHT_M = 2.0;

/**
 * Distance the camera is brought in along the design ray, in metres.
 *
 * Tightens the frame on the arena once elevation and height have fixed its
 * direction, so the robot fills more of a 720p clip than the floor does.
 */
inline constexpr double CAMERA_DOLLY_IN_M = 1.0;

/**
 * Height the camera gains after the dolly, in metres.
 *
 * Coming in flattens the view; lifting afterwards restores enough of the
 * downward angle to keep the far side of the disc in frame.
 */
inline constexpr double CAMERA_RISE_M = 1.0;

/**
 * Vertical field of view, in degrees.
 *
 * Wide enough that a policy which has drifted a metre off the disc is still
 * in shot when it falls, which is usually the frame worth watching.
 */
inline constexpr double CAMERA_FOVY_DEG = 65.0;

/**
 * Width of the offscreen buffer a recording renders into, in pixels.
 *
 * The offscreen framebuffer is sized once at context creation, so a clip
 * has this resolution whatever the on-screen window is doing, including
 * when there is no visible window at all.
 */
constexpr int RECORD_WIDTH = 1280;

/**
 * Height of the offscreen buffer a recording renders into, in pixels.
 */
constexpr int RECORD_HEIGHT = 720;

/**
 * Scene geom that marks the arena centre.
 *
 * The tour is drawn around `CENTRE_X_M`/`CENTRE_Y_M` while the mark is
 * placed by the XML, and nothing in either file enforces agreement, so
 * `centre_check` compares them and refuses a run that would record the
 * robot circling something other than what is being scored.
 */
inline constexpr char CENTRE_GEOM[] = "floor_mark";

/**
 * Whether TensorRT plans are built with half precision enabled.
 *
 * Off. Half precision changes what a checkpoint computes by an amount that
 * varies with the network, and a comparison between policies should not
 * turn on which of them tolerated the cast.
 */
inline constexpr bool ENGINE_FP16 = false;

/**
 * Directory serialised TensorRT plans are cached under.
 *
 * Building a plan costs far more than running the tour does and the field
 * is thirteen networks, so the first run pays and the rest do not. Kept
 * under `build/` because a plan is valid only for the machine that built
 * it and must never be committed.
 */
const char* PLAN_CACHE_DIR = "build/trt";

/**
 * Scratch memory the TensorRT builder may use per network, in bytes.
 *
 * A ceiling, not an allocation: it lets the builder consider the faster
 * tactics for these small networks without the choice depending on what
 * else happens to be resident on the card.
 */
const size_t WORKSPACE_BYTES = 1ull << 30;

/**
 * Index of each MuJoCo joint in IsaacLab's ordering.
 *
 * MuJoCo walks the kinematic tree depth first, so a whole leg is
 * contiguous; IsaacLab orders joints breadth first across the robot, so all
 * the hips come before all the knees. The two agree almost nowhere, and a
 * policy trained in IsaacLab therefore has to be permuted on both sides of
 * `step`: observations in, actions out.
 */
inline const Eigen::Vector<int, NUM_MOTOR> MUJOCO_TO_ISAACLAB = {
    0,  6,  12, 1,  7,  13, 2,  8,  14, 3,  9,  15, 22, 4, 10,
    16, 23, 5,  11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
};

/**
 * Index of each IsaacLab joint in MuJoCo's ordering.
 *
 * The inverse of `MUJOCO_TO_ISAACLAB`, spelled out rather than inverted at
 * startup so that a policy can scatter an action vector back into MuJoCo
 * order in a single indexed pass.
 */
inline const Eigen::Vector<int, NUM_MOTOR> ISAACLAB_TO_MUJOCO = {
    0,  3,  6,  9,  13, 17, 1,  4,  7,  10, 14, 18, 2,  5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
};

/**
 * Everything a policy is allowed to know about the robot itself.
 *
 * Exactly what a G1 publishes over its own bus: joint encoders and the
 * pelvis IMU. There is deliberately no world position here — a real machine
 * does not have one — so a policy that wants to know where it is has to
 * infer it from the target error it is handed instead.
 */
struct RobotState {
  Eigen::Vector<double, NUM_MOTOR> motor_q;   ///< joint angles, rad
  Eigen::Vector<double, NUM_MOTOR> motor_dq;  ///< joint rates, rad/s
  Eigen::Vector4d imu_quat;                   ///< pelvis attitude, w x y z
  Eigen::Vector3d imu_gyro;                   ///< pelvis rates, rad/s, body
};

/**
 * One control step's worth of task, as `step` receives it.
 *
 * The task is given as the raw error to the current target in the body
 * frame, never as a velocity command. Turning that error into a command is
 * the policy's own business, because that is precisely where a checkpoint's
 * trained envelope lives: a candidate that widened its limits to make the
 * tour easier would no longer be measuring the same thing. `Limits` and
 * `command_from_target` are offered for policies that want the usual
 * answer.
 */
struct Input {
  const RobotState* state;  ///< robot as its own sensors see it

  double target_forward_m;  ///< target ahead of the pelvis, body frame, m
  double target_left_m;     ///< target left of the pelvis, body frame, m

  bool new_target;  ///< first step on this target; latches must reset

  const double* arm_targets;  ///< NUM_MOTOR poses for joints left unowned

  double control_time;  ///< seconds since the crane released

  double dist_m;       ///< straight-line distance to the target, m
  double yaw_err_rad;  ///< heading error, wrapped to [-pi, pi]
  bool has_target;     ///< false during lead-in and after the last target
};

/**
 * One control step's worth of answer, as `step` returns it.
 *
 * A policy declares per joint whether it is driving it. Whatever it leaves
 * unowned is held at `DEFAULT_ANGLES` on the shared gains, which is what
 * lets a lower-body checkpoint run this tour at all rather than being
 * disqualified for having no opinion about the arms.
 */
struct Output {
  double q_target[NUM_MOTOR];  ///< position targets, rad
  float kp[NUM_MOTOR];         ///< position gains, Nm/rad
  float kd[NUM_MOTOR];         ///< velocity gains, Nm s/rad

  bool owns[NUM_MOTOR];  ///< joints this policy drives; the rest are held
};

/**
 * The envelope one policy asks to be driven inside.
 *
 * Owned by the policy, not by the harness, because it describes what that
 * checkpoint was trained to accept: a command outside it is not a harder
 * task but an undefined one. The arrival thresholds are two-sided on
 * purpose — a single threshold at the edge of the robot's own footstep
 * jitter makes a policy step in and out of "arrived" every control period
 * and shuffle in place for the rest of the target's clock.
 */
struct Limits {
  double vx_min, vx_max;  ///< forward command range, m/s
  double vy_abs;          ///< strafe command magnitude, m/s
  double yaw_rate_abs;    ///< turn command magnitude, rad/s
  double speed_norm;      ///< cap on the planar command's norm, m/s

  double pos_reached_enter_m = 0.10;    ///< distance that counts as arrived
  double pos_reached_exit_m = 0.20;     ///< distance that gives it back
  double yaw_reached_enter_rad = 0.05;  ///< heading error counted as faced
  double yaw_reached_exit_rad = 0.12;   ///< heading error that gives it back

  double walk_kp_pos = 1.5;  ///< metres of error to m/s of command
  double walk_kp_yaw = 1.5;  ///< radians of error to rad/s of command
};

/**
 * The one interface every candidate in the field is reduced to.
 *
 * Thirteen checkpoints from thirteen projects, each with its own
 * observation layout, action scaling, history buffers and joint ordering,
 * meet here and nowhere else. Everything upstream of `step` — the tour, the
 * gains, the stance, the punches, the timing — is shared, so a difference
 * in the table is a difference between policies rather than between
 * harnesses.
 */
class ModelPolicy {
 public:
  virtual ~ModelPolicy() = default;

  /**
   * Load weights and bring the network to a state where `step` can be
   * timed.
   *
   * Called once, outside the run, so that a plan build or a first lazy
   * allocation lands here rather than in the step timings. Throwing marks
   * this policy as an error and leaves the rest of the field to run.
   *
   * @throws std::runtime_error if the weights are missing or unusable
   * @exceptsafe basic
   */
  virtual void init() = 0;

  /**
   * Answer one control step.
   *
   * Called every `PERIOD_S` of simulated time and timed, so it must not
   * block on anything but its own inference.
   *
   * @param[in] in  robot state and the error to the current target
   * @returns targets and gains for the joints this policy owns
   * @exceptsafe basic
   */
  virtual Output step(const Input& in) = 0;

  /**
   * Name this policy is selected, reported and recorded under.
   *
   * The same string names its directory under `policies/`, its `--policy`
   * argument, its row in the table and the label painted on the floor of
   * its clip.
   *
   * @returns a string that outlives the policy
   * @exceptsafe no-throw
   */
  virtual const char* name() const = 0;
};

/**
 * Everything that decides which tour a seed produces.
 *
 * Held as a struct rather than read from the constants directly so that the
 * tour a report describes is the tour that ran, even if the defaults move.
 */
struct TourConfig {
  uint32_t seed = DEFAULT_SEED;        ///< draws waypoints and punches
  double walk_s = WALK_S;              ///< scored walk length, s
  double radius_m = DEFAULT_RADIUS_M;  ///< radius of the target disc, m
  double point_s = POINT_S;            ///< time per target, s
  double lead_in_s = LEAD_IN_S;        ///< quiet time before target 0, s
};

/**
 * One waypoint, in the frame the crane released.
 *
 * Position and heading are demanded together: arriving at the right spot
 * facing the wrong way is a failure to do the task, and separating the two
 * errors is what makes a policy that can only approach on an arc visible in
 * the table.
 */
struct TourTarget {
  double x;     ///< metres ahead of the release point
  double y;     ///< metres left of the release point
  double yaw;   ///< demanded heading, rad, wrapped to [-pi, pi]
  bool active;  ///< false for the placeholder returned outside the tour
  int index;    ///< position in the tour, or -1 when inactive
};

/**
 * A whole scenario: the waypoints, and the settings that drew them.
 *
 * Built once and handed to every policy in a sweep unchanged, which is what
 * makes a seed a task rather than a repetition.
 */
struct Tour {
  TourConfig config;                  ///< the settings these came from
  std::vector<TourTarget> waypoints;  ///< visited in order, one per clock
};

/**
 * How a run ended.
 *
 * Only `COMPLETE` makes the error columns of a report comparable: the
 * others leave a policy scored on whichever fragment of the tour it
 * reached, which flatters a short-lived candidate rather than penalising
 * it.
 */
enum class Outcome { COMPLETE, FELL, TIMEOUT, INTERRUPTED, ERROR };

/**
 * Where the robot actually was when a target's clock ran out.
 *
 * Both the demand and the outcome are kept, not just the error between
 * them, so a reader can tell a policy that stopped short of every target
 * from one that overshot them all — two failures with the same mean error
 * and nothing else in common.
 */
struct TargetScore {
  int index;           ///< which waypoint this scored
  double target_x;     ///< demanded position, released frame, m
  double target_y;     ///< demanded position, released frame, m
  double target_yaw;   ///< demanded heading, released frame, rad
  double actual_x;     ///< where the pelvis was, world frame, m
  double actual_y;     ///< where the pelvis was, world frame, m
  double actual_yaw;   ///< heading the pelvis held, world frame, rad
  double pos_err_cm;   ///< distance between the two, cm
  double yaw_err_deg;  ///< signed heading error, deg
};

/**
 * What one leg of the tour cost, joint by joint and body by body.
 *
 * A segment spans one target's clock: it opens when `index` becomes the
 * demand and closes when the demand moves on, so segment 5 is the leg the
 * robot walked while target 5 was being asked for, closed at the moment the
 * demand changed to target 6. The lead-in before target 0 is not a segment,
 * which keeps the segments one-for-one with the targets they are named
 * after and joinable onto `TargetScore` by `index`.
 *
 * `full_window` is what separates a leg from a fragment. A run that falls
 * part way through a leg still records it, because how much a policy was
 * burning while it lost its footing is the interesting part, but the flag
 * says plainly that the numbers cover less than a target clock and must not
 * be pooled with legs that ran their course.
 *
 * Only a fragment carries timestamps. A whole leg's span follows from its
 * index and the target clock, and writing it out on every row would be a
 * column of arithmetic the reader can do; where a run ended is the one time
 * the file cannot reconstruct.
 */
struct SegmentCost {
  int index = -1;            ///< the target whose clock this leg spans
  double t_start_s = 0.0;    ///< when that target became the demand, s
  double t_end_s = 0.0;      ///< when it stopped being it, or the run ended, s
  bool full_window = false;  ///< false if the run ended part way through

  /// Absolute mechanical work per group over the leg, J, in `GROUP_NAMES`
  /// order. Joules add, so a group's figure is its joints' work.
  std::array<double, NUM_GROUPS> group_energy_j{};

  /// Accumulated joint-space jerk per group over the leg, rad/s^2, in
  /// `GROUP_NAMES` order and so paired one-for-one with `group_energy_j`.
  /// Its joints' total variation added, which is how much the group shook
  /// altogether rather than how much its worst joint did.
  std::array<double, NUM_GROUPS> group_vibration{};
};

/**
 * What one policy's run is reduced to.
 *
 * `outcome` defaults to `ERROR` so that a run which throws before it can
 * say anything is recorded as a failure rather than as an empty success.
 * The per-target scores are kept alongside the means because the means
 * alone cannot show a policy that walked well until one punch.
 */
struct Report {
  std::string policy;                 ///< policy name, as `name()` gives it
  Outcome outcome = Outcome::ERROR;   ///< how the run ended
  std::string detail;                 ///< why, for everything but COMPLETE
  std::vector<TargetScore> targets;   ///< every target actually scored
  std::vector<SegmentCost> segments;  ///< what each leg of the tour cost
  double pos_err_cm = 0.0;            ///< mean position error, cm
  double yaw_err_deg = 0.0;           ///< mean absolute heading error, deg
  double duration_s = 0.0;            ///< simulated seconds the run lasted
  double step_mean_us = 0.0;          ///< mean time in `step`, us
  double step_max_us = 0.0;           ///< worst time in `step`, us
};

/**
 * How a run is executed, as opposed to what it is asked to do.
 *
 * Nothing here changes the task: the same seed produces the same trajectory
 * whether it is watched, recorded or run headless as fast as the machine
 * allows.
 */
struct SimConfig {
  uint32_t seed = DEFAULT_SEED;                    ///< seeds the punches
  std::string model_path = "assets/g1_29dof.xml";  ///< scene to load
  bool viewer = false;                             ///< open a window
  double realtime = 0.0;                           ///< pace factor; 0 is free
  double max_seconds = 0.0;                        ///< cap, 0 for the default
  std::string record_dir;                          ///< empty to not record
  bool quiet = false;                              ///< drop the per-run chatter
};

/**
 * A campaign over many seeds, run inside this one process.
 *
 * A seed is a whole task rather than a repetition, so a campaign is the
 * only thing that separates policies at all; one process rather than one
 * per seed is what makes it affordable, since the scene, the CUDA context
 * and the libtorch runtime are then loaded once instead of once per run in
 * flight.
 */
struct SweepConfig {
  bool enabled = false;  ///< true once --runs or --seeds asked for a sweep
  uint32_t first = 0;    ///< first seed, inclusive
  uint32_t last = 0;     ///< last seed, inclusive
  int parallel = 1;      ///< runs in flight at once
  std::string csv = DEFAULT_CSV;  ///< file every run appends to
};

/**
 * The command line, parsed.
 *
 * One invocation measures one candidate. Holding the whole field in a
 * single process meant holding every checkpoint's engines and every
 * network's CUDA kernels at once, which cost more memory than the runs
 * themselves; one policy per process bounds that by construction, and the
 * field is a loop over invocations.
 */
struct Args {
  std::string policy;   ///< the one candidate this invocation measures
  TourConfig scenario;  ///< the tour to walk
  SimConfig sim;        ///< how to run it
  SweepConfig sweep;    ///< the campaign, if one was asked for
};

/**
 * What the joints are asked for on one control step.
 *
 * Shaped like the packet a G1 actually takes, feedforward torque included,
 * so that the PD in `apply_pd` is the same arithmetic the robot's own
 * boards would do rather than a simulation-only shortcut.
 */
struct MotorCommand {
  Eigen::Vector<float, NUM_MOTOR> q_target;   ///< position targets, rad
  Eigen::Vector<float, NUM_MOTOR> dq_target;  ///< rate targets, rad/s
  Eigen::Vector<float, NUM_MOTOR> kp;         ///< position gains, Nm/rad
  Eigen::Vector<float, NUM_MOTOR> kd;         ///< rate gains, Nm s/rad
  Eigen::Vector<float, NUM_MOTOR> tau_ff;     ///< feedforward torque, Nm
};

/**
 * Where the pelvis is in the world, flattened to the floor.
 *
 * Read from the simulator and used only by the harness: to place the target
 * marker, to score arrival, and to fix the released frame. No policy ever
 * sees it.
 */
struct WorldPose {
  double x;    ///< world x, m
  double y;    ///< world y, m
  double yaw;  ///< heading, rad, from the pelvis attitude
};

/**
 * Which part of a run the loop is in.
 *
 * `INIT` is the craned ramp to the stance, `CONTROL` is the policy walking
 * the tour, and `DONE` holds the stance after the last target so that a
 * recording ends on a standing robot rather than on a freeze.
 */
enum class Phase { INIT, CONTROL, DONE };

/**
 * What one call to `loop_step` decided.
 *
 * Carries the errors as well as the command because the viewer overlay and
 * the target marker need them, and recomputing them there would risk
 * showing something other than what was scored.
 */
struct Tick {
  MotorCommand command;  ///< what the joints are asked for
  Phase phase;           ///< phase after this step
  double dist_m;         ///< distance to the current target, m
  double yaw_err_rad;    ///< heading error to the current target, rad
  bool has_target;       ///< whether a target was current at all
  int target;            ///< index of that target, or -1
};

/**
 * The state one policy's run accumulates as it walks.
 *
 * The `last_*` fields exist because a target is scored when it expires, by
 * which point the loop has already moved on: they hold the last state seen
 * while that target was still current, so the score describes the moment
 * the clock ran out rather than the step after it.
 */
struct Loop {
  std::shared_ptr<ModelPolicy> policy;  ///< the candidate being measured
  Tour tour;                            ///< waypoints it is walking
  bool quiet = false;                   ///< drop the per-run chatter

  Phase phase = Phase::INIT;         ///< where the run has got to
  double start_time = -1.0;          ///< sim time the run began, -1 unset
  double control_start_time = -1.0;  ///< sim time the crane released

  std::optional<WorldPose> anchor;  ///< released frame the tour is drawn in

  int current_target = -1;              ///< target the clock is on
  double last_yaw_err_rad = 0.0;        ///< its last heading error, rad
  WorldPose last_world{0.0, 0.0, 0.0};  ///< pelvis at that moment
  double last_target_x = 0.0;           ///< its demanded position, m
  double last_target_y = 0.0;           ///< its demanded position, m
  double last_target_yaw = 0.0;         ///< its demanded heading, rad

  std::vector<TargetScore> targets;  ///< scores closed out so far
  double step_total_us = 0.0;        ///< summed time in `step`, us
  double step_max_us = 0.0;          ///< worst single step, us
  long step_count = 0;               ///< steps taken, for the mean
};

/**
 * One scheduled shove at the robot.
 *
 * Everything is drawn up front from the seed rather than at the moment of
 * impact, so the campaign a policy faces depends on the seed alone and not
 * on where its own walking happened to put the body when the clock struck.
 */
struct Punch {
  double time;           ///< sim time the push starts, s
  int body;              ///< body hit, a MuJoCo body id
  double point[3];       ///< contact point in that body's frame, m
  double draw_point[3];  ///< same direction, pushed out to the surface
  double dir[3];         ///< unit push direction, world frame
  double force_n;        ///< magnitude held for `DURATION_S`, N
};

/**
 * The whole punch campaign for one run.
 *
 * One punch per waypoint, on the same clock as the targets, so every policy
 * is disturbed the same number of times at the same moments and the
 * campaign cannot be outrun by walking faster.
 */
struct Schedule {
  std::vector<Punch> punches;  ///< in time order, one per waypoint
};

/**
 * Every index into the model this run will ever need, resolved once.
 *
 * Names are looked up at load and never again: a control step touching a
 * dozen joints at 50 Hz should not be doing string comparisons, and
 * resolving up front turns a missing joint into a refusal to start rather
 * than into a run that silently drives nothing.
 */
struct Handles {
  mjModel* m = nullptr;     ///< the loaded scene, not owned
  mjData* d = nullptr;      ///< its state, not owned
  int qpos_adr[NUM_MOTOR];  ///< qpos slot per joint, MuJoCo order
  int dof_adr[NUM_MOTOR];   ///< qvel slot per joint, MuJoCo order
  int act_id[NUM_MOTOR];    ///< actuator per joint, MuJoCo order
  int base_qpos = 0;        ///< first of the free joint's seven qpos
  int base_dof = 0;         ///< first of the free joint's six dofs
  int sensor_quat = -1;     ///< pelvis attitude sensor
  int sensor_gyro = -1;     ///< pelvis rate sensor
  int mocap_target = -1;    ///< marker showing the current target
  int mocap_heading = -1;   ///< marker showing its demanded heading
};

/**
 * The window and GL state a visible or recorded run renders through.
 *
 * A recording needs the same context a window does, so this is created for
 * `--record` as well as for `--realtime`; in that case the window is opened
 * hidden and only the offscreen buffer is ever read.
 */
struct Viewer {
  GLFWwindow* window = nullptr;  ///< hidden when only recording
  mjvCamera camera;              ///< fixed framing of the arena
  mjvOption option;              ///< what is drawn
  mjvScene scene;                ///< geoms for the current frame
  mjrContext context;            ///< GL resources, window and offscreen
};

/**
 * An ffmpeg process being fed raw frames.
 *
 * Piping to ffmpeg rather than linking an encoder keeps the harness free of
 * a codec dependency, and the pipe backpressures: if encoding cannot keep
 * up the simulation waits rather than dropping frames, which would break
 * the one-frame-per-control-step contract that makes playback real time.
 */
struct Recorder {
  FILE* pipe = nullptr;            ///< stdin of the encoder
  int width = 0;                   ///< frame width, px
  int height = 0;                  ///< frame height, px
  std::vector<unsigned char> rgb;  ///< one frame, reused every time
  long frames = 0;                 ///< frames written so far
  std::string path;                ///< destination mp4
};

/**
 * One engine binding and the host buffer that mirrors it.
 *
 * The host side is allocated once at load, so a control step copies into
 * memory that already exists instead of allocating on the path being timed.
 */
struct Tensor {
  std::string name;            ///< binding name in the ONNX graph
  std::vector<int64_t> shape;  ///< resolved shape, dynamic axes pinned to 1
  size_t count = 0;            ///< elements, the product of `shape`
  std::vector<float> data;     ///< host mirror of the device buffer
};

/**
 * A loaded TensorRT network, ready to be run one observation at a time.
 *
 * Held entirely in owning pointers because a policy that throws part way
 * through `init` must not leave a stream or a device allocation behind: a
 * sweep runs the whole field in one process, and one failed candidate
 * cannot be allowed to cost the next twelve their memory.
 */
struct Engine {
  std::unique_ptr<nvinfer1::IRuntime> runtime;           ///< deserialiser
  std::unique_ptr<nvinfer1::ICudaEngine> cuda_engine;    ///< the plan
  std::unique_ptr<nvinfer1::IExecutionContext> context;  ///< execution state
  std::shared_ptr<CUstream_st> stream;         ///< stream every copy uses
  std::vector<std::shared_ptr<void>> buffers;  ///< device memory, owned
  std::vector<void*> device_in;   ///< input buffers, parallel to `inputs`
  std::vector<void*> device_out;  ///< output buffers, parallel to `outputs`
  std::vector<Tensor> inputs;     ///< input bindings, in engine order
  std::vector<Tensor> outputs;    ///< output bindings, in engine order
};

/**
 * A serialised engine and where it is cached.
 *
 * The path travels with the bytes so that a plan which turns out not to
 * deserialise can be deleted by whoever discovers it, rather than being
 * rebuilt into the same broken cache entry on every future run.
 */
struct Plan {
  std::vector<char> bytes;  ///< the serialised engine
  std::string cache_path;   ///< where it is, or is about to be, cached
};

/**
 * Routes TensorRT's own diagnostics into this program's output.
 *
 * TensorRT insists on a logger; this one drops everything below a warning,
 * because a plan build otherwise buries the benchmark's own output under
 * several hundred lines about layer fusion.
 */
struct EngineLogger : nvinfer1::ILogger {
  void log(
      Severity severity,
      const char* msg
  ) noexcept override;
};

/**
 * Collect the bodies a punch is allowed to land on.
 *
 * Mocap bodies are the target markers, which have no dynamics to disturb,
 * and anything under a tenth of a kilogram is a fastener or a sensor mount
 * where six hundred newtons would be a modelling artefact rather than a
 * shove. What survives is the robot's own mass.
 *
 * @param[in] m  the loaded scene
 * @returns body ids in model order, worldbody excluded
 * @exceptsafe basic
 */
std::vector<int> targets(const mjModel* m) {
  std::vector<int> ids;
  for (int b = 1; b < m->nbody; ++b) {
    if (m->body_mocapid[b] >= 0) continue;
    if (m->body_mass[b] < 0.1) continue;
    ids.push_back(b);
  }
  return ids;
}

/**
 * Accumulates what a leg of the tour costs, one physics step at a time.
 *
 * Two quantities, both summed over the 1 ms step rather than sampled at the
 * 50 Hz control rate, because both are about what happens between control
 * steps: a policy that buzzes its ankles at 200 Hz is invisible to anything
 * that only looks when the policy does.
 *
 * Energy is absolute mechanical work, `sum |tau * omega| dt` per joint. The
 * absolute value is what makes it an actuator cost rather than a physics
 * one — these motors burn current holding a limb against gravity whichever
 * way it is moving, and a signed integral would let a leg swinging down pay
 * back the leg that lifted it and report a policy as free.
 *
 * Vibration is the total variation of each joint's angular acceleration,
 * `sum |alpha(t) - alpha(t-dt)|`, which is jerk integrated over the leg. It
 * is measured in the same space as the energy, so the two columns for a
 * joint describe that one joint and can be read together: what it burned,
 * and how much of that was spent fighting itself.
 *
 * Acceleration alone would rank a policy that strides hard alongside one
 * that shakes; differencing it first leaves only the part that changes
 * direction faster than walking does.
 *
 * Both are read straight after `mj_step`, so torque and velocity are the
 * pair MuJoCo actually integrated with. The first two steps are spent
 * filling the two finite differences and contribute to neither sum.
 */
struct CostMeter {
  std::array<double, NUM_MOTOR> prev_qvel{};  ///< last joint velocity
  std::array<double, NUM_MOTOR> prev_qacc{};  ///< last joint acceleration
  int warmup = 0;                 ///< steps seen, capped once differences valid
  bool open = false;              ///< true while a leg is being accumulated
  SegmentCost leg;                ///< the leg being accumulated
  std::vector<SegmentCost> done;  ///< legs already closed, in order
};

/**
 * Build a meter sized to one scene.
 *
 * @param[in] m  the loaded scene
 * @returns a meter with nothing accumulated and no leg open
 * @exceptsafe basic
 */
CostMeter meter_make() { return CostMeter{}; }

/**
 * Start accumulating a new leg.
 *
 * @param[in,out] meter  the meter, with no leg open
 * @param[in] index      the target whose clock this leg spans
 * @param[in] t          simulated time the demand changed, s
 * @exceptsafe basic
 */
void meter_open(
    CostMeter& meter,
    int index,
    double t
) {
  meter.leg = SegmentCost{};
  meter.leg.index = index;
  meter.leg.t_start_s = t;
  meter.open = true;
}

/**
 * Close the open leg and keep it.
 *
 * @param[in,out] meter  the meter, with a leg open
 * @param[in] t          simulated time the leg ended, s
 * @param[in] full       true if the target's clock ran out, false if the
 *                       run ended part way through
 * @exceptsafe basic
 */
void meter_close(
    CostMeter& meter,
    double t,
    bool full
) {
  if (!meter.open) return;
  meter.leg.t_end_s = t;
  meter.leg.full_window = full;
  meter.done.push_back(meter.leg);
  meter.open = false;
}

/**
 * Follow the demand, opening and closing legs as it moves.
 *
 * Called once per control step with whatever the tour is asking for. A
 * target index of -1 — the lead-in, and everything after the last target —
 * closes whatever was open and starts nothing.
 *
 * @param[in,out] meter  the meter
 * @param[in] index      the target being demanded, or -1 for none
 * @param[in] t          simulated time, s
 * @exceptsafe basic
 */
void meter_mark(
    CostMeter& meter,
    int index,
    double t
) {
  if (meter.open && meter.leg.index == index) return;
  meter_close(meter, t, true);
  if (index >= 0) meter_open(meter, index, t);
}

/**
 * Add one physics step to the open leg.
 *
 * The per-body differences are advanced whether or not a leg is open, so
 * that the first step of a leg is differenced against the step before it
 * rather than against nothing and the boundary costs no accuracy.
 *
 * @param[in,out] meter  the meter
 * @param[in] h          resolved indices into the model and its state
 * @param[in] dt         the physics timestep just taken, s
 * @exceptsafe no-throw
 */
void meter_step(
    CostMeter& meter,
    const Handles& h,
    double dt
) {
  const mjData* d = h.d;
  const bool differences_valid = meter.warmup >= 2;
  for (int i = 0; i < NUM_MOTOR; ++i) {
    const double omega = d->qvel[h.dof_adr[i]];
    const double alpha = (omega - meter.prev_qvel[i]) / dt;
    if (differences_valid && meter.open) {
      const int g = JOINT_GROUP[i];
      meter.leg.group_vibration[g] += std::fabs(alpha - meter.prev_qacc[i]);
      const double tau = d->actuator_force[h.act_id[i]];
      meter.leg.group_energy_j[g] += std::fabs(tau * omega) * dt;
    }
    meter.prev_qvel[i] = omega;
    meter.prev_qacc[i] = alpha;
  }
  if (meter.warmup < 2) ++meter.warmup;
}

/**
 * Draw a uniform double from [0, 1).
 *
 * Rolled by hand rather than taken from `std::uniform_real_distribution`,
 * whose output is implementation defined: a seed has to name the same tour
 * on every machine that clones this repository, and that guarantee is
 * exactly what the standard declines to give. The shift keeps the top 24
 * bits, which divide exactly by 2^24 and so cannot round to 1.
 *
 * @param[in,out] rng  the generator, advanced by one draw
 * @returns a value in [0, 1)
 * @exceptsafe no-throw
 */
inline double unit(std::mt19937& rng) {
  return static_cast<double>(rng() >> 8) * (1.0 / 16777216.0);
}

/**
 * Draw a uniform double from [lo, hi).
 *
 * @param[in,out] rng  the generator, advanced by one draw
 * @param[in] lo       lower bound, included
 * @param[in] hi       upper bound, excluded
 * @returns a value in [lo, hi)
 * @exceptsafe no-throw
 */
inline double between(
    std::mt19937& rng,
    double lo,
    double hi
) {
  return lo + (hi - lo) * unit(rng);
}

/**
 * Draw a tour from its seed.
 *
 * The radius is the square root of a uniform draw, which spreads waypoints
 * evenly over the disc; drawing the radius directly would pile them near
 * the centre and turn the tour into a shuffle in place. Heading is drawn
 * independently of bearing, so arriving at a target is genuinely two
 * demands rather than one: a policy that can only face where it is walking
 * will be caught by the yaw column.
 *
 * @param[in] config  seed, arena and clock
 * @returns the waypoints, in the order they will be demanded
 * @throws std::runtime_error if the clock or the radius is not positive, or
 *         if the walk is shorter than a single target
 * @exceptsafe strong
 */
Tour tour_make(const TourConfig& config) {
  if (!(config.point_s > 0.0)) {
    throw std::runtime_error("scenario: the target clock must be positive");
  }
  if (!(config.radius_m > 0.0)) {
    throw std::runtime_error("scenario: the arena must have a positive radius");
  }
  const int targets = static_cast<int>(config.walk_s / config.point_s);
  if (targets < 1) {
    throw std::runtime_error("scenario: the run is shorter than one target");
  }
  Tour tour{config, {}};
  std::mt19937 rng(config.seed);
  for (int i = 0; i < targets; ++i) {
    const double radius = config.radius_m * std::sqrt(unit(rng));
    const double bearing = between(rng, 0.0, 2.0 * M_PI);
    const double yaw = between(rng, 0.0, 2.0 * M_PI);
    tour.waypoints.push_back(
        TourTarget{
            CENTRE_X_M + radius * std::cos(bearing),
            CENTRE_Y_M + radius * std::sin(bearing),
            std::remainder(yaw, 2.0 * M_PI),
            true,
            i
        }
    );
  }
  return tour;
}

/**
 * Name an outcome for the table and the transcript.
 *
 * @param[in] outcome  how a run ended
 * @returns a stable lower-case word, never null
 * @exceptsafe no-throw
 */
const char* outcome_name(Outcome outcome) {
  switch (outcome) {
    case Outcome::COMPLETE:
      return "complete";
    case Outcome::FELL:
      return "fell";
    case Outcome::TIMEOUT:
      return "timeout";
    case Outcome::INTERRUPTED:
      return "interrupted";
    case Outcome::ERROR:
      return "error";
  }
  return "error";
}

/**
 * Format one policy's row of the summary table.
 *
 * The error columns are left blank unless the run completed, rather than
 * filled with the means of a partial tour: a figure in that column would
 * invite comparison with the rows above it, and errors gathered over the
 * fragment a policy reached before falling are not the same measurement.
 * A long failure message is truncated so the columns cannot be pushed out
 * of alignment.
 *
 * @param[in] report  a finished run
 * @returns one line, no trailing newline
 * @exceptsafe basic
 */
std::string report_line(const Report& report) {
  std::ostringstream out;
  out << std::left << std::setw(15) << report.policy << std::right;
  if (report.outcome == Outcome::COMPLETE) {
    out << std::fixed << std::setprecision(2) << std::setw(9)
        << report.pos_err_cm << " cm" << std::setw(8) << report.yaw_err_deg
        << " deg";
  } else {
    out << std::setw(12) << outcome_name(report.outcome) << std::setw(12) << "";
  }
  out << std::setw(6) << report.targets.size() << " targets" << std::fixed
      << std::setprecision(0) << std::setw(7) << report.step_max_us << " us";
  if (!report.detail.empty()) {
    const size_t width = 52;
    out << "  "
        << (report.detail.size() > width
                ? report.detail.substr(0, width - 3) + "..."
                : report.detail);
  }
  return out.str();
}

/**
 * Print one run's full detail: every target, then the means and timings.
 *
 * Written to stdout target by target, because the means alone cannot
 * distinguish a policy that tracked well throughout from one that walked
 * cleanly until a punch and then spent the rest of the tour recovering.
 *
 * @param[in] report  a finished run
 * @exceptsafe basic
 */
void report_print(const Report& report) {
  std::cout << "policy   " << report.policy << std::endl;
  std::cout << "outcome  " << outcome_name(report.outcome)
            << (report.detail.empty() ? "" : " (" + report.detail + ")")
            << std::endl;
  std::cout << std::fixed << std::setprecision(3);
  for (const TargetScore& t : report.targets) {
    std::cout << "  target " << std::setw(2) << t.index << "  (" << std::setw(6)
              << t.target_x << ", " << std::setw(6) << t.target_y << ", "
              << std::setw(6) << t.target_yaw << ")  actual (" << std::setw(6)
              << t.actual_x << ", " << std::setw(6) << t.actual_y << ", "
              << std::setw(6) << t.actual_yaw << ")  err " << std::setw(7)
              << std::setprecision(2) << t.pos_err_cm << " cm " << std::setw(6)
              << t.yaw_err_deg << " deg" << std::setprecision(3) << std::endl;
  }
  for (const SegmentCost& leg : report.segments) {
    std::cout << "  leg    " << std::setw(2) << leg.index << "  "
              << std::setprecision(2) << std::setw(6) << leg.t_start_s << "-"
              << std::setw(6) << leg.t_end_s << " s";
    for (int g = 0; g < NUM_GROUPS; ++g) {
      std::cout << "  " << GROUP_NAMES[g] << " " << std::setprecision(1)
                << std::setw(7) << leg.group_energy_j[g] << " J/"
                << std::setw(7) << std::setprecision(0)
                << leg.group_vibration[g];
    }
    std::cout << (leg.full_window ? "" : "  (cut short)")
              << std::setprecision(3) << std::endl;
  }
  if (report.outcome == Outcome::COMPLETE) {
    std::cout << std::setprecision(2) << "walk pos " << report.pos_err_cm
              << " cm   walk yaw " << report.yaw_err_deg << " deg" << std::endl;
  }
  std::cout << std::setprecision(1) << "policy step  mean "
            << report.step_mean_us << " us   max " << report.step_max_us
            << " us   over " << report.duration_s << " s of sim" << std::endl;
}

/**
 * Total control time a tour occupies, in seconds.
 *
 * The lead-in plus one clock per waypoint. Used both to end the run and to
 * size the timeout, so the two can never disagree about how long a tour is.
 *
 * @param[in] tour  the scenario
 * @returns seconds from the crane release to the last target expiring
 * @exceptsafe no-throw
 */
double tour_duration(const Tour& tour) {
  return tour.config.lead_in_s +
         tour.config.point_s * static_cast<double>(tour.waypoints.size());
}

/**
 * Fill in a report's means from its per-target scores.
 *
 * Heading errors are averaged in absolute value while position errors are
 * already unsigned, so a policy that leans as far left of one target as it
 * does right of the next is not credited with having faced either of them
 * correctly. A run with nothing scored keeps its zeros: there is no mean to
 * take, and inventing one would put it in the table beside runs that walked.
 *
 * @param[in] report  a run with its target scores collected
 * @returns the same report with its means filled in
 * @exceptsafe basic
 */
Report report_finalize(Report report) {
  if (report.targets.empty()) return report;
  double pos = 0.0;
  double yaw = 0.0;
  for (const TargetScore& t : report.targets) {
    pos += t.pos_err_cm;
    yaw += std::fabs(t.yaw_err_deg);
  }
  report.pos_err_cm = pos / static_cast<double>(report.targets.size());
  report.yaw_err_deg = yaw / static_cast<double>(report.targets.size());
  return report;
}

/**
 * Start a run's control state for one policy and tour.
 *
 * @param[in] policy  the candidate to drive it
 * @param[in] tour    waypoints to walk
 * @returns a loop waiting on its first step
 * @exceptsafe basic
 */
std::shared_ptr<Loop> loop_make(
    std::shared_ptr<ModelPolicy> policy,
    Tour tour
) {
  std::shared_ptr<Loop> loop = std::make_shared<Loop>();
  loop->policy = std::move(policy);
  loop->tour = std::move(tour);
  return loop;
}

/**
 * Find which waypoint is current at a given control time.
 *
 * Returns an inactive placeholder before the lead-in has elapsed and after
 * the last clock has run out, which is how the loop learns both that there
 * is nothing to chase yet and that the tour is over.
 *
 * @param[in] tour  the scenario
 * @param[in] t     seconds since the crane released
 * @returns the current waypoint, or an inactive one with index -1
 * @exceptsafe no-throw
 */
TourTarget tour_at(
    const Tour& tour,
    double t
) {
  const double elapsed = t - tour.config.lead_in_s;
  if (elapsed < 0.0) return TourTarget{0.0, 0.0, 0.0, false, -1};
  const int index = static_cast<int>(elapsed / tour.config.point_s);
  if (index < 0 || index >= static_cast<int>(tour.waypoints.size())) {
    return TourTarget{0.0, 0.0, 0.0, false, -1};
  }
  return tour.waypoints[index];
}

/**
 * Hold a velocity command inside a policy's envelope.
 *
 * The planar norm is capped before the per-axis clamps so that a diagonal
 * command is scaled rather than squared off: clamping each axis on its own
 * would let a policy travel faster across the disc than along it, and turn
 * the envelope into something that depends on which way the target lies.
 *
 * @param[in] limits     the envelope to respect
 * @param[in,out] cmd    forward, left and yaw rate, clamped in place
 * @exceptsafe no-throw
 */
void clamp_command(
    const Limits& limits,
    double cmd[3]
) {
  if (limits.speed_norm > 0.0) {
    const double norm = std::hypot(cmd[0], cmd[1]);
    if (norm > limits.speed_norm) {
      cmd[0] *= limits.speed_norm / norm;
      cmd[1] *= limits.speed_norm / norm;
    }
  }
  cmd[0] = std::clamp(cmd[0], limits.vx_min, limits.vx_max);
  cmd[1] = std::clamp(cmd[1], -limits.vy_abs, limits.vy_abs);
  cmd[2] = std::clamp(cmd[2], -limits.yaw_rate_abs, limits.yaw_rate_abs);
}

/**
 * Turn a target error into a velocity command, the usual way.
 *
 * Offered to policies rather than imposed on them, and driven by the
 * `Limits` the policy itself owns. Arrival is latched with separate enter
 * and exit thresholds because a single threshold sitting inside the robot's
 * own footstep jitter makes a policy arrive and un-arrive every control
 * period, which reads on video as shuffling in place for the rest of the
 * clock. Position and heading latch independently, so a policy can stand
 * still and turn.
 *
 * @param[in] in                 state and the error to the target
 * @param[in] limits             the envelope to respect
 * @param[in,out] pos_reached    position latch, reset on a new target
 * @param[in,out] yaw_reached    heading latch, reset on a new target
 * @param[out] cmd               forward, left and yaw rate
 * @returns whether anything is still being asked for
 * @exceptsafe no-throw
 */
bool command_from_target(
    const Input& in,
    const Limits& limits,
    bool& pos_reached,
    bool& yaw_reached,
    double cmd[3]
) {
  cmd[0] = 0.0;
  cmd[1] = 0.0;
  cmd[2] = 0.0;
  if (in.new_target) {
    pos_reached = false;
    yaw_reached = false;
  }
  if (!in.has_target) return false;

  if (pos_reached) {
    if (in.dist_m > limits.pos_reached_exit_m) pos_reached = false;
  } else {
    if (in.dist_m < limits.pos_reached_enter_m) pos_reached = true;
  }
  if (yaw_reached) {
    if (std::fabs(in.yaw_err_rad) > limits.yaw_reached_exit_rad) {
      yaw_reached = false;
    }
  } else {
    if (std::fabs(in.yaw_err_rad) < limits.yaw_reached_enter_rad) {
      yaw_reached = true;
    }
  }

  if (!pos_reached) {
    cmd[0] = limits.walk_kp_pos * in.target_forward_m;
    cmd[1] = limits.walk_kp_pos * in.target_left_m;
  }
  if (!yaw_reached) {
    cmd[2] = limits.walk_kp_yaw * in.yaw_err_rad;
  }
  clamp_command(limits, cmd);
  return !pos_reached || !yaw_reached;
}

/**
 * Hold a fixed pose on the shared gains.
 *
 * The fallback behind everything: the joints a policy does not own, the
 * whole robot once the tour is finished, and the starting point that policy
 * output is written over. Feedforward torque is zero, so nothing here can
 * push the robot anywhere its own PD would not.
 *
 * @param[in] pose  NUM_MOTOR joint angles, MuJoCo order, rad
 * @returns the command that holds it
 * @exceptsafe no-throw
 */
MotorCommand hold_command(const double* pose) {
  MotorCommand command{};
  command.dq_target.setZero();
  command.tau_ff.setZero();
  command.kp = KPS;
  command.kd = KDS;
  for (int i = 0; i < NUM_MOTOR; ++i) {
    command.q_target[i] = static_cast<float>(pose[i]);
  }
  return command;
}

/**
 * Blend from wherever the joints are to the shared stance.
 *
 * Linear in time from the pose measured at the call rather than from the
 * pose at the start of the ramp, which makes the ramp self-correcting: a
 * joint that lags its target is pulled towards the stance from where it
 * actually is, and all of them arrive together at the release.
 *
 * @param[in] state     current joint angles
 * @param[in] time      seconds into the ramp
 * @param[in] duration  seconds the ramp lasts
 * @returns the command for this instant of the ramp
 * @exceptsafe no-throw
 */
MotorCommand ramp_command(
    const RobotState& state,
    double time,
    double duration
) {
  MotorCommand command{};
  command.dq_target.setZero();
  command.tau_ff.setZero();
  command.kp = KPS;
  command.kd = KDS;
  const double ratio = std::clamp(time / duration, 0.0, 1.0);
  for (int i = 0; i < NUM_MOTOR; ++i) {
    command.q_target[i] = static_cast<float>(
        state.motor_q[i] * (1.0 - ratio) + DEFAULT_ANGLES[i] * ratio
    );
  }
  return command;
}

/**
 * Close out the target whose clock has just run out.
 *
 * Scores the `last_*` state rather than the present one, because by the
 * time the loop notices the clock has turned over it is already looking at
 * the next waypoint. Does nothing before the first target, so the lead-in
 * is never scored as an arrival.
 *
 * @param[in,out] loop  run state; gains one entry in `targets`
 * @exceptsafe basic
 */
void score_target(Loop& loop) {
  if (loop.current_target < 0) return;
  const double dx = loop.last_target_x - loop.last_world.x;
  const double dy = loop.last_target_y - loop.last_world.y;
  loop.targets.push_back(
      TargetScore{
          loop.current_target,
          loop.last_target_x,
          loop.last_target_y,
          loop.last_target_yaw,
          loop.last_world.x,
          loop.last_world.y,
          loop.last_world.yaw,
          std::hypot(dx, dy) * 100.0,
          loop.last_yaw_err_rad * 180.0 / M_PI
      }
  );
}

/**
 * Advance one control step: ramp, or drive the policy and score the tour.
 *
 * The frame the tour is drawn in is fixed here, once, at the moment the
 * crane lets go. Every waypoint is then rotated into world coordinates
 * through that anchor, so a policy that comes off the crane facing slightly
 * off is asked to walk the same shape as one that does not, and the tour
 * cannot be made easier or harder by how the handover went.
 *
 * The policy is timed around this call alone. Everything else the step does
 * — resolving the target, transforming it, filling the command — sits
 * outside the measurement, so the table's step column is inference and
 * nothing else.
 *
 * @param[in,out] loop  run state, advanced by one step
 * @param[in] state     robot as its own sensors see it
 * @param[in] world     pelvis pose, for anchoring and scoring
 * @param[in] now       simulated time, s
 * @returns the command to apply and what the step decided
 * @exceptsafe basic
 */
Tick loop_step(
    Loop& loop,
    const RobotState& state,
    const WorldPose& world,
    double now
) {
  if (loop.start_time < 0.0) loop.start_time = now;
  const double elapsed = now - loop.start_time;

  Tick tick{};
  tick.phase = loop.phase;
  tick.target = -1;

  if (loop.phase == Phase::INIT) {
    if (elapsed >= INIT_DURATION_S) {
      loop.phase = Phase::CONTROL;
      loop.control_start_time = now;
      loop.anchor.emplace(world);
      if (!loop.quiet) {
        std::cout << "control: " << loop.policy->name() << " engaged, anchor ("
                  << world.x << ", " << world.y << ", " << world.yaw << ")"
                  << std::endl;
      }
    } else {
      tick.command = ramp_command(state, elapsed, INIT_DURATION_S);
      return tick;
    }
  }

  if (loop.phase == Phase::DONE) {
    tick.phase = Phase::DONE;
    tick.command = hold_command(DEFAULT_ANGLES.data());
    return tick;
  }

  const double control_time = now - loop.control_start_time;

  const TourTarget target = tour_at(loop.tour, control_time);

  const bool new_target = target.index != loop.current_target;
  if (new_target) {
    score_target(loop);
    loop.current_target = target.index;
  }

  double forward = 0.0;
  double left = 0.0;
  double dist = 0.0;
  double yaw_err = 0.0;
  bool has_target = false;

  if (target.active && loop.anchor.has_value()) {
    const double ayaw = loop.anchor->yaw;
    const double tx =
        loop.anchor->x + target.x * std::cos(ayaw) - target.y * std::sin(ayaw);
    const double ty =
        loop.anchor->y + target.x * std::sin(ayaw) + target.y * std::cos(ayaw);
    const double tyaw = ayaw + target.yaw;
    const double dx = tx - world.x;
    const double dy = ty - world.y;
    dist = std::hypot(dx, dy);
    yaw_err = std::remainder(tyaw - world.yaw, 2.0 * M_PI);
    has_target = true;

    const double cy = std::cos(world.yaw);
    const double sy = std::sin(world.yaw);
    forward = cy * dx + sy * dy;
    left = -sy * dx + cy * dy;

    loop.last_yaw_err_rad = yaw_err;
    loop.last_world = world;
    loop.last_target_x = target.x;
    loop.last_target_y = target.y;
    loop.last_target_yaw = target.yaw;
  } else if (control_time > tour_duration(loop.tour)) {
    loop.phase = Phase::DONE;
  }

  double arm_targets[NUM_MOTOR];
  for (int i = 0; i < NUM_MOTOR; ++i) {
    arm_targets[i] = DEFAULT_ANGLES[i];
  }

  const Input in{
      &state,
      forward,
      left,
      new_target,
      arm_targets,
      control_time,
      dist,
      yaw_err,
      has_target
  };

  const std::chrono::steady_clock::time_point t0 =
      std::chrono::steady_clock::now();
  const Output out = loop.policy->step(in);
  const double step_us = std::chrono::duration<double, std::micro>(
                             std::chrono::steady_clock::now() - t0
  )
                             .count();
  loop.step_total_us += step_us;
  loop.step_max_us = std::max(loop.step_max_us, step_us);
  ++loop.step_count;

  MotorCommand command = hold_command(DEFAULT_ANGLES.data());
  for (int i = 0; i < NUM_MOTOR; ++i) {
    if (!out.owns[i]) continue;
    command.q_target[i] = static_cast<float>(out.q_target[i]);
    command.dq_target[i] = 0.0f;
    command.kp[i] = out.kp[i];
    command.kd[i] = out.kd[i];
    command.tau_ff[i] = 0.0f;
  }

  tick.command = command;
  tick.phase = loop.phase;
  tick.dist_m = dist;
  tick.yaw_err_rad = yaw_err;
  tick.has_target = has_target;
  tick.target = target.index;
  return tick;
}

/**
 * Read a finished run's scores and timings out of its loop state.
 *
 * Leaves the outcome alone: only the caller stepping the world knows
 * whether the run completed, fell or was interrupted.
 *
 * @param[in] loop  a run that has finished
 * @returns a report with everything but its outcome and duration
 * @exceptsafe basic
 */
Report loop_report(const Loop& loop) {
  Report report;
  report.policy = loop.policy->name();
  report.targets = loop.targets;
  report.step_max_us = loop.step_max_us;
  report.step_mean_us =
      loop.step_count > 0
          ? loop.step_total_us / static_cast<double>(loop.step_count)
          : 0.0;
  return report;
}

/**
 * Draw a direction uniformly over the sphere.
 *
 * The height is drawn uniformly and the azimuth independently, which is
 * what makes the result uniform over the surface; drawing two angles
 * instead would crowd the poles and leave the campaign quietly biased
 * towards pushing the robot straight up and down.
 *
 * @param[in,out] rng  the generator, advanced by two draws
 * @param[out] out     unit vector, world frame
 * @exceptsafe no-throw
 */
void direction(
    std::mt19937& rng,
    double out[3]
) {
  const double z = between(rng, -1.0, 1.0);
  const double phi = between(rng, 0.0, 2.0 * M_PI);
  const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
  out[0] = r * std::cos(phi);
  out[1] = r * std::sin(phi);
  out[2] = z;
}

/**
 * How far from a body's origin a punch may be applied, in metres.
 *
 * The largest bounding radius among the body's geoms, capped at a quarter
 * metre. The cap matters for the torso, whose bounding sphere is far larger
 * than anything a hand could reach around: without it a shove would be
 * applied through a lever arm that does not exist on the robot.
 *
 * @param[in] m     the loaded scene
 * @param[in] body  body id
 * @returns the reach, in metres
 * @exceptsafe no-throw
 */
double reach(
    const mjModel* m,
    int body
) {
  double r = 0.0;
  for (int g = 0; g < m->body_geomnum[body]; ++g) {
    const int geom = m->body_geomadr[body] + g;
    r = std::max(r, static_cast<double>(m->geom_rbound[geom]));
  }
  return std::min(r, 0.25);
}

/**
 * Draw a run's whole punch campaign from its seed.
 *
 * One punch per target, on the target clock, so the disturbances arrive at
 * the same moments for every candidate and cannot be outrun by walking
 * faster. Force is drawn against a ceiling that climbs with time from
 * `RAMP_FLOOR` of `FORCE_MAX_N` at the first punch to all of it after
 * `RAMP_S`, which puts the hardest hits at the end of a tour where a policy
 * has already had to prove it can walk at all.
 *
 * Everything is drawn now rather than at the moment of impact: a policy
 * cannot change the campaign it faces by walking somewhere else.
 *
 * @param[in] m        the loaded scene, for bodies and their reach
 * @param[in] seed     seeds the bodies, points, directions and forces
 * @param[in] t_first  simulated time of the first punch, s
 * @param[in] period   time between punches, s
 * @param[in] count    how many to draw
 * @returns the campaign, in time order; empty if none were asked for
 * @throws std::runtime_error if the scene has no body worth hitting
 * @exceptsafe basic
 */
Schedule schedule_make(
    const mjModel* m,
    uint32_t seed,
    double t_first,
    double period,
    int count
) {
  Schedule schedule;
  if (count <= 0 || period <= 0.0) return schedule;

  const std::vector<int> bodies = targets(m);
  if (bodies.empty()) {
    throw std::runtime_error("punch: the model has no body worth hitting");
  }

  std::mt19937 rng(seed);
  for (int i = 0; i < count; ++i) {
    Punch p{};
    p.time = t_first + period * static_cast<double>(i);
    p.body = bodies[static_cast<size_t>(rng() % bodies.size())];
    const double r = reach(m, p.body);
    for (int k = 0; k < 3; ++k) p.point[k] = between(rng, -r, r);
    const double span = std::sqrt(
        p.point[0] * p.point[0] + p.point[1] * p.point[1] +
        p.point[2] * p.point[2]
    );
    for (int k = 0; k < 3; ++k) {
      p.draw_point[k] = span > 1e-9 ? p.point[k] * r / span : 0.0;
    }
    if (span <= 1e-9) p.draw_point[0] = r;
    direction(rng, p.dir);
    const double climbed = std::min((p.time - t_first) / RAMP_S, 1.0);
    const double ceiling =
        FORCE_MAX_N * (RAMP_FLOOR + (1.0 - RAMP_FLOOR) * climbed);
    p.force_n = ceiling * between(rng, FORCE_SCALE_MIN, 1.0);
    schedule.punches.push_back(p);
  }
  return schedule;
}

/**
 * Find the punch being applied at a given time, if any.
 *
 * @param[in] schedule  the campaign
 * @param[in] t         simulated time, s
 * @returns the punch inside its push window, or null
 * @exceptsafe no-throw
 */
const Punch* schedule_active(
    const Schedule& schedule,
    double t
) {
  for (const Punch& p : schedule.punches) {
    if (t >= p.time && t < p.time + DURATION_S) return &p;
  }
  return nullptr;
}

/**
 * Find the punch that should be drawn at a given time, if any.
 *
 * Its window outlasts the push by `PUNCH_HOLD_S`, so an impulse four frames
 * long still reads on video. Nothing physical follows from being visible.
 *
 * @param[in] schedule  the campaign
 * @param[in] t         simulated time, s
 * @returns the punch inside its drawing window, or null
 * @exceptsafe no-throw
 */
const Punch* schedule_visible(
    const Schedule& schedule,
    double t
) {
  for (const Punch& p : schedule.punches) {
    if (t >= p.time && t < p.time + DURATION_S + PUNCH_HOLD_S) return &p;
  }
  return nullptr;
}

/**
 * Take a point in a punched body's frame into world coordinates.
 *
 * The punch stores its contact point in the body frame, so the same drawn
 * campaign lands in the same place on the robot however the robot is
 * standing when the clock reaches it.
 *
 * @param[in] d      simulation state, for the body's current pose
 * @param[in] p      the punch, naming the body
 * @param[in] local  point in the body frame, m
 * @param[out] out   the same point in world coordinates, m
 * @exceptsafe no-throw
 */
void punch_body_point(
    const mjData* d,
    const Punch& p,
    const double local[3],
    double out[3]
) {
  const double* pos = d->xpos + 3 * p.body;
  const double* mat = d->xmat + 9 * p.body;
  for (int k = 0; k < 3; ++k) {
    out[k] = pos[k] + mat[3 * k + 0] * local[0] + mat[3 * k + 1] * local[1] +
             mat[3 * k + 2] * local[2];
  }
}

/**
 * Where a punch is currently being applied, in world coordinates.
 *
 * @param[in] d     simulation state
 * @param[in] p     the punch
 * @param[out] out  contact point in world coordinates, m
 * @exceptsafe no-throw
 */
void punch_world_point(
    const mjData* d,
    const Punch& p,
    double out[3]
) {
  punch_body_point(d, p, p.point, out);
}

/**
 * Apply one punch as a force and the torque it implies.
 *
 * MuJoCo takes an external wrench about a body's centre of mass, so the
 * off-centre contact has to be carried in as its own moment; dropping that
 * term would turn every shove into a push through the hips and never spin
 * the robot, which is the part a walking policy finds hardest.
 *
 * Assigns rather than accumulates, and the loop clears the slot when the
 * push ends, so a punch cannot outlive its window.
 *
 * @param[in] m     the loaded scene, unused but kept for symmetry
 * @param[in,out] d  simulation state; gains the wrench
 * @param[in] p     the punch to apply
 * @exceptsafe no-throw
 */
void punch_apply(
    const mjModel* m,
    mjData* d,
    const Punch& p
) {
  (void)m;
  double point[3];
  punch_world_point(d, p, point);

  const double* com = d->xipos + 3 * p.body;
  const double r[3] = {point[0] - com[0], point[1] - com[1], point[2] - com[2]};
  const double f[3] = {
      p.dir[0] * p.force_n,
      p.dir[1] * p.force_n,
      p.dir[2] * p.force_n
  };

  double* dst = d->xfrc_applied + 6 * p.body;
  dst[0] = f[0];
  dst[1] = f[1];
  dst[2] = f[2];
  dst[3] = r[1] * f[2] - r[2] * f[1];
  dst[4] = r[2] * f[0] - r[0] * f[2];
  dst[5] = r[0] * f[1] - r[1] * f[0];
}

/**
 * Draw a word into an RGB texture, centred and turned a quarter turn.
 *
 * The glyphs are laid out twice: once to measure the ink, and once to draw
 * it centred on that measurement rather than on the pen path, so a word
 * with descenders sits in the middle of the tile rather than above it. The
 * rotation is what makes the label read the right way up from a camera
 * looking along the floor rather than down at it.
 *
 * Blends towards white by coverage rather than writing opaque pixels, so
 * the lettering keeps its antialiasing against whatever the floor already
 * shows.
 *
 * @param[in,out] rgb  texture bytes, three channels, row major
 * @param[in] width    texture width, px
 * @param[in] height   texture height, px
 * @param[in] text     word to draw; nothing is drawn if it is empty
 * @throws std::runtime_error if freetype or the vendored font fails to load
 * @exceptsafe basic
 */
void paint_floor_text(
    mjtByte* rgb,
    int width,
    int height,
    const std::string& text
) {
  if (text.empty()) return;

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
  FT_Set_Pixel_Sizes(face, 0, static_cast<FT_UInt>(em_px));
  const double tracking_px = FLOOR_TRACKING_EM * em_px;

  double pen = 0.0;
  double min_x = 0.0, max_x = 0.0, min_y = 0.0, max_y = 0.0;
  bool any = false;
  for (const char c : text) {
    FT_Load_Char(face, static_cast<FT_ULong>(c), FT_LOAD_RENDER);
    const FT_GlyphSlot glyph = face->glyph;
    if (glyph->bitmap.width > 0 && glyph->bitmap.rows > 0) {
      const double gx = pen + glyph->bitmap_left;
      const double gy = -glyph->bitmap_top;
      if (!any) {
        min_x = gx;
        max_x = gx + glyph->bitmap.width;
        min_y = gy;
        max_y = gy + glyph->bitmap.rows;
        any = true;
      } else {
        min_x = std::min(min_x, gx);
        max_x = std::max(max_x, gx + glyph->bitmap.width);
        min_y = std::min(min_y, gy);
        max_y = std::max(max_y, gy + glyph->bitmap.rows);
      }
    }
    pen += (glyph->advance.x / 64.0) + tracking_px;
  }

  const double ink_x = 0.5 * (min_x + max_x);
  const double ink_y = 0.5 * (min_y + max_y);
  const double centre_x = 0.5 * width;
  const double centre_y = 0.5 * height;

  pen = 0.0;
  for (const char c : text) {
    FT_Load_Char(face, static_cast<FT_ULong>(c), FT_LOAD_RENDER);
    const FT_GlyphSlot glyph = face->glyph;
    const FT_Bitmap& bits = glyph->bitmap;
    for (unsigned int r = 0; r < bits.rows; ++r) {
      for (unsigned int q = 0; q < bits.width; ++q) {
        const unsigned char alpha =
            bits.buffer[static_cast<int>(r) * bits.pitch + static_cast<int>(q)];
        if (alpha == 0) continue;
        const double sx = pen + glyph->bitmap_left + static_cast<int>(q);
        const double sy = -glyph->bitmap_top + static_cast<int>(r);
        const long tx = std::lround(centre_x + (sy - ink_y));
        const long ty = std::lround(centre_y - (sx - ink_x));
        if (tx < 0 || tx >= width || ty < 0 || ty >= height) continue;
        mjtByte* pixel =
            rgb + (static_cast<size_t>(ty) * width + tx) * RGB_CHANNELS;
        const float ink = alpha / 255.0f;
        for (int channel = 0; channel < RGB_CHANNELS; ++channel) {
          const float ground = pixel[channel];
          pixel[channel] =
              static_cast<mjtByte>(ground + (255.0f - ground) * ink + 0.5f);
        }
      }
    }
    pen += (glyph->advance.x / 64.0) + tracking_px;
  }

  FT_Done_Face(face);
  FT_Done_FreeType(library);
}

/**
 * Paint a policy's name into the scene's floor label texture.
 *
 * Silently does nothing if the scene carries no such texture, since a
 * missing label costs a recording nothing; a texture that is present but
 * not RGB is an error, because painting it would corrupt the scene.
 *
 * @param[in,out] m   the loaded scene, whose texture data is written
 * @param[in] text    the name to paint
 * @throws std::runtime_error if the label texture is not RGB, or if the
 *         font cannot be loaded
 * @exceptsafe basic
 */
void paint_floor_texture(
    mjModel* m,
    const std::string& text
) {
  const int texture = mj_name2id(m, mjOBJ_TEXTURE, TEXTURE_NAME);
  if (texture < 0) return;
  if (m->tex_nchannel[texture] != RGB_CHANNELS) {
    throw std::runtime_error("floor: the label texture is not RGB");
  }
  paint_floor_text(
      m->tex_data + m->tex_adr[texture],
      m->tex_width[texture],
      m->tex_height[texture],
      text
  );
}

/**
 * Resolve a name in the model, or refuse to run.
 *
 * Every index this harness uses is resolved through here at load, so a
 * scene missing a joint stops the run with the name it could not find
 * rather than leaving a -1 to be indexed with later.
 *
 * @param[in] m     the loaded scene
 * @param[in] type  object type to search
 * @param[in] name  name to find
 * @param[in] what  word for the error message, such as "joint"
 * @returns the id, always non-negative
 * @throws std::runtime_error if the scene has no such object
 * @exceptsafe strong
 */
int require_id(
    const mjModel* m,
    mjtObj type,
    const char* name,
    const char* what
) {
  const int id = mj_name2id(m, type, name);
  if (id < 0) {
    throw std::runtime_error(
        std::string("sim: model has no ") + what + " '" + name + "'"
    );
  }
  return id;
}

/**
 * Resolve every index a run will need, once, at load.
 *
 * The free joint is checked rather than assumed: a scene that pinned the
 * base would otherwise run to completion and report a tour walked by a
 * robot that could not fall over. The target markers are optional, since a
 * headless run has nothing to show them to.
 *
 * @param[in] m  the loaded scene
 * @param[in] d  its state
 * @returns handles for every joint, actuator, sensor and marker
 * @throws std::runtime_error if a joint, actuator or sensor is missing, or
 *         if the floating base is not a free joint
 * @exceptsafe strong
 */
Handles handles_make(
    mjModel* m,
    mjData* d
) {
  Handles h;
  h.m = m;
  h.d = d;
  for (int i = 0; i < NUM_MOTOR; ++i) {
    const int joint = require_id(m, mjOBJ_JOINT, JOINT_NAMES[i], "joint");
    h.qpos_adr[i] = m->jnt_qposadr[joint];
    h.dof_adr[i] = m->jnt_dofadr[joint];
    h.act_id[i] = require_id(m, mjOBJ_ACTUATOR, JOINT_NAMES[i], "actuator");
  }
  const int base = require_id(m, mjOBJ_JOINT, "floating_base_joint", "joint");
  if (m->jnt_type[base] != mjJNT_FREE) {
    throw std::runtime_error("sim: floating_base_joint is not a free joint");
  }
  h.base_qpos = m->jnt_qposadr[base];
  h.base_dof = m->jnt_dofadr[base];

  h.sensor_quat = require_id(m, mjOBJ_SENSOR, "imu_quat", "sensor");
  h.sensor_gyro = require_id(m, mjOBJ_SENSOR, "imu_gyro", "sensor");

  const int target = mj_name2id(m, mjOBJ_BODY, "target");
  const int heading = mj_name2id(m, mjOBJ_BODY, "target_heading");
  h.mocap_target = target >= 0 ? m->body_mocapid[target] : -1;
  h.mocap_heading = heading >= 0 ? m->body_mocapid[heading] : -1;
  return h;
}

/**
 * Put the robot in its starting pose, upright and level.
 *
 * The pelvis is placed at standing height with an identity attitude and the
 * joints at `DEFAULT_ANGLES`, then the model is brought forward so that the
 * first control step reads a consistent state rather than the zeros
 * `mj_resetData` leaves behind.
 *
 * @param[in,out] h  handles whose data is reset in place
 * @exceptsafe no-throw
 */
void reset_to_stance(Handles& h) {
  mj_resetData(h.m, h.d);
  h.d->qpos[h.base_qpos + 0] = 0.0;
  h.d->qpos[h.base_qpos + 1] = 0.0;
  h.d->qpos[h.base_qpos + 2] = 0.793;
  h.d->qpos[h.base_qpos + 3] = 1.0;
  h.d->qpos[h.base_qpos + 4] = 0.0;
  h.d->qpos[h.base_qpos + 5] = 0.0;
  h.d->qpos[h.base_qpos + 6] = 0.0;
  for (int i = 0; i < NUM_MOTOR; ++i) {
    h.d->qpos[h.qpos_adr[i]] = DEFAULT_ANGLES[i];
  }
  mj_forward(h.m, h.d);
}

/**
 * Read what the robot's own sensors would report.
 *
 * Deliberately narrow: joint encoders and the pelvis IMU, taken through the
 * model's sensors rather than from the state directly, so a policy is given
 * what a G1 publishes and nothing the simulator happens to know as well.
 *
 * @param[in] h  resolved handles
 * @returns the observable state
 * @exceptsafe no-throw
 */
RobotState state_read(const Handles& h) {
  RobotState state{};
  for (int i = 0; i < NUM_MOTOR; ++i) {
    state.motor_q[i] = h.d->qpos[h.qpos_adr[i]];
    state.motor_dq[i] = h.d->qvel[h.dof_adr[i]];
  }
  const double* quat = h.d->sensordata + h.m->sensor_adr[h.sensor_quat];
  const double* gyro = h.d->sensordata + h.m->sensor_adr[h.sensor_gyro];
  state.imu_quat = Eigen::Vector4d(quat[0], quat[1], quat[2], quat[3]);
  state.imu_gyro = Eigen::Vector3d(gyro[0], gyro[1], gyro[2]);
  return state;
}

/**
 * Read a MuJoCo quaternion as an Eigen one.
 *
 * Both store the scalar first here, but Eigen's constructor takes it first
 * while its coefficient storage puts it last, which is the kind of thing
 * that is silently wrong for a long time; the conversion is written once.
 *
 * @param[in] q  quaternion as w, x, y, z
 * @returns the same rotation
 * @exceptsafe no-throw
 */
inline Eigen::Quaterniond quat_to_eigen(const Eigen::Vector4d& q) {
  return Eigen::Quaterniond(q[0], q[1], q[2], q[3]);
}

/**
 * Rotate a vector by a quaternion.
 *
 * Normalises first, because an attitude integrated over a two-minute run
 * drifts off the unit sphere by enough to scale what it rotates.
 *
 * @param[in] q  quaternion as w, x, y, z
 * @param[in] v  vector to rotate
 * @returns the rotated vector
 * @exceptsafe no-throw
 */
inline Eigen::Vector3d quat_rot_vec(
    const Eigen::Vector4d& q,
    const Eigen::Vector3d& v
) {
  return quat_to_eigen(q).normalized() * v;
}

/**
 * Extract a heading from an attitude.
 *
 * Takes the body's forward axis into the world and reads its bearing,
 * rather than decomposing the rotation into Euler angles. A walking robot
 * pitches and rolls constantly, and a yaw read out of a Euler triple swings
 * with that pitch — the bearing of the forward axis does not.
 *
 * @param[in] q  pelvis attitude as w, x, y, z
 * @returns heading in radians, in [-pi, pi]
 * @exceptsafe no-throw
 */
inline double quat_to_heading_yaw(const Eigen::Vector4d& q) {
  const Eigen::Vector3d ref_dir(1.0, 0.0, 0.0);
  const Eigen::Vector3d rot_dir = quat_rot_vec(q, ref_dir);
  return std::atan2(rot_dir[1], rot_dir[0]);
}

/**
 * Read the pelvis pose from the simulator, flattened to the floor.
 *
 * Used only by the harness — to anchor the tour, place the markers and
 * score arrival. No policy is shown any of it.
 *
 * @param[in] h  resolved handles
 * @returns pelvis position and heading in world coordinates
 * @exceptsafe no-throw
 */
WorldPose world_read(const Handles& h) {
  const Eigen::Vector4d quat(
      h.d->qpos[h.base_qpos + 3],
      h.d->qpos[h.base_qpos + 4],
      h.d->qpos[h.base_qpos + 5],
      h.d->qpos[h.base_qpos + 6]
  );
  return WorldPose{
      h.d->qpos[h.base_qpos + 0],
      h.d->qpos[h.base_qpos + 1],
      quat_to_heading_yaw(quat)
  };
}

/**
 * Close the joint loop and write actuator torques.
 *
 * The PD lives here rather than in the model's actuators so that every
 * policy is driven by the same arithmetic a G1's own boards would do, with
 * the gains it asked for. Torque is clamped to the actuator's declared
 * range where the scene declares one, so a policy cannot walk on a demand
 * the hardware could not have met.
 *
 * @param[in,out] h    handles whose `ctrl` is written
 * @param[in] command  targets, gains and feedforward torque
 * @exceptsafe no-throw
 */
void apply_pd(
    Handles& h,
    const MotorCommand& command
) {
  for (int i = 0; i < NUM_MOTOR; ++i) {
    const double q = h.d->qpos[h.qpos_adr[i]];
    const double dq = h.d->qvel[h.dof_adr[i]];
    double tau = command.kp[i] * (command.q_target[i] - q) +
                 command.kd[i] * (command.dq_target[i] - dq) +
                 command.tau_ff[i];
    const int act = h.act_id[i];
    if (h.m->actuator_forcelimited[act]) {
      tau = std::clamp(
          tau,
          h.m->actuator_forcerange[2 * act],
          h.m->actuator_forcerange[2 * act + 1]
      );
    }
    h.d->ctrl[act] = tau;
  }
}

/**
 * Hold the pelvis still: the crane, applied once per physics step.
 *
 * Written as a state override rather than as a constraint because it has to
 * be exact: every candidate must be released from the same pose at the same
 * height, or the handover itself becomes part of what is measured.
 *
 * @param[in,out] h  handles whose base position and velocity are pinned
 * @exceptsafe no-throw
 */
void hold_base(Handles& h) {
  h.d->qpos[h.base_qpos + 0] = 0.0;
  h.d->qpos[h.base_qpos + 1] = 0.0;
  h.d->qpos[h.base_qpos + 2] = 0.793;
  h.d->qpos[h.base_qpos + 3] = 1.0;
  h.d->qpos[h.base_qpos + 4] = 0.0;
  h.d->qpos[h.base_qpos + 5] = 0.0;
  h.d->qpos[h.base_qpos + 6] = 0.0;
  for (int i = 0; i < 6; ++i) h.d->qvel[h.base_dof + i] = 0.0;
}

/**
 * Move the scene's target markers to the current waypoint.
 *
 * Purely cosmetic: the markers are mocap bodies, so nothing they do reaches
 * the robot. When there is no target to show — during the lead-in, or once
 * the tour is over — they are parked well below the floor rather than
 * hidden, since a mocap body has nowhere to hide.
 *
 * @param[in,out] h       handles whose mocap slots are written
 * @param[in] tour        the scenario, for the waypoint being shown
 * @param[in] tick        what the last control step decided
 * @param[in] anchor      the frame the crane released
 * @param[in] have_anchor false before the release, when there is no frame
 * @exceptsafe no-throw
 */
void place_markers(
    Handles& h,
    const Tour& tour,
    const Tick& tick,
    const WorldPose& anchor,
    bool have_anchor
) {
  if (h.mocap_target < 0 || h.mocap_heading < 0) return;
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  bool show = false;
  if (have_anchor && tick.target >= 0 &&
      tick.target < static_cast<int>(tour.waypoints.size())) {
    const TourTarget& t = tour.waypoints[tick.target];
    const double c = std::cos(anchor.yaw);
    const double s = std::sin(anchor.yaw);
    x = anchor.x + t.x * c - t.y * s;
    y = anchor.y + t.x * s + t.y * c;
    yaw = anchor.yaw + t.yaw;
    show = true;
  }
  const double z = show ? 0.0 : -5.0;
  const double half = yaw * 0.5;
  for (int slot : {h.mocap_target, h.mocap_heading}) {
    h.d->mocap_pos[3 * slot + 0] = x;
    h.d->mocap_pos[3 * slot + 1] = y;
    h.d->mocap_pos[3 * slot + 2] = z + (slot == h.mocap_heading ? 0.9 : 0.0);
    h.d->mocap_quat[4 * slot + 0] = std::cos(half);
    h.d->mocap_quat[4 * slot + 1] = 0.0;
    h.d->mocap_quat[4 * slot + 2] = 0.0;
    h.d->mocap_quat[4 * slot + 3] = std::sin(half);
  }
}

/**
 * Open a rendering context, visible or not.
 *
 * A recording needs a GL context exactly as a window does, so this is also
 * how `--record` gets one: the window is created hidden and only the
 * offscreen buffer is ever read from. The camera is specified as a
 * viewpoint — a height, a look-down angle, then a dolly and a lift — and
 * converted here into the distance and elevation about the arena centre
 * that MuJoCo actually wants, because the first description is the one a
 * person can reason about and the second is not.
 *
 * @param[in,out] m   the loaded scene; its offscreen size and fov are set
 * @param[in] visible whether the window is shown
 * @returns the context, closed by `viewer_close`
 * @throws std::runtime_error if GLFW or the window cannot be created
 * @exceptsafe basic
 */
std::unique_ptr<Viewer> viewer_open(
    mjModel* m,
    bool visible
) {
  if (!glfwInit()) throw std::runtime_error("sim: glfwInit failed");
  std::unique_ptr<Viewer> v = std::make_unique<Viewer>();
  glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
  v->window =
      glfwCreateWindow(1200, 800, "teleop-walking-benchmark", nullptr, nullptr);
  if (v->window == nullptr) throw std::runtime_error("sim: no GLFW window");
  glfwMakeContextCurrent(v->window);
  glfwSwapInterval(0);
  mjv_defaultCamera(&v->camera);
  mjv_defaultOption(&v->option);
  mjv_defaultScene(&v->scene);
  mjr_defaultContext(&v->context);
  mjv_makeScene(m, &v->scene, 2000);
  m->vis.global.offwidth = RECORD_WIDTH;
  m->vis.global.offheight = RECORD_HEIGHT;
  mjr_makeContext(m, &v->context, mjFONTSCALE_150);
  const double design = CAMERA_ELEVATION_DEG * M_PI / 180.0;
  const double ray = CAMERA_HEIGHT_M / -std::sin(design) - CAMERA_DOLLY_IN_M;
  const double back = ray * std::cos(design);
  const double rise = ray * -std::sin(design) + CAMERA_RISE_M;
  v->camera.type = mjCAMERA_FREE;
  v->camera.azimuth = CAMERA_AZIMUTH_DEG;
  v->camera.elevation = -std::atan2(rise, back) * 180.0 / M_PI;
  v->camera.distance = std::hypot(back, rise);
  v->camera.lookat[0] = CENTRE_X_M;
  v->camera.lookat[1] = CENTRE_Y_M;
  v->camera.lookat[2] = 0.0;
  m->vis.global.fovy = CAMERA_FOVY_DEG;
  return v;
}

/**
 * Draw the arrow that shows where a punch landed and how hard.
 *
 * Sized by force so a glance separates a shove from a nudge, and drawn
 * behind the contact point so the head sits on the robot and the arrow
 * reads as going in rather than coming out. The orthonormal frame is built
 * from a seed axis chosen away from the push direction, since crossing two
 * nearly parallel vectors gives a length that normalises into nonsense.
 * Silently skipped if the scene is full: a missing arrow is better than a
 * frame that fails to render.
 *
 * @param[in,out] v  viewer whose scene gains one geom
 * @param[in] d      simulation state, for the contact point
 * @param[in] p      the punch to draw
 * @exceptsafe no-throw
 */
void viewer_punch_arrow(
    Viewer& v,
    const mjData* d,
    const Punch& p
) {
  if (v.scene.ngeom >= v.scene.maxgeom) return;
  double to[3];
  punch_body_point(d, p, p.draw_point, to);
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

/**
 * Render one on-screen frame and pump the window.
 *
 * Closing the window stops the whole sweep rather than just this run: it is
 * the only gesture a viewer has, and the alternative — moving silently on
 * to the next policy — is never what was meant by it.
 *
 * @param[in,out] v     viewer to render through
 * @param[in] m         the loaded scene
 * @param[in] d         simulation state
 * @param[in] overlay   text drawn in the corner
 * @param[in] active    punch to draw, or null
 * @exceptsafe no-throw
 */
void viewer_draw(
    Viewer& v,
    mjModel* m,
    mjData* d,
    const std::string& overlay,
    const Punch* active
) {
  mjrRect viewport = {0, 0, 0, 0};
  glfwGetFramebufferSize(v.window, &viewport.width, &viewport.height);
  mjv_updateScene(m, d, &v.option, nullptr, &v.camera, mjCAT_ALL, &v.scene);
  if (active != nullptr) viewer_punch_arrow(v, d, *active);
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
  if (glfwWindowShouldClose(v.window)) g_stop.store(true);
}

/**
 * Start an ffmpeg process to encode a run.
 *
 * Piping raw frames to ffmpeg keeps a codec out of this program entirely,
 * and the pipe applies backpressure: if encoding falls behind, the
 * simulation waits rather than dropping frames, which is what keeps one
 * frame per control step and therefore real-time playback true. The frames
 * are flipped on the way in because GL reads them bottom row first, and
 * written as yuv420p because a good deal of software will not play anything
 * else.
 *
 * @param[in] path  destination mp4
 * @returns the running encoder, closed by `recorder_close`
 * @throws std::runtime_error if ffmpeg cannot be started
 * @exceptsafe basic
 */
std::unique_ptr<Recorder> recorder_open(const std::string& path) {
  std::unique_ptr<Recorder> r = std::make_unique<Recorder>();
  r->width = RECORD_WIDTH;
  r->height = RECORD_HEIGHT;
  r->path = path;
  r->rgb.assign(
      static_cast<size_t>(r->width) * static_cast<size_t>(r->height) * 3,
      0
  );

  std::ostringstream cmd;
  cmd << "ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt rgb24"
      << " -s " << r->width << "x" << r->height << " -r " << RECORD_FPS
      << " -i - -vf vflip -c:v libx264 -preset veryfast -crf 20"
      << " -pix_fmt yuv420p '" << path << "'";
  r->pipe = popen(cmd.str().c_str(), "w");
  if (r->pipe == nullptr) {
    throw std::runtime_error("sim: cannot start ffmpeg for " + path);
  }
  return r;
}

/**
 * Render one frame offscreen and hand it to the encoder.
 *
 * Renders at the fixed recording size rather than at the window size, so a
 * clip is the same resolution whether anyone was watching, and restores the
 * window buffer afterwards so that a run being watched and recorded at once
 * keeps drawing. A write that fails means ffmpeg has gone, which ends the
 * run rather than filling a disk with a video nobody can play.
 *
 * @param[in,out] r  the encoder being fed
 * @param[in,out] v  viewer to render through
 * @param[in] m      the loaded scene
 * @param[in] d      simulation state
 * @param[in] active punch to draw, or null
 * @throws std::runtime_error if the encoder stops accepting frames
 * @exceptsafe basic
 */
void recorder_frame(
    Recorder& r,
    Viewer& v,
    mjModel* m,
    mjData* d,
    const Punch* active
) {
  const mjrRect viewport = {0, 0, r.width, r.height};
  mjv_updateScene(m, d, &v.option, nullptr, &v.camera, mjCAT_ALL, &v.scene);
  if (active != nullptr) viewer_punch_arrow(v, d, *active);
  mjr_setBuffer(mjFB_OFFSCREEN, &v.context);
  mjr_render(viewport, &v.scene, &v.context);
  mjr_readPixels(r.rgb.data(), nullptr, viewport, &v.context);
  mjr_setBuffer(mjFB_WINDOW, &v.context);
  if (fwrite(r.rgb.data(), 1, r.rgb.size(), r.pipe) != r.rgb.size()) {
    throw std::runtime_error("sim: ffmpeg stopped accepting frames");
  }
  ++r.frames;
}

/**
 * Close the encoder and report what was written.
 *
 * Takes ownership so the pipe cannot be closed twice, and prints the frame
 * count with the playback length it implies, which is the quickest way to
 * see that a clip covers the run rather than the first few seconds of it.
 *
 * @param[in] r  the encoder, consumed; a null pointer is accepted
 * @exceptsafe basic
 */
void recorder_close(std::unique_ptr<Recorder> r) {
  if (!r) return;
  if (r->pipe != nullptr) pclose(r->pipe);
  std::cout << "sim: wrote " << r->path << " (" << r->frames << " frames, "
            << std::fixed << std::setprecision(2)
            << static_cast<double>(r->frames) / RECORD_FPS << " s at "
            << RECORD_FPS << " fps)" << std::endl;
}

/**
 * Release the rendering context.
 *
 * Takes ownership for the same reason `recorder_close` does: a sweep opens
 * and closes one of these per policy, and a double free thirteen runs in
 * would be an expensive way to learn about it.
 *
 * @param[in] v  the viewer, consumed; a null pointer is accepted
 * @exceptsafe no-throw
 */
void viewer_close(std::unique_ptr<Viewer> v) {
  if (!v) return;
  mjr_freeContext(&v->context);
  mjv_freeScene(&v->scene);
  glfwDestroyWindow(v->window);
  glfwTerminate();
}

/**
 * Refuse to run if the scene's floor mark is not where the tour is drawn.
 *
 * The arena lives in two places — this file's constants and the scene XML —
 * and nothing links them. If they drift apart the run still scores
 * correctly but every recording shows the robot walking around a circle it
 * is not being scored against, which is a slow and unpleasant thing to
 * discover from a video. A scene with no mark at all is fine.
 *
 * @param[in] m  the loaded scene
 * @throws std::runtime_error if the mark and the constants disagree
 * @exceptsafe strong
 */
void centre_check(const mjModel* m) {
  const int mark = mj_name2id(m, mjOBJ_GEOM, CENTRE_GEOM);
  if (mark < 0) return;
  const double dx = m->geom_pos[3 * mark] - CENTRE_X_M;
  const double dy = m->geom_pos[3 * mark + 1] - CENTRE_Y_M;
  if (std::abs(dx) > 1e-9 || std::abs(dy) > 1e-9) {
    throw std::runtime_error(
        "sim: the tour is drawn around CENTRE_X_M/CENTRE_Y_M but the "
        "floor_mark geom sits somewhere else; move one to match the other"
    );
  }
}

/**
 * Load the scene that every run will share.
 *
 * Loaded once and lent to `sim_run`, which is what lets a sweep hold many
 * tours in one process: nothing in a run writes through the model, so the
 * same scene backs every worker thread and only `mjData` is per run. The
 * pointer is not const only because the viewer and marker helpers spell
 * their parameter `mjModel*`; no caller writes through it.
 *
 * @param[in] path  the scene to load
 * @returns the loaded model, owned
 * @throws std::runtime_error if the scene will not load or is off centre
 * @exceptsafe strong
 */
std::unique_ptr<
    mjModel,
    void (*)(mjModel*)>
scene_load(const std::string& path) {
  char error[1024] = "";
  mjModel* m = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
  if (m == nullptr) {
    throw std::runtime_error("sim: cannot load " + path + ": " + error);
  }
  centre_check(m);
  return {m, [](mjModel* p) { mj_deleteModel(p); }};
}

/**
 * Run one policy over one tour and report what happened.
 *
 * The loop keeps two clocks. Physics advances at the model's own timestep
 * while the policy is stepped every `PERIOD_S` of simulated time, so a slow
 * candidate is shown exactly the world a fast one is shown and the table
 * separates policies by their walking rather than by their latency.
 * `--realtime` only ever holds a run back, and never lets it get ahead.
 *
 * The crane holds the base until the first control step leaves the ramp;
 * after the release the pelvis height is watched, and a run that puts it on
 * the floor ends there rather than dragging the robot through the rest of
 * the tour to no purpose. A punch is cleared from the body it was applied
 * to before the next is looked up, since MuJoCo's external wrench persists
 * until something overwrites it.
 *
 * @param[in] config  scene, seed, pacing and recording
 * @param[in] policy  the candidate, already initialised
 * @param[in] tour    the waypoints to walk
 * @returns the finished report, means included
 * @throws std::runtime_error if the scene cannot be loaded or checked, if
 *         the recording cannot be started, or if the encoder fails
 * @exceptsafe basic
 */
Report sim_run(
    const SimConfig& config,
    std::shared_ptr<ModelPolicy> policy,
    const Tour& tour,
    mjModel* m
) {
  const std::unique_ptr<mjData, void (*)(mjData*)> data(
      mj_makeData(m),
      [](mjData* p) { mj_deleteData(p); }
  );
  mjData* d = data.get();
  if (d == nullptr) throw std::runtime_error("sim: cannot allocate mjData");
  Handles h = handles_make(m, d);
  reset_to_stance(h);
  CostMeter meter = meter_make();

  const std::shared_ptr<Loop> loop = loop_make(policy, tour);
  loop->quiet = config.quiet;

  const double cap =
      config.max_seconds > 0.0
          ? config.max_seconds
          : INIT_DURATION_S + tour_duration(tour) + TIMEOUT_MARGIN_S;

  const Schedule punches = schedule_make(
      m,
      config.seed,
      INIT_DURATION_S + tour.config.lead_in_s,
      tour.config.point_s,
      static_cast<int>(tour.waypoints.size())
  );

  MotorCommand command{};
  command.q_target.setZero();
  command.dq_target.setZero();
  command.tau_ff.setZero();
  command.kp = KPS;
  command.kd = KDS;
  for (int i = 0; i < NUM_MOTOR; ++i) {
    command.q_target[i] = static_cast<float>(DEFAULT_ANGLES[i]);
  }

  const Punch* active_punch = nullptr;

  Tick tick{};
  tick.phase = Phase::INIT;
  tick.target = -1;
  WorldPose anchor{0.0, 0.0, 0.0};
  bool have_anchor = false;
  bool released = false;

  Outcome outcome = Outcome::TIMEOUT;
  std::string detail;

  const bool recording = !config.record_dir.empty();
  std::unique_ptr<Viewer> viewer;
  std::unique_ptr<Recorder> recorder;
  if (config.viewer || recording) {
    paint_floor_texture(m, policy->name());
    viewer = viewer_open(m, config.viewer);
  }
  if (recording) {
    std::error_code ec;
    std::filesystem::create_directories(config.record_dir, ec);
    if (ec) {
      throw std::runtime_error("sim: cannot create " + config.record_dir);
    }
    recorder = recorder_open(config.record_dir + "/" + policy->name() + ".mp4");
  }
  double last_draw = -1.0;

  const std::chrono::steady_clock::time_point wall_start =
      std::chrono::steady_clock::now();
  double next_control = 0.0;
  double next_frame = 0.0;
  const double frame_s = 1.0 / RECORD_FPS;

  while (d->time < cap) {
    if (g_stop.load()) {
      outcome = Outcome::INTERRUPTED;
      detail = "stopped";
      break;
    }

    if (d->time >= next_control - 1e-9) {
      next_control += PERIOD_S;
      const RobotState state = state_read(h);
      const WorldPose world = world_read(h);
      tick = loop_step(*loop, state, world, d->time);
      command = tick.command;
      if (!released && tick.phase != Phase::INIT) {
        released = true;
        anchor = world;
        have_anchor = true;
        if (!config.quiet) {
          std::cout << "sim: crane released at t=" << d->time << " s"
                    << std::endl;
        }
      }
      place_markers(h, tour, tick, anchor, have_anchor);
      meter_mark(meter, tick.target, d->time);
      if (tick.phase == Phase::DONE) {
        outcome = Outcome::COMPLETE;
        break;
      }
    }

    apply_pd(h, command);
    if (!released) hold_base(h);

    if (active_punch != nullptr) {
      std::fill(
          d->xfrc_applied + 6 * active_punch->body,
          d->xfrc_applied + 6 * active_punch->body + 6,
          0.0
      );
    }
    active_punch = schedule_active(punches, d->time);
    if (active_punch != nullptr) punch_apply(m, d, *active_punch);

    mj_step(m, d);
    meter_step(meter, h, m->opt.timestep);

    if (released && d->qpos[h.base_qpos + 2] < FALL_PELVIS_Z) {
      outcome = Outcome::FELL;
      std::ostringstream why;
      why << "pelvis " << std::fixed << std::setprecision(2)
          << d->qpos[h.base_qpos + 2] << " m at t=" << std::setprecision(1)
          << d->time << " s";
      detail = why.str();
      break;
    }

    const double wall_now = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - wall_start
    )
                                .count();
    const bool draw_now =
        viewer && config.viewer && wall_now - last_draw >= 1.0 / 60.0;
    const bool frame_now = recorder && d->time >= next_frame - 1e-9;
    if (draw_now || frame_now) {
      const Punch* shown_punch = schedule_visible(punches, d->time);
      std::ostringstream hud;
      hud << policy->name() << "\nt " << std::fixed << std::setprecision(2)
          << d->time << " s\ntarget " << tick.target << "\ndist " << tick.dist_m
          << " m";
      if (active_punch != nullptr) {
        hud << "\nPUNCH " << std::setprecision(0) << active_punch->force_n
            << " N on " << mj_id2name(m, mjOBJ_BODY, active_punch->body);
      }
      if (frame_now) {
        next_frame += frame_s;
        recorder_frame(*recorder, *viewer, m, d, shown_punch);
      }
      if (draw_now) {
        last_draw = wall_now;
        viewer_draw(*viewer, m, d, hud.str(), shown_punch);
      }
    }

    if (config.realtime > 0.0) {
      const double target_wall = d->time / config.realtime;
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - wall_start
      )
                                 .count();
      if (target_wall > elapsed) {
        std::this_thread::sleep_for(
            std::chrono::duration<double>(target_wall - elapsed)
        );
      }
    }
  }

  recorder_close(std::move(recorder));
  viewer_close(std::move(viewer));

  meter_close(meter, d->time, false);

  Report report = loop_report(*loop);
  report.outcome = outcome;
  report.detail = detail;
  report.duration_s = d->time;
  report.segments = std::move(meter.done);

  return report_finalize(report);
}

/**
 * Print the command line synopsis.
 *
 * Short by design: everything that decides what a run measures — the target
 * clock, the arena, the punch forces, the gains — is a constant in this
 * file rather than a flag, so that two people quoting a policy's numbers
 * are quoting the same benchmark.
 *
 * @exceptsafe basic
 */
void usage() {
  std::cout
      << "usage: teleop-walking-benchmark --policy NAME [--realtime R] "
         "[--seed N] [--record DIR]\n"
         "       teleop-walking-benchmark --policy NAME --runs N "
         "[--parallel W] [--csv FILE]\n"
         "\n"
         "  --policy NAME     the one policy this invocation measures; "
         "required.\n"
         "                    One per process: holding the whole field at "
         "once\n"
         "                    costs more memory than the runs do. Run the "
         "binary\n"
         "                    once per candidate to sweep the field\n"
         "  --realtime R      open the viewer and pace to R x real time: 1 is "
         "real\n"
         "                    time, 0.1 ten times slower, 5 five times "
         "faster, 0 as\n"
         "                    fast as the machine can. R only ever holds a "
         "run back.\n"
         "                    WITHOUT this flag there is no window and no "
         "sleeping\n"
         "  --seed N          seeds the tour and the punch campaign "
         "(default "
      << DEFAULT_SEED
      << ")\n"
         "  --record DIR      write DIR/<policy>.mp4, one frame per 0.02 s "
         "of\n"
         "                    simulated time, so playback is exactly real "
         "time\n"
         "\n"
         "Campaign flags. A sweep runs the chosen policy over a range of "
         "seeds with\n"
         "several tours in flight, so the memory it needs is set by "
         "--parallel:\n"
         "\n"
         "  --runs N          run each policy over seeds 0..N-1\n"
         "  --seeds A-B       run each policy over seeds A..B inclusive\n"
         "  --parallel W      W tours in flight at once, one core each "
         "(default 1)\n"
         "  --csv FILE        record somewhere other than the default "
      << DEFAULT_CSV
      << ".\n"
         "                    Every run appends, sweep or not: one 'run' row "
         "plus\n"
         "                    one 'seg' row per leg of its tour, carrying "
         "per-joint\n"
         "                    energy and vibration, flushed as it goes. The "
         "header\n"
         "                    is written only to an empty file, so a campaign\n"
         "                    resumes onto the same file\n"
         "\n"
         "Everything else a run needs — the target clock, the arena, the "
         "punch forces,\n"
         "the scene and the precision — is fixed in main.cpp.\n"
      << std::endl;
}

/**
 * Parse a seed from the command line.
 *
 * Rejects anything that does not fit in 32 bits rather than truncating it,
 * because a silently narrowed seed would name a different tour from the one
 * that was asked for and quietly break a sweep's reproducibility.
 *
 * @param[in] flag   flag being parsed, for the error message
 * @param[in] value  the argument text
 * @returns the seed
 * @throws std::runtime_error if the text is not a seed in range
 * @exceptsafe strong
 */
uint32_t arg_seed(
    const std::string& flag,
    const char* value
) {
  try {
    const unsigned long parsed = std::stoul(value);
    if (parsed > 0xFFFFFFFFul) throw std::out_of_range(value);
    return static_cast<uint32_t>(parsed);
  } catch (const std::exception&) {
    throw std::runtime_error(
        flag + ": expected a seed in 0..4294967295, got '" +
        std::string(value) + "'"
    );
  }
}

/**
 * Parse a number from the command line.
 *
 * @param[in] flag   flag being parsed, for the error message
 * @param[in] value  the argument text
 * @returns the value
 * @throws std::runtime_error if the text is not a number
 * @exceptsafe strong
 */
double arg_double(
    const std::string& flag,
    const char* value
) {
  try {
    return std::stod(value);
  } catch (const std::exception&) {
    throw std::runtime_error(flag + ": expected a number, got '" + value + "'");
  }
}

/** Every policy this binary was built with. Defined beside the includes. */
std::vector<std::string> policy_names();

/**
 * Join the policy names for an error message.
 *
 * @returns "a, b, c"
 * @exceptsafe basic
 */
std::string policy_list() {
  std::string list;
  for (const std::string& n : policy_names()) {
    if (!list.empty()) list += ", ";
    list += n;
  }
  return list;
}

/**
 * Parse a positive count from the command line.
 *
 * @param[in] flag   flag being parsed, for the error message
 * @param[in] value  the argument text
 * @returns the value
 * @throws std::runtime_error if the text is not a positive number
 * @exceptsafe strong
 */
int arg_count(
    const std::string& flag,
    const char* value
) {
  try {
    const int parsed = std::stoi(value);
    if (parsed < 1) throw std::out_of_range(value);
    return parsed;
  } catch (const std::exception&) {
    throw std::runtime_error(
        flag + ": expected a count of 1 or more, got '" + std::string(value) +
        "'"
    );
  }
}

/**
 * Parse an inclusive seed range, `A-B` or a bare `A`.
 *
 * @param[in] flag   flag being parsed, for the error message
 * @param[in] value  the argument text
 * @returns first and last seed, inclusive
 * @throws std::runtime_error if the text will not parse or runs backwards
 * @exceptsafe strong
 */
std::pair<
    uint32_t,
    uint32_t>
arg_range(
    const std::string& flag,
    const char* value
) {
  const std::string text = value;
  const size_t dash = text.find('-', 1);
  if (dash == std::string::npos) {
    const uint32_t only = arg_seed(flag, text.c_str());
    return {only, only};
  }
  const uint32_t first = arg_seed(flag, text.substr(0, dash).c_str());
  const uint32_t last = arg_seed(flag, text.substr(dash + 1).c_str());
  if (last < first) {
    throw std::runtime_error(
        flag + ": the range '" + text + "' ends before it starts"
    );
  }
  return {first, last};
}

/**
 * Parse the command line.
 *
 * An unknown flag stops the program rather than being ignored: the flags
 * that exist change how a run is watched, never what it measures, so a
 * misspelled one that was silently dropped would produce a plausible table
 * answering a different question. `--record` is refused outright by a
 * binary built without the renderer, at parse time, rather than after the
 * first policy has run.
 *
 * @param[in] argc  argument count
 * @param[in] argv  arguments, `argv[0]` skipped
 * @returns the parsed configuration
 * @throws std::runtime_error on an unknown flag, a missing value or a value
 *         that will not parse
 * @exceptsafe basic
 */
Args parse(
    int argc,
    char** argv
) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    const auto next = [&](const char* what) -> const char* {
      if (i + 1 >= argc) {
        throw std::runtime_error(flag + ": expected " + what);
      }
      return argv[++i];
    };
    if (flag == "--policy") {
      if (!args.policy.empty()) {
        throw std::runtime_error(
            "--policy: one policy per invocation; run the binary once per "
            "candidate"
        );
      }
      args.policy = next("a policy name");
    } else if (flag == "--realtime") {
      args.sim.realtime = arg_double(flag, next("a rate"));
      if (args.sim.realtime < 0.0) {
        throw std::runtime_error("--realtime: a rate cannot be negative");
      }
      args.sim.viewer = true;
    } else if (flag == "--record") {
      args.sim.record_dir = next("a directory");
    } else if (flag == "--seed") {
      const uint32_t seed = arg_seed(flag, next("a seed"));
      args.scenario.seed = seed;
      args.sim.seed = seed;
    } else if (flag == "--runs") {
      args.sweep.enabled = true;
      args.sweep.first = 0;
      args.sweep.last =
          static_cast<uint32_t>(arg_count(flag, next("a run count"))) - 1;
    } else if (flag == "--seeds") {
      const auto range = arg_range(flag, next("a seed range"));
      args.sweep.enabled = true;
      args.sweep.first = range.first;
      args.sweep.last = range.second;
    } else if (flag == "--parallel") {
      args.sweep.parallel = arg_count(flag, next("a worker count"));
    } else if (flag == "--csv") {
      args.sweep.csv = next("a file path");
    } else if (flag == "--help" || flag == "-h") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument '" + flag + "'");
    }
  }

  if (args.policy.empty()) {
    throw std::runtime_error(
        "--policy is required: one policy per invocation. Valid names: " +
        policy_list()
    );
  }
  if (args.sweep.parallel > 1 && !args.sweep.enabled) {
    throw std::runtime_error(
        "--parallel: there is nothing to run in parallel without --runs or "
        "--seeds"
    );
  }

  if (args.sweep.parallel > 1 && args.sim.viewer) {
    throw std::runtime_error("--parallel: cannot pace a sweep with --realtime");
  }
  if (args.sweep.parallel > 1 && !args.sim.record_dir.empty()) {
    throw std::runtime_error("--parallel: cannot record a sweep with --record");
  }
  return args;
}

/**
 * SIGINT handler: ask the run to stop at its next opportunity.
 *
 * Does nothing but set the flag, which is all that is safe from a signal
 * handler; the loop notices and leaves with the report it has, so an
 * interrupted sweep still prints the policies that finished.
 *
 * @exceptsafe no-throw
 */
void on_sigint(int) { g_stop.store(true); }

/**
 * Pin one worker to one core.
 *
 * A sweep's workers are spread deliberately rather than left to the
 * scheduler: the process-wide pin that a single run uses would put every
 * worker on the same core, and migrating workers would make the step
 * column say more about the scheduler than about the networks.
 *
 * A failure is reported once and stepped over. An unpinned worker still
 * walks its tours correctly; only its timings become untrustworthy.
 *
 * @param[in] worker  index of this worker, used to pick the core
 * @exceptsafe no-throw
 */
void pin_worker_to_core(int worker) {
  const long cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (cores < 1) return;
  const int core = static_cast<int>(worker % cores);
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
    std::cout << "cpu: cannot pin worker " << worker << " to core " << core
              << ", leaving it unpinned" << std::endl;
  }
}

/**
 * Pin the process to one core.
 *
 * The step column is the point of this: a policy that migrates between
 * cores, or that quietly fans out across a thread pool, reports a time that
 * says more about the machine's scheduler than about the network. One core
 * and one thread make the figure mean the same thing for every candidate.
 *
 * Every failure here is reported and stepped over rather than raised. An
 * unpinnable run still walks the tour correctly; only its timings become
 * untrustworthy, and saying so is more useful than refusing to run.
 *
 * @exceptsafe basic
 */
void pin_to_one_core() {
  const int core = sched_getcpu();
  if (core < 0) {
    std::cout << "cpu: cannot read the current core, leaving the run unpinned"
              << std::endl;
    return;
  }

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    std::cout << "cpu: cannot pin to core " << core << " ("
              << std::strerror(errno) << "), leaving the run unpinned"
              << std::endl;
    return;
  }

  std::cout << "cpu: pinned to core " << core << std::endl;
}

/**
 * Print a TensorRT diagnostic, unless it is routine.
 *
 * Info and verbose are dropped: a plan build otherwise buries the
 * benchmark's own output under several hundred lines about layer fusion,
 * and everything worth reading is a warning or worse.
 *
 * @param[in] severity  TensorRT's own severity
 * @param[in] msg       the message
 * @exceptsafe no-throw
 */
void EngineLogger::log(
    Severity severity,
    const char* msg
) noexcept {
  if (severity == Severity::kINFO || severity == Severity::kVERBOSE) return;
  std::cout << "engine: " << msg << std::endl;
}

/**
 * The one logger every TensorRT object is built against.
 *
 * TensorRT keeps the pointer it is given for the lifetime of everything
 * built from it, so the logger has to outlive the engines; a function-local
 * static is the shortest way to promise that.
 *
 * @returns the shared logger
 * @exceptsafe no-throw
 */
EngineLogger& engine_logger() {
  static EngineLogger instance;
  return instance;
}

/**
 * Turn a CUDA status into an exception.
 *
 * Nothing on this path can carry on usefully after a CUDA failure, and a
 * policy that throws during `init` only costs itself: the sweep records it
 * as an error and runs the rest of the field.
 *
 * @param[in] status  the status to check
 * @param[in] what    the call being reported, for the message
 * @throws std::runtime_error if the status is not success
 * @exceptsafe strong
 */
void cuda_check(
    cudaError_t status,
    const char* what
) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("engine: ") + what + ": " + cudaGetErrorString(status)
    );
  }
}

/**
 * Write a built plan into the cache.
 *
 * Written to a unique temporary and renamed, so a build interrupted part
 * way through leaves no truncated plan for a later run to deserialise, and
 * two sweeps started at once cannot write over each other. Every failure is
 * swallowed: the cache is an optimisation, and a run that cannot save its
 * plan should still walk the tour.
 *
 * @param[in] path   final path in the cache
 * @param[in] bytes  the serialised engine
 * @exceptsafe no-throw
 */
void plan_write(
    const std::string& path,
    const std::vector<char>& bytes
) {
  std::error_code ec;
  std::filesystem::create_directories(PLAN_CACHE_DIR, ec);
  if (ec) return;

  const std::string temporary =
      path + ".tmp" +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()
      );
  {
    std::ofstream out(temporary, std::ios::binary);
    if (!out ||
        !out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
      std::filesystem::remove(temporary, ec);
      return;
    }
  }
  std::filesystem::rename(temporary, path, ec);
  if (ec) std::filesystem::remove(temporary, ec);
}

/**
 * Build a TensorRT plan from an ONNX file.
 *
 * Every dynamic axis is pinned to one. Deployment is a single robot at 50
 * Hz, so there is no batch to speak of, and a plan built for one shape is
 * both faster and free of the question of which profile a run happened to
 * select.
 *
 * @param[in] model_path  the ONNX file
 * @param[in] fp16        whether half precision kernels are allowed
 * @returns the serialised engine
 * @throws std::runtime_error if the file cannot be parsed or the build
 *         fails
 * @exceptsafe basic
 */
std::vector<char> plan_build(
    const std::string& model_path,
    bool fp16
) {
  const std::unique_ptr<nvinfer1::IBuilder> builder(
      nvinfer1::createInferBuilder(engine_logger())
  );
  if (!builder) throw std::runtime_error("engine: no builder");

  const std::unique_ptr<nvinfer1::INetworkDefinition> network(
      builder->createNetworkV2(0)
  );
  if (!network) throw std::runtime_error("engine: no network");

  const std::unique_ptr<nvonnxparser::IParser> parser(
      nvonnxparser::createParser(*network, engine_logger())
  );
  if (!parser || !parser->parseFromFile(
                     model_path.c_str(),
                     static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)
                 )) {
    throw std::runtime_error("engine: cannot parse " + model_path);
  }

  const std::unique_ptr<nvinfer1::IBuilderConfig> config(
      builder->createBuilderConfig()
  );
  if (!config) throw std::runtime_error("engine: no builder config");
  config->setMemoryPoolLimit(
      nvinfer1::MemoryPoolType::kWORKSPACE,
      WORKSPACE_BYTES
  );
  if (fp16) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
#pragma GCC diagnostic pop
  }

  nvinfer1::IOptimizationProfile* profile =
      builder->createOptimizationProfile();
  bool dynamic = false;
  for (int i = 0; i < network->getNbInputs(); ++i) {
    nvinfer1::ITensor* tensor = network->getInput(i);
    nvinfer1::Dims dims = tensor->getDimensions();
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] < 0) {
        dims.d[d] = 1;
        dynamic = true;
      }
    }
    for (nvinfer1::OptProfileSelector which :
         {nvinfer1::OptProfileSelector::kMIN,
          nvinfer1::OptProfileSelector::kOPT,
          nvinfer1::OptProfileSelector::kMAX}) {
      profile->setDimensions(tensor->getName(), which, dims);
    }
  }
  if (dynamic) config->addOptimizationProfile(profile);

  std::cout << "engine: building a TensorRT plan for " << model_path << " ("
            << (fp16 ? "fp16" : "fp32") << ")" << std::endl;

  const std::unique_ptr<nvinfer1::IHostMemory> serialised(
      builder->buildSerializedNetwork(*network, *config)
  );
  if (!serialised) {
    throw std::runtime_error("engine: build failed for " + model_path);
  }
  const char* data = static_cast<const char*>(serialised->data());
  return std::vector<char>(data, data + serialised->size());
}

/**
 * Checksum a file, to name it in the plan cache.
 *
 * An identifier, not an integrity check: it exists so that editing or
 * replacing a checkpoint misses the cache instead of silently running the
 * plan built from its predecessor.
 *
 * @param[in] path  file to read
 * @returns its CRC-32
 * @throws std::runtime_error if the file cannot be read
 * @exceptsafe basic
 */
uint32_t crc32_file(const std::string& path) {
  static const std::array<uint32_t, 256> table = [] {
    std::array<uint32_t, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) {
        c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      }
      t[i] = c;
    }
    return t;
  }();

  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("engine: cannot read " + path);

  uint32_t crc = 0xFFFFFFFFu;
  char buffer[8192];
  while (in) {
    in.read(buffer, sizeof(buffer));
    const std::streamsize n = in.gcount();
    for (std::streamsize i = 0; i < n; ++i) {
      crc = table[(crc ^ static_cast<unsigned char>(buffer[i])) & 0xFFu] ^
            (crc >> 8);
    }
  }
  return ~crc;
}

/**
 * Render a 32-bit value as eight lower-case hex digits.
 *
 * @param[in] v  the value
 * @returns exactly eight characters, zero padded
 * @exceptsafe basic
 */
std::string hex8(uint32_t v) {
  std::ostringstream out;
  out << std::hex << std::setw(8) << std::setfill('0') << v;
  return out.str();
}

/**
 * Name the CUDA runtime this process is using, for the cache key.
 *
 * Falls back to the version compiled against if the runtime will not answer,
 * which is conservative in the right direction: a wrong tag costs a rebuild,
 * where a missing one would risk loading a plan built elsewhere.
 *
 * @returns a tag such as "cu12.4"
 * @exceptsafe basic
 */
std::string cuda_tag() {
  int version = 0;
  if (cudaRuntimeGetVersion(&version) != cudaSuccess || version == 0) {
    version = CUDART_VERSION;
  }
  return "cu" + std::to_string(version / 1000) + "." +
         std::to_string((version % 1000) / 10);
}

/**
 * Name the TensorRT version, for the cache key.
 *
 * @returns a tag such as "trt10.3.0"
 * @exceptsafe basic
 */
std::string trt_tag() {
  return "trt" + std::to_string(NV_TENSORRT_MAJOR) + "." +
         std::to_string(NV_TENSORRT_MINOR) + "." +
         std::to_string(NV_TENSORRT_PATCH);
}

/**
 * Name the current GPU, for the cache key.
 *
 * Stripped to alphanumerics so it is safe in a filename. A card that cannot
 * be identified is tagged as unknown rather than left out, so that plans
 * from an unidentifiable machine still miss the cache on a different one.
 *
 * @returns a tag such as "NVIDIAGeForceRTX4090"
 * @exceptsafe basic
 */
std::string gpu_tag() {
  int device = 0;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return "unknowngpu";
  }
  std::string tag;
  for (const char c : std::string(properties.name)) {
    if (std::isalnum(static_cast<unsigned char>(c)) != 0) tag.push_back(c);
  }
  return tag.empty() ? std::string("unknowngpu") : tag;
}

/**
 * Where a given checkpoint's plan is cached.
 *
 * The key carries everything a plan is only valid under: the weights, the
 * TensorRT and CUDA versions, the card, and the precision. A plan is
 * machine-specific and there is no cheap way to ask one whether it belongs
 * here, so the name is made to answer that instead — a changed machine
 * misses the cache rather than deserialising something that was never meant
 * for it.
 *
 * @param[in] model_path  the ONNX file
 * @param[in] fp16        whether half precision kernels are allowed
 * @returns the path, under `PLAN_CACHE_DIR`
 * @throws std::runtime_error if the model file cannot be read
 * @exceptsafe basic
 */
std::string plan_cache_path(
    const std::string& model_path,
    bool fp16
) {
  std::string stem;
  for (const char c : model_path) {
    stem.push_back(std::isalnum(static_cast<unsigned char>(c)) != 0 ? c : '_');
  }
  return std::string(PLAN_CACHE_DIR) + "/" + stem + "_" +
         hex8(crc32_file(model_path)) + "_" + trt_tag() + "_" + cuda_tag() +
         "_" + gpu_tag() + "_" + (fp16 ? "fp16" : "fp32") + ".plan";
}

/**
 * Read a cached plan, if there is one to read.
 *
 * Every failure returns empty rather than throwing, because there is
 * nothing a caller could do about a missing or unreadable cache entry
 * except build the plan, which is what an empty result asks for.
 *
 * @param[in] path  the cache entry
 * @returns the bytes, or empty
 * @exceptsafe basic
 */
std::vector<char> plan_read(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) return {};
  const std::streamsize size = in.tellg();
  if (size <= 0) return {};
  in.seekg(0, std::ios::beg);
  std::vector<char> bytes(static_cast<size_t>(size));
  if (!in.read(bytes.data(), size)) return {};
  return bytes;
}

/**
 * Get a plan for a checkpoint: from the cache, or by building it.
 *
 * Building costs far more than the tour does, and a sweep loads thirteen
 * networks, so the first run pays for the rest.
 *
 * @param[in] model_path  the ONNX file
 * @param[in] fp16        whether half precision kernels are allowed
 * @returns the plan and the path it is cached at
 * @throws std::runtime_error if no plan can be built
 * @exceptsafe basic
 */
Plan plan_load_or_build(
    const std::string& model_path,
    bool fp16
) {
  const std::string path = plan_cache_path(model_path, fp16);
  std::vector<char> cached = plan_read(path);
  if (!cached.empty()) {
    std::cout << "engine: cached plan for " << model_path << " (" << path << ")"
              << std::endl;
    return Plan{std::move(cached), path};
  }
  Plan plan{plan_build(model_path, fp16), path};
  plan_write(path, plan.bytes);
  return plan;
}

/**
 * Deserialise a plan and bind every tensor it declares.
 *
 * A plan that will not deserialise is assumed to be a stale cache entry
 * rather than a broken installation: it is deleted, rebuilt once, and only
 * then given up on. That is the failure a version bump produces, and it
 * should cost a rebuild rather than a puzzled afternoon.
 *
 * Non-float bindings are refused here, at load, because a quantised
 * checkpoint would otherwise be fed float bytes and answer with plausible
 * nonsense for a whole tour.
 *
 * @param[in,out] e       engine to populate; left owning nothing on failure
 * @param[in] model_path  the ONNX file, for the cache key and messages
 * @param[in] fp16        whether half precision kernels are allowed
 * @throws std::runtime_error if there is no CUDA device, if the plan cannot
 *         be built or deserialised, or if a binding is not float
 * @exceptsafe basic
 */
void engine_open(
    Engine& e,
    const std::string& model_path,
    bool fp16
) {
  int devices = 0;
  cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
  if (devices < 1) throw std::runtime_error("engine: no CUDA device");

  Plan plan = plan_load_or_build(model_path, fp16);

  e.runtime.reset(nvinfer1::createInferRuntime(engine_logger()));
  if (!e.runtime) throw std::runtime_error("engine: no runtime");
  e.cuda_engine.reset(
      e.runtime->deserializeCudaEngine(plan.bytes.data(), plan.bytes.size())
  );
  if (!e.cuda_engine && !plan.cache_path.empty()) {
    std::cout << "engine: discarding an unreadable cached plan ("
              << plan.cache_path << ")" << std::endl;
    std::error_code ec;
    std::filesystem::remove(plan.cache_path, ec);
    plan.bytes = plan_build(model_path, fp16);
    plan_write(plan.cache_path, plan.bytes);
    e.cuda_engine.reset(
        e.runtime->deserializeCudaEngine(plan.bytes.data(), plan.bytes.size())
    );
  }
  if (!e.cuda_engine) {
    throw std::runtime_error(
        "engine: cannot deserialise the plan for " + model_path
    );
  }
  e.context.reset(e.cuda_engine->createExecutionContext());
  if (!e.context) throw std::runtime_error("engine: no execution context");

  cudaStream_t raw_stream = nullptr;
  cuda_check(cudaStreamCreate(&raw_stream), "cudaStreamCreate");
  e.stream.reset(raw_stream, [](cudaStream_t s) {
    if (s != nullptr) cudaStreamDestroy(s);
  });

  for (int i = 0; i < e.cuda_engine->getNbIOTensors(); ++i) {
    const char* name = e.cuda_engine->getIOTensorName(i);
    const bool is_input =
        e.cuda_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
    if (e.cuda_engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error(
          "engine: " + model_path + " tensor '" + std::string(name) +
          "' is not float"
      );
    }

    Tensor t;
    t.name = name;
    nvinfer1::Dims dims = e.cuda_engine->getTensorShape(name);
    t.count = 1;
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] < 0) dims.d[d] = 1;
      t.shape.push_back(dims.d[d]);
      t.count *= static_cast<size_t>(dims.d[d]);
    }
    t.data.assign(t.count, 0.0f);

    if (is_input && !e.context->setInputShape(name, dims)) {
      throw std::runtime_error(
          "engine: cannot pin the shape of input '" + t.name + "'"
      );
    }

    void* device = nullptr;
    cuda_check(cudaMalloc(&device, t.count * sizeof(float)), "cudaMalloc");
    e.buffers.emplace_back(device, [](void* p) { cudaFree(p); });
    if (!e.context->setTensorAddress(name, device)) {
      throw std::runtime_error("engine: cannot bind tensor '" + t.name + "'");
    }

    (is_input ? e.device_in : e.device_out).push_back(device);
    (is_input ? e.inputs : e.outputs).push_back(std::move(t));
  }
}

/**
 * Load a checkpoint into a ready-to-run engine.
 *
 * The file is checked before TensorRT is involved, so the common mistake —
 * a policy whose weights were never vendored — is reported as a missing
 * file rather than as a parser error.
 *
 * @param[in] model_path  the ONNX file
 * @returns the loaded engine
 * @throws std::runtime_error if the model is missing or cannot be loaded
 * @exceptsafe basic
 */
std::shared_ptr<Engine> engine_init(const std::string& model_path) {
  if (!std::ifstream(model_path).good()) {
    throw std::runtime_error("engine: cannot read model " + model_path);
  }
  std::shared_ptr<Engine> e = std::make_shared<Engine>();
  engine_open(*e, model_path, ENGINE_FP16);
  return e;
}

/**
 * List an engine's bindings for an error message.
 *
 * A policy asking for a tensor the network does not have is nearly always a
 * naming mismatch, and the fix is obvious the moment the real names are in
 * front of the reader.
 *
 * @param[in] tensors  the bindings to list
 * @returns "name[count], name[count], ..."
 * @exceptsafe basic
 */
std::string engine_tensor_list(const std::vector<Tensor>& tensors) {
  std::string list;
  for (const Tensor& t : tensors) {
    if (!list.empty()) list += ", ";
    list += t.name + "[" + std::to_string(t.count) + "]";
  }
  return list;
}

/**
 * Find an input binding by name.
 *
 * @param[in,out] e  the loaded engine
 * @param[in] name   binding name
 * @returns the binding, whose host buffer is written before a run
 * @throws std::runtime_error if there is no such input, listing those there
 *         are
 * @exceptsafe strong
 */
Tensor& engine_input(
    Engine& e,
    const std::string& name
) {
  for (Tensor& t : e.inputs) {
    if (t.name == name) return t;
  }
  throw std::runtime_error(
      "engine: no input '" + name + "'; model has " +
      engine_tensor_list(e.inputs)
  );
}

/**
 * Find an output binding by name.
 *
 * @param[in,out] e  the loaded engine
 * @param[in] name   binding name
 * @returns the binding, whose host buffer is read after a run
 * @throws std::runtime_error if there is no such output, listing those
 *         there are
 * @exceptsafe strong
 */
Tensor& engine_output(
    Engine& e,
    const std::string& name
) {
  for (Tensor& t : e.outputs) {
    if (t.name == name) return t;
  }
  throw std::runtime_error(
      "engine: no output '" + name + "'; model has " +
      engine_tensor_list(e.outputs)
  );
}

/**
 * Check a binding is the size a policy believes it is.
 *
 * Called from `init`, so a policy whose observation layout has drifted from
 * its checkpoint fails before the run instead of walking a whole tour on
 * misaligned numbers and reporting a plausible-looking fall.
 *
 * @param[in,out] e    the loaded engine
 * @param[in] name     binding name
 * @param[in] count    elements the policy expects
 * @param[in] is_input whether to look among inputs or outputs
 * @param[in] policy   policy name, for the message
 * @throws std::runtime_error if the binding is missing or the wrong size
 * @exceptsafe strong
 */
void engine_expect(
    Engine& e,
    const std::string& name,
    size_t count,
    bool is_input,
    const char* policy
) {
  const Tensor& t = is_input ? engine_input(e, name) : engine_output(e, name);
  if (t.count != count) {
    throw std::runtime_error(
        std::string(policy) + ": tensor '" + name + "' has " +
        std::to_string(t.count) + " elements, expected " + std::to_string(count)
    );
  }
}

/**
 * Write an Eigen quaternion back in MuJoCo's layout.
 *
 * @param[in] q  the rotation
 * @returns w, x, y, z
 * @exceptsafe no-throw
 */
inline Eigen::Vector4d quat_from_eigen(const Eigen::Quaterniond& q) {
  return Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
}

/**
 * Conjugate a quaternion.
 *
 * How a policy takes a world-frame quantity into the body frame: the
 * conjugate of the pelvis attitude is the rotation from world to body.
 *
 * @param[in] quat  quaternion as w, x, y, z
 * @returns the conjugate, in the same layout
 * @exceptsafe no-throw
 */
inline Eigen::Vector4d quat_conj(const Eigen::Vector4d& quat) {
  return quat_from_eigen(quat_to_eigen(quat).conjugate());
}

/**
 * Run the network once, host buffers in and host buffers out.
 *
 * Copies, execution and copies back are all queued on one stream and waited
 * on at the end, so what the caller times is the whole round trip rather
 * than the enqueue. That is the honest figure for a control loop that
 * cannot proceed without the answer.
 *
 * @param[in,out] e  the loaded engine; its output buffers are filled
 * @throws std::runtime_error if the engine is not ready, if a copy fails,
 *         or if execution fails
 * @exceptsafe basic
 */
void engine_run(Engine& e) {
  if (!e.context) throw std::runtime_error("engine: session not ready");
  for (size_t i = 0; i < e.inputs.size(); ++i) {
    cuda_check(
        cudaMemcpyAsync(
            e.device_in[i],
            e.inputs[i].data.data(),
            e.inputs[i].count * sizeof(float),
            cudaMemcpyHostToDevice,
            e.stream.get()
        ),
        "cudaMemcpyAsync (in)"
    );
  }
  if (!e.context->enqueueV3(e.stream.get())) {
    throw std::runtime_error("engine: enqueue failed");
  }
  for (size_t i = 0; i < e.outputs.size(); ++i) {
    cuda_check(
        cudaMemcpyAsync(
            e.outputs[i].data.data(),
            e.device_out[i],
            e.outputs[i].count * sizeof(float),
            cudaMemcpyDeviceToHost,
            e.stream.get()
        ),
        "cudaMemcpyAsync (out)"
    );
  }
  cuda_check(cudaStreamSynchronize(e.stream.get()), "cudaStreamSynchronize");
}

/**
 * Run a one-in, one-out network on one observation vector.
 *
 * Most of the field is exactly this shape, so the common case is written
 * once. A network with more bindings is refused rather than guessed at.
 *
 * @param[in,out] e  the loaded engine
 * @param[in] obs    observation, in the network's own layout
 * @param[in] count  its length, which must match the binding
 * @returns the output buffer, valid until the next run
 * @throws std::runtime_error if the network is not 1-in 1-out, if the
 *         observation is the wrong size, or if execution fails
 * @exceptsafe basic
 */
const std::vector<float>& engine_run_single(
    Engine& e,
    const float* obs,
    size_t count
) {
  if (e.inputs.size() != 1 || e.outputs.size() != 1) {
    throw std::runtime_error("engine: run_single: model is not 1-in 1-out");
  }
  if (count != e.inputs[0].count) {
    throw std::runtime_error("engine: run_single: bad obs size");
  }
  std::copy(obs, obs + count, e.inputs[0].data.begin());
  engine_run(e);
  return e.outputs[0].data;
}

#include "policies/asap/policy.cpp"

#include "policies/clobot/policy.cpp"

#include "policies/falcon/policy.cpp"

#include "policies/gr00t_wbc/policy.cpp"

#include "policies/holosoma/policy.cpp"

#include "policies/homie/policy.cpp"

#include "policies/openwbt/policy.cpp"

#include "policies/rl_lab/policy.cpp"

#include "policies/rl_mjlab/policy.cpp"

#include "policies/run_residual/policy.cpp"

#include "policies/amo/policy.cpp"

#include "policies/rl_gym/policy.cpp"

#include "policies/robomimic/policy.cpp"

/**
 * Every policy this binary was built with.
 *
 * Each name is a directory under `policies/` holding one `policy.cpp` and
 * its weights, and each of those sources is included above rather than
 * compiled separately: a policy is written against the harness types that
 * precede it and lives in its own namespace, so one translation unit costs
 * nothing and keeps the whole benchmark a single file to read.
 *
 * Adding a candidate means an include, an entry here and a branch in
 * `make_policy`.
 */
const char* const POLICY_NAMES[] = {
    "asap",
    "clobot",
    "clobot_with_arms",
    "falcon",
    "gr00t_wbc",
    "holosoma",
    "homie",
    "openwbt",
    "rl_lab",
    "rl_mjlab",
    "run_residual",
    "amo",
    "rl_gym",
    "robomimic"
};

/**
 * Every policy name the binary knows, sorted.
 *
 * @returns the names in alphabetical order
 * @exceptsafe basic
 */
std::vector<std::string> policy_names() {
  std::vector<std::string> names(
      std::begin(POLICY_NAMES),
      std::end(POLICY_NAMES)
  );
  std::sort(names.begin(), names.end());
  return names;
}

/**
 * Construct a policy by name.
 *
 * A misspelled name lists the whole field rather than simply refusing, so
 * that the answer to the mistake is in the same message as the complaint.
 *
 * @param[in] name  policy name, as `--policy` gives it
 * @returns the constructed policy
 * @throws std::runtime_error if no policy goes by that name
 * @exceptsafe strong
 */
std::shared_ptr<ModelPolicy> make_policy(const char* name) {
  const std::string wanted = name == nullptr ? "" : name;
  if (wanted == "asap") return std::make_shared<asap::Policy>();
  if (wanted == "clobot") return std::make_shared<clobot::Policy>();
  if (wanted == "clobot_with_arms")
    return std::make_shared<clobot::WithArmsPolicy>();
  if (wanted == "falcon") return std::make_shared<falcon::Policy>();
  if (wanted == "gr00t_wbc") return std::make_shared<gr00t_wbc::Policy>();
  if (wanted == "holosoma") return std::make_shared<holosoma::Policy>();
  if (wanted == "homie") return std::make_shared<homie::Policy>();
  if (wanted == "openwbt") return std::make_shared<openwbt::Policy>();
  if (wanted == "rl_lab") return std::make_shared<rl_lab::Policy>();
  if (wanted == "rl_mjlab") return std::make_shared<rl_mjlab::Policy>();
  if (wanted == "run_residual") return std::make_shared<run_residual::Policy>();
  if (wanted == "amo") return std::make_shared<amo::Policy>();
  if (wanted == "rl_gym") return std::make_shared<rl_gym::Policy>();
  if (wanted == "robomimic") return std::make_shared<robomimic::Policy>();

  std::string msg = "unknown --policy '" + wanted + "'. Valid names:";
  bool first = true;
  for (const std::string& n : policy_names()) {
    msg += (first ? " " : ", ") + n;
    first = false;
  }
  throw std::runtime_error(msg);
}

/**
 * Appends one row per finished run to a CSV file.
 *
 * A row is written and flushed as each run ends rather than at the close of
 * the campaign, so a sweep that is interrupted after four hours keeps every
 * run it had already finished. The header is written only when the file
 * starts empty, which makes resuming a campaign a matter of pointing
 * `--csv` at the same file again.
 *
 * One row is one run and nothing else, so the file can be read straight
 * into a frame and grouped by policy without first being filtered by row
 * kind. `outcome` gives the completion count and its interval, `survival_s`
 * the mean, median and worst, and `pos_err_cm` and `yaw_err_deg` the error
 * columns, poolable across runs because `targets` travels beside them as
 * the weight. The step timings a run measures are deliberately absent from
 * the published table, so they are absent from here too, and `seed` is kept
 * only so a campaign can be checked for gaps and resumed.
 *
 * What each leg of the tour cost follows as `s<i>_` columns, one block per
 * target clock: the work each joint group did and how much it shook while
 * the robot walked that leg. Energy is written in joules and vibration in
 * thousands of rad/s^2, both to one decimal place, which is all either is
 * worth: a group's figures span four orders of magnitude across the field,
 * and the raw jerk sums run to six digits before the decimal point means
 * anything. A run that fell leaves the legs it never reached empty.
 *
 * Nothing says here which legs ran their full clock, because nothing needs
 * to. Legs `0` to `targets - 1` each had their whole five seconds; a block
 * present at index `targets` is the leg the run died in, and it ran from
 * that leg's start to `survival_s`. Writing a flag and two timestamps
 * twelve times over would be a hundred and forty-four columns of arithmetic
 * the reader can already do.
 *
 * A file that already exists must have been written by this same header or
 * the append is refused: a campaign resumed after the scene or the metrics
 * changed would otherwise interleave two schemas in one file and quietly
 * ruin both halves.
 */
class CsvLog {
 public:
  /**
   * Open a campaign file, writing or checking its header.
   *
   * @param[in] path  file to append to, created if absent
   * @throws std::runtime_error if the file cannot be opened, or if it
   *         already holds a header this build would not have written
   * @exceptsafe basic
   */
  explicit CsvLog(const std::string& path) : header_(header_for()) {
    const std::filesystem::path parent =
        std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(parent, ec);
      if (ec) {
        throw std::runtime_error("csv: cannot create " + parent.string());
      }
    }
    {
      std::ifstream probe(path);
      std::string first;
      if (probe && std::getline(probe, first)) {
        if (!first.empty() && first.back() == '\r') first.pop_back();
        if (first != header_) {
          throw std::runtime_error(
              "csv: " + path +
              " was written with different columns; point --csv at a new file"
          );
        }
        existing_ = true;
      }
    }
    out_.open(path, std::ios::app);
    if (!out_) throw std::runtime_error("csv: cannot open " + path);
    if (!existing_) {
      out_ << header_ << '\n';
      out_.flush();
    }
  }

  /**
   * Record one finished run: its summary row, then one row per leg.
   *
   * The error columns are written whatever the outcome, because the table
   * takes them over every target actually scored; a reader who wants only
   * completed tours has the outcome column to filter on. All of a run's
   * rows are appended under one lock so that a sweep with sixteen tours in
   * flight cannot interleave them.
   *
   * @param[in] seed    the seed this run walked
   * @param[in] report  the finished run
   * @exceptsafe basic
   */
  void row(
      uint32_t seed,
      const Report& report
  ) {
    std::ostringstream line;
    line << std::fixed << std::setprecision(1);
    line << report.policy << ',' << seed << ',' << outcome_name(report.outcome)
         << ',' << report.duration_s << ',' << report.targets.size() << ','
         << report.pos_err_cm << ',' << report.yaw_err_deg;

    for (int i = 0; i < WAYPOINTS; ++i) {
      const SegmentCost* leg = nullptr;
      for (const SegmentCost& s : report.segments) {
        if (s.index == i) {
          leg = &s;
          break;
        }
      }
      if (leg == nullptr) {
        line << std::string(2 * NUM_GROUPS, ',');
        continue;
      }
      for (const double e : leg->group_energy_j) line << ',' << e;
      for (const double v : leg->group_vibration) line << ',' << v / 1000.0;
    }
    line << '\n';

    const std::lock_guard<std::mutex> lock(mutex_);
    out_ << line.str();
    out_.flush();
  }

 private:
  /**
   * Build the one header both kinds of row are written against.
   *
   * @returns the header line, without its newline
   * @exceptsafe basic
   */
  static std::string header_for() {
    std::string header =
        "policy,seed,outcome,survival_s,targets,pos_err_cm,yaw_err_deg";
    for (int i = 0; i < WAYPOINTS; ++i) {
      const std::string leg = "s" + std::to_string(i) + "_";
      for (const char* group : GROUP_NAMES) {
        header += "," + leg + "e_" + std::string(group) + "_j";
      }
      for (const char* group : GROUP_NAMES) {
        header += "," + leg + "v_" + std::string(group) + "_krads2";
      }
    }
    return header;
  }

  std::mutex mutex_;       ///< serialises the appends
  std::ofstream out_;      ///< the file, opened for append
  std::string header_;     ///< what this build writes and demands
  bool existing_ = false;  ///< true if the file already had a header
};

/**
 * Run every policy over a campaign of seeds and record each run.
 *
 * One policy at a time, each over the whole seed range with `--parallel`
 * tours in flight. Taking the policies in turn rather than interleaving
 * them is what bounds the memory: the process holds as many instances of
 * one candidate as there are workers, instead of the whole field
 * multiplied by the parallelism.
 *
 * Each run gets a policy of its own. Policies carry state -- history
 * buffers, recurrent hidden state -- and in the one-process-per-seed
 * arrangement this replaces, an instance walked exactly one tour; reusing
 * one across seeds would let a fallen run bias the next and would cost the
 * campaign its determinism.
 *
 * A policy that cannot load is recorded as an error on every seed rather
 * than skipped, so the denominator of its row stays the size of the
 * campaign and a missing checkpoint cannot read as a good score.
 *
 * @param[in] args  the parsed command line, with a sweep asked for
 * @returns 0 if the campaign ran to the end, 1 if it was interrupted
 * @throws std::runtime_error if nothing is installed or the scene will not
 *         load
 * @exceptsafe basic
 */
int sweep_run(const Args& args) {
  const std::string& name = args.policy;
  make_policy(name.c_str());

  const std::unique_ptr<mjModel, void (*)(mjModel*)> scene =
      scene_load(args.sim.model_path);

  std::unique_ptr<CsvLog> csv;
  if (!args.sweep.csv.empty()) {
    csv = std::make_unique<CsvLog>(args.sweep.csv);
  }

  const size_t seeds =
      static_cast<size_t>(args.sweep.last - args.sweep.first) + 1;
  const int workers =
      static_cast<int>(std::min<size_t>(args.sweep.parallel, seeds));

  std::cout << "sweep: " << name << " x " << seeds << " seeds ("
            << args.sweep.first << "-" << args.sweep.last << "), " << workers
            << " in flight" << std::endl;

  std::atomic<size_t> finished{0};
  const size_t total = seeds;

  {
    std::atomic<size_t> next{0};
    std::atomic<size_t> complete{0};
    std::mutex report_mutex;
    double survived = 0.0;

    const auto worker = [&](int index) {
      pin_worker_to_core(index);
      while (!g_stop.load()) {
        const size_t slot = next.fetch_add(1);
        if (slot >= seeds) break;
        const uint32_t seed = args.sweep.first + static_cast<uint32_t>(slot);

        SimConfig sim = args.sim;
        sim.seed = seed;
        sim.quiet = true;
        TourConfig scenario = args.scenario;
        scenario.seed = seed;

        Report report;
        report.policy = name;
        try {
          const std::shared_ptr<ModelPolicy> policy = make_policy(name.c_str());
          policy->init();
          report = sim_run(sim, policy, tour_make(scenario), scene.get());
        } catch (const std::exception& e) {
          report = Report();
          report.policy = name;
          report.outcome = Outcome::ERROR;
          report.detail = e.what();
        }

        if (report.outcome == Outcome::COMPLETE) complete.fetch_add(1);
        if (csv) csv->row(seed, report);

        const size_t done = finished.fetch_add(1) + 1;
        const std::lock_guard<std::mutex> lock(report_mutex);
        survived += report.duration_s;
        std::cout << "[" << std::setw(6) << done << "/" << total << "] "
                  << std::left << std::setw(14) << name << std::right
                  << " seed " << std::setw(5) << seed << "  " << std::setw(8)
                  << outcome_name(report.outcome) << std::fixed
                  << std::setprecision(1) << std::setw(8) << report.duration_s
                  << " s" << std::endl;
      }
    };

    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(workers));
    for (int w = 0; w < workers; ++w) threads.emplace_back(worker, w);
    for (std::thread& t : threads) t.join();

    const size_t ran = std::min<size_t>(next.load(), seeds);
    std::cout << name << ": " << complete.load() << "/" << ran
              << " complete, mean survival " << std::fixed
              << std::setprecision(1)
              << (ran > 0 ? survived / static_cast<double>(ran) : 0.0) << " s"
              << std::endl;
  }

  if (g_stop.load()) {
    std::cout << "sweep: interrupted after " << finished.load() << " runs"
              << std::endl;
    return 1;
  }
  std::cout << "sweep: " << finished.load() << " runs recorded" << std::endl;
  return 0;
}

/**
 * Run the requested policies over one tour and print the table.
 *
 * One invocation measures the one policy `--policy` names. With `--runs`
 * or `--seeds` this hands over to `sweep_run`, which walks a whole campaign
 * of seeds instead of a single tour.
 *
 * The exit status is non-zero unless the policy completed its tour, which
 * is what makes this usable from a script that runs the field one
 * candidate at a time.
 *
 * @param[in] argc  argument count
 * @param[in] argv  arguments as given to `main`
 * @returns 0 if every policy completed, 1 if any did not, 2 if the run
 *          could not be set up at all
 * @exceptsafe no-throw
 */
int run(
    int argc,
    char** argv
) {
  std::signal(SIGINT, on_sigint);
  pin_to_one_core();
  try {
    const Args args = parse(argc, argv);

    if (args.sweep.enabled) return sweep_run(args);

    const std::unique_ptr<mjModel, void (*)(mjModel*)> scene =
        scene_load(args.sim.model_path);
    const Tour tour = tour_make(args.scenario);

    std::cout << "=== " << args.policy << " ===" << std::endl;
    Report report;
    report.policy = args.policy;
    try {
      const std::shared_ptr<ModelPolicy> policy =
          make_policy(args.policy.c_str());
      policy->init();
      report = sim_run(args.sim, policy, tour, scene.get());
    } catch (const std::exception& e) {
      report.outcome = Outcome::ERROR;
      report.detail = e.what();
      std::cout << "ERROR: " << e.what() << std::endl;
    }
    if (!report.targets.empty()) report_print(report);

    if (!args.sweep.csv.empty()) {
      CsvLog(args.sweep.csv).row(args.scenario.seed, report);
      std::cout << "recorded to " << args.sweep.csv << std::endl;
    }

    std::cout << std::endl
              << std::left << std::setw(15) << "policy" << std::right
              << std::setw(12) << "walk pos" << std::setw(12) << "walk yaw"
              << std::setw(14) << "targets" << std::setw(10) << "step max"
              << std::endl
              << report_line(report) << std::endl;

    return report.outcome == Outcome::COMPLETE ? 0 : 1;
  } catch (const std::exception& e) {
    std::cout << "ERROR: " << e.what() << std::endl;
    return 2;
  }
}

}

/**
 * Entry point.
 *
 * Everything, including the argument parsing and the error reporting, is in
 * `benchmark::run`, so that the namespace holding the harness also holds
 * its top level and nothing outside it has to be reasoned about.
 *
 * @param[in] argc  argument count
 * @param[in] argv  arguments
 * @returns what `benchmark::run` decided
 * @exceptsafe no-throw
 */
int main(
    int argc,
    char** argv
) {
  return benchmark::run(argc, argv);
}
