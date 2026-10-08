"""Hand-written content for the project report (docs/build_report.py).

FILES: one entry per source file: a title, what the file is for, how it works,
and important notes. DESC: descriptions of classes, functions and methods that
have no docstring (key: "<path relative to src/slingpuck>::<qualified name>").
Functions with a docstring use its first sentence unless DESC overrides it.
"""

# --------------------------------------------------------------------------- files
FILES = {
    "config.py": {
        "title": "Configuration loading, parameter versions, randomization",
        "purpose": [
            "Every other module reads its numbers from one merged configuration dictionary built here. "
            "load_config() merges three YAML files from configs/: servo.yaml, then physics_params_v<N>.yaml, "
            "then env.yaml. A later file overrides an earlier one key by key.",
            "Physics parameters are versioned. sysid writes physics_params_v1.yaml, v2, ...; 'latest' loads the "
            "highest version, and the loader checks that the version inside the file matches its name.",
            "Every value that is not measured is listed under 'placeholders:' in its file. strict=True refuses to "
            "load while any placeholder or null value remains, so no result can be reported against real data by "
            "accident with guessed numbers.",
            "cfg['meta'] records the params version, the placeholder list, the git hash and whether the working "
            "tree had uncommitted changes. save_run_config() writes all of this, plus the seed, next to every "
            "training run.",
        ],
        "notes": ["sample_randomized() is the domain randomization: it returns a changed copy and never changes "
                  "the input, and it records the sampled values in meta['randomized']."],
    },
    "kinematics.py": {
        "title": "Goalkeeper kinematics shared by the sim and the robot",
        "purpose": [
            "GoalkeeperGeometry is the only mapping from a policy action in [-1, 1] to a board-frame pan angle and "
            "then to the SO-101 shoulder_pan joint command. The env and the LeRobot interface both use it, so the "
            "sim and the robot cannot disagree (the original code mapped action 1.0 to 0.19 rad in the sim and to "
            "0.5 rad on the robot).",
            "It also computes the paddle pose on its arc (center, angle, velocity, angular velocity) for the "
            "physics sim. pan = 0 puts the paddle center on the gate line; positive pan moves it toward +x.",
        ],
        "notes": ["from_config() checks that the base is behind the paddle line and that the paddle stays inside "
                  "the board at full pan; it raises an error otherwise."],
    },
    "physics/backend.py": {
        "title": "Physics backend interface",
        "purpose": [
            "Defines the PuckPhysicsBackend abstract class (reset, get_state, set_paddle, step, kinetic_energy) "
            "and the small data classes it uses: PaddleState (a kinematic box), GateCrossing and StepEvents.",
            "The envs talk only to this interface, so the fast 2D sim and the MuJoCo sim can replace each other "
            "without env changes (the plan asked for a swappable backend).",
            "Board frame: origin at the board center, x across the width, y along the length; the agent half is "
            "y < 0, the opponent half y > 0, and the divider with the gate is at y = 0.",
        ],
    },
    "physics/puck_dynamics.py": {
        "title": "Fast2DPuckSim: the fast planar puck sim",
        "purpose": [
            "Planar pucks with Coulomb sliding friction (constant deceleration mu*g, stop without reversal), side "
            "and end walls, the center divider with the gate gap, puck-puck contacts, and an optional kinematic "
            "paddle (an oriented box with linear and angular velocity).",
            "step(dt) returns StepEvents: the gate crossings (with direction) and the paddle contacts.",
        ],
        "how": [
            "Contacts reflect the normal velocity relative to the surface, scaled by the restitution, and keep "
            "the tangential velocity. Without a moving paddle the kinetic energy can never increase (tested).",
            "The per-puck contact code uses Python floats, not NumPy, because NumPy call overhead is larger than "
            "the arithmetic for 1-2 values. It is 5 times faster than the first NumPy version and gives the same "
            "events (max state difference 7e-13 over 200 random trials).",
        ],
    },
    "physics/servo_model.py": {
        "title": "Servo model (Feetech STS3215)",
        "purpose": [
            "Each step: command latency (exactly round(latency/dt) steps) -> deadband (errors smaller than the "
            "deadband are ignored) -> first-order lag (exact discretization of tau*dx/dt = target - x) -> rate "
            "limit (default 300 deg/s). Works for one joint or an array of joints.",
        ],
        "notes": ["The MuJoCo sim-to-sim test showed that this first-order model is much faster at the start of a "
                  "move than the torque-limited MuJoCo arm. The real step response must decide which is right."],
    },
    "physics/band_model.py": {
        "title": "Sling band: pre-tension, nonlinear tension, closed hysteresis loop, release",
        "purpose": [
            "The band runs between two pegs at (+-span/2, band_y). The arm pulls the puck back to point P; the "
            "band makes a V and pushes P forward with force T*(u_left + u_right).",
            "The band is pre-stretched (natural length < peg span). The pre-tension gives most of the launch "
            "energy: without it, a 3 cm pull would launch a 27 g puck at about 0.4 m/s, too slow to reach the gate.",
        ],
        "how": [
            "Loading tension T_l(s) = k*s*(s/1 cm)^(exponent-1), s = band length - natural length.",
            "Unloading T_u(s) = T_l(s) - h*4x(1-x), x = (s-s0)/(s_max-s0). The bump is zero at both ends, so the "
            "loop closes, and h is chosen so the cycle loses exactly hysteresis_loss_factor of the loading work.",
            "release() integrates the puck from rest under the unloading force and friction until it crosses the "
            "band line, then scales the speed by sqrt(energy_transfer). A pull too small to beat friction leaves "
            "the puck behind the band (launched = False).",
        ],
        "notes": ["Main finding: the pull is small compared with the band span, so u_left + u_right points almost "
                  "straight ahead for any pull angle. A 17 deg pull steers the puck by less than 1 deg. This is why "
                  "Phase 2 aims by sliding the puck along the band (placement)."],
    },
    "physics/arm_deflection.py": {
        "title": "Arm deflection under band load",
        "purpose": [
            "The arm gives way a little under the band force: actual pull = commanded pull - compliance * F_par, "
            "where F_par is the band force component that resists the pull. solve_pull() finds the actual pull by "
            "bisection, which always converges because d + c*F_par(d) increases with d.",
            "pull_direction(angle) gives the unit pull vector (angle 0 = straight back, away from the gate).",
        ],
        "notes": ["Only deflection along the pull direction is modeled. A bug (compliance 0 returned half the pull) "
                  "was found by a test and fixed."],
    },
    "physics/mujoco_goalkeeper.py": {
        "title": "MuJoCo physics for the goalkeeper (board, puck, SO-101 arm)",
        "purpose": [
            "Builds a MuJoCo scene in code (MjSpec) from the merged config: the real-size board, walls, divider "
            "and gate, the puck, and the RobotStudio SO-101 model (STS3215 actuators, kp 998, +-2.94 N m) holding "
            "a paddle in front of the gate. It implements PuckPhysicsBackend, so the Phase 1 env runs on it.",
        ],
        "how": [
            "Arm placement: the base is turned 90 deg so the arm reaches toward the gate, and placed so that the "
            "pan axis is exactly at robot.base_xy_m.",
            "Arm pose: least-squares inverse kinematics puts the gripper fingertips at the paddle center; the "
            "paddle is a box fixed to the gripper (a custom mount) and absorbs a 1 mm offset of the pitch-joint "
            "plane. The hold targets are corrected for gravity sag. The paddle is on the 2D arc within 0.2 mm.",
            "Puck: planar (x, y, spin) with an applied Coulomb friction force. A free puck tumbled off the board, "
            "and MuJoCo contact friction made it hop and lose friction after wall hits.",
            "Contacts: only explicit pairs collide, frictionless like the 2D sim. They use direct stiffness/damping; "
            "the damping is calibrated in the real scene (24 damping values x 5 impact phases) so the measured "
            "restitution matches the config. Paddle restitution is therefore the effective value on the compliant arm.",
            "Timestep 0.5 ms; about 180 episodes per second.",
        ],
    },
    "sensing/camera_model.py": {
        "title": "Overhead camera model",
        "purpose": [
            "60 Hz frames with configurable latency, Gaussian position noise and random dropouts, using a seeded "
            "random generator. Each Frame carries its capture time and its arrival time. observe(t, true_xy) "
            "returns only the frames that arrive at this step, so a frame is never delivered twice (the original "
            "code fed the same stale frame to the tracker again and again).",
        ],
    },
    "sensing/kalman_tracker.py": {
        "title": "Constant-velocity Kalman tracker with latency compensation",
        "purpose": [
            "State [x, y, vx, vy], white-noise-acceleration process model, measurement noise from the camera "
            "config. It starts from the first frame. update() filters each frame at its capture time; estimate(t) "
            "predicts to the current time, which removes the camera latency (error < 25% of the raw frame in tests).",
        ],
        "notes": ["Limits: it does not model wall bounces or a sudden launch from rest; 80 ms after a launch the "
                  "estimate is about 10 cm behind. For a still puck the Phase 2 env uses the plain frame mean instead."],
    },
    "envs/goalkeeper_env.py": {
        "title": "Phase 1 env: goalkeeper (recovery task)",
        "purpose": [
            "The opponent's puck rests at the opponent band line, then launches toward the gate. The policy "
            "commands the pan joint; the paddle swings on its arc. The policy sees only the Kalman-tracked puck, "
            "the pan encoder and its previous action (8 values). The true state is in info['privileged'] "
            "(16 values) for an asymmetric critic.",
            "Recovery task (user decision): the gate is only ~1.5 mm wider than the puck, so a settled paddle "
            "blocks everything. The paddle is therefore locked at a random offset until the launch (plus "
            "release_delay_s), as if the arm were busy with a sling. The result is a time-to-cover measurement.",
            "Reward: +1 save, -1 goal, 0 no threat, minus 0.01*|change of action|. With threats_only, shots are "
            "resampled until they would score with no paddle.",
        ],
        "how": [
            "Three physics hooks (_build_physics, _reset_physics, _physics_step) are the only place that touches "
            "the physics; MujocoGoalkeeperEnv replaces just these three.",
            "Domain randomization is sampled again at every reset. substep_hook lets the viewers record frames.",
        ],
    },
    "envs/mujoco_goalkeeper_env.py": {
        "title": "Phase 1 env on MuJoCo physics",
        "purpose": [
            "Same task, observation, action, reward and sensing as GoalkeeperEnv; only the three physics hooks "
            "change. A policy trained on the 2D sim runs here unchanged, which makes this a sim-to-sim test.",
        ],
        "notes": ["Pan dynamics come from the MuJoCo actuator, not ServoModel (only the command latency is shared). "
                  "The threat check at reset still uses the 2D sim."],
    },
    "envs/sling_env.py": {
        "title": "Phase 2 env: targeted shot through the gate",
        "purpose": [
            "Step 1 (aim): the policy sees the tracked puck at rest and chooses (slide x, pull angle, pull "
            "distance). The arm slides the puck along the band (placement noise), pulls it (angle and pull noise) "
            "and the arm deflects. Step 2 (correct): the policy sees the pulled puck and the band depth from the "
            "camera and applies a small correction; then the band releases and the puck flies on Fast2DPuckSim.",
            "Reward: +1 for crossing the gate, plus 0.3*exp(-(miss/1 cm)^2), where miss is |x| at the divider.",
        ],
        "how": [
            "realize_pull() and launch_and_fly() are module-level functions, shared by the env, the scripted "
            "planner, the synthetic sysid logs and the parity eval.",
            "_observe_still() uses the mean of 0.3 s of camera frames: for a still puck this gives ~0.7 mm error, "
            "while the constant-velocity Kalman filter gave ~2 mm and made the correction harmful.",
        ],
    },
    "envs/match_env.py": {
        "title": "Phase 3 env: event-level match against the scripted opponent",
        "purpose": [
            "The high-level policy picks one of three frozen primitives every 0.1 s: 0 block, 1 sling, 2 hold. "
            "Arm states: gate -> to_band -> slinging -> at_band -> to_gate. The gate is undefended from the sling "
            "command until the arm is back and settled (the explicit recovery time from the plan).",
            "Nothing is simulated at 1 kHz: opponent shots fly straight with friction (analytic arrival time); a "
            "threat is saved with the probability from the measured block table; a sling succeeds with the Phase "
            "2 probability. About 70,000 decisions per second.",
            "Observation (20): tracked incoming puck (after the camera latency), opponent hand, arm state, puck "
            "counts, time. Privileged (11): opponent phase, time to release, next shot, incoming threat.",
            "Reward and win rule come from 'scoring' (placeholder: +1 sent, -1 received, first empty side wins).",
        ],
    },
    "envs/wrappers.py": {
        "title": "Privileged observation wrapper",
        "purpose": ["Turns an env's Box observation into a Dict {'policy': ..., 'privileged': ...}, taking the "
                    "privileged vector from info. Used with the asymmetric actor-critic policy."],
    },
    "opponent/scripted_opponent.py": {
        "title": "Scripted human opponent (Phase 3)",
        "purpose": [
            "Cycle: reload (the hand fetches a puck) -> place (on the band) -> pull (pull ~ shot speed) -> hold "
            "(random pause) -> release, about 1.2 s per shot. Each cycle samples its timings, the launch x, the "
            "speed and whether the shot is a threat (launch |x| < 3 mm).",
            "The hand covers the loaded puck from the camera, so only the hand is observed (position and velocity "
            "along straight segments, plus noise).",
            "tell_strength: during place, pull and hold the hand x is tell*x_launch + (1-tell)*x_decoy, with the "
            "decoy from the same distribution, so 0 = no information and 1 = exact (tested).",
        ],
    },
    "primitives/block.py": {
        "title": "Frozen block primitive: save-probability table",
        "purpose": [
            "Measures, once, on the Phase 1 env with a frozen block policy: p_gate(speed) for a paddle settled at "
            "the gate when the shot launches, and p_edge(speed, delay) for a paddle that is just back from the band "
            "at the edge of its range and free 'delay' seconds after the launch. Cached in cache/ under a hash of "
            "everything that changes it (about 40 s to build).",
            "The table shows the vulnerability window: from the edge the paddle saves 100% of 1 m/s shots, 60% "
            "at 1.6 m/s and almost none above 2 m/s; a settled paddle saves 100% at every speed.",
        ],
    },
    "primitives/sling.py": {
        "title": "Frozen sling primitive: timings and success probability",
        "purpose": ["Holds the sling timing (move to the band, fetch a puck, sling core, flight) and the Phase 2 "
                    "success probability, from the config."],
    },
    "policies/asymmetric.py": {
        "title": "Asymmetric actor-critic for Stable-Baselines3 PPO",
        "purpose": [
            "The actor reads only obs['policy'] (deployable sensors); the critic reads obs['policy'] and "
            "obs['privileged'] (sim truth and randomized physics). Only the actor is deployed. A test checks that "
            "changing the privileged input does not change the actor output and does change the value.",
        ],
        "how": ["SB3 builds the actor feature extractor first and the critic extractor second; "
                "make_features_extractor() uses this order. AsymmetricMlpExtractor lets the actor and the critic "
                "networks have different input sizes."],
    },
    "policies/scripted.py": {
        "title": "Scripted goalkeeper baselines (Phase 1)",
        "purpose": ["CenterBlocker always commands pan = 0; HoldStart never moves. Both have the SB3 predict() "
                    "signature, so the eval code treats them like trained models."],
    },
    "policies/sling_scripted.py": {
        "title": "Scripted sling baselines (Phase 2)",
        "purpose": ["SlideCenter slides the puck to x = 0 and pulls straight back. SlideCenterCorrect also uses the "
                    "tracked pulled puck to shift it back to x = 0 in the correction step."],
    },
    "policies/match_scripted.py": {
        "title": "Scripted match selectors (Phase 3)",
        "purpose": [
            "AlwaysBlock, GreedySling (sling whenever possible, never return), ReloadRule (the plan's baseline: "
            "sling only when the opponent reloads, detected from the hand; otherwise go back to the gate), TellRule "
            "(also slings while the opponent aims if the hand x shows a likely miss) and OracleReload (ReloadRule "
            "with the true opponent phase).",
        ],
    },
    "train/common.py": {
        "title": "Shared training code",
        "purpose": [
            "Env factories (goalkeeper with a '2d' or 'mujoco' backend, sling), picklable factories for "
            "SubprocVecEnv, load_trained_run() (loads a model with either import path), the outcome-rate "
            "TensorBoard callback, and train_ppo()/train_cli(): one reproducible PPO run per seed with its run "
            "folder (config, params version, git hash, seed, checkpoints, final model).",
        ],
    },
    "train/train_goalkeeper.py": {"title": "Phase 1 training entry point",
                                  "purpose": ["Calls train_cli for the goalkeeper (configs/train_goalkeeper.yaml)."]},
    "train/train_sling.py": {"title": "Phase 2 training entry point",
                             "purpose": ["Calls train_cli for the sling shot (configs/train_sling.yaml)."]},
    "eval/stats.py": {"title": "Confidence intervals",
                      "purpose": ["wilson_ci() for proportions and t_ci() for means across runs (training seeds)."]},
    "eval/save_rate_vs_speed.py": {
        "title": "Phase 1 eval: save rate vs puck speed",
        "purpose": [
            "Runs trained goalkeeper runs and the scripted baselines on the same shots (paired seeds), with "
            "stratified speed bins, and reports the save rate vs speed, vs paddle start offset, and the speed x "
            "offset grid (the time-to-cover table), with 95% CIs. --backend mujoco runs the same eval on MuJoCo.",
            "PPO CI = union of a t interval across training seeds and a Wilson interval over one seed's shots "
            "(all seeds learned the same saturated policy, so the t interval alone had zero width).",
        ],
    },
    "eval/hole_rate.py": {
        "title": "Phase 2 eval: gate success rate and sim-vs-real parity",
        "purpose": [
            "Policy mode: success rate vs shot count and by puck start position, for trained runs and baselines.",
            "Parity mode: replays every logged shot many times in the sim and compares (a) success (Brier score, "
            "reliability diagram) and (b) the transit time from 2 cm past the band line to y = -4 cm. Success "
            "depends mostly on the lateral aim and cannot see a speed error; the transit time can.",
        ],
    },
    "eval/match_eval.py": {
        "title": "Phase 3 eval: match results",
        "purpose": ["Plays every policy on the same matches and reports win, loss and time-out rates, the mean puck "
                    "difference, slings, pucks conceded while away and the save rate, with 95% CIs. Supports a "
                    "tell_strength sweep (--tell) and the hand ablation (--no-hand). Each policy keeps a fixed color."],
    },
    "eval/visualize_goalkeeper.py": {
        "title": "Top-down slow-motion goalkeeper viewer",
        "purpose": ["Shows the true puck, the last camera frame, the Kalman estimate and the paddle (faded while "
                    "locked), in a live window or as a GIF. Records frames through the env's substep_hook."],
    },
    "eval/view_mujoco.py": {
        "title": "3D MuJoCo viewer",
        "purpose": ["Runs MujocoGoalkeeperEnv with a trained run or a scripted policy. Live window with mjpython "
                    "(required on macOS) or an offscreen GIF. The blue ring is the Kalman estimate; the small "
                    "sphere is the last camera frame. Cameras: gate, side, top."],
    },
    "sysid/schema.py": {
        "title": "Log schemas",
        "purpose": ["Column definitions for band_bench, servo_step and shots logs, load_log() (validates columns, "
                    "empty values and phase names) and find_logs() (finds one log of each kind in a folder)."],
    },
    "sysid/synthetic.py": {
        "title": "Synthetic logs with known true values",
        "purpose": ["Generates the three log types from the simulator with a 'truth' config that differs from v0 "
                    "(camera noise, dropouts, gauge noise, encoder quantization), plus truth.yaml, to test that "
                    "the fits recover known values before real data exists."],
    },
    "sysid/fit.py": {
        "title": "Physics fits",
        "purpose": [
            "fit_band_bench: band stiffness, exponent, hysteresis (natural length measured and held fixed, because "
            "the bench data cannot separate it from stiffness and exponent).",
            "fit_servo_steps: latency (1 ms grid), deadband (0.02 deg grid, no useful gradient), lag and max rate "
            "(least squares on all tests at once, simulated as one vector).",
            "fit_shots: compliance (measured pull depth vs command), friction (one deceleration shared by every "
            "free sliding segment: exit, through the gate, back from the divider), wall restitution (divider "
            "bounces) and energy transfer (exit speeds vs the band model).",
        ],
        "notes": ["Four problems were found on synthetic data and fixed: parameters that trade off, a deadband "
                  "stuck at its start value, frames selected by the noisy y (biased speeds 8% low; now a second "
                  "pass by the fitted y), and a bounce between two frames that put post-bounce frames into the "
                  "first segment. All shot parameters are now within ~1 standard error over 6 seeds."],
    },
    "sysid/run_sysid.py": {
        "title": "Sysid command",
        "purpose": ["Runs the fits for the logs present and writes physics_params_v<N+1>.yaml: the parent with the "
                    "fitted values, those values removed from 'placeholders', servo values in its servo: section, "
                    "and a sysid: record (parent, log checksums, git hash, fits with standard errors, suggested "
                    "+-2 sigma randomization ranges). Synthetic fits are never written to configs/."],
    },
    "deploy/export_onnx.py": {
        "title": "ONNX export (old code, M5)",
        "purpose": ["Exports the actor of an SB3 PPO model to ONNX. Written before the rework; M5 must update it "
                    "for the asymmetric (Dict-observation) policy."],
    },
    "deploy/latency_benchmark.py": {
        "title": "ONNX latency benchmark (old code, M5)",
        "purpose": ["Times ONNX inference on CPU (mean and 99th percentile) against the 30 Hz control period. "
                    "Written before the rework; M5 must update the observation shapes and run it on the Jetson."],
    },
    "deploy/lerobot_interface.py": {
        "title": "LeRobot interface stub (M5)",
        "purpose": ["Control-loop stub for the real SO-101: read sensors, run the ONNX actor, send a rate-limited "
                    "joint command. It already uses the shared GoalkeeperGeometry mapping. Its observation is still "
                    "the old 6-value layout (FIXME for M5); the hardware calls are commented out."],
    },
}

# ----------------------------------------------------------------- descriptions
DESC = {
    # config.py
    "config.py::ConfigError": "Error raised for a bad or uncalibrated config.",
    "config.py::config_dir": "Return the config folder: the argument, else $SLINGPUCK_CONFIG_DIR, else configs/.",
    "config.py::available_param_versions": "List the version numbers of the physics_params_v<N>.yaml files in a folder.",
    "config.py::_read_yaml": "Read one YAML file (empty file -> empty dict).",
    "config.py::deep_merge": "Merge two nested dicts; values from 'override' win, key by key.",
    "config.py::get_path": "Read a value by dotted path, e.g. 'puck.mass_kg'.",
    "config.py::set_path": "Write a value by dotted path.",
    "config.py::_find_nulls": "List the dotted paths of all null values.",
    "config.py::git_state": "Return the git hash and whether the working tree has uncommitted changes.",
    # kinematics
    "kinematics.py::GoalkeeperGeometry": "Base position, arm radius, pan range, paddle size and joint zero offset.",
    "kinematics.py::GoalkeeperGeometry.from_config": "Build from the config and check that the geometry is valid.",
    "kinematics.py::GoalkeeperGeometry.pan_to_action": "Inverse of action_to_pan (clipped to [-1, 1]).",
    "kinematics.py::GoalkeeperGeometry.joint_to_pan": "Joint reading to board-frame pan angle.",
    "kinematics.py::GoalkeeperGeometry.paddle_center": "Paddle center position on its arc for a pan angle.",
    "kinematics.py::GoalkeeperGeometry.paddle_state": "PaddleState (pose, velocity, angular velocity) for the physics sim.",
    # backend
    "physics/backend.py::GateCrossing": "A puck crossed y = 0: puck index, direction (+1 to the opponent), x.",
    "physics/backend.py::StepEvents": "Events of one step: gate crossings and paddle contacts.",
    "physics/backend.py::PuckPhysicsBackend": "Abstract interface every puck physics backend implements.",
    # puck_dynamics
    "physics/puck_dynamics.py::BoardGeometry": "Inner board size, gate width, divider thickness, wall restitution.",
    "physics/puck_dynamics.py::BoardGeometry.from_config": "Build from cfg['board'].",
    "physics/puck_dynamics.py::BoardGeometry.half_length": "Half of the inner length.",
    "physics/puck_dynamics.py::BoardGeometry.half_width": "Half of the inner width.",
    "physics/puck_dynamics.py::PuckParams": "Puck radius, mass, friction and puck-puck restitution.",
    "physics/puck_dynamics.py::PuckParams.from_config": "Build from cfg['puck'].",
    "physics/puck_dynamics.py::Fast2DPuckSim": "The fast planar puck simulator (implements PuckPhysicsBackend).",
    "physics/puck_dynamics.py::Fast2DPuckSim.__init__": "Check restitutions and that the gate is wider than the puck.",
    "physics/puck_dynamics.py::Fast2DPuckSim.from_config": "Build from the merged config.",
    "physics/puck_dynamics.py::Fast2DPuckSim.reset": "Set positions and velocities; reject pucks outside the board.",
    "physics/puck_dynamics.py::Fast2DPuckSim.get_state": "Copies of positions and velocities.",
    "physics/puck_dynamics.py::Fast2DPuckSim.set_paddle": "Set (or remove) the kinematic paddle for the next steps.",
    "physics/puck_dynamics.py::Fast2DPuckSim.kinetic_energy": "Total translational kinetic energy.",
    "physics/puck_dynamics.py::Fast2DPuckSim.step": "Advance dt: friction, motion, paddle, walls, divider, puck-puck; return events.",
    "physics/puck_dynamics.py::Fast2DPuckSim._collide_paddle": "Circle vs oriented box contact, using the paddle surface velocity.",
    "physics/puck_dynamics.py::Fast2DPuckSim._collide_pucks": "Equal-mass impulse between overlapping pucks.",
    # servo
    "physics/servo_model.py::ServoModel": "Position servo: latency, deadband, first-order lag, rate limit.",
    "physics/servo_model.py::ServoModel.__init__": "Read the servo config (deg -> rad) and set up the latency queue.",
    "physics/servo_model.py::ServoModel.effective_latency_s": "Latency after rounding to whole steps.",
    "physics/servo_model.py::ServoModel.reset": "Set the position and fill the latency queue with it.",
    "physics/servo_model.py::ServoModel.step": "Apply one command and return the new position.",
    # band
    "physics/band_model.py::BandParams": "Peg span, natural length, stiffness, exponent, hysteresis, energy transfer.",
    "physics/band_model.py::BandParams.from_config": "Build from cfg['band'].",
    "physics/band_model.py::ReleaseResult": "Exit position and velocity, release time, band work, launched flag.",
    "physics/band_model.py::BandModel": "Band geometry, tension laws and release dynamics.",
    "physics/band_model.py::BandModel.__init__": "Check the parameters and place the anchors on the band line.",
    "physics/band_model.py::BandModel.from_config": "Build from the config (band line from the end-wall offset).",
    "physics/band_model.py::BandModel.tension_load": "Loading tension for an elongation.",
    "physics/band_model.py::BandModel.elongation": "Band length through a point minus the natural length.",
    "physics/band_model.py::BandModel.release": "Integrate the release from a pulled point; return a ReleaseResult.",
    # deflection
    "physics/arm_deflection.py::PullResult": "Actual puck position, actual pull distance, band force there.",
    "physics/arm_deflection.py::solve_pull": "Solve actual pull = command - compliance * resisting force (bisection).",
    # mujoco
    "physics/mujoco_goalkeeper.py::_rotz": "Rotation matrix about z.",
    "physics/mujoco_goalkeeper.py::SceneLayout.from_config": "Compute base pose and paddle target from the config.",
    "physics/mujoco_goalkeeper.py::ArmHoldPose": "IK joints, sag-corrected hold targets, settled pose, paddle mount pose, errors.",
    "physics/mujoco_goalkeeper.py::_divider_boxes": "The two divider blocks beside the gate.",
    "physics/mujoco_goalkeeper.py::_build_spec": "Build the MjSpec: SO-101, board, walls, divider, puck, paddle, pairs, lights, cameras.",
    "physics/mujoco_goalkeeper.py::_stiffness": "Contact stiffness for the configured contact time.",
    "physics/mujoco_goalkeeper.py::ContactCalibration.solref": "Interpolate (-stiffness, -damping) for a target restitution.",
    "physics/mujoco_goalkeeper.py::_qadr": "qpos addresses of joints by name.",
    "physics/mujoco_goalkeeper.py::_pose_key": "Cache key of the config values that change the arm pose.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.__init__": "Solve (or reuse) the arm pose, compile the model, calibrate contacts.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.reset_arm": "Put the arm at a pan angle in the settled hold pose.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.set_pan_target": "Set the shoulder_pan actuator target.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.pan": "Current board-frame pan angle.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.pan_vel": "Current pan velocity.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.paddle_center": "Paddle center in the world.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.reset": "Set the puck position and velocity.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.pos": "Puck position (1, 2).",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.vel": "Puck velocity (zero below 1 mm/s, so 'stopped' works).",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.get_state": "Position and velocity.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.set_paddle": "No-op: the arm moves the paddle.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.kinetic_energy": "Puck kinetic energy.",
    "physics/mujoco_goalkeeper.py::MujocoGoalkeeperSim.step": "Apply friction, step MuJoCo, detect paddle contacts and gate crossings.",
    "physics/mujoco_goalkeeper.py::MujocoPanServo.__init__": "Set up the command latency queue.",
    "physics/mujoco_goalkeeper.py::MujocoPanServo.reset": "Reset the arm and fill the queue.",
    "physics/mujoco_goalkeeper.py::MujocoPanServo.step": "Delay the command, send it to the actuator, return the pan.",
    "physics/mujoco_goalkeeper.py::MujocoPanServo.pos": "Pan angle (same interface as ServoModel).",
    "physics/mujoco_goalkeeper.py::MujocoPanServo.vel": "Pan velocity.",
    # camera / tracker
    "sensing/camera_model.py::Frame": "Capture time, arrival time and measured position of one frame.",
    "sensing/camera_model.py::CameraModel": "Frame rate, latency, noise and dropouts.",
    "sensing/camera_model.py::CameraModel.__init__": "Read the camera config.",
    "sensing/camera_model.py::CameraModel.reset": "Restart frame timing and clear pending frames.",
    "sensing/camera_model.py::CameraModel.observe": "Capture when a frame is due; return frames that arrive now.",
    "sensing/camera_model.py::CameraModel._capture": "Take one frame: dropout or noisy measurement.",
    "sensing/kalman_tracker.py::_transition": "State transition and process noise for a time step.",
    "sensing/kalman_tracker.py::KalmanTracker": "Constant-velocity Kalman filter for one puck.",
    "sensing/kalman_tracker.py::KalmanTracker.__init__": "Set measurement and process noise.",
    "sensing/kalman_tracker.py::KalmanTracker.from_config": "Build from the camera and tracker config.",
    "sensing/kalman_tracker.py::KalmanTracker.reset": "Forget the state.",
    "sensing/kalman_tracker.py::KalmanTracker.initialized": "True after the first frame.",
    "sensing/kalman_tracker.py::KalmanTracker.update": "Predict to the frame capture time and update (Joseph form).",
    # goalkeeper env
    "envs/goalkeeper_env.py::Shot": "Start, velocity, speed, pre-launch time, threat flag and tries of one shot.",
    "envs/goalkeeper_env.py::GoalkeeperEnv": "Phase 1 Gymnasium env.",
    "envs/goalkeeper_env.py::GoalkeeperEnv.__init__": "Read the task config and define the spaces.",
    "envs/goalkeeper_env.py::GoalkeeperEnv._sample_shot": "Sample start, aim and speed; resample until it is a threat.",
    "envs/goalkeeper_env.py::GoalkeeperEnv.step": "Apply the action for one control period (33 physics steps); return reward and outcome.",
    "envs/goalkeeper_env.py::GoalkeeperEnv._reset_physics": "Physics hook: reset servo, puck and paddle.",
    "envs/goalkeeper_env.py::GoalkeeperEnv._physics_step": "Physics hook: servo step, paddle pose, sim step.",
    "envs/goalkeeper_env.py::GoalkeeperEnv._check_outcome": "Decide save / goal / none after a physics step.",
    "envs/goalkeeper_env.py::GoalkeeperEnv._final_outcome": "Outcome by side when the puck stops or time runs out.",
    "envs/goalkeeper_env.py::GoalkeeperEnv._obs": "The 8-value policy observation (tracked puck, encoder, previous action).",
    "envs/goalkeeper_env.py::GoalkeeperEnv.privileged_obs": "The 16-value critic observation (true state, randomized physics).",
    "envs/goalkeeper_env.py::GoalkeeperEnv._info": "Info dict: outcome, speed, threat, start pan, release delay, privileged.",
    # mujoco env
    "envs/mujoco_goalkeeper_env.py::MujocoGoalkeeperEnv": "GoalkeeperEnv with MuJoCo physics hooks.",
    "envs/mujoco_goalkeeper_env.py::MujocoGoalkeeperEnv.__init__": "Build the MuJoCo sim once.",
    "envs/mujoco_goalkeeper_env.py::MujocoGoalkeeperEnv._build_physics": "Apply randomized parameters in place; set sim and servo.",
    "envs/mujoco_goalkeeper_env.py::MujocoGoalkeeperEnv._reset_physics": "Reset the arm at the start pan and place the puck.",
    "envs/mujoco_goalkeeper_env.py::MujocoGoalkeeperEnv._physics_step": "Send the pan command and step MuJoCo.",
    # sling env
    "envs/sling_env.py::FlightResult": "Success, miss distance, exit state and optional trajectory of one shot.",
    "envs/sling_env.py::SlingEnv": "Phase 2 Gymnasium env.",
    "envs/sling_env.py::SlingEnv.__init__": "Read the task config and define the spaces.",
    "envs/sling_env.py::SlingEnv.scale_aim": "Map the step-1 action to slide x, pull angle and pull distance.",
    "envs/sling_env.py::SlingEnv.scale_correction": "Map the step-2 action to a lateral, angle and pull correction.",
    "envs/sling_env.py::SlingEnv.step": "Step 1: slide and pull (then observe). Step 2 (or step 1 without correction): correct, release, fly, reward.",
    "envs/sling_env.py::SlingEnv._obs": "The 7-value policy observation.",
    "envs/sling_env.py::SlingEnv.privileged_obs": "The 12-value critic observation.",
    "envs/sling_env.py::SlingEnv._info": "Info dict: start x, stage, outcome, miss, exit speed, commands, actual pull.",
    # match env
    "envs/match_env.py::MatchEnv": "Phase 3 Gymnasium env (event-level match).",
    "envs/match_env.py::MatchEnv.__init__": "Load (or build) the block table, set the sling primitive and the spaces.",
    "envs/match_env.py::MatchEnv.step": "Apply the action, advance 0.1 s in 10 ms steps, add win/loss reward.",
    "envs/match_env.py::MatchEnv.can_sling": "True if the arm is free (gate or band), a puck is left and none is in flight.",
    "envs/match_env.py::MatchEnv._apply_action": "Start a sling or a return to the gate (busy actions are ignored).",
    "envs/match_env.py::MatchEnv._advance": "Advance the arm state machine, own puck, opponent and incoming pucks by one step.",
    "envs/match_env.py::MatchEnv._flight": "Analytic arrival time of an opponent shot at the gate.",
    "envs/match_env.py::MatchEnv._resolve": "Resolve an arriving shot: miss, save (block table) or goal.",
    "envs/match_env.py::MatchEnv._match_over": "Apply the win rule (first side with no pucks).",
    "envs/match_env.py::MatchEnv._obs": "The 20-value policy observation.",
    "envs/match_env.py::MatchEnv.privileged_obs": "The 11-value critic observation.",
    "envs/match_env.py::MatchEnv._info": "Info dict: outcome, counts, arm state, opponent phase, stats at the end.",
    # wrappers
    "envs/wrappers.py::PrivilegedObsWrapper.__init__": "Define the Dict observation space.",
    "envs/wrappers.py::PrivilegedObsWrapper.reset": "Reset and keep the privileged vector from info.",
    "envs/wrappers.py::PrivilegedObsWrapper.step": "Step and keep the privileged vector from info.",
    "envs/wrappers.py::PrivilegedObsWrapper.observation": "Build the Dict observation.",
    # opponent
    "opponent/scripted_opponent.py::Shot": "Launch time, launch x, speed and threat flag of one opponent shot.",
    "opponent/scripted_opponent.py::_Cycle": "Phase boundary times, hand waypoints and the shot of one cycle.",
    "opponent/scripted_opponent.py::Opponent": "The scripted opponent.",
    "opponent/scripted_opponent.py::Opponent.__init__": "Read the opponent config and the band line.",
    "opponent/scripted_opponent.py::Opponent._launch_x": "Sample a launch x for a threat or a miss.",
    "opponent/scripted_opponent.py::Opponent._start_cycle": "Sample one full cycle: timings, shot, tell, hand waypoints.",
    "opponent/scripted_opponent.py::Opponent.reset": "Start at home; begin a cycle if the opponent has a puck.",
    "opponent/scripted_opponent.py::Opponent.phase": "Current phase name.",
    "opponent/scripted_opponent.py::Opponent.hand_observed": "True hand plus camera noise.",
    "opponent/scripted_opponent.py::Opponent.time_to_release": "Seconds until the next release (privileged).",
    "opponent/scripted_opponent.py::Opponent.next_shot": "The shot of the current cycle (privileged).",
    # primitives
    "primitives/block.py::BlockTable": "Speeds, delays, p_gate and p_edge arrays.",
    "primitives/block.py::BlockTable.p_save": "Save probability for a speed, gate/edge start and delay (interpolated).",
    "primitives/block.py::BlockTable._edge_at": "Bilinear interpolation in the edge table.",
    "primitives/block.py::_cache_key": "Hash of everything that changes the table.",
    "primitives/block.py::_policy": "The block policy: scripted center or a trained run.",
    "primitives/block.py::build_block_table": "Measure the table on the Phase 1 env.",
    "primitives/block.py::load_or_build_block_table": "Load the cached table, or build and cache it.",
    "primitives/sling.py::SlingPrimitive": "Sling timings and success probability.",
    "primitives/sling.py::SlingPrimitive.from_config": "Build from cfg['match'].",
    # policies
    "policies/asymmetric.py::KeySelectExtractor.__init__": "Feature size = sum of the selected keys.",
    "policies/asymmetric.py::KeySelectExtractor.forward": "Concatenate the selected keys.",
    "policies/asymmetric.py::AsymmetricMlpExtractor.__init__": "Actor MLP on the actor input size, critic MLP on the critic size.",
    "policies/asymmetric.py::AsymmetricMlpExtractor.forward": "Run both networks.",
    "policies/asymmetric.py::AsymmetricMlpExtractor.forward_actor": "Actor network.",
    "policies/asymmetric.py::AsymmetricMlpExtractor.forward_critic": "Critic network.",
    "policies/asymmetric.py::AsymmetricActorCriticPolicy": "SB3 ActorCriticPolicy with separate actor and critic inputs.",
    "policies/asymmetric.py::AsymmetricActorCriticPolicy.__init__": "Check the Dict space; force separate feature extractors.",
    "policies/asymmetric.py::AsymmetricActorCriticPolicy.make_features_extractor": "First call: actor keys; second call: critic keys.",
    "policies/asymmetric.py::AsymmetricActorCriticPolicy._build_mlp_extractor": "Use AsymmetricMlpExtractor.",
    "policies/scripted.py::CenterBlocker": "Always command pan = 0.",
    "policies/scripted.py::CenterBlocker.predict": "Return action 0 for every observation.",
    "policies/scripted.py::HoldStart": "Never move.",
    "policies/scripted.py::HoldStart.predict": "Return the previous action.",
    "policies/sling_scripted.py::SlideCenter": "Slide to x = 0, pull straight back, no correction.",
    "policies/sling_scripted.py::SlideCenter.__init__": "Set the pull fraction.",
    "policies/sling_scripted.py::SlideCenter.predict": "Aim action at step 1, zero correction at step 2.",
    "policies/sling_scripted.py::SlideCenterCorrect": "SlideCenter plus a lateral correction from the camera.",
    "policies/sling_scripted.py::SlideCenterCorrect.__init__": "Read the board length and the correction range.",
    "policies/sling_scripted.py::SlideCenterCorrect.predict": "At step 2, shift the puck back to x = 0.",
    "policies/match_scripted.py::_Base": "Common predict() for Box or Dict observations.",
    "policies/match_scripted.py::_Base.predict": "Call act() with the policy (and privileged) vector.",
    "policies/match_scripted.py::_Base.arm": "True if the arm is in the given state.",
    "policies/match_scripted.py::AlwaysBlock": "Never sling.",
    "policies/match_scripted.py::AlwaysBlock.act": "Return block.",
    "policies/match_scripted.py::GreedySling": "Sling whenever possible.",
    "policies/match_scripted.py::GreedySling.act": "Return sling.",
    "policies/match_scripted.py::ReloadRule.__init__": "Read the opponent band line and the margin.",
    "policies/match_scripted.py::ReloadRule.reloading": "Hand away from the opponent band line.",
    "policies/match_scripted.py::ReloadRule.safe": "Safe to sling = opponent reloading.",
    "policies/match_scripted.py::ReloadRule.act": "Sling if safe and no puck is incoming, else block.",
    "policies/match_scripted.py::TellRule.__init__": "Set the miss threshold for the hand x.",
    "policies/match_scripted.py::TellRule.safe": "Reloading, or the hand x shows a likely miss.",
    "policies/match_scripted.py::OracleReload.reloading": "True opponent phase is reload or idle.",
    # train
    "train/common.py::make_sling_env": "SlingEnv, wrapped for the asymmetric policy if needed.",
    "train/common.py::load_train_config": "Read a train_*.yaml file.",
    "train/common.py::new_run_dir": "Run id and folder: logs/<task>/<task>_<params>_s<seed>_<time>.",
    "train/common.py::OutcomeLoggerCallback.__init__": "Set the outcome names to count.",
    "train/common.py::OutcomeLoggerCallback._on_step": "Count outcomes of finished episodes.",
    "train/common.py::OutcomeLoggerCallback._on_rollout_end": "Log outcome rates to TensorBoard and reset the counts.",
    # eval
    "eval/save_rate_vs_speed.py::in_bin": "True if a value is in bin b (last bin includes its upper edge).",
    "eval/save_rate_vs_speed.py::plot": "Two panels: save rate vs speed and vs start offset, with CI bands.",
    "eval/save_rate_vs_speed.py::write_csv": "Write a list of dicts to CSV.",
    "eval/save_rate_vs_speed.py::main": "Command line: evaluate runs and baselines, write tables, plot and metadata.",
    "eval/hole_rate.py::run_shots": "Run one policy for n shots; return per-shot results.",
    "eval/hole_rate.py::summarize": "Success rates overall and by start position, with CIs.",
    "eval/hole_rate.py::_style": "Common axis style.",
    "eval/hole_rate.py::plot_policies": "Running success vs shot count, and success by start position.",
    "eval/hole_rate.py::_transit": "Transit time from 2 cm past the band line to y = -4 cm.",
    "eval/hole_rate.py::parity": "Replay logged shots in the sim for each params version.",
    "eval/hole_rate.py::parity_report": "Brier score, reliability table, transit bias; parity plot.",
    "eval/hole_rate.py::write_csv": "Write a list of dicts to CSV.",
    "eval/hole_rate.py::main": "Command line: policy mode or parity mode.",
    "eval/match_eval.py::play": "Play n matches with one policy; return per-match stats.",
    "eval/match_eval.py::summarize": "Rates and means with CIs for one policy and tell value.",
    "eval/match_eval.py::plot": "Win rate and puck difference: dot plot, or lines for a tell sweep.",
    "eval/match_eval.py::main": "Command line: scripted policies and trained runs, tell sweep, hand ablation.",
    "eval/visualize_goalkeeper.py::load_policy": "Load a trained run or a scripted baseline.",
    "eval/visualize_goalkeeper.py::draw_static": "Draw the board, divider, band line and paddle arc.",
    "eval/visualize_goalkeeper.py::animate": "Animate recorded episodes (window or GIF).",
    "eval/visualize_goalkeeper.py::main": "Command line for the 2D viewer.",
    "eval/view_mujoco.py::episode_options": "Reset options from the command line (speed, start offset, delay).",
    "eval/view_mujoco.py::live": "Live MuJoCo window, paced to the slow-motion factor.",
    "eval/view_mujoco.py::save_gif": "Render episodes offscreen and save a GIF with a text header.",
    "eval/view_mujoco.py::main": "Command line for the 3D viewer.",
    # sysid
    "sysid/schema.py::LogError": "Error raised for a log that does not match its schema.",
    "sysid/synthetic.py::truth_config": "Copy of the base config with the true values set.",
    "sysid/synthetic.py::band_bench_log": "Synthetic force-gauge cycles (load and unload) with gauge noise.",
    "sysid/synthetic.py::servo_step_log": "Synthetic step tests with encoder noise and 12-bit quantization.",
    "sysid/synthetic.py::generate": "Write band_bench.csv, servo_step.csv, shots.csv and truth.yaml.",
    "sysid/synthetic.py::main": "Command line for the generator.",
    "sysid/fit.py::FitValue": "Fitted value, standard error, number of data points, note.",
    "sysid/fit.py::FitValue.as_dict": "Plain dict for YAML.",
    "sysid/fit.py::with_values": "Copy of a config with fitted values set.",
    "sysid/fit.py::_servo_sim": "Simulate all step tests at once (vector ServoModel) for given parameters.",
    "sysid/fit.py::_Shot": "Rest position, pulled position, flight frames and commands of one logged shot.",
    "sysid/fit.py::_shots": "Group the shots log into _Shot objects.",
    "sysid/fit.py::_Segment": "Frames of one free sliding segment (exit, through, return).",
    "sysid/fit.py::_seg_params_guess": "Start values (position, heading, speed) for a segment.",
    "sysid/fit.py::fit_shots": "Compliance, friction, wall restitution and energy transfer from the shots log.",
    "sysid/run_sysid.py::_sha256": "Short checksum of a log file.",
    "sysid/run_sysid.py::run": "Run the fits and write the new params version; return its path.",
    "sysid/run_sysid.py::main": "Command line for sysid.",
    # deploy
    "deploy/lerobot_interface.py::SO101Controller": "Control loop stub for the real arm.",
    "deploy/lerobot_interface.py::SO101Controller.__init__": "Load the ONNX actor, the shared geometry and the limits.",
    "deploy/lerobot_interface.py::SO101Controller.run_loop": "Sense, infer, act, sleep to keep the control rate.",
}
