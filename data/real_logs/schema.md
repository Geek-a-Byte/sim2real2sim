# Real log schemas (sysid input)

`python -m src.slingpuck.sysid.run_sysid --logs <folder> --natural-length <m>` reads
the CSV files below from one folder (one file of each kind; the file name must
start with the kind, for example `shots_2026-10-10.csv`). Any subset works; each
fit runs only for the logs that are present. The code that checks these columns
is `src/slingpuck/sysid/schema.py`.

All positions are in the **board frame**: origin at the board center, x across the
width, y along the length, agent half y < 0, divider and gate at y = 0. Units are
SI (m, s, rad, N).

`synthetic_example/` holds a small set of **synthetic** logs in this format
(made by `python -m src.slingpuck.sysid.synthetic`; `truth.yaml` lists the true
values). Use it to try the pipeline. Do not report results from it as real data.

---

## 1. `band_bench*.csv`: band force test

Hook a force gauge to the band center. Pull it straight back from the band line
(toward the agent end wall) in small steps up to the maximum pull, then let it
back in small steps. Do 3 or more cycles, ideally with different maximum pulls.

| Column | Type | Description |
|---|---|---|
| `cycle` | int | Cycle number |
| `phase` | str | `load` (pulling back) or `unload` (letting back) |
| `pull_m` | float | Distance of the band center behind the band line, m |
| `force_n` | float | Gauge force along the pull direction, N |

Fits: `band.stiffness_k_n_m`, `band.stiffness_exponent`, `band.hysteresis_loss_factor`.

**Also measure the natural band length** (band unhooked, lying flat, with a
ruler) and pass it as `--natural-length`. The bench data alone cannot separate
it from stiffness and exponent.

## 2. `servo_step*.csv`: servo step response

Send step commands to one joint (normally `shoulder_pan`) **with the paddle
mounted**, and log the command and the encoder reading at a fixed rate
(5 ms or faster). Use several step sizes and directions, including small steps
(for the deadband).

| Column | Type | Description |
|---|---|---|
| `test_id` | int | Step test number |
| `t_s` | float | Time since the start of this test, s |
| `cmd_rad` | float | Commanded joint angle, rad |
| `meas_rad` | float | Measured joint angle (encoder), rad |

Fits: `servo.command_latency_s`, `servo.lag_tau_s`, `servo.max_rate_deg_s`,
`servo.deadband_deg`. This is the most important real measurement: the MuJoCo
sim-to-sim test showed that the servo model decides the Phase 1 results.

## 3. `shots*.csv`: sling shots seen by the camera

One row per camera frame. For each shot, log three phases:

1. `rest`: the puck still against the band, before the slide (about 0.3 s).
2. `pulled`: the puck held still by the arm after the slide and pull (about 0.3 s).
3. `flight`: after the release command, until the puck stops (about 1 s).

Include **off-center shots that hit the divider face and bounce back** (puck 3 to
6 cm from the gate center). Straight shots never touch the side walls, so these
bounces are the only source of the wall restitution.

| Column | Type | Description |
|---|---|---|
| `shot_id` | int | Shot number |
| `phase` | str | `rest`, `pulled` or `flight` |
| `t_s` | float | Frame capture time. For `flight`: seconds since the release command |
| `puck_x_m`, `puck_y_m` | float | Puck position from the camera tracker, board frame, m |
| `cmd_slide_m` | float | Commanded slide position along the band, m |
| `cmd_angle_rad` | float | Commanded pull angle, rad (0 = straight back, away from the gate) |
| `cmd_pull_m` | float | Commanded pull distance, m |
| `success` | int | 1 if the puck went through the gate, else 0 (same on every row of a shot) |

Fits: `arm.deflection_compliance_m_per_n` (measured pull depth vs command),
`puck.friction_kinetic` (sliding segments), `board.restitution` (divider
bounces), `band.energy_transfer` (exit speeds vs the band model).

The same file is the input to the sim-vs-real parity check:
`python -m src.slingpuck.eval.hole_rate --parity <shots.csv> --versions v0 v1`.
