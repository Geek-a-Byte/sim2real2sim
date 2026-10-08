# Real-World Teleoperation Log Schema

For Phase 2 (Targeted Shot) system identification, record teleoperated slingshot attempts. The system ID pipeline expects a CSV file (`real_shots_vX.csv`) with the following columns:

| Column | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | float | Time in seconds from the start of the recording. |
| `puck_x_m`, `puck_y_m` | float | Tracked puck coordinates from the overhead camera. |
| `cmd_pan`, `cmd_pull` | float | Commanded joint target (pan angle in rad, pullback extension in m). |
| `meas_pan`, `meas_pull` | float | Measured joint position from the SO-101 servos (captures deflection/latency). |
| `band_disp_m` | float | Visual displacement of the band center from its resting position. |
| `launch_vel_m_s` | float | Calculated exit velocity of the puck after slipping the band. |
| `gate_success` | int | 1 if the puck passed through the gate, 0 otherwise. |