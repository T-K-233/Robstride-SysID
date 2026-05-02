# Robstride Actuator Identification with MuJoCo

Identify `armature`, `damping`, and `frictionloss` of Robstride actuators
from CAN-bus recordings, using the MuJoCo system-identification
toolbox introduced in MuJoCo 3.5. The actuator is mounted on a rigid
stand with the output shaft unloaded; torque commands are streamed at
400 Hz over CAN; the optimizer fits a single-hinge MJCF to the recorded
position/velocity.

## Setup

```bash
uv sync --prerelease=allow

# bring up the CAN interface (once per boot)
sudo ip link set can0 down && sudo ip link set can0 up type can bitrate 1000000
```

## Validate the pipeline (no hardware needed)

`tests/sim_collect.py` drives the same MJCF with preset ground-truth
parameters and Gaussian sensor noise, producing MCAPs in the same format
`collect.py` would. End-to-end check:

```bash
uv run tests/sim_collect.py -o data/sim
uv run scripts/optimize.py --recordings data/sim/ --out-dir results/sim/
```

## Hardware workflow

```bash
uv run scripts/collect.py --channel can0 --id 1 -o data/rs-02/sample1_run1

uv run scripts/optimize.py \
  --recordings data/rs-02/sample1_run1/ \
  --out-dir results/rs-02/sample1_run1/
```

`optimize.py` writes `results.json` (params + initial values) and
`report.html` (MuJoCo's interactive sysid report with confidence
intervals). Useful flags:
- `--no-frictionloss` — freeze frictionloss at the initial value
- `--no-velocity-sensor` — fit position only (firmware velocity is
  differentiated and can have phase delay)

`visualize.py` writes one `fit_<stem>.png` per recording, overlaying
measured / initial / identified position, velocity, and feed-forward
torque with RMSE in the legend.

The collect loop typically lands at ~385 Hz rather than the requested
400 Hz; `optimize.py` reads back the empirical rate from the timestamps
so this jitter doesn't bias the simulation timestep.

## Storage format

Recordings are [MCAP](https://mcap.dev) files. Each file has:

* topic `/actuator/state`, JSON-schema message per timestep with
  `ctrl_torque`, `position`, `velocity`, `measured_torque`
* a `recording` metadata record with run-level fields (`sampling_rate`,
  `signal_type`, `actuator_id`; ground-truth fields for sim recordings)

`log_time` is nanoseconds since the start of the recording, not unix
wall-clock.

## Modeling notes

- The MJCF body has tiny fixed `diaginertia` so all rotational inertia
  is absorbed by the joint `armature` — that one parameter lumps rotor
  + reflected gearbox inertia at the output.
- `gear="1"` on the motor actuator means the recorded torque command in
  N·m is what the simulator applies directly.
- `integrator="implicitfast"` is recommended whenever `armature` is
  nonzero (matches the sysid notebook's robot-arm example).
- Gravity and contact are disabled — unloaded shaft, no contact.

## RS02 reference values

From the user manual:

| quantity | value |
|---|---|
| reduction | 7.75 : 1 |
| torque constant | 1.22 N·m / Arms |
| rated / peak torque | 6 / 17 N·m |
| no-load speed | 410 rpm ≈ 43 rad/s |
| weight | 380 g |
| encoder | 14-bit absolute (output side, post-gearbox) |
| CAN bitrate | 1 Mbps |

The manual doesn't publish a rotor inertia; `--init-armature` defaults
to `1e-3` kg·m² (output-side), which is a reasonable starting guess.
