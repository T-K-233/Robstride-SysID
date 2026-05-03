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

## Layout

```
scripts/
  signals.py        excitation waveform generators (multisine, chirp)
  streaming.py      bus open/close + the streaming-recording loop
  recording.py      MCAP I/O + Sequence dataclass + load_sequence/resample
  model.py          inlined MJCF + helper to build an MjSpec
  collect.py        CLI: drive feed-forward torque, log MCAPs into a run dir
  optimize.py       glob a run dir, run mujoco.sysid, write results dir
  visualize.py      forward-simulate per recording, write fit_<stem>.png
  discover.py       scan a CAN bus for Robstride device IDs
tests/
  sim_collect.py    twin of collect.py, MuJoCo-backed with known truth
  check_recovery.py compare optimizer output against ground-truth metadata
data/<class>/<run>/    per-run recordings (e.g. data/rs-02/sample1_run1/)
results/<class>/<run>/ per-run fit outputs (results.json, report.html, fit_*.png)
```

## Validate the pipeline (no hardware needed)

`tests/sim_collect.py` drives the same MJCF with preset ground-truth
parameters and Gaussian sensor noise, producing MCAPs in the same format
`collect.py` would. End-to-end check:

```bash
uv run tests/sim_collect.py -o data/sim
uv run scripts/optimize.py --recordings data/sim/ --out-dir results/sim/
uv run tests/check_recovery.py --recordings data/sim/ --out-dir results/sim/ --rel-tol 0.30
uv run scripts/visualize.py --recordings data/sim/ --out-dir results/sim/
```

`check_recovery.py` exits non-zero if any param exceeds the tolerance.

## Hardware workflow

Recordings are organized by actuator class (`--model`) and run number,
auto-incremented by `collect.py`:

```
data/<model>/run<N>/{multisine,chirp}.mcap
results/<model>/run<N>/{results.json, report.html, fit_*.png}
```

`collect.py` runs the actuator in MIT mode with kp=kd=0, so the firmware
emits a pure feed-forward torque -- the same signal that becomes the
MuJoCo `<motor>` actuator's `ctrl` during optimization. Each invocation
picks the next free `run<N>` slot under `data/<model>/`:

```bash
uv run scripts/collect.py --channel can0 --id 1 --model rs-02
# -> data/rs-02/run1/   (or run2, run3, ... if earlier runs exist)
```

`optimize.py --model rs-02` iterates *every* `data/rs-02/run<N>/` and
writes the corresponding `results/rs-02/run<N>/`:

```bash
uv run scripts/optimize.py --model rs-02
uv run scripts/visualize.py --model rs-02
```

`optimize.py` writes `results.json` (params + initial values) and
`report.html` (MuJoCo's interactive sysid report with confidence
intervals) per run. Useful flags:
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
