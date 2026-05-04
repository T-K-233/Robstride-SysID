# Robstride Actuator Identification with MuJoCo

Identify `armature`, `damping`, and `frictionloss` of Robstride actuators
from CAN-bus recordings, using the MuJoCo system-identification toolbox.
The actuator is mounted on a rigid stand with the output shaft unloaded;
torque commands are streamed at 400 Hz over CAN; the optimizer fits a
single-hinge MJCF to the recorded position/velocity.

## Setup

```bash
uv sync --prerelease=allow

# bring up the CAN interface (once per boot)
sudo ip link set can0 down && sudo ip link set can0 up type can bitrate 1000000
```

## Hardware workflow

```bash
uv run scripts/collect.py --channel can0 --id 1 --model rs-02
# writes data/rs-02/run<N>/{multisine,chirp}.mcap, N auto-incremented

uv run scripts/analyze.py --model rs-02
# iterates every data/rs-02/run<N>/ -> results/rs-02/run<N>/
```

`analyze.py` writes `results.json`, `report.html` (MuJoCo's interactive
sysid report), and one `fit_<stem>.png` per recording. Useful flags:
`--no-frictionloss` freezes frictionloss at the initial value;
`--no-velocity-sensor` fits position only (firmware velocity is
differentiated and can lag); `--no-plots` skips the per-recording fit
plots.

## Validate the pipeline (no hardware)

```bash
uv run tests/sim_collect.py -o data/sim
uv run scripts/analyze.py --recordings data/sim/ --out-dir results/sim/
uv run tests/check_recovery.py --recordings data/sim/ --out-dir results/sim/ --rel-tol 0.30
```

## Actuator parameters

Manufacturer specs and joint parameters identified by this pipeline
(output-side). `J` (armature) lumps rotor inertia + reflected gearbox
inertia; `b` is viscous damping; `Fc` is Coulomb friction. Identified
columns are filled in as `analyze.py --model <X>` converges across
multiple runs.

| model | reduction | rated / peak torque | torque const. | weight | `J` (kg·m²) | `b` (N·m·s/rad) | `Fc` (N·m) |
|-------|----------:|--------------------:|--------------:|-------:|------------:|----------------:|-----------:|
| rs-02 |   7.75:1  |        6 / 17 N·m   | 1.22 N·m/Arms | 380 g  |    TBD      |       TBD       |    TBD     |
