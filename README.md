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

Joint parameters identified by `analyze.py` (output-side), reported as
mean ± relative stddev across `n` runs. `J` (armature) lumps rotor
inertia + reflected gearbox inertia; `b` is viscous damping; `Fc` is
Coulomb friction.

| model | n | `J` (kg·m²)     | `b` (N·m·s/rad) | `Fc` (N·m)    |
|-------|--:|----------------:|----------------:|--------------:|
| rs-00 | 3 | 0.0149 ± 11%    | 0.054 ± 129%    | 0.233 ± 46%   |
| rs-02 | 3 | 0.01369 ± 1.7%  | 0.0035 ± 33%    | 0.159 ± 3.7%  |
| rs-05 | 3 | 0.00678 ± 6.5%  | 0 (rail)        | 0.309 ± 22%   |
| rs-06 | 3 | 0.01647 ± 5.9%  | 0.0093 ± 141%   | 0.169 ± 59%   |

`J` is the most robust output: 1.7–11% relative spread across runs.
`b` and `Fc` trade off when the multisine + chirp excitation doesn't
visit low-velocity dwell long enough to disambiguate viscous from
Coulomb dissipation, so high-variance columns are an identifiability
hint, not a reproducibility hint. RS-02 happens to give a clean fit
on `Fc` here; RS-05's `b` pins to the lower bound on every run (all
dissipation lands in `Fc`); RS-00's run 3 and RS-06's run 2 land in
alternate minima where `b` and `Fc` swap roles.
