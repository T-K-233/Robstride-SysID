"""
Identify joint armature, damping, and frictionloss from recordings.

Reads every `*.mcap` file inside a recordings directory and treats each as
an independent sequence in a single MuJoCo `ModelSequences` -- the standard
way to combine experiments for better identifiability (see the mujoco.sysid
notebook). Outputs go into the parallel results directory.

Usage:
    python scripts/optimize.py \
        --recordings data/rs-02/run1/ \
        --out-dir results/rs-02/run1/

Convention (mirroring /home/tk/Desktop/Robstride-SysID):
    data/<class>/<run>/<signal>.mcap   recordings produced by collect.py
    results/<class>/<run>/results.json fitted params
    results/<class>/<run>/report.html  mujoco.sysid HTML report
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.rollout as rollout
import numpy as np
from mujoco import sysid

from model import JOINT_NAME, make_spec
from recording import read_mcap


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class Sequence:
    name: str
    times: np.ndarray            # (N,) seconds, ~uniform, rebased to t=0
    ctrl_torque: np.ndarray      # (N,) N.m -- input to motor actuator
    position: np.ndarray         # (N,) rad, output side
    velocity: np.ndarray         # (N,) rad/s, output side
    sampling_rate: float         # Hz


def load_sequence(path: Path) -> Sequence:
    """Load an MCAP recording. The sampling rate is derived from the actual
    timestamps -- the metadata's `sampling_rate` is only the *target* rate,
    and Python loop jitter typically makes the actual rate 1-5% slower.
    Using the empirical rate eliminates that bias in the simulation."""
    rec = read_mcap(path)
    if len(rec.times) < 2:
        raise ValueError(f"{path}: need >= 2 samples")
    declared = float(rec.metadata.get("sampling_rate", 0.0))
    empirical = 1.0 / float(np.median(np.diff(rec.times)))
    if declared > 0.0 and abs(empirical - declared) / declared > 0.005:
        print(
            f"  {path.name}: empirical rate {empirical:.2f} Hz "
            f"(declared {declared:.1f} Hz, "
            f"mismatch {(empirical - declared) / declared:+.1%})"
        )
    return Sequence(
        name=path.stem,
        times=rec.times.astype(np.float64),
        ctrl_torque=rec.ctrl_torque.astype(np.float64),
        position=rec.position.astype(np.float64),
        velocity=rec.velocity.astype(np.float64),
        sampling_rate=empirical,
    )


def to_uniform_grid(seq: Sequence, dt: float) -> Sequence:
    """Resample onto a uniform t = k*dt grid spanning the recording."""
    t0 = float(seq.times[0])
    t_end = float(seq.times[-1])
    n = int(np.floor((t_end - t0) / dt)) + 1
    t_uniform = t0 + dt * np.arange(n)
    return Sequence(
        name=seq.name,
        times=t_uniform - t0,                     # rebase to start at 0
        ctrl_torque=np.interp(t_uniform, seq.times, seq.ctrl_torque),
        position=np.interp(t_uniform, seq.times, seq.position),
        velocity=np.interp(t_uniform, seq.times, seq.velocity),
        sampling_rate=1.0 / dt,
    )


# --------------------------------------------------------------------------- #
# MuJoCo sysid setup                                                          #
# --------------------------------------------------------------------------- #

def _set_scalar(joint, attr, value):
    setattr(joint, attr, float(value))


def _set_damping(joint, value):
    # Joint damping is stored as a length-3 vector internally (ball-joint capable),
    # but only the first component matters for a 1-DoF hinge.
    joint.damping = np.array([float(value), 0.0, 0.0], dtype=np.float64)


def make_modifier(joint_name: str, attr: str):
    """Return a modifier that writes a scalar onto spec.joint(joint_name).<attr>."""
    if attr == "damping":
        def _mod(spec, param):
            _set_damping(spec.joint(joint_name), param.value[0])
    else:
        def _mod(spec, param):
            _set_scalar(spec.joint(joint_name), attr, param.value[0])
    return _mod


def build_initial_params(
    init_armature: float,
    init_damping: float,
    init_frictionloss: float,
) -> sysid.ParameterDict:
    params = sysid.ParameterDict()
    params.add(sysid.Parameter(
        "armature",
        nominal=init_armature,
        min_value=1e-5,
        max_value=0.1,
        modifier=make_modifier(JOINT_NAME, "armature"),
    ))
    params.add(sysid.Parameter(
        "damping",
        nominal=init_damping,
        min_value=0.0,
        max_value=5.0,
        modifier=make_modifier(JOINT_NAME, "damping"),
    ))
    params.add(sysid.Parameter(
        "frictionloss",
        nominal=init_frictionloss,
        min_value=0.0,
        max_value=3.0,
        modifier=make_modifier(JOINT_NAME, "frictionloss"),
    ))
    params["armature"].value[:] = init_armature
    params["damping"].value[:] = init_damping
    params["frictionloss"].value[:] = init_frictionloss
    return params


def make_sequence_inputs(
    seq: Sequence,
    model: mujoco.MjModel,
) -> tuple[np.ndarray, sysid.TimeSeries, sysid.TimeSeries]:
    """Build (initial_state, control_ts, sensor_ts) for one sequence."""
    data = mujoco.MjData(model)
    data.qpos[0] = seq.position[0]
    data.qvel[0] = seq.velocity[0]
    initial_state = sysid.create_initial_state(
        model, data.qpos, data.qvel, data.act
    )
    control_ts = sysid.TimeSeries(
        seq.times, seq.ctrl_torque.reshape(-1, 1).astype(np.float64),
    )
    sensor_array = np.column_stack(
        [seq.position.astype(np.float64), seq.velocity.astype(np.float64)]
    )
    sensor_ts = sysid.TimeSeries.from_names(seq.times, sensor_array, model)
    return initial_state, control_ts, sensor_ts


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def discover_recordings(recordings_dir: Path) -> list[Path]:
    """Return all *.mcap files in `recordings_dir`, sorted alphabetically."""
    if not recordings_dir.is_dir():
        raise SystemExit(
            f"--recordings must be a directory, got {recordings_dir}"
        )
    paths = sorted(recordings_dir.glob("*.mcap"))
    if not paths:
        raise SystemExit(
            f"No *.mcap files found in {recordings_dir}. "
            "Did you run scripts/collect.py?"
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings", type=Path, required=True,
        help="Directory containing *.mcap files from collect.py. Every MCAP "
             "in the directory becomes a separate sequence in the fit.",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory to write results.json and report.html.",
    )
    parser.add_argument(
        "--init-armature", type=float, default=1e-3,
    )
    parser.add_argument(
        "--init-damping", type=float, default=0.05,
    )
    parser.add_argument(
        "--init-frictionloss", type=float, default=0.1,
    )
    parser.add_argument(
        "--no-frictionloss", action="store_true",
        help="Freeze frictionloss at the initial value (helpful when you only "
             "want to identify armature + damping for a smooth-bearing case).",
    )
    parser.add_argument(
        "--no-velocity-sensor", action="store_true",
        help="Drop the jointvel sensor; identify from position only.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    args = parser.parse_args()

    paths = discover_recordings(args.recordings)
    print(f"Found {len(paths)} recording(s) in {args.recordings}:")
    for p in paths:
        print(f"  {p.name}")

    raw_seqs = [load_sequence(p) for p in paths]

    # Use the median empirical rate across recordings as the common simulation
    # timestep; reject only if any recording diverges by more than 5% from it
    # (a likely sign of mis-configured collection, not jitter).
    rates = np.array([s.sampling_rate for s in raw_seqs])
    sampling_rate = float(np.median(rates))
    dt = 1.0 / sampling_rate
    for s, p in zip(raw_seqs, paths):
        if abs(s.sampling_rate - sampling_rate) / sampling_rate > 0.05:
            raise SystemExit(
                f"{p}: empirical rate {s.sampling_rate:.1f} Hz differs from "
                f"the median {sampling_rate:.1f} Hz by more than 5%. "
                "Re-record at a consistent rate."
            )
    print(f"Simulation timestep: dt = {dt*1000:.3f} ms ({sampling_rate:.2f} Hz)")
    seqs = [to_uniform_grid(s, dt) for s in raw_seqs]

    spec = make_spec(dt)
    if args.no_velocity_sensor:
        spec.delete(spec.sensor("velocity"))
    model = spec.compile()

    init_states, control_tss, sensor_tss = [], [], []
    for seq in seqs:
        s0, c_ts, y_ts = make_sequence_inputs(seq, model)
        init_states.append(s0)
        control_tss.append(c_ts)
        sensor_tss.append(y_ts)

    ms = sysid.ModelSequences(
        "rs02",
        spec,
        [s.name for s in seqs],
        init_states,
        control_tss,
        sensor_tss,
    )

    params = build_initial_params(
        args.init_armature, args.init_damping, args.init_frictionloss,
    )
    if args.no_frictionloss:
        params["frictionloss"].frozen = True

    residual_fn = sysid.build_residual_fn(models_sequences=[ms])

    print("Running optimizer...")
    opt_params, opt_result = sysid.optimize(
        initial_params=params,
        residual_fn=residual_fn,
        optimizer="mujoco",
        verbose=args.verbose,
    )

    summary = {
        name: float(opt_params[name].value[0])
        for name in ("armature", "damping", "frictionloss")
    }
    print("\nIdentified parameters:")
    for k, v in summary.items():
        frozen = " (frozen)" if params[k].frozen else ""
        print(f"  {k:>14s} = {v:.6f}{frozen}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_json = args.out_dir / "results.json"
    with results_json.open("w") as f:
        json.dump(
            {
                "armature": summary["armature"],
                "damping": summary["damping"],
                "frictionloss": summary["frictionloss"],
                "initial": {
                    "armature": args.init_armature,
                    "damping": args.init_damping,
                    "frictionloss": args.init_frictionloss,
                },
                "frozen": {
                    "armature": False,
                    "damping": False,
                    "frictionloss": bool(args.no_frictionloss),
                },
                "recordings": [p.name for p in paths],
                "recordings_dir": str(args.recordings),
                "sampling_rate": sampling_rate,
            },
            f,
            indent=2,
        )
    print(f"\nWrote params to {results_json}")

    print("Building report...")
    report = sysid.default_report(
        models_sequences=[ms],
        initial_params=build_initial_params(
            args.init_armature, args.init_damping, args.init_frictionloss,
        ),
        opt_params=opt_params,
        residual_fn=residual_fn,
        opt_result=opt_result,
        title="RS02 actuator system identification",
        generate_videos=False,
    )
    report_html = args.out_dir / "report.html"
    report_html.write_text(report.build())
    print(f"Wrote report to {report_html}")


if __name__ == "__main__":
    # silence: rollout import is needed for sysid even if not referenced directly
    _ = rollout
    main()
