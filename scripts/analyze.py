"""
Analyze Robstride actuator recordings: fit MuJoCo joint parameters and
plot per-recording fits.

Two modes:
  * ``--model rs-02`` iterates every ``data/rs-02/run<N>/`` and writes
    ``results/rs-02/run<N>/{results.json, report.html, fit_*.png}``.
  * ``--recordings DIR --out-dir DIR`` runs once on a single explicit pair;
    used by the sim test pipeline.

Each run runs the optimizer, writes results.json + report.html, then
forward-rolls the identified params and writes one fit_<stem>.png per
recording. Pass ``--no-plots`` to skip the plotting step.

Layout:
    data/<model>/run<N>/<signal>.mcap
    results/<model>/run<N>/results.json    fitted params
    results/<model>/run<N>/report.html     mujoco.sysid HTML report
    results/<model>/run<N>/fit_<stem>.png  per-recording overlay plots

Usage:
    python scripts/analyze.py --model rs-02
    python scripts/analyze.py --recordings data/sim/ --out-dir results/sim/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.rollout as rollout
import numpy as np
from mujoco import sysid

from model import JOINT_NAME, make_spec
from recording import list_run_dirs, load_sequence, resample


# --------------------------------------------------------------------------- #
# Sysid plumbing                                                              #
# --------------------------------------------------------------------------- #

def _set_scalar(joint, attr, value):
    setattr(joint, attr, float(value))


def _set_damping(joint, value):
    # Joint damping is stored as a length-3 vector internally (ball-joint
    # capable), but only the first component matters for a 1-DoF hinge.
    joint.damping = np.array([float(value), 0.0, 0.0], dtype=np.float64)


def _make_modifier(attr: str):
    if attr == "damping":
        def _mod(spec, param):
            _set_damping(spec.joint(JOINT_NAME), param.value[0])
    else:
        def _mod(spec, param):
            _set_scalar(spec.joint(JOINT_NAME), attr, param.value[0])
    return _mod


def _build_initial_params(
    init_armature: float,
    init_damping: float,
    init_frictionloss: float,
) -> sysid.ParameterDict:
    params = sysid.ParameterDict()
    params.add(sysid.Parameter(
        "armature", nominal=init_armature,
        min_value=1e-5, max_value=0.1,
        modifier=_make_modifier("armature"),
    ))
    params.add(sysid.Parameter(
        "damping", nominal=init_damping,
        min_value=0.0, max_value=5.0,
        modifier=_make_modifier("damping"),
    ))
    params.add(sysid.Parameter(
        "frictionloss", nominal=init_frictionloss,
        min_value=0.0, max_value=3.0,
        modifier=_make_modifier("frictionloss"),
    ))
    params["armature"].value[:] = init_armature
    params["damping"].value[:] = init_damping
    params["frictionloss"].value[:] = init_frictionloss
    return params


def _make_sequence_inputs(seq, model):
    data = mujoco.MjData(model)
    data.qpos[0] = seq.position[0]
    data.qvel[0] = seq.velocity[0]
    initial_state = sysid.create_initial_state(
        model, data.qpos, data.qvel, data.act,
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
# Stage 1: fit                                                                #
# --------------------------------------------------------------------------- #

def _discover_recordings(recordings_dir: Path) -> list[Path]:
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


def fit_run(
    recordings_dir: Path,
    out_dir: Path,
    *,
    init_armature: float,
    init_damping: float,
    init_frictionloss: float,
    no_frictionloss: bool,
    no_velocity_sensor: bool,
    verbose: bool,
) -> None:
    """Run mujoco.sysid on every *.mcap in `recordings_dir`, write to `out_dir`."""
    paths = _discover_recordings(recordings_dir)
    print(f"Found {len(paths)} recording(s) in {recordings_dir}:")
    raw_seqs = []
    for p in paths:
        s = load_sequence(p)
        print(f"  {s.name}.mcap  rate={s.sampling_rate:.2f} Hz")
        raw_seqs.append(s)

    # Use the median empirical rate across recordings as the common simulation
    # timestep; reject only if any recording diverges by more than 5% from it.
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
    seqs = [resample(s, dt) for s in raw_seqs]

    spec = make_spec(dt)
    if no_velocity_sensor:
        spec.delete(spec.sensor("velocity"))
    model = spec.compile()

    init_states, control_tss, sensor_tss = [], [], []
    for seq in seqs:
        s0, c_ts, y_ts = _make_sequence_inputs(seq, model)
        init_states.append(s0)
        control_tss.append(c_ts)
        sensor_tss.append(y_ts)

    ms = sysid.ModelSequences(
        "actuator", spec, [s.name for s in seqs],
        init_states, control_tss, sensor_tss,
    )

    params = _build_initial_params(init_armature, init_damping, init_frictionloss)
    if no_frictionloss:
        params["frictionloss"].frozen = True

    residual_fn = sysid.build_residual_fn(models_sequences=[ms])

    print("Running optimizer...")
    opt_params, opt_result = sysid.optimize(
        initial_params=params, residual_fn=residual_fn,
        optimizer="mujoco", verbose=verbose,
    )

    summary = {
        name: float(opt_params[name].value[0])
        for name in ("armature", "damping", "frictionloss")
    }
    print("\nIdentified parameters:")
    for k, v in summary.items():
        frozen = " (frozen)" if params[k].frozen else ""
        print(f"  {k:>14s} = {v:.6f}{frozen}")

    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "results.json"
    with results_json.open("w") as f:
        json.dump(
            {
                "armature": summary["armature"],
                "damping": summary["damping"],
                "frictionloss": summary["frictionloss"],
                "initial": {
                    "armature": init_armature,
                    "damping": init_damping,
                    "frictionloss": init_frictionloss,
                },
                "frozen": {
                    "armature": False,
                    "damping": False,
                    "frictionloss": bool(no_frictionloss),
                },
                "recordings": [p.name for p in paths],
                "recordings_dir": str(recordings_dir),
                "sampling_rate": sampling_rate,
            },
            f, indent=2,
        )
    print(f"Wrote params to {results_json}")

    print("Building report...")
    report = sysid.default_report(
        models_sequences=[ms],
        initial_params=_build_initial_params(
            init_armature, init_damping, init_frictionloss,
        ),
        opt_params=opt_params,
        residual_fn=residual_fn,
        opt_result=opt_result,
        title=f"Actuator sysid: {recordings_dir.name}",
        generate_videos=False,
    )
    report_html = out_dir / "report.html"
    report_html.write_text(report.build())
    print(f"Wrote report to {report_html}")


# --------------------------------------------------------------------------- #
# Stage 2: plot                                                               #
# --------------------------------------------------------------------------- #

def _simulate(
    armature: float,
    damping: float,
    frictionloss: float,
    dt: float,
    ctrl_torque: np.ndarray,
    qpos0: float,
    qvel0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out MuJoCo with the given params; return (position, velocity)."""
    spec = make_spec(dt)
    j = spec.joint(JOINT_NAME)
    j.armature = float(armature)
    j.damping = np.array([float(damping), 0.0, 0.0], dtype=np.float64)
    j.frictionloss = float(frictionloss)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[0] = qpos0
    data.qvel[0] = qvel0

    nu = model.nu
    n = len(ctrl_torque)
    ctrl = ctrl_torque.reshape(-1, nu).astype(np.float64)

    initial_state = np.zeros(
        (1, mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value))
    )
    mujoco.mj_getState(
        model, data, initial_state[0],
        mujoco.mjtState.mjSTATE_FULLPHYSICS.value,
    )

    _, sensor = rollout.rollout(
        model, data, initial_state, ctrl[:-1].reshape(1, n - 1, nu),
    )
    sensor = np.squeeze(sensor, axis=0)
    return sensor[:, 0], sensor[:, 1]


def _plot_fit(
    out_path: Path,
    seq,
    params: dict,
    pos_init: np.ndarray,
    vel_init: np.ndarray,
    pos_opt: np.ndarray,
    vel_opt: np.ndarray,
) -> dict:
    """Render the 3-pane plot for one sequence; return its RMSEs."""
    n = min(len(seq.times), len(pos_init), len(pos_opt))
    times = seq.times[:n]
    ctrl = seq.ctrl_torque[:n]
    pos_meas = seq.position[:n]
    vel_meas = seq.velocity[:n]
    pos_init = pos_init[:n]; vel_init = vel_init[:n]
    pos_opt = pos_opt[:n]; vel_opt = vel_opt[:n]

    rmse = {
        "pos_init": float(np.sqrt(np.mean((pos_init - pos_meas) ** 2))),
        "pos_opt": float(np.sqrt(np.mean((pos_opt - pos_meas) ** 2))),
        "vel_init": float(np.sqrt(np.mean((vel_init - vel_meas) ** 2))),
        "vel_opt": float(np.sqrt(np.mean((vel_opt - vel_meas) ** 2))),
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(times, pos_meas, color="0.2", lw=1.2, label="measured")
    axes[0].plot(times, pos_init, color="C3", lw=1.0, alpha=0.8,
                 label=f"initial (RMSE {rmse['pos_init']:.3f})")
    axes[0].plot(times, pos_opt, color="C0", lw=1.0,
                 label=f"identified (RMSE {rmse['pos_opt']:.3f})")
    axes[0].set_ylabel("position (rad)")
    axes[0].legend(loc="best", fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(times, vel_meas, color="0.2", lw=1.2, label="measured")
    axes[1].plot(times, vel_init, color="C3", lw=1.0, alpha=0.8,
                 label=f"initial (RMSE {rmse['vel_init']:.3f})")
    axes[1].plot(times, vel_opt, color="C0", lw=1.0,
                 label=f"identified (RMSE {rmse['vel_opt']:.3f})")
    axes[1].set_ylabel("velocity (rad/s)")
    axes[1].legend(loc="best", fontsize=9); axes[1].grid(alpha=0.3)

    axes[2].plot(times, ctrl, color="0.4", lw=0.9)
    axes[2].set_ylabel("ctrl torque (N.m)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"{seq.name}: armature={params['armature']:.4g}, "
        f"damping={params['damping']:.4g}, "
        f"frictionloss={params['frictionloss']:.4g}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return rmse


def plot_run(recordings_dir: Path, out_dir: Path) -> None:
    """Plot every *.mcap in `recordings_dir` against `out_dir/results.json`."""
    paths = sorted(recordings_dir.glob("*.mcap"))
    if not paths:
        return
    results_json = out_dir / "results.json"
    if not results_json.exists():
        raise SystemExit(
            f"{results_json} not found; fit_run() must run first."
        )
    params = json.loads(results_json.read_text())
    init = params["initial"]

    print(f"\nPlotting {len(paths)} recording(s) -> {out_dir}/")
    print(f"{'recording':>20s}  {'pos init':>9s}  {'pos opt':>9s}  "
          f"{'vel init':>9s}  {'vel opt':>9s}")
    print("-" * 72)
    for p in paths:
        raw = load_sequence(p)
        seq = resample(raw, 1.0 / raw.sampling_rate)
        dt = 1.0 / seq.sampling_rate
        pos_init, vel_init = _simulate(
            init["armature"], init["damping"], init["frictionloss"],
            dt, seq.ctrl_torque,
            qpos0=seq.position[0], qvel0=seq.velocity[0],
        )
        pos_opt, vel_opt = _simulate(
            params["armature"], params["damping"], params["frictionloss"],
            dt, seq.ctrl_torque,
            qpos0=seq.position[0], qvel0=seq.velocity[0],
        )
        rmse = _plot_fit(
            out_dir / f"fit_{p.stem}.png",
            seq, params,
            pos_init, vel_init, pos_opt, vel_opt,
        )
        print(f"{seq.name:>20s}  "
              f"{rmse['pos_init']:>9.4f}  {rmse['pos_opt']:>9.4f}  "
              f"{rmse['vel_init']:>9.4f}  {rmse['vel_opt']:>9.4f}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="Iterate every data/<model>/run<N>/ and write to "
             "results/<model>/run<N>/. Mutually exclusive with --recordings.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"),
        help="Root for data/<model>/ when --model is used.",
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results"),
        help="Root for results/<model>/ when --model is used.",
    )
    parser.add_argument(
        "--recordings", type=Path,
        help="Single run directory (overrides --model).",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        help="Single output directory; required with --recordings.",
    )
    parser.add_argument("--init-armature", type=float, default=1e-3)
    parser.add_argument("--init-damping", type=float, default=0.05)
    parser.add_argument("--init-frictionloss", type=float, default=0.1)
    parser.add_argument(
        "--no-frictionloss", action="store_true",
        help="Freeze frictionloss at the initial value.",
    )
    parser.add_argument(
        "--no-velocity-sensor", action="store_true",
        help="Drop the jointvel sensor; identify from position only.",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip the per-recording fit plots.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.recordings is not None:
        if args.model is not None:
            raise SystemExit("--model and --recordings are mutually exclusive")
        if args.out_dir is None:
            raise SystemExit("--out-dir is required when --recordings is set")
        pairs = [(args.recordings, args.out_dir)]
    elif args.model is not None:
        run_dirs = list_run_dirs(args.data_root / args.model)
        pairs = [
            (d, args.results_root / args.model / d.name) for d in run_dirs
        ]
    else:
        raise SystemExit("must specify --model or --recordings")

    for i, (rec_dir, out_dir) in enumerate(pairs, start=1):
        if len(pairs) > 1:
            print(f"\n========== run {i}/{len(pairs)}: {rec_dir.name} ==========")
        fit_run(
            rec_dir, out_dir,
            init_armature=args.init_armature,
            init_damping=args.init_damping,
            init_frictionloss=args.init_frictionloss,
            no_frictionloss=args.no_frictionloss,
            no_velocity_sensor=args.no_velocity_sensor,
            verbose=args.verbose,
        )
        if not args.no_plots:
            plot_run(rec_dir, out_dir)


if __name__ == "__main__":
    main()
