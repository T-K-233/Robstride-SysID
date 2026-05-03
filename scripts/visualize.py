"""
Plot measured vs. simulated (initial-guess and identified) trajectories for
every recording in a session.

Reads `out-dir/results.json` (produced by `optimize.py`) for the identified
and initial parameters, then for every `*.mcap` in `recordings`:
  * forward-rolls MuJoCo with both parameter sets
  * overlays measured / initial / identified position and velocity, plus the
    feed-forward (or PD-equivalent) torque, and writes `out-dir/fit_<stem>.png`

A summary table at the end aggregates the per-recording RMSEs.

Two modes (matching `optimize.py`):
  * ``--model rs-02`` iterates every ``data/rs-02/run<N>/`` and writes
    ``results/rs-02/run<N>/fit_<stem>.png``.
  * ``--recordings DIR --out-dir DIR`` runs once on a single explicit pair;
    used by the sim test pipeline.

Usage:
    python scripts/visualize.py --model rs-02
"""



import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.rollout as rollout
import numpy as np

from model import JOINT_NAME, make_spec
from recording import Sequence, list_run_dirs, load_sequence, resample


def simulate(
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


def plot_fit(
    out_path: Path,
    seq: Sequence,
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
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, vel_meas, color="0.2", lw=1.2, label="measured")
    axes[1].plot(times, vel_init, color="C3", lw=1.0, alpha=0.8,
                 label=f"initial (RMSE {rmse['vel_init']:.3f})")
    axes[1].plot(times, vel_opt, color="C0", lw=1.0,
                 label=f"identified (RMSE {rmse['vel_opt']:.3f})")
    axes[1].set_ylabel("velocity (rad/s)")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(times, ctrl, color="0.4", lw=0.9)
    axes[2].set_ylabel("ctrl torque (N.m)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"{seq.name}: "
        f"armature={params['armature']:.4g}, "
        f"damping={params['damping']:.4g}, "
        f"frictionloss={params['frictionloss']:.4g}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return rmse


def visualize_run(recordings_dir: Path, out_dir: Path) -> None:
    """Plot every *.mcap in `recordings_dir` against `out_dir/results.json`."""
    if not recordings_dir.is_dir():
        raise SystemExit(f"{recordings_dir} is not a directory")
    paths = sorted(recordings_dir.glob("*.mcap"))
    if not paths:
        raise SystemExit(f"no *.mcap files in {recordings_dir}")

    results_json = out_dir / "results.json"
    if not results_json.exists():
        raise SystemExit(
            f"{results_json} not found; run scripts/optimize.py first."
        )
    params = json.loads(results_json.read_text())
    init = params["initial"]

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting {len(paths)} recording(s) -> {out_dir}/")
    print(
        f"Identified params: armature={params['armature']:.6f}  "
        f"damping={params['damping']:.6f}  "
        f"frictionloss={params['frictionloss']:.6f}"
    )
    print()
    print(f"{'recording':>20s}  "
          f"{'pos init':>9s}  {'pos opt':>9s}  "
          f"{'vel init':>9s}  {'vel opt':>9s}")
    print("-" * 72)

    for p in paths:
        raw = load_sequence(p)
        seq = resample(raw, 1.0 / raw.sampling_rate)
        dt = 1.0 / seq.sampling_rate
        pos_init, vel_init = simulate(
            init["armature"], init["damping"], init["frictionloss"],
            dt, seq.ctrl_torque,
            qpos0=seq.position[0], qvel0=seq.velocity[0],
        )
        pos_opt, vel_opt = simulate(
            params["armature"], params["damping"], params["frictionloss"],
            dt, seq.ctrl_torque,
            qpos0=seq.position[0], qvel0=seq.velocity[0],
        )
        rmse = plot_fit(
            out_dir / f"fit_{p.stem}.png",
            seq, params,
            pos_init, vel_init, pos_opt, vel_opt,
        )
        print(
            f"{seq.name:>20s}  "
            f"{rmse['pos_init']:>9.4f}  {rmse['pos_opt']:>9.4f}  "
            f"{rmse['vel_init']:>9.4f}  {rmse['vel_opt']:>9.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        help="Iterate every data/<model>/run<N>/ and write plots to "
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
        help="Single run directory (overrides --model). Used by the sim test.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        help="Single output directory; required with --recordings.",
    )
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
        visualize_run(rec_dir, out_dir)


if __name__ == "__main__":
    main()
