"""
Plot measured vs. simulated (initial-guess and identified) trajectories for
every recording in a session.

Reads `out-dir/results.json` (produced by `optimize.py`) for the identified
and initial parameters, then for every `*.mcap` in `recordings`:
  * forward-rolls MuJoCo with both parameter sets
  * overlays measured / initial / identified position and velocity, plus the
    feed-forward torque, and writes `out-dir/fit_<stem>.png`

A summary table at the end aggregates the per-recording RMSEs.

Usage:
    python scripts/visualize.py \
        --recordings data/rs-02/run1/ \
        --out-dir results/rs-02/run1/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import mujoco.rollout as rollout
import numpy as np

from model import JOINT_NAME, make_spec
from recording import read_mcap


def load_recording(path: Path) -> dict:
    """Load an MCAP and derive its empirical sampling rate from timestamps,
    matching the convention in optimize.py."""
    rec = read_mcap(path)
    if len(rec.times) < 2:
        raise ValueError(f"{path}: need >= 2 samples")
    sampling_rate = 1.0 / float(np.median(np.diff(rec.times)))
    return {
        "times": rec.times,
        "ctrl_torque": rec.ctrl_torque,
        "position": rec.position,
        "velocity": rec.velocity,
        "sampling_rate": sampling_rate,
        "metadata": rec.metadata,
    }


def to_uniform(times, *signals, dt):
    """Resample irregularly-sampled signals onto t = k*dt grid."""
    t0 = float(times[0])
    n = int(np.floor((times[-1] - t0) / dt)) + 1
    t_new = t0 + dt * np.arange(n)
    out = [t_new - t0]
    for s in signals:
        out.append(np.interp(t_new, times, s))
    return out


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
        model, data, initial_state[0], mujoco.mjtState.mjSTATE_FULLPHYSICS.value,
    )

    state, sensor = rollout.rollout(
        model, data, initial_state, ctrl[:-1].reshape(1, n - 1, nu),
    )
    sensor = np.squeeze(sensor, axis=0)
    return sensor[:, 0], sensor[:, 1]


def plot_one(
    out_path: Path,
    label: str,
    params: dict,
    times: np.ndarray,
    ctrl: np.ndarray,
    pos_meas: np.ndarray,
    vel_meas: np.ndarray,
    pos_init: np.ndarray,
    vel_init: np.ndarray,
    pos_opt: np.ndarray,
    vel_opt: np.ndarray,
) -> dict:
    """Render the 3-pane plot for a single sequence and return its RMSEs."""
    n = min(len(times), len(pos_init), len(pos_opt))
    times = times[:n]; ctrl = ctrl[:n]
    pos_meas = pos_meas[:n]; vel_meas = vel_meas[:n]
    pos_init = pos_init[:n]; vel_init = vel_init[:n]
    pos_opt = pos_opt[:n]; vel_opt = vel_opt[:n]

    rmse_pos_init = float(np.sqrt(np.mean((pos_init - pos_meas) ** 2)))
    rmse_pos_opt = float(np.sqrt(np.mean((pos_opt - pos_meas) ** 2)))
    rmse_vel_init = float(np.sqrt(np.mean((vel_init - vel_meas) ** 2)))
    rmse_vel_opt = float(np.sqrt(np.mean((vel_opt - vel_meas) ** 2)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(times, pos_meas, color="0.2", lw=1.2, label="measured")
    axes[0].plot(times, pos_init, color="C3", lw=1.0, alpha=0.8,
                 label=f"initial (RMSE {rmse_pos_init:.3f})")
    axes[0].plot(times, pos_opt, color="C0", lw=1.0,
                 label=f"identified (RMSE {rmse_pos_opt:.3f})")
    axes[0].set_ylabel("position (rad)")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, vel_meas, color="0.2", lw=1.2, label="measured")
    axes[1].plot(times, vel_init, color="C3", lw=1.0, alpha=0.8,
                 label=f"initial (RMSE {rmse_vel_init:.3f})")
    axes[1].plot(times, vel_opt, color="C0", lw=1.0,
                 label=f"identified (RMSE {rmse_vel_opt:.3f})")
    axes[1].set_ylabel("velocity (rad/s)")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].plot(times, ctrl, color="0.4", lw=0.9)
    axes[2].set_ylabel("ctrl torque (N.m)")
    axes[2].set_xlabel("time (s)")
    axes[2].grid(alpha=0.3)

    fig.suptitle(
        f"{label}: armature={params['armature']:.4g}, "
        f"damping={params['damping']:.4g}, "
        f"frictionloss={params['frictionloss']:.4g}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {
        "pos_init": rmse_pos_init, "pos_opt": rmse_pos_opt,
        "vel_init": rmse_vel_init, "vel_opt": rmse_vel_opt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recordings", type=Path, required=True,
        help="Directory of *.mcap recordings (matches optimize.py).",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Directory containing results.json from optimize.py; "
             "fit_<stem>.png images are written here too.",
    )
    args = parser.parse_args()

    if not args.recordings.is_dir():
        raise SystemExit(f"--recordings must be a directory, got {args.recordings}")
    paths = sorted(args.recordings.glob("*.mcap"))
    if not paths:
        raise SystemExit(f"No *.mcap files found in {args.recordings}")

    results_json = args.out_dir / "results.json"
    if not results_json.exists():
        raise SystemExit(
            f"{results_json} not found; run scripts/optimize.py first."
        )
    params = json.loads(results_json.read_text())
    init = params["initial"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting {len(paths)} recording(s) -> {args.out_dir}/")
    print(
        f"Identified params: armature={params['armature']:.6f}  "
        f"damping={params['damping']:.6f}  "
        f"frictionloss={params['frictionloss']:.6f}"
    )
    print()
    print(f"{'recording':>20s}  {'pos init':>10s}  {'pos opt':>10s}  "
          f"{'vel init':>10s}  {'vel opt':>10s}")
    print("-" * 72)

    for p in paths:
        rec = load_recording(p)
        dt = 1.0 / rec["sampling_rate"]
        times, ctrl, pos_meas, vel_meas = to_uniform(
            rec["times"], rec["ctrl_torque"], rec["position"], rec["velocity"],
            dt=dt,
        )
        pos_init, vel_init = simulate(
            init["armature"], init["damping"], init["frictionloss"],
            dt, ctrl, qpos0=pos_meas[0], qvel0=vel_meas[0],
        )
        pos_opt, vel_opt = simulate(
            params["armature"], params["damping"], params["frictionloss"],
            dt, ctrl, qpos0=pos_meas[0], qvel0=vel_meas[0],
        )
        out_png = args.out_dir / f"fit_{p.stem}.png"
        rmses = plot_one(
            out_png, p.stem, params, times, ctrl,
            pos_meas, vel_meas, pos_init, vel_init, pos_opt, vel_opt,
        )
        print(
            f"{p.stem:>20s}  {rmses['pos_init']:>10.4f}  "
            f"{rmses['pos_opt']:>10.4f}  {rmses['vel_init']:>10.4f}  "
            f"{rmses['vel_opt']:>10.4f}"
        )

    print()
    print(f"Wrote {len(paths)} fit plots to {args.out_dir}/")


if __name__ == "__main__":
    main()
