"""
Simulate the collect.py experiment using MuJoCo with known parameters.

This validates the collect -> optimize -> visualize pipeline without hardware.
The MuJoCo model is the same one (from `scripts/model.py`) that the optimizer
uses, with the joint's armature/damping/frictionloss overridden to a preset
"ground-truth" set. The same excitation generators (`scripts/signals.py`)
that `collect.py` uses on real hardware are reused here, and the script
emits the same two MCAPs (`multisine.mcap`, `chirp.mcap`) into `--out-dir`.

Each MCAP additionally carries the ground-truth parameter values in its
metadata so `tests/check_recovery.py` can quantify recovery error.

Usage:
    python tests/sim_collect.py --out-dir data/sim
"""



import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.rollout as rollout
import numpy as np
from mujoco import sysid

# allow `from model import ...` etc. when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from model import JOINT_NAME, make_spec  # noqa: E402
from recording import write_mcap  # noqa: E402
from signals import make_chirp, make_multisine  # noqa: E402


# Default ground-truth parameters representative of an unloaded RS02.
TRUE_ARMATURE = 0.0035       # kg.m^2 at the output (rotor + reflected gearbox)
TRUE_DAMPING = 0.18          # N.m.s/rad
TRUE_FRICTIONLOSS = 0.25     # N.m

# Default sensor noise mimicking the 14-bit absolute encoder and the
# firmware's differentiated velocity estimate.
NOISE_POSITION_RAD = 1e-3
NOISE_VELOCITY_RADPS = 5e-2


def simulate_signal(
    rate: float,
    tau: np.ndarray,
    true_armature: float,
    true_damping: float,
    true_frictionloss: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the ground-truth model with `tau`. Returns (times, pos, vel)."""
    spec = make_spec(1.0 / rate)
    j = spec.joint(JOINT_NAME)
    j.armature = float(true_armature)
    j.damping = np.array([float(true_damping), 0.0, 0.0], dtype=np.float64)
    j.frictionloss = float(true_frictionloss)
    model = spec.compile()
    data = mujoco.MjData(model)
    s0 = sysid.create_initial_state(model, data.qpos, data.qvel, data.act)
    state, sensor = rollout.rollout(
        model, data, s0, tau.reshape(-1, 1).astype(np.float64),
    )
    state = np.squeeze(state, axis=0)
    sensor = np.squeeze(sensor, axis=0)
    return state[:, 0], sensor[:, 0], sensor[:, 1]


def write_signal(
    out_path: Path,
    *,
    label: str,
    amplitude: float,
    rate: float,
    times: np.ndarray,
    tau: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    rng: np.random.Generator,
    noise_position: float,
    noise_velocity: float,
    true_armature: float,
    true_damping: float,
    true_frictionloss: float,
) -> int:
    """Add measurement noise and write one MCAP. Returns sample count."""
    n = times.shape[0]
    pos_meas = pos + rng.normal(0.0, noise_position, n)
    vel_meas = vel + rng.normal(0.0, noise_velocity, n)
    write_mcap(
        out_path,
        times=times,
        ctrl_torque=tau[:n],
        position=pos_meas,
        velocity=vel_meas,
        measured_torque=np.zeros(n, dtype=np.float64),
        metadata={
            "sampling_rate": float(rate),
            "signal_type": label,
            "amplitude": float(amplitude),
            "actuator_id": 0,
            "actuator_model": "rs-02",
            "aborted": False,
            # ground-truth metadata (only present in simulated recordings):
            "true_armature": float(true_armature),
            "true_damping": float(true_damping),
            "true_frictionloss": float(true_frictionloss),
            "noise_position": float(noise_position),
            "noise_velocity": float(noise_velocity),
        },
    )
    print(
        f"[{label}] amp={amplitude:.1f} N.m  n={n}  "
        f"pos=[{pos_meas.min():+.2f},{pos_meas.max():+.2f}]  "
        f"vel=[{vel_meas.min():+.2f},{vel_meas.max():+.2f}]  -> {out_path}"
    )
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=400.0)
    parser.add_argument(
        "--amplitude", type=float, default=1.0,
        help="multisine and (low) chirp amplitude (N.m)",
    )
    parser.add_argument(
        "--freqs", type=float, nargs="+",
        default=[2.0, 5.0, 11.0, 19.0, 29.0],
        help="multisine component frequencies (Hz)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--true-armature", type=float, default=TRUE_ARMATURE)
    parser.add_argument("--true-damping", type=float, default=TRUE_DAMPING)
    parser.add_argument("--true-frictionloss", type=float, default=TRUE_FRICTIONLOSS)
    parser.add_argument(
        "--noise-position", type=float, default=NOISE_POSITION_RAD,
        help="Std of Gaussian noise added to position sensor (rad)",
    )
    parser.add_argument(
        "--noise-velocity", type=float, default=NOISE_VELOCITY_RADPS,
        help="Std of Gaussian noise added to velocity sensor (rad/s)",
    )
    parser.add_argument(
        "--out-dir", "-o", type=Path, default=Path("data/sim"),
        help="directory to write multisine.mcap and chirp.mcap",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed + 100)

    print(
        f"TRUE armature={args.true_armature:.4f}  "
        f"damping={args.true_damping:.4f}  "
        f"frictionloss={args.true_frictionloss:.4f}"
    )
    print(f"Writing simulated recordings to {args.out_dir}/")
    print()

    # ---- multisine ----
    _, tau = make_multisine(
        args.duration, args.rate, args.amplitude,
        tuple(args.freqs), args.seed,
    )
    times, pos, vel = simulate_signal(
        args.rate, tau,
        args.true_armature, args.true_damping, args.true_frictionloss,
    )
    write_signal(
        args.out_dir / "multisine.mcap",
        label="multisine", amplitude=args.amplitude, rate=args.rate,
        times=times, tau=tau, pos=pos, vel=vel, rng=rng,
        noise_position=args.noise_position, noise_velocity=args.noise_velocity,
        true_armature=args.true_armature, true_damping=args.true_damping,
        true_frictionloss=args.true_frictionloss,
    )

    # ---- chirp at user amplitude ----
    _, tau = make_chirp(args.duration, args.rate, args.amplitude)
    times, pos, vel = simulate_signal(
        args.rate, tau,
        args.true_armature, args.true_damping, args.true_frictionloss,
    )
    write_signal(
        args.out_dir / "chirp.mcap",
        label="chirp", amplitude=args.amplitude, rate=args.rate,
        times=times, tau=tau, pos=pos, vel=vel, rng=rng,
        noise_position=args.noise_position, noise_velocity=args.noise_velocity,
        true_armature=args.true_armature, true_damping=args.true_damping,
        true_frictionloss=args.true_frictionloss,
    )

if __name__ == "__main__":
    main()
