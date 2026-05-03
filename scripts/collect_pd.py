"""
Drive an unloaded Robstride actuator with a position-target excitation
through the firmware's built-in MIT-mode PD controller.

Each loop iteration sends:
    write_mit_control(position=p_des[i], velocity=0,
                      kp=KP, kd=KD,
                      torque=0)

The firmware applies tau = KP*(p_des - p_meas) + KD*(0 - v_meas) and that
closed-loop torque drives the actuator. The kp_cmd / kd_cmd values are
stored in the MCAP metadata so `optimize.py` can reconstruct the equivalent
torque (see `recording._equivalent_torque`) and feed it to the same
single-`<motor>` MuJoCo model used for torque-mode recordings.

Each invocation runs two excitations back-to-back:
  * multisine position target at `--amplitude` rad -> `<out-dir>/multisine.mcap`
  * linear chirp position target at `--amplitude` rad -> `<out-dir>/chirp.mcap`

Usage:
    python scripts/collect_pd.py --channel can0 --id 1 -o data/run1_pd \
        --kp 10.0 --kd 0.5 --amplitude 0.5
"""



import argparse
import time
from pathlib import Path

import numpy as np

from recording import write_mcap
from signals import make_chirp, make_multisine
from streaming import (
    close_bus, open_bus, stream, wait_for_first_state,
)


def _record(
    bus,
    label: str,
    p_des: np.ndarray,
    rate_hz: float,
    amplitude: float,
    kp: float,
    kd: float,
    out_path: Path,
    common_metadata: dict,
) -> None:
    print(
        f"\n[{label}] duration={len(p_des) / rate_hz:.1f}s "
        f"peak target={float(np.max(np.abs(p_des))):.3f} rad  "
        f"kp={kp} kd={kd}"
    )
    pos0 = wait_for_first_state(bus)
    # Targets are recorded relative to pos0; the firmware needs them in its
    # own absolute frame, so shift by pos0 in the command callback.
    result = stream(
        bus, len(p_des), rate_hz,
        command_fn=lambda i: dict(
            position=pos0 + float(p_des[i]),
            velocity=0.0,
            kp=float(kp), kd=float(kd),
            torque=0.0,
        ),
        pos0=pos0,
    )
    n = len(result["times"])
    write_mcap(
        out_path,
        times=result["times"],
        ctrl_torque=np.zeros(n, dtype=np.float64),  # no feed-forward in pure PD
        position_target=p_des[:n],
        position=result["position"],
        velocity=result["velocity"],
        measured_torque=result["measured_torque"],
        metadata={
            **common_metadata,
            "control_mode": "pd",
            "signal_type": label,
            "amplitude": float(amplitude),
            "kp_cmd": float(kp),
            "kd_cmd": float(kd),
            "aborted": bool(result["aborted"]),
        },
    )
    dur = result["times"][-1] if n else 0.0
    suffix = "  [ABORTED]" if result["aborted"] else ""
    print(f"[{label}] saved {n} samples ({dur:.2f}s) to {out_path}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0", help="SocketCAN channel")
    parser.add_argument("--id", type=int, default=1, help="Robstride CAN ID")
    parser.add_argument(
        "--model", default="rs-02",
        help="Robstride model string (rs-02, rs-00, ...)",
    )
    parser.add_argument(
        "--duration", type=float, default=20.0, help="seconds per signal"
    )
    parser.add_argument("--rate", type=float, default=400.0, help="Hz")
    parser.add_argument(
        "--amplitude", type=float, default=0.5,
        help="peak position-target amplitude in rad",
    )
    parser.add_argument(
        "--kp", type=float, default=10.0,
        help="firmware MIT kp (N.m/rad)",
    )
    parser.add_argument(
        "--kd", type=float, default=1.0,
        help="firmware MIT kd (N.m.s/rad)",
    )
    parser.add_argument(
        "--freqs", type=float, nargs="+",
        default=[2.0, 5.0, 11.0, 19.0, 29.0],
        help="multisine component frequencies (Hz)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rest", type=float, default=2.0,
        help="seconds to coast at zero torque between excitations",
    )
    parser.add_argument(
        "--out-dir", "-o", type=Path, default=Path("data/recording_pd"),
        help="directory to write multisine.mcap and chirp.mcap",
    )
    args = parser.parse_args()

    _, p_multi = make_multisine(
        args.duration, args.rate, args.amplitude,
        tuple(args.freqs), args.seed,
    )
    _, p_chirp = make_chirp(args.duration, args.rate, args.amplitude)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common_metadata = {
        "sampling_rate": float(args.rate),
        "actuator_id": int(args.id),
        "actuator_model": str(args.model),
    }

    bus = open_bus(args.channel, args.id, args.model)
    try:
        _record(
            bus, "multisine", p_multi, args.rate, args.amplitude,
            args.kp, args.kd,
            args.out_dir / "multisine.mcap", common_metadata,
        )
        print(f"\nresting {args.rest:.1f}s at zero torque...")
        time.sleep(args.rest)
        _record(
            bus, "chirp", p_chirp, args.rate, args.amplitude,
            args.kp, args.kd,
            args.out_dir / "chirp.mcap", common_metadata,
        )
    finally:
        close_bus(bus)


if __name__ == "__main__":
    main()
