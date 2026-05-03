"""
Drive an unloaded Robstride actuator with a feed-forward torque excitation.

The actuator is run in MIT mode with kp=kd=0, so the firmware applies
exactly the torque we send -- the same signal that becomes the MuJoCo
`<motor>` actuator's `ctrl` during optimization.

Each invocation runs two excitations back-to-back:
  * multisine at `--amplitude` -> `<out-dir>/multisine.mcap`
  * linear chirp at `--amplitude` -> `<out-dir>/chirp.mcap`

Usage:
    python scripts/collect_torque.py --channel can0 --id 1 -o data/run1
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
    tau: np.ndarray,
    rate_hz: float,
    amplitude: float,
    out_path: Path,
    common_metadata: dict,
) -> None:
    print(
        f"\n[{label}] duration={len(tau) / rate_hz:.1f}s "
        f"peak={float(np.max(np.abs(tau))):.2f} N.m"
    )
    pos0 = wait_for_first_state(bus)
    result = stream(
        bus, len(tau), rate_hz,
        command_fn=lambda i: dict(
            position=0.0, velocity=0.0,
            kp=0.0, kd=0.0,
            torque=float(tau[i]),
        ),
        pos0=pos0,
    )
    n = len(result["times"])
    write_mcap(
        out_path,
        times=result["times"],
        ctrl_torque=tau[:n],
        position_target=np.zeros(n, dtype=np.float64),
        position=result["position"],
        velocity=result["velocity"],
        measured_torque=result["measured_torque"],
        metadata={
            **common_metadata,
            "control_mode": "torque",
            "signal_type": label,
            "amplitude": float(amplitude),
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
        "--amplitude", type=float, default=1.0,
        help="peak torque amplitude in N.m (well below 17 N.m peak)",
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
        "--out-dir", "-o", type=Path, default=Path("data/recording_torque"),
        help="directory to write multisine.mcap and chirp.mcap",
    )
    args = parser.parse_args()

    _, tau_multi = make_multisine(
        args.duration, args.rate, args.amplitude,
        tuple(args.freqs), args.seed,
    )
    _, tau_chirp = make_chirp(args.duration, args.rate, args.amplitude)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common_metadata = {
        "sampling_rate": float(args.rate),
        "actuator_id": int(args.id),
        "actuator_model": str(args.model),
    }

    bus = open_bus(args.channel, args.id, args.model)
    try:
        _record(
            bus, "multisine", tau_multi, args.rate, args.amplitude,
            args.out_dir / "multisine.mcap", common_metadata,
        )
        print(f"\nresting {args.rest:.1f}s at zero torque...")
        time.sleep(args.rest)
        _record(
            bus, "chirp", tau_chirp, args.rate, args.amplitude,
            args.out_dir / "chirp.mcap", common_metadata,
        )
    finally:
        close_bus(bus)


if __name__ == "__main__":
    main()
