"""
Collect a torque-excitation trajectory from an unloaded Robstride RS02 for
MuJoCo system identification of armature / damping / frictionloss.

The actuator is driven in MIT mode with kp=kd=0 so the firmware emits a pure
feed-forward torque -- the same signal that will be fed into the MuJoCo `motor`
actuator during optimization. Position, velocity and (firmware-estimated)
torque are read every loop iteration.

Each invocation runs two excitations back-to-back:
  * multisine at `--amplitude` -> `<output-dir>/multisine.mcap`
  * linear chirp at `--amplitude` -> `<output-dir>/chirp.mcap`

Usage:
    python scripts/collect.py --channel can0 --id 1 -o data/run1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from loop_rate_limiters import RateLimiter

from actuator_control import Actuator, RobstrideBus

from recording import write_mcap


# --------------------------------------------------------------------------- #
# Excitation signals                                                          #
# --------------------------------------------------------------------------- #

def make_multisine(
    duration: float,
    rate: float,
    amplitude: float,
    freqs: tuple[float, ...] = (2.0, 5.0, 11.0, 19.0, 29.0),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum of sinusoids with random phases. Zero-mean, peak-bounded.

    Each component has equal amplitude `amplitude / sqrt(len(freqs))` so the
    expected RMS is `amplitude / sqrt(2)` regardless of the number of terms.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * rate))
    t = np.arange(n) / rate
    per_amp = amplitude / np.sqrt(len(freqs))
    phases = rng.uniform(0.0, 2 * np.pi, size=len(freqs))
    tau = np.zeros(n, dtype=np.float64)
    for f, ph in zip(freqs, phases):
        tau += per_amp * np.sin(2 * np.pi * f * t + ph)
    # smooth onset/offset envelope (Tukey-style) to avoid impulsive edges
    win = _tukey(n, alpha=0.1)
    tau *= win
    return t, tau


def make_chirp(
    duration: float,
    rate: float,
    amplitude: float,
    f0: float = 1.0,
    f1: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-frequency-sweep torque. Useful for broadband excitation."""
    n = int(round(duration * rate))
    t = np.arange(n) / rate
    # phase of a linear chirp: 2*pi*(f0*t + 0.5*k*t^2), k = (f1-f0)/duration
    k = (f1 - f0) / duration
    tau = amplitude * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    tau *= _tukey(n, alpha=0.1)
    return t, tau


def _tukey(n: int, alpha: float) -> np.ndarray:
    """Tukey (tapered cosine) window."""
    if alpha <= 0.0:
        return np.ones(n)
    if alpha >= 1.0:
        return np.hanning(n)
    w = np.ones(n)
    edge = int(alpha * (n - 1) / 2.0)
    if edge > 0:
        ramp = 0.5 * (1 + np.cos(np.pi * (np.arange(edge) / edge - 1)))
        w[:edge] = ramp
        w[-edge:] = ramp[::-1]
    return w


# --------------------------------------------------------------------------- #
# Main collection loop                                                        #
# --------------------------------------------------------------------------- #

def collect(
    bus: RobstrideBus,
    actuator_name: str,
    torque_cmd: np.ndarray,
    rate_hz: float,
) -> dict:
    """Stream torque commands and log feedback. Returns per-step arrays."""
    n = len(torque_cmd)
    times = np.zeros(n, dtype=np.float64)
    cmd = np.zeros(n, dtype=np.float64)
    pos = np.zeros(n, dtype=np.float64)
    vel = np.zeros(n, dtype=np.float64)
    tau_meas = np.zeros(n, dtype=np.float64)

    # Wait for first feedback frame so position[0] is meaningful.
    deadline = time.perf_counter() + 1.0
    while bus.get_state(actuator_name) is None:
        if time.perf_counter() > deadline:
            raise RuntimeError(
                f"No state received from actuator {actuator_name!r} after 1 s. "
                "Check CAN wiring and that the motor is powered."
            )
        time.sleep(0.01)

    state0 = bus.get_state(actuator_name)
    pos0 = state0.position

    rate = RateLimiter(frequency=rate_hz)
    t0 = time.perf_counter()
    aborted = False
    last_idx = n
    try:
        for i in range(n):
            tau = float(torque_cmd[i])
            bus.write_mit_control(
                actuator=actuator_name,
                position=0.0,
                velocity=0.0,
                kp=0.0,
                kd=0.0,
                torque=tau,
            )
            state = bus.get_state(actuator_name)
            if state is None:
                raise RuntimeError("Lost actuator state mid-run.")

            times[i] = time.perf_counter() - t0
            cmd[i] = tau
            pos[i] = state.position - pos0
            vel[i] = state.velocity
            tau_meas[i] = state.torque

            if state.faults:
                print(f"\n[!] fault reported: {state.faults}")
                aborted = True
                last_idx = i + 1
                break

            rate.sleep()
    finally:
        # send a zero-torque command before returning; caller will disable.
        bus.write_mit_control(
            actuator=actuator_name,
            position=0.0,
            velocity=0.0,
            kp=0.0,
            kd=0.0,
            torque=0.0,
        )

    return {
        "times": times[:last_idx],
        "ctrl_torque": cmd[:last_idx],
        "position": pos[:last_idx],
        "velocity": vel[:last_idx],
        "measured_torque": tau_meas[:last_idx],
        "aborted": aborted,
    }


def _record(
    bus: RobstrideBus,
    actuator_name: str,
    label: str,
    tau: np.ndarray,
    rate_hz: float,
    amplitude: float,
    output_path: Path,
    common_metadata: dict,
) -> None:
    """Run one excitation, save its MCAP, return."""
    print(
        f"\n[{label}] duration={len(tau) / rate_hz:.1f}s "
        f"peak={float(np.max(np.abs(tau))):.2f} N.m"
    )
    result = collect(bus, actuator_name, tau, rate_hz)
    write_mcap(
        output_path,
        times=result["times"],
        ctrl_torque=result["ctrl_torque"],
        position=result["position"],
        velocity=result["velocity"],
        measured_torque=result["measured_torque"],
        metadata={
            **common_metadata,
            "signal_type": label,
            "amplitude": float(amplitude),
            "aborted": bool(result["aborted"]),
        },
    )
    n = len(result["times"])
    dur = result["times"][-1] if n else 0.0
    suffix = "  [ABORTED]" if result["aborted"] else ""
    print(f"[{label}] saved {n} samples ({dur:.2f}s) to {output_path}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0", help="SocketCAN channel")
    parser.add_argument("--id", type=int, default=1, help="Robstride CAN ID")
    parser.add_argument(
        "--model", default="rs-02", help="Robstride model string (rs-02, rs-00, ...)"
    )
    parser.add_argument(
        "--duration", type=float, default=20.0, help="seconds per signal"
    )
    parser.add_argument("--rate", type=float, default=400.0, help="Hz")
    parser.add_argument(
        "--amplitude", type=float, default=2.0,
        help="peak torque amplitude in N.m (well below 17 N.m peak)",
    )
    parser.add_argument(
        "--freqs", type=float, nargs="+",
        default=[2.0, 5.0, 11.0, 19.0, 29.0],
        help="component frequencies for the multisine signal (Hz)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="multisine phase seed",
    )
    parser.add_argument(
        "--rest", type=float, default=2.0,
        help="seconds to coast at zero torque between excitations",
    )
    parser.add_argument(
        "--out-dir", "-o", type=Path,
        default=Path("data/recording"),
        help="directory to write multisine.mcap and chirp.mcap",
    )
    args = parser.parse_args()

    # pre-compute both excitations
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

    name = "shaft"
    actuators = {name: Actuator(id=args.id, model=args.model)}
    bus = RobstrideBus(channel=args.channel, actuators=actuators, bitrate=1_000_000)
    bus.connect()
    try:
        bus.enable(name)
        time.sleep(0.1)  # brief settle so the firmware has started reporting

        _record(
            bus, name, "multisine", tau_multi, args.rate, args.amplitude,
            args.out_dir / "multisine.mcap", common_metadata,
        )
        print(f"\nresting {args.rest:.1f}s at zero torque...")
        time.sleep(args.rest)
        _record(
            bus, name, "chirp", tau_chirp, args.rate, args.amplitude,
            args.out_dir / "chirp.mcap", common_metadata,
        )
    finally:
        try:
            bus.disable(name)
        except Exception as exc:
            print(f"disable failed: {exc}")
        bus.disconnect()


if __name__ == "__main__":
    main()
