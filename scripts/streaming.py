"""Bus setup and streaming-recording loop for `collect.py`.

The `command_fn` callback indirection is kept so the loop is reusable for
future driving modes; the connect/enable lifecycle, the wait-for-first-state
probe, the per-step state read, the fault-abort path, and the zero-on-exit
safety move all live here.
"""

import time
from typing import Any, Callable

import numpy as np
from loop_rate_limiters import RateLimiter

from actuator_control import Actuator, RobstrideBus


ACTUATOR_NAME = "shaft"   # the logical name we use throughout the codebase


def open_bus(
    channel: str,
    device_id: int,
    model: str,
    bitrate: int = 1_000_000,
) -> RobstrideBus:
    """Connect to a single Robstride and enable it under the logical name
    `ACTUATOR_NAME`. Returns the bus; caller is responsible for `close_bus`.
    """
    actuators = {ACTUATOR_NAME: Actuator(id=device_id, model=model)}
    bus = RobstrideBus(channel=channel, actuators=actuators, bitrate=bitrate)
    bus.connect()
    bus.enable(ACTUATOR_NAME)
    time.sleep(0.1)  # let the firmware start reporting
    return bus


def close_bus(bus: RobstrideBus) -> None:
    """Disable and disconnect, tolerating a failed disable (e.g. fault state)."""
    try:
        bus.disable(ACTUATOR_NAME)
    except Exception as exc:
        print(f"disable failed: {exc}")
    bus.disconnect()


def wait_for_first_state(
    bus: RobstrideBus,
    timeout_s: float = 1.0,
) -> float:
    """Block until the first feedback frame arrives. Returns absolute pos."""
    deadline = time.perf_counter() + timeout_s
    while bus.get_state(ACTUATOR_NAME) is None:
        if time.perf_counter() > deadline:
            raise RuntimeError(
                f"No state from {ACTUATOR_NAME!r} after {timeout_s:.1f}s. "
                "Check CAN wiring and that the motor is powered."
            )
        time.sleep(0.01)
    return bus.get_state(ACTUATOR_NAME).position


def stream(
    bus: RobstrideBus,
    n_steps: int,
    rate_hz: float,
    command_fn: Callable[[int], dict[str, Any]],
    pos0: float,
) -> dict[str, Any]:
    """Drive `n_steps` MIT commands at `rate_hz`; log feedback per step.

    `command_fn(i)` returns the kwargs (everything except ``actuator``) for
    `bus.write_mit_control` at step i. Logged positions are relative to
    `pos0`, so each recording starts at zero. A zero-torque safety command
    is sent on exit (success or exception). Faults reported by the firmware
    truncate the recording and set ``aborted=True``.

    Returns dict with: times, position, velocity, measured_torque, aborted.
    """
    times = np.zeros(n_steps, dtype=np.float64)
    pos = np.zeros(n_steps, dtype=np.float64)
    vel = np.zeros(n_steps, dtype=np.float64)
    tau_meas = np.zeros(n_steps, dtype=np.float64)

    rate = RateLimiter(frequency=rate_hz)
    t0 = time.perf_counter()
    aborted = False
    last_idx = n_steps
    try:
        for i in range(n_steps):
            bus.write_mit_control(actuator=ACTUATOR_NAME, **command_fn(i))
            state = bus.get_state(ACTUATOR_NAME)
            if state is None:
                raise RuntimeError("Lost actuator state mid-run.")
            times[i] = time.perf_counter() - t0
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
        bus.write_mit_control(
            actuator=ACTUATOR_NAME,
            position=0.0, velocity=0.0,
            kp=0.0, kd=0.0, torque=0.0,
        )

    return {
        "times": times[:last_idx],
        "position": pos[:last_idx],
        "velocity": vel[:last_idx],
        "measured_torque": tau_meas[:last_idx],
        "aborted": aborted,
    }
