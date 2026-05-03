"""I/O for actuator recordings stored as MCAP files.

Layout:
  * one topic ``/actuator/state``, JSON-schema message per timestep:
    ``{ctrl_torque, position_target, position, velocity, measured_torque}``
  * MCAP message ``log_time`` is nanoseconds since the start of the recording
    (relative timeline; not unix wall-clock)
  * a ``recording`` metadata record holds run-level fields (sampling rate,
    signal type, control mode, kp/kd for PD-mode runs, sim ground-truth, ...).
    MCAP metadata stores ``str -> str``, so values are JSON-encoded then decoded.

The MCAP files open cleanly in Foxglove and ROS 2 tooling.

Two recording flavors share one schema:
  * torque (``control_mode="torque"``): ``ctrl_torque`` is what we sent via
    MIT mode with kp=kd=0; ``position_target`` is 0.
  * pd (``control_mode="pd"``): ``position_target`` is what we sent; the
    firmware applied tau = kp_cmd*(p_des - p) + kd_cmd*(0 - v) + ctrl_torque,
    with ``kp_cmd`` / ``kd_cmd`` in the recording metadata.

Three layers of abstraction, increasing distance from disk:
  * ``Recording`` -- raw MCAP contents, irregular sample times.
  * ``Sequence``  -- ``Recording`` plus the empirical sampling rate and the
    *equivalent torque* (PD reconstructed via the firmware's PD law); still
    irregular times.
  * ``resample(seq, dt)`` -- ``Sequence`` on a uniform t = k*dt grid, ready
    for the optimizer.
"""


import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


TOPIC = "/actuator/state"

SCHEMA: dict[str, Any] = {
    "title": "ActuatorState",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "ctrl_torque", "position_target", "position", "velocity",
        "measured_torque",
    ],
    "properties": {
        "ctrl_torque": {"type": "number", "description": "feed-forward torque command (N.m)"},
        "position_target": {"type": "number", "description": "position target sent to firmware (rad); 0 in torque mode"},
        "position": {"type": "number", "description": "output-shaft position (rad)"},
        "velocity": {"type": "number", "description": "output-shaft velocity (rad/s)"},
        "measured_torque": {"type": "number", "description": "firmware-estimated output torque (N.m)"},
    },
}


# --------------------------------------------------------------------------- #
# Raw MCAP <-> dataclass                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class Recording:
    """One actuator-state recording exactly as it lives on disk."""
    times: np.ndarray              # (N,) seconds since recording start
    ctrl_torque: np.ndarray        # (N,) N.m -- feed-forward torque command
    position_target: np.ndarray    # (N,) rad -- position target (0 in torque mode)
    position: np.ndarray           # (N,) rad, output side, zeroed at t=0
    velocity: np.ndarray           # (N,) rad/s
    measured_torque: np.ndarray    # (N,) N.m
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.times.shape[0])


def _to_str_dict(d: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in d.items():
        if isinstance(v, np.generic):
            v = v.item()
        out[str(k)] = json.dumps(v)
    return out


def _from_str_dict(d: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        try:
            out[k] = json.loads(v)
        except (TypeError, json.JSONDecodeError):
            out[k] = v
    return out


def write_mcap(
    path: Path | str,
    *,
    times: np.ndarray,
    ctrl_torque: np.ndarray,
    position_target: np.ndarray,
    position: np.ndarray,
    velocity: np.ndarray,
    measured_torque: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Write one recording to an MCAP file."""
    from mcap.writer import Writer

    n = int(np.asarray(times).shape[0])
    arrays = {
        "ctrl_torque": ctrl_torque,
        "position_target": position_target,
        "position": position,
        "velocity": velocity,
        "measured_torque": measured_torque,
    }
    for name, a in arrays.items():
        if int(np.asarray(a).shape[0]) != n:
            raise ValueError(
                f"signal {name!r} has length {len(a)}, expected {n}"
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    times_arr = np.asarray(times, dtype=np.float64)
    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()}

    with path.open("wb") as f:
        writer = Writer(f)
        writer.start(profile="robstride-sysid", library="robstride-sysid")
        schema_id = writer.register_schema(
            name="ActuatorState",
            encoding="jsonschema",
            data=json.dumps(SCHEMA).encode(),
        )
        channel_id = writer.register_channel(
            topic=TOPIC,
            message_encoding="json",
            schema_id=schema_id,
        )
        for i in range(n):
            t_ns = int(round(float(times_arr[i]) * 1e9))
            data = json.dumps({
                k: float(v[i]) for k, v in arrays.items()
            }).encode()
            writer.add_message(
                channel_id=channel_id,
                log_time=t_ns,
                publish_time=t_ns,
                data=data,
                sequence=i,
            )
        writer.add_metadata("recording", _to_str_dict(metadata))
        writer.finish()


def read_mcap(path: Path | str) -> Recording:
    """Load one recording from an MCAP file."""
    from mcap.reader import make_reader

    times: list[float] = []
    cmd: list[float] = []
    p_des: list[float] = []
    pos: list[float] = []
    vel: list[float] = []
    tau: list[float] = []
    metadata: dict[str, Any] = {}

    with Path(path).open("rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages():
            if channel.topic != TOPIC:
                continue
            d = json.loads(message.data)
            times.append(message.log_time / 1e9)
            cmd.append(float(d["ctrl_torque"]))
            p_des.append(float(d["position_target"]))
            pos.append(float(d["position"]))
            vel.append(float(d["velocity"]))
            tau.append(float(d["measured_torque"]))
        for record in reader.iter_metadata():
            if record.name == "recording":
                metadata = _from_str_dict(dict(record.metadata))

    return Recording(
        times=np.asarray(times, dtype=np.float64),
        ctrl_torque=np.asarray(cmd, dtype=np.float64),
        position_target=np.asarray(p_des, dtype=np.float64),
        position=np.asarray(pos, dtype=np.float64),
        velocity=np.asarray(vel, dtype=np.float64),
        measured_torque=np.asarray(tau, dtype=np.float64),
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Sequence: Recording + empirical rate + equivalent torque                    #
# --------------------------------------------------------------------------- #

@dataclass
class Sequence:
    """One recording prepared for the optimizer.

    `ctrl_torque` here is the *equivalent* torque -- the one the firmware
    actually applied -- regardless of control mode. For torque-mode that's
    just what we sent; for PD-mode it's reconstructed from the PD law.
    `times` may be either the raw irregular timestamps (from `load_sequence`)
    or a uniform grid (from `resample`).
    """
    name: str
    times: np.ndarray
    ctrl_torque: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    sampling_rate: float           # empirical, Hz
    control_mode: str              # "torque" or "pd"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.times.shape[0])


def _equivalent_torque(rec: Recording) -> np.ndarray:
    """Reconstruct the torque the firmware applied during the recording."""
    mode = rec.metadata.get("control_mode", "torque")
    if mode == "torque":
        return rec.ctrl_torque.astype(np.float64)
    if mode == "pd":
        try:
            kp = float(rec.metadata["kp_cmd"])
            kd = float(rec.metadata["kd_cmd"])
        except KeyError as exc:
            raise ValueError(
                f"PD-mode recording missing {exc.args[0]!r} in metadata"
            ) from exc
        return (
            kp * (rec.position_target - rec.position)
            - kd * rec.velocity
            + rec.ctrl_torque
        ).astype(np.float64)
    raise ValueError(f"unknown control_mode {mode!r}")


def load_sequence(path: Path | str) -> Sequence:
    """Read an MCAP, derive empirical rate, reconstruct equivalent torque."""
    path = Path(path)
    rec = read_mcap(path)
    if len(rec.times) < 2:
        raise ValueError(f"{path}: need >= 2 samples")
    sampling_rate = 1.0 / float(np.median(np.diff(rec.times)))
    return Sequence(
        name=path.stem,
        times=rec.times,
        ctrl_torque=_equivalent_torque(rec),
        position=rec.position,
        velocity=rec.velocity,
        sampling_rate=sampling_rate,
        control_mode=str(rec.metadata.get("control_mode", "torque")),
        metadata=rec.metadata,
    )


def resample(seq: Sequence, dt: float) -> Sequence:
    """Resample onto a uniform t = k*dt grid spanning the recording.

    The returned sequence is rebased to t=0 so the optimizer sees a clean
    aligned timeline. `sampling_rate` is updated to 1/dt.
    """
    t0 = float(seq.times[0])
    n = int(np.floor((seq.times[-1] - t0) / dt)) + 1
    t_uniform = t0 + dt * np.arange(n)
    return Sequence(
        name=seq.name,
        times=t_uniform - t0,
        ctrl_torque=np.interp(t_uniform, seq.times, seq.ctrl_torque),
        position=np.interp(t_uniform, seq.times, seq.position),
        velocity=np.interp(t_uniform, seq.times, seq.velocity),
        sampling_rate=1.0 / dt,
        control_mode=seq.control_mode,
        metadata=seq.metadata,
    )
