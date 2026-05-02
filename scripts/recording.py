"""I/O helpers for actuator recordings stored as MCAP files.

Layout:
  * one topic, ``/actuator/state``, with a JSON-schema message per timestep:
    ``{ctrl_torque, position, velocity, measured_torque}``
  * MCAP message ``log_time`` is nanoseconds since the start of the recording
    (relative timeline; not unix wall-clock)
  * a single ``recording`` metadata record holds the run-level metadata
    (sampling rate, signal type, actuator ID, sim ground-truth, ...) -- MCAP
    metadata stores ``str -> str``, so values are JSON-encoded then decoded.

The MCAP files open cleanly in Foxglove and ROS 2 tooling.
"""

from __future__ import annotations

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
    "required": ["ctrl_torque", "position", "velocity", "measured_torque"],
    "properties": {
        "ctrl_torque": {"type": "number", "description": "feed-forward torque command (N.m)"},
        "position": {"type": "number", "description": "output-shaft position (rad)"},
        "velocity": {"type": "number", "description": "output-shaft velocity (rad/s)"},
        "measured_torque": {"type": "number", "description": "firmware-estimated output torque (N.m)"},
    },
}


@dataclass
class Recording:
    """One actuator-state recording, fully loaded into memory."""
    times: np.ndarray              # (N,) seconds since recording start
    ctrl_torque: np.ndarray        # (N,) N.m
    position: np.ndarray           # (N,) rad, output side, zeroed at t=0
    velocity: np.ndarray           # (N,) rad/s
    measured_torque: np.ndarray    # (N,) N.m
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.times.shape[0])


# --------------------------------------------------------------------------- #
# metadata helpers (MCAP metadata stores str->str, so we JSON-encode values)  #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# write / read                                                                #
# --------------------------------------------------------------------------- #

def write_mcap(
    path: Path | str,
    *,
    times: np.ndarray,
    ctrl_torque: np.ndarray,
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
    times = np.asarray(times, dtype=np.float64)
    ctrl_torque = np.asarray(ctrl_torque, dtype=np.float64)
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    measured_torque = np.asarray(measured_torque, dtype=np.float64)

    with path.open("wb") as f:
        writer = Writer(f)
        writer.start(profile="rs02-sysid", library="rs02-sysid")
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
            t_ns = int(round(float(times[i]) * 1e9))
            data = json.dumps({
                "ctrl_torque": float(ctrl_torque[i]),
                "position": float(position[i]),
                "velocity": float(velocity[i]),
                "measured_torque": float(measured_torque[i]),
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
            pos.append(float(d["position"]))
            vel.append(float(d["velocity"]))
            tau.append(float(d["measured_torque"]))
        for record in reader.iter_metadata():
            if record.name == "recording":
                metadata = _from_str_dict(dict(record.metadata))

    return Recording(
        times=np.asarray(times, dtype=np.float64),
        ctrl_torque=np.asarray(cmd, dtype=np.float64),
        position=np.asarray(pos, dtype=np.float64),
        velocity=np.asarray(vel, dtype=np.float64),
        measured_torque=np.asarray(tau, dtype=np.float64),
        metadata=metadata,
    )
