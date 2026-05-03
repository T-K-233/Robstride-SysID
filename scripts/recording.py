"""I/O for actuator recordings stored as MCAP files.

Layout:
  * one topic ``/actuator/state``, JSON-schema message per timestep:
    ``{ctrl_torque, position, velocity, measured_torque}``
  * MCAP message ``log_time`` is nanoseconds since the start of the recording
    (relative timeline; not unix wall-clock)
  * a ``recording`` metadata record holds run-level fields (sampling rate,
    signal type, actuator ID, sim ground-truth, ...). MCAP metadata stores
    ``str -> str``, so values are JSON-encoded then decoded.

The MCAP files open cleanly in Foxglove and ROS 2 tooling.

Three layers of abstraction, increasing distance from disk:
  * ``Recording`` -- raw MCAP contents, irregular sample times.
  * ``Sequence``  -- ``Recording`` plus the empirical sampling rate;
    still irregular times.
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
    "required": ["ctrl_torque", "position", "velocity", "measured_torque"],
    "properties": {
        "ctrl_torque": {"type": "number", "description": "feed-forward torque command (N.m)"},
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


# --------------------------------------------------------------------------- #
# Sequence: Recording + empirical rate                                        #
# --------------------------------------------------------------------------- #

@dataclass
class Sequence:
    """One recording prepared for the optimizer.

    `times` may be either the raw irregular timestamps (from `load_sequence`)
    or a uniform grid (from `resample`).
    """
    name: str
    times: np.ndarray
    ctrl_torque: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    sampling_rate: float           # empirical, Hz
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.times.shape[0])


def load_sequence(path: Path | str) -> Sequence:
    """Read an MCAP and derive its empirical sampling rate from timestamps."""
    path = Path(path)
    rec = read_mcap(path)
    if len(rec.times) < 2:
        raise ValueError(f"{path}: need >= 2 samples")
    sampling_rate = 1.0 / float(np.median(np.diff(rec.times)))
    return Sequence(
        name=path.stem,
        times=rec.times,
        ctrl_torque=rec.ctrl_torque,
        position=rec.position,
        velocity=rec.velocity,
        sampling_rate=sampling_rate,
        metadata=rec.metadata,
    )


# --------------------------------------------------------------------------- #
# Run-directory layout: data/<model>/run<N>/  (and results/<model>/run<N>/)   #
# --------------------------------------------------------------------------- #

def next_run_dir(base: Path | str) -> Path:
    """Return ``base/run<N>`` with N = (max existing run number) + 1.

    Used by `collect.py` to write each new session into a fresh subdir.
    """
    base = Path(base)
    n = 1
    if base.exists():
        nums = [
            int(p.name[3:]) for p in base.iterdir()
            if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit()
        ]
        if nums:
            n = max(nums) + 1
    return base / f"run{n}"


def list_run_dirs(base: Path | str) -> list[Path]:
    """Return every ``base/run<N>`` directory, sorted numerically by N.

    Used by `optimize.py` and `visualize.py` to iterate every captured run
    for a given actuator class.
    """
    base = Path(base)
    if not base.is_dir():
        raise SystemExit(f"{base} does not exist")
    runs = []
    for p in base.iterdir():
        if p.is_dir() and p.name.startswith("run") and p.name[3:].isdigit():
            runs.append((int(p.name[3:]), p))
    if not runs:
        raise SystemExit(f"no run<N> directories found in {base}")
    return [p for _, p in sorted(runs)]


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
        metadata=seq.metadata,
    )
