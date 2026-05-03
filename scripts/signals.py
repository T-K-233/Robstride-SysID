"""Excitation waveform generators.

`collect_torque.py` and `collect_pd.py` run the same multisine and chirp
shapes; the only difference is interpretation -- in torque mode the samples
are N.m, in PD mode they are position targets in rad. `tests/sim_collect.py`
pulls from here so torque-mode sim and torque-mode hardware see identical
excitation.
"""



import numpy as np


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
    sig = np.zeros(n, dtype=np.float64)
    for f, ph in zip(freqs, phases):
        sig += per_amp * np.sin(2 * np.pi * f * t + ph)
    sig *= _tukey(n, alpha=0.1)
    return t, sig


def make_chirp(
    duration: float,
    rate: float,
    amplitude: float,
    f0: float = 1.0,
    f1: float = 40.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear-frequency-sweep waveform."""
    n = int(round(duration * rate))
    t = np.arange(n) / rate
    k = (f1 - f0) / duration
    sig = amplitude * np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))
    sig *= _tukey(n, alpha=0.1)
    return t, sig


def _tukey(n: int, alpha: float) -> np.ndarray:
    """Tukey (tapered cosine) window for smooth onset/offset."""
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
