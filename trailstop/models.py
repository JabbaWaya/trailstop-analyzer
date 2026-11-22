"""Pfadgeneratoren. Jeder liefert Tages-Tupel (log) der Form (N, T):
r = Close/VorClose, h = High/VorClose, l = Low/VorClose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Paths:
    r: np.ndarray
    h: np.ndarray
    l: np.ndarray
    info: dict = field(default_factory=dict)


def _demean(hist: np.ndarray, zero_drift: bool) -> np.ndarray:
    hist = hist.astype(np.float64).copy()
    if zero_drift:
        hist -= hist[:, 0].mean()  # Verschiebt Close, High und Low gleichermaßen
    return hist


# ---------------------------------------------------------------- GBM
def simulate_gbm(sig_day: float, mu_day: float, n_paths: int, n_days: int,
                 rng: np.random.Generator) -> Paths:
    """Log-Kurs = Brownsche Bewegung mit Drift. Intraday-Hoch/Tief aus der exakten
    Verteilung von Max/Min einer Brownschen Brücke zwischen VorClose und Close."""
    r = mu_day + sig_day * rng.standard_normal((n_paths, n_days))
    var = sig_day ** 2
    u1 = rng.random((n_paths, n_days))
    u2 = rng.random((n_paths, n_days))
    h = 0.5 * (r + np.sqrt(r * r - 2.0 * var * np.log(u1)))
    lo = 0.5 * (r - np.sqrt(r * r - 2.0 * var * np.log(u2)))
    return Paths(r, h, lo, {"sigma_tag": sig_day, "drift_tag": mu_day})


# ---------------------------------------------------------------- Block-Bootstrap
def simulate_bootstrap(hist: np.ndarray, mean_block: float, n_paths: int, n_days: int,
                       rng: np.random.Generator, zero_drift: bool = True) -> Paths:
    """Stationärer Bootstrap (Politis/Romano): Blöcke echter Handelstage mit
    geometrisch verteilter Länge (Mittel `mean_block`)."""
    data = _demean(hist, zero_drift)
    n = len(data)
    p_new = 1.0 / max(mean_block, 1.0)
    idx = np.empty((n_paths, n_days), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_paths)
    for t in range(1, n_days):
        restart = rng.random(n_paths) < p_new
        nxt = (idx[:, t - 1] + 1) % n
        idx[:, t] = np.where(restart, rng.integers(0, n, n_paths), nxt)
    d = data[idx]
    return Paths(d[..., 0], d[..., 1], d[..., 2], {"mittlere_blocklaenge": mean_block})


# ---------------------------------------------------------------- Markov-Kette
@dataclass
class MarkovModel:
    edges: np.ndarray          # Bucket-Grenzen der Tagesrendite (k+1)
    states: np.ndarray         # Zustand je historischem Tag
    transition: np.ndarray     # (k, k), Zeilen summieren auf 1
    start_state: int
    labels: list[str]


def fit_markov(hist: np.ndarray, n_states: int) -> MarkovModel:
    r = hist[:, 0]
    qs = np.quantile(r, np.linspace(0, 1, n_states + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    states = np.clip(np.searchsorted(qs, r, side="right") - 1, 0, n_states - 1)
    counts = np.full((n_states, n_states), 0.5)  # Laplace-Glättung
    np.add.at(counts, (states[:-1], states[1:]), 1.0)
    trans = counts / counts.sum(axis=1, keepdims=True)
    labels = []
    for i in range(n_states):
        lo = "−∞" if not np.isfinite(qs[i]) else f"{qs[i] * 100:+.2f}%"
        hi = "+∞" if not np.isfinite(qs[i + 1]) else f"{qs[i + 1] * 100:+.2f}%"
        labels.append(f"Z{i + 1} [{lo}, {hi})")
    return MarkovModel(qs, states, trans, int(states[-1]), labels)


def simulate_markov(hist: np.ndarray, model: MarkovModel, n_paths: int, n_days: int,
                    rng: np.random.Generator, zero_drift: bool = True) -> Paths:
    """Zustandsfolge aus der Übergangsmatrix ziehen (Start = heutiger Zustand),
    je Zustand einen echten historischen Tag (inkl. Hoch/Tief) aus diesem Bucket."""
    data = _demean(hist, zero_drift)
    k = model.transition.shape[0]
    cum = np.cumsum(model.transition, axis=1)
    cum[:, -1] = 1.0
    order = np.argsort(model.states, kind="stable")
    counts = np.bincount(model.states, minlength=k)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])

    state = np.full(n_paths, model.start_state, dtype=np.int64)
    idx = np.empty((n_paths, n_days), dtype=np.int64)
    for t in range(n_days):
        u = rng.random(n_paths)
        state = (u[:, None] > cum[state]).sum(axis=1)
        pick = starts[state] + (rng.random(n_paths) * counts[state]).astype(np.int64)
        idx[:, t] = order[pick]
    d = data[idx]
    return Paths(d[..., 0], d[..., 1], d[..., 2], {"zustaende": k})
