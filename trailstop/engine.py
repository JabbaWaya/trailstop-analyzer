"""Trailing-Stop-Auswertung auf simulierten OHLC-Pfaden.

Der Stop folgt dem höchsten bisher erreichten Kurs (Tageshoch, Start = aktueller Kurs)
und löst aus, sobald ein Tagestief den Stop berührt. Innerhalb eines Tages ist die
Reihenfolge Hoch/Tief unbekannt; sie wird je Tag zufällig (50/50) festgelegt.

Kernobjekt ist der laufende maximale Drawdown (log) je Pfad: Ein Stop mit Log-Abstand b
ist bis Tag t genau dann ausgelöst, wenn maxdd[:, t] >= b. Dadurch lassen sich alle
Stop-Abstände auf denselben Pfaden auswerten (gemeinsame Zufallszahlen).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import Paths


@dataclass
class StopResult:
    name: str
    maxdd: np.ndarray        # (N, T) laufender max. Drawdown, log, nichtfallend
    median_dd: np.ndarray    # (T,) Median über Pfade -> Halbwertszeit
    close_log: np.ndarray    # (N, T) log(Close_t / Close_0)

    @property
    def n_days(self) -> int:
        return self.maxdd.shape[1]

    def stop_for_confidence(self, confidence: float, horizon: int) -> float:
        """Log-Abstand b mit P(Auslösung bis Tag `horizon`) = 1 - confidence."""
        return float(np.quantile(self.maxdd[:, horizon - 1], confidence))

    def p_trigger(self, b: float, days: int) -> float:
        return float(np.mean(self.maxdd[:, days - 1] >= b))

    def trigger_curve(self, b: float) -> np.ndarray:
        """P(Auslösung bis Tag t), t = 1..T."""
        return np.mean(self.maxdd >= b, axis=0)

    def halflife(self, b: float) -> float:
        """Erster Tag, an dem P(Auslösung) >= 50 %; inf, falls nicht im Simulationsfenster."""
        i = int(np.searchsorted(self.median_dd, b, side="left"))
        return float(i + 1) if i < self.n_days else float("inf")

    def p_maxdd_at_least(self, depth_log: float, days: int) -> float:
        return float(np.mean(self.maxdd[:, days - 1] >= depth_log))


def evaluate(name: str, paths: Paths, rng: np.random.Generator) -> StopResult:
    r, h, lo = paths.r, paths.h, paths.l
    close = np.cumsum(r, axis=1)
    prev_close = close - r
    high = prev_close + h
    low = prev_close + lo
    # Hoch bis einschließlich Vortag (Start: aktueller Kurs = 0)
    peak_prev = np.maximum.accumulate(np.concatenate([np.zeros((r.shape[0], 1)), high[:, :-1]], axis=1), axis=1)
    peak_prev = np.maximum(peak_prev, 0.0)
    high_first = rng.random(r.shape) < 0.5
    peak_eff = np.where(high_first, np.maximum(peak_prev, high), peak_prev)
    dd = np.maximum(peak_eff - low, 0.0)
    maxdd = np.maximum.accumulate(dd, axis=1).astype(np.float32)
    return StopResult(name, maxdd, np.median(maxdd, axis=0), close.astype(np.float32))
