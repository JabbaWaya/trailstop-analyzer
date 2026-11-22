"""Einheitliche Schnittstelle für Simulations- und Formelmodelle + Fazit-Tabelle."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import analytic as an
from .engine import StopResult

MODEL_MARKOV = "Markov-Kette"
MODEL_BOOT = "Block-Bootstrap"
MODEL_GBM = "GBM-Monte-Carlo"
MODEL_LEVY = "Formel: nachlaufend (Lévy)"
MODEL_REFL = "Formel: fester Stop (Reflexion)"
SIM_MODELS = [MODEL_MARKOV, MODEL_BOOT, MODEL_GBM]
ALL_MODELS = SIM_MODELS + [MODEL_LEVY, MODEL_REFL]

MODEL_COLORS = {
    MODEL_MARKOV: "#2a6fdb",
    MODEL_BOOT: "#e07a1f",
    MODEL_GBM: "#2e9e6a",
    MODEL_LEVY: "#8a5cc2",
    MODEL_REFL: "#8c8c8c",
}


class ModelView:
    """Gleiche Methoden für Simulation und Formel (Abstände immer als Log-Abstand b)."""

    def __init__(self, name: str, sim: StopResult | None = None, sig_day: float | None = None):
        self.name, self.sim, self.sig_day = name, sim, sig_day

    @property
    def is_sim(self) -> bool:
        return self.sim is not None

    def stop(self, confidence: float, horizon: int) -> float:
        if self.sim is not None:
            return self.sim.stop_for_confidence(confidence, horizon)
        if self.name == MODEL_LEVY:
            return an.b_trailing(confidence, self.sig_day, horizon)
        return an.z_fixed(confidence) * self.sig_day * math.sqrt(horizon)

    def p(self, b: float, days: int) -> float:
        if self.sim is not None:
            return self.sim.p_trigger(b, days)
        if self.name == MODEL_LEVY:
            return an.p_trailing(b, self.sig_day, days)
        return an.p_fixed(b, self.sig_day, days)

    def curve(self, b: float, days: int) -> np.ndarray:
        if self.sim is not None:
            return self.sim.trigger_curve(b)[:days]
        return np.array([self.p(b, t) for t in range(1, days + 1)])

    def halflife(self, b: float) -> float:
        if self.sim is not None:
            return self.sim.halflife(b)
        if self.name == MODEL_LEVY:
            return an.halflife_trailing(b, self.sig_day)
        return an.halflife_fixed(b, self.sig_day)

    def p_mdd(self, depth_log: float, days: int) -> float:
        if self.sim is not None:
            return self.sim.p_maxdd_at_least(depth_log, days)
        return self.p(depth_log, days)


def fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x * 100:,.{digits}f} %".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_num(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_days(x: float, limit: int | None = None) -> str:
    if math.isinf(x):
        return f"> {limit}" if limit else "∞"
    return fmt_num(x, 1) if x != int(x) else f"{int(x)}"


def distance_rows(d_opt: float) -> list[tuple[str, float]]:
    base = round(d_opt * 100)
    rows = [("Berechnet", d_opt)]
    for off in (-2, -1, 0, 1, 2):
        v = base + off
        if v > 0:
            label = f"Gerundet {off:+d} %-P." if off else "Gerundet"
            rows.append((label, v / 100.0))
    return rows


def summary_table(models: dict[str, ModelView], lead: str, d_opt: float, price: float,
                  horizon: int, sim_days: int) -> pd.DataFrame:
    recs = []
    for label, d in distance_rows(d_opt):
        b = an.pct_to_log(d)
        rec = {
            "Variante": label,
            "Trailing-Abstand": d,
            "Stop-Kurs": price * (1 - d),
            f"P(Auslösung ≤ {horizon} T)": models[lead].p(b, horizon),
            "Halbwertszeit (T)": models[lead].halflife(b),
        }
        for name, m in models.items():
            if name == lead:
                continue
            rec[f"P {name}"] = m.p(b, horizon)
            rec[f"HWZ {name}"] = m.halflife(b)
        recs.append(rec)
    return pd.DataFrame(recs)
