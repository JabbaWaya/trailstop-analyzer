"""Geschlossene Formeln (driftlose Brownsche Bewegung der Log-Kurse).

- Fester Stop (Reflexionsprinzip):   P(min_{s<=T} W_s <= -b) = 2 * Phi(-b / (sigma*sqrt(T)))
- Nachlaufender Stop (Lévy):         M_t - W_t hat dieselbe Verteilung wie |W_t| (als Prozess),
                                     also max. Drawdown ~ sup_{s<=T} |W_s|  (Zwei-Schranken-Problem)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

TRADING_DAYS = 252


def sigma_day(vol_annual: float) -> float:
    return vol_annual / math.sqrt(TRADING_DAYS)


def sigma_horizon(vol_annual: float, days: float) -> float:
    return sigma_day(vol_annual) * math.sqrt(days)


def log_to_pct(b: float) -> float:
    """Log-Abstand -> prozentualer Abstand vom Kurs."""
    return 1.0 - math.exp(-b)


def pct_to_log(d: float) -> float:
    return -math.log(1.0 - d)


# ---------- fester Stop ----------
def p_fixed(b: float, sig_day: float, days: float) -> float:
    if days <= 0:
        return 0.0
    return float(min(1.0, 2.0 * norm.cdf(-b / (sig_day * math.sqrt(days)))))


def z_fixed(confidence: float) -> float:
    """z so, dass 2*Phi(-z) = 1 - confidence."""
    return float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))


def halflife_fixed(b: float, sig_day: float) -> float:
    """Tage bis P(Auslösung) = 50 %:  2*Phi(-b/(sigma*sqrt(n))) = 0.5."""
    return float((b / (sig_day * norm.ppf(0.75))) ** 2)


# ---------- nachlaufender Stop ----------
def p_trailing(b: float, sig_day: float, days: float) -> float:
    """P(max. Drawdown >= b in `days`) = 1 - P(sup|W| < b), Spiegelungsreihe."""
    if days <= 0:
        return 0.0
    s = sig_day * math.sqrt(days)
    x = b / s
    kmax = 10 + int(8.0 / max(x, 1e-6))
    k = np.arange(-kmax, kmax + 1)
    stay = np.sum((-1.0) ** k * (norm.cdf((1 - 2 * k) * x) - norm.cdf((-1 - 2 * k) * x)))
    return float(min(1.0, max(0.0, 1.0 - stay)))


def b_trailing(confidence: float, sig_day: float, days: float) -> float:
    """Log-Abstand b so, dass P_trailing(b) = 1 - confidence."""
    target = 1.0 - confidence
    s = sig_day * math.sqrt(days)
    return float(brentq(lambda b: p_trailing(b, sig_day, days) - target, 0.01 * s, 20 * s))


def halflife_trailing(b: float, sig_day: float) -> float:
    """Tage bis P(Auslösung nachlaufender Stop) = 50 %."""
    # P_trailing hängt nur von b/(sigma*sqrt(n)) ab -> einmal x50 bestimmen
    x50 = brentq(lambda x: p_trailing(x, 1.0, 1.0) - 0.5, 0.1, 10.0)
    return float((b / (sig_day * x50)) ** 2)


def p_mdd_1y_trailing(mdd_log: float, sig_day: float, days: int = TRADING_DAYS) -> float:
    """P(1-Jahres-Max-Drawdown >= realer MDD) bei driftloser BB (Lévy)."""
    return p_trailing(mdd_log, sig_day, days)
