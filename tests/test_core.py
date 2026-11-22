import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trailstop import analytic as an  # noqa: E402
from trailstop.data import is_valid_isin, load_csv  # noqa: E402
from trailstop.engine import evaluate  # noqa: E402
from trailstop.models import Paths, fit_markov, simulate_bootstrap, simulate_gbm, simulate_markov  # noqa: E402
from trailstop.stats import compute_base_stats, max_drawdown  # noqa: E402


def test_isin_checksum():
    assert is_valid_isin("IE00B4L5YC18")
    assert is_valid_isin("US0378331005")
    assert not is_valid_isin("IE00B4L5YC19")


def test_reflection_z():
    assert an.z_fixed(0.95) == pytest.approx(1.95996, abs=1e-4)
    b = 1.96 * 0.01 * math.sqrt(10)
    assert an.p_fixed(b, 0.01, 10) == pytest.approx(0.05, abs=1e-3)


def test_trailing_small_prob_is_twice_fixed():
    # Für kleine Wahrscheinlichkeiten gilt P_trailing ≈ 2 * P_fixed (zwei Schranken)
    b = 3.0 * 0.01 * math.sqrt(10)
    assert an.p_trailing(b, 0.01, 10) == pytest.approx(2 * an.p_fixed(b, 0.01, 10), rel=1e-3)


def test_halflife_fixed_consistent():
    n = an.halflife_fixed(0.05, 0.01)
    assert an.p_fixed(0.05, 0.01, n) == pytest.approx(0.5, abs=1e-9)


def test_halflife_trailing_consistent():
    n = an.halflife_trailing(0.05, 0.01)
    assert an.p_trailing(0.05, 0.01, n) == pytest.approx(0.5, abs=1e-6)


def test_gbm_simulation_matches_levy_formula():
    """Fein diskretisierte driftlose BB: Simulierter Trailing-Drawdown ≈ Lévy-Formel."""
    rng = np.random.default_rng(1)
    steps_per_day, days, sig = 40, 10, 0.012
    sub = sig / math.sqrt(steps_per_day)
    n = 20000
    r = sub * rng.standard_normal((n, days * steps_per_day))
    paths = Paths(r, np.maximum(r, 0), np.minimum(r, 0))
    res = evaluate("gbm", paths, rng)
    b = an.b_trailing(0.95, sig, days)
    p_sim = res.p_trigger(b, days * steps_per_day)
    # Diskrete Beobachtung unterschätzt leicht -> Toleranz
    assert 0.035 < p_sim < 0.055


def test_gbm_bridge_daily_matches_levy():
    rng = np.random.default_rng(2)
    sig = 0.012
    res = evaluate("gbm", simulate_gbm(sig, 0.0, 30000, 10, rng), rng)
    b = an.b_trailing(0.95, sig, 10)
    assert res.p_trigger(b, 10) == pytest.approx(0.05, abs=0.015)


def test_stop_quantile_and_probability_consistent():
    rng = np.random.default_rng(3)
    res = evaluate("gbm", simulate_gbm(0.01, 0.0, 20000, 30, rng), rng)
    b = res.stop_for_confidence(0.95, 10)
    assert res.p_trigger(b, 10) == pytest.approx(0.05, abs=0.002)
    assert res.trigger_curve(b)[9] == pytest.approx(res.p_trigger(b, 10))
    hl = res.halflife(b)
    assert 10 < hl <= 30 or math.isinf(hl)


def _fake_hist(n=800, seed=0):
    rng = np.random.default_rng(seed)
    r = 0.01 * rng.standard_normal(n)
    h = r + np.abs(0.004 * rng.standard_normal(n))
    lo = r - np.abs(0.004 * rng.standard_normal(n))
    return np.column_stack([r, h, lo])


def test_bootstrap_and_markov_shapes_and_drift():
    hist = _fake_hist()
    rng = np.random.default_rng(4)
    pb = simulate_bootstrap(hist, 5, 2000, 50, rng)
    assert pb.r.shape == (2000, 50)
    assert abs(pb.r.mean()) < 1e-3
    mm = fit_markov(hist, 5)
    assert np.allclose(mm.transition.sum(axis=1), 1.0)
    pm = simulate_markov(hist, mm, 2000, 50, rng)
    assert pm.r.shape == (2000, 50)
    assert np.all(pm.h >= pm.r - 1e-12) and np.all(pm.l <= pm.r + 1e-12)


def test_max_drawdown_peak_to_trough():
    s = pd.Series([100, 120, 90, 110, 80, 130], index=pd.date_range("2024-01-01", periods=6))
    dd = max_drawdown(s, s)
    assert dd.depth == pytest.approx(1 - 80 / 120)
    assert dd.peak_price == 120 and dd.trough_price == 80


def test_load_csv_german_format():
    csv = "Datum;Eröffnung;Hoch;Tief;Schlusskurs\n02.01.2024;1.234,50;1.240,00;1.230,00;1.238,00\n03.01.2024;1.238,00;1.250,00;1.235,00;1.245,10\n"
    df = load_csv(csv.encode("utf-8"))
    assert df["Close"].iloc[-1] == pytest.approx(1245.10)
    assert df["High"].iloc[0] == pytest.approx(1240.0)


def test_base_stats_runs():
    idx = pd.bdate_range("2021-01-01", periods=900)
    rng = np.random.default_rng(5)
    c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(900)))
    df = pd.DataFrame({"Open": c, "High": c * 1.005, "Low": c * 0.995, "Close": c}, index=idx)
    st, dm = compute_base_stats(df, 3)
    assert 0.1 < st.vol_annual < 0.2
    assert st.mdd_intraday.depth >= st.mdd_close.depth
