"""Kennzahlen aus historischen Kursen."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Drawdown:
    depth: float          # positiv, z. B. 0.18 = -18 %
    peak_date: pd.Timestamp
    peak_price: float
    trough_date: pd.Timestamp
    trough_price: float


@dataclass
class BaseStats:
    price: float
    date: pd.Timestamp
    vol_annual: float          # aus Modellhistorie
    vol_annual_1y: float       # nur letzte 252 Tage
    mean_log_return: float     # täglich, Modellhistorie
    high_52w: float
    high_52w_date: pd.Timestamp
    low_52w: float
    low_52w_date: pd.Timestamp
    mdd_close: Drawdown        # Schlusskurs-Peak -> Schlusskurs-Tief, 1 Jahr
    mdd_intraday: Drawdown     # Tageshoch-Peak -> Tagestief, 1 Jahr
    n_days: int
    history_start: pd.Timestamp
    has_intraday: bool


def last_year(df: pd.DataFrame) -> pd.DataFrame:
    start = df.index[-1] - pd.DateOffset(years=1)
    return df[df.index > start]


def max_drawdown(peak_series: pd.Series, low_series: pd.Series) -> Drawdown:
    """Größter Rückgang vom laufenden Hoch (peak_series) zum späteren Tief (low_series)."""
    running = peak_series.cummax()
    dd = low_series / running - 1.0
    trough_date = dd.idxmin()
    peak_window = peak_series.loc[:trough_date]
    peak_date = peak_window.idxmax()
    return Drawdown(
        depth=float(-dd.min()),
        peak_date=peak_date,
        peak_price=float(peak_series.loc[peak_date]),
        trough_date=trough_date,
        trough_price=float(low_series.loc[trough_date]),
    )


def day_tuples(df: pd.DataFrame) -> np.ndarray:
    """Pro Handelstag (log): [Close/VorClose, High/VorClose, Low/VorClose], Form (n, 3)."""
    prev = df["Close"].shift(1)
    r = np.log(df["Close"] / prev)
    h = np.log(df["High"] / prev)
    lo = np.log(df["Low"] / prev)
    arr = np.column_stack([r, h, lo])[1:]
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr[:, 1] = np.maximum(arr[:, 1], arr[:, 0])
    arr[:, 2] = np.minimum(arr[:, 2], arr[:, 0])
    return arr


def compute_base_stats(df_full: pd.DataFrame, model_years: int) -> tuple[BaseStats, pd.DataFrame]:
    """Kennzahlen + das für die Modelle verwendete Historienfenster."""
    start = df_full.index[-1] - pd.DateOffset(years=model_years)
    df_model = df_full[df_full.index > start]
    y1 = last_year(df_full)

    r_model = np.log(df_model["Close"]).diff().dropna()
    r_1y = np.log(y1["Close"]).diff().dropna()

    has_intraday = bool(((df_full["High"] - df_full[["Open", "Close"]].max(axis=1)) > 1e-12).mean() > 0.2)

    stats = BaseStats(
        price=float(df_full["Close"].iloc[-1]),
        date=df_full.index[-1],
        vol_annual=float(r_model.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        vol_annual_1y=float(r_1y.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        mean_log_return=float(r_model.mean()),
        high_52w=float(y1["High"].max()),
        high_52w_date=y1["High"].idxmax(),
        low_52w=float(y1["Low"].min()),
        low_52w_date=y1["Low"].idxmin(),
        mdd_close=max_drawdown(y1["Close"], y1["Close"]),
        mdd_intraday=max_drawdown(y1["High"], y1["Low"]),
        n_days=len(df_model),
        history_start=df_model.index[0],
        has_intraday=has_intraday,
    )
    return stats, df_model
