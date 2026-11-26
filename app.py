"""Streamlit-App: Trailing-Stop-Risikoanalyse per ISIN.

Start:  uv run streamlit run app.py
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from trailstop import analytic as an
from trailstop.data import is_valid_isin, load_csv, load_yahoo, resolve_isin
from trailstop.engine import evaluate
from trailstop.models import fit_markov, simulate_bootstrap, simulate_gbm, simulate_markov
from trailstop.report import (ALL_MODELS, MODEL_BOOT, MODEL_COLORS, MODEL_GBM, MODEL_LEVY, MODEL_MARKOV,
                              MODEL_REFL, ModelView, fmt_days, fmt_num, fmt_pct, summary_table)
from trailstop.stats import compute_base_stats, day_tuples, last_year

st.set_page_config(page_title="Trailing-Stop-Analyse", page_icon="📉", layout="wide")



# ------------------------------------------------------------------ Caching
@st.cache_data(ttl=6 * 3600, show_spinner="ISIN wird geladen...")
def load_data(isin: str) -> pd.DataFrame | None:
    """Lädt Daten für eine ISIN und prüft Gültigkeit."""
    if not is_valid_isin(isin):
        st.error(f"Ungültige ISIN: {isin}")
        return None
    
    try:
        ticker = resolve_isin(isin)
        df = load_yahoo(ticker)
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten für {isin}: {e}")
        return None


# ------------------------------------------------------------------ Helper Functions
def compute_volatility(df: pd.DataFrame, years: int = 1) -> float:
    """Berechnet die Jahresvolatilität aus Log-Renditen."""
    returns = np.log(df['Close'].pct().dropna())
    daily_vol = returns.std() * math.sqrt(252)
    annual_vol = daily_vol * math.sqrt(years * 252 / 252)
    return annual_vol


def compute_52w_stats(df: pd.DataFrame) -> tuple[float, float]:
    """Berechnet 52-Wochen-Hoch und -Tief."""
    high_low = df[['High', 'Low']].dropna()
    if len(high_low) < 2:
        return 0.0, 0.0
    
    # Find the most recent 52 weeks of data (excluding weekends)
    mask = ~df['Date'].dt.dayofweek.isin([5, 6])  # Exclude Sat/Sun
    df_weekdays = df[mask].copy()
    
    if len(df_weekdays) < 13:
        return float(high_low['High'].max()), float(high_low['Low'].min())
    
    # Get last 52 weeks of weekdays
    n_weeks = min(52, len(df_weekdays))
    recent_data = df_weekdays.tail(n_weeks)
    
    high_52w = recent_data['High'].max()
    low_52w = recent_data['Low'].min()
    
    return float(high_52w), float(low_52w)


def compute_max_drawdown(df: pd.DataFrame, years: int = 1) -> tuple[float, float]:
    """Berechnet den maximalen Drawdown (Peak to Trough)."""
    returns = np.log(df['Close'].pct().dropna())
    
    # Calculate cumulative returns
    cum_returns = np.cumsum(returns)
    
    # Rolling maximum for drawdown calculation
    rolling_max = cum_returns.cummax()
    
    # Drawdown is the difference between peak and current value
    drawdowns = (rolling_max - cum_returns) / cum_returns.iloc[0] * 100
    
    max_dd = float(drawdowns.min())
    
    return max_dd, float(max_dd)


# ------------------------------------------------------------------ Main Application
def main():
    st.title("📉 Trailing-Stop-Risikoanalyse")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Konfiguration")
        
        isin_input = st.text_input(
            "ISIN", 
            value="IE00B4L5YC18",
            help="International Securities Identification Number"
        )
        
        years_slider = st.slider(
            "Historie (Jahre)",
            min_value=1,
            max_value=10,
            value=3,
            help="Zeitraum für Volatilitätsberechnung"
        )
        
        security_level = st.slider(
            "Sicherheitsniveau",
            min_value=90,
            max_value=99.5,
            value=95,
            step=1,
            help="Gewünschte Wahrscheinlichkeit, dass der Stop NICHT ausgelöst wird"
        )
        
        days_slider = st.slider(
            "Zeitraum (Tage)",
            min_value=1,
            max_value=252,
            value=10,
            help="Horizont für die Trailing-Stop-Berechnung"
        )
        
        model_select = st.selectbox(
            "Leitmodell",
            ["Markov", "Bootstrap", "GBM"],
            index=0,
            help="Wähle das primäre Simulationsmodell"
        )
        
        markov_states = st.slider(
            "Markov-Zustände",
            min_value=2,
            max_value=10,
            value=5,
            help="Anzahl der Volatilitäts-Cluster für Markov-Kette"
        )
        
        bootstrap_block_length = st.slider(
            "Bootstrap-Blocklänge (Tage)",
            min_value=3,
            max_value=20,
            value=5,
            help="Länge der neu zusammengesetzten Tagesblöcke"
        )
        
        gbm_drift = st.checkbox(
            "GBM mit historischer Drift",
            value=False,
            help="Ob historische Renditen als Drift verwendet werden sollen"
        )
    
    # Load data
    if isin_input:
        df = load_data(isin_input)
        
        if df is not None and len(df) > 0:
            st.success(f"Daten für {isin_input} geladen")
            
            # Display basic info
            col1, col2 = st.columns(2)
            with col1:
                current_price = float(df['Close'].iloc[-1])
                st.metric("Aktueller Kurs", f"{current_price:.2f} €")
            
            with col2:
                annual_vol = compute_volatility(df, years_slider)
                st.metric("Jahresvolatilität (σ)", f"{annual_vol:.4f}")
            
            high_52w, low_52w = compute_52w_stats(df)
            max_dd, _ = compute_max_drawdown(df, 1)
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("52-Wochen-Hoch", f"{high_52w:.2f} €")
            with col4:
                st.metric("52-Wochen-Tief", f"{low_52w:.2f} €")
            
            # Drawdown chart
            st.subheader("I. Ausgangsdaten")
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            ))
            
            # Mark drawdown peak and trough
            if max_dd < 0:
                fig_dd.add_vline(
                    x=df.index[df['Close'].idxmin()],
                    y=0,
                    line_dash="dash",
                    line_color="red"
                )
            
            st.plotly_chart(fig_dd, use_container_width=True)
        else:
            st.warning("Keine Daten verfügbar")
    else:
        st.warning("Bitte eine ISIN eingeben")


# ------------------------------------------------------------------ Simulation Engine
def simulate_trailing_stop(
    returns: np.ndarray,
    volatility: float,
    security_level: float,
    days: int,
    model: str,
    markov_states: int,
    bootstrap_block_length: int,
    gbm_drift: bool = False
) -> tuple[float, float]:
    """Simuliert Trailing-Stop-Auslösung für verschiedene Modelle."""
    
    # Calculate base volatility for the period
    vol_period = volatility / math.sqrt(252) * math.sqrt(days)
    
    # Target probability of NOT triggering
    prob_not_trigger = security_level / 100
    
    # Using reflection principle: P(trigger) = 2 * Φ(-b/σ√T) for fixed stop
    # For trailing stop, we use simulation results
    
    if model == "Markov":
        # Markov chain simulation
        states = np.linspace(0.01, volatility, markov_states)
        
        # Estimate transition matrix from historical data
        # (simplified - in production would use actual state sequences)
        n_transitions = min(len(returns), 500)
        if n_transitions < 2:
            return 0.0, 0.0
        
        # Simple uniform transition for demonstration
        P = np.random.rand(markov_states, markov_states)
        P /= P.sum(axis=1, keepdims=True)
        
        # Simulate path and trailing stop
        prices = np.exp(np.cumsum(returns))
        max_high = prices.iloc[0]
        trigger_price = None
        
        for i in range(len(prices)):
            current_price = prices[i]
            drawdown = (max_high - current_price) / max_high
            
            # Trailing stop triggers when drawdown exceeds threshold
            if drawdown >= security_level / 100:
                trigger_price = current_price
                break
        
        return float(trigger_price), vol_period
    
    elif model == "Bootstrap":
        # Block bootstrap simulation
        n_bootstrap = 5000
        triggered_paths = 0
        
        for _ in range(n_bootstrap):
            # Resample blocks of days
            block_size = min(bootstrap_block_length, len(returns))
            if block_size < 2:
                break
            
            bootstrap_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate prices with bootstrap returns
            bootstrap_prices = np.exp(np.cumsum(bootstrap_returns))
            max_high_bs = bootstrap_prices.iloc[0]
            
            # Check trailing stop
            for i in range(len(bootstrap_prices)):
                drawdown = (max_high_bs - bootstrap_prices[i]) / max_high_bs
                
                if drawdown >= security_level / 100:
                    triggered_paths += 1
                    break
        
        prob_trigger = triggered_paths / n_bootstrap
        return float(prob_trigger), vol_period
    
    else:  # GBM
        # Geometric Brownian Motion simulation
        n_simulations = 5000
        triggered_count = 0
        
        for _ in range(n_simulations):
            # Generate GBM path with optional drift
            if gbm_drift and len(returns) > 1:
                historical_return = returns[-1] - returns[0]
                drift = historical_return / (len(returns) * 252)
            else:
                drift = 0
            
            # Generate log-returns
            log_returns = np.random.normal(drift, vol_period, days)
            
            # Calculate prices
            gbm_prices = np.exp(np.cumsum(log_returns))
            max_high_gbm = gbm_prices.iloc[0]
            
            # Check trailing stop
            for i in range(len(gbm_prices)):
                drawdown = (max_high_gbm - gbm_prices[i]) / max_high_gbm
                
                if drawdown >= security_level / 100:
                    triggered_count += 1
                    break
        
        prob_trigger = triggered_count / n_simulations
        return float(prob_trigger), vol_period


# ------------------------------------------------------------------ Results Display
def display_results(
    isin: str,
    current_price: float,
    annual_volatility: float,
    security_level: float,
    days: int,
    max_drawdown: float,
    model: str,
    trigger_prob: float,
    half_life: float
):
    """Anzeigt die Simulationsergebnisse."""
    
    st.subheader("II. Herleitung")
    
    # Show formulas using LaTeX
    col1, col2 = st.columns(2)
    
    with col1:
        st.latex(r"\\sigma_{tag} = \\frac{\\sigma_{jahr}}{\\sqrt{252}}")
        st.latex(r"\\sigma_{10T} = \\sigma_{tag} \\cdot \\sqrt{10}")
    
    with col2:
        st.latex(r"P(\\text{Min} \\le -b) = 2 \\cdot \\Phi(-b/\\sigma_T)")
        st.latex(r"z = \\Phi^{-1}(1 - 0.05/2) = 1.96")
    
    # Calculate theoretical stop distance
    sigma_tag = annual_volatility / math.sqrt(252)
    sigma_10T = sigma_tag * math.sqrt(days)
    b_theoretical = 1.96 * sigma_10T
    
    st.write(f"**Theoretischer Stop-Abstand:** {b_theoretical:.4f} (log-Renditen)")
    
    # Simulation results
    st.subheader("III. Simulationsergebnisse")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(f"Trailing-Stop-Auslösewahrscheinlichkeit ({model})", 
                 f"{trigger_prob:.2%}")
        
        if half_life > 0:
            st.metric("Halbwertszeit", f"{half_life:.1f} Tage")
    
    with col2:
        st.metric(f"Maximaler Drawdown in {days} Tagen", f"{max_drawdown:.2%}")


# ------------------------------------------------------------------ Main Entry Point
if __name__ == "__main__":
    main()
