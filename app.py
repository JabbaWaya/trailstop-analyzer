"""Streamlit-App: Trailing-Stop-Risikoanalyse per ISIN.

Start:  .\\.conda\\python.exe -m streamlit run app.py
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
@st.cache_data(ttl=6 * 3600, show_spinner="ISIN wird aufgelöst …")
def cached_resolve(isin: str) -> list[tuple[str, str]]:
    return [(c.symbol, c.label) for c in resolve_isin(isin)]


@st.cache_data(ttl=3600, show_spinner="Kursdaten werden geladen …")
def cached_yahoo(symbol: str, years: int):
    return load_yahoo(symbol, years)


@st.cache_data(show_spinner=False)
def cached_csv(content: bytes) -> pd.DataFrame:
    return load_csv(content)


@st.cache_resource(max_entries=3, show_spinner="Pfade werden simuliert …")
def run_simulations(hist: np.ndarray, sig_day: float, mu_day: float, zero_drift: bool,
                    n_paths: int, sim_days: int, n_states: int, block: float, seed: int):
    ss = np.random.SeedSequence(seed)
    r_markov, r_boot, r_gbm = (np.random.default_rng(s) for s in ss.spawn(3))
    mm = fit_markov(hist, n_states)
    res = {
        MODEL_MARKOV: evaluate(MODEL_MARKOV, simulate_markov(hist, mm, n_paths, sim_days, r_markov, zero_drift), r_markov),
        MODEL_BOOT: evaluate(MODEL_BOOT, simulate_bootstrap(hist, block, n_paths, sim_days, r_boot, zero_drift), r_boot),
        MODEL_GBM: evaluate(MODEL_GBM, simulate_gbm(sig_day, 0.0 if zero_drift else mu_day, n_paths, sim_days, r_gbm), r_gbm),
    }
    return res, mm


def latex(expr: str):
    """LaTeX mit deutschem Dezimalkomma."""
    st.latex(re.sub(r"(\d)\.(\d)", r"\1{,}\2", expr))


def plot(fig: go.Figure, height: int = 380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------ Sidebar
with st.sidebar:
    st.header("Eingaben")
    isin = st.text_input("ISIN", value="IE00B4L5YC18").strip().upper()
    upload = st.file_uploader("…oder Kurs-CSV hochladen (Datum, Open, High, Low, Close)", type=["csv", "txt"])

    symbols: list[str] = []  # gewählter Ticker zuerst, danach Ausweich-Ticker
    if upload is None:
        if not is_valid_isin(isin):
            st.error("Ungültige ISIN (Format/Prüfziffer).")
            st.stop()
        candidates = cached_resolve(isin)
        manual = st.text_input("Ticker manuell (optional, Yahoo-Format, z. B. EUNM.DE)", value="").strip()
        if manual:
            symbols = [manual]
        elif candidates:
            labels = [lbl for _, lbl in candidates]
            choice = st.selectbox("Börsenplatz / Ticker", labels, index=0)
            i = labels.index(choice)
            symbols = [s for s, _ in candidates[i:]] + [s for s, _ in candidates[:i]]
        else:
            st.error("Kein Ticker zur ISIN gefunden – bitte Ticker manuell eingeben oder CSV hochladen.")
            st.stop()

    st.divider()
    confidence = st.slider("Sicherheitsniveau (keine Auslösung)", 0.80, 0.995, 0.95, 0.005, format="%.3f")
    horizon = st.number_input("Überwachungszeitraum (Börsentage)", 1, 60, 10)
    model_years = st.slider("Historie für Modelle (Jahre)", 1, 10, 3)
    vol_basis = st.radio("Vola-Basis für Formeln und GBM", ["1 Jahr", "Modellhistorie"], horizontal=True)
    lead = st.selectbox("Leitmodell (bestimmt den empfohlenen Abstand)", ALL_MODELS, index=0)
    drift_mode = st.radio("Markttrend", ["Ohne Trend (reiner Zufallspfad)", "Historischer Trend"])
    zero_drift = drift_mode.startswith("Ohne")

    with st.expander("Simulationseinstellungen"):
        n_paths = st.select_slider("Anzahl Pfade", [2000, 5000, 10000, 20000, 40000], value=10000)
        sim_days = st.select_slider("Simulationslänge (Börsentage)", [252, 378, 504, 756], value=504)
        n_states = st.slider("Markov-Zustände", 3, 9, 5)
        block = st.slider("Mittlere Blocklänge Bootstrap (Tage)", 1, 20, 5)
        seed = st.number_input("Zufalls-Seed", 0, 10_000, 42)

# ------------------------------------------------------------------ Daten
if upload is not None:
    try:
        df_full = cached_csv(upload.getvalue())
    except Exception as exc:  # noqa: BLE001
        st.error(f"CSV konnte nicht gelesen werden: {exc}")
        st.stop()
    name, source = upload.name, "CSV"
    currency = st.sidebar.text_input("Währung der CSV", "EUR").upper()
else:
    errors = []
    for symbol in symbols:
        try:
            df_full, currency, name = cached_yahoo(symbol, model_years)
            if len(df_full) >= 120:
                break
            errors.append(f"{symbol}: nur {len(df_full)} Tage")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: {exc}")
    else:
        st.error("Kursdaten konnten nicht geladen werden:\n\n" + "\n\n".join(errors))
        st.stop()
    if symbol != symbols[0]:
        st.warning(f"{symbols[0]} lieferte keine verwertbaren Daten – verwende {symbol}.")
    source = f"Yahoo Finance ({symbol})"

if len(df_full) < 120:
    st.error("Zu wenige Kursdaten (< 120 Handelstage).")
    st.stop()

cur = "€" if currency in ("EUR", "") else currency
stats, df_model = compute_base_stats(df_full, model_years)
hist = day_tuples(df_model)
vol = stats.vol_annual_1y if vol_basis == "1 Jahr" else stats.vol_annual
sig_d = an.sigma_day(vol)
sig_h = an.sigma_horizon(vol, horizon)
P0 = stats.price

if sim_days < 252:
    sim_days = 252
sims, markov_model = run_simulations(hist, sig_d, stats.mean_log_return, zero_drift,
                                     int(n_paths), int(sim_days), int(n_states), float(block), int(seed))
models = {name_: ModelView(name_, sim=sims[name_]) for name_ in sims}
models[MODEL_LEVY] = ModelView(MODEL_LEVY, sig_day=sig_d)
models[MODEL_REFL] = ModelView(MODEL_REFL, sig_day=sig_d)

b_opt = models[lead].stop(confidence, horizon)
d_opt = an.log_to_pct(b_opt)
stop_price = P0 * (1 - d_opt)
alpha = 1 - confidence

# ------------------------------------------------------------------ Kopf
st.title("Trailing-Stop-Risikoanalyse")
st.caption(f"{name} · ISIN {isin if upload is None else '–'} · Quelle: {source} · "
           f"Stand {stats.date:%d.%m.%Y} · Modellhistorie ab {stats.history_start:%d.%m.%Y} ({stats.n_days} Tage)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Empfohlener Trailing-Abstand", fmt_pct(d_opt), help=f"Leitmodell: {lead}")
c2.metric("Erster Stop-Kurs", f"{fmt_num(stop_price)} {cur}")
c3.metric(f"P(Auslösung ≤ {horizon} Tage)", fmt_pct(models[lead].p(b_opt, horizon), 1))
c4.metric("Halbwertszeit", f"{fmt_days(models[lead].halflife(b_opt), sim_days)} Tage")

if not stats.has_intraday:
    st.warning("Die Daten enthalten kaum Tageshoch/-tief-Spannen. Simulierte Auslösungen beruhen dann "
               "faktisch nur auf Schlusskursen und unterschätzen das Risiko.")

# ================================================================== I
st.header("I. Ausgangsdaten")
y1 = last_year(df_full)
m = st.columns(6)
m[0].metric("Aktueller Kurs", f"{fmt_num(P0)} {cur}", help=f"Schlusskurs {stats.date:%d.%m.%Y}")
m[1].metric("Jahres-Vola (1 J.)", fmt_pct(stats.vol_annual_1y), help="σ der täglichen Log-Renditen · √252")
m[2].metric(f"Jahres-Vola ({model_years} J.)", fmt_pct(stats.vol_annual))
m[3].metric("52-Wochen-Hoch", f"{fmt_num(stats.high_52w)} {cur}", help=f"Tageshoch am {stats.high_52w_date:%d.%m.%Y}")
m[4].metric("52-Wochen-Tief", f"{fmt_num(stats.low_52w)} {cur}", help=f"Tagestief am {stats.low_52w_date:%d.%m.%Y}")
m[5].metric("Max. Drawdown 1 J.", fmt_pct(-stats.mdd_close.depth),
            help=f"Schlusskurse: Peak {stats.mdd_close.peak_date:%d.%m.%Y} ({fmt_num(stats.mdd_close.peak_price)}) "
                 f"→ Tief {stats.mdd_close.trough_date:%d.%m.%Y} ({fmt_num(stats.mdd_close.trough_price)})")

st.markdown(
    f"- **Max. Drawdown (Schlusskurse):** {fmt_pct(stats.mdd_close.depth)} — Peak "
    f"{stats.mdd_close.peak_date:%d.%m.%Y} ({fmt_num(stats.mdd_close.peak_price)} {cur}) → Tief "
    f"{stats.mdd_close.trough_date:%d.%m.%Y} ({fmt_num(stats.mdd_close.trough_price)} {cur})\n"
    f"- **Max. Drawdown (Tageshoch → Tagestief, maßgeblich für Stops):** {fmt_pct(stats.mdd_intraday.depth)} — "
    f"{stats.mdd_intraday.peak_date:%d.%m.%Y} → {stats.mdd_intraday.trough_date:%d.%m.%Y}\n"
    f"- Abstand zum 52W-Hoch: {fmt_pct(P0 / stats.high_52w - 1)}, zum 52W-Tief: {fmt_pct(P0 / stats.low_52w - 1)}"
)

fig = go.Figure()
fig.add_trace(go.Scatter(x=y1.index, y=y1["High"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=y1.index, y=y1["Low"], fill="tonexty", line=dict(width=0),
                         fillcolor="rgba(42,111,219,0.18)", name="Tagesspanne"))
fig.add_trace(go.Scatter(x=y1.index, y=y1["Close"], name="Schlusskurs", line=dict(color="#2a6fdb", width=2)))
fig.add_hline(y=stats.high_52w, line_dash="dot", line_color="#2e9e6a", annotation_text="52W-Hoch")
fig.add_hline(y=stats.low_52w, line_dash="dot", line_color="#c0392b", annotation_text="52W-Tief")
fig.add_hline(y=stop_price, line_dash="dash", line_color="#e07a1f", annotation_text="Erster Stop")
fig.add_trace(go.Scatter(x=[stats.mdd_close.peak_date, stats.mdd_close.trough_date],
                         y=[stats.mdd_close.peak_price, stats.mdd_close.trough_price], mode="markers+lines",
                         marker=dict(size=10, color="#c0392b"), line=dict(dash="dot", color="#c0392b"),
                         name=f"Max. Drawdown {fmt_pct(-stats.mdd_close.depth, 1)}"))
fig.update_layout(title="Kursverlauf letzte 12 Monate", yaxis_title=cur)
plot(fig, 420)

# ================================================================== II
st.header("II. Herleitung des optimalen Stop-Abstands")

st.subheader("1. Wurzel-T-Gesetz")
latex(rf"\sigma_{{Tag}} = \frac{{\sigma_{{Jahr}}}}{{\sqrt{{252}}}} = \frac{{{vol*100:.2f}\,\%}}{{15{{,}}87}} = {sig_d*100:.3f}\,\%")
latex(rf"\sigma_{{{horizon}T}} = \sigma_{{Tag}}\cdot\sqrt{{{horizon}}} = {sig_d*100:.3f}\,\% \cdot {math.sqrt(horizon):.3f} = {sig_h*100:.3f}\,\%")

st.subheader("2. Kontinuierlicher Pfad-Faktor (Reflexionsprinzip)")
z_naive = norm.ppf(confidence)
z_refl = an.z_fixed(confidence)
b_levy = an.b_trailing(confidence, sig_d, horizon)
z_levy = b_levy / sig_h
st.markdown(
    f"Eine reine Endpunkt-Betrachtung prüft nur, ob der Kurs **am Ende** von {horizon} Tagen unter dem Stop liegt "
    f"(z = Φ⁻¹({fmt_num(confidence, 3)}) = {fmt_num(z_naive, 3)}). Der Stop löst aber aus, sobald der Kurs ihn **irgendwann** "
    f"im Zeitraum berührt. Nach dem Reflexionsprinzip wird jeder Pfad, der die Schranke berührt, mit gleicher "
    f"Wahrscheinlichkeit darunter oder darüber enden – die Berührungswahrscheinlichkeit ist daher **doppelt** so "
    f"groß wie die Endpunkt-Wahrscheinlichkeit (Pfad-Faktor 2):")
latex(rf"P\Big(\min_{{t\le T}} \ln\tfrac{{S_t}}{{S_0}} \le -b\Big) = 2\,\Phi\!\Big(-\frac{{b}}{{\sigma_T}}\Big) \stackrel{{!}}{{=}} {alpha:.3f}"
         rf"\;\Rightarrow\; z = \Phi^{{-1}}\!\Big(1-\tfrac{{{alpha:.3f}}}{{2}}\Big) = {z_refl:.3f}")

st.subheader("3. Sigma-Niveau und Stop-Abstand")
rows = []
for label, z, b in [
    ("Naiv (nur Endpunkt)", z_naive, z_naive * sig_h),
    ("Fester Stop (Reflexion, Pfad-Faktor 2)", z_refl, z_refl * sig_h),
    ("Nachlaufender Stop (Lévy-Formel)", z_levy, b_levy),
]:
    d = an.log_to_pct(b)
    rows.append({"Methode": label, "z (σ-Niveau)": z, "Abstand": d, "Stop-Kurs": P0 * (1 - d)})
for name_ in (MODEL_MARKOV, MODEL_BOOT, MODEL_GBM):
    b = models[name_].stop(confidence, horizon)
    d = an.log_to_pct(b)
    rows.append({"Methode": f"Simulation: {name_}", "z (σ-Niveau)": b / sig_h, "Abstand": d, "Stop-Kurs": P0 * (1 - d)})
deriv = pd.DataFrame(rows)
latex(rf"b = z\cdot\sigma_{{{horizon}T}} = {z_refl:.3f}\cdot{sig_h*100:.3f}\,\% = {z_refl*sig_h*100:.3f}\,\%"
         rf"\quad\Rightarrow\quad \text{{Abstand}} = 1-e^{{-b}} = {an.log_to_pct(z_refl*sig_h)*100:.2f}\,\%")
st.dataframe(
    deriv.style.format({"z (σ-Niveau)": lambda x: fmt_num(x, 3), "Abstand": lambda x: fmt_pct(x),
                        "Stop-Kurs": lambda x: f"{fmt_num(x)} {cur}"}),
    hide_index=True, width="stretch")
st.info(
    f"**Wichtig:** Das Reflexionsprinzip gilt für einen *festen* Stop. Ein *nachlaufender* Stop wird mit jedem neuen "
    f"Hoch nachgezogen; sein Auslöser ist der Drawdown vom laufenden Hoch. Nach Lévy verhält sich dieser Drawdown "
    f"wie |W_t| – ein Zwei-Schranken-Problem, daher z ≈ {fmt_num(z_levy)} statt {fmt_num(z_refl)}. Die Simulationen bilden "
    f"zusätzlich Fat Tails, Volatilitäts-Cluster (Markov) und echte Tagesspannen ab.")
st.success(f"**Empfehlung (Leitmodell {lead}):** Trailing-Abstand **{fmt_pct(d_opt)}** → erster Stop-Kurs "
           f"**{fmt_num(stop_price)} {cur}** (Kurs {fmt_num(P0)} {cur}).")

# Kurve P(Auslösung) über Abstand
grid = np.linspace(0.002, max(0.25, d_opt * 2.5), 160)
fig = go.Figure()
for name_, mv in models.items():
    ps = [mv.p(an.pct_to_log(d), horizon) for d in grid]
    fig.add_trace(go.Scatter(x=grid * 100, y=np.array(ps) * 100, name=name_,
                             line=dict(color=MODEL_COLORS[name_], width=3 if name_ == lead else 1.6,
                                       dash="dot" if not mv.is_sim else None)))
fig.add_hline(y=alpha * 100, line_dash="dash", line_color="#c0392b", annotation_text=f"{alpha*100:.1f} %")
fig.add_vline(x=d_opt * 100, line_dash="dash", line_color="#444")
fig.update_layout(title=f"Wahrscheinlichkeit der Auslösung innerhalb {horizon} Tagen je Stop-Abstand",
                  xaxis_title="Trailing-Abstand (%)", yaxis_title="P(Auslösung) %", yaxis_range=[0, 60])
plot(fig)

with st.expander("Simulierte Beispielpfade und Modelldetails"):
    lead_sim = sims.get(lead, sims[MODEL_MARKOV])
    show_days = max(horizon, 20)
    t = np.arange(0, show_days + 1)
    fig = go.Figure()
    sample = lead_sim.close_log[:60, :show_days]
    for i in range(sample.shape[0]):
        fig.add_trace(go.Scatter(x=t, y=P0 * np.exp(np.concatenate([[0], sample[i]])), mode="lines",
                                 line=dict(width=0.8, color="rgba(42,111,219,0.35)"), showlegend=False,
                                 hoverinfo="skip"))
    qs = np.quantile(lead_sim.close_log[:, :show_days], [0.05, 0.5, 0.95], axis=0)
    for q, lbl in zip(qs, ["5 %-Quantil", "Median", "95 %-Quantil"]):
        fig.add_trace(go.Scatter(x=t, y=P0 * np.exp(np.concatenate([[0], q])), name=lbl, line=dict(width=2)))
    fig.add_hline(y=stop_price, line_dash="dash", line_color="#e07a1f", annotation_text="Erster Stop")
    fig.add_vline(x=horizon, line_dash="dot")
    fig.update_layout(title=f"Schlusskurspfade ({lead_sim.name})", xaxis_title="Börsentage", yaxis_title=cur)
    plot(fig)

    st.markdown(f"**Markov-Kette:** {n_states} Zustände nach Tagesrendite-Quantilen, Start in "
                f"*{markov_model.labels[markov_model.start_state]}* (heutiger Zustand). Pro Zustand wird ein echter "
                f"historischer Tag inkl. Hoch/Tief gezogen. Diagonale > {1/n_states:.2f} bedeutet Persistenz.")
    fig = go.Figure(go.Heatmap(z=markov_model.transition, x=markov_model.labels, y=markov_model.labels,
                               colorscale="Blues", text=np.round(markov_model.transition, 2),
                               texttemplate="%{text}", colorbar=dict(title="P")))
    fig.update_layout(title="Übergangsmatrix (Zeile = heute, Spalte = morgen)", yaxis_autorange="reversed")
    plot(fig, 420)
    st.caption("Innerhalb eines Tages ist unbekannt, ob Hoch oder Tief zuerst kam; die Simulation legt die "
               "Reihenfolge je Tag zufällig fest (50/50). GBM-Tagesspannen folgen der exakten Max/Min-Verteilung "
               "einer Brownschen Brücke.")

# ================================================================== III
st.header("III. Gegenrechnungen & Realitätschecks")
st.subheader("Halbwertszeit")
hl_fixed = an.halflife_fixed(b_opt, sig_d)
latex(rf"2\,\Phi\!\Big(-\frac{{b}}{{\sigma_{{Tag}}\sqrt{{n}}}}\Big)=0{{,}}5 \;\Rightarrow\; "
         rf"n = \Big(\frac{{b}}{{0{{,}}6745\cdot\sigma_{{Tag}}}}\Big)^2 = "
         rf"\Big(\frac{{{b_opt*100:.3f}\,\%}}{{0{{,}}6745\cdot{sig_d*100:.3f}\,\%}}\Big)^2 = {hl_fixed:.1f}\ \text{{Tage}}")
hl_rows = [{"Modell": n_, "Halbwertszeit (Börsentage)": fmt_days(mv.halflife(b_opt), sim_days)} for n_, mv in models.items()]
cA, cB = st.columns([1, 2])
cA.dataframe(pd.DataFrame(hl_rows), hide_index=True, width="stretch")
cA.caption(f"Für den Abstand {fmt_pct(d_opt)}. Ohne Markttrend: 50 % aller Zufallspfade haben den Stop bis "
           f"dahin ausgelöst.")
with cB:
    days_axis = np.arange(1, sim_days + 1)
    fig = go.Figure()
    for name_, mv in models.items():
        fig.add_trace(go.Scatter(x=days_axis, y=mv.curve(b_opt, sim_days) * 100, name=name_,
                                 line=dict(color=MODEL_COLORS[name_], width=3 if name_ == lead else 1.6,
                                           dash="dot" if not mv.is_sim else None)))
    fig.add_hline(y=50, line_dash="dash", annotation_text="50 %")
    fig.add_vline(x=horizon, line_dash="dot", annotation_text=f"{horizon} T")
    fig.update_layout(title=f"P(Auslösung) über die Zeit bei {fmt_pct(d_opt)} Abstand",
                      xaxis_title="Börsentage", yaxis_title="%", xaxis_type="log")
    plot(fig)

st.subheader("Abgleich mit dem realen 1-Jahres-Max-Drawdown")
mdd = stats.mdd_intraday.depth
mdd_log = an.pct_to_log(mdd)
st.markdown(
    f"Realer 1J-Max-Drawdown (Hoch → Tief): **{fmt_pct(mdd)}**. Der Stop-Abstand {fmt_pct(d_opt)} entspricht "
    f"**{fmt_num(d_opt / mdd)} ×** dem MDD — der Stop hätte im letzten Jahr also "
    f"{'**ausgelöst**' if d_opt < mdd else '**nicht** ausgelöst'} "
    f"(sofern er am Peak {stats.mdd_intraday.peak_date:%d.%m.%Y} aktiv gewesen wäre).")
mdd_rows = []
for name_, mv in models.items():
    mdd_rows.append({"Modell": name_,
                     "P(1J-MDD ≥ realer MDD)": fmt_pct(mv.p_mdd(mdd_log, 252), 1),
                     "P(1J-MDD ≥ Stop-Abstand)": fmt_pct(mv.p_mdd(b_opt, 252), 1)})
cA, cB = st.columns([1, 2])
cA.dataframe(pd.DataFrame(mdd_rows), hide_index=True, width="stretch")
cA.caption("„Fester Stop (Reflexion)“ misst hier nur den Fall vom heutigen Kurs, nicht vom laufenden Hoch.")
with cB:
    fig = go.Figure()
    for name_ in (MODEL_MARKOV, MODEL_BOOT, MODEL_GBM):
        dd1 = (1 - np.exp(-sims[name_].maxdd[:, 251].astype(float))) * 100
        fig.add_trace(go.Histogram(x=dd1, name=name_, opacity=0.55, nbinsx=80, histnorm="percent",
                                   marker_color=MODEL_COLORS[name_]))
    fig.add_vline(x=mdd * 100, line_color="#c0392b", line_dash="dash", annotation_text="realer MDD")
    fig.add_vline(x=d_opt * 100, line_color="#444", line_dash="dot", annotation_text="Stop")
    fig.update_layout(barmode="overlay", title="Verteilung des simulierten 1-Jahres-Max-Drawdowns",
                      xaxis_title="Max. Drawdown (%)", yaxis_title="% der Pfade")
    plot(fig)

# ================================================================== IV
st.header("IV. Strategisches Fazit")
table = summary_table(models, lead, d_opt, P0, horizon, sim_days)
p_col = f"P(Auslösung ≤ {horizon} T)"
show = pd.DataFrame({
    "Variante": table["Variante"],
    "Trailing-Abstand": table["Trailing-Abstand"].map(lambda x: fmt_pct(x)),
    "Stop-Kurs": table["Stop-Kurs"].map(lambda x: f"{fmt_num(x)} {cur}"),
    p_col: table[p_col].map(lambda x: fmt_pct(x, 1)),
    "Halbwertszeit (Börsentage)": table["Halbwertszeit (T)"].map(lambda x: fmt_days(x, sim_days)),
})
st.markdown(f"Leitmodell: **{lead}** · Sicherheitsniveau {fmt_num(confidence*100, 1)} % · {horizon} Börsentage")
st.dataframe(show, hide_index=True, width="stretch")

with st.expander("Vergleich aller Modelle"):
    comp = {"Variante": table["Variante"], "Abstand": table["Trailing-Abstand"].map(lambda x: fmt_pct(x))}
    comp[f"P {lead}"] = table[p_col].map(lambda x: fmt_pct(x, 1))
    comp[f"HWZ {lead}"] = table["Halbwertszeit (T)"].map(lambda x: fmt_days(x, sim_days))
    for col in table.columns:
        if col.startswith("P "):
            comp[col] = table[col].map(lambda x: fmt_pct(x, 1))
        elif col.startswith("HWZ "):
            comp[col] = table[col].map(lambda x: fmt_days(x, sim_days))
    st.dataframe(pd.DataFrame(comp), hide_index=True, width="stretch")

st.download_button("Fazit als CSV herunterladen", table.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                   file_name=f"trailing_stop_{isin}.csv", mime="text/csv")

p_lead = models[lead].p(b_opt, horizon)
st.markdown(
    f"**Kurzfazit:** Mit {fmt_pct(d_opt)} Abstand (Stop {fmt_num(stop_price)} {cur}) liegt die Auslösewahrscheinlichkeit "
    f"in {horizon} Börsentagen bei {fmt_pct(p_lead, 1)}. Ohne Trend wird der Stop nach etwa "
    f"{fmt_days(models[lead].halflife(b_opt), sim_days)} Börsentagen mit 50 % Wahrscheinlichkeit erreicht. "
    f"Der reale 1J-Max-Drawdown von {fmt_pct(mdd)} ist {fmt_num(mdd / d_opt, 1)}-mal so groß wie der Abstand.")
st.caption("Statistische Modellrechnung auf Basis historischer Kurse, keine Anlageberatung. Kurslücken, "
           "Ausführungs-Slippage und Regimewechsel außerhalb der Historie sind nicht vollständig abgebildet.")
