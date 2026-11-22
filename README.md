# trailstop-analyzer

Interactive risk analysis for trailing stops. Enter an ISIN, and the app finds the security, loads its price history and works out **how far below the price a trailing stop should sit** so that it does *not* trigger within a chosen period (default: 95 % probability over the next 10 trading days).

Instead of relying only on a volatility formula, it **simulates tens of thousands of price paths** with three models (Markov chain, historical block bootstrap, geometric Brownian motion) and checks the stop against each path, including intraday highs and lows. Closed-form formulas are shown next to the simulations for transparency.

The app's interface is in German. The output follows four sections:

| Section | Content |
|---|---|
| **I. Ausgangsdaten** | Current price, annual volatility (1 year and model history), real 52-week high/low, real 1-year maximum drawdown (closing prices and high→low) |
| **II. Herleitung** | √T scaling of volatility, reflection principle (path factor 2), trailing-stop formula (Lévy), simulated stop distance per model, recommended distance and first stop price |
| **III. Gegenrechnungen** | Half-life (days until the stop is hit with 50 % probability), comparison with the real 1-year max drawdown, probability of reaching that drawdown again within a year |
| **IV. Strategisches Fazit** | Table: computed distance and rounded distance ±1/±2 percentage points, with stop price, trigger probability and half-life, CSV download |

How the numbers are derived is documented in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

> This is a statistical model calculation based on historical prices, not investment advice.

---

## Setup

Requires [Miniconda/Anaconda](https://docs.conda.io/). The environment lives inside the project folder (`.conda/`).

```powershell
conda env create -p .\.conda -f environment.yml
```

The environment needs roughly 1–1.5 GB of disk space.

## Start the app

```powershell
.\.conda\python.exe -m streamlit run app.py
```

The app opens at http://localhost:8501.

## Run the tests

```powershell
.\.conda\python.exe -m pytest -q tests
```

---

## Using the app

### 1. Choose the security (sidebar)

- **ISIN** – e.g. `IE00B4L5YC18`. The ISIN is checked for format and check digit.
- **Börsenplatz / Ticker** – all listings found for the ISIN, sorted by preference (Xetra first, then other German exchanges, euro exchanges, others). If a listing returns no usable data, the app automatically falls back to the next one and shows a warning.
- **Ticker manuell** – override with any Yahoo Finance symbol (e.g. `EUNM.DE`) if the lookup picks the wrong security.
- **CSV-Upload** – use your own price file instead of an ISIN (see [CSV format](#csv-format)).

### 2. Set the analysis parameters

| Setting | Default | Meaning |
|---|---|---|
| Sicherheitsniveau | 0.95 | Probability that the stop does **not** trigger within the period |
| Überwachungszeitraum | 10 | Period in trading days |
| Historie für Modelle | 3 years | Price history used to fit the Markov chain and bootstrap and for the "model history" volatility |
| Vola-Basis | 1 Jahr | Volatility used by the formulas and the GBM simulation: last year or the full model history |
| Leitmodell | Markov-Kette | Model that sets the recommended stop distance (any simulation or formula can be chosen) |
| Markttrend | Ohne Trend | *Without trend*: pure random walk (historical average return removed). *Historical trend*: keeps the average return |

**Simulationseinstellungen** (expander):

| Setting | Default | Meaning |
|---|---|---|
| Anzahl Pfade | 10 000 | Simulated paths per model |
| Simulationslänge | 504 days | Length of each path; half-lives longer than this are shown as "> 504" |
| Markov-Zustände | 5 | Number of return buckets (states) in the Markov chain |
| Mittlere Blocklänge | 5 days | Average block length in the bootstrap |
| Zufalls-Seed | 42 | Fixed seed, so results are reproducible |

Every change recalculates the page. Simulation results are cached, so switching the leading model or the confidence level is instant; changing history, paths or model settings re-runs the simulation (a few seconds).

### 3. Read the results

- **Top row:** recommended trailing distance, first stop price, trigger probability within the period, half-life.
- **Section II table:** stop distance and implied σ-level for every method side by side.
- **Chart "Wahrscheinlichkeit der Auslösung … je Stop-Abstand":** for any distance on the x-axis, how likely each model says the stop is hit within the period. Where a curve crosses the red line is that model's optimal distance.
- **Expander "Simulierte Beispielpfade und Modelldetails":** sample paths with 5/50/95 % quantiles and the Markov transition matrix.
- **Section III:** half-life per model with the trigger probability over time; histogram of simulated 1-year max drawdowns against the real one.
- **Section IV:** final table and **"Vergleich aller Modelle"** with probability and half-life for every model. "Fazit als CSV herunterladen" saves it (semicolon-separated, decimal comma, opens directly in German Excel).

### CSV format

At least a date column and a closing price; open/high/low are optional but strongly recommended (without them, trailing-stop triggers can only be simulated on closing prices, which understates the risk; the app warns about this).

Recognised column names (case-insensitive): `Date/Datum`, `Open/Eröffnung`, `High/Hoch`, `Low/Tief`, `Close/Schluss/Schlusskurs/Kurs/Adj Close`. Comma or semicolon separators, German number formats (`1.234,50`) and dates (`31.12.2025`) are detected automatically.

```csv
Datum;Eröffnung;Hoch;Tief;Schlusskurs
02.01.2026;51,20;51,60;50,90;51,45
05.01.2026;51,45;51,80;51,10;51,70
```

---

## Project structure

```
app.py                  Streamlit app (UI, charts, sections I–IV)
environment.yml         Conda environment
trailstop/
  data.py               ISIN validation, ISIN → ticker (Yahoo search, OpenFIGI), price download, CSV import
  stats.py              Volatility, 52-week high/low, max drawdown, daily (close, high, low) tuples
  analytic.py           Closed-form formulas: √T scaling, reflection principle, Lévy trailing-stop formula, half-lives
  models.py             Path generators: Markov chain, stationary block bootstrap, GBM with Brownian-bridge intraday range
  engine.py             Trailing-stop evaluation on simulated paths (running max drawdown per path)
  report.py             Common interface for simulations and formulas, number formatting, final table
tests/test_core.py      Unit tests (formulas, simulation vs. formula, CSV import, statistics)
docs/METHODOLOGY.md     Detailed description of data, models and formulas
PLAN.md                 Original implementation plan
```

## Data sources

- **Yahoo Finance** (via [yfinance](https://github.com/ranaroussi/yfinance)) for daily OHLC prices, adjusted for distributions.
- **OpenFIGI** and Yahoo search to map an ISIN to exchange tickers.

Both are free and need no API key, but they can be rate-limited or incomplete. If a lookup fails, enter the ticker manually or upload a CSV.

## Limitations

- Free data: occasional gaps or bad intraday highs/lows; the most recent day may be incomplete during trading hours.
- Within a day the order of high and low is unknown; the simulation assumes 50/50.
- The models only know the chosen history: regimes that never occurred in it (e.g. a crash in a calm 3-year window) are not represented.
- No slippage or gap-through-stop execution prices are modelled; the analysis is about *whether* the stop triggers, not at which fill price.
