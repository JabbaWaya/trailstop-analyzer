# Methodology

What the app computes, step by step, and where in the code it happens. Notation: $S_t$ price, $\ln$ natural log, $\Phi$ standard normal CDF, $\alpha = 1 - \text{confidence}$ (e.g. 0.05), $T$ = monitoring period in trading days (e.g. 10).

All distances are handled internally as **log distances** $b$ and converted to percent at the end:

$$d = 1 - e^{-b}, \qquad b = -\ln(1-d)$$

A stop at distance $d$ below a peak $P$ sits at $P\,(1-d)$.

---

## 1. Data (`trailstop/data.py`)

1. **ISIN check:** format (2 letters + 9 alphanumerics + check digit) and Luhn check digit.
2. **ISIN → tickers:** Yahoo Finance search and the OpenFIGI mapping API. OpenFIGI exchange codes are translated to Yahoo suffixes (e.g. `GY → .DE`, `GR → .F`, `NA → .AS`). Because Yahoo often omits the Xetra line, a `.DE` candidate is derived from every German regional listing (`.F`, `.SG`, `.MU`, …).
3. **Ranking:** Xetra (`.DE`) → other German exchanges → euro exchanges (`.AS`, `.PA`, `.MI`, …) → `.SW`, `.L` → US → others. If the chosen ticker fails or has fewer than 120 days, the next one is tried.
4. **Prices:** daily OHLC from Yahoo with `auto_adjust=True` (adjusted for dividends/splits), history = model years + 1 year.
5. **Cleaning:** rows without a positive close are dropped; missing open = previous close; `High ≥ max(Open, Close)` and `Low ≤ min(Open, Close)` are enforced; one row per calendar day.

## 2. Input data – Section I (`trailstop/stats.py`)

| Figure | Definition |
|---|---|
| Current price $S_0$ | Last close |
| Daily log return | $r_t = \ln(C_t / C_{t-1})$ |
| Annual volatility | $\sigma_{Jahr} = \text{std}(r_t)\cdot\sqrt{252}$ (sample std), once for the **last year** and once for the **model history** |
| 52-week high / low | Highest daily **high** / lowest daily **low** of the last 365 calendar days |
| Max drawdown (closing prices) | $\max_t \big(1 - C_t / \max_{s\le t} C_s\big)$ over the last year, with peak and trough dates |
| Max drawdown (high → low) | $\max_t \big(1 - L_t / \max_{s\le t} H_s\big)$ – the relevant figure for stops, since a stop trails highs and is hit by lows |

For the simulations each historical day is stored as a **day tuple** relative to the previous close:

$$\big(r_t,\ h_t,\ l_t\big) = \Big(\ln\tfrac{C_t}{C_{t-1}},\ \ln\tfrac{H_t}{C_{t-1}},\ \ln\tfrac{L_t}{C_{t-1}}\Big), \quad h_t \ge r_t \ge l_t$$

This keeps real intraday ranges, gaps and fat tails together with each day's return.

## 3. Formulas – Section II (`trailstop/analytic.py`)

All formulas assume a driftless Brownian motion of the log price with daily volatility $\sigma_{Tag}$ (from the chosen *Vola-Basis*).

### 3.1 Square-root-of-time rule

$$\sigma_{Tag} = \frac{\sigma_{Jahr}}{\sqrt{252}}, \qquad \sigma_T = \sigma_{Tag}\sqrt{T}$$

### 3.2 Naive endpoint view

Only checks whether the price is below the stop **at the end** of the period: $P(\ln S_T/S_0 \le -b) = \Phi(-b/\sigma_T)$, so $z = \Phi^{-1}(1-\alpha)$ (1.645 for 95 %). This understates the risk because a stop triggers as soon as it is touched **at any time**.

### 3.3 Fixed stop: reflection principle (path factor 2)

Every path that touches the level $-b$ is, from that moment on, equally likely to end above or below it. Hence the probability of touching is twice the probability of ending below:

$$P\Big(\min_{t\le T}\ln\tfrac{S_t}{S_0} \le -b\Big) = 2\,\Phi\Big(-\frac{b}{\sigma_T}\Big) \stackrel{!}{=} \alpha
\;\Rightarrow\; z = \Phi^{-1}\big(1-\tfrac{\alpha}{2}\big),\quad b = z\,\sigma_T$$

$z = 1.960$ for 95 %, $2.576$ for 99 %. This is the method described in the original task. It is exact for a **fixed** stop placed below today's price.

### 3.4 Trailing stop: Lévy's theorem

A trailing stop is moved up with every new high, so it is triggered by the **drawdown from the running maximum** $M_t - W_t$, not by the distance from the start. By Lévy's theorem, $M_t - W_t$ has the same law as $|W_t|$ (reflected Brownian motion). The maximum drawdown up to $T$ therefore has the law of $\sup_{s\le T}|W_s|$ – a two-barrier exit problem. With $x = b/\sigma_T$ and the method of images:

$$P(\text{trigger by } T) = 1 - \sum_{k=-\infty}^{\infty} (-1)^k\Big[\Phi\big((1-2k)\,x\big) - \Phi\big((-1-2k)\,x\big)\Big]$$

For small probabilities this is ≈ $4\,\Phi(-x)$, i.e. **twice the fixed-stop probability**. Solving for $\alpha$ (root finding, `b_trailing`) gives $z \approx 2.241$ for 95 % and $2.807$ for 99 %. The app shows this as *"Formel: nachlaufend (Lévy)"*.

### 3.5 Half-life

Number of trading days $n$ after which the stop is hit with exactly 50 % probability (no trend):

- Fixed stop: $2\Phi\big(-b/(\sigma_{Tag}\sqrt n)\big) = 0.5 \Rightarrow n = \big(b / (0.6745\,\sigma_{Tag})\big)^2$
- Trailing stop (Lévy): $P$ depends only on $x = b/(\sigma_{Tag}\sqrt n)$; it is 50 % at $x_{50} \approx 1.149$, so $n = \big(b/(1.149\,\sigma_{Tag})\big)^2$

## 4. Simulations (`trailstop/models.py`)

Each model produces $N$ paths (default 10 000) of $D$ days (default 504) of day tuples $(r, h, l)$. Each model gets its own random generator derived from the seed.

**Trend:** with *"Ohne Trend"* the mean daily return of the history is subtracted from $r$, $h$ and $l$ alike (the shape of each day stays intact, the drift becomes zero). With *"Historischer Trend"* the data is used as is, and the GBM gets the historical mean log return as drift.

### 4.1 Markov chain (default leading model)

1. Split the historical daily returns into $k$ quantile buckets (default 5) = states, e.g. *strong down, down, flat, up, strong up*.
2. Estimate the transition matrix $P_{ij}$ = P(state $j$ tomorrow | state $i$ today) from consecutive days, with Laplace smoothing (+0.5 per cell).
3. Simulate: start in **today's** state, draw the next state from the row of the current state, then draw a **real historical day** (including its high and low) uniformly from that state's bucket.

This captures volatility clustering (large moves tend to follow large moves) and starts from the current market regime, while each simulated day is a real day. The matrix is shown as a heatmap in the app.

### 4.2 Stationary block bootstrap

Politis–Romano stationary bootstrap: paths are stitched together from **blocks of consecutive real days**. At each step the path continues with the next historical day, or with probability $1/L$ (default $L = 5$) jumps to a random day. Block lengths are therefore geometric with mean $L$. Model-free; keeps fat tails and short-term autocorrelation.

### 4.3 GBM Monte Carlo

$r_t = \mu + \sigma_{Tag}\,Z_t$ with $Z_t \sim N(0,1)$ ($\mu = 0$ without trend). Intraday high and low come from the exact distribution of the maximum and minimum of a **Brownian bridge** from $0$ to $r_t$ over one day:

$$h_t = \tfrac12\Big(r_t + \sqrt{r_t^2 - 2\sigma_{Tag}^2\ln U_1}\Big),\qquad l_t = \tfrac12\Big(r_t - \sqrt{r_t^2 - 2\sigma_{Tag}^2\ln U_2}\Big)$$

($U_1, U_2$ uniform; drawn independently, so their joint behaviour is approximate.) This is the baseline: it should – and does – reproduce the Lévy formula.

## 5. Trailing-stop evaluation (`trailstop/engine.py`)

For every path:

1. Log close $c_t = \sum_{s\le t} r_s$ ($c_0 = 0$ is today's price), daily high $c_{t-1} + h_t$, daily low $c_{t-1} + l_t$.
2. **Peak** = highest daily high so far, starting at today's price. Within a day it is unknown whether the high came before the low; the order is drawn 50/50 per day. If the high came first, today's high already lifts the stop before the low is tested.
3. **Drawdown** on day $t$: peak − daily low (log). **Running max drawdown** $DD_t$ = largest drawdown up to day $t$; it never decreases.

A stop with log distance $b$ has triggered by day $t$ exactly when $DD_t \ge b$. This makes every distance evaluable on the **same** paths (common random numbers), without re-simulating:

| Result | Computation |
|---|---|
| Optimal distance | $b^* = $ confidence-quantile of $DD_T$ over all paths (e.g. 95th percentile of the 10-day max drawdown) |
| Trigger probability for any $b$ | share of paths with $DD_T \ge b$ |
| Trigger probability over time | share of paths with $DD_t \ge b$, for $t = 1..D$ |
| Half-life | first day $t$ on which the **median** of $DD_t$ over all paths is $\ge b$; "> D" if never within the simulated length |
| P(1-year MDD ≥ real MDD) | share of paths with $DD_{252} \ge \ln\big(1/(1-\text{MDD}_{real})\big)$, using the high→low drawdown |

## 6. Sections III and IV (`trailstop/report.py`, `app.py`)

- **Leading model:** any of the three simulations or two formulas can be chosen. All results use one interface (`ModelView`: stop, probability, curve, half-life), so tables and charts treat simulations and formulas alike.
- **Max-drawdown check:** ratio stop distance / real 1-year max drawdown; whether the stop would have triggered last year; probability per model that a 1-year path reaches the real max drawdown (for the fixed-stop formula this is the probability of falling that far below today's price, not below a running peak).
- **Final table:** the computed distance $d^*$ plus $\text{round}(d^*)$ and $\pm 1$, $\pm 2$ percentage points (non-positive values are skipped). For each: stop price $S_0(1-d)$, trigger probability within $T$ days and half-life for the leading model; the comparison expander adds the same figures for every other model.

## 7. Validation (`tests/test_core.py`)

- $z$ for the reflection principle equals 1.96 and gives exactly 5 %.
- Lévy formula ≈ 2 × fixed-stop formula for small probabilities.
- Half-life formulas return exactly 50 %.
- A finely discretised driftless random walk (40 steps per day) and the daily GBM with Brownian-bridge ranges both reproduce the Lévy trailing-stop probability.
- Quantile-based stop and empirical trigger probability agree; Markov transition rows sum to 1; bootstrap without trend has zero mean.
- Max drawdown peak/trough detection, German CSV import, base statistics.

## 8. Why the results differ between methods

- **Fixed vs. trailing:** the reflection principle is for a stop that does not move. A trailing stop is hit more often, so it needs a wider distance ($z \approx 2.24$ instead of 1.96 at 95 %).
- **Volatility basis:** the formulas and the GBM use the volatility chosen under *Vola-Basis* (default: last year). The Markov chain and the bootstrap always use the whole model history (default: 3 years). If the last year was more volatile than the three-year average, the formulas give wider stops than these two simulations, and vice versa.
- **Fat tails and clustering:** the bootstrap and the Markov chain use real days, so occasional large moves and calm or volatile phases are included. The Markov chain also starts from today's state: after a large down day, a wider stop comes out than in a calm market.
