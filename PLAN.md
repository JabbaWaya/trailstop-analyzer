# Plan: Trailing-Stop-Risikoanalyse per ISIN (Streamlit)

## Entscheidungen (aus Rückfragen)
| Thema | Wahl |
|---|---|
| Oberfläche | Streamlit-App, Plotly-Charts, deutsche Beschriftung |
| Daten | ISIN → Ticker (OpenFIGI / Yahoo-Suche, EUR-Listing bevorzugt, z. B. `IS3N.DE`), Kurse via yfinance; CSV-Upload (OHLC) als Fallback; Ticker im UI überschreibbar |
| Modelle | Markov-Kette (Regime), historischer Block-Bootstrap, GBM-Monte-Carlo + analytische Formel als Referenz |
| Stop-Regel | Stop folgt dem laufenden Hoch (Tageshoch), löst aus, wenn ein Tagestief den Stop berührt |
| „Optimaler“ Stop | Leitmodell per Dropdown wählbar (Default Markov); Abstand so, dass simulierte P(Auslösung in 10 Tagen) = 5 % |
| Historie | Slider 1–10 Jahre, Default 3 Jahre (52W-Hoch/Tief und 1J-Max-Drawdown immer letztes Jahr) |
| Umgebung | Lokale Conda-Umgebung im Projektordner: `./.conda` (Miniconda vorhanden) |

## Projektstruktur
```
finance_calculator/
  environment.yml          # python=3.12, numpy, pandas, scipy, plotly, streamlit, yfinance, requests, pytest
  app.py                   # Streamlit-UI
  trailstop/
    data.py                # ISIN→Ticker, Download (auto_adjust OHLC), CSV-Import, Caching
    stats.py               # Vola, 52W-Hoch/Tief, Max-Drawdown (Peak→Tief), Tages-Tupel
    analytic.py            # Wurzel-T, Reflexionsprinzip, Halbwertszeit, Drawdown-Wahrscheinlichkeit
    models/
      gbm.py               # GBM + Brownsche-Brücke für Intraday-Hoch/Tief
      bootstrap.py         # Stationärer Block-Bootstrap echter Handelstage
      markov.py            # Zustände + Übergangsmatrix, Ziehen echter Tage je Zustand
    engine.py              # Vektorisierte Trailing-Stop-Auswertung auf simulierten OHLC-Pfaden
    report.py              # Abschnitte I–IV als Tabellen/Text
  tests/                   # u. a. GBM-Simulation ≈ analytische Formel
```

## Kernidee der Simulation
Jeder simulierte Handelstag ist ein Tupel *(Close-Rendite, Hoch/VorClose, Tief/VorClose)*, damit Intraday-Spannen realistisch sind.

- **Markov-Kette:** Tagesrenditen in k Zustände (Default 5, Quantil-Buckets) einteilen, Übergangsmatrix P(i→j) aus der Historie schätzen. Simulation: Zustandsfolge ziehen, pro Zustand einen echten historischen Tag (inkl. Hoch/Tief) aus diesem Bucket ziehen. Startzustand = Zustand von heute → bildet Volatilitäts-Cluster ab. Übergangsmatrix wird als Heatmap gezeigt.
- **Block-Bootstrap:** Blöcke echter Tage (mittlere Blocklänge ~5) neu zusammensetzen → Fat Tails + kurzfristige Autokorrelation, modellfrei.
- **GBM:** Normalverteilte Log-Renditen mit historischer Vola, Drift wählbar (0 oder historisch); Intraday-Tief/-Hoch per exakter Verteilung des Minimums/Maximums einer Brownschen Brücke.

**Trailing-Stop-Auswertung (effizient):** Pro Pfad wird der laufende Drawdown *Tief_t / max(Hoch_≤t) − 1* berechnet. Ein Stop mit Abstand d löst aus, sobald dieser Drawdown ≤ −d ist. Daraus folgt direkt:
- optimaler Abstand = 95 %-Quantil des maximalen 10-Tage-Drawdowns über alle Pfade (keine Bisektion nötig, gleiche Zufallszahlen für alle Abstände),
- Auslösewahrscheinlichkeit für beliebige d = Anteil der Pfade mit Max-DD ≥ d,
- Halbwertszeit = Median der ersten Auslösezeit auf langen Pfaden (bis 750 Tage).

Default 20 000 Pfade, fester Seed (reproduzierbar), Ergebnisse mit `st.cache_data` gecacht.

## Ausgabe = vier Abschnitte (wie in der Aufgabe)
**I. Ausgangsdaten:** Jahresvola (σ der Log-Renditen · √252), aktueller Kurs, reales 52W-Hoch/-Tief, realer 1J-Max-Drawdown (mit Peak- und Tiefdatum) + Kurschart mit markiertem Drawdown.

**II. Herleitung (transparent, mit Formeln via `st.latex`):**
1. σ_Tag = σ_Jahr/√252, σ_10T = σ_Tag·√10
2. Reflexionsprinzip: P(Min ≤ −b in T) = 2·Φ(−b/σ_T) → Pfadfaktor 2 → z = Φ⁻¹(1 − 0,05/2) = 1,96 statt 1,645
3. Abstand b = 1,96·σ_10T (log) → Prozent und erster Stop-Kurs in €
4. Hinweis + Zahl: Das Reflexionsprinzip gilt für einen *festen* Stop; ein *nachlaufender* Stop löst häufiger aus. Daher daneben der simulierte Wert jedes Modells — Leitmodell bestimmt den empfohlenen Abstand.

**III. Gegenrechnungen:**
- Halbwertszeit analytisch n = (b / (0,6745·σ_Tag))² und simuliert (Median-Auslösezeit), plus Kurve „P(Auslösung) über Handelstage“.
- Max-Drawdown-Abgleich: Verhältnis Stop-Abstand / realer 1J-MDD; P(MDD in 1 Jahr ≥ realer MDD) aus jedem Modell + Verteilung des simulierten 1J-MDD als Histogramm.

**IV. Fazit-Tabelle:** Zeilen = berechneter Abstand, gerundet ±0, ±1, ±2 %-Punkte; Spalten = Abstand, Stop-Kurs €, P(Auslösung in 10 Tagen), Halbwertszeit — je für Leitmodell, weitere Modelle als zusätzliche Spalten/Tab. CSV-Download.

## UI (Sidebar)
ISIN (Default `IE00B4L5YC18`), Ticker-Auswahl, CSV-Upload, Historie (Jahre), Sicherheitsniveau (Default 95 %), Zeitraum (Default 10 Tage), Leitmodell, Pfadanzahl, Markov-Zustände, Bootstrap-Blocklänge, GBM-Drift an/aus.

## Umsetzungsschritte
1. `environment.yml` + `conda env create -p ./.conda -f environment.yml`
2. `data.py` + `stats.py`, prüfen mit IE00B4L5YC18
3. `analytic.py` + `engine.py` + drei Modelle
4. Tests: GBM ohne Drift, Close-only, fester Stop ≈ 2Φ(−b/σ√T) (inkl. bekannter Abweichung durch diskrete Beobachtung); Quantil/Wahrscheinlichkeit konsistent
5. `app.py` mit Abschnitten I–IV, Charts
6. Lauf der App im Browser-Pane, Ergebnisse für IE00B4L5YC18 prüfen

## Bekannte Grenzen
- Kostenlose Datenquellen: ISIN-Auflösung kann scheitern → CSV/Ticker-Override.
- Keine Gaps über Wochenenden/News gesondert modelliert (Bootstrap/Markov enthalten sie implizit über echte Tage).
- Keine Anlageberatung; reine statistische Modellrechnung.

Start der App später:
```
.\.conda\python.exe -m streamlit run app.py
```
