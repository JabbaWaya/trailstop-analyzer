# Methodik

Was die App Schritt für Schritt berechnet und wo im Code das passiert. Notation: $S_t$ Kurs, $\ln$ natürlicher Logarithmus, $\Phi$ Verteilungsfunktion der Standardnormalverteilung, $\alpha = 1 - \text{Sicherheitsniveau}$ (z. B. 0,05), $T$ = Überwachungszeitraum in Börsentagen (z. B. 10).

Abstände werden intern als **Log-Abstände** $b$ geführt und erst am Ende in Prozent umgerechnet:

$$d = 1 - e^{-b}, \qquad b = -\ln(1-d)$$

Ein Stop mit Abstand $d$ unter einem Hoch $P$ liegt bei $P\,(1-d)$.

---

## 1. Daten (`trailstop/data.py`)

1. **ISIN-Prüfung:** Format (2 Buchstaben + 9 alphanumerische Zeichen + Prüfziffer) und Luhn-Prüfziffer.
2. **ISIN → Ticker:** Yahoo-Finance-Suche und OpenFIGI-Mapping-API. OpenFIGI-Börsencodes werden in Yahoo-Suffixe übersetzt (z. B. `GY → .DE`, `GR → .F`, `NA → .AS`). Da Yahoo das Xetra-Listing oft nicht liefert, wird aus jedem deutschen Regionalbörsen-Listing (`.F`, `.SG`, `.MU`, …) ein `.DE`-Kandidat abgeleitet.
3. **Reihenfolge:** Xetra (`.DE`) → andere deutsche Börsen → Euro-Börsen (`.AS`, `.PA`, `.MI`, …) → `.SW`, `.L` → USA → Rest. Scheitert der gewählte Ticker oder hat er weniger als 120 Tage, wird der nächste versucht.
4. **Kurse:** tägliche OHLC-Daten von Yahoo mit `auto_adjust=True` (um Dividenden/Splits bereinigt), Historie = Modelljahre + 1 Jahr.
5. **Bereinigung:** Zeilen ohne positiven Schlusskurs entfallen; fehlende Eröffnung = Vortagesschluss; `Hoch ≥ max(Eröffnung, Schluss)` und `Tief ≤ min(Eröffnung, Schluss)` werden erzwungen; eine Zeile pro Kalendertag.

## 2. Ausgangsdaten – Abschnitt I (`trailstop/stats.py`)

| Kennzahl | Definition |
|---|---|
| Aktueller Kurs $S_0$ | Letzter Schlusskurs |
| Tägliche Log-Rendite | $r_t = \ln(C_t / C_{t-1})$ |
| Jahresvolatilität | $\sigma_{Jahr} = \text{Std}(r_t)\cdot\sqrt{252}$ (Stichproben-Standardabweichung), einmal für das **letzte Jahr** und einmal für die **Modellhistorie** |
| 52-Wochen-Hoch / -Tief | Höchstes Tages**hoch** / tiefstes Tages**tief** der letzten 365 Kalendertage |
| Max-Drawdown (Schlusskurse) | $\max_t \big(1 - C_t / \max_{s\le t} C_s\big)$ über das letzte Jahr, mit Datum von Hoch und Tief |
| Max-Drawdown (Hoch → Tief) | $\max_t \big(1 - L_t / \max_{s\le t} H_s\big)$ – maßgeblich für Stops, denn ein Stop folgt den Hochs und wird von Tiefs ausgelöst |

Für die Simulationen wird jeder historische Tag als **Tages-Tupel** relativ zum Vortagesschluss gespeichert:

$$\big(r_t,\ h_t,\ l_t\big) = \Big(\ln\tfrac{C_t}{C_{t-1}},\ \ln\tfrac{H_t}{C_{t-1}},\ \ln\tfrac{L_t}{C_{t-1}}\Big), \quad h_t \ge r_t \ge l_t$$

So bleiben echte Tagesspannen, Kurslücken und Fat Tails mit der Rendite des jeweiligen Tages verbunden.

## 3. Formeln – Abschnitt II (`trailstop/analytic.py`)

Alle Formeln setzen eine driftlose Brownsche Bewegung des Log-Kurses mit Tagesvolatilität $\sigma_{Tag}$ voraus (aus der gewählten *Vola-Basis*).

### 3.1 Wurzel-T-Gesetz

$$\sigma_{Tag} = \frac{\sigma_{Jahr}}{\sqrt{252}}, \qquad \sigma_T = \sigma_{Tag}\sqrt{T}$$

### 3.2 Naive Endpunkt-Betrachtung

Prüft nur, ob der Kurs **am Ende** des Zeitraums unter dem Stop liegt: $P(\ln S_T/S_0 \le -b) = \Phi(-b/\sigma_T)$, also $z = \Phi^{-1}(1-\alpha)$ (1,645 bei 95 %). Das unterschätzt das Risiko, weil ein Stop auslöst, sobald er **irgendwann** berührt wird.

### 3.3 Fester Stop: Reflexionsprinzip (Pfad-Faktor 2)

Jeder Pfad, der die Schranke $-b$ berührt, endet ab diesem Moment mit gleicher Wahrscheinlichkeit darüber oder darunter. Die Berührungswahrscheinlichkeit ist daher doppelt so groß wie die Wahrscheinlichkeit, darunter zu enden:

$$P\Big(\min_{t\le T}\ln\tfrac{S_t}{S_0} \le -b\Big) = 2\,\Phi\Big(-\frac{b}{\sigma_T}\Big) \stackrel{!}{=} \alpha
\;\Rightarrow\; z = \Phi^{-1}\big(1-\tfrac{\alpha}{2}\big),\quad b = z\,\sigma_T$$

$z = 1{,}960$ bei 95 %, $2{,}576$ bei 99 %. Das ist die in der ursprünglichen Aufgabe beschriebene Methode. Sie ist exakt für einen **festen** Stop unter dem heutigen Kurs.

### 3.4 Nachlaufender Stop: Satz von Lévy

Ein Trailing Stop wird mit jedem neuen Hoch nachgezogen. Er wird also vom **Drawdown vom laufenden Maximum** $M_t - W_t$ ausgelöst, nicht vom Abstand zum Startkurs. Nach dem Satz von Lévy hat $M_t - W_t$ dieselbe Verteilung wie $|W_t|$ (reflektierte Brownsche Bewegung). Der maximale Drawdown bis $T$ ist daher verteilt wie $\sup_{s\le T}|W_s|$ – ein Austrittsproblem mit zwei Schranken. Mit $x = b/\sigma_T$ und der Spiegelungsmethode:

$$P(\text{Auslösung bis } T) = 1 - \sum_{k=-\infty}^{\infty} (-1)^k\Big[\Phi\big((1-2k)\,x\big) - \Phi\big((-1-2k)\,x\big)\Big]$$

Für kleine Wahrscheinlichkeiten ist das ≈ $4\,\Phi(-x)$, also **doppelt so viel wie beim festen Stop**. Nach $\alpha$ aufgelöst (Nullstellensuche, `b_trailing`) ergibt sich $z \approx 2{,}241$ bei 95 % und $2{,}807$ bei 99 %. Die App zeigt dies als *„Formel: nachlaufend (Lévy)“*.

### 3.5 Halbwertszeit

Anzahl Börsentage $n$, nach der der Stop mit genau 50 % Wahrscheinlichkeit erreicht ist (ohne Trend):

- Fester Stop: $2\Phi\big(-b/(\sigma_{Tag}\sqrt n)\big) = 0{,}5 \Rightarrow n = \big(b / (0{,}6745\,\sigma_{Tag})\big)^2$
- Nachlaufender Stop (Lévy): $P$ hängt nur von $x = b/(\sigma_{Tag}\sqrt n)$ ab und beträgt 50 % bei $x_{50} \approx 1{,}149$, also $n = \big(b/(1{,}149\,\sigma_{Tag})\big)^2$

## 4. Simulationen (`trailstop/models.py`)

Jedes Modell erzeugt $N$ Pfade (Standard 10 000) mit $D$ Tagen (Standard 504) aus Tages-Tupeln $(r, h, l)$. Jedes Modell erhält einen eigenen, aus dem Seed abgeleiteten Zufallsgenerator.

**Trend:** Bei *„Ohne Trend“* wird die mittlere Tagesrendite der Historie von $r$, $h$ und $l$ gleichermaßen abgezogen (die Form jedes Tages bleibt erhalten, die Drift wird null). Bei *„Historischer Trend“* bleiben die Daten unverändert, und die GBM erhält die historische mittlere Log-Rendite als Drift.

### 4.1 Markov-Kette (Standard-Leitmodell)

1. Die historischen Tagesrenditen werden in $k$ Quantilklassen (Standard 5) eingeteilt = Zustände, z. B. *stark fallend, fallend, seitwärts, steigend, stark steigend*.
2. Die Übergangsmatrix $P_{ij}$ = P(Zustand $j$ morgen | Zustand $i$ heute) wird aus aufeinanderfolgenden Tagen geschätzt, mit Laplace-Glättung (+0,5 je Zelle).
3. Simulation: Start im **heutigen** Zustand; der nächste Zustand wird aus der Zeile des aktuellen Zustands gezogen, dann ein **echter historischer Tag** (mit Hoch und Tief) gleichverteilt aus der Klasse dieses Zustands.

So werden Volatilitäts-Cluster erfasst (große Bewegungen folgen eher auf große Bewegungen), die Simulation startet in der aktuellen Marktphase, und jeder simulierte Tag ist ein echter Tag. Die Matrix erscheint in der App als Heatmap.

### 4.2 Stationärer Block-Bootstrap

Stationärer Bootstrap nach Politis und Romano: Pfade werden aus **Blöcken aufeinanderfolgender echter Tage** zusammengesetzt. In jedem Schritt geht der Pfad mit dem nächsten historischen Tag weiter oder springt mit Wahrscheinlichkeit $1/L$ (Standard $L = 5$) zu einem zufälligen Tag. Die Blocklängen sind damit geometrisch verteilt mit Mittelwert $L$. Modellfrei; erhält Fat Tails und kurzfristige Autokorrelation.

### 4.3 GBM-Monte-Carlo

$r_t = \mu + \sigma_{Tag}\,Z_t$ mit $Z_t \sim N(0,1)$ ($\mu = 0$ ohne Trend). Tageshoch und -tief stammen aus der exakten Verteilung von Maximum und Minimum einer **Brownschen Brücke** von $0$ nach $r_t$ über einen Tag:

$$h_t = \tfrac12\Big(r_t + \sqrt{r_t^2 - 2\sigma_{Tag}^2\ln U_1}\Big),\qquad l_t = \tfrac12\Big(r_t - \sqrt{r_t^2 - 2\sigma_{Tag}^2\ln U_2}\Big)$$

($U_1, U_2$ gleichverteilt; unabhängig gezogen, das gemeinsame Verhalten ist daher genähert.) Das ist die Referenz: Sie sollte die Lévy-Formel reproduzieren – und tut es.

## 5. Trailing-Stop-Auswertung (`trailstop/engine.py`)

Für jeden Pfad:

1. Log-Schlusskurs $c_t = \sum_{s\le t} r_s$ ($c_0 = 0$ ist der heutige Kurs), Tageshoch $c_{t-1} + h_t$, Tagestief $c_{t-1} + l_t$.
2. **Hoch** = höchstes bisheriges Tageshoch, Start beim heutigen Kurs. Ob innerhalb eines Tages zuerst das Hoch oder das Tief kam, ist unbekannt; die Reihenfolge wird je Tag 50/50 gezogen. Kam das Hoch zuerst, zieht es den Stop bereits nach, bevor das Tief geprüft wird.
3. **Drawdown** an Tag $t$: Hoch − Tagestief (log). **Laufender Max-Drawdown** $DD_t$ = größter Drawdown bis Tag $t$; er fällt nie.

Ein Stop mit Log-Abstand $b$ ist bis Tag $t$ genau dann ausgelöst, wenn $DD_t \ge b$. Dadurch lässt sich jeder Abstand auf **denselben** Pfaden auswerten (gemeinsame Zufallszahlen), ohne neu zu simulieren:

| Ergebnis | Berechnung |
|---|---|
| Optimaler Abstand | $b^* = $ Quantil von $DD_T$ über alle Pfade zum Sicherheitsniveau (z. B. 95. Perzentil des 10-Tage-Max-Drawdowns) |
| Auslösewahrscheinlichkeit für beliebiges $b$ | Anteil der Pfade mit $DD_T \ge b$ |
| Auslösewahrscheinlichkeit über die Zeit | Anteil der Pfade mit $DD_t \ge b$, für $t = 1..D$ |
| Halbwertszeit | Erster Tag $t$, an dem der **Median** von $DD_t$ über alle Pfade $\ge b$ ist; „> D“, falls nie innerhalb der Simulationslänge |
| P(1-Jahres-MDD ≥ realer MDD) | Anteil der Pfade mit $DD_{252} \ge \ln\big(1/(1-\text{MDD}_{real})\big)$, mit dem Hoch→Tief-Drawdown |

## 6. Abschnitte III und IV (`trailstop/report.py`, `app.py`)

- **Leitmodell:** Jede der drei Simulationen und beide Formeln sind wählbar. Alle Ergebnisse laufen über eine gemeinsame Schnittstelle (`ModelView`: Stop, Wahrscheinlichkeit, Kurve, Halbwertszeit), sodass Tabellen und Diagramme Simulationen und Formeln gleich behandeln.
- **Max-Drawdown-Abgleich:** Verhältnis Stop-Abstand / realer 1-Jahres-Max-Drawdown; ob der Stop im letzten Jahr ausgelöst hätte; Wahrscheinlichkeit je Modell, dass ein 1-Jahres-Pfad den realen Max-Drawdown erreicht (bei der Formel für den festen Stop ist das die Wahrscheinlichkeit, so weit unter den heutigen Kurs zu fallen, nicht unter ein laufendes Hoch).
- **Fazit-Tabelle:** der berechnete Abstand $d^*$ sowie $\text{round}(d^*)$ und $\pm 1$, $\pm 2$ Prozentpunkte (Werte ≤ 0 entfallen). Je Zeile: Stop-Kurs $S_0(1-d)$, Auslösewahrscheinlichkeit innerhalb von $T$ Tagen und Halbwertszeit des Leitmodells; der aufklappbare Vergleich ergänzt dieselben Werte für alle übrigen Modelle.

## 7. Validierung (`tests/test_core.py`)

- $z$ für das Reflexionsprinzip ist 1,96 und ergibt genau 5 %.
- Lévy-Formel ≈ 2 × Formel für den festen Stop bei kleinen Wahrscheinlichkeiten.
- Die Halbwertszeit-Formeln liefern genau 50 %.
- Ein fein diskretisierter driftloser Zufallspfad (40 Schritte je Tag) und die tägliche GBM mit Brownscher-Brücke-Spannen reproduzieren beide die Lévy-Wahrscheinlichkeit für den nachlaufenden Stop.
- Quantil-basierter Stop und empirische Auslösewahrscheinlichkeit stimmen überein; Zeilen der Markov-Matrix summieren auf 1; Bootstrap ohne Trend hat Mittelwert null.
- Erkennung von Hoch und Tief beim Max-Drawdown, deutscher CSV-Import, Basiskennzahlen.

## 8. Warum die Methoden unterschiedliche Ergebnisse liefern

- **Fest vs. nachlaufend:** Das Reflexionsprinzip gilt für einen Stop, der sich nicht bewegt. Ein Trailing Stop wird häufiger ausgelöst und braucht daher einen größeren Abstand ($z \approx 2{,}24$ statt 1,96 bei 95 %).
- **Vola-Basis:** Formeln und GBM verwenden die unter *Vola-Basis* gewählte Volatilität (Standard: letztes Jahr). Markov-Kette und Bootstrap nutzen immer die ganze Modellhistorie (Standard: 3 Jahre). War das letzte Jahr volatiler als der 3-Jahres-Schnitt, ergeben die Formeln größere Abstände als diese beiden Simulationen, und umgekehrt.
- **Fat Tails und Cluster:** Bootstrap und Markov-Kette verwenden echte Tage, enthalten also gelegentliche große Bewegungen sowie ruhige und volatile Phasen. Die Markov-Kette startet zudem im heutigen Zustand: Nach einem starken Verlusttag ergibt sich ein größerer Abstand als in einem ruhigen Markt.
