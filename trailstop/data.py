"""ISIN-Auflösung und Kursdaten (yfinance, CSV-Fallback)."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

import pandas as pd
import requests

ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")

# Bloomberg-Börsencode (OpenFIGI) -> Yahoo-Suffix
FIGI_EXCH_TO_YAHOO = {
    "GY": ".DE", "GR": ".F", "GF": ".F", "GS": ".SG", "GM": ".MU", "GD": ".DU",
    "GB": ".BE", "GH": ".HM", "GI": ".HM", "NA": ".AS", "FP": ".PA", "IM": ".MI",
    "LN": ".L", "SW": ".SW", "SE": ".SW", "BB": ".BR", "ID": ".IR", "AV": ".VI",
    "US": "", "UN": "", "UW": "", "UQ": "", "UP": "", "UA": "",
}

# Präferenz: Xetra, deutsche Plätze, Euro-Börsen, Rest
SUFFIX_RANK = [".DE", ".F", ".SG", ".MU", ".DU", ".BE", ".HM", ".AS", ".PA", ".MI",
               ".BR", ".IR", ".VI", ".SW", ".L", ""]


@dataclass
class TickerCandidate:
    symbol: str
    name: str = ""
    exchange: str = ""
    source: str = ""

    @property
    def suffix(self) -> str:
        return "." + self.symbol.rsplit(".", 1)[1] if "." in self.symbol else ""

    @property
    def rank(self) -> int:
        try:
            return SUFFIX_RANK.index(self.suffix)
        except ValueError:
            return len(SUFFIX_RANK)

    @property
    def label(self) -> str:
        parts = [self.symbol]
        if self.exchange:
            parts.append(self.exchange)
        if self.name:
            parts.append(self.name)
        return " | ".join(parts)


def is_valid_isin(isin: str) -> bool:
    """Formatprüfung + Luhn-Prüfziffer."""
    isin = isin.strip().upper()
    if not ISIN_RE.match(isin):
        return False
    digits = "".join(str(int(c, 36)) for c in isin)
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            d = d - 9 if d > 9 else d
        total += d
    return total % 10 == 0


def _yahoo_search(isin: str) -> list[TickerCandidate]:
    try:
        import yfinance as yf

        quotes = yf.Search(isin, max_results=15, news_count=0).quotes
    except Exception:
        return []
    out = []
    for q in quotes or []:
        sym = q.get("symbol")
        if sym:
            out.append(TickerCandidate(sym, q.get("longname") or q.get("shortname") or "",
                                       q.get("exchDisp") or q.get("exchange") or "", "Yahoo"))
    return out


def _openfigi(isin: str) -> list[TickerCandidate]:
    try:
        resp = requests.post(
            "https://api.openfigi.com/v3/mapping",
            json=[{"idType": "ID_ISIN", "idValue": isin}],
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()[0].get("data", [])
    except Exception:
        return []
    out = []
    for row in data:
        suffix = FIGI_EXCH_TO_YAHOO.get(row.get("exchCode", ""))
        ticker = (row.get("ticker") or "").replace("/", "-")
        if suffix is None or not ticker or " " in ticker:
            continue
        out.append(TickerCandidate(ticker + suffix, row.get("name", ""), row.get("exchCode", ""), "OpenFIGI"))
    return out


def resolve_isin(isin: str) -> list[TickerCandidate]:
    """Alle gefundenen Yahoo-Ticker zur ISIN, nach Börsenpräferenz sortiert."""
    isin = isin.strip().upper()
    seen: dict[str, TickerCandidate] = {}
    found = _yahoo_search(isin) + _openfigi(isin)
    german = (".F", ".SG", ".MU", ".DU", ".BE", ".HM")
    derived = [TickerCandidate(c.symbol.rsplit(".", 1)[0] + ".DE", c.name, "Xetra (abgeleitet)", "abgeleitet")
               for c in found if c.suffix in german]
    for cand in found + derived:
        seen.setdefault(cand.symbol, cand)
    return sorted(seen.values(), key=lambda c: (c.rank, c.symbol))


def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["Open", "High", "Low", "Close"]].astype(float).copy()
    df = df[df["Close"] > 0].dropna(subset=["Close"])
    for col in ("Open", "High", "Low"):
        df[col] = df[col].where(df[col] > 0)
    df["Open"] = df["Open"].fillna(df["Close"].shift(1)).fillna(df["Close"])
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index.name = "Datum"
    return df


def load_yahoo(symbol: str, years: int) -> tuple[pd.DataFrame, str, str]:
    """OHLC (dividendenbereinigt) für `years` Jahre + 1 Jahr Puffer. Rückgabe: (df, Währung, Name)."""
    import yfinance as yf

    t = yf.Ticker(symbol)
    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years + 1)
    hist = t.history(start=start.strftime("%Y-%m-%d"), auto_adjust=True, actions=False)
    if hist is None or hist.empty:
        raise ValueError(f"Keine Kursdaten für {symbol}")
    currency, name = "", symbol
    try:
        currency = t.fast_info.get("currency") or ""
    except Exception:
        pass
    try:
        info = t.get_info()
        name = info.get("longName") or info.get("shortName") or symbol
        currency = currency or info.get("currency", "")
    except Exception:
        pass
    return _clean_ohlc(hist), currency.upper(), name


_COL_ALIASES = {
    "Date": ["date", "datum", "zeit", "time"],
    "Open": ["open", "eröffnung", "eroeffnung", "erster", "start"],
    "High": ["high", "hoch", "max"],
    "Low": ["low", "tief", "min"],
    "Close": ["adj close", "close", "schluss", "schlusskurs", "letzter", "kurs", "price"],
}


def load_csv(content: bytes) -> pd.DataFrame:
    """CSV mit Datum + Close (Open/High/Low optional); Komma/Semikolon, Dezimalkomma werden erkannt."""
    text = content.decode("utf-8-sig", errors="replace")
    sep = ";" if text.count(";") > text.count(",") else ","
    raw = pd.read_csv(io.StringIO(text), sep=sep)
    cols = {c: c.strip().lower() for c in raw.columns}
    mapping = {}
    for target, aliases in _COL_ALIASES.items():
        for alias in aliases:  # Reihenfolge = Priorität
            match = next((c for c, lc in cols.items() if lc == alias and c not in mapping), None)
            if match is None:
                match = next((c for c, lc in cols.items() if alias in lc and c not in mapping), None)
            if match is not None:
                mapping[match] = target
                break
    df = raw.rename(columns=mapping)
    if "Date" not in df or "Close" not in df:
        raise ValueError("CSV braucht mindestens die Spalten Datum und Schlusskurs (Close).")
    for col in ("Open", "High", "Low", "Close"):
        if col not in df:
            df[col] = float("nan")
        elif not pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].astype(str).str.strip()
            if sep == ";" or s.str.contains(r"^\d{1,3}(\.\d{3})*,\d+$|^\d+,\d+$").any():
                s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(s, errors="coerce")
    dayfirst = df["Date"].astype(str).str.match(r"^\d{1,2}\.\d{1,2}\.\d{2,4}").any()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=dayfirst, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    return _clean_ohlc(df)
