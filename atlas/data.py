"""Market data: daily OHLC history from Yahoo Finance's chart endpoint, cached in SQLite, validated.

Adjusted close (dividends/splits) is used for research (total return); raw close * scale is used
to value positions and size orders.
"""
from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import requests

from .config import UNIVERSE
from .db import DB

log = logging.getLogger("atlas.data")
UA = {"User-Agent": "Mozilla/5.0 (atlas-x research client)"}


class DataError(RuntimeError):
    pass


def fetch_yahoo(symbol: str, start: int = 946684800, retries: int = 3) -> pd.DataFrame:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={start}&period2={int(time.time())}&interval=1d&events=div,split")
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({
                "date": pd.to_datetime(res["timestamp"], unit="s").normalize(),
                "close": q["close"], "volume": q["volume"],
                "adjclose": res["indicators"]["adjclose"][0]["adjclose"],
            }).dropna(subset=["close", "adjclose"])
            return df.drop_duplicates("date", keep="last").set_index("date")
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise DataError(f"{symbol}: fetch failed after {retries} tries: {last}")


def validate(symbol: str, df: pd.DataFrame) -> list[str]:
    """Sanity checks; returns list of problems (empty = clean)."""
    issues = []
    if df.empty:
        return ["no data"]
    if (df["adjclose"] <= 0).any() or (df["close"] <= 0).any():
        issues.append("non-positive prices")
    r = df["adjclose"].pct_change().dropna()
    big = r[r.abs() > 0.25]
    if len(big):
        issues.append(f"{len(big)} daily moves >25% (first {big.index[0].date()})")
    return issues


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bad prints.

    1. Isolated: a >25% move fully reversed next day (classic Yahoo LSE glitch).
    2. Short runs: a price more than 25% away from its centred 21-day median. This catches
       multi-day unit glitches (e.g. SGLN.L in Apr-Jun 2011 quoted ~3,000 instead of ~1,800 for
       several days at a time) that rule 1 misses, while keeping genuine regime moves such as the
       Sep-2022 gilt crash, which persist and so move the median with them."""
    r = df["adjclose"].pct_change()
    nxt = r.shift(-1)
    bad = (r.abs() > 0.25) & (nxt.abs() > 0.2) & (np.sign(r) != np.sign(nxt))
    med = df["adjclose"].rolling(21, center=True, min_periods=5).median()
    bad |= (df["adjclose"] / med - 1).abs() > 0.25
    return df[~bad]


def refresh(db: DB, symbols: list[str] | None = None) -> dict:
    out = {}
    for s in symbols or list(UNIVERSE):
        raw = fetch_yahoo(s)
        df = clean(raw)
        if UNIVERSE.get(s, {}).get("history_from"):
            df = df[df.index >= pd.Timestamp(UNIVERSE[s]["history_from"])]
        dropped = raw.index.difference(df.index)
        if len(dropped):  # remove bad prints already stored by earlier refreshes
            db.many("DELETE FROM prices WHERE symbol=? AND date=?",
                    [(s, d.strftime("%Y-%m-%d")) for d in dropped])
        issues = validate(s, df)
        db.many("INSERT OR REPLACE INTO prices(symbol,date,close,adjclose,volume) VALUES(?,?,?,?,?)",
                [(s, d.strftime("%Y-%m-%d"), float(row.close), float(row.adjclose),
                  float(row.volume or 0)) for d, row in df.iterrows()])
        out[s] = {"rows": len(df), "last": df.index[-1].strftime("%Y-%m-%d"), "issues": issues}
        if issues:
            db.event("warn", "data_quality", f"{s}: {'; '.join(issues)}")
    db.set("data_refreshed_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return out


def load_panel(db: DB, field: str = "adjclose", symbols: list[str] | None = None) -> pd.DataFrame:
    syms = symbols or list(UNIVERSE)
    ph = ",".join("?" * len(syms))
    rows = db.q(f"SELECT symbol,date,{field} AS v FROM prices WHERE symbol IN ({ph})", tuple(syms))
    if not rows:
        raise DataError("no prices in database; run data refresh")
    df = pd.DataFrame(rows, columns=["symbol", "date", "v"])
    p = df.pivot(index="date", columns="symbol", values="v").sort_index()
    p.index = pd.to_datetime(p.index)
    return p[[s for s in syms if s in p.columns]]


def latest_prices_gbp(db: DB) -> tuple[dict[str, float], dict[str, str]]:
    """Latest raw close converted to GBP, plus the date of each price (for staleness checks)."""
    prices, dates = {}, {}
    for s, meta in UNIVERSE.items():
        r = db.q("SELECT date, close FROM prices WHERE symbol=? ORDER BY date DESC LIMIT 1", (s,))
        if r:
            prices[s] = r[0]["close"] * meta["scale"]
            dates[s] = r[0]["date"]
    return prices, dates


def staleness_days(d: str) -> int:
    """Business days between the price date and today."""
    return int(np.busday_count(date.fromisoformat(d), date.today()))
