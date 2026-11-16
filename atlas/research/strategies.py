"""Candidate strategy families. Each returns TARGET weights decided at the close of day t, using only
information available up to and including t. The backtester applies them from t+1 (no look-ahead).

Weights are long-only, sum <= 1; any residual is held in the cash-proxy ETF.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from ..config import BENCHMARK, CASH_PROXY, UNIVERSE

RISKY = [s for s in UNIVERSE if s != CASH_PROXY]


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    return pd.DatetimeIndex(s.groupby([idx.year, idx.month]).max().values)


def _week_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx, index=idx)
    iso = idx.isocalendar()
    return pd.DatetimeIndex(s.groupby([iso.year.values, iso.week.values]).max().values)


def _hold(rebal: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill weights between rebalance dates and park the residual in the cash proxy."""
    w = rebal.reindex(idx).ffill().fillna(0.0)
    for c in UNIVERSE:
        if c not in w:
            w[c] = 0.0
    w[CASH_PROXY] = w[CASH_PROXY] + (1 - w.sum(axis=1)).clip(lower=0)
    return w[list(UNIVERSE)]


def _cap(w: pd.Series, cap: float) -> pd.Series:
    """Iteratively cap weights, redistributing the excess pro-rata; leftover goes to cash."""
    w = w.copy()
    caps = pd.Series({s: min(cap, UNIVERSE[s].get("max_w", cap)) for s in w.index})
    for _ in range(20):
        over = w > caps + 1e-12
        if not over.any():
            break
        excess = (w[over] - caps[over]).sum()
        w[over] = caps[over]
        under = (~over) & (w > 0) & (w < caps - 1e-12)
        if under.any():
            w[under] += excess * w[under] / w[under].sum()
    return w.clip(upper=caps)


def buy_hold(px: pd.DataFrame, symbol: str = BENCHMARK) -> pd.DataFrame:
    rb = pd.DataFrame(0.0, index=px.index[:1], columns=list(UNIVERSE))
    rb[symbol] = 1.0
    return _hold(rb, px.index)


def static_mix(px: pd.DataFrame, mix: tuple = (("VWRL.L", 0.6), ("IGLT.L", 0.4))) -> pd.DataFrame:
    me = _month_ends(px.index)
    rb = pd.DataFrame(0.0, index=me, columns=list(UNIVERSE))
    for s, wt in mix:
        rb[s] = wt
    return _hold(rb, px.index)


def inverse_vol(px: pd.DataFrame, lookback: int = 120, cap: float = 0.35) -> pd.DataFrame:
    r = px[RISKY].pct_change()
    vol = r.rolling(lookback).std()
    me = _month_ends(px.index)
    rows = {}
    for d in me:
        v = vol.loc[d].dropna()
        if len(v) < 3:
            continue
        w = (1 / v) / (1 / v).sum()
        rows[d] = _cap(w, cap)
    return _hold(pd.DataFrame(rows).T, px.index)


def trend_vol_target(px: pd.DataFrame, lookback: int = 252, target_vol: float = 0.10,
                     vol_window: int = 60, cap: float = 0.35) -> pd.DataFrame:
    """Time-series momentum (Moskowitz/Ooi/Pedersen 2012; Hurst et al. 'A Century of Evidence on
    Trend-Following'): hold an asset only if its trailing excess return over the cash proxy is
    positive; size inverse-vol; scale the book to a portfolio volatility target (no leverage)."""
    r = px.pct_change()
    mom = px.pct_change(lookback)
    excess = mom[RISKY].sub(mom[CASH_PROXY], axis=0)
    vol = r[RISKY].rolling(vol_window).std() * np.sqrt(252)
    cov_r = r[RISKY]
    me = _month_ends(px.index)
    rows = {}
    for d in me:
        e = excess.loc[d].dropna()
        v = vol.loc[d].reindex(e.index)
        on = e[(e > 0) & v.notna() & (v > 0)].index
        if len(on) == 0:
            rows[d] = pd.Series(0.0, index=RISKY)
            continue
        w = (1 / v[on]) / (1 / v[on]).sum()
        win = cov_r.loc[:d, on].tail(vol_window).dropna()
        if len(win) >= 20:
            pvol = float(np.sqrt(w.values @ (win.cov().values * 252) @ w.values))
            scale = min(1.0, target_vol / pvol) if pvol > 0 else 0.0
        else:
            scale = 0.5
        rows[d] = _cap(w * scale, cap).reindex(RISKY).fillna(0.0)
    return _hold(pd.DataFrame(rows).T, px.index)


def dual_momentum(px: pd.DataFrame, lookback: int = 252, top_n: int = 3, cap: float = 0.35) -> pd.DataFrame:
    """Cross-sectional + absolute momentum (Antonacci): top-N by trailing return, only if > cash."""
    mom = px.pct_change(lookback)
    me = _month_ends(px.index)
    rows = {}
    for d in me:
        m = mom.loc[d].dropna()
        if CASH_PROXY not in m or len(m) < 4:
            continue
        hurdle = m[CASH_PROXY]
        cand = m[RISKY].dropna().sort_values(ascending=False).head(top_n)
        picks = cand[cand > hurdle].index
        w = pd.Series(0.0, index=RISKY)
        if len(picks):
            w[picks] = min(cap, 1.0 / top_n)
        rows[d] = _cap(w, cap)
    return _hold(pd.DataFrame(rows).T, px.index)


def short_term_reversal(px: pd.DataFrame, lookback: int = 5, top_n: int = 2) -> pd.DataFrame:
    """Weekly short-term reversal among risky ETFs: buy the worst recent performers. Included as a
    deliberately cost-sensitive hypothesis; expected to fail after costs."""
    ret = px[RISKY].pct_change(lookback)
    we = _week_ends(px.index)
    rows = {}
    for d in we:
        m = ret.loc[d].dropna()
        if len(m) < 4:
            continue
        w = pd.Series(0.0, index=RISKY)
        w[m.sort_values().head(top_n).index] = 1.0 / top_n
        rows[d] = _cap(w, 0.5)
    return _hold(pd.DataFrame(rows).T, px.index)


def _grid(**kw):
    keys = list(kw)
    for vals in itertools.product(*kw.values()):
        yield dict(zip(keys, vals))


# family -> (function, parameter grid, is_benchmark)
FAMILIES = {
    "buy_hold_global_equity": (buy_hold, [{}], True),
    "static_60_40": (static_mix, [{}], True),
    "inverse_vol": (inverse_vol, list(_grid(lookback=[60, 120, 250])), False),
    "trend_vol_target": (trend_vol_target, list(_grid(lookback=[126, 189, 252], target_vol=[0.08, 0.10, 0.12])), False),
    "dual_momentum": (dual_momentum, list(_grid(lookback=[126, 189, 252], top_n=[2, 3, 4])), False),
    "short_term_reversal": (short_term_reversal, list(_grid(lookback=[5, 10, 20], top_n=[2, 3])), False),
}


def target_weights(family: str, params: dict, px: pd.DataFrame) -> pd.Series:
    """Latest target weights for live use."""
    fn = FAMILIES[family][0]
    return fn(px, **params).iloc[-1]
