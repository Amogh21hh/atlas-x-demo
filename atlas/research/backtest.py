"""Vectorised daily backtester with next-day execution and transaction costs."""
from __future__ import annotations

import pandas as pd

from ..config import settings


def run(px: pd.DataFrame, weights: pd.DataFrame, cost_bps: float | None = None) -> tuple[pd.Series, pd.Series]:
    """Weights decided at close t earn returns from t to t+1 (shift(1) => no look-ahead).
    Costs = turnover * (commission + half/whole spread) charged on the day the trade executes.
    Weights drift with prices between rebalances (buy-and-hold drift), so turnover is realistic.
    Returns (net daily returns, daily one-way turnover)."""
    cost = (settings.commission_bps + settings.spread_bps) / 1e4 if cost_bps is None else cost_bps / 1e4
    r = px.pct_change().fillna(0.0)[weights.columns]
    target = weights.shift(1).dropna(how="all").fillna(0.0)
    r = r.loc[target.index]

    # Simulate drift: trade only on days where the target changes; otherwise let weights drift.
    changed = target.ne(target.shift(1)).any(axis=1)
    held = []
    w = target.iloc[0].copy()
    turn = []
    for i, d in enumerate(target.index):
        if i == 0 or changed.iloc[i]:
            new = target.iloc[i]
            turn.append(float((new - w).abs().sum()) / 2 if i else float(new.abs().sum()) / 2)
            w = new.copy()
        else:
            turn.append(0.0)
        held.append(w.values.copy())
        gross = 1 + (w * r.iloc[i]).sum()
        w = w * (1 + r.iloc[i]) / gross if gross > 0 else w
    held_df = pd.DataFrame(held, index=target.index, columns=target.columns)
    turnover = pd.Series(turn, index=target.index)
    gross_ret = (held_df * r).sum(axis=1)
    net = gross_ret - turnover * 2 * cost  # one-way turnover * 2 sides traded
    return net, turnover
