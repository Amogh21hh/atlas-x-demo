"""Performance statistics incl. Probabilistic & Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2012/2014)."""
from __future__ import annotations

import math

from statistics import NormalDist

import numpy as np
import pandas as pd

class _Norm:  # stdlib replacement for scipy.stats.norm (avoids a heavy dependency)
    _d = NormalDist()
    cdf = staticmethod(_d.cdf)
    ppf = staticmethod(_d.inv_cdf)


norm = _Norm()

TD = 252
EULER = 0.5772156649


def sharpe(r: pd.Series) -> float:
    s = r.std()
    return float(r.mean() / s * math.sqrt(TD)) if s > 0 else 0.0


def max_drawdown(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def cagr(r: pd.Series) -> float:
    if len(r) == 0:
        return 0.0
    total = float((1 + r).prod())
    yrs = len(r) / TD
    return total ** (1 / yrs) - 1 if total > 0 else -1.0


def psr(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """Probability the true (per-period) Sharpe exceeds sr_benchmark, adjusting for skew/kurtosis."""
    r = r.dropna()
    n = len(r)
    if n < 30 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std()
    g3 = float(r.skew())
    g4 = float(r.kurt()) + 3.0
    denom = math.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sr(trial_srs: list[float]) -> float:
    """Expected maximum per-period Sharpe among N unskilled trials (False Strategy Theorem)."""
    n = len(trial_srs)
    if n < 2:
        return 0.0
    v = float(np.var(trial_srs, ddof=1))
    return math.sqrt(v) * ((1 - EULER) * norm.ppf(1 - 1 / n) + EULER * norm.ppf(1 - 1 / (n * math.e)))


def dsr(r: pd.Series, trial_srs_per_period: list[float]) -> float:
    return psr(r, expected_max_sr(trial_srs_per_period))


def summary(r: pd.Series, turnover: pd.Series | None = None) -> dict:
    r = r.dropna()
    vol = float(r.std() * math.sqrt(TD))
    mdd = max_drawdown(r)
    c = cagr(r)
    out = {
        "cagr": c, "vol": vol, "sharpe": sharpe(r), "max_dd": mdd,
        "calmar": c / abs(mdd) if mdd < 0 else 0.0, "days": len(r),
        "start": str(r.index[0].date()) if len(r) else None,
        "end": str(r.index[-1].date()) if len(r) else None,
    }
    if turnover is not None:
        out["turnover_ann"] = float(turnover.reindex(r.index).fillna(0).sum() / (len(r) / TD))
    return out
