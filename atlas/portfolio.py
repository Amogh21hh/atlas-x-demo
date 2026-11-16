"""Portfolio construction: target weights -> whole-share orders, with a no-trade band to limit costs."""
from __future__ import annotations

import math

from .config import CASH_PROXY, settings
from .risk import ProposedOrder

CASH_BUFFER = 0.01  # keep 1% in broker cash for commissions / rounding


SMALL_CAPITAL_GBP = 25_000.0     # below this, small-capital deployment rules apply
SMALL_MIN_POSITION_GBP = 1_000.0  # no position smaller than this (keeps £3 commission <= 0.3%)
SMALL_BAND = 0.05                 # wider no-trade band (fewer, larger trades)
SMALL_MIN_ORDER_GBP = 250.0


def small_capital_plan(target: dict[str, float], capital: float):
    """Conservative sizing for small accounts. Returns (target, band, min_order_gbp).

    Drops any holding whose allocation would be under £1,000, renormalises the survivors while
    respecting each ETF's cap, and leaves anything that cannot be placed as uninvested cash (never
    forced into the cash-proxy ETF below the minimum size). Above £25k returns the target unchanged."""
    if capital >= SMALL_CAPITAL_GBP or capital <= 0:
        return target, None, None
    from .config import UNIVERSE
    risky = {s: w for s, w in target.items() if w > 0 and s != CASH_PROXY}
    keep = {s: w for s, w in risky.items() if w * capital >= SMALL_MIN_POSITION_GBP}
    out = {s: 0.0 for s in target}
    if keep:
        total = sum(keep.values())
        invest = min(1.0, sum(risky.values()))  # never invest more than the strategy intended
        for s, w in keep.items():
            out[s] = min(UNIVERSE[s]["max_w"], invest * w / total)
    rest = 1.0 - sum(out.values())
    if rest * capital >= SMALL_MIN_POSITION_GBP and target.get(CASH_PROXY, 0) > 0:
        out[CASH_PROXY] = rest
    return out, SMALL_BAND, SMALL_MIN_ORDER_GBP


def plan_orders(target: dict[str, float], equity: float, holdings_qty: dict[str, float],
                prices: dict[str, float], band: float | None = None, force: bool = False,
                min_order_gbp: float | None = None) -> list[ProposedOrder]:
    """Contribution-led rebalancing (see docs/RESEARCH_AND_DECISION.md, experiment E1).

    1. If no holding is more than `band` ABOVE its target, nothing is sold: spare cash (deposits,
       dividends, idle cash above the 1% buffer) is spent on the underweight holdings, in proportion
       to how far below target each one is.
    2. Only when a holding is overweight by `band` or more (or `force`) is the book fully rebalanced,
       which may sell. This avoids most taxable disposals and trading costs without letting risk drift.
    """
    band = settings.rebalance_band if band is None else band
    investable = equity * (1 - CASH_BUFFER)
    cur_val = {s: holdings_qty.get(s, 0.0) * prices[s] for s in prices}
    cur_w = {s: v / equity for s, v in cur_val.items()} if equity > 0 else {}
    over = max((cur_w.get(s, 0.0) - target.get(s, 0.0) for s in prices), default=0.0)
    if not force and over < band:
        free_cash = equity - sum(cur_val.values()) - CASH_BUFFER * equity
        floor_gbp = min_order_gbp or settings.limits.min_order_gbp
        if free_cash < floor_gbp:
            return []
        deficit = {s: max(0.0, target.get(s, 0.0) * investable - cur_val[s]) for s in prices}
        total_def = sum(deficit.values())
        if total_def <= 0:
            return []
        buys = []
        for s, d in deficit.items():
            spend = min(d, free_cash * d / total_def)
            qty = math.floor(spend / prices[s]) if prices[s] > 0 else 0
            if qty >= 1 and qty * prices[s] >= floor_gbp:
                buys.append(ProposedOrder(s, "buy", float(qty), prices[s]))
        return buys
    orders = []
    for s, px in prices.items():
        tgt_qty = math.floor(investable * target.get(s, 0.0) / px) if px > 0 else 0
        delta = tgt_qty - holdings_qty.get(s, 0.0)
        if abs(delta) < 1:
            continue
        # skip tiny legs that are inside the band unless the target is a full exit
        leg_drift = abs(target.get(s, 0.0) - cur_w.get(s, 0.0))
        if not force and leg_drift < band / 3 and target.get(s, 0.0) > 0:
            continue
        if min_order_gbp and abs(delta) * px < min_order_gbp and target.get(s, 0.0) > 0:
            continue
        orders.append(ProposedOrder(s, "buy" if delta > 0 else "sell", float(abs(delta)), px))
    return orders


def liquidation_orders(holdings_qty: dict[str, float], prices: dict[str, float]) -> list[ProposedOrder]:
    """Emergency: sell every risky holding (cash proxy retained)."""
    return [ProposedOrder(s, "sell", float(q), prices[s]) for s, q in holdings_qty.items()
            if q > 0 and s != CASH_PROXY and s in prices]
