"""Risk engine — FINAL AUTHORITY over every transaction.

Execution only accepts `ApprovedOrder` objects, which can only be minted here (guarded by a private
sentinel). Every decision — approve or reject, and why — is persisted with the order.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .config import CASH_PROXY, PAPER_ONLY, UNIVERSE, RiskLimits, settings
from .data import staleness_days
from .db import DB

_MINT = object()
GILT_CAP = 0.30


def position_cap(symbol: str, limits: RiskLimits | None = None) -> float:
    L = limits or settings.limits
    return float(UNIVERSE.get(symbol, {}).get("max_w", L.max_position_weight))


@dataclass
class ProposedOrder:
    symbol: str
    side: str           # buy | sell
    qty: float
    ref_price: float    # GBP


@dataclass
class ApprovedOrder:
    order: ProposedOrder
    checks: list[str]
    _token: object = field(repr=False, default=None)

    def __post_init__(self):
        if self._token is not _MINT:
            raise PermissionError("ApprovedOrder can only be created by the RiskEngine")


@dataclass
class RiskReport:
    approved: list[ApprovedOrder]
    rejected: list[tuple[ProposedOrder, str]]
    state: dict


class RiskEngine:
    def __init__(self, db: DB, limits: RiskLimits | None = None, live=None):
        """`live` = verified LiveLimits from atlas.live.gate(); None for paper. Without it, every order
        on a live account is rejected."""
        self.db = db
        self.L = limits or settings.limits
        self.live = live
        self.allowed_gilts: set[str] = set()   # conventional UK gilts the sleeve may buy this cycle

    # ---- persistent state ------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "kill_switch": bool(self.db.get("kill_switch", False)),
            "suspended": self.db.get("suspended", None),        # reason string or None
            "live_authorised": self.db.get("live_authorisation", None),
            "peak_equity": self.db.get("peak_equity", None),
            "day": self.db.get("risk_day", None),
            "day_start_equity": self.db.get("day_start_equity", None),
            "day_turnover": self.db.get("day_turnover", 0.0),
        }

    def update_equity(self, equity: float) -> dict:
        """Mark-to-market hook: updates peak/day state and trips circuit breakers."""
        today = date.today().isoformat()
        if self.db.get("risk_day") != today:
            self.db.set("risk_day", today)
            self.db.set("day_start_equity", equity)
            self.db.set("day_turnover", 0.0)
        peak = max(self.db.get("peak_equity", equity) or equity, equity)
        self.db.set("peak_equity", peak)
        dd = equity / peak - 1 if peak else 0.0
        day0 = self.db.get("day_start_equity", equity) or equity
        dl = equity / day0 - 1 if day0 else 0.0
        if self.live is not None and not self.db.get("suspended"):
            # Live limits are in pounds against the CAPPED capital, so a large account with a small
            # authorised cap is still stopped by small losses.
            cap_peak = min(peak, self.live.max_capital_gbp)
            loss_from_peak = peak - equity
            day_loss = day0 - equity
            if loss_from_peak >= self.live.max_drawdown_pct * cap_peak > 0:
                self.suspend(f"live drawdown £{loss_from_peak:,.0f} >= {self.live.max_drawdown_pct:.0%} of capped capital")
            elif day_loss >= self.live.max_daily_loss_gbp:
                self.suspend(f"live daily loss £{day_loss:,.0f} >= limit £{self.live.max_daily_loss_gbp:,.0f}")
        if dd <= -self.L.max_drawdown and not self.db.get("suspended"):
            self.suspend(f"max drawdown breached ({dd:.1%} from peak £{peak:,.0f})")
        elif dl <= -self.L.daily_loss_limit and not self.db.get("suspended"):
            self.suspend(f"daily loss limit breached ({dl:.1%})")
        return {"drawdown": dd, "day_pnl": dl, "peak": peak}

    def suspend(self, reason: str) -> None:
        self.db.set("suspended", reason)
        self.db.event("critical", "risk_suspend", f"Trading suspended: {reason}. Only risk-reducing orders allowed until resumed.")

    def resume(self, by: str) -> None:
        self.db.set("suspended", None)
        self.db.set("peak_equity", None)  # reset peak so a resumed book is not instantly re-tripped
        self.db.event("warn", "risk_resume", f"Trading resumed by {by}")

    def set_kill_switch(self, on: bool, by: str) -> None:
        self.db.set("kill_switch", on)
        self.db.event("critical" if on else "warn", "kill_switch", f"Kill switch {'ENGAGED' if on else 'released'} by {by}")

    # ---- the gate --------------------------------------------------------------------------
    def review(self, orders: list[ProposedOrder], equity: float, cash: float,
               holdings_value: dict[str, float], price_dates: dict[str, str], broker_is_live: bool,
               reduce_only: bool = False) -> RiskReport:
        st = self.state()
        approved, rejected = [], []
        proj = dict(holdings_value)
        proj_cash = cash
        turnover = float(st["day_turnover"] or 0.0)

        # sells first so their proceeds fund buys
        for o in sorted(orders, key=lambda o: o.side != "sell"):
            notional = o.qty * o.ref_price
            reducing = o.side == "sell" or o.symbol == CASH_PROXY
            checks = []

            def reject(msg):
                rejected.append((o, msg))

            if st["kill_switch"]:
                reject("kill switch engaged — all trading halted"); continue
            if broker_is_live is not False and PAPER_ONLY:
                reject("paper-only lock: broker account is live or unverified"); continue
            if broker_is_live and self.live is None:
                reject("live account but no verified live authorisation for this cycle"); continue
            is_gilt = o.symbol.startswith("GILT:")
            if o.symbol not in UNIVERSE and not (is_gilt and (o.symbol in self.allowed_gilts or o.side == "sell")):
                reject(f"{o.symbol} not in authorised instrument whitelist"); continue
            if o.side not in ("buy", "sell") or o.qty <= 0 or o.ref_price <= 0:
                reject("malformed order"); continue
            if (st["suspended"] or reduce_only) and not reducing:
                reject(f"risk-increasing order blocked: {st['suspended'] or 'reduce-only mode'}"); continue
            d = price_dates.get(o.symbol)
            max_stale = self.live.max_data_staleness_days if self.live else self.L.max_data_staleness_days
            if not d or staleness_days(d) > max_stale:
                reject(f"stale or missing price ({d})"); continue
            checks.append("fresh_price")
            if notional < self.L.min_order_gbp:
                reject(f"below minimum order size £{self.L.min_order_gbp:.0f}"); continue
            if equity <= 0:
                reject("non-positive equity"); continue
            if self.live is not None and o.side == "buy":
                if notional > self.live.max_order_gbp:
                    reject(f"order £{notional:,.0f} > live max order £{self.live.max_order_gbp:,.0f}"); continue
                if turnover + notional > self.live.turnover_cap:
                    reject(f"live daily turnover would exceed £{self.live.turnover_cap:,.0f}"); continue
                deployed = sum(v for s, v in proj.items() if v > 0) + notional
                if deployed > self.live.max_capital_gbp + 1e-6:
                    reject(f"would invest £{deployed:,.0f} > live capital cap £{self.live.max_capital_gbp:,.0f}"); continue
            if notional > self.L.max_order_fraction * equity and not reducing:
                reject(f"order £{notional:,.0f} > {self.L.max_order_fraction:.0%} of equity"); continue
            if (turnover + notional) > self.L.max_daily_turnover * equity and not reducing:
                reject("daily turnover limit"); continue
            if o.side == "sell":
                if proj.get(o.symbol, 0.0) + 1e-6 < notional * 0.999 and not self.L.allow_short:
                    reject("sell exceeds holding (short selling prohibited)"); continue
                proj[o.symbol] = proj.get(o.symbol, 0.0) - notional
                proj_cash += notional
            else:
                if notional > proj_cash + 1e-6 and not self.L.allow_leverage:
                    reject("insufficient settled cash (leverage prohibited)"); continue
                base = min(equity, self.live.max_capital_gbp) if self.live else equity
                new_w = (proj.get(o.symbol, 0.0) + notional) / base
                if is_gilt:
                    cap = GILT_CAP   # per-gilt cap; the sleeve itself is bounded by the strategy's cash/bond share
                else:
                    cap = self.live.cap_for(o.symbol) if self.live else position_cap(o.symbol, self.L)
                if new_w > cap + 1e-9:
                    reject(f"post-trade weight {new_w:.1%} > cap {cap:.0%}"); continue
                gross = (sum(proj.values()) + notional) / base
                if gross > self.L.max_gross_exposure + 1e-6:
                    reject(f"gross exposure {gross:.2f} > {self.L.max_gross_exposure}"); continue
                proj[o.symbol] = proj.get(o.symbol, 0.0) + notional
                proj_cash -= notional
            checks += ["whitelist", "size", "turnover", "cash", "weight_cap", "gross"]
            turnover += notional
            approved.append(ApprovedOrder(o, checks, _MINT))
        return RiskReport(approved, rejected, st)

    def record_turnover(self, notional: float) -> None:
        self.db.set("day_turnover", float(self.db.get("day_turnover", 0.0) or 0.0) + notional)
