"""ATLAS X — read-only public demo (Streamlit).

Safeguards (do not remove):
  * No broker is imported or reachable. There is no IBKR adapter and no paper-broker execution in this
    repository, so no order can be sent anywhere.
  * `atlas.config.PAPER_ONLY` stays True; the app refuses to start otherwise.
  * Orders are only *planned* and *reviewed* by the real risk engine against a throwaway in-memory
    simulation. Nothing is executed, filled or stored beyond the session.
  * No credentials or secrets are needed. Notification webhooks are forced off.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

# Pin the environment BEFORE importing atlas: paper mode, a throwaway database, no webhook.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
_TMP = Path(tempfile.mkdtemp(prefix="atlas-demo-"))
os.environ["ATLAS_MODE"] = "paper"
os.environ["ATLAS_DB"] = str(_TMP / "demo.db")
os.environ["ATLAS_NOTIFY_WEBHOOK"] = ""
os.environ["ATLAS_BROKER"] = "none"

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from atlas import config  # noqa: E402
from atlas.config import CASH_PROXY, UNIVERSE, settings  # noqa: E402
from atlas.db import DB  # noqa: E402
from atlas.portfolio import plan_orders, small_capital_plan  # noqa: E402
from atlas.research import lab  # noqa: E402
from atlas.research.strategies import FAMILIES, target_weights  # noqa: E402
import atlas.risk as _risk  # noqa: E402
from atlas.risk import ProposedOrder, RiskEngine, position_cap  # noqa: E402

if not config.PAPER_ONLY or config.MODE != "paper":
    raise SystemExit("Refusing to start: the demo must run with PAPER_ONLY = True in paper mode.")

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "demo_data" / "prices.csv.gz"

st.set_page_config(page_title="ATLAS X demo", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- data ---------------------------
@st.cache_data(show_spinner=False)
def load_prices() -> pd.DataFrame:
    df = pd.read_csv(SNAPSHOT, parse_dates=["date"])
    return df[df["symbol"].isin(list(UNIVERSE))]


def panel(df: pd.DataFrame, field: str = "adjclose") -> pd.DataFrame:
    p = df.pivot(index="date", columns="symbol", values=field).sort_index()
    return p[[s for s in UNIVERSE if s in p.columns]]


@st.cache_data(show_spinner="Running the research lab on every strategy configuration…")
def run_lab_cached(_key: str) -> tuple[list[dict], dict]:
    px = panel(load_prices())
    results = lab.evaluate(px)
    choice = lab.select(results)
    for r in results:  # drop non-serialisable Series before caching
        r.pop("net", None)
    return results, choice


def fresh_db(prices: pd.DataFrame) -> DB:
    """A throwaway database holding only the latest price per ETF, for the risk engine's checks."""
    db = DB(Path(":memory:"))  # private to this call; vanishes when the page reruns
    last = prices.sort_values("date").groupby("symbol").tail(1)
    db.many("INSERT OR REPLACE INTO prices(symbol,date,close,adjclose,volume) VALUES(?,?,?,?,?)",
            [(r.symbol, r.date.strftime("%Y-%m-%d"), float(r.close), float(r.adjclose), float(r.volume or 0))
             for r in last.itertuples()])
    return db


def latest(prices: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    last = prices.sort_values("date").groupby("symbol").tail(1)
    px = {r.symbol: float(r.close) * UNIVERSE[r.symbol]["scale"] for r in last.itertuples()}
    dates = {r.symbol: r.date.strftime("%Y-%m-%d") for r in last.itertuples()}
    return px, dates


def pct(x: float) -> str:
    return f"{x:.1%}"


prices = load_prices()
last_date = prices["date"].max().date()

# The demo prices are a fixed snapshot, so the risk engine's stale-price check counts business days
# from each price to the snapshot date instead of to today. Every other check is unchanged.
_risk.staleness_days = lambda d: max(0, int(pd.bdate_range(d, last_date).size) - 1)


@st.cache_data(show_spinner="Computing target weights…")
def cached_target(fam: str, params_key: str) -> dict:
    return target_weights(fam, json.loads(params_key), panel(load_prices()).dropna()).to_dict()

# ---------------------------------------------------------------- header -------------------------
st.title("ATLAS X — tax- and cost-aware allocator (read-only demo)")
st.warning(
    "**Paper-only, read-only demo.** No broker is connected and nothing on this page can place, send or "
    "simulate a real trade. It shows the research lab, portfolio construction and risk engine of ATLAS X "
    f"on a snapshot of public daily prices (to {last_date:%d %b %Y}). Nothing here is investment advice, "
    "and a backtest is not evidence of future returns.", icon="🛡️")

tab_over, tab_data, tab_lab, tab_plan, tab_risk = st.tabs(
    ["Overview", "Market data", "Research lab", "Order planner", "Risk engine sandbox"])

# ---------------------------------------------------------------- overview -----------------------
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trading mode", "Paper only")
    c2.metric("Broker", "None (demo)")
    c3.metric("ETFs in universe", len(UNIVERSE))
    c4.metric("Price history to", f"{last_date:%d %b %Y}")
    st.markdown(
        "ATLAS X researches, chooses, sizes and monitors a long-only portfolio of London-listed UCITS "
        "ETFs. The money individual investors reliably lose goes to tax, costs, idle cash and "
        "over-trading, so the platform captures diversified market returns and stops those leaks. "
        "Every proposed trade goes through a **risk engine that has the final say**.\n\n"
        "In the full installation the platform runs on the investor's own PC against an IBKR **paper** "
        "account. This public demo contains no broker code at all.")
    st.subheader("Instrument whitelist and concentration caps")
    uni = pd.DataFrame([{"Ticker": s, "Name": m["name"], "Asset class": m["cls"],
                         "Quote unit": "pence" if m["scale"] == 0.01 else "pounds",
                         "Max weight": pct(position_cap(s))} for s, m in UNIVERSE.items()])
    st.dataframe(uni, hide_index=True, width="stretch")
    L = settings.limits
    st.subheader("Hard risk limits")
    st.dataframe(pd.DataFrame([
        ("Short selling", "prohibited" if not L.allow_short else "allowed"),
        ("Leverage", "prohibited" if not L.allow_leverage else "allowed"),
        ("Max gross exposure", pct(L.max_gross_exposure)),
        ("Max single order", f"{pct(L.max_order_fraction)} of equity"),
        ("Max daily turnover", f"{pct(L.max_daily_turnover)} of equity"),
        ("Daily loss breaker", f"-{pct(L.daily_loss_limit)}"),
        ("Drawdown breaker", f"-{pct(L.max_drawdown)} from peak"),
        ("Stale-price block", f"> {L.max_data_staleness_days} business days"),
        ("Minimum order", f"£{L.min_order_gbp:,.0f}"),
        ("Paper-only lock", "ON (code constant, not configurable)"),
    ], columns=["Limit", "Value"]), hide_index=True, width="stretch")

# ---------------------------------------------------------------- market data --------------------
with tab_data:
    syms = st.multiselect("ETFs", list(UNIVERSE), default=["VWRL.L", "IGLT.L", "SGLN.L"],
                          format_func=lambda s: f"{s} — {UNIVERSE[s]['name']}")
    start = st.slider("From year", 2008, int(last_date.year), 2014)
    if syms:
        p = panel(prices)[syms]
        p = p[p.index >= pd.Timestamp(f"{start}-01-01")].dropna(how="all")
        norm = p / p.bfill().iloc[0] * 100
        long = norm.reset_index().melt("date", var_name="ETF", value_name="Growth of 100")
        st.altair_chart(alt.Chart(long).mark_line().encode(
            x="date:T", y="Growth of 100:Q", color="ETF:N",
            tooltip=["date:T", "ETF:N", alt.Tooltip("Growth of 100:Q", format=".1f")]).interactive(),
            width="stretch")
        st.caption("Total return (dividend-adjusted closes), rebased to 100. Source: Yahoo Finance daily "
                   "history, cleaned of bad prints by `atlas.data.clean`.")
    cov = prices.groupby("symbol")["date"].agg(["min", "max", "count"]).reset_index()
    cov.columns = ["Ticker", "First date", "Last date", "Rows"]
    st.dataframe(cov, hide_index=True, width="stretch")

# ---------------------------------------------------------------- research lab -------------------
with tab_lab:
    st.markdown(
        "Every strategy family is tested under one protocol fixed in advance: costs and next-day "
        "execution included, parameters chosen in-sample only, a **Deflated Sharpe Ratio** across every "
        "configuration tried, a robustness check, and active strategies must beat passive ones. "
        "If nothing passes, the platform holds cash.")
    if st.button("Run the research lab", type="primary") or st.session_state.get("lab_ran"):
        st.session_state["lab_ran"] = True
        results, choice = run_lab_cached(str(last_date))
        st.success(f"**Selected: {choice['name']}** — {choice['rationale']}")
        rows = []
        for r in results:
            rows.append({"Strategy": r["name"], "Passed": "✅" if r["passed"] else "❌",
                         "Benchmark": "yes" if r["benchmark"] else "",
                         "OOS Sharpe": round(r["oos"]["sharpe"], 2), "DSR": round(r["dsr"], 2),
                         "OOS CAGR": pct(r["oos"]["cagr"]), "OOS max DD": pct(r["oos"]["max_dd"]),
                         "Robustness": pct(r["robustness"]), "Configs": r["n_configs"],
                         "Best params": str(r["params"] or "—")})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        eq = pd.concat([pd.DataFrame(r["equity"], columns=["date", "equity"]).assign(Strategy=r["name"])
                        for r in results])
        eq["date"] = pd.to_datetime(eq["date"])
        split = pd.DataFrame({"date": [pd.Timestamp(results[0]["split"])]})
        chart = alt.Chart(eq).mark_line().encode(x="date:T", y=alt.Y("equity:Q", title="Growth of £1 (net)"),
                                                 color="Strategy:N", tooltip=["Strategy", "date:T", "equity:Q"])
        rule = alt.Chart(split).mark_rule(strokeDash=[4, 4]).encode(x="date:T")
        st.altair_chart((chart + rule).interactive(), width="stretch")
        st.caption(f"Dashed line: start of the out-of-sample period ({results[0]['split']}), never used to "
                   "choose parameters.")
        with st.expander("Why strategies were rejected"):
            for r in results:
                if r["reasons"]:
                    st.markdown(f"**{r['name']}**")
                    for why in r["reasons"]:
                        st.markdown(f"- {why}")
    else:
        st.info("Press the button to evaluate all 41 configurations (takes about a minute the first time).")

# ---------------------------------------------------------------- order planner ------------------
with tab_plan:
    st.markdown(
        "Choose a strategy and a **simulated** starting balance. ATLAS X turns the target weights into "
        "whole-share orders and the risk engine reviews every one. Nothing is executed.")
    c1, c2 = st.columns(2)
    fam = c1.selectbox("Strategy", list(FAMILIES), index=list(FAMILIES).index("static_60_40"))
    grid = FAMILIES[fam][1]
    params = c1.selectbox("Parameters", grid, format_func=lambda p: str(p) if p else "default")
    equity = c2.number_input("Simulated cash (£)", min_value=100.0, max_value=10_000_000.0,
                             value=100_000.0, step=1_000.0)
    px_now, dates = latest(prices)
    target = cached_target(fam, json.dumps(params, sort_keys=True))
    tgt, band, min_order = small_capital_plan(target, equity)
    tw = pd.DataFrame([{"Ticker": s, "Name": UNIVERSE[s]["name"], "Target weight": pct(w)}
                       for s, w in tgt.items() if w > 0.0005])
    c2.dataframe(tw, hide_index=True, width="stretch")
    if band is not None:
        st.info("Small-capital mode (< £25k): holdings under £1,000 are dropped and the no-trade band widens.")
    orders = plan_orders(tgt, equity, {}, px_now, band=band, min_order_gbp=min_order)
    db = fresh_db(prices)
    engine = RiskEngine(db)
    rep = engine.review(orders, equity=equity, cash=equity, holdings_value={}, price_dates=dates,
                        broker_is_live=False)
    st.subheader(f"{len(rep.approved)} approved, {len(rep.rejected)} rejected")
    out = [{"Decision": "approved", "Side": a.order.side, "Ticker": a.order.symbol, "Qty": int(a.order.qty),
            "Price £": round(a.order.ref_price, 2), "Notional £": round(a.order.qty * a.order.ref_price, 2),
            "Reason / checks": ", ".join(a.checks)} for a in rep.approved]
    out += [{"Decision": "rejected", "Side": o.side, "Ticker": o.symbol, "Qty": int(o.qty),
             "Price £": round(o.ref_price, 2), "Notional £": round(o.qty * o.ref_price, 2),
             "Reason / checks": why} for o, why in rep.rejected]
    if out:
        st.dataframe(pd.DataFrame(out), hide_index=True, width="stretch")
        spent = sum(a.order.qty * a.order.ref_price for a in rep.approved)
        st.caption(f"Approved notional £{spent:,.0f} of £{equity:,.0f}; the rest stays as cash "
                   f"(1% buffer for commissions plus rounding to whole shares). Not executed.")
    else:
        st.info("No orders: the targets are too small to trade at this balance.")

# ---------------------------------------------------------------- risk sandbox -------------------
with tab_risk:
    st.markdown(
        "Try to get a bad order past the risk engine. This is the same `RiskEngine.review` the platform "
        "uses before every trade, run here against a throwaway simulation. Nothing is sent anywhere.")
    c1, c2, c3 = st.columns(3)
    sym = c1.text_input("Ticker", "VUSA.L").strip().upper()
    side = c1.radio("Side", ["buy", "sell"], horizontal=True)
    qty = c2.number_input("Quantity", min_value=0.0, value=1500.0, step=10.0)
    ref = c2.number_input("Price £ (0 = latest)", min_value=0.0, value=0.0, step=1.0)
    eq2 = c3.number_input("Account equity £", min_value=0.0, value=100_000.0, step=1_000.0)
    held = c3.number_input(f"Currently held (£ of {sym or 'ticker'})", min_value=0.0, value=0.0, step=100.0)
    c4, c5 = st.columns(2)
    kill = c4.checkbox("Kill switch engaged")
    suspended = c4.checkbox("Trading suspended (drawdown breaker tripped)")
    live = c5.checkbox("Pretend the broker account is LIVE")
    stale = c5.checkbox("Pretend the price is 10 days old")
    if st.button("Review this order", type="primary"):
        px_now, dates = latest(prices)
        price = ref or px_now.get(sym, 0.0)
        db = fresh_db(prices)
        eng = RiskEngine(db)
        if kill:
            db.set("kill_switch", True)
        if suspended:
            db.set("suspended", "demo: drawdown breaker")
        d2 = dict(dates)
        if stale and sym in d2:
            d2[sym] = (pd.Timestamp(d2[sym]) - pd.offsets.BDay(10)).strftime("%Y-%m-%d")
        rep = eng.review([ProposedOrder(sym, side, float(qty), float(price))], equity=eq2,
                         cash=max(0.0, eq2 - held), holdings_value={sym: held} if held else {},
                         price_dates=d2, broker_is_live=live)
        if rep.approved:
            a = rep.approved[0]
            st.success(f"Approved by the risk engine (checks: {', '.join(a.checks)}). "
                       f"Notional £{qty * price:,.0f}. **Not executed** — this demo has no broker.")
        else:
            st.error(f"Rejected: {rep.rejected[0][1]}")

st.divider()
st.caption("ATLAS X demo · paper-only · no broker · no orders executed · not investment advice.")
