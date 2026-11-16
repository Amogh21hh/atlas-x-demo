"""Research lab: evaluates every candidate strategy under a fixed protocol and gates them.

Protocol (fixed BEFORE looking at results to limit researcher degrees of freedom):
  1. Sample = dates where every universe ETF has a price (no back-filling of missing ETFs).
  2. In-sample (IS) = first 55% of the sample; out-of-sample (OOS) = remainder, never used to choose
     parameters. Parameters per family are chosen by IS Sharpe only.
  3. All returns are net of commission + spread; execution is next-day close.
  4. Deflated Sharpe Ratio uses the per-period Sharpe of EVERY configuration tried (all families),
     so multiple-testing is penalised.
  5. Gate (all must pass):
       - OOS Sharpe >= 0.30
       - full-sample DSR >= 0.90
       - OOS max drawdown >= -25%
       - parameter robustness: >= 60% of the family's grid has positive IS Sharpe (plateau, not spike)
       - active strategies must beat the passive benchmark OOS on Sharpe, OR match it (>= -0.05)
         with a drawdown at least 25% shallower
  6. Selection: highest full-sample DSR among passing strategies; if none pass, hold the cash proxy.
Known residual biases (documented, not hidden): the ETF universe was chosen in 2026 (mild
survivorship/selection bias); Yahoo adjusted closes are the only data source; daily closes ignore
intraday liquidity; fixed £ minimum commissions are approximated in bps.
"""
from __future__ import annotations

import json
import math

import pandas as pd

from ..config import CASH_PROXY, settings
from ..data import load_panel
from ..db import DB, now
from ..risk import position_cap
from . import stats
from .backtest import run
from .strategies import FAMILIES

IS_FRACTION = 0.55
GATE = {"oos_sharpe_min": 0.30, "dsr_min": 0.90, "oos_max_dd": -0.25, "robust_min": 0.60}


def evaluate(px: pd.DataFrame) -> list[dict]:
    px = px.dropna()
    split = px.index[int(len(px) * IS_FRACTION)]
    all_trials, fam_results = [], {}
    for fam, (fn, grid, is_bench) in FAMILIES.items():
        trials = []
        for params in grid:
            w = fn(px, **params)
            net, turn = run(px, w)
            is_r, oos_r = net[net.index < split], net[net.index >= split]
            sr_pp = float(net.mean() / net.std()) if net.std() > 0 else 0.0
            breaches = [s for s in w.columns if w[s].max() > position_cap(s) + 1e-6]
            trials.append({"params": params, "net": net, "turn": turn, "is_sharpe": stats.sharpe(is_r),
                           "oos": oos_r, "is": is_r, "breaches": breaches})
            all_trials.append(sr_pp)
        fam_results[fam] = (trials, is_bench)

    bench_oos = None
    results = []
    for fam, (trials, is_bench) in fam_results.items():
        best = max(trials, key=lambda t: t["is_sharpe"])
        robust = sum(t["is_sharpe"] > 0 for t in trials) / len(trials)
        full = stats.summary(best["net"], best["turn"])
        oos = stats.summary(best["oos"], best["turn"])
        ins = stats.summary(best["is"], best["turn"])
        d = stats.dsr(best["net"], all_trials)
        res = {"name": fam, "params": best["params"], "benchmark": is_bench, "full": full, "is": ins,
               "oos": oos, "dsr": d, "psr0": stats.psr(best["net"]), "robustness": robust,
               "n_configs": len(trials), "n_trials_total": len(all_trials), "split": str(split.date()),
               "breaches": best["breaches"],
               "equity": _equity_curve(best["net"])}
        if fam == "buy_hold_global_equity":
            bench_oos = oos
        results.append(res)

    for res in results:
        reasons = []
        o = res["oos"]
        if res["breaches"]:
            reasons.append(f"not executable within risk limits (concentration cap breached: {', '.join(res['breaches'])})")
        if o["sharpe"] < GATE["oos_sharpe_min"]:
            reasons.append(f"OOS Sharpe {o['sharpe']:.2f} < {GATE['oos_sharpe_min']}")
        if res["dsr"] < GATE["dsr_min"]:
            reasons.append(f"DSR {res['dsr']:.2f} < {GATE['dsr_min']} (multiple-testing adjusted)")
        if o["max_dd"] < GATE["oos_max_dd"]:
            reasons.append(f"OOS max DD {o['max_dd']:.1%} worse than {GATE['oos_max_dd']:.0%}")
        if res["robustness"] < GATE["robust_min"]:
            reasons.append(f"only {res['robustness']:.0%} of parameter grid profitable IS (fragile)")
        if not res["benchmark"] and bench_oos:
            beats = o["sharpe"] > bench_oos["sharpe"]
            matches_safer = (o["sharpe"] >= bench_oos["sharpe"] - 0.05 and
                             o["max_dd"] >= 0.75 * bench_oos["max_dd"])
            if not (beats or matches_safer):
                reasons.append(f"does not beat passive benchmark OOS (Sharpe {o['sharpe']:.2f} vs "
                               f"{bench_oos['sharpe']:.2f}, DD {o['max_dd']:.1%} vs {bench_oos['max_dd']:.1%})")
        res["passed"] = not reasons
        res["reasons"] = reasons
    return results


def _equity_curve(r: pd.Series, points: int = 260) -> list[list]:
    eq = (1 + r).cumprod()
    step = max(1, len(eq) // points)
    s = eq.iloc[::step]
    return [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in s.items()]


def select(results: list[dict]) -> dict:
    passing = [r for r in results if r["passed"]]
    if not passing:
        return {"name": "cash", "params": {}, "rationale": "No strategy passed the evidence gate; holding cash proxy."}
    best = max(passing, key=lambda r: (r["dsr"], r["oos"]["calmar"]))
    return {"name": best["name"], "params": best["params"],
            "rationale": (f"Highest deflated Sharpe ({best['dsr']:.2f}) among {len(passing)} passing "
                          f"strategies; OOS Sharpe {best['oos']['sharpe']:.2f}, OOS max DD "
                          f"{best['oos']['max_dd']:.1%}, ann. turnover {best['full'].get('turnover_ann', 0):.1f}x.")}


def run_lab(db: DB) -> dict:
    px = load_panel(db)
    results = evaluate(px)
    ts = now()
    for r in results:
        db.x("INSERT INTO strategy_runs(run_at,name,params,metrics,passed,reasons,equity) VALUES(?,?,?,?,?,?,?)",
             (ts, r["name"], json.dumps(r["params"]),
              json.dumps({k: r[k] for k in ("full", "is", "oos", "dsr", "psr0", "robustness", "n_configs",
                                             "n_trials_total", "split", "benchmark")}, default=_safe),
              int(r["passed"]), json.dumps(r["reasons"]), json.dumps(r["equity"])))
    choice = select(results)
    prev = db.q("SELECT strategy FROM selection ORDER BY id DESC LIMIT 1")
    db.x("INSERT INTO selection(decided_at,strategy,rationale) VALUES(?,?,?)",
         (ts, json.dumps({"name": choice["name"], "params": choice["params"]}), choice["rationale"]))
    if not prev or json.loads(prev[0]["strategy"])["name"] != choice["name"]:
        db.event("warn", "strategy_change", f"Active strategy -> {choice['name']}: {choice['rationale']}")
    db.set("lab_ran_at", ts)
    return {"results": results, "selection": choice}


def _safe(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return str(o)


def current_selection(db: DB) -> dict:
    r = db.q("SELECT * FROM selection ORDER BY id DESC LIMIT 1")
    if not r:
        return {"name": "cash", "params": {}, "rationale": "Lab has not run yet."}
    s = json.loads(r[0]["strategy"])
    return {**s, "rationale": r[0]["rationale"], "decided_at": r[0]["decided_at"]}


__all__ = ["run_lab", "evaluate", "select", "current_selection", "CASH_PROXY", "settings"]
