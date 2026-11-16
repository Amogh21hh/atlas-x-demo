# ATLAS X — read-only Streamlit demo

A public, read-only demo of **ATLAS X**, a tax- and cost-aware allocator for a long-only portfolio of
London-listed UCITS ETFs. It runs the project's real research lab, portfolio construction and risk
engine on a snapshot of public daily prices.

> **Paper-only, read-only.** This repository contains **no broker code** (no IBKR adapter, no paper
> broker execution). Nothing in the app can place, send or simulate a real trade. It is not
> investment advice, and a backtest is not evidence of future returns.

## What it shows

| Tab | What it does |
|---|---|
| Overview | The instrument whitelist, per-ETF concentration caps and the hard risk limits |
| Market data | Total-return price history for the 10 ETFs (dividend-adjusted, bad prints cleaned) |
| Research lab | Evaluates 6 strategy families / 41 configurations: next-day execution, costs, in-sample parameter choice, out-of-sample check, Deflated Sharpe Ratio, robustness gate; selects one or holds cash |
| Order planner | Turns a strategy's target weights into whole-share orders for a *simulated* balance, then the risk engine approves or rejects each one. Nothing is executed |
| Risk engine sandbox | Try to get a bad order past the risk engine (oversized, short sale, stale price, kill switch, "live" account…) |

## Safeguards

* `atlas/config.py` keeps `PAPER_ONLY = True` (a code constant). The app refuses to start if it is
  ever off or the mode is not `paper`.
* No broker modules are included, so there is nothing that could connect to a brokerage.
* Every simulation uses a throwaway in-memory database that disappears on the next page rerun.
* Notification webhooks are forced off. The app needs **no secrets**, and none are stored here.
* Because the prices are a fixed snapshot, the risk engine's stale-price check counts business days
  to the snapshot date instead of to today. Every other check is the unchanged production code.

## Data

`demo_data/prices.csv.gz` holds public daily closes for the 10 ETFs, from Yahoo Finance and
cleaned by `atlas.data.clean`, up to 13 Nov 2026. It contains no account, order or personal data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

`python smoke_test.py` runs every tab headlessly with Streamlit's AppTest.

## Deploy (Streamlit Community Cloud)

Repository: this repo · Branch: `main` · Main file: `streamlit_app.py` · Python 3.12.
No secrets need to be configured.
