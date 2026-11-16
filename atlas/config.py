"""Central configuration. Secrets come ONLY from environment variables / .env, never from code."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

# Universe: London-listed UCITS ETFs available to UK retail investors (US-domiciled ETFs are
# generally not purchasable by UK retail under PRIIPs). `scale` converts the quote unit to GBP
# (GBp/pence lines quote in pence -> 0.01).
# `max_w` = per-instrument concentration cap enforced by the risk engine. Broad, highly diversified
# index funds get a higher cap than single-country / single-asset funds.
UNIVERSE: dict[str, dict] = {
    "VWRL.L": {"name": "Vanguard FTSE All-World", "cls": "equity", "scale": 1.0, "max_w": 0.65},
    "VUSA.L": {"name": "Vanguard S&P 500", "cls": "equity", "scale": 1.0, "max_w": 0.40},
    "VUKE.L": {"name": "Vanguard FTSE 100", "cls": "equity", "scale": 1.0, "max_w": 0.35},
    "EMIM.L": {"name": "iShares Core MSCI EM IMI", "cls": "equity", "scale": 0.01, "max_w": 0.25},
    "IUKP.L": {"name": "iShares UK Property", "cls": "real_estate", "scale": 0.01, "max_w": 0.20},
    "IGLT.L": {"name": "iShares Core UK Gilts", "cls": "bond", "scale": 1.0, "max_w": 0.45},
    "INXG.L": {"name": "iShares UK Index-Linked Gilts", "cls": "bond", "scale": 1.0, "max_w": 0.35},
    "SLXX.L": {"name": "iShares Core GBP Corp Bond", "cls": "bond", "scale": 1.0, "max_w": 0.35},
    # Yahoo's SGLN history before Jul 2011 mixes USD-cent and GBp quotes for weeks at a time
    # (≈3,000 vs ≈1,860), so it is discarded. Research uses data from 2014 anyway.
    "SGLN.L": {"name": "iShares Physical Gold", "cls": "commodity", "scale": 0.01, "max_w": 0.20,
               "history_from": "2011-07-01"},
    "ERNS.L": {"name": "iShares GBP Ultrashort Bond (cash proxy)", "cls": "cash", "scale": 1.0, "max_w": 1.0},
}
# Expected ISINs, checked against IBKR contract details by `atlas.cli ibkr-check`. SLXX and ERNS
# corrected 2026-11-14 after IBKR and justETF both reported different ISINs from the original list.
# A mismatch is reported as a failure; the platform never silently substitutes an instrument.
ISINS = {
    "VWRL.L": "IE00B3RBWM25", "VUSA.L": "IE00B3XXRP09", "VUKE.L": "IE00B810Q511",
    "EMIM.L": "IE00BKM4GZ66", "IUKP.L": "IE00B1TXLS18", "IGLT.L": "IE00B1FZSB30",
    "INXG.L": "IE00B1FZSD53", "SLXX.L": "IE00B00FV011", "SGLN.L": "IE00B4ND3602",
    "ERNS.L": "IE00BCRY6441",
}
CASH_PROXY = "ERNS.L"

# HARD PAPER-ONLY LOCK. Deliberately not configurable through env/.env: with this set, any
# broker account that is live (or whose type cannot be verified) engages the kill switch and
# every order is rejected, and the live-authorisation endpoint is disabled. Turning it off is a
# reviewed code change, never a runtime setting.
PAPER_ONLY = True
BENCHMARK = "VWRL.L"

# Operating mode, fixed at process start. "paper" and "live" use SEPARATE databases, so no state
# (authorisation, kill switch, peaks, ledger) can carry over from paper to live. Live additionally
# requires PAPER_ONLY = False (code change) and a valid signed authorisation for one exact account.
MODE = os.environ.get("ATLAS_MODE", "paper").strip().lower()
if MODE not in ("paper", "live"):
    raise SystemExit(f"ATLAS_MODE must be 'paper' or 'live', not {MODE!r}")


def _default_db() -> str:
    if os.environ.get("ATLAS_DB"):
        return os.environ["ATLAS_DB"]
    if MODE == "live":
        (DATA_DIR / "live").mkdir(exist_ok=True)
        return str(DATA_DIR / "live" / "atlas-live.db")
    return str(DATA_DIR / "atlas.db")


@dataclass
class RiskLimits:
    """Hard limits enforced by the risk engine. The risk engine has final authority."""
    max_position_weight: float = 0.35      # default cap; per-instrument `max_w` in UNIVERSE overrides
    max_gross_exposure: float = 1.00       # long-only, no leverage
    max_order_fraction: float = 0.65       # fat-finger guard: single order <= 65% of equity
    max_daily_turnover: float = 1.00       # traded notional per day <= 100% equity
    daily_loss_limit: float = 0.04         # -4% day -> suspend risk-increasing trades
    max_drawdown: float = 0.20             # -20% from peak -> suspend, require human resume
    max_data_staleness_days: int = 5       # refuse to trade on stale prices
    min_order_gbp: float = 50.0            # ignore dust trades (commission drag)
    allow_short: bool = False
    allow_leverage: bool = False


@dataclass
class Settings:
    db_path: Path = field(default_factory=lambda: Path(_default_db()))
    secret_key: str = field(default_factory=lambda: os.environ.get("ATLAS_SECRET_KEY", ""))
    broker: str = field(default_factory=lambda: os.environ.get("ATLAS_BROKER", "paper"))
    paper_starting_cash: float = float(os.environ.get("ATLAS_PAPER_CASH", "100000"))
    # Trading cost model (IBKR UK tiered-ish for LSE ETFs; UK ETFs are stamp-duty exempt)
    commission_bps: float = 5.0
    commission_min_gbp: float = 3.0
    spread_bps: float = 5.0
    rebalance_band: float = 0.05           # sell only when a holding is >= 5pp overweight (experiment E1)
    cycle_seconds: int = int(os.environ.get("ATLAS_CYCLE_SECONDS", "3600"))
    notify_webhook: str = field(default_factory=lambda: os.environ.get("ATLAS_NOTIFY_WEBHOOK", ""))
    limits: RiskLimits = field(default_factory=RiskLimits)


settings = Settings()
