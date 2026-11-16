"""SQLite persistence (WAL mode so the API and background worker can share it)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL, salt TEXT NOT NULL,
  created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS prices(
  symbol TEXT NOT NULL, date TEXT NOT NULL, close REAL, adjclose REAL NOT NULL, volume REAL,
  PRIMARY KEY(symbol, date));
CREATE TABLE IF NOT EXISTS strategy_runs(
  id INTEGER PRIMARY KEY, run_at TEXT NOT NULL, name TEXT NOT NULL, params TEXT NOT NULL,
  metrics TEXT NOT NULL, passed INTEGER NOT NULL, reasons TEXT NOT NULL, equity TEXT);
CREATE TABLE IF NOT EXISTS selection(
  id INTEGER PRIMARY KEY, decided_at TEXT NOT NULL, strategy TEXT NOT NULL, rationale TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS targets(
  id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, strategy TEXT NOT NULL, weights TEXT NOT NULL,
  asof TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS orders(
  id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, broker TEXT NOT NULL, broker_order_id TEXT,
  symbol TEXT NOT NULL, side TEXT NOT NULL, qty REAL NOT NULL, ref_price REAL, status TEXT NOT NULL,
  filled_qty REAL DEFAULT 0, avg_fill_price REAL, commission REAL DEFAULT 0, risk_decision TEXT,
  updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS paper_account(
  id INTEGER PRIMARY KEY CHECK(id=1), cash REAL NOT NULL);
CREATE TABLE IF NOT EXISTS paper_positions(symbol TEXT PRIMARY KEY, qty REAL NOT NULL, avg_cost REAL NOT NULL);
CREATE TABLE IF NOT EXISTS equity_snapshots(
  ts TEXT PRIMARY KEY, broker TEXT NOT NULL, equity REAL NOT NULL, cash REAL NOT NULL, positions TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, level TEXT NOT NULL, kind TEXT NOT NULL, message TEXT NOT NULL,
  data TEXT);
CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cash_flows(
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL, broker TEXT NOT NULL, amount REAL NOT NULL, note TEXT);
"""

_lock = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = Path(path or settings.db_path)
    conn = sqlite3.connect(p, timeout=30, check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    return conn


class DB:
    def __init__(self, path: Path | None = None):
        self.conn = connect(path)

    def q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with _lock:
            return self.conn.execute(sql, args).fetchall()

    def x(self, sql: str, args: tuple = ()) -> int:
        with _lock:
            cur = self.conn.execute(sql, args)
            return cur.lastrowid

    def many(self, sql: str, rows: list[tuple]) -> None:
        with _lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.executemany(sql, rows)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    # key/value state (kill switch, trading mode, heartbeat, peak equity ...)
    def get(self, k: str, default=None):
        r = self.q("SELECT v FROM kv WHERE k=?", (k,))
        return json.loads(r[0]["v"]) if r else default

    def set(self, k: str, v) -> None:
        self.x("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, json.dumps(v)))

    def event(self, level: str, kind: str, message: str, data: dict | None = None) -> None:
        self.x("INSERT INTO events(ts,level,kind,message,data) VALUES(?,?,?,?,?)",
               (now(), level, kind, message, json.dumps(data or {}, default=str)))
        if level in ("warn", "critical"):
            from .notify import notify
            notify(level, kind, message)
