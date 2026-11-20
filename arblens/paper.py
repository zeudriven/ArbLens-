"""Paper-trading ledger in SQLite. Fake money, real prices.

A position is always a PAIR of legs: one side on Kalshi, the opposite side on Polymarket.
If both platforms resolve the event the same way, exactly one leg pays $1 per contract.
If their rules differ and they resolve differently, you get $0 or $2. That's the risk this tool measures.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS account (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  cash REAL NOT NULL,
  starting_cash REAL NOT NULL,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  pair_key TEXT NOT NULL,
  title TEXT NOT NULL,
  direction TEXT NOT NULL,
  qty REAL NOT NULL,
  k_ticker TEXT NOT NULL, k_side TEXT NOT NULL, k_price REAL NOT NULL, k_fee REAL NOT NULL,
  p_market TEXT NOT NULL, p_token TEXT NOT NULL, p_side TEXT NOT NULL, p_price REAL NOT NULL, p_fee REAL NOT NULL,
  outlay REAL NOT NULL,
  opened_at TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open',   -- open | closed (sold early) | settled (resolved)
  closed_at TEXT,
  proceeds REAL,
  pnl REAL,
  note TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
  pair_key TEXT PRIMARY KEY,
  verdict TEXT NOT NULL,
  at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rule_checks (
  pair_key TEXT NOT NULL,
  rules_hash TEXT NOT NULL,
  result TEXT NOT NULL,
  model TEXT,
  prompt_version TEXT,
  at TEXT NOT NULL,
  PRIMARY KEY (pair_key, rules_hash)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path, starting_cash: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.db() as c:
            c.executescript(SCHEMA)
            if not c.execute("SELECT 1 FROM account").fetchone():
                c.execute("INSERT INTO account VALUES (1, ?, ?, ?)", (starting_cash, starting_cash, now()))

    @contextmanager
    def db(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ---- account
    def account(self) -> dict:
        with self.db() as c:
            return dict(c.execute("SELECT cash, starting_cash, created_at FROM account").fetchone())

    def reset(self, starting_cash: float) -> None:
        with self.db() as c:
            c.execute("DELETE FROM positions")
            c.execute("UPDATE account SET cash = ?, starting_cash = ?, created_at = ?", (starting_cash, starting_cash, now()))

    # ---- positions
    def open_position(self, *, pair_key: str, title: str, direction: str, qty: float,
                      k_ticker: str, k_side: str, k_price: float, k_fee: float,
                      p_market: str, p_token: str, p_side: str, p_price: float, p_fee: float) -> int:
        outlay = round((k_price + p_price) * qty + k_fee + p_fee, 5)
        with self.db() as c:
            cash = c.execute("SELECT cash FROM account").fetchone()["cash"]
            if outlay > cash + 1e-9:
                raise ValueError(f"Not enough paper cash: need ${outlay:,.2f}, have ${cash:,.2f}")
            c.execute("UPDATE account SET cash = cash - ?", (outlay,))
            cur = c.execute(
                """INSERT INTO positions (pair_key, title, direction, qty, k_ticker, k_side, k_price, k_fee,
                   p_market, p_token, p_side, p_price, p_fee, outlay, opened_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (pair_key, title, direction, qty, k_ticker, k_side, k_price, k_fee,
                 p_market, p_token, p_side, p_price, p_fee, outlay, now()))
            return int(cur.lastrowid)

    def positions(self, status: Optional[str] = None) -> list[dict]:
        with self.db() as c:
            q = "SELECT * FROM positions" + (" WHERE status = ?" if status else "") + " ORDER BY id DESC"
            return [dict(r) for r in c.execute(q, (status,) if status else ())]

    def position(self, pid: int) -> Optional[dict]:
        with self.db() as c:
            r = c.execute("SELECT * FROM positions WHERE id = ?", (pid,)).fetchone()
            return dict(r) if r else None

    def close(self, pid: int, proceeds: float, status: str, note: str) -> dict:
        with self.db() as c:
            p = c.execute("SELECT * FROM positions WHERE id = ? AND status = 'open'", (pid,)).fetchone()
            if not p:
                raise ValueError("Position not found or already closed")
            pnl = round(proceeds - p["outlay"], 5)
            c.execute("UPDATE positions SET status=?, closed_at=?, proceeds=?, pnl=?, note=? WHERE id=?",
                      (status, now(), round(proceeds, 5), pnl, note, pid))
            c.execute("UPDATE account SET cash = cash + ?", (proceeds,))
        return self.position(pid) or {}

    # ---- match reviews
    def reviews(self) -> dict[str, str]:
        with self.db() as c:
            return {r["pair_key"]: r["verdict"] for r in c.execute("SELECT * FROM reviews")}

    def set_review(self, pair_key: str, verdict: str) -> None:
        with self.db() as c:
            if verdict == "clear":
                c.execute("DELETE FROM reviews WHERE pair_key = ?", (pair_key,))
            else:
                c.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?)", (pair_key, verdict, now()))

    # ---- cached rule checks
    def get_rule_check(self, pair_key: str, rules_hash: str) -> Optional[dict]:
        with self.db() as c:
            r = c.execute("SELECT * FROM rule_checks WHERE pair_key=? AND rules_hash=?", (pair_key, rules_hash)).fetchone()
            return {**json.loads(r["result"]), "model": r["model"], "prompt_version": r["prompt_version"],
                    "checked_at": r["at"], "cached": True} if r else None

    def save_rule_check(self, pair_key: str, rules_hash: str, result: dict, model: str, prompt_version: str) -> None:
        with self.db() as c:
            c.execute("INSERT OR REPLACE INTO rule_checks VALUES (?,?,?,?,?,?)",
                      (pair_key, rules_hash, json.dumps(result), model, prompt_version, now()))
