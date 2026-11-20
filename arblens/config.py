"""Settings, read once from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader so no extra package is needed. Real env vars win."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    kalshi_base: str = "https://api.elections.kalshi.com/trade-api/v2"
    gamma_base: str = "https://gamma-api.polymarket.com"
    clob_base: str = "https://clob.polymarket.com"

    # How much of each catalogue to scan. Kalshi: 1000 markets per page. Polymarket: 100 events per page (max 20).
    kalshi_pages: int = int(os.getenv("ARBLENS_KALSHI_PAGES", "40"))
    poly_pages: int = int(os.getenv("ARBLENS_POLY_PAGES", "20"))
    # Only markets that close within this many days. Money locked up for years is a poor arb.
    horizon_days: int = int(os.getenv("ARBLENS_HORIZON_DAYS", "180"))

    # A scan older than this is refreshed on the next request.
    scan_ttl_seconds: int = int(os.getenv("ARBLENS_SCAN_TTL", "90"))
    # Minimum text-similarity score (0-1) for two markets to be shown as a candidate pair.
    min_match_score: float = float(os.getenv("ARBLENS_MIN_MATCH", "0.45"))

    # Kalshi's published taker fee: ceil(0.07 x C x P x (1-P)), scaled by each series' fee_multiplier.
    kalshi_base_fee_rate: float = float(os.getenv("ARBLENS_KALSHI_FEE_RATE", "0.07"))

    # On Vercel only /tmp is writable, and it is wiped between cold starts, so paper trades there
    # are temporary until a hosted database (e.g. Supabase Postgres) replaces SQLite.
    db_path: Path = Path(os.getenv("ARBLENS_DB") or (
        "/tmp/arblens.db" if os.getenv("VERCEL") else str(ROOT / "data" / "arblens.db")))
    starting_cash: float = float(os.getenv("ARBLENS_STARTING_CASH", "1000"))

    # Optional AI rules check. Leave ANTHROPIC_API_KEY unset to run without it.
    ai_model: str = os.getenv("ARBLENS_AI_MODEL", "claude-opus-5-5")
    ai_enabled: bool = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


settings = Settings()
