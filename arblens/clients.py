"""Read-only clients for the public Kalshi and Polymarket APIs. No keys, no orders."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import httpx

from .config import settings
from .models import Market, Rejected

log = logging.getLogger("arblens.clients")

HEADERS = {"User-Agent": "ArbLens/0.2 (read-only research tool)", "Accept": "application/json"}


def _f(v: Any) -> Optional[float]:
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def _dt(v: Any) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None


def _price(v: Optional[float]) -> Optional[float]:
    """A tradeable price is strictly between 0 and 1. 0 or 1 means no offers on that side."""
    return v if v is not None and 0 < v < 1 else None


async def get_json(client: httpx.AsyncClient, url: str, params: dict | None = None, tries: int = 3) -> Any:
    """GET with retry and backoff for rate limits and server errors."""
    delay = 0.5
    for attempt in range(tries):
        try:
            r = await client.get(url, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            retryable = status is None or status == 429 or status >= 500
            if attempt == tries - 1 or not retryable:
                raise
            await asyncio.sleep(delay)
            delay *= 2


# --------------------------------------------------------------------------- Kalshi

def parse_kalshi_market(m: dict, event: dict | None = None) -> Market:
    event = event or {}
    ev_title = (event.get("title") or "").strip()
    title = (m.get("title") or "").strip()
    sub = (m.get("yes_sub_title") or "").strip()
    base = ev_title or title
    # yes_sub_title says what YES means ("Washington", "Above 4.0%"). Keep it unless the title
    # already reads as a full yes/no question that names it.
    full_question = base.lower().startswith("will ") and sub.lower() in base.lower()
    display = f"{base} — {sub}" if sub and not full_question else base
    yes_bid, no_bid = _price(_f(m.get("yes_bid_dollars"))), _price(_f(m.get("no_bid_dollars")))
    yes_ask, no_ask = _price(_f(m.get("yes_ask_dollars"))), _price(_f(m.get("no_ask_dollars")))
    series = event.get("series_ticker") or m.get("series_ticker") or m.get("event_ticker", "").split("-")[0]
    rules = "\n\n".join(x for x in [m.get("rules_primary"), m.get("rules_secondary")] if x)
    # Close time: prefer expected expiration (Kalshi sets far-future close_time on long-dated markets).
    close = _dt(m.get("expected_expiration_time")) or _dt(m.get("close_time"))
    return Market(
        platform="kalshi",
        id=m["ticker"],
        title=display,
        match_text=f"{display} {title if title not in display else ''}",
        yes_ask=yes_ask, yes_bid=yes_bid, no_ask=no_ask, no_bid=no_bid,
        yes_ask_size=_f(m.get("yes_ask_size_fp")),
        # Buying NO on Kalshi fills against YES bids, so NO-ask depth = YES-bid size.
        no_ask_size=_f(m.get("no_ask_size_fp")) or _f(m.get("yes_bid_size_fp")),
        close_time=close,
        rules=rules,
        url=f"https://kalshi.com/markets/{series.lower()}" if series else "https://kalshi.com",
        fee_rate=settings.kalshi_base_fee_rate,
        category=event.get("category"),
        series=series,
        event=m.get("event_ticker"),
        subtitle=sub or None,
        volume_24h=_f(m.get("volume_24h_fp")),
        status=m.get("status", "active"),
    )


async def fetch_kalshi(client: httpx.AsyncClient) -> tuple[list[Market], list[Rejected]]:
    """Open Kalshi markets closing within the horizon. Long-dated markets tie up money for years,
    so they make poor arbs even when the gap is real."""
    markets: list[Market] = []
    rejected: list[Rejected] = []
    cursor: str | None = None
    max_close = int(time.time()) + settings.horizon_days * 86400
    for _ in range(settings.kalshi_pages):
        params = {"status": "open", "limit": 1000, "mve_filter": "exclude", "max_close_ts": max_close}
        if cursor:
            params["cursor"] = cursor
        data = await get_json(client, f"{settings.kalshi_base}/markets", params)
        for m in data.get("markets", []):
            if m.get("market_type", "binary") != "binary" or m.get("mve_collection_ticker"):
                continue  # skip multi-leg combo products
            try:
                mk = parse_kalshi_market(m)
            except Exception as e:  # noqa: BLE001 - keep the reason, keep going
                rejected.append(Rejected(platform="kalshi", id=str(m.get("ticker")), reason=str(e)[:200]))
                continue
            if mk.yes_ask is None and mk.no_ask is None:
                rejected.append(Rejected(platform="kalshi", id=mk.id, reason="no offers on either side"))
                continue
            markets.append(mk)
        cursor = data.get("cursor")
        if not cursor:
            break
    return markets, rejected


_event_cache: dict[str, dict] = {}


async def kalshi_event(client: httpx.AsyncClient, event_ticker: str) -> dict:
    """Event metadata (title, category). Cached for the life of the process: titles don't change."""
    if event_ticker not in _event_cache:
        ev = (await get_json(client, f"{settings.kalshi_base}/events/{event_ticker}", tries=4)).get("event", {})
        _event_cache[event_ticker] = {k: ev.get(k) for k in ("title", "sub_title", "category", "series_ticker")}
    return _event_cache[event_ticker]


def needs_event_title(m: Market) -> bool:
    """Kalshi market titles like 'X wins the award' only make sense with the event title."""
    return m.platform == "kalshi" and bool(m.event) and not m.title.lower().startswith("will ")


def with_event(m: Market, ev: dict) -> Market:
    """Rebuild a Kalshi market's display title using its event title."""
    title = (ev.get("title") or "").strip()
    if not title:
        return m
    sub = m.subtitle or ""
    m.title = f"{title} — {sub}" if sub and sub.lower() not in title.lower() else title
    m.match_text = m.title
    m.category = ev.get("category") or m.category
    return m


async def kalshi_market(client: httpx.AsyncClient, ticker: str) -> Market:
    data = await get_json(client, f"{settings.kalshi_base}/markets/{ticker}")
    m = data["market"]
    mk = parse_kalshi_market(m)
    if needs_event_title(mk):
        try:
            with_event(mk, await kalshi_event(client, mk.event or ""))
        except Exception:  # noqa: BLE001 - title is cosmetic here
            pass
    mk.status = m.get("status", "active")
    return mk


async def kalshi_result(client: httpx.AsyncClient, ticker: str) -> Optional[str]:
    """'yes' / 'no' once Kalshi has settled the market, else None."""
    m = (await get_json(client, f"{settings.kalshi_base}/markets/{ticker}"))["market"]
    if m.get("status") in ("finalized", "settled") and m.get("result") in ("yes", "no"):
        return m["result"]
    return None


_series_fee_cache: dict[str, float] = {}


async def kalshi_fee_rate(client: httpx.AsyncClient, series: str | None) -> float:
    """Base fee rate x the series' fee_multiplier. Cached; falls back to the base rate."""
    if not series:
        return settings.kalshi_base_fee_rate
    if series not in _series_fee_cache:
        try:
            s = (await get_json(client, f"{settings.kalshi_base}/series/{series}")).get("series", {})
            mult = _f(s.get("fee_multiplier"))
            _series_fee_cache[series] = settings.kalshi_base_fee_rate * (mult if mult is not None else 1)
        except Exception:  # noqa: BLE001
            return settings.kalshi_base_fee_rate
    return _series_fee_cache[series]


async def kalshi_book(client: httpx.AsyncClient, ticker: str) -> dict[str, list[tuple[float, float]]]:
    """Ask ladders (cheapest first) and bid ladders (highest first), as (price, size).

    Kalshi only publishes bids. Buying YES at p fills against a NO bid at 1-p, and vice versa.
    """
    ob = (await get_json(client, f"{settings.kalshi_base}/markets/{ticker}/orderbook")).get("orderbook_fp", {})
    yes_bids = sorted(((float(p), float(s)) for p, s in ob.get("yes_dollars") or []), reverse=True)
    no_bids = sorted(((float(p), float(s)) for p, s in ob.get("no_dollars") or []), reverse=True)
    yes_asks = sorted((round(1 - p, 4), s) for p, s in no_bids)
    no_asks = sorted((round(1 - p, 4), s) for p, s in yes_bids)
    return {"yes": yes_asks, "no": no_asks, "yes_bids": yes_bids, "no_bids": no_bids}


# --------------------------------------------------------------------------- Polymarket

def parse_poly_market(m: dict, event: dict | None = None) -> Market:
    outcomes = json.loads(m.get("outcomes") or "[]")
    if len(outcomes) != 2:
        raise ValueError(f"not a two-outcome market: {outcomes}")
    tokens = json.loads(m.get("clobTokenIds") or "[]")
    if len(tokens) != 2:
        raise ValueError("missing CLOB token ids")
    bid, ask = _f(m.get("bestBid")), _f(m.get("bestAsk"))
    ev = event or (m.get("events") or [{}])[0]
    q = (m.get("question") or "").strip()
    # Two-outcome markets that aren't Yes/No (e.g. "Wizards vs. Bulls" -> ["Wizards", "Bulls"]):
    # treat the first outcome as YES and say so in the title.
    if [o.lower() for o in outcomes] != ["yes", "no"]:
        q = f"{q} — {outcomes[0]} (vs {outcomes[1]})"
    fee = m.get("feeSchedule") or {}
    fee_rate = float(fee.get("rate", 0)) if m.get("feesEnabled") else 0.0
    return Market(
        platform="polymarket",
        id=str(m["id"]),
        title=q,
        match_text=f"{q} {m.get('groupItemTitle') or ''}",
        yes_ask=_price(ask),
        yes_bid=_price(bid),
        # Polymarket's YES and NO books mirror each other: NO ask = 1 - YES bid.
        no_ask=_price(round(1 - bid, 4)) if bid is not None else None,
        no_bid=_price(round(1 - ask, 4)) if ask is not None else None,
        close_time=_dt(m.get("endDate")),
        rules=(m.get("description") or "").strip(),
        url=f"https://polymarket.com/event/{ev.get('slug') or m.get('slug')}",
        fee_rate=fee_rate,
        category=(ev.get("category") or None),
        yes_token=tokens[0],
        no_token=tokens[1],
        volume_24h=_f(m.get("volume24hr")),
        status="closed" if m.get("closed") else "open",
    )


async def fetch_polymarket(client: httpx.AsyncClient) -> tuple[list[Market], list[Rejected]]:
    """Open Polymarket events (each holds many markets), busiest first.

    Gamma caps `offset` at 2000, so at most 20 pages of 100 events.
    """
    horizon = datetime.now(timezone.utc) + timedelta(days=settings.horizon_days)

    async def page(offset: int) -> list[dict]:
        params = {"active": "true", "closed": "false", "limit": 100, "offset": offset,
                  "order": "volume24hr", "ascending": "false"}
        try:
            return await get_json(client, f"{settings.gamma_base}/events", params)
        except Exception as e:  # noqa: BLE001
            log.warning("polymarket page %s failed: %s", offset, e)
            return []

    sem = asyncio.Semaphore(6)

    async def guarded(offset: int) -> list[dict]:
        async with sem:
            return await page(offset)

    pages = await asyncio.gather(*(guarded(i * 100) for i in range(min(settings.poly_pages, 20))))
    markets: list[Market] = []
    rejected: list[Rejected] = []
    seen: set[str] = set()
    for events in pages:
        for ev in events:
            for m in ev.get("markets") or []:
                mid = str(m.get("id"))
                if mid in seen:
                    continue
                seen.add(mid)
                if m.get("closed") or not m.get("enableOrderBook") or not m.get("acceptingOrders"):
                    continue
                try:
                    mk = parse_poly_market(m, ev)
                except Exception as e:  # noqa: BLE001
                    rejected.append(Rejected(platform="polymarket", id=mid, reason=str(e)[:200]))
                    continue
                if mk.close_time and mk.close_time > horizon:
                    continue
                if mk.yes_ask is None and mk.no_ask is None:
                    rejected.append(Rejected(platform="polymarket", id=mid, reason="no offers on either side"))
                    continue
                markets.append(mk)
    return markets, rejected


async def poly_market(client: httpx.AsyncClient, market_id: str) -> Market:
    m = await get_json(client, f"{settings.gamma_base}/markets/{market_id}")
    return parse_poly_market(m)


async def poly_result(client: httpx.AsyncClient, market_id: str) -> Optional[str]:
    """'yes' / 'no' once Polymarket has closed the market at 1/0, else None."""
    m = await get_json(client, f"{settings.gamma_base}/markets/{market_id}")
    if not m.get("closed"):
        return None
    prices = [_f(x) for x in json.loads(m.get("outcomePrices") or "[]")]
    if prices == [1.0, 0.0]:
        return "yes"
    if prices == [0.0, 1.0]:
        return "no"
    return None


async def poly_book(client: httpx.AsyncClient, token_id: str) -> list[tuple[float, float]]:
    """Ask ladder for one outcome token, cheapest first."""
    data = await get_json(client, f"{settings.clob_base}/book", {"token_id": token_id})
    return sorted((float(a["price"]), float(a["size"])) for a in data.get("asks") or [])


async def poly_bids(client: httpx.AsyncClient, token_id: str) -> list[tuple[float, float]]:
    """Bid ladder for one outcome token, best (highest) first."""
    data = await get_json(client, f"{settings.clob_base}/book", {"token_id": token_id})
    return sorted(((float(b["price"]), float(b["size"])) for b in data.get("bids") or []), reverse=True)
