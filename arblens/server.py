"""FastAPI app: live scanner, pair detail, rules check and paper trading.

Run:  python -m uvicorn arblens.server:app --reload
Open: http://127.0.0.1:8000
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from . import clients, rules
from .config import ROOT, settings
from .matching import build_index, make_pair, match, pair_warnings
from .models import Pair, ResetRequest, ReviewRequest, TradeRequest
from .paper import Ledger
from .pricing import (days_until, fill, kalshi_fee_on_ladder, max_profitable_qty, poly_fee_on_ladder,
                      quote_from_books, quote_top_of_book)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("arblens")

app = FastAPI(title="ArbLens", version="0.2")
ledger = Ledger(settings.db_path, settings.starting_cash)

_scan: dict[str, Any] = {"at": 0.0, "pairs": {}, "meta": {}}
_scan_lock = asyncio.Lock()


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(limits=httpx.Limits(max_connections=20), follow_redirects=True)


# --------------------------------------------------------------------------- scanning

async def run_scan() -> None:
    t0 = time.time()
    errors: list[str] = []
    async with _client() as c:
        k_res, p_res = await asyncio.gather(clients.fetch_kalshi(c), clients.fetch_polymarket(c), return_exceptions=True)
        if isinstance(k_res, Exception):
            errors.append(f"Kalshi fetch failed: {k_res}")
            k_res = ([], [])
        if isinstance(p_res, Exception):
            errors.append(f"Polymarket fetch failed: {p_res}")
            p_res = ([], [])
        (kalshi, k_rej), (poly, p_rej) = k_res, p_res
        idx = build_index(kalshi, poly)
        pairs = match(kalshi, poly, settings.min_match_score, idx=idx)

        sem = asyncio.Semaphore(6)  # Kalshi's public tier allows roughly 20 reads per second

        # Pass 2: titles like "X wins the award" need the event title ("Heisman Trophy 2026").
        async def event(t: str) -> tuple[str, dict]:
            async with sem:
                try:
                    return t, await clients.kalshi_event(c, t)
                except Exception:  # noqa: BLE001
                    return t, {}

        # Each lookup is one rate-limited Kalshi call, so cap them per scan (cached ones are free).
        # Pairs that look profitable go first; the rest fill in over later scans.
        need = [p for p in pairs if clients.needs_event_title(p.kalshi) and p.kalshi.event]
        need.sort(key=lambda p: -(p.best.net_per if p.best else -999))
        uncached = [t for t in dict.fromkeys(p.kalshi.event for p in need) if t not in clients._event_cache]
        to_fetch = set(uncached[:settings.event_lookups_per_scan])
        wanted = {p.kalshi.event for p in need if p.kalshi.event in clients._event_cache} | to_fetch
        events = dict(await asyncio.gather(*(event(t) for t in wanted)))
        kept = []
        for p in pairs:
            if clients.needs_event_title(p.kalshi) and not events.get(p.kalshi.event or ""):
                p.warnings.append("Kalshi title lacks its event name (not loaded yet); check what YES means")
            if p.kalshi.event in events and events[p.kalshi.event]:
                clients.with_event(p.kalshi, events[p.kalshi.event])
                score = idx.similarity(p.kalshi.match_text, p.polymarket.match_text)
                ok, warnings = pair_warnings(p.kalshi, p.polymarket)
                if score < settings.min_match_score or not ok:
                    continue
                p = make_pair(p.kalshi, p.polymarket, score, warnings)
            kept.append(p)
        pairs = kept

    # The scanner prices every Kalshi leg at the standard 0.07 rate. Looking up each series' exact
    # multiplier took hundreds of calls (~10 s), so that happens only when you open a pair.
    reviews = ledger.reviews()
    for p in pairs:
        p.review = reviews.get(p.key)

    _scan.update(
        at=time.time(),
        pairs={p.key: p for p in pairs},
        meta={
            "kalshi_markets": len(kalshi), "polymarket_markets": len(poly),
            "rejected": len(k_rej) + len(p_rej),
            "rejected_sample": [r.model_dump() for r in (k_rej + p_rej)[:15]],
            "pairs": len(pairs), "seconds": round(time.time() - t0, 1), "errors": errors,
        },
    )
    log.info("scan: %s Kalshi, %s Polymarket, %s pairs in %.1fs", len(kalshi), len(poly), len(pairs), time.time() - t0)


async def _locked_scan() -> None:
    if _scan_lock.locked():
        return  # a scan is already running
    async with _scan_lock:
        try:
            await run_scan()
        except Exception as e:  # noqa: BLE001 - keep serving the last good scan
            log.exception("scan failed")
            _scan["meta"] = {**_scan.get("meta", {}), "errors": [f"Scan failed: {e}"]}


async def _await_scan() -> None:
    """Run a scan in this request, or wait for the one already running."""
    if _scan_lock.locked():
        async with _scan_lock:
            return
    await _locked_scan()


async def ensure_scan(force: bool = False) -> None:
    stale = force or time.time() - _scan["at"] > settings.scan_ttl_seconds
    if not _scan["at"] or (stale and settings.serverless):
        # No data yet, or on Vercel where background work is frozen: scan inside this request.
        await _await_scan()
    elif stale:
        # Long-running server: serve the last scan now, refresh in the background.
        asyncio.create_task(_locked_scan())


@app.on_event("startup")
async def _warm() -> None:
    if not settings.serverless:
        asyncio.create_task(_locked_scan())


def _summary(p: Pair) -> dict:
    """Just what the scanner table needs (Vercel caps responses at 4.5 MB).
    Full pair data loads on demand from /api/pair."""
    b = p.best
    return {
        "key": p.key, "score": p.score, "review": p.review, "warnings": p.warnings, "word_diff": p.word_diff,
        "days_to_close": round(p.days_to_close, 1) if p.days_to_close is not None else None,
        "kalshi": {"title": p.kalshi.title}, "polymarket": {"title": p.polymarket.title},
        "best": {"direction": b.direction, "gross_per": b.gross_per, "net_per": b.net_per,
                 "annualized": b.annualized, "max_qty": b.max_qty},
    }


@app.get("/api/scan")
async def scan(force: bool = False) -> dict:
    await ensure_scan(force)
    pairs = sorted((p for p in _scan["pairs"].values() if p.best), key=lambda p: p.best.net_per, reverse=True)
    return {"updated_at": _scan["at"], "ttl": settings.scan_ttl_seconds, "scanning": _scan_lock.locked(),
            "serverless": settings.serverless, **_scan["meta"], "pairs": [_summary(p) for p in pairs]}


# --------------------------------------------------------------------------- pair detail

def _split_key(key: str) -> tuple[str, str]:
    if "~" not in key:
        raise HTTPException(400, "Bad pair key")
    k, p = key.split("~", 1)
    return k, p


async def _live_pair(key: str) -> dict:
    """Fresh prices, full order books and fee rates for one pair."""
    ticker, pid = _split_key(key)
    async with _client() as c:
        try:
            k, p = await asyncio.gather(clients.kalshi_market(c, ticker), clients.poly_market(c, pid))
        except httpx.HTTPError as e:
            raise HTTPException(502, f"Could not load live market data: {e}") from e
        k.fee_rate = await clients.kalshi_fee_rate(c, k.series)
        kb, p_yes, p_no = await asyncio.gather(clients.kalshi_book(c, ticker),
                                               clients.poly_book(c, p.yes_token), clients.poly_book(c, p.no_token))
    return {"k": k, "p": p, "kb": kb, "p_yes": p_yes, "p_no": p_no}


def _ladders(live: dict, direction: str) -> tuple[list, list]:
    if direction == "K_YES+P_NO":
        return live["kb"]["yes"], live["p_no"]
    return live["kb"]["no"], live["p_yes"]


@app.get("/api/pair")
async def pair_detail(key: str, qty: float = Query(100, gt=0, le=100_000)) -> dict:
    live = await _live_pair(key)
    k, p = live["k"], live["p"]
    days = days_until(min([t for t in (k.close_time, p.close_time) if t], default=None))
    directions = []
    for d in ("K_YES+P_NO", "K_NO+P_YES"):
        kl, pl = _ladders(live, d)
        q = quote_from_books(d, qty, kl, pl, k.fee_rate, p.fee_rate, days)
        directions.append({
            "direction": d,
            "quote": q.model_dump() if q else None,
            "max_profitable_qty": max_profitable_qty(kl, pl, k.fee_rate, p.fee_rate),
            "kalshi_book": kl[:8], "polymarket_book": pl[:8],
        })
    directions.sort(key=lambda x: x["quote"]["net"] if x["quote"] else -1e9, reverse=True)
    cached = ledger.get_rule_check(key, rules.rules_hash(k.rules, p.rules))
    cached_pair = _scan["pairs"].get(key)
    return {
        "key": key,
        "score": cached_pair.score if cached_pair else None,
        "review": ledger.reviews().get(key),
        "kalshi": k.model_dump(mode="json"),
        "polymarket": p.model_dump(mode="json"),
        "days_to_close": days,
        "directions": directions,
        "top_of_book": [q.model_dump() for q in quote_top_of_book(k, p, qty)],
        "code_check": rules.code_check(k.rules, p.rules),
        "ai_check": cached,
        "ai_enabled": settings.ai_enabled,
    }


@app.post("/api/pair/review")
async def review(key: str, body: ReviewRequest) -> dict:
    _split_key(key)
    ledger.set_review(key, body.verdict)
    if key in _scan["pairs"]:
        _scan["pairs"][key].review = None if body.verdict == "clear" else body.verdict
    return {"ok": True}


@app.post("/api/pair/ai-check")
async def ai_check(key: str, force: bool = False) -> dict:
    if not settings.ai_enabled:
        raise HTTPException(400, "AI rules check is off. Add ANTHROPIC_API_KEY to .env and restart the server.")
    ticker, pid = _split_key(key)
    async with _client() as c:
        k, p = await asyncio.gather(clients.kalshi_market(c, ticker), clients.poly_market(c, pid))
    h = rules.rules_hash(k.rules, p.rules)
    if not force and (cached := ledger.get_rule_check(key, h)):
        return cached
    try:
        result = await asyncio.to_thread(rules.ai_check, k.title, p.title, k.rules, p.rules)
    except RuntimeError as e:
        raise HTTPException(502, str(e)) from e
    ledger.save_rule_check(key, h, result, settings.ai_model, rules.PROMPT_VERSION)
    return {**result, "model": settings.ai_model, "prompt_version": rules.PROMPT_VERSION, "cached": False}


# --------------------------------------------------------------------------- paper trading

@app.post("/api/paper/trade")
async def paper_trade(body: TradeRequest) -> dict:
    live = await _live_pair(body.pair_key)
    k, p = live["k"], live["p"]
    if k.status not in ("active", "open") or p.status != "open":
        raise HTTPException(400, "One of the markets is no longer open.")
    kl, pl = _ladders(live, body.direction)
    days = days_until(min([t for t in (k.close_time, p.close_time) if t], default=None))
    q = quote_from_books(body.direction, body.qty, kl, pl, k.fee_rate, p.fee_rate, days)
    if not q:
        raise HTTPException(400, "No liquidity on one side right now.")
    k_side, p_side = q.kalshi.side, q.polymarket.side
    try:
        pid = ledger.open_position(
            pair_key=body.pair_key, title=k.title, direction=body.direction, qty=q.qty,
            k_ticker=k.id, k_side=k_side, k_price=q.kalshi.price, k_fee=q.kalshi.fee,
            p_market=p.id, p_token=p.yes_token if p_side == "yes" else p.no_token, p_side=p_side,
            p_price=q.polymarket.price, p_fee=q.polymarket.fee)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {"position_id": pid, "filled_qty": q.qty, "requested_qty": body.qty, "quote": q.model_dump(),
            "partial": q.qty < body.qty}


async def _exit_value(c: httpx.AsyncClient, pos: dict) -> dict:
    """What you'd get selling both legs right now, walking the bid books, after taker fees."""
    kb, p_bids = await asyncio.gather(clients.kalshi_book(c, pos["k_ticker"]), clients.poly_bids(c, pos["p_token"]))
    k_ladder = kb["yes_bids"] if pos["k_side"] == "yes" else kb["no_bids"]
    qty = pos["qty"]
    k_rate = await clients.kalshi_fee_rate(c, pos["k_ticker"].split("-")[0])
    kp, k_got = fill(k_ladder, qty)
    pp, p_got = fill(p_bids, qty)
    p_rate = 0.0
    try:
        p_rate = (await clients.poly_market(c, pos["p_market"])).fee_rate
    except Exception:  # noqa: BLE001
        pass
    k_val = (kp or 0) * k_got - kalshi_fee_on_ladder(k_ladder, k_got, k_rate)
    p_val = (pp or 0) * p_got - poly_fee_on_ladder(p_bids, p_got, p_rate)
    return {"value": round(k_val + p_val, 4), "kalshi_bid": kp, "polymarket_bid": pp,
            "fully_sellable": k_got >= qty - 1e-9 and p_got >= qty - 1e-9}


@app.get("/api/paper")
async def paper() -> dict:
    acct = ledger.account()
    open_pos = ledger.positions("open")
    async with _client() as c:
        async def mark(pos: dict) -> dict:
            try:
                ev = await _exit_value(c, pos)
            except Exception as e:  # noqa: BLE001
                ev = {"value": None, "error": str(e)[:120]}
            return {**pos, "exit": ev, "hold_value": pos["qty"]}
        marked = await asyncio.gather(*(mark(p) for p in open_pos))
    closed = ledger.positions("closed") + ledger.positions("settled")
    realized = round(sum(p["pnl"] or 0 for p in closed), 4)
    invested = round(sum(p["outlay"] for p in open_pos), 4)
    exit_total = sum(m["exit"]["value"] or 0 for m in marked)
    hold_total = sum(m["hold_value"] for m in marked)
    return {
        "account": acct,
        "open": marked,
        "closed": sorted(closed, key=lambda p: p["closed_at"] or "", reverse=True),
        "realized_pnl": realized,
        "invested": invested,
        "equity_if_sold_now": round(acct["cash"] + exit_total, 4),
        "equity_if_held": round(acct["cash"] + hold_total, 4),
    }


@app.post("/api/paper/{pid}/close")
async def paper_close(pid: int) -> dict:
    pos = ledger.position(pid)
    if not pos or pos["status"] != "open":
        raise HTTPException(404, "Open position not found")
    async with _client() as c:
        ev = await _exit_value(c, pos)
    if not ev["fully_sellable"]:
        raise HTTPException(400, "Not enough bids to sell the whole position right now.")
    return ledger.close(pid, ev["value"], "closed", "Sold both legs at live bids")


@app.post("/api/paper/settle")
async def paper_settle() -> dict:
    """Check every open position for resolution on both platforms and pay out."""
    results = []
    async with _client() as c:
        for pos in ledger.positions("open"):
            try:
                kr, pr = await asyncio.gather(clients.kalshi_result(c, pos["k_ticker"]), clients.poly_result(c, pos["p_market"]))
            except Exception as e:  # noqa: BLE001
                results.append({"id": pos["id"], "status": "error", "detail": str(e)[:120]})
                continue
            if kr is None or pr is None:
                results.append({"id": pos["id"], "status": "pending", "kalshi": kr, "polymarket": pr})
                continue
            payout = pos["qty"] * ((pos["k_side"] == kr) + (pos["p_side"] == pr))
            note = f"Kalshi resolved {kr.upper()}, Polymarket resolved {pr.upper()}."
            if kr != pr:
                note += " The platforms disagreed: this is the rules risk."
            ledger.close(pos["id"], payout, "settled", note)
            results.append({"id": pos["id"], "status": "settled", "payout": payout, "note": note})
    return {"results": results}


@app.post("/api/paper/reset")
async def paper_reset(body: ResetRequest = Body(default_factory=ResetRequest)) -> dict:
    ledger.reset(body.starting_cash)
    return ledger.account()


# --------------------------------------------------------------------------- misc

@app.get("/api/health")
async def health() -> dict:
    """Pings each data source so you can see which API is up, how fast, and a sample market."""
    checks = {
        "kalshi": (f"{settings.kalshi_base}/markets", {"status": "open", "limit": 1, "mve_filter": "exclude"}),
        "polymarket_gamma": (f"{settings.gamma_base}/markets", {"active": "true", "closed": "false", "limit": 1}),
        "polymarket_clob": (f"{settings.clob_base}/time", None),
    }

    async def ping(c: httpx.AsyncClient, url: str, params: dict | None) -> dict:
        t = time.time()
        try:
            r = await c.get(url, params=params, headers=clients.HEADERS, timeout=10)
            out: dict[str, Any] = {"ok": r.is_success, "status": r.status_code, "ms": round((time.time() - t) * 1000)}
            if r.is_success and "markets" in url:
                body = r.json()
                m = (body.get("markets") or [{}])[0] if isinstance(body, dict) else (body or [{}])[0]
                out["sample"] = m.get("ticker") or m.get("question")
            return out
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)[:160], "ms": round((time.time() - t) * 1000)}

    async with _client() as c:
        results = await asyncio.gather(*(ping(c, u, p) for u, p in checks.values()))
    return {
        "ok": all(r["ok"] for r in results),
        "sources": dict(zip(checks, results)),
        "ai_enabled": settings.ai_enabled,
        "serverless": settings.serverless,
        "scan_age_seconds": round(time.time() - _scan["at"]) if _scan["at"] else None,
        "last_scan": {k: v for k, v in _scan.get("meta", {}).items() if k != "rejected_sample"},
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/prototype")
async def prototype() -> FileResponse:
    return FileResponse(ROOT / "prototype.html")
