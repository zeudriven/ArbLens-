"""All money math lives here, in plain testable code. The AI never touches a number."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Optional

from .models import Direction, Leg, Market, Quote

Ladder = list[tuple[float, float]]  # (price, size), cheapest first


def kalshi_fee(contracts: float, price: float, rate: float) -> float:
    """Kalshi taker fee: rate x C x P x (1-P), rounded UP to the next cent."""
    raw = rate * contracts * price * (1 - price)
    return math.ceil(round(raw * 100, 6)) / 100


def poly_fee(contracts: float, price: float, rate: float) -> float:
    """Polymarket taker fee: C x rate x p x (1-p), rounded to 5 decimals (their published precision)."""
    return round(contracts * rate * price * (1 - price), 5)


def fill(ladder: Ladder, qty: float) -> tuple[Optional[float], float]:
    """Walk an ask ladder. Returns (average price, quantity actually fillable)."""
    remaining, spent, got = qty, 0.0, 0.0
    for price, size in ladder:
        take = min(size, remaining)
        spent += take * price
        got += take
        remaining -= take
        if remaining <= 1e-9:
            break
    return (spent / got if got else None), got


def kalshi_fee_on_ladder(ladder: Ladder, qty: float, rate: float) -> float:
    """Kalshi charges per price level filled, so sum the fee level by level."""
    remaining, fee = qty, 0.0
    for price, size in ladder:
        take = min(size, remaining)
        if take > 0:
            fee += kalshi_fee(take, price, rate)
        remaining -= take
        if remaining <= 1e-9:
            break
    return round(fee, 2)


def poly_fee_on_ladder(ladder: Ladder, qty: float, rate: float) -> float:
    remaining, fee = qty, 0.0
    for price, size in ladder:
        take = min(size, remaining)
        if take > 0:
            fee += poly_fee(take, price, rate)
        remaining -= take
        if remaining <= 1e-9:
            break
    return round(fee, 5)


def days_until(when: Optional[datetime], now: Optional[datetime] = None) -> Optional[float]:
    if when is None:
        return None
    now = now or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - now).total_seconds() / 86400)


def _build(direction: Direction, qty: float, k_price: float, p_price: float, k_fee: float, p_fee: float,
           k_side: str, p_side: str, k_size: Optional[float], p_size: Optional[float],
           days: Optional[float]) -> Quote:
    cost = (k_price + p_price) * qty
    fees = k_fee + p_fee
    net = qty - cost - fees
    outlay = cost + fees
    roi = net / outlay * 100 if outlay > 0 else 0.0
    sizes = [s for s in (k_size, p_size) if s is not None]
    return Quote(
        direction=direction,
        qty=qty,
        kalshi=Leg(platform="kalshi", side=k_side, price=k_price, fee=k_fee, size_available=k_size),
        polymarket=Leg(platform="polymarket", side=p_side, price=p_price, fee=p_fee, size_available=p_size),
        cost=round(cost, 4),
        fees=round(fees, 4),
        gross_per=round((1 - k_price - p_price) * 100, 3),
        net=round(net, 4),
        net_per=round(net / qty * 100, 3) if qty else 0.0,
        roi=round(roi, 3),
        annualized=round(roi * 365 / max(days, 1), 2) if days is not None else None,
        max_qty=min(sizes) if sizes else None,
    )


def quote_top_of_book(k: Market, p: Market, qty: float = 100, k_fee_rate: Optional[float] = None) -> list[Quote]:
    """Both directions using only best prices. Fast; used for the scanner."""
    rate = k.fee_rate if k_fee_rate is None else k_fee_rate
    days = days_until(min([t for t in (k.close_time, p.close_time) if t], default=None))
    out = []
    combos = [("K_YES+P_NO", k.yes_ask, p.no_ask, "yes", "no", k.yes_ask_size, p.no_ask_size),
              ("K_NO+P_YES", k.no_ask, p.yes_ask, "no", "yes", k.no_ask_size, p.yes_ask_size)]
    for direction, kp, pp, ks, ps, kz, pz in combos:
        if kp is None or pp is None:
            continue
        out.append(_build(direction, qty, kp, pp, kalshi_fee(qty, kp, rate), poly_fee(qty, pp, p.fee_rate),
                          ks, ps, kz, pz, days))
    return sorted(out, key=lambda q: q.net, reverse=True)


def quote_from_books(direction: Direction, qty: float, k_ladder: Ladder, p_ladder: Ladder,
                     k_rate: float, p_rate: float, days: Optional[float]) -> Optional[Quote]:
    """Realistic quote: walk both order books for `qty` contracts.

    If either book is too thin, quantity is cut to what both sides can actually fill.
    """
    k_side, p_side = ("yes", "no") if direction == "K_YES+P_NO" else ("no", "yes")
    k_depth = sum(s for _, s in k_ladder)
    p_depth = sum(s for _, s in p_ladder)
    q = min(qty, k_depth, p_depth)
    if q <= 0:
        return None
    kp, _ = fill(k_ladder, q)
    pp, _ = fill(p_ladder, q)
    if kp is None or pp is None:
        return None
    return _build(direction, q, kp, pp, kalshi_fee_on_ladder(k_ladder, q, k_rate),
                  poly_fee_on_ladder(p_ladder, q, p_rate), k_side, p_side, k_depth, p_depth, days)


def max_profitable_qty(k_ladder: Ladder, p_ladder: Ladder, k_rate: float, p_rate: float) -> float:
    """How many contracts to buy before the NEXT contract would lose money after fees.

    Walks both ask ladders together. Prices only get worse as you go deeper, so stop at the first
    level pair where 1 - price_k - price_p - fees_per_contract <= 0.
    """
    i = j = 0
    k_rem = k_ladder[0][1] if k_ladder else 0
    p_rem = p_ladder[0][1] if p_ladder else 0
    qty = 0.0
    while i < len(k_ladder) and j < len(p_ladder):
        kp, pp = k_ladder[i][0], p_ladder[j][0]
        marginal = 1 - kp - pp - k_rate * kp * (1 - kp) - p_rate * pp * (1 - pp)
        if marginal <= 0:
            break
        step = min(k_rem, p_rem)
        qty += step
        k_rem -= step
        p_rem -= step
        if k_rem <= 1e-9:
            i += 1
            k_rem = k_ladder[i][1] if i < len(k_ladder) else 0
        if p_rem <= 1e-9:
            j += 1
            p_rem = p_ladder[j][1] if j < len(p_ladder) else 0
    return float(math.floor(qty))
