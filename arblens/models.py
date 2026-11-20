"""Validated shapes for everything that crosses a boundary (API in, JSON out)."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

Platform = Literal["kalshi", "polymarket"]
Direction = Literal["K_YES+P_NO", "K_NO+P_YES"]


class Market(BaseModel):
    """One binary YES/NO market, normalised so both platforms look the same.

    Prices are in dollars per contract (0-1). A contract pays $1 if its side wins.
    """

    platform: Platform
    id: str  # Kalshi ticker, or Polymarket market id
    title: str
    match_text: str
    yes_ask: Optional[float] = None  # cheapest price to BUY YES
    yes_bid: Optional[float] = None  # best price to SELL YES
    no_ask: Optional[float] = None
    no_bid: Optional[float] = None
    yes_ask_size: Optional[float] = None  # contracts available at yes_ask
    no_ask_size: Optional[float] = None
    close_time: Optional[datetime] = None
    rules: str = ""
    url: str = ""
    fee_rate: float = 0.0
    category: Optional[str] = None
    series: Optional[str] = None  # Kalshi series ticker (fee multiplier lives here)
    event: Optional[str] = None  # Kalshi event ticker
    subtitle: Optional[str] = None  # what YES means, e.g. "Washington" or "Above 4.0%"
    yes_token: Optional[str] = None  # Polymarket CLOB token ids
    no_token: Optional[str] = None
    volume_24h: Optional[float] = None
    status: str = "open"


class Rejected(BaseModel):
    """A row from an API that failed validation. Kept so nothing disappears silently."""

    platform: Platform
    id: str
    reason: str


class Leg(BaseModel):
    platform: Platform
    side: Literal["yes", "no"]
    price: float  # average fill price per contract
    fee: float
    size_available: Optional[float] = None


class Quote(BaseModel):
    """What it costs to lock in $1 per contract by buying opposite sides on each platform."""

    direction: Direction
    qty: float
    kalshi: Leg
    polymarket: Leg
    cost: float  # both prices x qty
    fees: float
    gross_per: float  # cents per contract before fees
    net: float  # dollars, if held to resolution and both platforms resolve the same way
    net_per: float  # cents per contract after fees
    roi: float  # percent on money put in
    annualized: Optional[float] = None
    max_qty: Optional[float] = None  # limited by the thinner side of the book


class Pair(BaseModel):
    key: str
    score: float
    kalshi: Market
    polymarket: Market
    best: Optional[Quote] = None
    quotes: list[Quote] = Field(default_factory=list)
    days_to_close: Optional[float] = None
    review: Optional[str] = None  # "confirmed" / "rejected" by you
    warnings: list[str] = Field(default_factory=list)  # reasons this text match might be a different bet
    word_diff: dict[str, list[str]] = Field(default_factory=dict)  # words only in one title


class TradeRequest(BaseModel):
    pair_key: str
    direction: Direction
    qty: float = Field(gt=0, le=100_000)


class ReviewRequest(BaseModel):
    verdict: Literal["confirmed", "rejected", "clear"]


class ResetRequest(BaseModel):
    starting_cash: float = Field(default=1000, gt=0, le=10_000_000)


# ---- AI rules check output (validated with Pydantic) ----

class RuleDifference(BaseModel):
    type: Literal["source", "deadline", "definition", "edge_case", "other"]
    quote_kalshi: str
    quote_polymarket: str
    why_it_matters: str
    verified: bool = False  # set in code, never by the model


class RulesVerdict(BaseModel):
    same_event: Literal["same", "different", "unsure"]
    differences: list[RuleDifference]
    risk: Literal["LOW", "MEDIUM", "HIGH"]
    reason: str
