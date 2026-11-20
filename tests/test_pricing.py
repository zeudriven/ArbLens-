import pytest

from arblens.models import Market
from arblens.pricing import (fill, kalshi_fee, kalshi_fee_on_ladder, max_profitable_qty, poly_fee,
                             quote_from_books, quote_top_of_book)


# Polymarket's published fee table (100 shares): Crypto 0.07 @ $0.50 -> $1.75, Sports 0.05 @ $0.30 -> $1.05
@pytest.mark.parametrize("rate,price,expected", [(0.07, 0.50, 1.75), (0.05, 0.30, 1.05), (0.04, 0.10, 0.36),
                                                 (0.07, 0.01, 0.0693), (0.0, 0.5, 0.0)])
def test_poly_fee_matches_published_table(rate, price, expected):
    assert poly_fee(100, price, rate) == pytest.approx(expected, abs=1e-5)


def test_poly_fee_is_symmetric():
    assert poly_fee(100, 0.3, 0.05) == poly_fee(100, 0.7, 0.05)


def test_kalshi_fee_rounds_up_to_cent():
    # 0.07 * 100 * 0.41 * 0.59 = 1.6933 -> $1.70
    assert kalshi_fee(100, 0.41, 0.07) == 1.70
    # exact cents stay put: 0.07 * 100 * 0.5 * 0.5 = 1.75
    assert kalshi_fee(100, 0.50, 0.07) == 1.75
    # tiny fees still round up to one cent
    assert kalshi_fee(1, 0.01, 0.07) == 0.01


def test_fill_walks_ladder():
    ladder = [(0.40, 10), (0.42, 10), (0.50, 100)]
    avg, got = fill(ladder, 15)
    assert got == 15
    assert avg == pytest.approx((0.40 * 10 + 0.42 * 5) / 15)
    avg, got = fill(ladder, 500)
    assert got == 120  # can't fill more than the book holds


def test_kalshi_fee_charged_per_level():
    ladder = [(0.40, 10), (0.42, 10)]
    assert kalshi_fee_on_ladder(ladder, 15, 0.07) == kalshi_fee(10, 0.40, 0.07) + kalshi_fee(5, 0.42, 0.07)


def _m(platform, **kw):
    base = dict(platform=platform, id="x", title="t", match_text="t")
    return Market(**{**base, **kw})


def test_top_of_book_picks_profitable_direction():
    k = _m("kalshi", yes_ask=0.41, no_ask=0.61, fee_rate=0.07)
    p = _m("polymarket", yes_ask=0.45, no_ask=0.56, fee_rate=0.0)
    q = quote_top_of_book(k, p, qty=100)
    best = q[0]
    assert best.direction == "K_YES+P_NO"
    assert best.gross_per == pytest.approx(3.0)
    # net = 100 - 97 - kalshi fee 1.70 - 0 = 1.30
    assert best.net == pytest.approx(1.30)
    assert best.net_per == pytest.approx(1.30)


def test_gap_that_fees_erase():
    k = _m("kalshi", yes_ask=0.49, fee_rate=0.07)
    p = _m("polymarket", no_ask=0.50, fee_rate=0.05)
    best = quote_top_of_book(k, p, qty=100)[0]
    assert best.gross_per == pytest.approx(1.0)
    assert best.net < 0  # 1.00 gross - 1.75 Kalshi fee - 1.25 Polymarket fee


def test_book_quote_is_capped_by_thinner_side():
    q = quote_from_books("K_YES+P_NO", 100, [(0.40, 30)], [(0.55, 1000)], 0.07, 0.0, days=10)
    assert q.qty == 30


def test_max_profitable_qty_stops_when_marginal_contract_loses():
    k = [(0.40, 50), (0.47, 50)]  # second level: 0.47 + 0.55 > 1, loses
    p = [(0.55, 200)]
    assert max_profitable_qty(k, p, 0.07, 0.0) == 50


def test_max_profitable_qty_zero_when_no_edge():
    assert max_profitable_qty([(0.50, 10)], [(0.51, 10)], 0.07, 0.0) == 0
