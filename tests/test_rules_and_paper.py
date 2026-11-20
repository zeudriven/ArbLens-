import pytest

from arblens.models import RuleDifference, RulesVerdict
from arblens.paper import Ledger
from arblens.rules import code_check, verify_quotes

KALSHI = "Based on the CF Benchmarks Bitcoin Real Time Index at 4:00 PM ET on December 31, 2026."
POLY = "Resolves using the Binance BTC/USDT 1-minute candle close at 12:00 PM ET on December 31, 2026."


def test_code_check_finds_time_difference():
    flags = {f["type"]: f for f in code_check(KALSHI, POLY)["flags"]}
    assert "time" in flags
    assert "4:00 pm et" in flags["time"]["only_kalshi"]
    assert "12:00 pm et" in flags["time"]["only_polymarket"]


def test_code_check_reports_missing_rules():
    assert code_check(KALSHI, "")["missing_rules"] == ["Polymarket"]


def test_verify_quotes_catches_made_up_quote():
    v = RulesVerdict(same_event="same", risk="HIGH", reason="r", differences=[
        RuleDifference(type="source", quote_kalshi="CF Benchmarks Bitcoin Real Time Index",
                       quote_polymarket="Binance BTC/USDT 1-minute candle close", why_it_matters="x"),
        RuleDifference(type="deadline", quote_kalshi="at 5:00 PM ET",  # not in the text
                       quote_polymarket="(not addressed)", why_it_matters="x"),
    ])
    v, all_ok = verify_quotes(v, KALSHI, POLY)
    assert v.differences[0].verified
    assert not v.differences[1].verified
    assert not all_ok


@pytest.fixture
def ledger(tmp_path):
    return Ledger(tmp_path / "t.db", 1000)


def _open(ledger, qty=100, kp=0.41, pp=0.56):
    return ledger.open_position(pair_key="K~P", title="t", direction="K_YES+P_NO", qty=qty,
                                k_ticker="K", k_side="yes", k_price=kp, k_fee=1.70,
                                p_market="P", p_token="tok", p_side="no", p_price=pp, p_fee=0.0)


def test_open_deducts_cash_including_fees(ledger):
    _open(ledger)
    assert ledger.account()["cash"] == pytest.approx(1000 - 97 - 1.70)


def test_cannot_overspend(ledger):
    with pytest.raises(ValueError):
        _open(ledger, qty=5000)


def test_settle_same_outcome_pays_one_dollar_per_contract(ledger):
    pid = _open(ledger)
    pos = ledger.close(pid, 100.0, "settled", "both NO")
    assert pos["pnl"] == pytest.approx(100 - 98.70)
    assert ledger.account()["cash"] == pytest.approx(1000 + 100 - 98.70)


def test_settle_disagreement_can_lose_everything(ledger):
    pid = _open(ledger)
    pos = ledger.close(pid, 0.0, "settled", "Kalshi NO, Polymarket YES")
    assert pos["pnl"] == pytest.approx(-98.70)


def test_close_twice_fails(ledger):
    pid = _open(ledger)
    ledger.close(pid, 99.0, "closed", "sold")
    with pytest.raises(ValueError):
        ledger.close(pid, 99.0, "closed", "sold")


def test_reset_and_reviews(ledger):
    _open(ledger)
    ledger.set_review("K~P", "confirmed")
    assert ledger.reviews() == {"K~P": "confirmed"}
    ledger.reset(500)
    assert ledger.account()["cash"] == 500
    assert ledger.positions() == []
