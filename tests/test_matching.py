from datetime import datetime, timedelta, timezone

from arblens.matching import compare_numbers, match, numbers, tokens, word_diff
from arblens.models import Market

NOW = datetime.now(timezone.utc)


def _m(platform, id, title, close_days=30, yes=0.5):
    return Market(platform=platform, id=id, title=title, match_text=title,
                  yes_ask=yes + 0.01, yes_bid=yes - 0.01, no_ask=1 - yes + 0.01, no_bid=1 - yes - 0.01,
                  close_time=NOW + timedelta(days=close_days))


def test_numbers_normalised_and_dates_ignored():
    assert numbers("S&P 500 above 6,800 on Dec 31, 2026?") == ({"2026"}, {"500", "6800"})
    assert numbers("CPI above 3.0%") == (set(), {"3"})


def test_different_thresholds_rejected():
    ok, _ = compare_numbers("S&P 500 above 6,800?", "S&P 500 above 6,900?")
    assert not ok


def test_one_sided_number_warns():
    ok, warnings = compare_numbers("Will Sinner make the top 10?", "Will Sinner win?")
    assert ok and warnings


def test_disjoint_years_rejected():
    ok, _ = compare_numbers("Win the 2026 title?", "Win the 2027 title?")
    assert not ok


def test_plural_folding():
    assert tokens("rates") == tokens("rate")


def test_match_pairs_same_question():
    k = [_m("kalshi", "K1", "Will Japan win the 2027 AFC Asian Cup?"),
         _m("kalshi", "K2", "Will Iran win the 2027 AFC Asian Cup?")]
    p = [_m("polymarket", "P1", "Will Iran win the 2027 AFC Asian Cup?"),
         _m("polymarket", "P2", "Will Japan win the 2027 AFC Asian Cup?")]
    pairs = {x.key for x in match(k, p, 0.4)}
    assert pairs == {"K1~P2", "K2~P1"}


def test_match_flags_price_disagreement_and_date_gap():
    k = [_m("kalshi", "K1", "Will Japan win the 2027 AFC Asian Cup?", close_days=10, yes=0.2)]
    p = [_m("polymarket", "P1", "Will Japan win the 2027 AFC Asian Cup?", close_days=30, yes=0.7)]
    [pair] = match(k, p, 0.4)
    joined = " ".join(pair.warnings)
    assert "Prices disagree" in joined and "Closing dates differ" in joined


def test_match_drops_far_apart_dates():
    k = [_m("kalshi", "K1", "Will Japan win the 2027 AFC Asian Cup?", close_days=10)]
    p = [_m("polymarket", "P1", "Will Japan win the 2027 AFC Asian Cup?", close_days=200)]
    assert match(k, p, 0.4) == []


def test_word_diff_exposes_different_bets():
    d = word_diff("Will Lisa Cook visit the White House before 2027?", "Will Lisa Cook be arrested before 2027?")
    assert "visit" in d["kalshi"] and "arrested" in d["polymarket"]
