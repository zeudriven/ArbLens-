"""Find likely same-question pairs across the two platforms.

TF-IDF cosine similarity on the titles, then hard guards:
  * the important numbers must agree (6,800 vs 6,900 is a different bet), and
  * the closing dates must be near each other.
A text match is only a candidate. You confirm it in the UI, and the rules check decides how risky it is.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import timedelta

from .models import Market, Pair
from .pricing import days_until, quote_top_of_book

STOP = set("""a an and are as at be before by for from has have he her his if in into is it its of on or
s she that the their there they this to was were will with would who what which when than then over
under after yes no market markets end ends price prices close closes""".split())

MONTHS = {"jan": "january", "feb": "february", "mar": "march", "apr": "april", "jun": "june", "jul": "july",
          "aug": "august", "sep": "september", "sept": "september", "oct": "october", "nov": "november",
          "dec": "december"}

_num = re.compile(r"\d+(?:\.\d+)?")
_month_day = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.? \d{1,2}(?:st|nd|rd|th)?\b", re.I)


def tokens(text: str) -> list[str]:
    t = text.lower().replace(",", "")
    t = re.sub(r"[^a-z0-9.%$ ]+", " ", t)
    out = []
    for w in t.split():
        w = w.strip(".")
        if not w or w in STOP:
            continue
        w = MONTHS.get(w, w)
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]  # crude plural folding: "rates" -> "rate"
        out.append(w)
    return out


def numbers(text: str) -> tuple[set[str], set[str]]:
    """(years, threshold numbers) in a title, normalised ("6,800" -> "6800", "3.0" -> "3").

    Day-of-month numbers ("Nov 28") are dropped: dates are compared separately via close times.
    """
    years, other = set(), set()
    text = _month_day.sub(" ", text.replace(",", ""))
    for n in _num.findall(text):
        v = n.rstrip("0").rstrip(".") if "." in n else n
        (years if re.fullmatch(r"20\d\d", v) else other).add(v)
    return years, other


def compare_numbers(a: str, b: str) -> tuple[bool, list[str]]:
    """(compatible, warnings). Different thresholds mean a different bet, so those are rejected."""
    ya, na = numbers(a)
    yb, nb = numbers(b)
    if na and nb and na != nb:
        return False, []
    if ya and yb and not (ya & yb):
        return False, []
    warnings = []
    if na != nb:
        only = sorted(na or nb)
        warnings.append(f"Only one title names the number(s) {', '.join(only)}")
    return True, warnings


def numbers_agree(a: str, b: str) -> bool:
    return compare_numbers(a, b)[0]


class Index:
    def __init__(self, docs: list[list[str]]):
        self.n = len(docs)
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log((1 + self.n) / (1 + c)) + 1 for t, c in df.items()}

    def vec(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        v = {t: c * self.idf.get(t, 1.0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}


    def similarity(self, a: str, b: str) -> float:
        va, vb = self.vec(tokens(a)), self.vec(tokens(b))
        return sum(w * vb.get(t, 0.0) for t, w in va.items())


def build_index(kalshi: list[Market], poly: list[Market]) -> Index:
    return Index([tokens(m.match_text) for m in kalshi] + [tokens(m.match_text) for m in poly])


def _mid(bid: float | None, ask: float | None) -> float | None:
    if bid is not None and ask is not None:
        return (bid + ask) / 2
    return bid if bid is not None else ask


def pair_warnings(k: Market, p: Market, max_days_apart: float = 60, warn_days_apart: float = 3) -> tuple[bool, list[str]]:
    """(still a candidate, warnings). Runs on every candidate and again after titles are enriched."""
    ok, warnings = compare_numbers(k.title, p.title)
    if not ok:
        return False, []
    if k.close_time and p.close_time:
        gap = abs(k.close_time - p.close_time)
        if gap > timedelta(days=max_days_apart):
            return False, []
        if gap > timedelta(days=warn_days_apart):
            warnings.append(f"Closing dates differ by {gap.days} days")
    km, pm = _mid(k.yes_bid, k.yes_ask), _mid(p.yes_bid, p.yes_ask)
    if km is not None and pm is not None and abs(km - pm) > 0.15:
        warnings.append(f"Prices disagree by {abs(km - pm) * 100:.0f}¢: likely a different question, or a thin book")
    return True, warnings


def word_diff(a: str, b: str) -> dict[str, list[str]]:
    """Meaningful words that appear in only one title. "visit" vs "arrested" is how you spot a fake pair."""
    ta, tb = set(tokens(a)), set(tokens(b))
    keep = lambda s: sorted(w for w in s if len(w) > 2 and not w.isdigit())  # noqa: E731
    return {"kalshi": keep(ta - tb)[:8], "polymarket": keep(tb - ta)[:8]}


def make_pair(k: Market, p: Market, score: float, warnings: list[str]) -> Pair:
    quotes = quote_top_of_book(k, p)
    return Pair(
        key=f"{k.id}~{p.id}", score=round(score, 3), kalshi=k, polymarket=p,
        best=quotes[0] if quotes else None, quotes=quotes, warnings=warnings,
        days_to_close=days_until(min([t for t in (k.close_time, p.close_time) if t], default=None)),
        word_diff=word_diff(k.title, p.title),
    )


def match(kalshi: list[Market], poly: list[Market], min_score: float = 0.45,
          max_days_apart: float = 60, warn_days_apart: float = 3, idx: Index | None = None) -> list[Pair]:
    k_toks = [tokens(m.match_text) for m in kalshi]
    p_toks = [tokens(m.match_text) for m in poly]
    idx = idx or Index(k_toks + p_toks)
    p_vecs = [idx.vec(t) for t in p_toks]
    inverted: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for j, v in enumerate(p_vecs):
        for t, w in v.items():
            inverted[t].append((j, w))
    # Words like "2026" or "vs" appear in thousands of titles. Using them to find candidates makes
    # the search quadratic, so skip them here; they still count in the exact score below.
    # Measured on a live scan: cap 1600 -> 13.4 s, cap 400 -> 4.6 s with 8 of 1,415 pairs lost.
    common_cap = max(300, len(poly) // 80)

    candidates: list[tuple[float, int, int, list[str]]] = []
    for i, toks in enumerate(k_toks):
        kv = idx.vec(toks)
        partial: dict[int, float] = defaultdict(float)
        for t, w in kv.items():
            posting = inverted.get(t, ())
            if len(posting) > common_cap:
                continue
            for j, pw in posting:
                partial[j] += w * pw
        top = sorted(partial.items(), key=lambda x: -x[1])[:10]
        exact = [(j, sum(w * p_vecs[j].get(t, 0.0) for t, w in kv.items())) for j, _ in top]
        for j, s in sorted(exact, key=lambda x: -x[1])[:5]:
            if s < min_score:
                break
            ok, warnings = pair_warnings(kalshi[i], poly[j], max_days_apart, warn_days_apart)
            if ok:
                candidates.append((s, i, j, warnings))

    # One-to-one: best scores claim their partners first.
    pairs: list[Pair] = []
    used_k, used_p = set(), set()
    for s, i, j, warnings in sorted(candidates, key=lambda c: -c[0]):
        if i in used_k or j in used_p:
            continue
        used_k.add(i)
        used_p.add(j)
        pairs.append(make_pair(kalshi[i], poly[j], s, warnings))
    return pairs
