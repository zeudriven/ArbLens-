# ArbLens

Finds the same event listed on Kalshi and Polymarket, prices the "buy both sides" trade after each platform's real fees, and flags anything suggesting the two markets aren't actually the same bet. Includes a paper-trading account (fake money, live prices).

**Read-only.** No trading keys, no wallets, no orders. Analysis only, not financial advice. Polymarket is not available to Ontario residents.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env          # optional: add ANTHROPIC_API_KEY for the AI rules check
python -m uvicorn arblens.server:app --reload
```

Open http://127.0.0.1:8000. The first scan takes 30 to 40 seconds.

## Test

```bash
python -m pytest -q
```

## Layout

| File | Job |
|---|---|
| `arblens/clients.py` | Public Kalshi + Polymarket API clients (markets, order books, results) |
| `arblens/matching.py` | Pairs markets by title (TF-IDF) with number, date and price guards |
| `arblens/pricing.py` | Fee formulas, order-book fills, net edge, max profitable size |
| `arblens/rules.py` | Code diff of the two rulebooks + optional Claude check with quote verification |
| `arblens/paper.py` | SQLite paper-trading ledger |
| `arblens/server.py` | FastAPI routes |
| `index.html` | The live app |
| `prototype.html` | The original static prototype |
