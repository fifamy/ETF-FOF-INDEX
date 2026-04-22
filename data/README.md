# Data Notes

`universe.csv` is the candidate ETF universe used to select one representative instrument per asset bucket.

Selection logic in v1:

- Only rows with `enabled = 1` are considered.
- If market data is missing for a symbol, that symbol is ignored for the current run.
- The framework ranks each bucket using:
  - `priority`
  - `liquidity_score`
  - `size_score`
  - `fee_bps`
  - `tracking_error_bps`

Edit this file when you want to refresh the eligible ETF pool.

