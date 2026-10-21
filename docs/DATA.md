# Data contract and provenance

No market data, factor values, per-stock predictions, vendor exports or trained weights are distributed. Obtain your own data license. The MIT license covers the repository's original code; it grants no rights over vendor data or brokerage reports.

The research used locally supplied vendor exports: daily backward-adjusted OHLC and supplied average price/VWAP, **observed share volume**, amount, shares outstanding, market capitalization, index returns, historical membership changes, industry implementation records and suspension records. Membership source folders were labelled `CSI300_CHANGE_WIND` and equivalent CSI500 exports. We have not independently verified a single vendor attribution for every supplied file; do not describe every input as Wind or CSMAR without checking the original license and metadata.

## Private directory

```
private_data/
  data/csi300/market.parquet
  data/csi300/index.parquet
  data/csi500/market.parquet
  data/csi500/index.parquet
  factor_store/processed/universe=csi300/year=2024/part-0.parquet
  factor_store/processed/universe=csi500/year=2024/part-0.parquet
  factor_store/_factor_catalog.json
```

`market.parquet`: unique `(date, symbol)`; `symbol` is a zero-padded string. Include nonmember observations for trailing histories. Required factor fields: `open_hfq, high_hfq, low_hfq, close_hfq, vwap_hfq, volume, amount, return_1d, turnover_total, market_cap, is_member, factor_input_valid, history_valid_days, return_quality`. Verified adjusted-price rows use `return_quality='ok_hfq'`; `history_valid_days` is the consecutive valid-history count. Volume and amount are not assumed adjusted merely because the download is labelled backward-adjusted. Never form VWAP from incompatible price/volume units.

`index.parquet`: unique trading `date`, decimal `index_ret`. Report formulas that use the benchmark also need `open` and `close` index levels. Do not fabricate missing index opens. Historical return data and new-period extensions must use an audited consistent bridge; preserve the original historical series.

Industry records supplied to `build_factors.py`: `Symbol, ImplementDate, IndustryClassificationID, IndustryCode`. CSI neutralization uses the date-dependent scheme in `src/preprocessing.py`; pre-2012 coverage uses the documented CSRC fallback. Backward as-of joins only; never fill industry from a future record. Effective dates are an availability assumption, not independently verified publication timestamps.

`formula_industry` in the market input is separate from neutralization: internal report-formula groups use P0212 (numeric code after removing `ZX`, first four characters) before 2019-12-01 and P0217 (first two digits plus 100) thereafter. Unknown group = 0. Supply these from past-effective records; the code does not invent them.

Processed factors contain metadata plus factor columns; daily members only. Missing factors stay missing until train-fold median imputation inside the model. Signals with fewer than 50 of the selected 100 factors are excluded from inference, regardless of future labels. Preserve the factor catalog order when replicating tie-breaking.

## Membership, prices and corporate events

Use a verified prior constituent list and apply dated additions/removals on their effective session. Check each trading day's membership count and uniqueness. An old constituent must not disappear from the price history simply because it leaves the index. Suspensions and missing prices are not the same as ST status.

The local extension includes a 300114→302132 identity/adjustment-scale bridge. Alias resolution must be date-specific, with duplicates removed only after checking the valid identity. Keep original provider prices and bridge audit separately. Missing pre-entry history remains NaN; do not use future prices. Do not apply a static code rename or a guessed adjustment coefficient to a new vendor dataset.

## Reusing the original local project

`scripts/prepare_local_inputs.py --source ORIGINAL_PROJECT --out PRIVATE_DIRECTORY` assembles exactly the corrected processed input columns used by the reported models, preserving old history and normalized new-period identities. It is a local-layout adapter, not a vendor downloader, and expects the original audited project files. Its processed output is ready for prediction; rebuilding factors from scratch additionally needs the field/industry contract above.

The single-factor historical engine retains filenames `data/{universe}/parquet/{universe}_daily_market_2010_2024.parquet` and the analogous index file. Extra execution fields are `market_data_status, source_transition, vwap_source, provider_trade_status, provider_is_st`, plus the price/volume fields above. The local adapter copies these private historical files. It does not package them for publication.
