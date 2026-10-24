# predictgar

Explore Iranian livestock and feed price history, compare forecasting models, and export predictions. Import a CSV or fetch available source data, then check a walk-forward backtest before using a forecast.

## Try it first

Open **Data & Charts** and upload a CSV with `date` and `price` columns. Train a small model in **Training**, inspect **Backtest**, then export from **Forecast**. The built-in product catalog contains names, not example price observations.

## Run locally

Verified existing environments: Python 3.13.5 / Gradio 5.49.1 and Python 3.10.19 / Gradio 6.14.0. Use a compatible existing environment; `requirements.txt` records the base dependency versions. BeautifulSoup is needed only for live HTML discovery, not CSV workflows or the offline product catalog.

```powershell
conda activate base  # Or use your existing compatible environment.
# Install dependencies if they are not already available in your environment.
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:7861. Startup uses the cached or built-in product catalog without network requests; **Refresh product list** explicitly checks the source. Catalog names are not price observations. The app binds to loopback and does not create a public sharing link.

## Workflows

- **Quick Predict:** fetch or reuse real cached prices, train candidates, and generate a forecast.
- **Data & Charts:** inspect product history or upload a CSV with `date` and `price` columns. Uploaded data is used by training; a failed download never substitutes synthetic prices.
- **Training:** choose lags, windows, detrending, models, feature selection, and voting ensembles. Save complete model packages locally.
- **Forecast:** forecast from the trained model and its matching history, with optional statistical blending and CSV export.
- **Backtest:** evaluate successive historical forecast windows without training on those future observations.
- **Analyze:** inspect relationships with available macroeconomic and related-product series.

Frequency means daily (`D`), completed Sunday-ending weeks (`W`), or completed calendar months (`M`). Horizons, lags, and windows are measured in those periods. Missing dates carry forward the last observed price. The latest incomplete weekly/monthly period is excluded.

Tables use a readable, keyboard-scrollable view by default. **Table display > Use interactive tables** restores sortable grids. Readable previews show at most 200 rows; interactive grids and available CSV downloads retain the full results. Switching views preserves the dataset.

## Forecasting and evaluation

Features use past observations. Missing history is never backfilled from the future. Feature imputation and scaling are fitted on training rows, and target scaling stays inside the estimator so saved-model predictions retain their units. Log detrending is inverted back to price units. Final future forecasts refit the selected estimator on available history and do not add cosmetic random noise.

The chronological validation segment is used for model comparison. Optional external-feature selection uses an inner training split. Reported comparison scores are **validation results**, not an untouched final test. Use the separate walk-forward backtest to assess recursive, multi-period performance. One-step validation and multi-period backtest errors are not directly interchangeable.

Forecast ranges are residual-based estimates with square-root horizon widening, not calibrated guarantees. Hybrid blending is a heuristic using historical errors; it does not guarantee improvement. Macro variables are carried forward for future periods, not independently forecast. Cached or forward-filled prices may be stale. Inspect source dates and compare against a last-price baseline.

Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, and voting are supported. XGBoost and LightGBM are optional, separately installed, and outside the tested base environment. Ordinary K-fold stacking is disabled because it is unsuitable for this temporal workflow.

## Data and model storage

Live sources are ITPNews and TGJU; availability and upstream formats can change. Data access uses timeouts and local caches. No API key is required by the current source adapters. Uploaded CSVs are limited to 50 MB.

`data/` contains local product caches, external series, saved models, and forecast exports. `report/` contains your existing reports. Both are excluded from new commits and preserved locally; data already published remains in older Git history. App startup does not delete model files.

Only load model packages you trust: joblib/pickle can execute code. New packages include forecasting metadata and dependency versions; legacy incomplete packages require retraining. Matching versions are required for import.

## Source layout

- `app.py`: Gradio interface and workspace workflows
- `data_pipeline.py`: source adapters, caching, temporal preprocessing, and features
- `model_pipeline.py`: training, forecasting, backtests, and persistence
- `tests/`: synthetic regression coverage without live market requests

## Checks

```powershell
# Install dependencies if they are not already available in your environment.
python -m pip install pytest==9.1.1
python -m pytest tests -q
python -m pip check
```

Tests cover future-data independence, price scaling, calendar alignment, log inversion, saved models, backtesting, and unavailable-source errors. The launcher supports the installed Gradio 5 and Gradio 6 styling APIs; see its [official migration guide](https://gradio.app/guides/gradio-6-migration-guide) for framework changes.

Current lightweight check (Windows): **18 passed** in each existing environment above. Tests include CSV upload, training, forecast and model restore, offline operation without BeautifulSoup, escaped table content, and bounded previews. No dependency installs, model downloads, live publishing, or heavy datasets were used.

Browser verification covered CSV import, all six tabs, table switching with retained data, and a 390-pixel layout. Default readable tables avoid the native grid's nested-control findings. Gradio 6 still reports accessibility findings in hidden tab scaffolding and empty plot placeholders; full screen-reader conformance is not claimed. Live market endpoints were not exercised.

No license file is included.
