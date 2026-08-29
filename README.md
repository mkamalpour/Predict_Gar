# ITPNews Price Intelligence v13

Gradio app for Iranian livestock/feed commodity price forecasting. Fetches live data from ITPNews, enriches with macro indicators (USD/IRR, gold, CPI), trains ML models, blends with statistical methods, and forecasts with confidence intervals.

## Quick Start

```
pip install gradio scikit-learn pandas numpy requests beautifulsoup4 plotly scipy jdatetime
pip install xgboost lightgbm joblib   # optional
python app.py                         # opens at http://127.0.0.1:7861
```

## Architecture

Three files with strict import isolation:

| File | Responsibility | Imports |
|---|---|---|
| `data_pipeline.py` | Web scraping, caching, preprocessing, feature engineering, detrending | requests, bs4, jdatetime, pandas, numpy |
| `model_pipeline.py` | ML training, forecasting, hybrid blending, backtesting, reports | sklearn, xgboost, lightgbm, plotly, scipy, pandas, numpy |
| `app.py` | Gradio UI (5 tabs) | gradio, plotly + both pipelines |

## Prediction Pipeline (end-to-end)

```
ITPNews AJAX API
     |
     v
[1] FETCH & CACHE
    Fetcher.discover_products() -> {name: id}
    Fetcher.fetch_chart(pid) -> DataFrame
    Cache stores only date + price (incremental updates)
     |
     v
[2] PREPROCESS
    preprocess(): ffill().bfill().interpolate() (no row loss)
    -> IQR clip outliers -> add is_holiday (Thu/Fri + 10 Iranian solar holidays)
     |
     v
[3] ENRICH (per product profile)
    merge_external_features(): USD/IRR, gold, CPI via merge_asof
    fetch_related_prices(): feed prices (corn, soy, barley, wheat)
    Smart External Select: tests each feature, keeps only MAE-improving ones (>5%)
     |
     v
[4] DETREND (4 branches)
    linear/quad: target - polyfit(trend)
    smooth_diff: rolling(7).mean().diff()
    diff: target.diff()
    none: pass-through
     |
     v
[5] FEATURE ENGINEERING (zero data leakage)
    All features shift(1):
    - Lags: lag_1, lag_7, lag_14, lag_30
    - Rolling: ma_7/14/30, std_7/14/30
    - Momentum: roc, position, cv, oscillator
    - Seasonal: month/dow sin/cos, is_holiday
    - Cross-product: ratio, margin, corr with other products
    - External: ext_usd_irr_free, ext_gold_price, ext_cpi_inflation (+ rolling)
     |
     v
[6] TRAIN (single canonical split)
    compute_split_date() -> 80/20 train/test by date
    StandardScaler on features (fit train only)
    Target standardized: (y - mean) / std
    Models: Ridge, Lasso, ElasticNet, Random Forest (+ optional XGBoost, LightGBM)
    Filter: R-squared > 0, MAE < naive baseline
    Optional: Voting/Stacking ensemble
     |
     v
[7] FORECAST (recursive)
    iterative_forecast(): for each future day:
    - Build lag features from previous predictions
    - model.predict() -> detrended forecast
    - Undetrend (cumsum for diff, +trend for linear)
    - Add AR(1) deterministic texture noise
    - CI = sqrt(horizon) * residual quantiles
     |
     v
[8] HYBRID BLEND (SmartHybridPredictor)
    Evaluate 6 simple methods on walk-forward window:
    MA(7), MA(14), MA(30), Holt, LinearTrend, Theta
    Blend: ML weight = max(30%, 1 - simple/ML ratio)
    CI shifts with blended forecast
     |
     v
[9] OUTPUT
    Status with R-squared, MAE, model name
    Interactive Plotly chart with confidence bands
    Data table with date/forecast/lower/upper
    CSV download, HTML report generation
```

## App Tabs

| Tab | Purpose |
|---|---|
| Quick Predict | One-click: fetch -> train -> forecast. Hybrid blending, smart external feature selection |
| Data & Charts | Browse products, fetch/upload data, preview, history chart |
| Training | Customize models/features, see comparison plot, feature importance (own vs external), save |
| Forecast | Generate predictions from trained model with anomaly detection |
| Backtest | Walk-forward validation (n folds), actual vs predicted plots |

## Key Behaviors

- **Smart External Select**: Tests each macro feature independently. Only keeps those that reduce MAE by >5%. If none help, falls back to price-only model.
- **Hybrid Blending**: Compares ML against 6 simple methods. Blends with dynamic weights (ML always >=30%). If ML fails, uses best simple method.
- **R-squared Filter**: Models with R-squared <= 0 are automatically excluded. Only valid models compete for best.
- **Confidence Intervals**: Derived from detrended residuals, shifted with forecast, textured with deterministic AR(1) noise.
- **Anomaly Detection**: Step-move detection flags regime changes in forecast vs actual.

## File Structure

```
data_pipeline.py      ~1189 lines
model_pipeline.py     ~1474 lines
app.py                ~1244 lines
system_prompt.md      specification
README.md             this file
data/cache/           cached price CSVs
data/external/        external data cache (USD, gold, CPI)
data/models/          saved trained models (.joblib/.pkl)
data/forecasts/       exported forecast CSVs
test/
  test_full_system.py   116 unit tests
  debug_test.py         end-to-end pipeline test
  test_data_pipeline.py basic data pipeline test
```

## Tests

```
python test/test_full_system.py    # 116 tests, all public functions
python test/debug_test.py          # live pipeline integration test
```
