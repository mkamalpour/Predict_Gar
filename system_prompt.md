# SPEC: ITPNews Price Intelligence v13

## Architecture (3 files, strict isolation)

```
data_pipeline.py  → web scraping, caching, preprocessing, feature engineering, detrending
                    NEVER imports: sklearn, xgboost, lightgbm

model_pipeline.py → ML training, forecasting, hybrid blending, backtesting, reports
                    NEVER imports: requests, bs4

app.py            → Gradio UI (5 tabs), imports both pipelines
```

## Rules (violate any = fail)

1. No emoji. Plain text only.
2. No data leakage: all features use shift(1), scalers fit on train only, split by date.
3. Single canonical split: `compute_split_date()` called ONCE, passed as param everywhere.
4. File isolation (rule 3-4 above).
5. Default detrend = "linear", default models = ["Lasso", "Ridge", "ElasticNet", "Random Forest"].
6. R-squared <= 0 models excluded from winner selection.
7. ML weight >= 30% in hybrid blend. Dynamic weights (not hardcoded).
8. CI bands shift with blended forecast. AR(1) deterministic texture on all forecasts.
9. Unicode NFC on all Persian strings. No product-name branches -- only PRODUCT_PROFILES.
10. Every callback wrapped in try/except returning correct output count.
11. Data filling: unlimited `ffill().bfill().interpolate()` (no `limit`), never use `asfreq`.
12. External features use only windows <= 14 days in create_features. auto_params caps window at 30.
13. Model training always scales target: y_scaled = (y - mean) / std. StandardScaler on features.
14. Metrics display uses detrended MAE/R-squared from trained dict (indices 2,3).
15. Confidence intervals from trained_models[best_name] indices 4 (std), 5 (q10), 6 (q90).
16. backtest() passes smooth_w to iterative_forecast; extracts smooth_w before the detrend if/elif chain.
17. other_cols always excludes [target, "date", detrended_target, "smooth"].

## Constants

```
data_pipeline.py:
  CACHE_DIR = "data/cache"
  EXTERNAL_CACHE_DIR = "data/external"
  MODELS_DIR = "data/models"
  FORECASTS_DIR = "data/forecasts"
  MAX_CACHE_DAYS = 1825
  DEFAULT_SMOOTH_WINDOW = 7
  PRODUCTS_CACHE_FILE = "data/products.json"
  EMERGENCY_FALLBACK = {"گوساله کشتار":508, "گوساله زنده":506, "گوسفند کشتار":507,
    "گوسفند زنده":505, "مرغ کشتار":67, "مرغ زنده":42, "تخم مرغ":19,
    "ذرت":13, "کنجاله سویا":1, "جو":95, "گندم":39}
  IRAN_HOLIDAYS_JALALI = [(1,1),(1,2),(1,3),(1,4),(1,12),(1,13),(3,14),(3,15),(11,22),(12,29)]
  RELATED_PRODUCTS = {"گوساله کشتار":["ذرت","کنجاله سویا","جو","گوساله زنده"], ...}
  EXTERNAL_FEATURE_REGISTRY = {
    "usd_irr_free": {"fetcher":"fetch_usd_irr_free", "freq":"D", "lag_days":1},
    "gold_price":   {"fetcher":"fetch_gold_price",   "freq":"D", "lag_days":1},
    "cpi_inflation":{"fetcher":"fetch_cpi_inflation", "freq":"M", "lag_days":0},
  }
  PRODUCT_PROFILES = {
    "_default": {"related_products":True, "external_features":[]},
    "گوساله کشتار": {"related_products":True, "external_features":["usd_irr_free","gold_price","cpi_inflation"]},
  }

model_pipeline.py:
  DEFAULT_MODELS = ["Lasso", "Ridge", "ElasticNet", "Random Forest"]
  DEFAULT_DETREND = "linear"
  MIN_ML_WEIGHT = 0.30
```

## detrend_series (4 branches)

Signature: `detrend_series(train_df, target, freq="D", method="linear", smooth_w=None, split_date=None)`
Returns: `(coeffs, ref_date)` or `(None, None, w)` for smooth_diff.

| method | return value | detrended_target assignment |
|---|---|---|
| smooth_diff | (None, None, w) | `df["smooth"] = rolling(w).mean()` then `df[target].diff()` |
| linear/quadratic | (coeffs_array, ref_date) | `df[target] - apply_trend(date, coeffs, ref_date)` |
| none | (None, Timestamp.today()) | `df[target]` (pass-through) |
| diff | (None, None) | `df[target].diff()` |

`use_diff = coeffs is None and ref_date is None` -- controls reconstruction in metrics and forecast.

## data_pipeline.py Public API

```python
# Utilities
_nfc(s) -> str                       # Unicode NFC normalize
_nfc_keys(d) -> dict                 # NFC all string keys
_polite_wait()                       # Rate limiting (thread-safe with Lock)
_jalali_to_gregorian(jy,jm,jd) -> (year,month,day)
_gregorian_to_jalali_tuple(dt) -> (year,month,day)
clean_number(text) -> float|None     # Parses Persian/Arabic numerals, commas
clean_text(text) -> str              # Strips + normalizes
_is_iran_holiday(date_val) -> int    # 1 if Thu/Fri or solar holiday
_add_holiday_feature(df) -> df       # Adds is_holiday column

# Product profiles
get_product_profile(name) -> dict    # NFC lookup, fallback _default
list_external_features(name) -> list # Keys from profile's external_features

# Classes
class ExternalDataSource:
    get(key, force_refresh=False) -> df|None  # NEVER raises
    fetch_usd_irr_free() -> df|None
    fetch_gold_price() -> df|None
    fetch_cpi_inflation() -> df|None
external_data = ExternalDataSource()  # singleton

class Fetcher:
    discover_products() -> (dict, source_str)  # {name: product_id}
    fetch_chart(pid, fy,fm,fd,ty,tm,td, sub=None, prd=None) -> df
    fetch_monthly(pid) -> df
    fetch_yearly(pid) -> df
    fetch_week_compare(pid) -> df
    fetch_year_compare(pid) -> df
    fetch_detail(pid) -> df

class Cache:
    __init__(fetcher)
    load(pid) -> df
    save(pid, df)
    update(pid, force=False) -> (df, source_str)  # incremental

# Core functions
preprocess(df, target, freq="D", iqr_k=3.0) -> df
    # no asfreq; ffill().bfill().interpolate() unlimited; IQR clip; holiday feature

merge_external_features(df, product_name, freq="D") -> df
    # fast-path if no features; merge_asof per registry key; ffill().bfill()

fetch_related_prices(df, product_name, fetcher, cache, freq="D") -> df
    # Fetches feed prices for related products; ffill().bfill()

create_features(df, target_col, lags, windows, freq="D", other_cols=None,
                external_cols=None, add_seasonal=True) -> (df, feature_cols, seasonal_cols)
    # Seasonal: month_sin/cos, dow_sin/cos, month, quarter, dayofweek, dayofyear, is_holiday
    # Lags: target_lag_N (shift(N))
    # Rolling: target_ma_W, target_std_W (on shift(1))
    # Momentum: target_roc_W, target_position_W, target_cv_W, target_oscillator_W (on shift(1))
    # Cross-product: other_col_ratio, other_col_margin, other_col_corr (both sides shift(1))
    # External: ext_* added directly (pre-lagged), rolling mean/std on shift(1) for windows <=14
    # NEVER fills target. Replaces inf with NaN.

check_data_quality(df, target_col) -> list[str]

detrend_series(train_df, target, freq, method, smooth_w, split_date) -> tuple  # see table above

apply_trend(dates, coeffs, ref_date, freq="D") -> np.ndarray  # polynomial eval

select_features(df, feature_cols, split_date=None, max_corr=0.95) -> list
    # Returns early if len <= 5; drops high-correlation features (upper triangle)

rank_features(model, feature_cols) -> list[(name, importance)]
remove_low_importance_features(df, feature_cols, model, threshold_quantile=0.05) -> list

auto_params(df, target_col, freq="D") -> (lags, windows)
    # max window 30 for daily with >1500 rows; min window = max(3, N//100)

generate_fallback(n_days=730, base=8_000_000) -> df  # synthetic data with trend+seasonality+noise
```

## model_pipeline.py Public API

```python
# Split
compute_split_date(df, date_col="date", train_frac=0.8) -> Timestamp
_resolve_col(df, key) -> str  # NFC column name resolver

# Models
get_model(name) -> (model, param_grid)
    # Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, XGBoost (optional), LightGBM (optional)
    # Unknown names fall back to Ridge with warning

create_ensemble(models_dict, ensemble_type="Voting") -> model|None
    # VotingRegressor or StackingRegressor

# Training
train_model(X_train, y_train, X_test, y_test, model, params=None, fast=False) -> dict
    # Keys: model, scaler, mae, rmse, r2, mape, std_res, residual_q10, residual_q90
    # Always scales target: (y - mean) / std; StandardScaler on features
    # Clips predictions to 0.8*min..1.2*max of train target

train_all(df, detrended_target, feature_cols, selected_models, split_date,
          use_ensemble=False, ensemble_type="Voting", fast=True, progress=None, train_log=None)
    -> (results_dict, trained_dict, best_model, best_scaler, best_name, best_std, best_mae)
    # trained_dict: {name: (model, scaler, mae, r2, std_res, residual_q10, residual_q90)}
    # Filters: R-squared > 0, MAE < naive MAE
    # ThreadPoolExecutor(4) in fast mode
    # Best = lowest MAE among valid models

# Persistence
save_trained_models(trained, product_name, metadata=None) -> list[str]
load_trained_model(path) -> dict|None  # {model, scaler, metadata}
list_saved_models(product_name=None) -> list[str]
cleanup_old_models() -> int  # keeps 1 per product, removes older

# Metrics
naive_benchmark(test_actuals) -> dict  # MAE, RMSE, R-squared, MAPE%, std_res, residual_q10, residual_q90
model_improvement_vs_naive(model_mae, naive_mae) -> float

compute_original_scale_metrics(trained_models, df_feat, feature_cols, target_name,
    coeffs, ref_date, freq, split_date, smooth_w=0, pred_col=None)
    -> (original_metrics_dict, y_test_original_array)
    # pred_col defaults to target_name; callers MUST pass pred_col=detrended_target
    # Undetrends predictions: cumsum for diff, +trend for linear, smooth+cumsum for smooth_diff

feature_importance_by_horizon(model, feature_cols, external_cols=None)
    -> {"own_graph": [{feature, importance, share}], "external": [...]}

# Forecasting
_ar1_texture(n, seed_key, std_res, phi=0.35) -> np.ndarray  # deterministic AR(1) noise

iterative_forecast(df, target, detrended_target, feature_cols, model, scaler, std_res,
    horizon, lags, windows, freq, coeffs, ref_date, residual_q10, residual_q90,
    external_cols=None, smooth_w=None, noise_threshold=0.0, add_texture=True,
    other_cols=None) -> DataFrame
    # Recomputes detrended_target from scratch using 4-branch logic
    # Recursive: NaN row -> create_features -> predict -> append -> repeat
    # External values forward-filled from last known
    # Undetrend: cumsum for use_diff, +trend for linear, pass-through for none
    # CI: sqrt(horizon) * residual quantiles + AR(1) texture
    # other_cols passed through to create_features for cross-product features

create_forecast_plot(df, target, pred_df, test_preds=None) -> go.Figure

# Anomalies / Volatility
detect_volatility_regime(df, target_col, window=30) -> float  # ratio of recent vs global std
adjust_forecast_ci(pred_df, regime_ratio, base_q10, base_q90) -> df  # widens CI in volatile regimes
detect_anomalies(actual, predicted, std_res, threshold=3.0) -> list[dict]
format_anomaly_report(anomalies, dates=None) -> str

# Hybrid blending
class SmartHybridPredictor:
    __init__(ml_func, test_window=60)
    _simple_methods() -> dict  # MA7/14/30, Holt, LinearTrend, Theta
    _ma_forecast(series, horizon, window) -> np.ndarray
    _holt_forecast(series, horizon, alpha=0.3, beta=0.1) -> np.ndarray
    _trend_forecast(series, horizon) -> np.ndarray
    _theta_forecast(series, horizon, theta=2.0) -> np.ndarray
    _evaluate_method(series, horizon, method_fn) -> float  # walk-forward MAE
    predict(product_name, product_id, fetcher, cache, horizon=30, ...) -> tuple
        # 1. Call ml_func (quick_predict)
        # 2. If ML fails: fallback to best simple method
        # 3. Else: evaluate simple, blend (ML>=30%, dynamic weights)
        # 4. Return (pred_df, best_name, status, metrics, trained, test_preds, comp_df)

# Quick Predict pipeline
quick_predict(product_name, product_id, fetcher, cache, horizon=30, force_refresh=False,
    lags=None, windows=None, selected_models=None, detrend_method="linear",
    freq="D", fast=True, train_log=None) -> (pred_df, best_name, status, metrics, trained, test_preds, comp_df)
    # 1. cache.update -> preprocess -> auto_params -> merge_external_features
    # 2. compute_split_date (ONCE)
    # 3. detrend_series -> apply 4-branch detrend logic
    # 4. create_features (with external_cols) -> select_features
    # 5. Smart External Select: test each ext feature, keep if MAE improvement > 5%
    # 6. train_all -> filter R-squared <= 0
    # 7. compute_original_scale_metrics(pred_col=detrended_target)
    # 8. iterative_forecast -> volatility adjust
    # 9. comp_df from trained dict (detrended MAE, R-squared)
    # 10. Return tuple

quick_predict_hybrid(...) -> tuple  # wraps quick_predict in SmartHybridPredictor

# Backtest
backtest(df, target, lags, windows, freq, detrend_method, selected_models,
    horizon=30, n_folds=3, fast=True) -> DataFrame
    # Walk-forward n_folds. Per fold:
    #   _resolve_col -> train_slice/test_slice -> detrend_series -> 4-branch
    #   create_features(train_slice, detrended_target, ...) NOT target
    #   select_features -> train_all -> iterative_forecast(smooth_w=smooth_w)
    # Returns: {fold, best_model, mae, rmse, n_test}

generate_report(product_name, df, pred_df, best_model_name, metrics,
    backtest_df=None, anomalies=None, feature_importance_split=None) -> str  # self-contained HTML

# Utility functions (also used internally)
train_direct_models(...) -> train_all wrapper
forecast_direct_ensemble(...) -> simple ensemble forecast
_naive_fallback_forecast(df, target, horizon, freq) -> pred_df
make_safe_filename(name) -> str  # NFC + strip special chars
```

## app.py (5 tabs, Gradio)

```
Imports from data_pipeline: Fetcher, Cache, preprocess, create_features, detrend_series,
  apply_trend, generate_fallback, merge_external_features, get_product_profile,
  list_external_features, check_data_quality, auto_params, select_features,
  CACHE_DIR, MODELS_DIR, FORECASTS_DIR

Imports from model_pipeline: train_all, compute_original_scale_metrics, iterative_forecast,
  create_forecast_plot, save_trained_models, load_trained_model, list_saved_models,
  cleanup_old_models, backtest, quick_predict, quick_predict_hybrid,
  detect_volatility_regime, detect_anomalies, format_anomaly_report,
  feature_importance_by_horizon, compute_split_date,
  XGB_AVAILABLE, LGBM_AVAILABLE, DEFAULT_MODELS, DEFAULT_DETREND, _resolve_col,
  SmartHybridPredictor

States:
  products_state    = gr.State(dict)         # {name: product_id}
  data_state        = gr.State(DataFrame)
  model_state       = gr.State({...})        # model, scaler, feature_cols, target_col,
                                             # detrended_target, trend_coeffs, ref_date,
                                             # std_residual, residual_q10, residual_q90,
                                             # best_model_name, lags, windows, best_mae,
                                             # freq, trained_models, detrend_method,
                                             # seasonal_cols, external_cols, smooth_w
  selected_model_name_state = gr.State(None)
  current_product_state     = gr.State(default)
  log_entries               = gr.State([])

Tab 1 - Quick Predict:
  Inputs:  Dropdown(product), Number(horizon=30,1-365), Checkbox(hybrid,fast,force)
  Outputs: status(3L), quality(2L), alerts(3L), metrics_df, chart, table, download
  Callback: qp_fn -> if hybrid: quick_predict_hybrid else quick_predict
            preprocess with "price" THEN rename to product_name
            Create chart, quality, anomalies, save CSV

Tab 2 - Data & Charts:
  Accordion: Available Products (refresh, table)
  Accordion: Fetch/Upload (target, extras multiselect, freq D/W/M, force, fetch_btn)
             Profile indicator (get_product_profile text)
             File upload CSV
  Outputs: status, dataset_info, preview(20), history_chart

Tab 3 - Training:
  Inputs:  lags_input("1,7,14,30"), windows_input("7,14,30,60")
           auto_params_btn, train_btn(primary)
           fast_mode, add_seasonal(True), auto_feature_select(True)
           detrend_method(default "linear"), models_check(default DEFAULT_MODELS)
           use_ensemble, ensemble_type
  Outputs: status, dataset_info, comparison_plot, feature_plot(split own_vs_external),
           comparison_table, save_btn, save_status
  Callback train_fn:
    4-branch detrend with smooth_w extraction
    other_cols excludes [target, "date", detrended_target, "smooth"]
    create_features with external_cols
    Auto-remove gate for external features
    compute_original_scale_metrics(pred_col=detrended_target)
    Feature importance split (own vs external)
    comp_df from trained_models (detrended MAE, R-squared)
    Extract best_std/q10/q90 from trained_models[best_name] indices 4,5,6
    Update state including smooth_w

Tab 4 - Forecast:
  Inputs:  Number(horizon=14), model_selector dropdown, refresh_btn,
           checkbox(hybrid=True), slider(anomaly_threshold=2)
  Outputs: status, alerts, chart, table, download
  Callback: iterative_forecast(smooth_w=state["smooth_w"])

Tab 5 - Backtest:
  Inputs:  Dropdown(models), Number(bt_horizon=30), bt_btn
  Outputs: status, actual_vs_pred plot, errors plot, metrics_df, details_df

Global:
  refresh_products_btn -> refresh all dropdowns
  sync_dropdowns helper
  target.change -> sync
  app.launch(share=False, server="127.0.0.1", port=7861)
```

## Critical implementation details

1. `pred_col` in compute_original_scale_metrics defaults to target_name, NOT "target". Callers pass `pred_col=detrended_target`.
2. `smooth_w` stored in app state, updated in training callback, threaded to iterative_forecast and backtest.
3. backtest() computes its own per-fold splits (walk-forward) -- deliberate exception to single-split rule.
4. Quick Predict preprocess: `preprocess(hist_df, "price")` THEN rename to product_name.
5. Smart External Select tests each external feature independently, keeps only those with >5% MAE improvement.
6. Auto-remove gate: retrain without external features, if MAE_without < MAE_with * 0.95, drop external.
7. naive_fallback uses `df[target].dropna().std()` with NaN guard (not raw `.std()`).
8. fetch_data_fn removes duplicate columns after merging: `merged = merged.loc[:, ~merged.columns.duplicated()]`.
9. Training tab comparison table built from trained_models (detrended), not original-scale metrics.
10. Status line shows "Detrended R-squared" from the trained model.

## Tests

- `test/test_full_system.py` -- 116 tests covering every public function
- `test/debug_test.py` -- End-to-end pipeline test with live data
- `test/test_data_pipeline.py` -- Basic data pipeline test
- Run: `python test/test_full_system.py` from project root
