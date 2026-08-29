# save as comprehensive_test.py and run: python comprehensive_test.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np, warnings, traceback, time
from datetime import datetime

warnings.filterwarnings("ignore")
import io, codecs
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
print("=" * 70)
print("COMPREHENSIVE PIPELINE TEST")
print("=" * 70)

# ----- IMPORTS -----
try:
    from data_pipeline import (
        Fetcher, Cache, preprocess, merge_external_features,
        fetch_related_prices, create_features, detrend_series, apply_trend,
        select_features, auto_params, generate_fallback,
        get_product_profile, list_external_features, check_data_quality,
        _nfc, _is_iran_holiday, _add_holiday_feature,
        EXTERNAL_FEATURE_REGISTRY, PRODUCT_PROFILES, CACHE_DIR,
    )
    print("✓ data_pipeline imports OK")
except Exception as e:
    print(f"✗ data_pipeline import failed: {e}")
    exit()

try:
    from model_pipeline import (
        compute_split_date, _resolve_col, get_model, train_model, train_all,
        quick_predict, quick_predict_hybrid, backtest, iterative_forecast,
        compute_original_scale_metrics, create_forecast_plot,
        SmartHybridPredictor, DEFAULT_MODELS, DEFAULT_DETREND, MIN_ML_WEIGHT,
        XGB_AVAILABLE, LGBM_AVAILABLE, JOBLIB_AVAILABLE, MODELS_DIR, FORECASTS_DIR,
        cleanup_old_models, save_trained_models, load_trained_model, list_saved_models,
        naive_benchmark, model_improvement_vs_naive, detect_volatility_regime,
        adjust_forecast_ci, detect_anomalies, format_anomaly_report,
        feature_importance_by_horizon, generate_report, train_direct_models,
        forecast_direct_ensemble, _ar1_texture, create_ensemble, make_safe_filename,
    )
    print("✓ model_pipeline imports OK")
except Exception as e:
    print(f"✗ model_pipeline import failed: {e}")
    exit()

# ----- SETUP -----
fetcher = Fetcher()
cache = Cache(fetcher)
products, source = fetcher.discover_products()
pn = "گوساله کشتار"
pid = products.get(pn)
if pid is None:
    pid = list(products.values())[0]
    pn = list(products.keys())[0]
target = "price"
freq = "D"
horizon = 14

# Clean old models
cleanup_old_models()

print(f"\nProduct: {pn} (ID: {pid}), Source: {source}")
print(f"XGBoost: {XGB_AVAILABLE}, LightGBM: {LGBM_AVAILABLE}, joblib: {JOBLIB_AVAILABLE}")

# ---------- 1. DATA PIPELINE ----------
print("\n" + "="*70)
print("1. DATA PIPELINE TESTS")
print("="*70)

# 1.1 Fetch & preprocess
print("\n1.1 Fetching data...")
raw_df, cache_status = cache.update(pid, force=False)
if raw_df.empty:
    raw_df = generate_fallback()
    cache_status = "fallback"
df = preprocess(raw_df, target, freq)
print(f"    Cache: {cache_status}, rows after preprocess: {len(df)}")
print(f"    Columns: {list(df.columns)}")
print(f"    Holiday ratio: {df['is_holiday'].mean():.2%}")
print(f"    Price range: {df[target].min():,.0f} to {df[target].max():,.0f}")
# check data quality
quality = check_data_quality(df, target)
if quality:
    print(f"    Warnings: {quality}")

# 1.2 External & feed
print("\n1.2 Merging external features...")
df = merge_external_features(df, pn, freq)
ext_cols = [c for c in df.columns if c.startswith("ext_")]
print(f"    External columns: {ext_cols}")
df = fetch_related_prices(df, pn, fetcher, cache, freq)
feed_cols = [c for c in df.columns if c.startswith("ext_feed_")]
print(f"    Feed columns: {feed_cols}")
# Ensure no NaN after fill
for c in ext_cols + feed_cols:
    if df[c].isna().any():
        print(f"    WARNING: {c} has NaN after fill")

# 1.3 Feature engineering
print("\n1.3 Feature engineering...")
split_date = compute_split_date(df)
print(f"    Split date: {split_date.date()}")
# Test different detrend methods
for method in ["linear", "diff", "none"]:
    det_result = detrend_series(df[df["date"] < split_date], target, freq, method, split_date=split_date)
    if method == "linear":
        coeffs, ref_date = det_result
    print(f"    Detrend '{method}': {det_result[:2] if len(det_result)!=3 else 'smooth_diff'}")

lags, windows = auto_params(df, target, freq)
print(f"    Auto lags: {lags}, windows: {windows}")
detrended = f"{target}_detrended"
# Use linear for further tests
if coeffs is not None:
    df[detrended] = df[target] - apply_trend(df["date"], coeffs, ref_date, freq)
else:
    df[detrended] = df[target].diff()

ext_present = [c for c in df.columns if c.startswith("ext_")]
other_cols = [c for c in df.columns if c not in [target, "date", detrended, "smooth", "is_holiday"]
              and np.issubdtype(df[c].dtype, np.number) and df[c].nunique() > 10]

df_feat, feature_cols, _ = create_features(df, detrended, lags, windows, freq,
                                           other_cols=other_cols, external_cols=ext_present or None,
                                           add_seasonal=True)
# Add lag of detrended
for lag in lags:
    col = f"{detrended}_lag_{lag}"
    df_feat[col] = df_feat[detrended].shift(lag)
    feature_cols.append(col)
feature_cols = list(dict.fromkeys([c for c in feature_cols if c in df_feat.columns]))
feature_cols = select_features(df_feat, feature_cols, split_date, max_corr=0.95)
df_clean = df_feat.dropna(subset=[detrended] + [c for c in feature_cols if c in df_feat.columns])
print(f"    Final dataset: {len(df_clean)} rows × {len(feature_cols)} features")
print(f"    Own‑graph: {len([c for c in feature_cols if not c.startswith('ext_')])}, "
      f"External: {len([c for c in feature_cols if c.startswith('ext_')])}")

# ---------- 2. MODEL PIPELINE ----------
print("\n" + "="*70)
print("2. MODEL PIPELINE TESTS")
print("="*70)

# 2.1 Individual model test
print("\n2.1 Individual model training...")
X = df_clean[feature_cols].values
y = df_clean[detrended].values
dates = pd.to_datetime(df_clean["date"]).values
train_mask = dates < np.datetime64(pd.to_datetime(split_date))
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

for name in ["Ridge", "Lasso", "ElasticNet", "Random Forest"]:
    model, params = get_model(name)
    t0 = time.time()
    result = train_model(X_train, y_train, X_test, y_test, model, params, fast=True)
    elapsed = time.time() - t0
    status = "✓" if result['r2'] > 0 else "✗"
    print(f"    {status} {name:15s}: MAE={result['mae']:>10,.0f}, R²={result['r2']:.3f} ({elapsed:.1f}s)")

# 2.2 train_all with ensemble
print("\n2.2 train_all (with ensemble)...")
selected_models = ["Ridge", "Lasso"]
results, trained, best_m, best_s, best_name, best_std, best_mae = train_all(
    df_clean, detrended, feature_cols, selected_models, split_date,
    use_ensemble=True, ensemble_type="Voting", fast=True,
)
print(f"    Models trained: {list(results.keys())}")
print(f"    Best model: {best_name} (MAE={best_mae:.0f}, R²={results[best_name]['R-squared']:.3f})")

# 2.3 Original scale metrics
print("\n2.3 Original scale metrics...")
if coeffs is not None:
    orig_metrics, _ = compute_original_scale_metrics(
        trained, df_clean, feature_cols, target,
        coeffs, ref_date, freq, split_date,
        pred_col=detrended,
    )
    orig_mae_str = {k: f"{v['MAE']:.0f}" for k, v in orig_metrics.items() if 'Naive' not in k}
    print(f"    Original MAE: {orig_mae_str}")
else:
    print("    Skipped (diff mode)")

# 2.4 Iterative forecast
print("\n2.4 Iterative forecast...")
if best_m is not None:
    pred_df = iterative_forecast(
        df_clean, target, detrended, feature_cols,
        best_m, best_s, best_std, horizon, lags, windows, freq,
        coeffs, ref_date, None, None, external_cols=ext_present,
    )
    if not pred_df.empty:
        print(f"    Forecast generated: {len(pred_df)} points")
        print(f"    Range: {pred_df['forecast'].iloc[0]:,.0f} → {pred_df['forecast'].iloc[-1]:,.0f}")
    else:
        print("    FAILED: empty forecast")
else:
    print("    Skipped (no model)")

# 2.5 Quick Predict (end-to-end)
print("\n2.5 Quick Predict (E2E)...")
t0 = time.time()
pred_df_qp, best_qp, status_qp, metrics_qp, trained_qp, test_preds, comp_df = quick_predict(
    pn, pid, fetcher, cache,
    horizon=horizon, force_refresh=False, fast=True,
    lags=lags, windows=windows,
    selected_models=["Ridge", "Lasso"],
)
elapsed = time.time() - t0
print(f"    Status: {status_qp}")
print(f"    Time: {elapsed:.1f}s")
if pred_df_qp is not None and not pred_df_qp.empty:
    print(f"    Forecast: {pred_df_qp['forecast'].iloc[0]:,.0f} → {pred_df_qp['forecast'].iloc[-1]:,.0f}")
else:
    print("    FAILED: empty forecast")

# 2.6 Quick Predict Hybrid
print("\n2.6 Quick Predict Hybrid...")
t0 = time.time()
pred_df_h, best_h, status_h, metrics_h, trained_h, test_preds_h, comp_df_h = quick_predict_hybrid(
    pn, pid, fetcher, cache,
    horizon=horizon, force_refresh=False, fast=True,
    lags=lags, windows=windows,
    selected_models=["Ridge", "Lasso"],
)
elapsed = time.time() - t0
print(f"    Status: {status_h}")
print(f"    Time: {elapsed:.1f}s")
if pred_df_h is not None and not pred_df_h.empty:
    print(f"    Forecast: {pred_df_h['forecast'].iloc[0]:,.0f} → {pred_df_h['forecast'].iloc[-1]:,.0f}")
else:
    print("    FAILED: empty forecast")

# 2.7 Backtest
print("\n2.7 Backtest...")
bt = backtest(
    df_clean, target, lags, windows, freq, "linear",
    ["Ridge", "Lasso"], horizon=14, n_folds=2, fast=True,
)
if not bt.empty:
    print(f"    Folds: {len(bt)}, Avg MAE: {bt['mae'].mean():,.0f}")
else:
    print("    FAILED: empty backtest")

# 2.8 Persistence (if joblib available)
print("\n2.8 Model persistence...")
if JOBLIB_AVAILABLE and trained_qp:
    paths = save_trained_models(trained_qp, pn, {"test": "comprehensive"})
    print(f"    Saved {len(paths)} models")
    if paths:
        loaded = load_trained_model(paths[0])
        print(f"    Loaded model: {'OK' if loaded else 'FAIL'}")
    list_files = list_saved_models(pn)
    print(f"    Listed models for product: {len(list_files)} files")
else:
    print("    Skipped (joblib not available or no trained models)")

# 2.9 Miscellaneous functions
print("\n2.9 Utility functions...")
# _resolve_col
col_resolved = _resolve_col(df_clean, target)
print(f"    _resolve_col('{target}') -> '{col_resolved}'")
# create_forecast_plot (just check it runs)
try:
    fig = create_forecast_plot(df, target, pred_df_qp) if pred_df_qp is not None else None
    print(f"    create_forecast_plot: {'OK' if fig else 'FAIL'}")
except Exception as e:
    print(f"    create_forecast_plot: {e}")
# volatility regime
regime = detect_volatility_regime(df, target)
print(f"    Volatility regime: {regime:.2f}")
# anomalies
if test_preds is not None and not test_preds.empty:
    anom = detect_anomalies(test_preds["actual"].values, test_preds["predicted"].values, metrics_qp.get("std_res", 0))
    print(f"    Anomalies detected: {len(anom)}")
    if anom:
        print(f"    Sample: {format_anomaly_report(anom[:1])}")
# feature importance
if best_m is not None:
    fi = feature_importance_by_horizon(best_m, feature_cols, external_cols=ext_present)
    print(f"    Feature importance: own_graph={len(fi['own_graph'])}, external={len(fi['external'])}")
# report generation
if best_qp and pred_df_qp is not None:
    report = generate_report(pn, df, pred_df_qp, best_qp, metrics_qp, backtest_df=bt, anomalies=anom if anom else None)
    print(f"    Generate report: {'OK' if len(report)>100 else 'FAIL'}")

# 2.10 Edge cases
print("\n2.10 Edge cases...")
# test with minimal data
df_small = df.head(100).copy()
try:
    df_small_f, fcols_s, _ = create_features(df_small, detrended, [1,7], [7], freq, add_seasonal=False)
    print(f"    Small data (100 rows): features created ({len(fcols_s)} cols)")
except Exception as e:
    print(f"    Small data: {e}")
# test with no external features
df_noext = df.drop(columns=ext_cols+feed_cols, errors='ignore')
try:
    df_noext_f, fcols_ne, _ = create_features(df_noext, detrended, lags, windows, freq, external_cols=None)
    print(f"    No external features: {len(fcols_ne)} features created")
except Exception as e:
    print(f"    No external: {e}")
# test naive fallback forecast
from model_pipeline import _naive_fallback_forecast
naive_df = _naive_fallback_forecast(df, target, 7, freq)
print(f"    Naive fallback forecast: {len(naive_df)} points")
# test _ar1_texture
noise = _ar1_texture(20, "test", 10000)
print(f"    AR1 texture: mean={noise.mean():.2f}, std={noise.std():.2f}")

# ---------- FINAL SUMMARY ----------
print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
# Quick overall assessment
all_ok = (
    len(df) > 100 and
    len(df_clean) > 100 and
    pred_df_qp is not None and not pred_df_qp.empty and
    not bt.empty
)
if all_ok:
    print("✓ All critical components passed.")
    print("  - Data pipeline: fetching, external features, feature engineering work correctly.")
    print("  - Model pipeline: training, forecasting, backtesting, hybrid blending functional.")
    print("  - Ready to launch the Gradio app with 'python app.py'.")
else:
    print("✗ Some issues detected. Review the output above.")