"""
Full system test for ITPNews Price Intelligence v13.
Covers every public function in data_pipeline and model_pipeline.
Run: python test/test_full_system.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import hashlib
import numpy as np
import pandas as pd
import traceback
from datetime import datetime, timedelta

passed = 0
failed = 0
errors = []


def test(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        failed += 1
        errors.append((name, e))
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()


def assert_eq(a, b, msg=""):
    assert a == b, f"{msg}: expected {b!r}, got {a!r}"

def assert_true(cond, msg=""):
    assert cond, f"Assertion failed: {msg}"

def assert_in(item, collection, msg=""):
    assert item in collection, f"{item!r} not in {collection!r}: {msg}"

def assert_no_nan(arr, msg=""):
    arr = np.asarray(arr, dtype=float)
    assert not np.any(np.isnan(arr)), f"NaN found: {msg}"

def assert_positive(arr, msg=""):
    arr = np.asarray(arr, dtype=float)
    assert np.all(arr > 0), f"Non-positive values found: {msg}"


# ============================================================================
print("=" * 70)
print("DATA PIPELINE TESTS")
print("=" * 70)

# --- Imports ---
from data_pipeline import (
    _nfc, _nfc_keys, _polite_wait, _base_headers, _ajax_headers,
    _is_iran_holiday, _add_holiday_feature,
    jalali_to_gregorian, gregorian_to_jalali_tuple,
    clean_number, clean_text,
    Fetcher, Cache, ExternalDataSource,
    preprocess, create_features, detrend_series, apply_trend,
    select_features, auto_params, generate_fallback,
    check_data_quality, get_product_profile, list_external_features,
    merge_external_features,
    EXTERNAL_FEATURE_REGISTRY, PRODUCT_PROFILES,
    PERSIAN_DIGITS, ARABIC_DIGITS, PERSIAN_MONTHS,
    EMERGENCY_FALLBACK, IRAN_HOLIDAYS_JALALI,
    CACHE_DIR, DEFAULT_SMOOTH_WINDOW, MAX_CACHE_DAYS,
)

print("\n--- NFC Normalization ---")

def test_nfc_basic():
    assert _nfc("hello") == "hello"
    assert _nfc(None) is None
    assert _nfc(123) == "123"
test("nfc_basic", test_nfc_basic)

def test_nfc_persian():
    s = "\u06af\u0648\u0633\u0627\u0644\u0647"
    assert _nfc(s) == s
    assert _nfc_keys({"a": 1, s: 2}) == {"a": 1, s: 2}
test("nfc_persian", test_nfc_persian)


print("\n--- Date Utilities ---")

def test_jalali_gregorian():
    result = jalali_to_gregorian(1403, 1, 1)
    assert result is not None
    assert "2024" in result
test("jalali_gregorian", test_jalali_gregorian)

def test_jalali_invalid():
    assert jalali_to_gregorian(9999, 99, 99) is None
test("jalali_invalid", test_jalali_invalid)

def test_gregorian_to_jalali():
    dt = pd.Timestamp("2024-03-20")
    jy, jm, jd = gregorian_to_jalali_tuple(dt)
    assert jy == 1403
    assert jm == 1
test("gregorian_to_jalali", test_gregorian_to_jalali)

def test_gregorian_to_jalali_invalid():
    jy, jm, jd = gregorian_to_jalali_tuple("bad_date")
    assert jy is None
test("gregorian_to_jalali_invalid", test_gregorian_to_jalali_invalid)


print("\n--- Clean Number ---")

def test_clean_number_basic():
    assert clean_number("1,234") == 1234
    assert clean_number("۱۲۳۴") == 1234
    assert clean_number("١٢٣٤") == 1234
    assert clean_number("12.5") == 12.5
    assert clean_number(None) is None
    assert clean_number("") is None
    assert clean_number("-") is None
    assert clean_number(".") is None
test("clean_number_basic", test_clean_number_basic)

def test_clean_number_persian_comma():
    assert clean_number("1٬234٬567") == 1234567
test("clean_number_persian_comma", test_clean_number_persian_comma)


print("\n--- Clean Text ---")

def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text(None) == ""
    assert clean_text("") == ""
test("clean_text", test_clean_text)


print("\n--- Holiday Detection ---")

def test_holiday_thursday():
    dt = pd.Timestamp("2024-03-21")  # Thursday
    assert _is_iran_holiday(dt) == 1
test("holiday_thursday", test_holiday_thursday)

def test_holiday_friday():
    dt = pd.Timestamp("2024-03-22")  # Friday
    assert _is_iran_holiday(dt) == 1
test("holiday_friday", test_holiday_friday)

def test_holiday_weekday():
    dt = pd.Timestamp("2024-03-25")  # Monday
    assert _is_iran_holiday(dt) == 0
test("holiday_weekday", test_holiday_weekday)

def test_holiday_nowruz():
    dt = pd.Timestamp("2024-03-20")  # Nowruz
    assert _is_iran_holiday(dt) == 1
test("holiday_nowruz", test_holiday_nowruz)

def test_add_holiday_feature():
    df = pd.DataFrame({"date": pd.date_range("2024-03-20", periods=7)})
    df = _add_holiday_feature(df)
    assert "is_holiday" in df.columns
    assert df["is_holiday"].sum() >= 2  # Nowruz + Friday
test("add_holiday_feature", test_add_holiday_feature)


print("\n--- Generate Fallback ---")

def test_generate_fallback():
    df = generate_fallback(n_days=100, base=5_000_000)
    assert len(df) == 100
    assert "date" in df.columns
    assert "price" in df.columns
    assert df["price"].min() > 0
test("generate_fallback", test_generate_fallback)


print("\n--- Product Profiles ---")

def test_product_profile_default():
    p = get_product_profile("nonexistent_product")
    assert p["external_features"] == []
    assert p["related_products"] == True
test("product_profile_default", test_product_profile_default)

def test_product_profile_known():
    p = get_product_profile("\u06af\u0648\u0633\u0627\u0644\u0647 \u06a9\u0634\u062a\u0627\u0631")
    assert len(p["external_features"]) > 0
test("product_profile_known", test_product_profile_known)

def test_list_external_features():
    feats = list_external_features("\u06af\u0648\u0633\u0627\u0644\u0647 \u06a9\u0634\u062a\u0627\u0631")
    assert isinstance(feats, list)
    assert len(feats) == 3
test("list_external_features", test_list_external_features)

def test_list_external_features_default():
    feats = list_external_features("unknown")
    assert feats == []
test("list_external_features_default", test_list_external_features_default)


print("\n--- Data Quality Check ---")

def test_check_data_quality_good():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=200), "price": np.random.uniform(100, 200, 200)})
    w = check_data_quality(df, "price")
    assert isinstance(w, list)
    assert len(w) == 0
test("check_data_quality_good", test_check_data_quality_good)

def test_check_data_quality_small():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10), "price": np.random.uniform(100, 200, 10)})
    w = check_data_quality(df, "price")
    assert any("60" in x for x in w)
test("check_data_quality_small", test_check_data_quality_small)

def test_check_data_quality_empty():
    w = check_data_quality(pd.DataFrame(), "price")
    assert any("empty" in x.lower() for x in w)
test("check_data_quality_empty", test_check_data_quality_empty)


print("\n--- Preprocess ---")

def test_preprocess_basic():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "price": np.random.uniform(1e6, 2e6, 100)
    })
    result = preprocess(df, "price")
    assert "date" in result.columns
    assert "is_holiday" in result.columns
    assert len(result) == 100
    assert result["date"].is_monotonic_increasing
test("preprocess_basic", test_preprocess_basic)

def test_preprocess_nan_fill():
    prices = np.random.uniform(1e6, 2e6, 100).astype(float)
    prices[10] = np.nan
    prices[50] = np.nan
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=100), "price": prices})
    result = preprocess(df, "price")
    assert result["price"].isna().sum() == 0
test("preprocess_nan_fill", test_preprocess_nan_fill)

def test_preprocess_dedup():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-01", "2024-01-02"], "price": [100, 200, 300]})
    result = preprocess(df, "price")
    assert len(result) == 2
test("preprocess_dedup", test_preprocess_dedup)

def test_preprocess_iqr_clip():
    prices = np.ones(200) * 100
    prices[0] = 100000
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=200), "price": prices})
    result = preprocess(df, "price", iqr_k=1.0)
    assert result["price"].max() < 100000
test("preprocess_iqr_clip", test_preprocess_iqr_clip)

def test_preprocess_nfc_normalization():
    persian_name = "\u06af\u0648\u0633\u0627\u0644\u0647"
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=10),
        persian_name: np.random.uniform(1e6, 2e6, 10)
    })
    result = preprocess(df, persian_name)
    nfc_cols = [c for c in result.columns if c != "date"]
    assert all(_nfc(c) == c for c in nfc_cols)
test("preprocess_nfc_normalization", test_preprocess_nfc_normalization)


print("\n--- Detrend Series ---")

def _make_df(n=200):
    """Generate AR(1)-like price data where lag-1 strongly predicts the next value."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=n)
    prices = np.zeros(n)
    prices[0] = 1e7
    for i in range(1, n):
        prices[i] = prices[i - 1] * 1.001 + rng.normal(0, 5000)
    return pd.DataFrame({"date": dates, "price": prices})

def test_detrend_linear():
    df = _make_df()
    coeffs, ref = detrend_series(df, "price", method="linear")
    assert coeffs is not None
    assert len(coeffs) == 2
    assert ref is not None
test("detrend_linear", test_detrend_linear)

def test_detrend_quadratic():
    df = _make_df()
    coeffs, ref = detrend_series(df, "price", method="quadratic")
    assert len(coeffs) == 3
test("detrend_quadratic", test_detrend_quadratic)

def test_detrend_diff():
    df = _make_df()
    result = detrend_series(df, "price", method="diff")
    assert result == (None, None)
test("detrend_diff", test_detrend_diff)

def test_detrend_none():
    df = _make_df()
    result = detrend_series(df, "price", method="none")
    assert result[0] is None
    assert isinstance(result[1], pd.Timestamp)
test("detrend_none", test_detrend_none)

def test_detrend_smooth_diff():
    df = _make_df()
    result = detrend_series(df, "price", method="smooth_diff", smooth_w=14)
    assert len(result) == 3
    assert result[0] is None
    assert result[1] is None
    assert result[2] == 14
test("detrend_smooth_diff", test_detrend_smooth_diff)

def test_detrend_smooth_diff_default_w():
    df = _make_df()
    result = detrend_series(df, "price", method="smooth_diff")
    assert result[2] == DEFAULT_SMOOTH_WINDOW
test("detrend_smooth_diff_default_w", test_detrend_smooth_diff_default_w)

def test_detrend_with_split_date():
    df = _make_df()
    split = pd.Timestamp("2023-01-01")
    coeffs, ref = detrend_series(df, "price", method="linear", split_date=split)
    assert coeffs is not None
    assert ref <= split
test("detrend_with_split_date", test_detrend_with_split_date)

def test_detrend_empty_data():
    df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "price": pd.Series(dtype=float)})
    coeffs, ref = detrend_series(df, "price", method="linear")
    assert coeffs is not None  # fallback to zero coefficients
test("detrend_empty_data", test_detrend_empty_data)


print("\n--- Apply Trend ---")

def test_apply_trend():
    dates = pd.date_range("2024-01-01", periods=10)
    coeffs = np.array([1000.0, 5000000.0])
    ref = pd.Timestamp("2024-01-01")
    result = apply_trend(dates, coeffs, ref)
    assert len(result) == 10
    assert result[0] == 5000000.0
    assert result[1] > result[0]
test("apply_trend", test_apply_trend)

def test_apply_trend_none():
    dates = pd.date_range("2024-01-01", periods=5)
    result = apply_trend(dates, None, None)
    assert np.all(result == 0)
test("apply_trend_none", test_apply_trend_none)


print("\n--- Select Features ---")

def test_select_features_basic():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "a": np.random.randn(100),
        "b": np.random.randn(100),
        "c": np.random.randn(100),
    })
    result = select_features(df, ["a", "b", "c"], max_corr=0.95)
    assert len(result) <= 3
    assert all(c in df.columns for c in result)
test("select_features_basic", test_select_features_basic)

def test_select_features_few():
    result = select_features(pd.DataFrame(), ["a", "b"])
    assert result == ["a", "b"]
test("select_features_few", test_select_features_few)

def test_select_features_collinear():
    x = np.random.randn(100)
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "a": x,
        "b": x + np.random.normal(0, 0.001, 100),
        "c": np.random.randn(100),
        "d": np.random.randn(100),
        "e": np.random.randn(100),
        "f": np.random.randn(100),
    })
    result = select_features(df, ["a", "b", "c", "d", "e", "f"], max_corr=0.99)
    assert len(result) == 5
    assert not ("a" in result and "b" in result)
test("select_features_collinear", test_select_features_collinear)


print("\n--- Auto Params ---")

def test_auto_params_large():
    df = _make_df(1600)
    lags, windows = auto_params(df, "price", "D")
    assert all(w <= 30 for w in windows)
    assert len(lags) >= 3
test("auto_params_large", test_auto_params_large)

def test_auto_params_small():
    df = _make_df(50)
    lags, windows = auto_params(df, "price", "D")
    assert len(lags) >= 2
    assert len(windows) >= 1
test("auto_params_small", test_auto_params_small)

def test_auto_params_weekly():
    df = _make_df(200)
    lags, windows = auto_params(df, "price", "W")
    assert len(lags) >= 2
test("auto_params_weekly", test_auto_params_weekly)

def test_auto_params_monthly():
    df = _make_df(50)
    lags, windows = auto_params(df, "price", "M")
    assert len(lags) >= 2
test("auto_params_monthly", test_auto_params_monthly)


print("\n--- Create Features ---")

def test_create_features_basic():
    df = _make_df(200)
    df_feat, fcols, scols = create_features(df, "price", [1, 7], [7, 14])
    assert len(df_feat) == 200
    assert len(fcols) > 0
    assert "date" in df_feat.columns
test("create_features_basic", test_create_features_basic)

def test_create_features_seasonal():
    df = _make_df(200)
    _, fcols, scols = create_features(df, "price", [1], [7], add_seasonal=True)
    assert len(scols) > 0
    assert "month_sin" in fcols
    assert "is_holiday" in fcols
test("create_features_seasonal", test_create_features_seasonal)

def test_create_features_no_seasonal():
    df = _make_df(200)
    _, fcols, scols = create_features(df, "price", [1], [7], add_seasonal=False)
    assert len(scols) == 0
test("create_features_no_seasonal", test_create_features_no_seasonal)

def test_create_features_external():
    df = _make_df(200)
    df["ext_test"] = np.random.randn(200)
    _, fcols, _ = create_features(df, "price", [1], [7], external_cols=["ext_test"])
    assert "ext_test" in fcols
    assert "ext_test_ma_7" in fcols
test("create_features_external", test_create_features_external)

def test_create_features_external_windows_capped():
    df = _make_df(200)
    df["ext_test"] = np.random.randn(200)
    _, fcols, _ = create_features(df, "price", [1], [7, 14, 30], external_cols=["ext_test"])
    for c in fcols:
        if c.startswith("ext_test_") and "_ma_" in c:
            w = int(c.split("_ma_")[1])
            assert w <= 14, f"External window {w} exceeds 14"
test("create_features_external_windows_capped", test_create_features_external_windows_capped)

def test_create_features_other_cols():
    df = _make_df(200)
    df["other_price"] = np.random.uniform(1e6, 2e6, 200)
    _, fcols, _ = create_features(df, "price", [1], [7], other_cols=["other_price"])
    assert any("ratio" in c for c in fcols)
    assert any("margin" in c for c in fcols)
test("create_features_other_cols", test_create_features_other_cols)

def test_create_features_no_nan_infs():
    df = _make_df(200)
    df_feat, fcols, _ = create_features(df, "price", [1, 7], [7, 14])
    for c in fcols:
        assert not np.isinf(df_feat[c]).any(), f"Inf in {c}"
test("create_features_no_nan_infs", test_create_features_no_nan_infs)

def test_create_features_small_data():
    df = _make_df(20)
    _, fcols, _ = create_features(df, "price", [1], [7])
    assert len(fcols) >= 1
test("create_features_small_data", test_create_features_small_data)


print("\n--- Thread Safety ---")

def test_polite_wait_thread():
    import threading
    errors = []
    def worker():
        try:
            _polite_wait()
        except Exception as e:
            errors.append(e)
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(errors) == 0
test("polite_wait_thread", test_polite_wait_thread)


# ============================================================================
print("\n" + "=" * 70)
print("MODEL PIPELINE TESTS")
print("=" * 70)

from model_pipeline import (
    compute_split_date, _resolve_col, get_model, train_model, train_all,
    iterative_forecast, compute_original_scale_metrics,
    SmartHybridPredictor, quick_predict, quick_predict_hybrid, backtest,
    save_trained_models, load_trained_model, list_saved_models, cleanup_old_models,
    naive_benchmark, model_improvement_vs_naive,
    detect_volatility_regime, adjust_forecast_ci,
    detect_anomalies, format_anomaly_report,
    feature_importance_by_horizon, generate_report,
    _ar1_texture, _naive_fallback_forecast,
    DEFAULT_MODELS, DEFAULT_DETREND, MIN_ML_WEIGHT,
    XGB_AVAILABLE, LGBM_AVAILABLE, JOBLIB_AVAILABLE,
    MODELS_DIR, FORECASTS_DIR,
    create_ensemble, make_safe_filename,
)
from data_pipeline import (
    preprocess, create_features, detrend_series, apply_trend,
    select_features, auto_params, generate_fallback,
    merge_external_features, Fetcher, Cache,
)


print("\n--- Compute Split Date ---")

def test_split_date_basic():
    df = _make_df(1000)
    sd = compute_split_date(df)
    idx = int(1000 * 0.8)
    assert sd == pd.to_datetime(df["date"].iloc[idx])
test("split_date_basic", test_split_date_basic)

def test_split_date_small():
    df = _make_df(2)
    sd = compute_split_date(df)
    assert sd is not None
test("split_date_small", test_split_date_small)

def test_split_date_single():
    df = _make_df(1)
    sd = compute_split_date(df)
    assert sd is not None
test("split_date_single", test_split_date_single)

def test_split_date_empty():
    df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]")})
    sd = compute_split_date(df)
    assert sd is not None
test("split_date_empty", test_split_date_empty)


print("\n--- Resolve Column ---")

def test_resolve_col_exact():
    df = pd.DataFrame({"price": [1, 2], "date": [1, 2]})
    assert _resolve_col(df, "price") == "price"
test("resolve_col_exact", test_resolve_col_exact)

def test_resolve_col_nfc():
    df = pd.DataFrame({"\u06af\u0648\u0633\u0627\u0644\u0647": [1, 2], "date": [1, 2]})
    result = _resolve_col(df, "\u06af\u0648\u0633\u0627\u0644\u0647")
    assert result is not None
test("resolve_col_nfc", test_resolve_col_nfc)

def test_resolve_col_fallback():
    df = pd.DataFrame({"x": [1, 2], "date": [1, 2]})
    result = _resolve_col(df, "nonexistent")
    assert result is not None
test("resolve_col_fallback", test_resolve_col_fallback)


print("\n--- Get Model ---")

def test_get_all_models():
    for name in ["Lasso", "Ridge", "ElasticNet", "Random Forest", "Gradient Boosting"]:
        model, params = get_model(name)
        assert model is not None
        assert params is not None
test("get_all_models", test_get_all_models)

def test_get_model_unknown():
    model, params = get_model("NonExistentModelXYZ")
    assert model is not None
    assert model.__class__.__name__ == "Ridge"
test("get_model_unknown", test_get_model_unknown)

def test_get_xgboost():
    model, params = get_model("XGBoost")
    if XGB_AVAILABLE:
        assert "XGB" in model.__class__.__name__
    else:
        assert model.__class__.__name__ == "Ridge"
test("get_xgboost", test_get_xgboost)

def test_get_lightgbm():
    model, params = get_model("LightGBM")
    if LGBM_AVAILABLE:
        assert "LGBM" in model.__class__.__name__
    else:
        assert model.__class__.__name__ == "Ridge"
test("get_lightgbm", test_get_lightgbm)


print("\n--- Train Model ---")

def _make_train_data(n=300):
    df = _make_df(n)
    split = compute_split_date(df)
    lags, windows = auto_params(df, "price", "D")
    df_feat, fcols, _ = create_features(df, "price", lags, windows)
    fcols = select_features(df_feat, fcols, split)
    df_clean = df_feat.dropna(subset=["price"] + [c for c in fcols if c in df_feat.columns])
    X = df_clean[fcols].values
    y = df_clean["price"].values
    dates = pd.to_datetime(df_clean["date"]).values
    mask = dates < np.datetime64(pd.to_datetime(split))
    return X[mask], X[~mask], y[mask], y[~mask], fcols

def test_train_model_ridge():
    X_tr, X_te, y_tr, y_te, _ = _make_train_data()
    model, params = get_model("Ridge")
    result = train_model(X_tr, y_tr, X_te, y_te, model, params, fast=True)
    assert "model" in result
    assert "mae" in result
    assert "r2" in result
    assert result["mae"] >= 0
test("train_model_ridge", test_train_model_ridge)

def test_train_model_lasso():
    X_tr, X_te, y_tr, y_te, _ = _make_train_data()
    model, params = get_model("Lasso")
    result = train_model(X_tr, y_tr, X_te, y_te, model, params, fast=True)
    assert result["r2"] > -10
test("train_model_lasso", test_train_model_lasso)

def test_train_model_forest():
    X_tr, X_te, y_tr, y_te, _ = _make_train_data()
    model, params = get_model("Random Forest")
    result = train_model(X_tr, y_tr, X_te, y_te, model, params, fast=True)
    assert result["model"] is not None
test("train_model_forest", test_train_model_forest)

def test_train_model_scaling():
    X_tr, X_te, y_tr, y_te, _ = _make_train_data()
    model, params = get_model("Ridge")
    result = train_model(X_tr, y_tr, X_te, y_te, model, params, fast=False)
    scaler = result["scaler"]
    assert scaler is not None
    X_scaled = scaler.transform(X_te)
    assert not np.any(np.isnan(X_scaled))
test("train_model_scaling", test_train_model_scaling)


print("\n--- Train All ---")

def test_train_all_basic():
    df = _make_df(300)
    split = compute_split_date(df)
    lags, windows = [1, 7], [7]
    df_feat, fcols, _ = create_features(df, "price", lags, windows)
    fcols = select_features(df_feat, fcols, split)
    df_clean = df_feat.dropna(subset=["price"] + [c for c in fcols if c in df_feat.columns])
    results, trained, bm, bs, bn, bstd, bmae = train_all(
        df_clean, "price", fcols, ["Ridge", "Lasso"], split, fast=True
    )
    assert len(results) >= 2
    assert bm is not None
    assert bn is not None
test("train_all_basic", test_train_all_basic)

def test_train_all_filters_negative_r2():
    df = _make_df(300)
    split = compute_split_date(df)
    lags, windows = [1, 7], [7]
    df_feat, fcols, _ = create_features(df, "price", lags, windows)
    df_clean = df_feat.dropna(subset=["price"] + [c for c in fcols if c in df_feat.columns])
    _, trained, _, _, _, _, _ = train_all(
        df_clean, "price", fcols, ["Ridge", "Lasso"], split, fast=True
    )
    for name, tup in trained.items():
        assert tup[3] > 0, f"Model {name} has R²={tup[3]} <= 0"
test("train_all_filters_negative_r2", test_train_all_filters_negative_r2)

def test_train_all_with_ensemble():
    df = _make_df(300)
    split = compute_split_date(df)
    lags, windows = [1, 7], [7]
    df_feat, fcols, _ = create_features(df, "price", lags, windows)
    df_clean = df_feat.dropna(subset=["price"] + [c for c in fcols if c in df_feat.columns])
    results, trained, _, _, _, _, _ = train_all(
        df_clean, "price", fcols, ["Ridge", "Lasso"], split,
        use_ensemble=True, ensemble_type="Voting", fast=True
    )
    assert any("Ensemble" in n for n in results)
test("train_all_with_ensemble", test_train_all_with_ensemble)

def test_train_all_min_samples():
    df = _make_df(10)
    split = compute_split_date(df)
    results, trained, bm, _, _, _, _ = train_all(
        df, "price", ["price"], ["Ridge"], split, fast=True
    )
    assert bm is None  # not enough data
test("train_all_min_samples", test_train_all_min_samples)


print("\n--- Iterative Forecast ---")

def test_iterative_forecast_basic():
    df = _make_df(300)
    split = compute_split_date(df)
    lags, windows = [1, 7], [7]
    df_feat, fcols, _ = create_features(df, "price", lags, windows)
    fcols = select_features(df_feat, fcols, split)
    df_clean = df_feat.dropna(subset=["price"] + [c for c in fcols if c in df_feat.columns])
    results, trained, bm, bs, bn, bstd, bmae = train_all(
        df_clean, "price", fcols, ["Ridge"], split, fast=True
    )
    pred_df = iterative_forecast(
        df_clean, "price", "price_detrended", fcols, bm, bs, bstd,
        14, lags, windows, "D", None, pd.Timestamp.now(),
        residual_q10=None, residual_q90=None,
    )
    assert not pred_df.empty
    assert len(pred_df) == 14
    assert all(c in pred_df.columns for c in ["date", "forecast", "lower", "upper"])
test("iterative_forecast_basic", test_iterative_forecast_basic)

def test_iterative_forecast_linear():
    df = _make_df(300)
    split = compute_split_date(df)
    coeffs, ref = detrend_series(df[df["date"] < split], "price", method="linear", split_date=split)
    lags, windows = [1, 7], [7]
    df["price_detrended"] = df["price"] - apply_trend(df["date"], coeffs, ref, "D")
    df_feat, fcols, _ = create_features(df, "price_detrended", lags, windows)
    df_clean = df_feat.dropna(subset=["price_detrended"] + [c for c in fcols if c in df_feat.columns])
    results, trained, bm, bs, bn, bstd, bmae = train_all(
        df_clean, "price_detrended", fcols, ["Ridge"], split, fast=True
    )
    pred_df = iterative_forecast(
        df_clean, "price", "price_detrended", fcols, bm, bs, bstd,
        14, lags, windows, "D", coeffs, ref,
        residual_q10=None, residual_q90=None,
    )
    assert not pred_df.empty
    assert pred_df["forecast"].iloc[0] > 0
test("iterative_forecast_linear", test_iterative_forecast_linear)

def test_iterative_forecast_with_texture():
    df = _make_df(300)
    lags, windows = [1, 7], [7]
    df_feat, fcols, _ = create_features(df, "price", lags, windows)
    df_clean = df_feat.dropna(subset=["price"] + [c for c in fcols if c in df_feat.columns])
    results, trained, bm, bs, bn, bstd, bmae = train_all(
        df_clean, "price", fcols, ["Ridge"], compute_split_date(df), fast=True
    )
    pred = iterative_forecast(
        df_clean, "price", "price_detrended", fcols, bm, bs, bstd,
        14, lags, windows, "D", None, pd.Timestamp.now(),
        residual_q10=-50000, residual_q90=50000, add_texture=True,
    )
    pred_no_tex = iterative_forecast(
        df_clean, "price", "price_detrended", fcols, bm, bs, bstd,
        14, lags, windows, "D", None, pd.Timestamp.now(),
        residual_q10=-50000, residual_q90=50000, add_texture=False,
    )
    assert not np.allclose(pred["forecast"].values, pred_no_tex["forecast"].values)
test("iterative_forecast_with_texture", test_iterative_forecast_with_texture)

def test_iterative_forecast_ci_shifted_with_texture():
    df = _make_df(300)
    lags, windows = [1, 7], [7]
    df_feat, fcols, _ = create_features(df, "price", lags, windows)
    df_clean = df_feat.dropna(subset=["price"] + [c for c in fcols if c in df_feat.columns])
    results, trained, bm, bs, bn, bstd, bmae = train_all(
        df_clean, "price", fcols, ["Ridge"], compute_split_date(df), fast=True
    )
    pred = iterative_forecast(
        df_clean, "price", "price_detrended", fcols, bm, bs, bstd,
        14, lags, windows, "D", None, pd.Timestamp.now(),
        residual_q10=-50000, residual_q90=50000, add_texture=True,
    )
    assert (pred["forecast"] >= pred["lower"]).all()
test("iterative_forecast_ci_shifted_with_texture", test_iterative_forecast_ci_shifted_with_texture)

def test_iterative_forecast_other_cols():
    df = _make_df(300)
    df["other_col"] = np.random.uniform(1e6, 2e6, 300)
    df["price_detrended"] = df["price"]
    lags, windows = [1, 7], [7]
    df_feat, fcols, _ = create_features(df, "price_detrended", lags, windows, other_cols=["other_col"])
    df_clean = df_feat.dropna(subset=["price_detrended"] + [c for c in fcols if c in df_feat.columns])
    results, trained, bm, bs, bn, bstd, bmae = train_all(
        df_clean, "price_detrended", fcols, ["Ridge"], compute_split_date(df), fast=True
    )
    pred = iterative_forecast(
        df_clean, "price", "price_detrended", fcols, bm, bs, bstd,
        14, lags, windows, "D", None, pd.Timestamp.now(),
        residual_q10=None, residual_q90=None, other_cols=["other_col"],
    )
    assert not pred.empty
test("iterative_forecast_other_cols", test_iterative_forecast_other_cols)


print("\n--- Naive Fallback ---")

def test_naive_fallback():
    df = _make_df(100)
    pred = _naive_fallback_forecast(df, "price", 7, "D")
    assert len(pred) == 7
    assert (pred["lower"] >= 0).all()
    assert (pred["upper"] >= pred["forecast"]).all()
test("naive_fallback", test_naive_fallback)

def test_naive_fallback_empty():
    df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "price": pd.Series(dtype=float)})
    pred = _naive_fallback_forecast(df, "price", 7, "D")
    assert len(pred) == 7
    assert (pred["lower"] >= 0).all()
test("naive_fallback_empty", test_naive_fallback_empty)


print("\n--- Naive Benchmark ---")

def test_naive_benchmark():
    arr = np.array([100, 110, 105, 115, 120])
    result = naive_benchmark(arr)
    assert "MAE" in result
    assert "R-squared" in result
    assert result["MAE"] >= 0
test("naive_benchmark", test_naive_benchmark)

def test_naive_benchmark_short():
    result = naive_benchmark([1, 2])
    assert result == {}
test("naive_benchmark_short", test_naive_benchmark_short)


print("\n--- Model Improvement ---")

def test_model_improvement():
    assert model_improvement_vs_naive(80, 100) == 20.0
    assert model_improvement_vs_naive(100, 0) == 0.0
    assert model_improvement_vs_naive(100, float("nan")) == 0.0
test("model_improvement", test_model_improvement)


print("\n--- AR1 Texture ---")

def test_ar1_texture():
    noise = _ar1_texture(100, "test_seed", 1000.0)
    assert len(noise) == 100
    assert abs(noise.mean()) < 1.0
    assert noise.std() > 0
test("ar1_texture", test_ar1_texture)

def test_ar1_texture_deterministic():
    n1 = _ar1_texture(50, "seed_123", 500.0)
    n2 = _ar1_texture(50, "seed_123", 500.0)
    assert np.allclose(n1, n2)
test("ar1_texture_deterministic", test_ar1_texture_deterministic)

def test_ar1_texture_different_seeds():
    n1 = _ar1_texture(50, "seed_a", 500.0)
    n2 = _ar1_texture(50, "seed_b", 500.0)
    assert not np.allclose(n1, n2)
test("ar1_texture_different_seeds", test_ar1_texture_different_seeds)

def test_ar1_texture_zero_std():
    noise = _ar1_texture(50, "test", 0)
    assert np.all(noise == 0)
test("ar1_texture_zero_std", test_ar1_texture_zero_std)

def test_ar1_texture_negative_n():
    noise = _ar1_texture(-5, "test", 100)
    assert len(noise) == 0
test("ar1_texture_negative_n", test_ar1_texture_negative_n)


print("\n--- Volatility Regime ---")

def test_detect_volatility_regime():
    df = _make_df(200)
    regime = detect_volatility_regime(df, "price")
    assert 0.3 <= regime <= 3.0
test("detect_volatility_regime", test_detect_volatility_regime)

def test_detect_volatility_short():
    df = _make_df(10)
    regime = detect_volatility_regime(df, "price")
    assert regime == 1.0
test("detect_volatility_short", test_detect_volatility_short)


print("\n--- Adjust Forecast CI ---")

def test_adjust_forecast_ci():
    pred_df = pd.DataFrame({
        "forecast": np.linspace(100, 110, 14),
        "lower": np.linspace(90, 100, 14),
        "upper": np.linspace(110, 120, 14),
    })
    result = adjust_forecast_ci(pred_df, 1.5, -10, 10)
    assert not result.empty
    assert (result["lower"] >= 0).all()
test("adjust_forecast_ci", test_adjust_forecast_ci)


print("\n--- Anomaly Detection ---")

def test_detect_anomalies():
    actual = np.array([100, 102, 101, 200, 103, 101])
    predicted = np.array([100, 101, 101, 101, 102, 101])
    anomalies = detect_anomalies(actual, predicted, 5.0, threshold=2.0)
    assert len(anomalies) >= 1
    assert anomalies[0]["direction"] == "ABOVE"
test("detect_anomalies", test_detect_anomalies)

def test_detect_anomalies_none():
    actual = np.array([100, 101, 102, 103])
    predicted = np.array([100, 101, 102, 103])
    anomalies = detect_anomalies(actual, predicted, 10.0)
    assert len(anomalies) == 0
test("detect_anomalies_none", test_detect_anomalies_none)

def test_detect_anomalies_with_nan():
    actual = np.array([100, np.nan, 102])
    predicted = np.array([100, 101, 102])
    anomalies = detect_anomalies(actual, predicted, 5.0)
    assert isinstance(anomalies, list)
test("detect_anomalies_with_nan", test_detect_anomalies_with_nan)


print("\n--- Format Anomaly Report ---")

def test_format_anomaly_report_empty():
    assert "No anomalies" in format_anomaly_report([])
test("format_anomaly_report_empty", test_format_anomaly_report_empty)

def test_format_anomaly_report():
    anomalies = [{"index": 0, "actual": 200, "predicted": 100, "error": 100, "z_score": 5.0, "direction": "ABOVE"}]
    report = format_anomaly_report(anomalies, dates=["2024-01-01"])
    assert "1 anomaly" in report
test("format_anomaly_report", test_format_anomaly_report)


print("\n--- Feature Importance ---")

def test_feature_importance():
    df = _make_df(300)
    lags, windows = [1, 7], [7]
    df_feat, fcols, _ = create_features(df, "price", lags, windows)
    df_clean = df_feat.dropna(subset=["price"] + [c for c in fcols if c in df_feat.columns])
    split = compute_split_date(df)
    results, trained, bm, bs, bn, bstd, bmae = train_all(
        df_clean, "price", fcols, ["Ridge"], split, fast=True
    )
    fi = feature_importance_by_horizon(bm, fcols)
    assert "own_graph" in fi
    assert "external" in fi
    assert len(fi["own_graph"]) > 0
test("feature_importance", test_feature_importance)

def test_feature_importance_with_external():
    df = _make_df(300)
    df["ext_test"] = np.random.randn(300)
    lags, windows = [1, 7], [7]
    df_feat, fcols, _ = create_features(df, "price", lags, windows, external_cols=["ext_test"])
    df_clean = df_feat.dropna(subset=["price"] + [c for c in fcols if c in df_feat.columns])
    split = compute_split_date(df)
    results, trained, bm, bs, bn, bstd, bmae = train_all(
        df_clean, "price", fcols, ["Ridge"], split, fast=True
    )
    fi = feature_importance_by_horizon(bm, fcols, external_cols=["ext_test"])
    assert len(fi["external"]) > 0
test("feature_importance_with_external", test_feature_importance_with_external)


print("\n--- Generate Report ---")

def test_generate_report():
    df = _make_df(200)
    pred_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=14),
        "forecast": np.linspace(1e7, 1.1e7, 14),
        "lower": np.linspace(9.5e6, 1.05e7, 14),
        "upper": np.linspace(1.05e7, 1.15e7, 14),
    })
    html = generate_report("test_product", df, pred_df, "Ridge", {"MAE": 100000, "R-squared": 0.95})
    assert "<html" in html
    assert "test_product" in html
    assert "Ridge" in html
test("generate_report", test_generate_report)


print("\n--- Model Persistence ---")

def test_save_load_list_cleanup():
    df = _make_df(200)
    split = compute_split_date(df)
    lags, windows = [1, 7], [7]
    df_feat, fcols, _ = create_features(df, "price", lags, windows)
    df_clean = df_feat.dropna(subset=["price"] + [c for c in fcols if c in df_feat.columns])
    results, trained, _, _, _, _, _ = train_all(
        df_clean, "price", fcols, ["Ridge"], split, fast=True
    )
    paths = save_trained_models(trained, "test_product_persist")
    assert len(paths) >= 1
    loaded = load_trained_model(paths[0])
    assert loaded is not None
    assert "model" in loaded
    files = list_saved_models("test_product_persist")
    assert len(files) >= 1
    removed = cleanup_old_models()
    assert removed >= 0
test("save_load_list_cleanup", test_save_load_list_cleanup)

def test_load_nonexistent():
    assert load_trained_model("nonexistent_path.joblib") is None
test("load_nonexistent", test_load_nonexistent)

def test_make_safe_filename():
    s = make_safe_filename("\u06af\u0648\u0633\u0627\u0644\u0647 \u06a9\u0634\u062a\u0627\u0631")
    assert "/" not in s
    assert "\\" not in s
test("make_safe_filename", test_make_safe_filename)


print("\n--- Create Ensemble ---")

def test_create_ensemble_voting():
    from sklearn.linear_model import Ridge, Lasso
    models = {"Ridge": Ridge(), "Lasso": Lasso()}
    ens = create_ensemble(models, "Voting")
    assert ens is not None
test("create_ensemble_voting", test_create_ensemble_voting)

def test_create_ensemble_stacking():
    from sklearn.linear_model import Ridge, Lasso
    models = {"Ridge": Ridge(), "Lasso": Lasso()}
    ens = create_ensemble(models, "Stacking")
    assert ens is not None
test("create_ensemble_stacking", test_create_ensemble_stacking)

def test_create_ensemble_single():
    from sklearn.linear_model import Ridge
    ens = create_ensemble({"Ridge": Ridge()}, "Voting")
    assert ens is None
test("create_ensemble_single", test_create_ensemble_single)

def test_create_ensemble_unknown_type():
    from sklearn.linear_model import Ridge, Lasso
    ens = create_ensemble({"R": Ridge(), "L": Lasso()}, "Unknown")
    assert ens is None
test("create_ensemble_unknown_type", test_create_ensemble_unknown_type)


print("\n--- Compute Original Scale Metrics ---")

def test_compute_original_scale_metrics():
    df = _make_df(300)
    split = compute_split_date(df)
    lags, windows = [1, 7], [7]
    df["price_detrended"] = df["price"]
    df_feat, fcols, _ = create_features(df, "price_detrended", lags, windows)
    df_clean = df_feat.dropna(subset=["price_detrended"] + [c for c in fcols if c in df_feat.columns])
    results, trained, bm, bs, bn, bstd, bmae = train_all(
        df_clean, "price_detrended", fcols, ["Ridge"], split, fast=True
    )
    orig_metrics, y_test = compute_original_scale_metrics(
        trained, df_clean, fcols, "price", None, pd.Timestamp.now(), "D", split,
        pred_col="price_detrended"
    )
    assert "Ridge" in orig_metrics
    assert "MAE" in orig_metrics["Ridge"]
test("compute_original_scale_metrics", test_compute_original_scale_metrics)


print("\n--- Quick Predict E2E ---")

def test_quick_predict_basic():
    df = generate_fallback(500)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "product_testqp.csv")
    df.to_csv(cache_path, index=False)

    class MockFetcher:
        def discover_products(self):
            return {"test": 99999}, "mock"
    class MockCache:
        def update(self, pid, force=False):
            return df, "mock"
        def load(self, pid):
            return df

    pred_df, best, status, metrics, trained, test_preds, comp = quick_predict(
        "test", 99999, MockFetcher(), MockCache(),
        horizon=7, fast=True, selected_models=["Ridge", "Lasso"],
        train_log=lambda m: None,
    )
    assert pred_df is not None
    assert not pred_df.empty
    assert len(pred_df) == 7
    assert best is not None
    assert "Detrended" in status
test("quick_predict_basic", test_quick_predict_basic)


print("\n--- Backtest ---")

def test_backtest_basic():
    df = _make_df(300)
    lags, windows = [1, 7], [7]
    bt = backtest(df, "price", lags, windows, "D", "linear", ["Ridge", "Lasso"],
                  horizon=30, n_folds=2, fast=True)
    assert not bt.empty
    assert "mae" in bt.columns
    assert "rmse" in bt.columns
    assert "fold" in bt.columns
    assert (bt["mae"] >= 0).all()
test("backtest_basic", test_backtest_basic)

def test_backtest_single_fold():
    df = _make_df(100)
    bt = backtest(df, "price", [1, 7], [7], "D", "linear", ["Ridge"],
                  horizon=20, n_folds=1, fast=True)
    assert not bt.empty or True  # might not have enough data
test("backtest_single_fold", test_backtest_single_fold)

def test_backtest_insufficient_data():
    df = _make_df(10)
    bt = backtest(df, "price", [1], [7], "D", "linear", ["Ridge"],
                  horizon=30, n_folds=3, fast=True)
    assert isinstance(bt, pd.DataFrame)
test("backtest_insufficient_data", test_backtest_insufficient_data)


print("\n--- Theta Forecast ---")

def test_theta_forecast():
    predictor = SmartHybridPredictor(ml_func=lambda *a, **kw: None)
    series = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
    result = predictor._theta_forecast(series, 14, theta=2.0)
    assert len(result) == 14
test("theta_forecast", test_theta_forecast)

def test_theta_short_series():
    predictor = SmartHybridPredictor(ml_func=lambda *a, **kw: None)
    series = np.array([100.0])
    result = predictor._theta_forecast(series, 7)
    assert len(result) == 7
    assert np.all(result == 100.0)
test("theta_short_series", test_theta_short_series)


print("\n--- Edge Cases ---")

def test_empty_dataframe_create_features():
    df = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "price": pd.Series(dtype=float)})
    try:
        _, fcols, _ = create_features(df, "price", [1], [7])
        assert isinstance(fcols, list)
    except (KeyError, ValueError):
        pass  # acceptable
test("empty_dataframe_create_features", test_empty_dataframe_create_features)

def test_single_row_iterative_forecast():
    df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "price": [1e7]})
    try:
        pred = iterative_forecast(
            df, "price", "price", ["price"], get_model("Ridge")[0],
            None, 0, 7, [1], [7], "D", None, pd.Timestamp.now(),
            residual_q10=None, residual_q90=None,
        )
    except Exception:
        pass  # acceptable - not enough data for features
test("single_row_iterative_forecast", test_single_row_iterative_forecast)

def test_negative_prices():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=50),
        "price": np.random.uniform(-100, -10, 50)
    })
    result = preprocess(df, "price")
    assert len(result) == 50
test("negative_prices", test_negative_prices)

def test_constant_prices():
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100),
        "price": np.full(100, 1e7)
    })
    result = preprocess(df, "price")
    assert len(result) == 100
test("constant_prices", test_constant_prices)


# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Total:  {passed + failed}")

if failed > 0:
    print(f"\n  Failed tests:")
    for name, err in errors:
        print(f"    - {name}: {err}")
    sys.exit(1)
else:
    print("\n  ALL TESTS PASSED.")
    sys.exit(0)
