"""Chronological model evaluation, recursive forecasting and complete model packages."""

from __future__ import annotations

from pathlib import Path
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import logging
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import (
    RandomForestRegressor,
    VotingRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("itpnews.model_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

try:
    import joblib

    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

if TYPE_CHECKING:
    pass

MODELS_DIR = str(Path(__file__).resolve().parent / "data" / "models")
FORECASTS_DIR = str(Path(__file__).resolve().parent / "data" / "forecasts")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FORECASTS_DIR, exist_ok=True)

# 1. CONSTANTS

DEFAULT_MODELS = ["Lasso", "Ridge", "ElasticNet", "Random Forest"]
# XGBoost and LightGBM will be added dynamically if available
DEFAULT_DETREND = "linear"
MIN_ML_WEIGHT = 0.30

# Human-readable labels for external features (matches data_pipeline)
EXTERNAL_LABELS = {
    "ext_usd_irr_free": "USD/IRR (Free)",
    "ext_gold_price": "Gold (18k/gram)",
    "ext_eur_irr": "EUR/IRR",
    "ext_try_irr": "TRY/IRR",
    "ext_cad_irr": "CAD/IRR",
    "ext_gbp_irr": "GBP/IRR",
}


# 2. SPLIT UTILITY


def is_log_trend(coeffs):
    return isinstance(coeffs, dict) and coeffs.get("method") == "log"


def future_periods(last_date, horizon, freq="D"):
    if type(horizon) is not int or not 1 <= horizon <= 365:
        raise ValueError("Forecast horizon must be 1 to 365 periods.")
    if freq not in {"D", "W", "M"}:
        raise ValueError("Frequency must be D, W, or M.")
    rule = {"D": "D", "W": "W-SUN", "M": "ME"}[freq]
    start = pd.Timestamp(last_date) + pd.tseries.frequencies.to_offset(rule)
    return pd.date_range(start=start, periods=horizon, freq=rule)


def compute_split_date(
    df: pd.DataFrame, date_col: str = "date", train_frac: float = 0.8
) -> pd.Timestamp:
    """Compute the split date for train/test. Always uses the CANONICAL split."""
    dates = pd.to_datetime(df[date_col]).sort_values().reset_index(drop=True)
    n = len(dates)
    if n < 2:
        return dates.iloc[-1] if n else pd.Timestamp.now()
    idx = max(1, min(n - 1, int(n * train_frac)))
    return dates.iloc[idx]


def _resolve_col(df: pd.DataFrame, key: str) -> str:
    """Find actual column name matching key (NFC normalized)."""
    key_n = unicodedata.normalize("NFC", str(key))
    for c in df.columns:
        if unicodedata.normalize("NFC", str(c)) == key_n:
            return c
    # If not found, return first non-date numeric column
    for c in df.columns:
        if c != "date" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return key


# 3. MODELS


def get_model(name: str):
    """Return (model_instance, param_grid) for a given model name."""

    if name == "Lasso":
        return Lasso(
            max_iter=10000, random_state=42, tol=1e-3, selection="random", alpha=0.01
        ), {"alpha": [0.001, 0.01, 0.1]}
    if name == "Ridge":
        return Ridge(random_state=42, max_iter=10000, tol=1e-3), {
            "alpha": [0.1, 1.0, 10.0]
        }
    if name == "ElasticNet":
        return ElasticNet(
            max_iter=10000, random_state=42, tol=1e-3, selection="random"
        ), {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.3, 0.5, 0.7]}
    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        ), {"n_estimators": [100, 200], "max_depth": [8, 12, 15]}
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42
        ), {"n_estimators": [100, 200], "learning_rate": [0.03, 0.05, 0.1]}
    if name == "XGBoost" and XGB_AVAILABLE:
        return xgb.XGBRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        ), {"n_estimators": [100, 200], "max_depth": [4, 6, 8]}
    if name == "LightGBM" and LGBM_AVAILABLE:
        return lgb.LGBMRegressor(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        ), {"n_estimators": [100, 200], "max_depth": [4, 6, 8]}
    raise ValueError(f"Unknown or unavailable model: {name}")


def create_ensemble(models_dict: dict, ensemble_type: str = "Voting"):
    """Create a Voting or Stacking ensemble from trained models."""
    estimators = [
        (re.sub(r"\W+", "_", name), model) for name, model in models_dict.items()
    ]
    if len(estimators) < 2:
        return None
    if ensemble_type == "Voting":
        return VotingRegressor(estimators=estimators)
    elif ensemble_type == "Stacking":
        raise ValueError(
            "Stacking is disabled: ordinary K-fold stacking leaks future observations. Choose Voting."
        )
    return None


# 4. TRAINING — ALL MODELS, R² FILTER


def train_model(
    X_train, y_train, X_test, y_test, model, params=None, fast=False
) -> dict:
    """
    Train a single model with scaling and clipping.
    Returns: {model, scaler, mae, rmse, r2, mape, std_res, residual_q10, residual_q90}
    """

    scaler = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
    )
    X_train_s = scaler.fit_transform(np.asarray(X_train, dtype=float))
    X_test_s = scaler.transform(np.asarray(X_test, dtype=float))
    model = TransformedTargetRegressor(regressor=model, transformer=StandardScaler())
    model.fit(X_train_s, np.asarray(y_train, dtype=float))
    preds = model.predict(X_test_s)

    residuals = y_test - preds
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    try:
        r2 = float(r2_score(y_test, preds))
    except Exception:
        r2 = -999.0
    try:
        mape = float(mean_absolute_percentage_error(y_test, preds)) * 100
    except Exception:
        mape = float("nan")
    std_res = float(np.std(residuals)) if len(residuals) else 0.0
    q10 = (
        float(np.percentile(residuals, 10)) if len(residuals) > 10 else -1.28 * std_res
    )
    q90 = float(np.percentile(residuals, 90)) if len(residuals) > 10 else 1.28 * std_res

    return {
        "model": model,
        "scaler": scaler,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "std_res": std_res,
        "residual_q10": q10,
        "residual_q90": q90,
    }


def train_all(
    df,
    detrended_target,
    feature_cols,
    selected_models,
    split_date,
    use_ensemble=False,
    ensemble_type="Voting",
    fast=True,
    progress=None,
    train_log=None,
):
    """
    Train ALL selected models, apply R² filter, return best model.
    trained[name] = (model, scaler, mae, r2, std_res, q10, q90)
    """

    def _log(msg):
        if train_log is not None:
            if callable(train_log):
                train_log(msg)
            elif isinstance(train_log, list):
                train_log.append(msg)
            elif hasattr(train_log, "append"):
                train_log.append(msg)
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"))
        return msg

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    X = df[feature_cols].values
    y = df[detrended_target].values
    dates = pd.to_datetime(df["date"]).values
    mask = np.isfinite(y)
    X, y, dates = X[mask], y[mask], dates[mask]

    if len(X) < 30:
        _log("Warning: Less than 30 samples available for training")
        return {}, {}, None, None, None, 0, 0

    train_bool = dates < np.datetime64(pd.to_datetime(split_date))
    if train_bool.sum() < 10 or (~train_bool).sum() < 5:
        # Fallback if split yields too few samples
        raise ValueError(
            "Not enough rows on each side of the chosen chronological split."
        )

    X_train, X_test = X[train_bool], X[~train_bool]
    y_train, y_test = y[train_bool], y[~train_bool]

    # Prepare model list
    if selected_models is None:
        selected_models = DEFAULT_MODELS.copy()
    results, trained = {}, {}
    best_mae = float("inf")
    best_model = best_scaler = best_name = None
    best_std = 0

    # Parallel training (fast mode)
    if fast and len(selected_models) > 1:

        def _train_wrapper(name):
            try:
                mo, pr = get_model(name)
                r = train_model(X_train, y_train, X_test, y_test, mo, pr, fast)
                return name, r, None
            except Exception as exc:
                return name, None, str(exc)

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(_train_wrapper, name): name for name in selected_models
            }
            for fut in as_completed(futures):
                name, r, err = fut.result()
                if err:
                    _log(f"⚠ {name} failed: {err}")
                    continue
                results[name] = {
                    "MAE": r["mae"],
                    "RMSE": r["rmse"],
                    "R-squared": r["r2"],
                    "MAPE%": r["mape"],
                }
                trained[name] = (
                    r["model"],
                    r["scaler"],
                    r["mae"],
                    r["r2"],
                    r["std_res"],
                    r["residual_q10"],
                    r["residual_q90"],
                )
                if r["mae"] < best_mae and r["r2"] > 0:
                    best_mae, best_model, best_scaler, best_name, best_std = (
                        r["mae"],
                        r["model"],
                        r["scaler"],
                        name,
                        r["std_res"],
                    )
                _log(f"{name}: MAE={r['mae']:,.0f}, R2={r['r2']:.3f}")
    else:
        for name in selected_models:
            try:
                mo, pr = get_model(name)
                r = train_model(X_train, y_train, X_test, y_test, mo, pr, fast)
                results[name] = {
                    "MAE": r["mae"],
                    "RMSE": r["rmse"],
                    "R-squared": r["r2"],
                    "MAPE%": r["mape"],
                }
                trained[name] = (
                    r["model"],
                    r["scaler"],
                    r["mae"],
                    r["r2"],
                    r["std_res"],
                    r["residual_q10"],
                    r["residual_q90"],
                )
                if r["mae"] < best_mae and r["r2"] > 0:
                    best_mae, best_model, best_scaler, best_name, best_std = (
                        r["mae"],
                        r["model"],
                        r["scaler"],
                        name,
                        r["std_res"],
                    )
                _log(f"{name}: MAE={r['mae']:,.0f}, R2={r['r2']:.3f}")
            except Exception as exc:
                _log(f"⚠ {name} failed: {exc}")

    # ═══ Fallback: if ALL models have R² ≤ 0, pick best MAE ═══
    if best_model is None and trained:
        best_mae_fb = min(trained.items(), key=lambda kv: kv[1][2])
        name_fb = best_mae_fb[0]
        t_fb = best_mae_fb[1]
        best_model, best_scaler, best_mae, best_name, best_std = (
            t_fb[0],
            t_fb[1],
            t_fb[2],
            name_fb,
            t_fb[4],
        )
        _log(
            f"⚠️ All models had R² ≤ 0 → Fallback (best MAE): {name_fb} (MAE={best_mae:,.0f})"
        )

    # Ensemble (optional)
    if use_ensemble and len(trained) >= 2:
        try:
            scaler_ens = make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
            )
            X_train_s = scaler_ens.fit_transform(X_train)
            ensemble = create_ensemble(
                {name: t[0] for name, t in trained.items()}, ensemble_type
            )
            if ensemble:
                ensemble.fit(X_train_s, y_train)
                preds = ensemble.predict(scaler_ens.transform(X_test))
                mae = float(mean_absolute_error(y_test, preds))
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                r2 = float(r2_score(y_test, preds))
                mape = float(mean_absolute_percentage_error(y_test, preds)) * 100
                std_res = float(np.std(y_test - preds))
                resids = y_test - preds
                eq10 = (
                    float(np.percentile(resids, 10))
                    if len(resids) > 10
                    else -1.28 * std_res
                )
                eq90 = (
                    float(np.percentile(resids, 90))
                    if len(resids) > 10
                    else 1.28 * std_res
                )
                name = f"{ensemble_type} Ensemble"
                results[name] = {
                    "MAE": mae,
                    "RMSE": rmse,
                    "R-squared": r2,
                    "MAPE%": mape,
                }
                trained[name] = (ensemble, scaler_ens, mae, r2, std_res, eq10, eq90)
                if mae < best_mae and r2 > 0:
                    best_mae, best_model, best_scaler, best_name, best_std = (
                        mae,
                        ensemble,
                        scaler_ens,
                        name,
                        std_res,
                    )
                _log(f"{name}: MAE={mae:,.0f}, R2={r2:.3f}")
        except Exception as e:
            _log(f"Ensemble training failed: {e}")

    return results, trained, best_model, best_scaler, best_name, best_std, best_mae


# 5. SMART FEATURE SELECTION — Greedy, >5% Improvement


def smart_feature_selection(
    df,
    detrended_target,
    feature_cols,
    selected_models,
    split_date,
    train_log=None,
    max_workers=4,
):
    """
    FAST parallel smart feature selection.
    SAME logic as v13 — tests each feature, keeps if >5% improvement.
    Uses parallel execution + correlation pre-filter for speed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _log(msg):
        if train_log is not None:
            if callable(train_log):
                train_log(msg)
            elif isinstance(train_log, list):
                train_log.append(msg)
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"))

    df = df[pd.to_datetime(df["date"]) < pd.Timestamp(split_date)].copy()
    if len(df) < 50:
        return [c for c in feature_cols if not c.startswith("ext_")]
    split_date = compute_split_date(df)
    # Step 1: Baseline (product own-graph)
    base_cols = [
        c
        for c in feature_cols
        if not c.startswith("ext_feed_") and not c.startswith("ext_")
    ]
    if not base_cols:
        _log("⚠️ No base features found; returning all.")
        return feature_cols

    _, _, _, _, _, _, baseline_mae = train_all(
        df,
        detrended_target,
        base_cols,
        selected_models,
        split_date=split_date,
        use_ensemble=False,
        fast=True,
        train_log=[],
    )
    if baseline_mae is None or np.isinf(baseline_mae):
        _log("⚠️ Baseline training failed; returning all features.")
        return feature_cols

    _log(f"📊 Baseline MAE (product own-graph): {baseline_mae:,.0f}")

    # Step 2: Get candidates
    all_candidates = [c for c in feature_cols if c.startswith("ext_")]

    if not all_candidates:
        return base_cols

    # Step 3: Quick correlation filter (FAST, no training)
    filtered = []
    for col in all_candidates:
        if col in df.columns:
            corr = abs(df[[detrended_target, col]].corr().iloc[0, 1])
            if not np.isnan(corr) and corr > 0.05:  # At least some relationship
                filtered.append(col)

    _log(f"📊 {len(filtered)}/{len(all_candidates)} passed correlation filter")

    if not filtered:
        _log("📊 No features passed correlation filter → using product only.")
        return base_cols

    # Step 4: Test features in PARALLEL
    _log(f"🧪 Testing {len(filtered)} features in parallel ({max_workers} workers)...")

    def test_feature(col):
        test_cols = base_cols + [col]
        try:
            _, _, _, _, _, _, mae = train_all(
                df,
                detrended_target,
                test_cols,
                selected_models,
                split_date=split_date,
                use_ensemble=False,
                fast=True,
                train_log=[],
            )
            return col, mae
        except Exception:
            return col, float("inf")

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(test_feature, col): col for col in filtered}
        for future in as_completed(futures):
            col, mae = future.result()
            results[col] = mae

    # Step 5: Keep >5% improvement
    kept = []
    for col in filtered:
        mae = results.get(col, float("inf"))
        if mae is not None and mae < baseline_mae * 0.95:
            kept.append(col)
            improvement = (1 - mae / baseline_mae) * 100
            _log(f"  ✅ {col} helps (+{improvement:.1f}%) → KEEP")
        else:
            _log(f"  ❌ {col} hurts or neutral → REMOVE")

    final_cols = base_cols + kept
    _log(f"📊 Final: {len(base_cols)} product features + {len(kept)} kept features")
    return final_cols


# 6. METRICS & EVALUATION


def naive_benchmark(test_actuals) -> dict:
    """Naive forecast: repeat last value."""
    arr = np.asarray(test_actuals, dtype=float)
    if len(arr) < 3:
        return {}
    preds = np.roll(arr, 1)
    preds[0] = arr[0]
    resids = arr - preds
    std_r = float(np.std(resids))
    return {
        "MAE": float(mean_absolute_error(arr, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(arr, preds))),
        "R-squared": float(r2_score(arr, preds)),
        "MAPE%": float(mean_absolute_percentage_error(arr, preds)) * 100,
        "std_res": std_r,
        "residual_q10": float(np.percentile(resids, 10))
        if len(resids) > 10
        else -1.28 * std_r,
        "residual_q90": float(np.percentile(resids, 90))
        if len(resids) > 10
        else 1.28 * std_r,
    }


def model_improvement_vs_naive(model_mae, naive_mae) -> float:
    if not naive_mae or np.isnan(naive_mae) or naive_mae == 0:
        return 0.0
    return float((naive_mae - model_mae) / naive_mae * 100)


def compute_original_scale_metrics(
    trained_models,
    df_feat,
    feature_cols,
    target_name,
    coeffs,
    ref_date,
    freq,
    split_date,
    smooth_w=0,
    pred_col=None,
):
    """
    Compute metrics on original price scale (not detrended).
    Used for final model comparison.
    """
    from data_pipeline import apply_trend

    if pred_col is None:
        pred_col = target_name
    use_diff = coeffs is None and ref_date is None

    df = df_feat.copy()
    df["date"] = pd.to_datetime(df["date"])
    X = df[feature_cols].values
    y_predicted = df[pred_col].values
    dates = pd.to_datetime(df["date"]).values
    mask = np.isfinite(y_predicted)
    X, y_predicted, dates = X[mask], y_predicted[mask], dates[mask]

    train_bool = dates < np.datetime64(pd.to_datetime(split_date))
    X_test, test_dates = X[~train_bool], dates[~train_bool]
    if len(X_test) < 2:
        return {}, np.array([])

    train_dates = dates[train_bool]
    # Use unique dates to avoid duplicate index
    df_unique = df.drop_duplicates("date", keep="last")
    orig_prices = df_unique.set_index("date")[target_name]
    last_train_price = (
        orig_prices.get(pd.Timestamp(train_dates[-1]), np.nan)
        if len(train_dates)
        else np.nan
    )
    y_test_original = np.array(
        [orig_prices.get(pd.Timestamp(d), np.nan) for d in test_dates]
    )
    valid = ~np.isnan(y_test_original)
    X_test, y_test_original, test_dates = (
        X_test[valid],
        y_test_original[valid],
        test_dates[valid],
    )

    y_test_smooth, smooth_anchor = None, last_train_price
    if use_diff and smooth_w > 0 and "smooth" in df.columns:
        smooth_prices = df_unique.set_index("date")["smooth"]
        y_test_smooth = np.array(
            [smooth_prices.get(pd.Timestamp(d), np.nan) for d in test_dates]
        )
        smooth_anchor = (
            smooth_prices.get(pd.Timestamp(train_dates[-1]), last_train_price)
            if len(train_dates)
            else last_train_price
        )

    trend_test = apply_trend(test_dates, coeffs, ref_date, freq)

    original_metrics = {}
    for model_name, tup in trained_models.items():
        model, scaler_m = tup[0], tup[1]
        pred_model = np.nan_to_num(
            model.predict(scaler_m.transform(X_test)), nan=0.0, posinf=0.0, neginf=0.0
        )
        # Reconstruct original scale predictions
        if use_diff and smooth_w > 0 and y_test_smooth is not None:
            pred_original = np.roll(y_test_smooth, 1)
            pred_original[0] = (
                smooth_anchor if smooth_anchor is not None else y_test_smooth[0]
            )
            pred_original += pred_model
            actual_ref = y_test_smooth
        elif use_diff:
            anchors = (
                df_unique.set_index("date")[target_name]
                .shift(1)
                .reindex(pd.to_datetime(test_dates))
                .to_numpy()
            )
            pred_original = anchors + pred_model
            actual_ref = y_test_original
        else:
            pred_original = pred_model + trend_test
            if is_log_trend(coeffs):
                pred_original = np.exp(np.clip(pred_original, -700, 700))
            actual_ref = y_test_original

        # Clip to sensible range
        pred_original = np.clip(pred_original, 0, None)

        score_mae = mean_absolute_error(actual_ref, pred_original)
        score_rmse = np.sqrt(mean_squared_error(actual_ref, pred_original))
        try:
            score_r2 = r2_score(actual_ref, pred_original)
        except Exception:
            score_r2 = -999.0
        try:
            score_mape = mean_absolute_percentage_error(actual_ref, pred_original) * 100
        except Exception:
            score_mape = float("nan")
        score_res = actual_ref - pred_original
        score_std = float(np.std(score_res))
        score_q10 = (
            float(np.percentile(score_res, 10))
            if len(score_res) > 10
            else -1.28 * score_std
        )
        score_q90 = (
            float(np.percentile(score_res, 90))
            if len(score_res) > 10
            else 1.28 * score_std
        )

        original_metrics[model_name] = {
            "MAE": score_mae,
            "RMSE": score_rmse,
            "R-squared": score_r2,
            "MAPE%": score_mape,
            "std_res": score_std,
            "residual_q10": score_q10,
            "residual_q90": score_q90,
        }

    # Add naive benchmarks
    naive = naive_benchmark(y_test_original)
    if naive:
        original_metrics["Naive (last value)"] = naive
    if use_diff and smooth_w > 0 and y_test_smooth is not None:
        smooth_naive = naive_benchmark(y_test_smooth)
        if smooth_naive:
            original_metrics["Smooth Naive"] = smooth_naive

    return original_metrics, y_test_original


def feature_importance_by_horizon(model, feature_cols, external_cols=None) -> dict:
    """Return feature importance split into own-graph and external."""
    from data_pipeline import rank_features

    external_cols = set(external_cols or [])
    ranked = rank_features(model, feature_cols)
    total = sum(v for _, v in ranked) or 1.0
    own_graph, external = [], []
    for name, val in ranked:
        entry = {"feature": name, "importance": float(val), "share": float(val / total)}
        if name in external_cols or name.startswith("ext_"):
            external.append(entry)
        else:
            own_graph.append(entry)
    return {"own_graph": own_graph, "external": external}


# 7. Recursive forecasting


def iterative_forecast(
    df,
    target,
    detrended_target,
    feature_cols,
    model,
    scaler,
    std_res,
    horizon,
    lags,
    windows,
    freq,
    coeffs,
    ref_date,
    residual_q10,
    residual_q90,
    external_cols=None,
    smooth_w=None,
    noise_threshold=0.0,
    add_texture=False,
    other_cols=None,
):
    """
    Generate iterative multi-step forecast with confidence intervals.
    ═══ add_texture=False by default for natural-looking charts ═══
    """
    from data_pipeline import apply_trend, create_features

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    target = _resolve_col(df, target)
    detrended_target = unicodedata.normalize("NFC", detrended_target)
    use_diff = coeffs is None and ref_date is None

    # Build detrended target column for history
    if use_diff and smooth_w and smooth_w > 0:
        df["smooth"] = df[target].rolling(smooth_w, min_periods=1).mean()
        df[detrended_target] = df["smooth"].diff()
    elif use_diff:
        df[detrended_target] = df[target].diff()
    elif coeffs is not None and ref_date is not None:
        df[detrended_target] = (
            np.log(df[target]) if is_log_trend(coeffs) else df[target]
        ) - apply_trend(df["date"], coeffs, ref_date, freq)
    else:
        df[detrended_target] = df[target]

    # Preserve external columns for recursive feature construction
    keep_cols = (
        ["date", detrended_target, "gap_days"]
        + (external_cols or [])
        + (other_cols or [])
    )
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in df.columns]))
    hist = df[keep_cols].copy().reset_index(drop=True)

    future_dates = future_periods(hist["date"].max(), horizon, freq)

    # Carry last known external values forward
    last_external_vals = {}
    if external_cols or other_cols:
        for ec in dict.fromkeys((external_cols or []) + (other_cols or [])):
            if ec in df.columns:
                valid = df[ec].dropna()
                last_external_vals[ec] = valid.iloc[-1] if len(valid) else np.nan

    preds = []
    for fdate in future_dates:
        new_row = {
            "date": fdate,
            detrended_target: np.nan,
            "gap_days": (fdate - df["date"].max()).days,
        }
        for ec, val in last_external_vals.items():
            new_row[ec] = val
        temp = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        temp_feat, _, _ = create_features(
            temp,
            detrended_target,
            lags,
            windows,
            freq,
            other_cols=other_cols,
            external_cols=external_cols,
        )
        if temp_feat.empty:
            break
        for col in set(feature_cols) - set(temp_feat.columns):
            temp_feat[col] = np.nan
        X_step = temp_feat[feature_cols].iloc[-1:].values
        pred = float(model.predict(scaler.transform(X_step))[0])
        if not np.isfinite(pred):
            pred = 0.0
        preds.append({"date": fdate, "forecast_detrended": pred})
        # Append to history for next step
        hist_row = {
            "date": fdate,
            detrended_target: pred,
            "gap_days": (fdate - df["date"].max()).days,
        }
        for ec, val in last_external_vals.items():
            hist_row[ec] = val
        hist = pd.concat([hist, pd.DataFrame([hist_row])], ignore_index=True)

    if not preds:
        return pd.DataFrame()

    pred_df = pd.DataFrame(preds)

    # Reconstruct original scale
    if use_diff:
        if smooth_w and smooth_w > 0 and "smooth" in df.columns:
            anchor = (
                float(df["smooth"].dropna().iloc[-1])
                if not df.empty
                else float(df[target].iloc[-1])
            )
        else:
            anchor = float(df[target].iloc[-1]) if not df.empty else 0
        pred_df["forecast"] = anchor + np.cumsum(pred_df["forecast_detrended"].values)
    elif coeffs is not None and ref_date is not None:
        pred_df["forecast"] = pred_df["forecast_detrended"] + apply_trend(
            pred_df["date"], coeffs, ref_date, freq
        )
    else:
        pred_df["forecast"] = pred_df["forecast_detrended"]

    if is_log_trend(coeffs):
        pred_df["forecast"] = np.exp(np.clip(pred_df["forecast"], -700, 700))
    pred_df["forecast"] = pred_df["forecast"].clip(lower=0)

    # Confidence intervals
    cum = np.sqrt(np.arange(1, len(pred_df) + 1))
    if residual_q10 is not None and residual_q90 is not None:
        pred_df["lower"] = np.clip(pred_df["forecast"] + residual_q10 * cum, 0, None)
        pred_df["upper"] = pred_df["forecast"] + residual_q90 * cum
    else:
        cum_std = std_res * cum
        pred_df["lower"] = np.clip(pred_df["forecast"] - 1.28 * cum_std, 0, None)
        pred_df["upper"] = pred_df["forecast"] + 1.28 * cum_std

    # ═══ AR(1) Texture: ON by default for natural-looking charts ═══
    pred_df["lower"] = np.minimum(pred_df["lower"].clip(lower=0), pred_df["forecast"])
    pred_df["upper"] = np.maximum(pred_df["upper"], pred_df["forecast"])
    return pred_df[["date", "forecast", "lower", "upper"]]


def create_forecast_plot(df, target, pred_df, test_preds=None) -> go.Figure:
    """Create interactive Plotly chart with historical and forecast."""
    target = _resolve_col(df, target)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Historical & Forecast",
            "Price Distribution",
            "Forecast Trend",
            "Forecast Range",
        ),
        specs=[
            [{"secondary_y": False}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "box"}],
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df[target],
            mode="lines",
            name="Historical",
            line=dict(color="#10b981", width=2),
        ),
        row=1,
        col=1,
    )

    if test_preds is not None and not test_preds.empty:
        fig.add_trace(
            go.Scatter(
                x=test_preds["date"],
                y=test_preds["predicted"],
                mode="lines+markers",
                name="Test Prediction",
                line=dict(color="#f59e0b", width=1.5, dash="dot"),
                marker=dict(size=4),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=test_preds["date"],
                y=test_preds["actual"],
                mode="lines",
                name="Test Actual",
                line=dict(color="#10b981", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

    if not pred_df.empty:
        last_val = float(df[target].iloc[-1]) if not df.empty else 0.0
        last_date = df["date"].iloc[-1] if not df.empty else pred_df["date"].iloc[0]
        bd = pd.concat([pd.Series([last_date]), pred_df["date"]], ignore_index=True)
        bf = pd.concat([pd.Series([last_val]), pred_df["forecast"]], ignore_index=True)
        bu = pd.concat([pd.Series([last_val]), pred_df["upper"]], ignore_index=True)
        bl = pd.concat([pd.Series([last_val]), pred_df["lower"]], ignore_index=True)

        fig.add_trace(
            go.Scatter(
                x=bd,
                y=bf,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#ef4444", width=2),
                marker=dict(size=6),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=bd, y=bu, mode="lines", line=dict(width=0), showlegend=False),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=bd,
                y=bl,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(239,68,68,0.2)",
                name="80% CI",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=bd,
                y=bf,
                mode="lines+markers",
                name="Trend",
                line=dict(color="#ef4444", width=3),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Box(
                y=pred_df["forecast"],
                name="Forecast Range",
                marker_color="#ef4444",
                boxmean="sd",
            ),
            row=2,
            col=2,
        )

    fig.add_trace(
        go.Histogram(
            x=df[target], name="Distribution", marker_color="#10b981", nbinsx=30
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        height=700, template="plotly_dark", showlegend=True, hovermode="x unified"
    )
    return fig


def detect_volatility_regime(
    df: pd.DataFrame, target_col: str, window: int = 30
) -> float:
    """Return volatility ratio (recent / full)."""
    series = df[target_col].dropna()
    if len(series) < window * 2:
        return 1.0
    recent_std = series.tail(window).pct_change().dropna().std()
    full_std = series.pct_change().dropna().std()
    if not full_std or full_std == 0:
        return 1.0
    return float(np.clip(recent_std / full_std, 0.3, 3.0))


def adjust_forecast_ci(
    pred_df: pd.DataFrame, regime_ratio: float, base_q10, base_q90
) -> pd.DataFrame:
    """Adjust CI width based on volatility regime."""
    if pred_df.empty:
        return pred_df
    out = pred_df.copy()
    scale = np.sqrt(regime_ratio)
    cum = np.sqrt(np.arange(1, len(out) + 1))
    if "lower" in out.columns and "upper" in out.columns:
        out["lower"] = (out["forecast"] + base_q10 * scale * cum).clip(lower=0)
        out["upper"] = out["forecast"] + base_q90 * scale * cum
    return out


def detect_anomalies(actual, predicted, std_res, threshold: float = 3.0) -> list:
    """Detect anomalous steps in forecast vs actual."""
    actual, predicted = (
        np.asarray(actual, dtype=float),
        np.asarray(predicted, dtype=float),
    )
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) < 2 or std_res == 0:
        return []
    z = np.abs(actual - predicted) / max(std_res, 1)
    anomalies = []
    for i in range(len(actual)):
        if z[i] > threshold:
            anomalies.append(
                {
                    "index": i,
                    "actual": float(actual[i]),
                    "predicted": float(predicted[i]),
                    "error": float(actual[i] - predicted[i]),
                    "z_score": float(z[i]),
                    "direction": "ABOVE" if actual[i] > predicted[i] else "BELOW",
                }
            )
    return anomalies


def format_anomaly_report(anomalies, dates=None) -> str:
    if not anomalies:
        return "No anomalies detected."
    lines = [f"{len(anomalies)} anomaly(ies) found:"]
    for a in anomalies[:10]:
        d = (
            f" at {dates[a['index']]}"
            if dates is not None and a["index"] < len(dates)
            else ""
        )
        lines.append(
            f"  - {d}: actual={a['actual']:,.0f} vs pred={a['predicted']:,.0f} ({a['direction']}, z={a['z_score']:.1f} sigma)"
        )
    if len(anomalies) > 10:
        lines.append(f"  ... and {len(anomalies) - 10} more")
    return "\n".join(lines)


# 8. SMART HYBRID PREDICTOR


class SmartHybridPredictor:
    """Blends ML forecast with simple statistical methods (MA, Holt, Trend, Theta)."""

    def __init__(self, ml_func, test_window: int = 60):
        self.ml_func = ml_func
        self.test_window = test_window

    def _simple_methods(self) -> dict:
        return {
            "MovingAverage(7d)": lambda s, h: self._ma_forecast(s, h, 7),
            "MovingAverage(14d)": lambda s, h: self._ma_forecast(s, h, 14),
            "MovingAverage(30d)": lambda s, h: self._ma_forecast(s, h, 30),
            "Holt": self._holt_forecast,
            "LinearTrend": self._trend_forecast,
            "Theta": self._theta_forecast,
        }

    def _ma_forecast(self, series, horizon, window) -> np.ndarray:
        if len(series) < window:
            return np.full(horizon, series[-1])
        return np.full(horizon, np.mean(series[-window:]))

    def _holt_forecast(self, series, horizon, alpha=0.3, beta=0.1) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        n = len(series)
        if n < 4:
            return np.full(horizon, series[-1])
        level, trend = series[0], (series[1] - series[0] if n > 1 else 0)
        for t in range(1, n):
            prev_level = level
            level = alpha * series[t] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        return np.array([level + trend * (i + 1) for i in range(horizon)])

    def _trend_forecast(self, series, horizon) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        if len(series) < 2:
            return np.full(horizon, series[-1])
        coeffs = np.polyfit(np.arange(len(series)), series, 1)
        return np.polyval(coeffs, np.arange(len(series), len(series) + horizon))

    def _theta_forecast(self, series, horizon, theta=2.0) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        n = len(series)
        if n < 4:
            return np.full(horizon, series[-1])
        x = np.arange(n)
        slope, intercept, _, _, _ = stats.linregress(x, series)
        trend = intercept + slope * np.arange(n, n + horizon)
        avg_residual = np.mean((series - (intercept + slope * x))[-min(30, n) :])
        return trend + avg_residual * (1.0 / theta)

    def _evaluate_method(self, series, horizon, method_fn) -> float:
        """Walk-forward MAE for a simple method."""
        n = len(series)
        if n < self.test_window + horizon:
            return float("inf")
        errors = []
        for start in range(self.test_window, n - horizon + 1):
            train = series[:start]
            actual = series[start : start + horizon]
            if len(actual) < horizon:
                continue
            pred = method_fn(train, horizon)
            errors.append(np.mean(np.abs(pred - actual)))
        return np.mean(errors) if errors else float("inf")

    def predict(
        self,
        product_name,
        product_id,
        fetcher,
        cache,
        horizon=30,
        force_refresh=False,
        freq="D",
        fast=True,
        train_log=None,
    ):
        """
        Run ML forecast, evaluate simple methods, blend with ML weight >=30%.
        """
        log = train_log if isinstance(train_log, list) else []
        ml_result = self.ml_func(
            product_name,
            product_id,
            fetcher,
            cache,
            horizon=horizon,
            force_refresh=force_refresh,
            freq=freq,
            fast=fast,
            train_log=log,
        )

        delta_map = {
            "D": pd.Timedelta(days=1),
            "W": pd.Timedelta(weeks=1),
            "M": pd.DateOffset(months=1),
        }
        _delta = delta_map.get(freq, pd.Timedelta(days=1))

        # If ML failed, use best simple method directly
        if ml_result is None or ml_result[0] is None:
            df = cache.load(product_id) if cache is not None else pd.DataFrame()
            if df.empty:
                raise ValueError(
                    "Market data unavailable; no synthetic fallback is used."
                )
            series = (
                df["price"].values
                if "price" in df.columns
                else (df.iloc[:, 1].values if df.shape[1] > 1 else np.array([]))
            )
            if len(series) < 10:
                return ml_result
            methods = self._simple_methods()
            scores = {
                name: self._evaluate_method(series, horizon, fn)
                for name, fn in methods.items()
            }
            best_simple = min(scores, key=scores.get)
            pred = methods[best_simple](series, horizon)
            pred_df = pd.DataFrame(
                {
                    "date": future_periods(df["date"].max(), horizon, freq),
                    "forecast": pred,
                    "lower": pred * 0.9,
                    "upper": pred * 1.1,
                }
            )
            status = f"[hybrid fallback] ML failed, using {best_simple}"
            if isinstance(train_log, list):
                train_log.append(status)
            elif callable(train_log):
                train_log(status)
            return pred_df, best_simple, status, {}, {}, pd.DataFrame(), pd.DataFrame()

        (
            pred_df_ml,
            ml_best_name,
            ml_status,
            ml_metrics,
            ml_trained,
            ml_test_preds,
            ml_comp_df,
        ) = ml_result

        # Load historical data for simple methods evaluation
        df = cache.load(product_id) if cache is not None else pd.DataFrame()
        if df.empty or "price" not in df.columns:
            return ml_result
        from data_pipeline import preprocess

        df = preprocess(df, "price", freq)
        series = df.sort_values("date")["price"].dropna().values
        if len(series) < self.test_window + horizon:
            return ml_result

        # Evaluate simple methods
        methods = self._simple_methods()
        simple_scores = {
            name: self._evaluate_method(series, horizon, fn)
            for name, fn in methods.items()
        }
        ml_mae = (
            ml_metrics.get(ml_best_name, {}).get("MAE", float("inf"))
            if isinstance(ml_metrics, dict)
            else float("inf")
        )

        # Weighted blend: invert MAE for weights
        epsilon = 1e-6
        inv_scores = {k: 1.0 / (v + epsilon) for k, v in simple_scores.items()}
        total_inv = sum(inv_scores.values())
        if not np.isfinite(total_inv) or total_inv <= 0:
            return ml_result
        simple_weights = {k: v / total_inv for k, v in inv_scores.items()}
        best_simple_name = max(simple_weights, key=simple_weights.get)

        # Determine ML weight (boosted by R² when confident)
        r2_boost = 0.0

        # Direct MAE comparison: lower MAE → higher weight
        best_simple_mae = simple_scores.get(best_simple_name, float("inf"))
        ml_inv = 1.0 / max(ml_mae, epsilon)
        simple_inv = 1.0 / max(best_simple_mae, epsilon)
        ml_weight = max(ml_inv / (ml_inv + simple_inv), MIN_ML_WEIGHT)
        ml_weight = min(max(ml_weight + r2_boost, MIN_ML_WEIGHT), 0.95)
        simple_weight = 1.0 - ml_weight

        # Blend forecasts
        simple_pred = methods[best_simple_name](series, horizon)
        ml_pred = pred_df_ml["forecast"].values
        blended_pred = ml_weight * ml_pred + simple_weight * simple_pred
        blended_pred = np.maximum(blended_pred, 0)

        # Rebuild CI around blended forecast
        if not pred_df_ml.empty and "lower" in pred_df_ml.columns:
            shift = blended_pred - ml_pred
            lower = np.clip(pred_df_ml["lower"].values + shift, 0, None)
            upper = pred_df_ml["upper"].values + shift
        else:
            cum_std = (
                simple_scores.get(best_simple_name, 0)
                * 0.5
                * np.sqrt(np.arange(1, horizon + 1))
            )
            lower = np.clip(blended_pred - 1.28 * cum_std, 0, None)
            upper = blended_pred + 1.28 * cum_std

        pred_df = pd.DataFrame(
            {
                "date": future_periods(df["date"].max(), horizon, freq),
                "forecast": blended_pred,
                "lower": lower,
                "upper": upper,
            }
        )

        status = f"[hybrid] ML({ml_best_name}) {ml_weight:.0%} + {best_simple_name} {simple_weight:.0%}"
        if isinstance(train_log, list):
            train_log.append(status)
        elif callable(train_log):
            train_log(status)

        return (
            pred_df,
            ml_best_name,
            status,
            ml_metrics,
            ml_trained,
            ml_test_preds,
            ml_comp_df,
        )


# 9. QUICK PREDICT — Smart Selection + Volatility Suggestion


def _intelligent_fallback_forecast(df, target, horizon, freq):
    """Simple trend-based fallback when ML fails."""
    if target in df.columns and df[target].notna().any():
        series = df[target].dropna()
        last_val = series.iloc[-1]
        lookback = min(30, len(series))
        if len(series) > 5:
            recent = series.tail(lookback).values
            x = np.arange(len(recent))
            coeffs = np.polyfit(x, recent, 1)
            trend = coeffs[0]
        else:
            trend = 0
    else:
        last_val = 0.0
        trend = 0

    if "date" in df.columns and not df.empty:
        last_date = pd.to_datetime(df["date"]).max()
    else:
        last_date = pd.Timestamp.now()

    step = {
        "D": pd.Timedelta(days=1),
        "W": pd.Timedelta(weeks=1),
        "M": pd.DateOffset(months=1),
    }.get(freq, pd.Timedelta(days=1))
    dates = [last_date + step * (i + 1) for i in range(horizon)]
    forecast = [max(0, last_val + trend * (i + 1)) for i in range(horizon)]
    std = df[target].dropna().std() if target in df.columns else 0.0
    if std is None or np.isnan(std):
        std = 0.0
    expansion = np.sqrt(np.arange(1, horizon + 1))
    return pd.DataFrame(
        {
            "date": dates,
            "forecast": forecast,
            "lower": np.clip(
                [f - std * expansion[i] for i, f in enumerate(forecast)], 0, None
            ),
            "upper": [f + std * expansion[i] for i, f in enumerate(forecast)],
        }
    )


def quick_predict(
    product_name,
    product_id,
    fetcher,
    cache,
    horizon=30,
    force_refresh=False,
    lags=None,
    windows=None,
    selected_models=None,
    detrend_method=DEFAULT_DETREND,
    freq="D",
    fast=True,
    train_log=None,
    product_type="Auto-detect",
):
    """
    End-to-end prediction with smart feature selection.
    ═══ NEVER forces freq change, NEVER limits models ═══
    """
    from data_pipeline import (
        apply_trend,
        auto_params,
        create_features,
        detrend_series,
        merge_external_features,
        preprocess,
        select_features,
        fetch_related_prices,  # ← ADD THIS
    )

    def _log(msg):
        if train_log is not None:
            if callable(train_log):
                train_log(msg)
            elif isinstance(train_log, list):
                train_log.append(msg)
            elif hasattr(train_log, "append"):
                train_log.append(msg)
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"))
        return msg

    target = "price"

    # Fetch data
    try:
        raw_df, cache_status = cache.update(product_id, force=force_refresh)
    except Exception:
        raise ValueError(
            "Market data is unavailable. Retry later or upload a real CSV."
        )

    if raw_df is None or raw_df.empty:
        raise ValueError(
            "Market data is unavailable. Retry later or upload a real CSV."
        )

    df = preprocess(raw_df, target, freq=freq)
    if df.empty:
        return (
            None,
            None,
            "No data available after preprocessing.",
            {},
            {},
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # ═══ VOLATILITY DETECTION: SUGGEST only, NEVER force changes ═══
    price_range_ratio = df[target].max() / df[target].min()
    is_volatile = price_range_ratio > 10

    volatile_mode = False
    if product_type == "Auto-detect" and is_volatile:
        _log(f"🔄 Auto-detected volatile product (range: {price_range_ratio:.0f}x)")
        _log("💡 Suggestion: Use 'log' detrend method for better accuracy")
        _log("✅ Keeping all models and your selected frequency")
        # ═══ ONLY change detrend method if user hasn't overridden ═══

        volatile_mode = True
    elif product_type == "Volatile":
        _log("🔄 Volatile mode enabled by user")
        detrend_method = "log"
        volatile_mode = True
    else:
        _log(
            f"✅ Product stable (range: {price_range_ratio:.0f}x) — using standard mode"
        )

    # Auto params if not provided
    if lags is None or windows is None:
        auto_lags, auto_windows = auto_params(df, target, freq)
        lags = lags or auto_lags
        windows = windows or auto_windows

    # Merge external features (all 5 Iranian)
    df = merge_external_features(df, product_name, freq)
    # Merge related products
    df = fetch_related_prices(df, product_name, fetcher, cache, freq)

    external_cols_present = [c for c in df.columns if c.startswith("ext_")]

    split_date = compute_split_date(df, "date", train_frac=0.8)

    # Detrend
    detrended_target = f"{target}_detrended"
    detrend_result = detrend_series(
        df[df["date"] < split_date],
        target,
        freq=freq,
        method=detrend_method,
        split_date=split_date,
    )
    if isinstance(detrend_result, tuple) and len(detrend_result) == 3:
        coeffs, ref_date = None, None
        smooth_w = detrend_result[2]
    else:
        coeffs, ref_date = detrend_result
        smooth_w = None

    # Apply detrending
    if coeffs is None and smooth_w is not None:
        df["smooth"] = df[target].rolling(smooth_w, min_periods=1).mean()
        df[detrended_target] = df["smooth"].diff()
    elif coeffs is not None:
        df[detrended_target] = (
            np.log(df[target]) if is_log_trend(coeffs) else df[target]
        ) - apply_trend(df["date"], coeffs, ref_date, freq)
    elif ref_date is not None:
        df[detrended_target] = df[target]
    else:
        df[detrended_target] = df[target].diff()

    # Feature engineering
    other_cols = [
        c
        for c in df.columns
        if c not in [target, "date", detrended_target, "smooth", "is_holiday"]
        and np.issubdtype(df[c].dtype, np.number)
        and df[c].nunique() > 10
    ]

    df_feat, feature_cols, seasonal_cols = create_features(
        df,
        detrended_target,
        lags,
        windows,
        freq=freq,
        other_cols=other_cols,
        external_cols=external_cols_present or None,
        add_seasonal=True,
    )
    # Ensure lags are included
    for lag in lags:
        col = f"{detrended_target}_lag_{lag}"
        if col not in df_feat.columns:
            df_feat[col] = df_feat[detrended_target].shift(lag)
        feature_cols.append(col)
    feature_cols = list(
        dict.fromkeys([c for c in feature_cols if c in df_feat.columns])
    )

    # Initial correlation-based feature selection
    feature_cols = select_features(
        df_feat, feature_cols, split_date=split_date, max_corr=0.95
    )
    own_graph_cols = [
        c for c in feature_cols if not c.startswith("ext_") and c in df_feat.columns
    ]
    external_feature_cols = [
        c for c in feature_cols if c.startswith("ext_") and c in df_feat.columns
    ]

    # Only drop NaN for target + own-graph (lag/window padding)
    df_feat = df_feat.dropna(
        subset=[detrended_target] + [c for c in own_graph_cols if "_lag_" in c]
    )

    # Fill external NaN with 0 — preserves all rows without leaking data
    for col in external_feature_cols:
        if col in df_feat.columns and df_feat[col].isna().any():
            df_feat[col] = df_feat[col].ffill()

    # Prepare models
    if selected_models is None:
        selected_models = DEFAULT_MODELS.copy()
    # ═══ SMART FEATURE SELECTION ═══
    _log("🧠 Running smart feature selection...")
    final_feature_cols = smart_feature_selection(
        df_feat,
        detrended_target,
        feature_cols,
        selected_models,
        split_date=split_date,
        train_log=train_log,
    )

    # Final training with selected features
    results, trained, best_model, best_scaler, best_name, best_std, best_mae = (
        train_all(
            df_feat,
            detrended_target,
            final_feature_cols,
            selected_models,
            split_date=split_date,
            use_ensemble=True,
            ensemble_type="Voting",
            fast=fast,
            train_log=train_log,
        )
    )

    # If all models failed, fallback
    if best_model is None:
        _log("⚠️ No eligible model found; using intelligent fallback.")
        pred_df = _intelligent_fallback_forecast(df, target, horizon, freq)
        return (
            pred_df,
            "Intelligent Fallback",
            "Fallback used (no model had R² > 0)",
            {},
            {},
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # Compute original-scale metrics for comparison
    orig_metrics, _ = compute_original_scale_metrics(
        trained,
        df_feat,
        final_feature_cols,
        target,
        coeffs,
        ref_date,
        freq,
        split_date,
        smooth_w=smooth_w or 0,
        pred_col=detrended_target,
    )

    # Pick best model again based on original metrics (R² > 0)
    naive_key = (
        "Smooth Naive" if (coeffs is None and smooth_w) else "Naive (last value)"
    )
    naive_mae = orig_metrics.get(naive_key, {}).get("MAE", float("inf"))
    valid_models = {
        mk: mv
        for mk, mv in orig_metrics.items()
        if mk not in (naive_key, "Naive (last value)")
        and mv.get("R-squared", -999) > 0
        and mv.get("MAE", float("inf")) < naive_mae
    }
    if valid_models:
        best_name = min(valid_models, key=lambda n: valid_models[n]["MAE"])
        if best_name in trained:
            bt = trained[best_name]
            best_model, best_scaler, best_std, best_mae = (
                bt[0],
                bt[1],
                bt[4] if len(bt) > 4 else best_std,
                bt[2] if len(bt) > 2 else float("inf"),
            )

    uses_naive = best_name == "Intelligent Fallback"
    actual_best_name = best_name

    # Get residual stats
    best_metrics = orig_metrics.get(actual_best_name, {}) or {}
    best_std = (
        best_metrics.get("std_res", best_std) if best_metrics else (best_std or 0)
    )
    best_q10 = (
        best_metrics.get("residual_q10", -1.28 * (best_std or 0))
        if best_metrics
        else -1.28 * (best_std or 0)
    )
    best_q90 = (
        best_metrics.get("residual_q90", 1.28 * (best_std or 0))
        if best_metrics
        else 1.28 * (best_std or 0)
    )

    valid_fit = np.isfinite(df_feat[detrended_target].to_numpy(dtype=float))
    best_scaler = clone(best_scaler)
    full_x = best_scaler.fit_transform(
        df_feat.loc[valid_fit, final_feature_cols].to_numpy(dtype=float)
    )
    best_model = clone(best_model).fit(
        full_x, df_feat.loc[valid_fit, detrended_target].to_numpy(dtype=float)
    )
    # Generate forecast (add_texture=False for natural charts)
    try:
        pred_df = iterative_forecast(
            df_feat,
            target,
            detrended_target,
            final_feature_cols,
            best_model,
            best_scaler,
            best_std,
            horizon,
            lags,
            windows,
            freq,
            coeffs,
            ref_date,
            residual_q10=best_q10,
            residual_q90=best_q90,
            external_cols=external_cols_present or None,
            smooth_w=smooth_w or 0,
            other_cols=other_cols,
            add_texture=False,  # ═══ ON by default for visual ═══
        )
    except Exception as exc:
        _log(f"Forecast failed: {exc}; using fallback.")
        pred_df = _intelligent_fallback_forecast(df, target, horizon, freq)

    if pred_df.empty:
        return (
            None,
            None,
            "Forecast generation failed.",
            {},
            {},
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # Volatility regime adjustment (if needed)
    regime = detect_volatility_regime(raw_df, target)
    if abs(regime - 1.0) > 0.2:
        pred_df = adjust_forecast_ci(pred_df, regime, best_q10, best_q90)

    # Build test predictions (if available)
    test_preds = pd.DataFrame()
    try:
        if not uses_naive:
            best_model_obj = trained.get(actual_best_name, (None, None))[0]
            best_scaler_obj = trained.get(actual_best_name, (None, None))[1]
            if best_model_obj is not None and best_scaler_obj is not None:
                test_mask = df_feat["date"].isin(
                    df[df["date"] >= split_date]["date"].values
                )
                test_mask = test_mask & df_feat[final_feature_cols].notna().all(axis=1)
                X_test_full = df_feat.loc[test_mask, final_feature_cols].values
                test_dates_full = df_feat.loc[test_mask, "date"].values
                if len(X_test_full) > 0:
                    X_te_scaled = best_scaler_obj.transform(X_test_full)
                    pred_te_det = best_model_obj.predict(X_te_scaled)
                    if coeffs is None and smooth_w:
                        if "smooth" in df.columns:
                            last_train_price = (
                                float(
                                    df[df["date"] < split_date]["smooth"]
                                    .dropna()
                                    .iloc[-1]
                                )
                                if len(df[df["date"] < split_date]) > 0
                                else 0
                            )
                        else:
                            last_train_price = (
                                float(df[df["date"] < split_date][target].iloc[-1])
                                if len(df[df["date"] < split_date]) > 0
                                else 0
                            )
                        pred_te = last_train_price + np.cumsum(pred_te_det)
                    elif coeffs is None:
                        last_train_price = (
                            float(df[df["date"] < split_date][target].iloc[-1])
                            if len(df[df["date"] < split_date]) > 0
                            else 0
                        )
                        pred_te = last_train_price + np.cumsum(pred_te_det)
                    else:
                        trend_te = apply_trend(test_dates_full, coeffs, ref_date, freq)
                        pred_te = pred_te_det + trend_te
                    actual_te = (
                        df.set_index("date")
                        .reindex(pd.to_datetime(test_dates_full))[target]
                        .values
                    )
                    test_preds = pd.DataFrame(
                        {
                            "date": test_dates_full,
                            "actual": actual_te,
                            "predicted": np.clip(pred_te, 0, None),
                        }
                    )
    except Exception:
        pass

    # Build comparison DataFrame
    comp_data = []
    for name, tup in trained.items():
        mae = tup[2] if len(tup) > 2 else None
        r2 = tup[3] if len(tup) > 3 else None
        comp_data.append({"model": name, "mae": mae, "r2": r2})
    comp_df = (
        pd.DataFrame(comp_data).sort_values("mae") if comp_data else pd.DataFrame()
    )

    _det_r2 = 0
    if actual_best_name and actual_best_name in trained:
        _bt = trained[actual_best_name]
        _det_r2 = _bt[3] if len(_bt) > 3 else 0

    # Count kept external features (excluding ext_feed_)
    kept_external = [
        c
        for c in final_feature_cols
        if c.startswith("ext_") and not c.startswith("ext_feed_")
    ]
    ext_status = (
        f"{len(kept_external)}/{len(external_cols_present)}"
        if external_cols_present
        else "none"
    )

    status = (
        f"{'🔄 Volatile mode' if volatile_mode else '✅'} {product_name} | "
        f"{cache_status} | {len(df)} rows | "
        f"Best: {best_name} | "
        f"MAE={best_mae:,.0f} | "
        f"R²={_det_r2:.3f} | "
        f"External kept: {ext_status}"
    )

    return pred_df, best_name, status, orig_metrics, trained, test_preds, comp_df


def quick_predict_hybrid(
    product_name,
    product_id,
    fetcher,
    cache,
    horizon=30,
    force_refresh=False,
    lags=None,
    windows=None,
    selected_models=None,
    detrend_method=DEFAULT_DETREND,
    freq="D",
    fast=True,
    train_log=None,
    product_type="Auto-detect",
):
    """Quick Predict with hybrid blending."""
    log_list = []

    def ml_wrapper(pn, pid, f, c, **kw):
        return quick_predict(
            pn,
            pid,
            f,
            c,
            horizon=horizon,
            force_refresh=force_refresh,
            lags=lags,
            windows=windows,
            selected_models=selected_models,
            detrend_method=detrend_method,
            freq=freq,
            fast=fast,
            train_log=log_list,
            product_type=product_type,
        )

    predictor = SmartHybridPredictor(ml_func=ml_wrapper, test_window=60)
    result = predictor.predict(
        product_name,
        product_id,
        fetcher,
        cache,
        horizon=horizon,
        force_refresh=force_refresh,
        freq=freq,
        fast=fast,
        train_log=log_list,
    )

    if callable(train_log) and train_log is not log_list:
        for msg in log_list:
            train_log(msg)

    return result


# 10. BACKTEST


def backtest(
    df,
    target,
    lags,
    windows,
    freq,
    detrend_method,
    selected_models,
    horizon=30,
    n_folds=3,
    fast=True,
    train_log=None,
    add_seasonal=True,
):
    from data_pipeline import (
        apply_trend,
        create_features,
        detrend_series,
        select_features,
    )

    log = train_log if train_log is not None else []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    target = _resolve_col(df, target)
    n = len(df)
    if n < horizon * (n_folds + 1):
        n_folds = max(1, n // (horizon * 2))

    rows = []
    for fold in range(n_folds):
        cut_idx = n - horizon * (n_folds - fold)
        if cut_idx < 20:
            continue
        train_slice = df.iloc[:cut_idx].copy()
        test_slice = df.iloc[cut_idx : cut_idx + horizon].copy()
        if test_slice.empty:
            continue

        split_date = compute_split_date(train_slice, "date", train_frac=0.85)
        detrended_target = f"{target}_detrended"

        detrend_result = detrend_series(
            train_slice, target, freq, detrend_method, split_date=split_date
        )
        if detrend_method == "smooth_diff":
            coeffs, ref_date = None, None
            smooth_w = detrend_result[2]
        elif detrend_method == "diff":
            coeffs, ref_date = None, None
            smooth_w = None
        elif detrend_method == "none":
            coeffs, ref_date = None, detrend_result[1]
            smooth_w = None
        else:
            coeffs, ref_date = detrend_result
            smooth_w = None

        if coeffs is None and smooth_w is not None:
            train_slice["smooth"] = (
                train_slice[target].rolling(smooth_w, min_periods=1).mean()
            )
            train_slice[detrended_target] = train_slice["smooth"].diff()
        elif coeffs is not None:
            train_slice[detrended_target] = (
                np.log(train_slice[target])
                if is_log_trend(coeffs)
                else train_slice[target]
            ) - apply_trend(train_slice["date"], coeffs, ref_date, freq)
        elif ref_date is not None:
            train_slice[detrended_target] = train_slice[target]
        else:
            train_slice[detrended_target] = train_slice[target].diff()

        external_cols = [c for c in train_slice.columns if c.startswith("ext_")]
        other_cols = [c for c in train_slice.columns if c.startswith("ext_feed_")]

        feat_df, feature_cols, _ = create_features(
            train_slice,
            detrended_target,
            lags,
            windows,
            freq=freq,
            external_cols=external_cols or None,
            other_cols=other_cols or None,
            add_seasonal=add_seasonal,
        )
        feature_cols = select_features(feat_df, feature_cols, split_date=split_date)
        _, trained, best_model, best_scaler, best_name, best_std, best_mae = train_all(
            feat_df,
            detrended_target,
            feature_cols,
            selected_models,
            split_date,
            fast=fast,
            train_log=log,
        )
        if best_model is None:
            continue

        try:
            pred_df = iterative_forecast(
                train_slice,
                target,
                detrended_target,
                feature_cols,
                best_model,
                best_scaler,
                best_std,
                len(test_slice),
                lags,
                windows,
                freq,
                coeffs,
                ref_date,
                None,
                None,
                smooth_w=smooth_w,
                add_texture=False,
                external_cols=external_cols,
                other_cols=other_cols,
            )  # No noise for backtest
        except Exception as exc:
            if isinstance(log, list):
                log.append(f"backtest fold {fold} forecast failed: {exc}")
            continue

        merged = test_slice[["date", target]].merge(
            pred_df[["date", "forecast"]], on="date", how="inner"
        )
        if merged.empty:
            continue
        mae = float(mean_absolute_error(merged[target], merged["forecast"]))
        rmse = float(np.sqrt(mean_squared_error(merged[target], merged["forecast"])))
        rows.append(
            {
                "fold": fold,
                "best_model": best_name,
                "mae": mae,
                "rmse": rmse,
                "n_test": len(merged),
            }
        )

    return pd.DataFrame(rows)


# 11. ANALYSIS TAB HELPER


def _external_label(col: str) -> str:
    return EXTERNAL_LABELS.get(col, col.replace("ext_feed_", "").replace("ext_", ""))


def analyze_product_relationships(
    df: pd.DataFrame,
    target: str = "price",
    external_cols: list | None = None,
    max_lag: int = 30,
    rolling_window: int = 60,
) -> dict:
    """Correlation analysis between product and external features."""
    from scipy import stats

    if external_cols is None:
        external_cols = [
            c for c in df.columns if c.startswith("ext_") and df[c].notna().sum() > 20
        ]

    empty = {
        "normalized_df": pd.DataFrame(),
        "correlations": [],
        "cross_corrs": {},
        "rolling_corrs": {},
        "lags": [],
    }
    if not external_cols:
        return {**empty, "summary_md": "No external feature data found."}

    work = df[["date", target] + external_cols].dropna().copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date").reset_index(drop=True)
    if len(work) < 10:
        return {**empty, "summary_md": "Not enough overlapping data points."}

    # Normalize for overlay
    norm_data = {"date": work["date"]}
    for col in [target] + external_cols:
        s = work[col]
        mn, mx = s.min(), s.max()
        norm_data[col] = ((s - mn) / (mx - mn) * 100) if mx > mn else 50.0
    normalized_df = pd.DataFrame(norm_data)

    correlations = []
    for col in external_cols:
        r, p = stats.pearsonr(work[target], work[col])
        correlations.append(
            {
                "col": col,
                "label": _external_label(col),
                "r": round(float(r), 4),
                "p_value": float(p),
            }
        )
    correlations.sort(key=lambda x: abs(x["r"]), reverse=True)

    lags = list(range(-max_lag, max_lag + 1))
    cross_corrs = {}
    for col in external_cols:
        vals = []
        for lag in lags:
            if lag >= 0:
                a = work[target].values[lag:]
                b = work[col].values[: len(a)]
            else:
                b = work[col].values[-lag:]
                a = work[target].values[: len(b)]
            if len(a) == len(b) and len(a) >= 10:
                try:
                    r = stats.pearsonr(a, b)[0]
                    vals.append(round(float(r), 4))
                except Exception:
                    vals.append(0.0)
            else:
                vals.append(0.0)
        cross_corrs[col] = vals

    for corr in correlations:
        cross_r = cross_corrs.get(corr["col"], [])
        if cross_r:
            best_idx = int(np.argmax(np.abs(cross_r)))
            corr["lag_at_max"], corr["r_at_max"] = lags[best_idx], cross_r[best_idx]
        else:
            corr["lag_at_max"], corr["r_at_max"] = 0, corr["r"]

    rolling_corrs = {
        col: work[target]
        .rolling(rolling_window, min_periods=max(10, rolling_window // 3))
        .corr(work[col])
        for col in external_cols
    }

    lines = []
    if correlations:
        top = correlations[0]
        abs_r = abs(top["r"])
        if abs_r >= 0.7:
            strength = "Very Strong"
        elif abs_r >= 0.5:
            strength = "Strong"
        elif abs_r >= 0.3:
            strength = "Moderate"
        elif abs_r >= 0.1:
            strength = "Weak"
        else:
            strength = "Very Weak / None"
        direction = "positive" if top["r"] > 0 else "negative"
        r_sq = top["r"] ** 2
        lag_note = ""
        if abs(top["lag_at_max"]) >= 1:
            lead = "leads" if top["lag_at_max"] > 0 else "lags"
            lag_note = f" — {top['label']} **{lead}** price by **{abs(top['lag_at_max'])} days**"
        lines.append("### Top Finding")
        lines.append(
            f"**{top['label']}** shows a **{strength}** {direction} correlation with {target} "
            f"(R = {top['r']:.3f}, R² = {r_sq:.1%}, p = {top['p_value']:.2e}){lag_note}."
        )

        sig = [c for c in correlations if c["p_value"] < 0.05]
        if sig:
            lines.append("\n### Statistically Significant (p < 0.05)")
            for c in sig[:5]:
                arrow = "↑" if c["r"] > 0 else "↓"
                lines.append(
                    f"- **{c['label']}**: R = {c['r']:+.3f} {arrow} (R² = {c['r'] ** 2:.1%})"
                )
        weak = [c for c in correlations if c["p_value"] >= 0.05]
        if weak:
            lines.append("\n### Not Significant (p ≥ 0.05)")
            lines.append(
                ", ".join(f"{c['label']}" for c in weak)
                + " — no reliable linear relationship detected."
            )

        lines.append(
            f"\n*Analysed {len(correlations)} external features over {len(work)} data points.*"
        )

    return {
        "normalized_df": normalized_df,
        "correlations": correlations,
        "cross_corrs": cross_corrs,
        "rolling_corrs": rolling_corrs,
        "lags": lags,
        "summary_md": "\n\n".join(lines) or "No correlations computed.",
    }


# 12. PERSISTENCE & REPORT


def make_safe_filename(name: str) -> str:
    s = unicodedata.normalize("NFC", str(name))
    s = re.sub(r"[^\w\s-]", "", s).strip().replace(" ", "_")
    return s or "model"


def runtime_versions():
    import importlib.metadata

    return {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scipy", "scikit-learn", "joblib"]
    }


def save_trained_models(
    trained: dict, product_name: str, metadata: dict = None
) -> list:
    import uuid

    required = {
        "feature_cols",
        "lags",
        "windows",
        "freq",
        "detrended_target",
        "target_col",
    }
    if not metadata or not required.issubset(metadata):
        raise ValueError(
            "Complete feature and forecasting metadata is required to save a reusable model."
        )
    paths = []
    for model_name, tup in trained.items():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = (
            Path(MODELS_DIR)
            / f"{make_safe_filename(product_name)}_{make_safe_filename(model_name)}_{stamp}_{uuid.uuid4().hex[:8]}.joblib"
        )
        model, scaler, mae, r2, std_res = tup[:5]
        payload = {
            "format": "predictgar-v2",
            "versions": runtime_versions(),
            "model": model,
            "scaler": scaler,
            "metadata": {
                **metadata,
                "product_name": product_name,
                "model_name": model_name,
                "date_trained": stamp,
                "mae": mae,
                "r2": r2,
                "std_res": std_res,
                "residual_q10": tup[5],
                "residual_q90": tup[6],
            },
        }
        temporary = path.with_suffix(".tmp")
        try:
            joblib.dump(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        paths.append(str(path))
    return paths


def load_trained_model(path: str):
    """Load only trusted locally-created packages; pickle can execute code."""
    payload = joblib.load(path)
    if not isinstance(payload, dict) or payload.get("format") != "predictgar-v2":
        raise ValueError(
            "Legacy model package is incomplete; retrain and save with this version."
        )
    if payload.get("versions") != runtime_versions():
        raise ValueError(
            "Model dependency versions differ; restore the original environment or retrain."
        )
    if not {"model", "scaler", "metadata"}.issubset(payload):
        raise ValueError("Incomplete model package.")
    return payload


def list_saved_models(product_name: str = None) -> list:
    if not os.path.isdir(MODELS_DIR):
        return []
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith((".joblib", ".pkl"))]
    if product_name:
        safe = make_safe_filename(product_name)
        files = [f for f in files if f.startswith(safe)]
    return sorted(files, reverse=True)


def generate_report(
    product_name,
    df,
    pred_df,
    best_model_name,
    metrics,
    backtest_df=None,
    anomalies=None,
    feature_importance_split=None,
) -> str:
    from html import escape

    safe_name = escape(unicodedata.normalize("NFC", str(product_name)))
    best_model_name = escape(str(best_model_name))
    mae = (
        metrics.get("mae", metrics.get("MAE", float("nan")))
        if isinstance(metrics, dict)
        else float("nan")
    )
    rmse = (
        metrics.get("rmse", metrics.get("RMSE", float("nan")))
        if isinstance(metrics, dict)
        else float("nan")
    )
    r2 = (
        metrics.get("r2", metrics.get("R-squared", float("nan")))
        if isinstance(metrics, dict)
        else float("nan")
    )

    forecast_rows = ""
    if pred_df is not None and not pred_df.empty:
        for _, row in pred_df.iterrows():
            forecast_rows += (
                f"<tr><td>{row['date']}</td><td>{row['forecast']:,.0f}</td>"
                f"<td>{row.get('lower', float('nan')):,.0f}</td>"
                f"<td>{row.get('upper', float('nan')):,.0f}</td></tr>\n"
            )

    backtest_html = ""
    if backtest_df is not None and not backtest_df.empty:
        backtest_html = backtest_df.to_html(index=False)

    feature_html = ""
    if feature_importance_split:
        feature_html += "<h3>Own-Graph Features</h3><ul>"
        for item in feature_importance_split.get("own_graph", [])[:15]:
            feature_html += (
                f"<li>{escape(str(item['feature']))}: {item['share'] * 100:.1f}%</li>"
            )
        feature_html += "</ul><h3>External Features</h3><ul>"
        for item in feature_importance_split.get("external", [])[:15]:
            feature_html += (
                f"<li>{escape(str(item['feature']))}: {item['share'] * 100:.1f}%</li>"
            )
        feature_html += "</ul>"

    anomaly_html = ""
    if anomalies:
        anomaly_html = (
            "<ul>"
            + "".join(
                f"<li>index {a['index']}: actual={a['actual']:,.0f} predicted={a['predicted']:,.0f} z={a['z_score']:.2f}</li>"
                for a in anomalies
            )
            + "</ul>"
        )
    else:
        anomaly_html = "<p>No anomalies detected.</p>"

    html = f"""<!DOCTYPE html>
<html lang="fa">
<head>
<meta charset="UTF-8">
<title>Price Intelligence Report - {safe_name}</title>
<style>
body {{ font-family: Tahoma, Arial, sans-serif; margin: 24px; color: #222; }}
h1 {{ border-bottom: 2px solid #444; padding-bottom: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
th {{ background: #f2f2f2; }}
.metric-box {{ display: inline-block; margin: 8px 16px 8px 0; padding: 8px 14px; background: #f7f7f7; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Price Intelligence Report: {safe_name}</h1>
<p>Best model: {best_model_name}</p>
<div class="metric-box">MAE: {mae:,.0f}</div>
<div class="metric-box">RMSE: {rmse:,.0f}</div>
<div class="metric-box">R2: {r2:.3f}</div>

<h2>Forecast</h2>
<table>
<tr><th>Date</th><th>Forecast</th><th>Lower</th><th>Upper</th></tr>
{forecast_rows}
</table>

<h2>Feature Importance</h2>
{feature_html}

<h2>Backtest Results</h2>
{backtest_html if backtest_html else "<p>No backtest data.</p>"}

<h2>Anomalies</h2>
{anomaly_html}

</body>
</html>
"""
    return html
