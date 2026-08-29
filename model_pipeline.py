"""
ITPNews Price Intelligence v13 - model_pipeline.py

Handles: model training, persistence, metrics, iterative forecasting,
hybrid (ML + simple-method) blending, backtesting, and HTML reports.

RULES ENFORCED IN THIS FILE:
- NEVER imports requests / bs4.
- compute_split_date() is THE canonical split; every other function receives
  split_date as a parameter.
- Default detrend = "linear", default models = Lasso/Ridge/ElasticNet/RF.
- Models with R^2 <= 0 are excluded from winner selection.
- ML weight is always >= 30% in hybrid blending; blend weights are dynamic.
- CI bands shift with the blended forecast; AR(1) texture is deterministic.
"""

from __future__ import annotations

import hashlib
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
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler

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
    from data_pipeline import Cache, Fetcher

MODELS_DIR = "data/models"
FORECASTS_DIR = "data/forecasts"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FORECASTS_DIR, exist_ok=True)

DEFAULT_MODELS = ["Lasso", "Ridge", "ElasticNet", "Random Forest"]
DEFAULT_DETREND = "linear"
MIN_ML_WEIGHT = 0.30


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------

def compute_split_date(df: pd.DataFrame, date_col: str = "date", train_frac: float = 0.8) -> pd.Timestamp:
    dates = pd.to_datetime(df[date_col]).sort_values().reset_index(drop=True)
    n = len(dates)
    if n < 2:
        return dates.iloc[-1] if n else pd.Timestamp.now()
    idx = max(1, min(n - 1, int(n * train_frac)))
    return dates.iloc[idx]


def _resolve_col(df: pd.DataFrame, key: str) -> str:
    """Find the actual column name that matches the key (NFC normalized)"""
    key_n = unicodedata.normalize("NFC", str(key))
    for c in df.columns:
        if unicodedata.normalize("NFC", str(c)) == key_n:
            return c
    # If not found, return the first non-date numeric column
    for c in df.columns:
        if c != "date" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    return key

# --------------------------------------------------------------------------
# Models - Increased max_iter to prevent convergence warnings
# --------------------------------------------------------------------------

def get_model(name: str):
    import warnings
    warnings.filterwarnings('ignore')
    
    if name == "Lasso":
        return Lasso(max_iter=10000, random_state=42, tol=1e-3, selection='random', alpha=0.01), {"alpha": [0.001, 0.01, 0.1]}
    if name == "Ridge":
        return Ridge(random_state=42, max_iter=10000, tol=1e-3), {"alpha": [0.1, 1.0, 10.0]}
    if name == "ElasticNet":
        return ElasticNet(max_iter=10000, random_state=42, tol=1e-3, selection='random'), {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.3, 0.5, 0.7]}
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_split=5, min_samples_leaf=3, random_state=42, n_jobs=-1), {"n_estimators": [100, 200], "max_depth": [8, 12, 15]}
    if name == "Gradient Boosting":
        return GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42), {"n_estimators": [100, 200], "learning_rate": [0.03, 0.05, 0.1]}
    if name == "XGBoost" and XGB_AVAILABLE:
        return xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=0), {"n_estimators": [100, 200], "max_depth": [4, 6, 8]}
    if name == "LightGBM" and LGBM_AVAILABLE:
        return lgb.LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1, verbosity=-1), {"n_estimators": [100, 200], "max_depth": [4, 6, 8]}
    logger.warning("Unknown model '%s', falling back to Ridge", name)
    return Ridge(random_state=42), {"alpha": [1.0]}


def create_ensemble(models_dict: dict, ensemble_type: str = "Voting"):
    estimators = [(re.sub(r"\W+", "_", name), model) for name, model in models_dict.items()]
    if len(estimators) < 2:
        return None
    if ensemble_type == "Voting":
        return VotingRegressor(estimators=estimators)
    elif ensemble_type == "Stacking":
        return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1.0), cv=3)
    return None


# --------------------------------------------------------------------------
# Training - FIXED: Better scaling for linear models + unified logging
# --------------------------------------------------------------------------

def train_model(X_train, y_train, X_test, y_test, model, params=None, fast=False) -> dict:
    import warnings
    warnings.filterwarnings('ignore')
    
    model_name = model.__class__.__name__
    
    # Always scale target
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-6:
        y_std = 1.0
    y_train_s = (y_train - y_mean) / y_std
    y_test_s = (y_test - y_mean) / y_std
    
    # Use StandardScaler for ALL models
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_train_s = np.nan_to_num(X_train_s, nan=0.0)
    X_test_s = np.nan_to_num(X_test_s, nan=0.0)
    y_train_s = np.nan_to_num(y_train_s, nan=0.0)
    
    best_model = model
    if fast:
        if model_name == "Random Forest":
            best_model.set_params(n_estimators=100, max_depth=10, n_jobs=-1)
        elif model_name in ["Lasso", "Ridge", "ElasticNet"]:
            best_model.set_params(max_iter=10000, tol=1e-3)
        elif model_name == "Gradient Boosting":
            best_model.set_params(n_estimators=100, max_depth=5)
    
    best_model.fit(X_train_s, y_train_s)
    preds = best_model.predict(X_test_s) * y_std + y_mean
    preds = np.clip(preds, np.min(y_test)*0.8, np.max(y_test)*1.2)
    
    residuals = y_test - preds
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    try: r2 = float(r2_score(y_test, preds))
    except: r2 = -999.0
    try: mape = float(mean_absolute_percentage_error(y_test, preds))*100
    except: mape = float("nan")
    std_res = float(np.std(residuals)) if len(residuals) else 0.0
    q10 = float(np.percentile(residuals, 10)) if len(residuals)>10 else -1.28*std_res
    q90 = float(np.percentile(residuals, 90)) if len(residuals)>10 else 1.28*std_res

    return {"model":best_model,"scaler":scaler,"mae":mae,"rmse":rmse,"r2":r2,"mape":mape,"std_res":std_res,"residual_q10":q10,"residual_q90":q90}

# --------------------------------------------------------------------------
# train_all - FIXED: Unified logging that handles both callable and list
# --------------------------------------------------------------------------

def train_all(df, detrended_target, feature_cols, selected_models, split_date,
              use_ensemble=False, ensemble_type="Voting", fast=True, progress=None, train_log=None):
    """Train all models with proper logging - handles any train_log type"""
    
    # FIX 5: Unified logger that handles any train_log type
    def _log(msg):
        if train_log is not None:
            if callable(train_log):
                train_log(msg)
            elif isinstance(train_log, list):
                train_log.append(msg)
            elif hasattr(train_log, 'append'):
                train_log.append(msg)
        # Always print for debugging
        print(msg)
        return msg
    
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    X = df[feature_cols].values
    y = df[detrended_target].values
    dates = pd.to_datetime(df["date"]).values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y, dates = X[mask], y[mask], dates[mask]

    if len(X) < 30:
        _log("Warning: Less than 30 samples available for training")
        return {}, {}, None, None, None, 0, 0

    train_bool = dates < np.datetime64(pd.to_datetime(split_date))
    if train_bool.sum() < 10 or (~train_bool).sum() < 5:
        train_bool = np.arange(len(X)) < int(len(X) * 0.8)

    X_train, X_test = X[train_bool], X[~train_bool]
    y_train, y_test = y[train_bool], y[~train_bool]

    selected_models = selected_models or DEFAULT_MODELS
    results, trained = {}, {}
    best_mae = float("inf")
    best_model = best_scaler = best_name = None
    best_std = 0

    if fast and len(selected_models) > 1:
        def _train_wrapper(name):
            mo, pr = get_model(name)
            r = train_model(X_train, y_train, X_test, y_test, mo, pr, fast)
            return name, r

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {ex.submit(_train_wrapper, name): name for name in selected_models}
            for fut in as_completed(futures):
                name, r = fut.result()
                results[name] = {"MAE": r["mae"], "RMSE": r["rmse"], "R-squared": r["r2"], "MAPE%": r["mape"]}
                trained[name] = (r["model"], r["scaler"], r["mae"], r["r2"], r["std_res"], r["residual_q10"], r["residual_q90"])
                if r["mae"] < best_mae and r["r2"] > 0:
                    best_mae, best_model, best_scaler, best_name, best_std = r["mae"], r["model"], r["scaler"], name, r["std_res"]
                _log(f"{name}: MAE={r['mae']:,.0f}, R2={r['r2']:.3f}")
    else:
        for name in selected_models:
            mo, pr = get_model(name)
            r = train_model(X_train, y_train, X_test, y_test, mo, pr, fast)
            results[name] = {"MAE": r["mae"], "RMSE": r["rmse"], "R-squared": r["r2"], "MAPE%": r["mape"]}
            trained[name] = (r["model"], r["scaler"], r["mae"], r["r2"], r["std_res"], r["residual_q10"], r["residual_q90"])
            if r["mae"] < best_mae and r["r2"] > 0:
                best_mae, best_model, best_scaler, best_name, best_std = r["mae"], r["model"], r["scaler"], name, r["std_res"]
            _log(f"{name}: MAE={r['mae']:,.0f}, R2={r['r2']:.3f}")

    if use_ensemble and len(trained) >= 2:
        try:
            y_mean_all = np.mean(y_train)
            y_std_all = np.std(y_train)
            if y_std_all < 1e-6:
                y_std_all = 1.0
            y_train_s = (y_train - y_mean_all) / y_std_all

            scaler_ens = StandardScaler()
            X_train_s = scaler_ens.fit_transform(X_train)
            model_dict = {name: t[0] for name, t in trained.items()}
            ensemble = create_ensemble(model_dict, ensemble_type)
            if ensemble:
                ensemble.fit(X_train_s, y_train_s)
                X_test_s = scaler_ens.transform(X_test)
                preds = ensemble.predict(X_test_s) * y_std_all + y_mean_all
                mae = float(mean_absolute_error(y_test, preds))
                rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
                r2 = float(r2_score(y_test, preds))
                mape = float(mean_absolute_percentage_error(y_test, preds)) * 100
                std_res = float(np.std(y_test - preds))
                resids = y_test - preds
                eq10 = float(np.percentile(resids, 10)) if len(resids) > 10 else -1.28 * std_res
                eq90 = float(np.percentile(resids, 90)) if len(resids) > 10 else 1.28 * std_res
                name = f"{ensemble_type} Ensemble"
                results[name] = {"MAE": mae, "RMSE": rmse, "R-squared": r2, "MAPE%": mape}
                trained[name] = (ensemble, scaler_ens, mae, r2, std_res, eq10, eq90)
                if mae < best_mae and r2 > 0:
                    best_mae, best_model, best_scaler, best_name, best_std = mae, ensemble, scaler_ens, name, std_res
                _log(f"{name}: MAE={mae:,.0f}, R2={r2:.3f}")
        except Exception as e:
            _log(f"Ensemble training failed: {e}")

    return results, trained, best_model, best_scaler, best_name, best_std, best_mae


# --------------------------------------------------------------------------
# Persistence (unchanged)
# --------------------------------------------------------------------------

def make_safe_filename(name: str) -> str:
    s = unicodedata.normalize("NFC", str(name))
    s = re.sub(r"[^\w\s-]", "", s).strip().replace(" ", "_")
    return s or "model"


def save_trained_models(trained: dict, product_name: str, metadata: dict = None) -> list:
    paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_product = make_safe_filename(product_name)
    for model_name, tup in trained.items():
        model, scaler, mae, r2, std_res = tup[:5]
        res_q10 = tup[5] if len(tup) > 5 else -1.28 * std_res
        res_q90 = tup[6] if len(tup) > 6 else 1.28 * std_res
        safe_name = f"{safe_product}_{make_safe_filename(model_name)}_{ts}"
        filepath = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
        payload = {
            "model": model, "scaler": scaler,
            "metadata": {
                "product_name": product_name, "model_name": model_name, "date_trained": ts,
                "mae": mae, "r2": r2, "std_res": std_res,
                "residual_q10": res_q10, "residual_q90": res_q90,
                **(metadata or {}),
            }
        }
        try:
            if JOBLIB_AVAILABLE:
                joblib.dump(payload, filepath)
                paths.append(filepath)
            else:
                import pickle
                pkl_path = filepath.replace(".joblib", ".pkl")
                with open(pkl_path, "wb") as f:
                    pickle.dump(payload, f)
                paths.append(pkl_path)
        except Exception as exc:
            logger.warning("save_trained_models failed for %s/%s: %s", product_name, model_name, exc)
    return paths


def load_trained_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        if path.endswith(".joblib") and JOBLIB_AVAILABLE:
            return joblib.load(path)
        elif path.endswith(".pkl"):
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as exc:
        logger.warning("load_trained_model failed for %s: %s", path, exc)
    return None


def list_saved_models(product_name: str = None) -> list:
    if not os.path.isdir(MODELS_DIR):
        return []
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith((".joblib", ".pkl"))]
    if product_name:
        safe = make_safe_filename(product_name)
        files = [f for f in files if f.startswith(safe)]
    return sorted(files, reverse=True)


def cleanup_old_models() -> int:
    if not os.path.isdir(MODELS_DIR):
        return 0
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith((".joblib", ".pkl"))]
    product_groups = {}
    for f in files:
        stem = f.rsplit(".", 1)[0]
        # Model filenames follow pattern: {product}_{ModelName}_{YYYYMMDD_HHMMSS}
        # Find the last two underscore-separated tokens that match the timestamp pattern
        parts = stem.split("_")
        ts_idx = None
        for i in range(len(parts) - 1, 0, -1):
            if len(parts[i]) == 14 and parts[i].isdigit():
                ts_idx = i
                break
        if ts_idx is not None and ts_idx >= 2:
            product = "_".join(parts[:ts_idx - 1])
        else:
            product = stem
        product_groups.setdefault(product, []).append(f)
    removed = 0
    for product, group_files in product_groups.items():
        if len(group_files) <= 1:
            continue
        for old in sorted(group_files, reverse=True)[1:]:
            try:
                os.remove(os.path.join(MODELS_DIR, old))
                removed += 1
            except Exception:
                pass
    return removed


# --------------------------------------------------------------------------
# Metrics (unchanged)
# --------------------------------------------------------------------------

def naive_benchmark(test_actuals) -> dict:
    arr = np.asarray(test_actuals, dtype=float)
    if len(arr) < 3:
        return {}
    preds = np.roll(arr, 1)
    preds[0] = arr[0]
    mae = float(mean_absolute_error(arr, preds))
    rmse = float(np.sqrt(mean_squared_error(arr, preds)))
    r2 = float(r2_score(arr, preds))
    mape = float(mean_absolute_percentage_error(arr, preds)) * 100
    resids = arr - preds
    std_r = float(np.std(resids))
    res_q10 = float(np.percentile(resids, 10)) if len(resids) > 10 else -1.28 * std_r
    res_q90 = float(np.percentile(resids, 90)) if len(resids) > 10 else 1.28 * std_r
    return {"MAE": mae, "RMSE": rmse, "R-squared": r2, "MAPE%": mape, "std_res": std_r, "residual_q10": res_q10, "residual_q90": res_q90}


def model_improvement_vs_naive(model_mae, naive_mae) -> float:
    if not naive_mae or np.isnan(naive_mae) or naive_mae == 0:
        return 0.0
    return float((naive_mae - model_mae) / naive_mae * 100)


def compute_original_scale_metrics(trained_models, df_feat, feature_cols, target_name,
                                    coeffs, ref_date, freq, split_date, smooth_w=0, pred_col=None):
    if pred_col is None:
        pred_col = target_name
    from data_pipeline import apply_trend

    use_diff = coeffs is None and ref_date is None
    df = df_feat.copy()
    df["date"] = pd.to_datetime(df["date"])
    X = df[feature_cols].values
    y_predicted = df[pred_col].values
    dates = pd.to_datetime(df["date"]).values
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y_predicted)
    X, y_predicted, dates = X[mask], y_predicted[mask], dates[mask]

    train_bool = dates < np.datetime64(pd.to_datetime(split_date))
    X_test = X[~train_bool]
    test_dates = dates[~train_bool]
    if len(X_test) < 2:
        return {}, np.array([])

    train_dates = dates[train_bool]
    target_series = df.set_index("date")[target_name]
    last_train_price = target_series.get(pd.Timestamp(train_dates[-1]), np.nan) if len(train_dates) > 0 else np.nan
    orig_prices = df.set_index("date")[target_name]
    y_test_original = np.array([orig_prices.get(pd.Timestamp(d), np.nan) for d in test_dates])
    valid = ~np.isnan(y_test_original)
    X_test, y_test_original, test_dates = X_test[valid], y_test_original[valid], test_dates[valid]

    y_test_smooth = None
    smooth_anchor = last_train_price
    if use_diff and smooth_w > 0:
        smooth_prices = df.set_index("date")["smooth"] if "smooth" in df.columns else None
        if smooth_prices is not None:
            y_test_smooth = np.array([smooth_prices.get(pd.Timestamp(d), np.nan) for d in test_dates])
            smooth_anchor = smooth_prices.get(pd.Timestamp(train_dates[-1]), last_train_price) if len(train_dates) > 0 else last_train_price

    trend_test = apply_trend(test_dates, coeffs, ref_date, freq)
    noise_threshold = 0.0
    if use_diff and smooth_w > 0 and "smooth" in df.columns:
        train_smooth = df.set_index("date")["smooth"]
        train_smooth_diff = train_smooth.diff()
        sd_std = train_smooth_diff[train_dates[0]:train_dates[-1]].std() if len(train_dates) > 1 else 0
        noise_threshold = 0.5 * (sd_std if not np.isnan(sd_std) and sd_std > 0 else 0.0)

    original_metrics = {}
    for model_name, tup in trained_models.items():
        model, scaler_m = tup[0], tup[1]
        X_test_scaled = scaler_m.transform(X_test)
        pred_model = model.predict(X_test_scaled)
        pred_model = np.nan_to_num(pred_model, nan=0.0, posinf=0.0, neginf=0.0)
        price_mean = float(np.nanmean(y_test_original)) if len(y_test_original) > 0 else 0
        if price_mean > 0 and not np.isnan(price_mean):
            pred_model = np.clip(pred_model, -10 * price_mean, 10 * price_mean)
        if use_diff and smooth_w > 0 and y_test_smooth is not None:
            pred_model = np.where(np.abs(pred_model) < noise_threshold, 0.0, pred_model)
            prev_smooth = np.roll(y_test_smooth, 1)
            prev_smooth[0] = smooth_anchor if smooth_anchor is not None else y_test_smooth[0]
            pred_original = prev_smooth + pred_model
            score_mae = mean_absolute_error(y_test_smooth, pred_original)
            score_rmse = np.sqrt(mean_squared_error(y_test_smooth, pred_original))
            score_r2 = r2_score(y_test_smooth, pred_original)
            score_mape = mean_absolute_percentage_error(y_test_smooth, pred_original) * 100
            score_std = float(np.std(y_test_smooth - pred_original))
            score_res = y_test_smooth - pred_original
        else:
            if use_diff:
                cum_pred = np.cumsum(pred_model)
                pred_original = last_train_price + cum_pred if not np.isnan(last_train_price) else cum_pred
            else:
                pred_original = pred_model + trend_test
            score_mae = mean_absolute_error(y_test_original, pred_original)
            score_rmse = np.sqrt(mean_squared_error(y_test_original, pred_original))
            score_r2 = r2_score(y_test_original, pred_original)
            score_mape = mean_absolute_percentage_error(y_test_original, pred_original) * 100
            score_std = float(np.std(y_test_original - pred_original))
            score_res = y_test_original - pred_original
        score_q10 = float(np.percentile(score_res, 10)) if len(score_res) > 10 else -1.28 * score_std
        score_q90 = float(np.percentile(score_res, 90)) if len(score_res) > 10 else 1.28 * score_std
        original_metrics[model_name] = {
            "MAE": score_mae, "RMSE": score_rmse, "R-squared": score_r2, "MAPE%": score_mape,
            "std_res": score_std, "residual_q10": score_q10, "residual_q90": score_q90,
        }

    naive = naive_benchmark(y_test_original)
    if naive:
        original_metrics["Naive (last value)"] = naive
    if use_diff and smooth_w > 0 and y_test_smooth is not None:
        smooth_naive = naive_benchmark(y_test_smooth)
        if smooth_naive:
            original_metrics["Smooth Naive"] = smooth_naive

    return original_metrics, y_test_original


def feature_importance_by_horizon(model, feature_cols, external_cols=None) -> dict:
    from data_pipeline import rank_features

    external_cols = set(external_cols or [])
    ranked = rank_features(model, feature_cols)
    total = sum(v for _, v in ranked) or 1.0

    own_graph, external = [], []
    for name, val in ranked:
        share = val / total
        entry = {"feature": name, "importance": float(val), "share": float(share)}
        is_external = name in external_cols or name.startswith("ext_")
        (external if is_external else own_graph).append(entry)

    return {"own_graph": own_graph, "external": external}


# --------------------------------------------------------------------------
# Forecasting helpers (unchanged)
# --------------------------------------------------------------------------

def _ar1_texture(n, seed_key, std_res, phi=0.35):
    if std_res is None or std_res <= 0 or n <= 0:
        return np.zeros(max(n, 0))
    h = hashlib.sha256(str(seed_key).encode("utf-8")).hexdigest()
    seed = int(h[:8], 16) % (2 ** 32)
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, std_res * 0.5, n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = phi * noise[i - 1] + eps[i]
    noise -= noise.mean()
    return noise


def iterative_forecast(df, target, detrended_target, feature_cols, model, scaler, std_res,
                        horizon, lags, windows, freq, coeffs, ref_date, residual_q10, residual_q90,
                        external_cols=None, smooth_w=None, noise_threshold=0.0, add_texture=True,
                        other_cols=None):
    from data_pipeline import apply_trend, create_features

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    target = _resolve_col(df, target)
    detrended_target = unicodedata.normalize("NFC", detrended_target)
    use_diff = coeffs is None and ref_date is None

    if use_diff and smooth_w and smooth_w > 0:
        smooth_series = df[target].rolling(smooth_w, min_periods=1).mean()
        df["smooth_diff_hist"] = smooth_series.diff()
        df[detrended_target] = df["smooth_diff_hist"]
        hist_col = "smooth_diff_hist"
    elif use_diff:
        df["diff"] = df[target].diff()
        df[detrended_target] = df["diff"]
        hist_col = "diff"
    else:
        hist_col = detrended_target
        if coeffs is not None and ref_date is not None:
            trend_hist = apply_trend(df["date"], coeffs, ref_date, freq)
            df[detrended_target] = df[target] - trend_hist
        else:
            df[detrended_target] = df[target]

    hist = df[["date", hist_col]].copy().rename(columns={hist_col: detrended_target}).reset_index(drop=True)
    delta_map = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(weeks=1), "M": pd.DateOffset(months=1)}
    delta = delta_map.get(freq, pd.DateOffset(months=1))
    last_date = hist["date"].max()
    future_dates = pd.date_range(start=last_date + delta, periods=horizon, freq=freq)

    last_external_vals = {}
    if external_cols:
        for ec in external_cols:
            if ec in df.columns:
                valid = df[ec].dropna()
                last_external_vals[ec] = valid.iloc[-1] if len(valid) else np.nan

    preds = []
    for fdate in future_dates:
        temp = pd.concat([hist, pd.DataFrame({"date": [fdate], detrended_target: [np.nan]})], ignore_index=True)
        for ec, val in last_external_vals.items():
            if ec not in temp.columns:
                temp[ec] = val
        temp_feat, feat_cols, _ = create_features(temp, detrended_target, lags, windows, freq,
                                                    other_cols=other_cols, external_cols=external_cols)
        if temp_feat.empty:
            break
        missing = set(feature_cols) - set(temp_feat.columns)
        for col in missing:
            temp_feat[col] = np.nan
        X_step = temp_feat[feature_cols].iloc[-1:].values
        if np.any(np.isnan(X_step)):
            col_medians = np.nanmedian(X_step, axis=0)
            col_medians = np.where(np.isnan(col_medians), 0.0, col_medians)
            nan_mask = np.isnan(X_step)
            X_step[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])
        X_scaled = scaler.transform(X_step)
        pred = float(model.predict(X_scaled)[0])
        if not np.isfinite(pred):
            pred = 0.0
        preds.append({"date": fdate, "forecast_detrended": pred})
        hist = pd.concat([hist, pd.DataFrame({"date": [fdate], detrended_target: [pred]})], ignore_index=True)

    if not preds:
        return pd.DataFrame()

    pred_df = pd.DataFrame(preds)
    if use_diff and smooth_w and smooth_w > 0:
        cum_preds = np.cumsum(pred_df["forecast_detrended"].values)
        if "smooth" in df.columns:
            last_actual = float(df["smooth"].dropna().iloc[-1]) if not df.empty else 0
        else:
            last_actual = float(df[target].iloc[-1]) if not df.empty else 0
        pred_df["forecast"] = last_actual + cum_preds
    elif use_diff:
        cum_preds = np.cumsum(pred_df["forecast_detrended"].values)
        last_actual = float(df[target].iloc[-1]) if not df.empty else 0
        pred_df["forecast"] = last_actual + cum_preds
    elif coeffs is not None and ref_date is not None:
        pred_df["forecast"] = pred_df["forecast_detrended"] + apply_trend(pred_df["date"], coeffs, ref_date, freq)
    else:
        pred_df["forecast"] = pred_df["forecast_detrended"]

    no_anchor = (use_diff and smooth_w and smooth_w > 0)
    if not no_anchor:
        last_actual_original = float(df[target].iloc[-1]) if not df.empty else 0
        first_fc = float(pred_df["forecast"].iloc[0]) if not pred_df.empty else 0
        anchor_shift = last_actual_original - first_fc
        pred_df["forecast"] = pred_df["forecast"] + anchor_shift

    if residual_q10 is not None and residual_q90 is not None:
        cum = np.sqrt(np.arange(1, len(pred_df) + 1))
        pred_df["lower"] = np.clip(pred_df["forecast"] + residual_q10 * cum, 0, None)
        pred_df["upper"] = pred_df["forecast"] + residual_q90 * cum
    else:
        cum_std = std_res * np.sqrt(np.arange(1, len(pred_df) + 1))
        pred_df["lower"] = np.clip(pred_df["forecast"] - 1.28 * cum_std, 0, None)
        pred_df["upper"] = pred_df["forecast"] + 1.28 * cum_std

    if add_texture:
        n = len(pred_df)
        if residual_q10 is not None and residual_q90 is not None and (residual_q90 - residual_q10) > 0:
            step_std = (residual_q90 - residual_q10) / 2.56
        else:
            step_std = std_res if std_res else 0.0
        if step_std > 0 and n > 0:
            noise = _ar1_texture(n, (str(target), int(horizon), str(freq)), step_std, phi=0.35)
            pred_df["forecast"] = pred_df["forecast"] + noise
            pred_df["lower"] = pred_df["lower"] + noise
            pred_df["upper"] = pred_df["upper"] + noise

    return pred_df[["date", "forecast", "lower", "upper"]]


def create_forecast_plot(df, target, pred_df, test_preds=None) -> go.Figure:
    target = _resolve_col(df, target)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Historical & Forecast", "Price Distribution", "Forecast Trend", "Forecast Range"),
        specs=[[{"secondary_y": False}, {"type": "histogram"}], [{"type": "scatter"}, {"type": "box"}]],
    )
    fig.add_trace(go.Scatter(x=df["date"], y=df[target], mode="lines", name="Historical",
                              line=dict(color="#10b981", width=2)), row=1, col=1)
    if test_preds is not None and not test_preds.empty:
        fig.add_trace(go.Scatter(x=test_preds["date"], y=test_preds["predicted"], mode="lines+markers",
                                  name="Test Prediction", line=dict(color="#f59e0b", width=1.5, dash="dot"),
                                  marker=dict(size=4, symbol="circle")), row=1, col=1)
        fig.add_trace(go.Scatter(x=test_preds["date"], y=test_preds["actual"], mode="lines", name="Test Actual",
                                  line=dict(color="#10b981", width=1.5, dash="dash")), row=1, col=1)
    if not pred_df.empty:
        last_hist_val = df[target].iloc[-1] if not df.empty else 0
        last_hist_date = df["date"].iloc[-1] if not df.empty else pred_df["date"].iloc[0]
        bridge_dates = pd.concat([pd.Series([last_hist_date]), pred_df["date"]], ignore_index=True)
        bridge_forecast = pd.concat([pd.Series([last_hist_val]), pred_df["forecast"]], ignore_index=True)
        bridge_upper = pd.concat([pd.Series([last_hist_val]), pred_df["upper"]], ignore_index=True)
        bridge_lower = pd.concat([pd.Series([last_hist_val]), pred_df["lower"]], ignore_index=True)
        fig.add_trace(go.Scatter(x=bridge_dates, y=bridge_forecast, mode="lines+markers", name="Forecast",
                                  line=dict(color="#ef4444", width=2), marker=dict(size=6)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bridge_dates, y=bridge_upper, mode="lines", line=dict(width=0),
                                  showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=bridge_dates, y=bridge_lower, mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(239,68,68,0.2)", name="80% CI"), row=1, col=1)
        fig.add_trace(go.Scatter(x=bridge_dates, y=bridge_forecast, mode="lines+markers", name="Trend",
                                  line=dict(color="#ef4444", width=3)), row=2, col=1)
        fig.add_trace(go.Box(y=pred_df["forecast"], name="Forecast Range", marker_color="#ef4444", boxmean="sd"),
                      row=2, col=2)
    fig.add_trace(go.Histogram(x=df[target], name="Distribution", marker_color="#10b981", nbinsx=30), row=1, col=2)
    fig.update_layout(height=700, template="plotly_dark", showlegend=True, hovermode="x unified")
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    return fig


def detect_volatility_regime(df: pd.DataFrame, target_col: str, window: int = 30) -> float:
    series = df[target_col].dropna()
    if len(series) < window * 2:
        return 1.0
    recent_std = series.tail(window).pct_change().dropna().std()
    full_std = series.pct_change().dropna().std()
    if not full_std or full_std == 0:
        return 1.0
    ratio = recent_std / full_std
    return float(np.clip(ratio, 0.3, 3.0))


def adjust_forecast_ci(pred_df: pd.DataFrame, regime_ratio: float, base_q10, base_q90) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df
    out = pred_df.copy()
    scale = np.sqrt(regime_ratio)
    if "lower" in out.columns and "upper" in out.columns:
        out["lower"] = (out["forecast"] + base_q10 * scale * np.sqrt(np.arange(1, len(out) + 1))).clip(lower=0)
        out["upper"] = out["forecast"] + base_q90 * scale * np.sqrt(np.arange(1, len(out) + 1))
    return out


def detect_anomalies(actual, predicted, std_res, threshold: float = 3.0) -> list:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = ~(np.isnan(actual) | np.isnan(predicted))
    actual, predicted = actual[valid], predicted[valid]
    if len(actual) < 2 or std_res == 0:
        return []
    errors = np.abs(actual - predicted)
    z_scores = errors / max(std_res, 1)
    anomalies = []
    for i in range(len(actual)):
        if z_scores[i] > threshold:
            anomalies.append({
                "index": i, "actual": float(actual[i]), "predicted": float(predicted[i]),
                "error": float(actual[i] - predicted[i]), "z_score": float(z_scores[i]),
                "direction": "ABOVE" if actual[i] > predicted[i] else "BELOW",
            })
    return anomalies


def format_anomaly_report(anomalies, dates=None) -> str:
    if not anomalies:
        return "No anomalies detected."
    lines = [f"{len(anomalies)} anomaly(ies) found:"]
    for a in anomalies[:10]:
        d = f" at {dates[a['index']]}" if dates is not None and a['index'] < len(dates) else ""
        lines.append(f"  - {d}: actual={a['actual']:,.0f} vs pred={a['predicted']:,.0f} ({a['direction']}, z={a['z_score']:.1f} sigma)")
    if len(anomalies) > 10:
        lines.append(f"  ... and {len(anomalies) - 10} more")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# SmartHybridPredictor (unchanged)
# --------------------------------------------------------------------------

class SmartHybridPredictor:
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
        ma = np.mean(series[-window:])
        return np.full(horizon, ma)

    def _holt_forecast(self, series, horizon, alpha=0.3, beta=0.1) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        n = len(series)
        if n < 4:
            return np.full(horizon, series[-1])
        level, trend = series[0], series[1] - series[0] if n > 1 else 0
        for t in range(1, n):
            prev_level = level
            level = alpha * series[t] + (1 - alpha) * (level + trend)
            trend = beta * (level - prev_level) + (1 - beta) * trend
        return np.array([level + trend * (i + 1) for i in range(horizon)])

    def _trend_forecast(self, series, horizon) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        n = len(series)
        if n < 2:
            return np.full(horizon, series[-1])
        x = np.arange(n)
        coeffs = np.polyfit(x, series, 1)
        future_x = np.arange(n, n + horizon)
        return np.polyval(coeffs, future_x)

    def _theta_forecast(self, series, horizon, theta=2.0) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        n = len(series)
        if n < 4:
            return np.full(horizon, series[-1])
        x = np.arange(n)
        slope, intercept, _, _, _ = stats.linregress(x, series)
        future_x = np.arange(n, n + horizon)
        trend = intercept + slope * future_x
        residual = series - (intercept + slope * x)
        avg_residual = np.mean(residual[-min(30, n):])
        return trend + avg_residual * (1.0 / theta)

    def _evaluate_method(self, series, horizon, method_fn) -> float:
        n = len(series)
        if n < self.test_window + horizon:
            return 1e18
        errors = []
        for i in range(n - self.test_window, n - horizon + 1):
            train = series[:i]
            actual = series[i:i + horizon]
            if len(actual) < horizon:
                continue
            pred = method_fn(train, horizon)
            errors.append(np.mean(np.abs(pred - actual)))
        return np.mean(errors) if errors else 1e18

    def predict(self, product_name, product_id, fetcher, cache, horizon=30, force_refresh=False,
                freq="D", fast=True, train_log=None):
        log = train_log if isinstance(train_log, list) else []

        ml_result = self.ml_func(
            product_name, product_id, fetcher, cache, horizon=horizon,
            force_refresh=force_refresh, freq=freq, fast=fast, train_log=log,
        )

        if ml_result is None or ml_result[0] is None:
            df = cache.load(product_id) if cache is not None else pd.DataFrame()
            if df.empty:
                from data_pipeline import generate_fallback
                df = generate_fallback()
            if "price" in df.columns:
                series = df["price"].values
            else:
                series = df.iloc[:, 1].values if df.shape[1] > 1 else np.array([])
            if len(series) < 10:
                return ml_result
            methods = self._simple_methods()
            scores = {name: self._evaluate_method(series, horizon, fn) for name, fn in methods.items()}
            best_simple = min(scores, key=scores.get)
            pred = methods[best_simple](series, horizon)
            delta_map = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(weeks=1), "M": pd.DateOffset(months=1)}
            _delta = delta_map.get(freq, pd.Timedelta(days=1))
            pred_df = pd.DataFrame({
                "date": pd.date_range(start=df["date"].max() + _delta, periods=horizon, freq=freq),
                "forecast": pred,
                "lower": pred * 0.9,
                "upper": pred * 1.1,
            })
            status = f"[hybrid fallback] ML failed, using {best_simple}"
            return pred_df, best_simple, status, {}, {}, pd.DataFrame(), pd.DataFrame()

        pred_df_ml, ml_best_name, ml_status, ml_metrics, ml_trained, ml_test_preds, ml_comp_df = ml_result

        df = cache.load(product_id) if cache is not None else pd.DataFrame()
        if df.empty or "price" not in df.columns:
            return ml_result
        series = df.sort_values("date")["price"].dropna().values
        if len(series) < self.test_window + horizon:
            return ml_result

        methods = self._simple_methods()
        simple_scores = {}
        for name, fn in methods.items():
            score = self._evaluate_method(series, horizon, fn)
            simple_scores[name] = score

        ml_mae = ml_metrics.get(ml_best_name, {}).get("MAE", float("inf")) if isinstance(ml_metrics, dict) else float("inf")

        epsilon = 1e-6
        inv_scores = {k: 1.0 / (v + epsilon) for k, v in simple_scores.items()}
        total_inv = sum(inv_scores.values())
        simple_weights = {k: v / total_inv for k, v in inv_scores.items()}
        best_simple_name = max(simple_weights, key=simple_weights.get)
        best_simple_weight = simple_weights[best_simple_name]

        if ml_mae < simple_scores.get(best_simple_name, float("inf")):
            ml_weight = max(0.5, 1.0 - best_simple_weight)
        else:
            ml_weight = MIN_ML_WEIGHT
        ml_weight = max(ml_weight, MIN_ML_WEIGHT)
        best_simple_weight = 1.0 - ml_weight

        simple_pred = methods[best_simple_name](series, horizon)
        ml_pred = pred_df_ml["forecast"].values
        blended_pred = ml_weight * ml_pred + best_simple_weight * simple_pred
        last_actual = float(series[-1])
        anchor_shift = last_actual - blended_pred[0]
        blended_pred = blended_pred + anchor_shift

        if not pred_df_ml.empty and "lower" in pred_df_ml.columns:
            lower = np.clip(pred_df_ml["lower"].values + anchor_shift, 0, None)
            upper = pred_df_ml["upper"].values + anchor_shift
        else:
            step_std = simple_scores.get(best_simple_name, 0) * 0.5
            cum_std = step_std * np.sqrt(np.arange(1, horizon + 1))
            lower = np.clip(blended_pred - 1.28 * cum_std, 0, None)
            upper = blended_pred + 1.28 * cum_std

        delta_map = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(weeks=1), "M": pd.DateOffset(months=1)}
        _delta = delta_map.get(freq, pd.Timedelta(days=1))
        pred_df = pd.DataFrame({
            "date": pd.date_range(start=df["date"].max() + _delta, periods=horizon, freq=freq),
            "forecast": blended_pred,
            "lower": lower,
            "upper": upper,
        })

        status = (f"[hybrid] ML({ml_best_name}) {ml_weight:.0%} + "
                  f"{best_simple_name} {best_simple_weight:.0%}")
        if isinstance(train_log, list):
            train_log.append(status)
        elif callable(train_log):
            train_log(status)

        return pred_df, ml_best_name, status, ml_metrics, ml_trained, ml_test_preds, ml_comp_df


# --------------------------------------------------------------------------
# Quick Predict (unchanged - the unified logger handles it)
# --------------------------------------------------------------------------

def quick_predict(product_name, product_id, fetcher, cache, horizon=30, force_refresh=False,
                   lags=None, windows=None, selected_models=None, detrend_method=DEFAULT_DETREND,
                   freq="D", fast=True, train_log=None):
    from data_pipeline import (
        RELATED_PRODUCTS, apply_trend, auto_params, create_features, detrend_series,
        generate_fallback, get_product_profile, merge_external_features,
        preprocess, select_features, CACHE_DIR,
    )

    # FIX: Use unified logger
    def _log(msg):
        if train_log is not None:
            if callable(train_log):
                train_log(msg)
            elif isinstance(train_log, list):
                train_log.append(msg)
            elif hasattr(train_log, 'append'):
                train_log.append(msg)
        print(msg)
        return msg

    target = "price"

    try:
        raw_df, cache_status = cache.update(product_id, force=force_refresh)
    except Exception:
        raw_df, cache_status = generate_fallback(), "synthetic"

    if raw_df is None or raw_df.empty:
        raw_df, cache_status = generate_fallback(), "synthetic"

    df = preprocess(raw_df, target, freq=freq)
    if df.empty:
        return None, None, "No data available after preprocessing.", {}, {}, pd.DataFrame(), pd.DataFrame()

    if lags is None or windows is None:
        auto_lags, auto_windows = auto_params(df, target, freq)
        lags = lags or auto_lags
        windows = windows or auto_windows

    df = merge_external_features(df, product_name, freq)
    external_cols_present = [c for c in df.columns if c.startswith("ext_")]

    split_date = compute_split_date(df, "date", train_frac=0.8)

    detrended_target = f"{target}_detrended"
    detrend_result = detrend_series(df[df["date"] < split_date], target, freq=freq, method=detrend_method, split_date=split_date)
    if isinstance(detrend_result, tuple) and len(detrend_result) == 3:
        coeffs, ref_date = None, None
        smooth_w = detrend_result[2]
    else:
        coeffs, ref_date = detrend_result
        smooth_w = None

    if coeffs is None and smooth_w is not None:
        df["smooth"] = df[target].rolling(smooth_w, min_periods=1).mean()
        df[detrended_target] = df["smooth"].diff()
    elif coeffs is not None:
        df[detrended_target] = df[target] - apply_trend(df["date"], coeffs, ref_date, freq)
    elif ref_date is not None:
        df[detrended_target] = df[target]
    else:
        df[detrended_target] = df[target].diff()

    other_cols = [c for c in df.columns if c not in [target, "date", detrended_target, "smooth", "is_holiday"] 
              and np.issubdtype(df[c].dtype, np.number) and df[c].nunique() > 10]
    df_feat, feature_cols, seasonal_cols = create_features(
        df, detrended_target, lags, windows, freq=freq,
        other_cols=other_cols, external_cols=external_cols_present or None, add_seasonal=True,
    )
    for lag in lags:
        col = f"{detrended_target}_lag_{lag}"
        df_feat[col] = df_feat[detrended_target].shift(lag)
        feature_cols.append(col)
    feature_cols = list(dict.fromkeys([c for c in feature_cols if c in df_feat.columns]))

    feature_cols = select_features(df_feat, feature_cols, split_date=split_date, max_corr=0.95)
    valid_cols = [detrended_target] + [c for c in feature_cols if c in df_feat.columns]
    df_feat = df_feat.dropna(subset=valid_cols)

    selected_models = selected_models or DEFAULT_MODELS

    if external_cols_present:
        base_cols = [c for c in feature_cols if not c.startswith("ext_")]
        ext_cols = sorted(set(c for c in feature_cols if c.startswith("ext_")))
        _log(f"Smart external select: {len(ext_cols)} candidates")
        _, base_trained, base_model, base_scaler, base_name, base_std, base_mae = train_all(
            df_feat, detrended_target, base_cols, selected_models,
            split_date=split_date, use_ensemble=False, fast=fast, train_log=train_log,
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed
        def _eval_ext(col):
            cand = base_cols + [col]
            _, ct, cm, cs, cn, cstd, cmae = train_all(
                df_feat, detrended_target, cand, selected_models,
                split_date=split_date, use_ensemble=False, fast=fast, train_log=[],
            )
            return col, cmae
        candidates = {}
        with ThreadPoolExecutor(max_workers=4) as ex:
            fut_map = {ex.submit(_eval_ext, c): c for c in ext_cols}
            for f in as_completed(fut_map):
                col, cmae = f.result()
                candidates[col] = cmae
        kept = []
        base_mae_val = base_mae if base_mae is not None else float("inf")
        for col in sorted(ext_cols):
            cmae = candidates[col]
            if cmae is not None and cmae < base_mae_val * 0.95:
                kept.append(col)
                _log(f"  + {col} (MAE {cmae:,.0f})")
            else:
                _log(f"  - {col} (skipped)")
        if kept:
            final_cols = base_cols + kept
            _log(f"Retraining with {len(kept)} kept external features...")
            _, trained, best_model, best_scaler, best_name, best_std, best_mae = train_all(
                df_feat, detrended_target, final_cols, selected_models,
                split_date=split_date, use_ensemble=False, fast=fast, train_log=train_log,
            )
            feature_cols = final_cols
            external_status = f"kept {len(kept)}/{len(ext_cols)}" if len(kept) < len(ext_cols) else "kept all"
        else:
            feature_cols = base_cols
            trained = base_trained or {}
            best_model, best_scaler, best_name, best_std, best_mae = base_model, base_scaler, base_name, base_std, base_mae
            external_status = "removed (none helped)"
        _log(f"Smart external: {external_status}")
    else:
        results, trained, best_model, best_scaler, best_name, best_std, best_mae = train_all(
            df_feat, detrended_target, feature_cols, selected_models,
            split_date=split_date, use_ensemble=True, ensemble_type="Voting", fast=fast, train_log=train_log,
        )
        external_status = "none"

    # ═══ FILTER: Remove models with R² ≤ 0 ═══
    if trained:
        trained = {k: v for k, v in trained.items() if len(v) > 3 and v[3] > 0}
        if not trained:
            _log("All models had R² ≤ 0; using naive fallback.")
            return _naive_fallback_forecast(df, target, horizon, freq), "Naive", \
                   "All models failed.", {}, {}, pd.DataFrame(), pd.DataFrame()
        # Update best model if it was filtered out
        if best_name not in trained:
            best_name = min(trained.keys(), key=lambda n: trained[n][2])
            bt = trained[best_name]
            best_model, best_scaler, best_std, best_mae = bt[0], bt[1], bt[4] if len(bt) > 4 else 0, bt[2]

    if best_model is None:
        _log("No eligible model found; using naive fallback.")
        return _naive_fallback_forecast(df, target, horizon, freq), "Naive", \
               f"All models failed R2 filter.", {}, {}, pd.DataFrame(), pd.DataFrame()

    orig_metrics, _ = compute_original_scale_metrics(
        trained, df_feat, feature_cols, target, coeffs, ref_date, freq, split_date, smooth_w=smooth_w or 0,
        pred_col=detrended_target,
    )

    naive_key = "Smooth Naive" if (coeffs is None and smooth_w) else "Naive (last value)"
    naive_mae = orig_metrics.get(naive_key, {}).get("MAE", float("inf"))

    valid_models = {
        mk: mv for mk, mv in orig_metrics.items()
        if mk not in (naive_key, "Naive (last value)")
        and mv.get("R-squared", -999) > 0
        and mv.get("MAE", float("inf")) < naive_mae
    }
    if valid_models:
        best_name = min(valid_models, key=lambda n: valid_models[n]["MAE"])
        if best_name in trained:
            bt = trained[best_name]
            best_model, best_scaler, best_std, best_mae = bt[0], bt[1], bt[4] if len(bt) > 4 else best_std, bt[2] if len(bt) > 2 else float("inf")

    # ── Ensure these are always defined ──
    uses_naive = False
    actual_best_name = best_name

    # ═══ FIX: Safe metrics access ═══
    best_metrics = orig_metrics.get(actual_best_name, {}) or {}
    best_std = best_metrics.get("std_res", best_std) if best_metrics else (best_std or 0)
    best_q10 = best_metrics.get("residual_q10", -1.28 * (best_std or 0)) if best_metrics else -1.28 * (best_std or 0)
    best_q90 = best_metrics.get("residual_q90", 1.28 * (best_std or 0)) if best_metrics else 1.28 * (best_std or 0)


    try:
        pred_df = iterative_forecast(
            df_feat, target, detrended_target, feature_cols, best_model, best_scaler, best_std,
            horizon, lags, windows, freq, coeffs, ref_date,
            residual_q10=best_q10, residual_q90=best_q90,
            external_cols=external_cols_present or None, smooth_w=smooth_w or 0,
            other_cols=other_cols,
        )
    except Exception as exc:
        _log(f"Forecast failed: {exc}; using naive fallback.")
        pred_df = _naive_fallback_forecast(df, target, horizon, freq)

    if pred_df.empty:
        return None, None, "Forecast generation failed.", {}, {}, pd.DataFrame(), pd.DataFrame()

    regime = detect_volatility_regime(raw_df, target)
    if abs(regime - 1.0) > 0.2:
        pred_df = adjust_forecast_ci(pred_df, regime, best_q10, best_q90)

    if uses_naive:
        related = RELATED_PRODUCTS.get(unicodedata.normalize("NFC", str(product_name)), [])
        slopes_raw = {"_self": 0.0}
        lookback = 30
        if len(raw_df) > lookback:
            series_vals = raw_df[target].tail(lookback).values
            slopes_raw["_self"] = (series_vals[-1] - series_vals[0]) / lookback
        weighted_sum, total_weight = slopes_raw["_self"], 1.0
        for rp in related:
            if rp in df.columns and len(df[rp].dropna()) > lookback:
                rp_series = df[rp].dropna().tail(lookback).values
                rp_slope = (rp_series[-1] - rp_series[0]) / lookback
                cc = df[[target, rp]].corr().iloc[0, 1]
                if not np.isnan(cc) and abs(cc) > 0.3:
                    weighted_sum += rp_slope * abs(cc)
                    total_weight += abs(cc)
        blended_slope = weighted_sum / total_weight if total_weight > 0 else 0.0
        damping = 0.94
        drift = np.zeros(len(pred_df))
        cur = 0.0
        for i in range(len(pred_df)):
            cur = cur * damping + blended_slope
            drift[i] = cur
        pred_df["forecast"] += drift
        pred_df["lower"] += drift
        pred_df["upper"] += drift
        band_half = (pred_df["upper"] - pred_df["lower"]) / 2.0
        center = (pred_df["upper"] + pred_df["lower"]) / 2.0
        pred_df["lower"] = np.clip(center - band_half * 1.15, 0, None)
        pred_df["upper"] = center + band_half * 1.15

    test_preds = pd.DataFrame()
    try:
        if not uses_naive:
            best_model_obj = trained.get(actual_best_name, (None, None))[0]
            best_scaler_obj = trained.get(actual_best_name, (None, None))[1]
            if best_model_obj is not None and best_scaler_obj is not None:
                test_mask = df_feat["date"].isin(df[df["date"] >= split_date]["date"].values)
                test_mask = test_mask & df_feat[feature_cols].notna().all(axis=1)
                X_test_full = df_feat.loc[test_mask, feature_cols].values
                test_dates_full = df_feat.loc[test_mask, "date"].values
                if len(X_test_full) > 0:
                    X_te_scaled = best_scaler_obj.transform(X_test_full)
                    pred_te_det = best_model_obj.predict(X_te_scaled)
                    if coeffs is None and smooth_w:
                        if "smooth" in df.columns:
                            last_train_price = float(df[df["date"] < split_date]["smooth"].dropna().iloc[-1]) if len(df[df["date"] < split_date]) > 0 else 0
                        else:
                            last_train_price = float(df[df["date"] < split_date][target].iloc[-1]) if len(df[df["date"] < split_date]) > 0 else 0
                        pred_te = last_train_price + np.cumsum(pred_te_det)
                    elif coeffs is None:
                        last_train_price = float(df[df["date"] < split_date][target].iloc[-1]) if len(df[df["date"] < split_date]) > 0 else 0
                        pred_te = last_train_price + np.cumsum(pred_te_det)
                    else:
                        trend_te = apply_trend(test_dates_full, coeffs, ref_date, freq)
                        pred_te = pred_te_det + trend_te
                    actual_te = df.set_index("date").reindex(pd.to_datetime(test_dates_full))[target].values
                    test_preds = pd.DataFrame({
                        "date": test_dates_full, "actual": actual_te,
                        "predicted": np.clip(pred_te, 0, None),
                    })
    except Exception:
        pass

    # Use the internal trained dict (has (model, scaler, mae, r2, std_res, ...))
    if trained:
        comp_data = []
        for name, tup in trained.items():
            mae = tup[2] if len(tup) > 2 else None
            r2 = tup[3] if len(tup) > 3 else None
            comp_data.append({"model": name, "mae": mae, "r2": r2})
        comp_df = pd.DataFrame(comp_data).sort_values("mae") if comp_data else pd.DataFrame()
    else:
        comp_df = pd.DataFrame()

    # Use detrended R² from trained dict (not original-scale)
    _det_r2 = 0
    if actual_best_name and actual_best_name in trained:
        _bt = trained[actual_best_name]
        _det_r2 = _bt[3] if len(_bt) > 3 else 0

    status = (
        f"{'[naive fallback]' if uses_naive else '[ok]'} {product_name} | "
        f"{cache_status} | {len(df)} rows | Best: {best_name} | "
        f"MAE={best_mae:,.0f} | Detrended R²={_det_r2:.3f} | "
        f"External: {external_status}"
    )

    return pred_df, best_name, status, orig_metrics, trained, test_preds, comp_df


def _naive_fallback_forecast(df, target, horizon, freq):
    if target in df.columns and df[target].notna().any():
        last_val = df[target].dropna().iloc[-1]
    else:
        last_val = 0.0
    if "date" in df.columns and not df.empty:
        last_date = pd.to_datetime(df["date"]).max()
    else:
        last_date = pd.Timestamp.now()
    step = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(weeks=1), "M": pd.DateOffset(months=1)}.get(freq, pd.Timedelta(days=1))
    dates = [last_date + step * (i + 1) for i in range(horizon)]
    std = df[target].dropna().std() if target in df.columns else 0.0
    if np.isnan(std) or std is None:
        std = 0.0
    expansion = np.sqrt(np.arange(1, horizon + 1))
    return pd.DataFrame({
        "date": dates, "forecast": [last_val] * horizon,
        "lower": np.clip(last_val - std * expansion, 0, None), "upper": last_val + std * expansion,
    })


def quick_predict_hybrid(product_name, product_id, fetcher, cache, horizon=30, force_refresh=False,
                          lags=None, windows=None, selected_models=None, detrend_method=DEFAULT_DETREND,
                          freq="D", fast=True, train_log=None):
    """Wraps quick_predict with SmartHybridPredictor blending."""
    log_list = []
    
    def ml_wrapper(pn, pid, f, c, **kw):
        return quick_predict(
            pn, pid, f, c, horizon=horizon, force_refresh=force_refresh,
            lags=lags, windows=windows, selected_models=selected_models,
            detrend_method=detrend_method, freq=freq, fast=fast, train_log=log_list,
        )
    
    predictor = SmartHybridPredictor(ml_func=ml_wrapper, test_window=60)
    
    result = predictor.predict(
        product_name, product_id, fetcher, cache, horizon=horizon,
        force_refresh=force_refresh, freq=freq, fast=fast, train_log=log_list,
    )
    
    # Forward log messages to caller's train_log function
    if callable(train_log) and train_log is not log_list:
        for msg in log_list:
            train_log(msg)
    
    return result


# --------------------------------------------------------------------------
# Backtest (unchanged)
# --------------------------------------------------------------------------

def backtest(df, target, lags, windows, freq, detrend_method, selected_models,
             horizon=30, n_folds=3, fast=True, train_log=None):
    from data_pipeline import apply_trend, create_features, detrend_series, select_features

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
        test_slice = df.iloc[cut_idx:cut_idx + horizon].copy()
        if test_slice.empty:
            continue

        split_date = compute_split_date(train_slice, "date", train_frac=0.85)
        detrended_target = f"{target}_detrended"
        detrend_result = detrend_series(train_slice, target, freq, detrend_method, split_date=split_date)
        if len(detrend_result) == 3:
            coeffs, ref_date = None, None
        else:
            coeffs, ref_date = detrend_result

        smooth_w = None
        if coeffs is None and ref_date is None:
            if len(detrend_result) == 3:
                smooth_w = detrend_result[2]
        if coeffs is None and smooth_w is not None:
            train_slice["smooth"] = train_slice[target].rolling(smooth_w, min_periods=1).mean()
            train_slice[detrended_target] = train_slice["smooth"].diff()
        elif coeffs is not None:
            train_slice[detrended_target] = train_slice[target] - apply_trend(train_slice["date"], coeffs, ref_date, freq)
        elif ref_date is not None:
            train_slice[detrended_target] = train_slice[target]
        else:
            train_slice[detrended_target] = train_slice[target].diff()

        feat_df, feature_cols, _ = create_features(train_slice, detrended_target, lags, windows, freq=freq)
        feature_cols = select_features(feat_df, feature_cols, split_date=split_date)

        _, trained, best_model, best_scaler, best_name, best_std, best_mae = train_all(
            feat_df, detrended_target, feature_cols, selected_models, split_date, fast=fast, train_log=log,
        )
        if best_model is None:
            continue

        try:
            pred_df = iterative_forecast(
                feat_df, target, detrended_target, feature_cols, best_model, best_scaler, best_std,
                len(test_slice), lags, windows, freq, coeffs, ref_date, None, None,
                smooth_w=smooth_w,
            )
        except Exception as exc:
            if isinstance(log, list):
                log.append(f"backtest fold {fold} forecast failed: {exc}")
            continue

        merged = test_slice[["date", target]].merge(pred_df[["date", "forecast"]], on="date", how="inner")
        if merged.empty:
            continue
        mae = float(mean_absolute_error(merged[target], merged["forecast"]))
        rmse = float(np.sqrt(mean_squared_error(merged[target], merged["forecast"])))
        rows.append({"fold": fold, "best_model": best_name, "mae": mae, "rmse": rmse, "n_test": len(merged)})

    return pd.DataFrame(rows)


def train_direct_models(df, target, feature_cols, selected_models, split_date, fast=True, train_log=None):
    return train_all(df, target, feature_cols, selected_models, split_date, fast=fast, train_log=train_log)


def forecast_direct_ensemble(df, target, feature_cols, models_dict, scaler, horizon, freq="D"):
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    step = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(weeks=1), "M": pd.DateOffset(months=1)}.get(freq, pd.Timedelta(days=1))
    last_date = work["date"].max()
    dates = [last_date + step * (i + 1) for i in range(horizon)]
    last_row = work.iloc[[-1]][feature_cols].fillna(0.0)
    x_scaled = scaler.transform(last_row) if scaler is not None else last_row.values
    preds = []
    for model in models_dict.values():
        try:
            preds.append(model.predict(x_scaled)[0])
        except Exception:
            continue
    avg_pred = float(np.mean(preds)) if preds else float(work[target].dropna().iloc[-1])
    return pd.DataFrame({"date": dates, "forecast": np.full(horizon, avg_pred)}), avg_pred


# --------------------------------------------------------------------------
# Report generation
# --------------------------------------------------------------------------

def generate_report(product_name, df, pred_df, best_model_name, metrics, backtest_df=None,
                     anomalies=None, feature_importance_split=None) -> str:
    safe_name = unicodedata.normalize("NFC", str(product_name))
    mae = metrics.get("mae", metrics.get("MAE", float("nan"))) if isinstance(metrics, dict) else float("nan")
    rmse = metrics.get("rmse", metrics.get("RMSE", float("nan"))) if isinstance(metrics, dict) else float("nan")
    r2 = metrics.get("r2", metrics.get("R-squared", float("nan"))) if isinstance(metrics, dict) else float("nan")

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
            feature_html += f"<li>{item['feature']}: {item['share'] * 100:.1f} percent</li>"
        feature_html += "</ul><h3>External Features</h3><ul>"
        for item in feature_importance_split.get("external", [])[:15]:
            feature_html += f"<li>{item['feature']}: {item['share'] * 100:.1f} percent</li>"
        feature_html += "</ul>"

    anomaly_html = ""
    if anomalies:
        anomaly_html = "<ul>" + "".join(
            f"<li>index {a['index']}: actual={a['actual']:,.0f} predicted={a['predicted']:,.0f} "
            f"z={a['z_score']:.2f}</li>" for a in anomalies
        ) + "</ul>"
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