# app.py
"""
ITPNews Price Intelligence v13 — Gradio Application
5-tab interface: Quick Predict, Data & Charts, Training, Forecast, Backtest
Integrates data_pipeline.py and model_pipeline.py
"""

import os, json, re, unicodedata, traceback as tb_mod
from datetime import datetime
import numpy as np, pandas as pd
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data Pipeline imports
from data_pipeline import (
    Fetcher, Cache, preprocess, create_features, detrend_series, apply_trend,
    generate_fallback, merge_external_features, get_product_profile,
    list_external_features, check_data_quality, auto_params, select_features,
    CACHE_DIR, MODELS_DIR, FORECASTS_DIR
)

# Model Pipeline imports
from model_pipeline import (
    train_all, compute_original_scale_metrics, iterative_forecast,
    create_forecast_plot, save_trained_models, load_trained_model,
    list_saved_models, cleanup_old_models, backtest,
    quick_predict, quick_predict_hybrid,
    detect_volatility_regime, detect_anomalies, format_anomaly_report,
    feature_importance_by_horizon, compute_split_date,
    XGB_AVAILABLE, LGBM_AVAILABLE, DEFAULT_MODELS, DEFAULT_DETREND,
    _resolve_col, SmartHybridPredictor
)

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _log(entries, msg, ok=True):
    prefix = "✓" if ok else "✗"
    entries = list(entries or [])
    entries.append(f"[{_ts()}] {prefix} {msg}")
    return entries, "\n".join(entries)

def _log_err(entries, msg, exc):
    entries = list(entries or [])
    entries.append(f"[{_ts()}] ✗ {msg}")
    entries.append(f"  {type(exc).__name__}: {exc}")
    return entries, "\n".join(entries)

def create_app():
    removed = cleanup_old_models()
    if removed:
        print(f"Cleaned up {removed} old model files")
    
    fetcher = Fetcher()
    cache = Cache(fetcher)

    def load_products():
        return fetcher.discover_products()

    initial_products, initial_source = load_products()
    product_names = sorted(initial_products.keys()) if initial_products else ["(no products found)"]
    default_target = product_names[0] if product_names else None

    models_available = ["Ridge", "Lasso", "ElasticNet", "Random Forest", "Gradient Boosting"]
    if XGB_AVAILABLE:
        models_available.append("XGBoost")
    if LGBM_AVAILABLE:
        models_available.append("LightGBM")

    with gr.Blocks(
        title="ITPNews Price Intelligence v13 — Hybrid + Profiles",
        theme=gr.themes.Soft(),
        css="""
        .log-box textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 11px !important; }
        .alert-box textarea { font-family: 'Consolas', 'Courier New', monospace !important; }
        """
    ) as demo:

        # ═══════════════════════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════════════════════
        gr.Markdown("# 📊 ITPNews Price Intelligence & Forecasting")
        gr.Markdown("**v13 Hybrid + Profiles** — ML-driven forecasts with automatic external feature selection and hybrid blending")

        log_display = gr.Textbox(
            label="📋 Activity Log",
            lines=8, max_lines=20, interactive=False,
            elem_classes=["log-box"],
            value=f"[{_ts()}] ✓ App ready — {len(initial_products)} products loaded from {initial_source}"
        )

        # ═══════════════════════════════════════════════════════════
        # GETTING STARTED GUIDE
        # ═══════════════════════════════════════════════════════════
        with gr.Accordion("📖 Getting Started", open=True):
            gr.Markdown("""
### How to use this application

1. **⚡ Quick Predict** — Pick a product, click predict. The system automatically:
   - Fetches historical prices from ITPNews
   - Enriches smart products with macro data (USD/IRR, gold, CPI) via product profiles
   - Trains multiple ML models + tests simple statistical methods
   - Blends the best performers with dynamic weights (ML always ≥30%)
   - Removes any external feature that hurts accuracy (auto-remove gate)
   - Produces a forecast with 80% confidence intervals with AR(1) texture

2. **📈 Data & Charts** — Browse products, fetch raw data, upload your own CSV files. See product profiles and which external features are available.

3. **🔧 Training** — Customize which models to train, adjust feature parameters, and see exactly which features drive predictions (own-graph vs external, color-coded).

4. **🔮 Forecast** — Generate predictions from a trained model with optional hybrid blending.

5. **🧪 Backtest** — Validate model accuracy on historical data before trusting future forecasts.

**Reading results:** Every forecast is compared against a naive baseline (repeating the last price).
- `[ok]` — Model beat the baseline
- `[naive fallback]` — Model underperformed; damped momentum shown instead (clearly labeled, wider CI)
- `[hybrid]` — Blended forecast with contribution percentages shown
            """)

        # ═══════════════════════════════════════════════════════════
        # STATE
        # ═══════════════════════════════════════════════════════════
        products_state = gr.State(initial_products)
        data_state = gr.State(pd.DataFrame())
        model_state = gr.State({
            "model": None, "scaler": None, "feature_cols": [], "target_col": None,
            "detrended_target": None, "trend_coeffs": None, "ref_date": None,
            "std_residual": None, "residual_q10": None, "residual_q90": None,
            "best_model_name": None,
            "lags": [1, 7, 14, 30], "windows": [7, 14, 30, 60],
            "best_mae": None, "freq": "D", "trained_models": {},
            "detrend_method": "linear", "seasonal_cols": [], "external_cols": [],
            "smooth_w": None,
        })
        selected_model_name_state = gr.State(None)
        current_product_state = gr.State(default_target)
        log_entries = gr.State([])

        def sync_dropdowns(product_name, products):
            try:
                if product_name in products:
                    return (gr.update(value=product_name), gr.update(),
                            gr.update(value=product_name), product_name)
                return gr.update(), gr.update(), gr.update(), product_name
            except Exception:
                return gr.update(), gr.update(), gr.update(), product_name

        # ═══════════════════════════════════════════════════════════
        # TAB 1: QUICK PREDICT
        # ═══════════════════════════════════════════════════════════
        with gr.Tab("⚡ Quick Predict"):
            gr.Markdown("### ⚡ One-Click Forecast: Fetch → Enrich → Train → Predict")

            with gr.Row():
                with gr.Column(scale=1):
                    qp_product = gr.Dropdown(
                        choices=product_names, value=default_target,
                        label="Product to Forecast"
                    )
                    qp_horizon = gr.Number(
                        value=30, label="Forecast Horizon (days)",
                        precision=0, minimum=1, maximum=365
                    )
                    with gr.Row():
                        qp_hybrid = gr.Checkbox(
                            value=True, label="Hybrid Mode",
                            info="Blends ML with simple statistical methods for better accuracy"
                        )
                        qp_fast = gr.Checkbox(
                            value=True, label="Fast Mode",
                            info="Quicker training with slightly less hyperparameter tuning"
                        )
                        qp_force = gr.Checkbox(
                            value=False, label="Force Refresh",
                            info="Ignore cache and re-download all data"
                        )
                    qp_btn = gr.Button("⚡ Run Quick Predict", variant="primary", size="lg")

                with gr.Column(scale=2):
                    qp_status = gr.Textbox(label="📊 Result", lines=3, value="Select a product and click predict")
                    qp_quality = gr.Textbox(label="🔍 Data Quality", lines=2)
                    qp_alerts = gr.Textbox(
                        label="⚠️ Model Confidence Notes", lines=3,
                        elem_classes=["alert-box"],
                        info="Flags when test predictions were unusually far from actual values"
                    )
                    qp_metrics = gr.Dataframe(label="📈 Model Performance", max_height=180)

            qp_chart = gr.Plot(label="📉 Forecast Chart")
            with gr.Row():
                qp_table = gr.Dataframe(label="📋 Forecast Values", max_height=200)
                qp_download = gr.File(label="💾 Download CSV")

            def qp_fn(product_name, horizon, hybrid, fast, force, products, entries):
                try:
                    entries, _ = _log(entries, f"Quick Predict started: {product_name} ({int(horizon)}d) | Hybrid={hybrid}")
                    
                    if not products or product_name not in products:
                        entries, log_text = _log(entries, "No product selected", ok=False)
                        return ("No product selected", "", "", pd.DataFrame(), None,
                                pd.DataFrame(), None, entries, log_text)

                    pid = products[product_name]

                    profile = get_product_profile(product_name)
                    ext_feats = profile.get("external_features", [])
                    profile_note = ""
                    if ext_feats:
                        profile_note = f" | Profile: smart ({', '.join(ext_feats)})"
                    else:
                        profile_note = " | Profile: default (own graph only)"

                    train_log_capture = []
                    
                    if hybrid:
                        pred_df, best_name, status, metrics, trained, test_preds, comp_df = quick_predict_hybrid(
                            product_name, pid, fetcher, cache, int(horizon),
                            force_refresh=force, fast=fast,
                            train_log=lambda msg: train_log_capture.append(msg)
                        )
                    else:
                        pred_df, best_name, status, metrics, trained, test_preds, comp_df = quick_predict(
                            product_name, pid, fetcher, cache, int(horizon),
                            force_refresh=force, fast=fast,
                            train_log=lambda msg: train_log_capture.append(msg)
                        )
                    
                    for line in train_log_capture:
                        entries, _ = _log(entries, line)

                    if pred_df is None:
                        entries, log_text = _log(entries, f"Prediction failed: {status}", ok=False)
                        return (status, "", "", pd.DataFrame(), None,
                                pd.DataFrame(), None, entries, log_text)

                    # Load historical data for chart
                    hist_df = cache.load(pid)
                    if "price" in hist_df.columns:
                        hist_df = preprocess(hist_df, "price", "D")
                        hist_df = hist_df.rename(columns={"price": product_name})
                    else:
                        hist_df = preprocess(hist_df, product_name, "D")

                    # Create forecast plot
                    fig = create_forecast_plot(hist_df, product_name, pred_df, test_preds)

                    # Data quality check
                    warnings_list = check_data_quality(hist_df, product_name)
                    quality_text = "✓ No data quality issues detected." if not warnings_list else " | ".join(warnings_list)

                    # Anomaly detection on test set
                    alert_text = "✓ No unusual prediction errors on the test set."
                    if test_preds is not None and not test_preds.empty and "actual" in test_preds.columns:
                        std_res = metrics.get(best_name, {}).get("std_res", 0) if isinstance(metrics, dict) else 0
                        if not std_res:
                            std_res = float(np.nanstd(test_preds["actual"] - test_preds["predicted"]))
                        anomalies = detect_anomalies(
                            test_preds["actual"].values, test_preds["predicted"].values,
                            std_res, threshold=2.5
                        )
                        if anomalies:
                            dates_list = pd.to_datetime(test_preds["date"]).dt.strftime("%Y-%m-%d").tolist()
                            alert_text = format_anomaly_report(anomalies, dates=dates_list)

                    # Save forecast to CSV
                    csv_path = os.path.join(CACHE_DIR, "quick_forecast.csv")
                    pred_df.to_csv(csv_path, index=False)

                    full_status = status + profile_note
                    entries, log_text = _log(entries, f"Quick Predict complete — Best model: {best_name}")
                    
                    return (full_status, quality_text, alert_text, comp_df, fig,
                            pred_df, csv_path, entries, log_text)

                except Exception as e:
                    entries, log_text = _log_err(entries, "Quick Predict crashed", e)
                    return (f"✗ ERROR: {e}", "", "", pd.DataFrame(), None,
                            pd.DataFrame(), None, entries, log_text)

            qp_btn.click(
                fn=qp_fn,
                inputs=[qp_product, qp_horizon, qp_hybrid, qp_fast, qp_force, products_state, log_entries],
                outputs=[qp_status, qp_quality, qp_alerts, qp_metrics, qp_chart,
                        qp_table, qp_download, log_entries, log_display]
            )

        # ═══════════════════════════════════════════════════════════
        # TAB 2: DATA & CHARTS
        # ═══════════════════════════════════════════════════════════
        with gr.Tab("📈 Data & Charts"):
            with gr.Accordion("📦 Available Products", open=False):
                refresh_products_btn = gr.Button("🔄 Refresh Product List")
                products_status = gr.Textbox(
                    label="Status", value=f"✓ {len(initial_products)} products loaded from {initial_source}"
                )
                products_table = gr.Dataframe(
                    value=pd.DataFrame(list(initial_products.items()), columns=["Product", "ID"]),
                    label="All Available Products", max_height=400
                )

            with gr.Accordion("📡 Fetch or Upload Data", open=True):
                with gr.Row():
                    with gr.Column():
                        target = gr.Dropdown(
                            choices=product_names, value=default_target,
                            label="Target Product"
                        )
                        extras = gr.Dropdown(
                            choices=product_names, label="Additional Products to Merge",
                            multiselect=True,
                            info="Add related products (e.g. feed costs) as extra feature columns"
                        )
                        freq = gr.Dropdown(choices=["D", "W", "M"], value="D", label="Frequency")
                        force_full = gr.Checkbox(value=False, label="Force full re-download")
                        fetch_btn = gr.Button("📡 Fetch Data", variant="primary", size="lg")
                    with gr.Column():
                        file_upload = gr.File(
                            label="📂 Or upload your own CSV (columns: date, price)",
                            file_types=[".csv"]
                        )
                        gr.Markdown("_Uploaded CSV replaces fetched data. Must have `date` and `price` columns._")

                profile_indicator = gr.Textbox(
                    label="🔍 Product Profile",
                    interactive=False,
                    info="Shows which external features this product uses (configured in PRODUCT_PROFILES)"
                )

                status = gr.Textbox(label="Status")
                dataset_info = gr.Textbox(label="📊 Dataset Summary", lines=4)
                preview = gr.Dataframe(label="👁️ Data Preview (last 20 rows)", max_height=200)
                history_chart = gr.Plot(label="📈 Historical Price Chart")

                def show_profile_fn(product_name):
                    try:
                        profile = get_product_profile(product_name)
                        ext = profile.get("external_features", [])
                        related = profile.get("related_products", True)
                        if ext:
                            return f"Smart Profile — External: {', '.join(ext)} | Related Products: {related}"
                        return f"Default Profile — Own graph only | Related Products: {related}"
                    except Exception:
                        return "Profile unavailable"

                def fetch_data_fn(target_name, extra_names, freq_val, force, products, entries):
                    try:
                        entries, _ = _log(entries, f"Fetching data: {target_name}")
                        target_name = unicodedata.normalize("NFC", target_name)
                        norm_products = {unicodedata.normalize("NFC", k): v for k, v in products.items()}
                        
                        if not norm_products or target_name not in norm_products:
                            entries, log_text = _log(entries, "No product selected", ok=False)
                            return ("No product selected", "No data", pd.DataFrame(), None,
                                    pd.DataFrame(), entries, log_text)

                        selected = [target_name] + [unicodedata.normalize("NFC", x) for x in (extra_names or [])]
                        # Deduplicate while keeping the first occurrence (target stays first)
                        seen = set()
                        unique_selected = []
                        for name in selected:
                            if name not in seen:
                                seen.add(name)
                                unique_selected.append(name)
                        selected = unique_selected

                        merged = None
                        for name in selected:
                            pid = norm_products.get(name)
                            if not pid:
                                continue
                            df, cache_status = cache.update(pid, force=force)
                            if df.empty:
                                df = generate_fallback()
                            df = df.rename(columns={"price": name})
                            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                            merged = df if merged is None else pd.merge(merged, df, on="date", how="outer")
                        # Remove duplicate columns (e.g., price_x, price_y) from merging same product twice
                        merged = merged.loc[:, ~merged.columns.duplicated()]

                        if merged is None or merged.empty:
                            merged = generate_fallback().rename(columns={"price": target_name})

                        merged = preprocess(merged, target_name, freq=freq_val)
                        merged = merge_external_features(merged, target_name, freq_val)
                        
                        ext_cols = [c for c in merged.columns if c.startswith("ext_")]
                        ext_note = f"\nExternal features: {len(ext_cols)} merged" if ext_cols else ""

                        info = (
                            f"Product: {target_name}\n"
                            f"Rows: {len(merged):,}\n"
                            f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}\n"
                            f"Source: {cache_status}{ext_note}"
                        )

                        fig = go.Figure()
                        col_name = merged.columns[1] if merged.columns[1] != "date" else merged.columns[2]
                        fig.add_trace(go.Scatter(
                            x=merged["date"], y=merged[col_name],
                            mode="lines", name=target_name,
                            line=dict(color="#10b981", width=2)
                        ))
                        fig.update_layout(
                            title=f"{target_name} — Historical Prices",
                            template="plotly_dark", height=400,
                            hovermode="x unified"
                        )

                        entries, log_text = _log(entries, f"Fetched {len(merged):,} rows for {target_name}")
                        return (f"✓ Ready: {len(merged):,} rows", info, merged.head(20), fig,
                                merged, entries, log_text)

                    except Exception as e:
                        entries, log_text = _log_err(entries, "Data fetch failed", e)
                        return (f"✗ ERROR: {e}", "No data", pd.DataFrame(), None,
                                pd.DataFrame(), entries, log_text)

                def upload_csv_fn(file_obj, target_name, freq_val, entries):
                    try:
                        if file_obj is None:
                            entries, log_text = _log(entries, "No file selected", ok=False)
                            return ("No file selected", "No data", pd.DataFrame(), None,
                                    pd.DataFrame(), entries, log_text)

                        entries, _ = _log(entries, f"Uploading CSV: {os.path.basename(file_obj.name)}")
                        raw = pd.read_csv(file_obj.name)
                        raw.columns = [c.strip().lower() for c in raw.columns]

                        if "date" not in raw.columns:
                            entries, log_text = _log(entries, "CSV must have a 'date' column", ok=False)
                            return ("CSV must have a 'date' column", "No data", pd.DataFrame(), None,
                                    pd.DataFrame(), entries, log_text)

                        price_col = "price" if "price" in raw.columns else None
                        if price_col is None:
                            numeric_cols = [c for c in raw.columns if c != "date" and pd.api.types.is_numeric_dtype(raw[c])]
                            if not numeric_cols:
                                entries, log_text = _log(entries, "CSV must have a numeric price column", ok=False)
                                return ("CSV must have a numeric price column", "No data", pd.DataFrame(), None,
                                        pd.DataFrame(), entries, log_text)
                            price_col = numeric_cols[0]

                        use_name = target_name if target_name else price_col
                        raw = raw.rename(columns={price_col: use_name})
                        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
                        raw = raw.dropna(subset=["date", use_name])
                        merged = preprocess(raw[["date", use_name]], use_name, freq=freq_val)

                        info = (
                            f"Source: uploaded CSV\n"
                            f"Column: {use_name}\n"
                            f"Rows: {len(merged):,}\n"
                            f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}"
                        )

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=merged["date"], y=merged[use_name],
                            mode="lines", name=use_name,
                            line=dict(color="#10b981", width=2)
                        ))
                        fig.update_layout(
                            title=f"{use_name} — Uploaded Data",
                            template="plotly_dark", height=400
                        )

                        entries, log_text = _log(entries, f"CSV uploaded: {len(merged):,} rows")
                        return (f"✓ Ready: {len(merged):,} rows (from upload)", info, merged.head(20), fig,
                                merged, entries, log_text)

                    except Exception as e:
                        entries, log_text = _log_err(entries, "CSV upload failed", e)
                        return (f"✗ ERROR: {e}", "No data", pd.DataFrame(), None,
                                pd.DataFrame(), entries, log_text)

                target.change(fn=show_profile_fn, inputs=[target], outputs=[profile_indicator])
                fetch_btn.click(
                    fn=fetch_data_fn,
                    inputs=[target, extras, freq, force_full, products_state, log_entries],
                    outputs=[status, dataset_info, preview, history_chart, data_state,
                             log_entries, log_display]
                )
                file_upload.upload(
                    fn=upload_csv_fn,
                    inputs=[file_upload, target, freq, log_entries],
                    outputs=[status, dataset_info, preview, history_chart, data_state,
                             log_entries, log_display]
                )

        # ═══════════════════════════════════════════════════════════
        # TAB 3: TRAINING
        # ═══════════════════════════════════════════════════════════
        with gr.Tab("🔧 Training"):
            gr.Markdown("### 🔧 Train Custom Models")

            with gr.Row():
                with gr.Column():
                    lags_input = gr.Textbox(value="1,7,14,30", label="Look-back Periods (lag days)")
                    windows_input = gr.Textbox(value="7,14,30,60", label="Rolling Window Sizes")
                    with gr.Row():
                        auto_params_btn = gr.Button("🎯 Auto Params", size="sm")
                        fast_mode = gr.Checkbox(value=False, label="Fast Mode")
                    add_seasonal = gr.Checkbox(value=True, label="Include Seasonal Patterns")
                    auto_feature_select = gr.Checkbox(value=True, label="Auto Feature Selection")
                    detrend_method = gr.Dropdown(
                        choices=["smooth_diff", "diff", "linear", "quadratic", "none"],
                        value="linear", label="Detrend Method",
                        info="'linear' works with ALL models. 'smooth_diff'/'diff' ONLY for Ridge/Lasso/ElasticNet."
                    )
                with gr.Column():
                    models_check = gr.CheckboxGroup(
                        choices=models_available, value=DEFAULT_MODELS,
                        label="Algorithms to Train"
                    )
                    use_ensemble = gr.Checkbox(value=False, label="Also Train an Ensemble")
                    ensemble_type = gr.Dropdown(
                        choices=["Voting", "Stacking"], value="Voting",
                        label="Ensemble Method"
                    )
                    train_btn = gr.Button("🔥 TRAIN MODELS", variant="primary", size="lg")

            with gr.Row():
                train_status = gr.Textbox(label="📊 Training Result", scale=2)
                dataset_info_display = gr.Textbox(label="📋 Dataset Used", scale=1, lines=5)

            with gr.Row():
                comparison_plot = gr.Plot(label="📊 Model Comparison", scale=2)
                feature_plot = gr.Plot(label="🔍 Feature Importance (own-graph vs external)", scale=1)

            comparison_table = gr.Dataframe(label="🏆 All Models Ranked", max_height=250)

            with gr.Row():
                save_models_btn = gr.Button("💾 Save Trained Models")
                save_status = gr.Textbox(label="💾 Save Status", scale=1)

            def train_fn(df, target_name, lags_str, windows_str, fast, add_seasonal_feat,
                         auto_select, detrend_meth, selected_models, freq_val, use_ens, ens_type,
                         state, entries, progress=gr.Progress()):
                log = []
                try:
                    entries, _ = _log(entries, f"Training started: {target_name}")

                    if df is None or df.empty:
                        entries, log_text = _log(entries, "No data loaded — fetch data first", ok=False)
                        return ("✗ No data — fetch data first", "No dataset", None, None, pd.DataFrame(),
                                state, entries, log_text)

                    df = df.copy()
                    df["date"] = pd.to_datetime(df["date"])

                    # Find the target column
                    if target_name not in df.columns:
                        found = [c for c in df.columns if target_name in c or c in target_name]
                        if found:
                            target_name = found[0]
                        else:
                            entries, log_text = _log(entries, f"Target '{target_name}' not found in data", ok=False)
                            return (f"✗ Target '{target_name}' not found", "No dataset", None, None, pd.DataFrame(),
                                    state, entries, log_text)

                    # Parse lags and windows
                    try:
                        lags = [int(x.strip()) for x in lags_str.split(",") if x.strip().isdigit()]
                        windows = [int(x.strip()) for x in windows_str.split(",") if x.strip().isdigit()]
                    except:
                        lags, windows = [1, 7, 14, 30], [7, 14, 30, 60]

                    external_cols_present = [c for c in df.columns if c.startswith("ext_")]

                    progress(0.05, desc="Splitting data...")
                    split_date = compute_split_date(df)

                    progress(0.10, desc="Detrending...")
                    detrend_result = detrend_series(
                        df[df["date"] < split_date], target_name, freq_val, detrend_meth,
                        split_date=split_date
                    )
                    smooth_w = None
                    if isinstance(detrend_result, tuple) and len(detrend_result) == 3:
                        coeffs, ref_date = None, None
                        smooth_w = detrend_result[2]
                    else:
                        coeffs, ref_date = detrend_result

                    price_col = target_name if target_name in df.columns else [c for c in df.columns if c != "date"][0]
                    detrended_target = "detrended_target"

                    if coeffs is None and smooth_w is not None:
                        df["smooth"] = df[price_col].rolling(smooth_w, min_periods=1).mean()
                        df[detrended_target] = df["smooth"].diff()
                    elif coeffs is not None:
                        df[detrended_target] = df[price_col] - apply_trend(df["date"], coeffs, ref_date, freq_val)
                    elif ref_date is not None:
                        df[detrended_target] = df[price_col]
                    else:
                        df[detrended_target] = df[price_col].diff()

                    progress(0.20, desc="Engineering features...")
                    other_cols = [
                        c for c in df.columns 
                        if c not in [target_name, "date", detrended_target, "smooth", "is_holiday"] 
                        and np.issubdtype(df[c].dtype, np.number) 
                        and df[c].nunique() > 10
                    ]
                    
                    df_feat, feature_cols, seasonal_cols = create_features(
                        df, detrended_target, lags, windows, freq_val,
                        other_cols=other_cols, external_cols=external_cols_present or None,
                        add_seasonal=add_seasonal_feat
                    )

                    all_feature_cols = list(dict.fromkeys(feature_cols))
                    valid_cols = [detrended_target] + [c for c in all_feature_cols if c in df_feat.columns]
                    df_feat = df_feat.dropna(subset=valid_cols)

                    if auto_select and len(all_feature_cols) > 5:
                        progress(0.30, desc="Selecting features...")
                        all_feature_cols = select_features(df_feat, all_feature_cols, split_date=split_date, max_corr=0.92)

                    entries, _ = _log(entries, f"Features engineered: {len(all_feature_cols)} total "
                                               f"({len([c for c in all_feature_cols if c.startswith('ext_')])} external)")

                    progress(0.40, desc="Training models...")
                    results, trained_models, best_model, best_scaler, best_name, best_std, best_mae = train_all(
                        df_feat, detrended_target, all_feature_cols, selected_models,
                        split_date=split_date, use_ensemble=use_ens, ensemble_type=ens_type,
                        fast=fast, progress=progress,
                        train_log=lambda msg: log.append(msg)
                    )

                    # ═══ AUTO-REMOVE GATE (v13) ═══
                    external_status = "none"
                    if external_cols_present and best_mae is not None:
                        progress(0.80, desc="Running auto-remove gate...")
                        v12_cols = [c for c in all_feature_cols if not c.startswith("ext_")]
                        (_, trained_v12, best_model_v12, best_scaler_v12, best_name_v12,
                         best_std_v12, best_mae_v12) = train_all(
                            df_feat, detrended_target, v12_cols, selected_models,
                            split_date=split_date, use_ensemble=False, fast=fast,
                            train_log=lambda msg: log.append(msg)
                        )
                        if best_mae_v12 is not None and best_mae_v12 < best_mae * 0.95:
                            pct = ((best_mae / best_mae_v12) - 1) * 100
                            log.append(f"⚠️ Auto-remove: External features HURT accuracy by {pct:.1f}%. Removing them.")
                            all_feature_cols = v12_cols
                            trained_models = trained_v12
                            best_model, best_scaler, best_name = best_model_v12, best_scaler_v12, best_name_v12
                            best_std, best_mae = best_std_v12, best_mae_v12
                            external_cols_present = []
                            external_status = "removed (hurt accuracy)"
                        elif best_mae_v12 is not None and best_mae < best_mae_v12 * 0.95:
                            pct = ((best_mae_v12 / best_mae) - 1) * 100
                            log.append(f"✓ Auto-remove: External features IMPROVE accuracy by {pct:.1f}%. Keeping them.")
                            external_status = "kept (improved accuracy)"
                        else:
                            log.append("→ Auto-remove: External features are neutral. Keeping them.")
                            external_status = "kept (neutral)"

                    if not results:
                        entries, log_text = _log(entries, "Training failed — all models had R² ≤ 0", ok=False)
                        for line in log:
                            entries, _ = _log(entries, line)
                        return ("✗ Training failed — all models had R² ≤ 0", "No dataset", None, None, pd.DataFrame(),
                                state, entries, log_text)

                    progress(0.90, desc="Computing final metrics...")
                    original_metrics, _ = compute_original_scale_metrics(
                        trained_models, df_feat, all_feature_cols,
                        target_name, coeffs, ref_date, freq_val, split_date,
                        smooth_w=smooth_w or 0,
                        pred_col=detrended_target
                    )

                    # Use detrended training metrics (real model performance)
                    comp_data = []
                    for name, tup in trained_models.items():
                        mae = tup[2] if len(tup) > 2 else None
                        r2 = tup[3] if len(tup) > 3 else None
                        comp_data.append({"Model": name, "MAE": mae, "R-squared": r2})
                    comp_df = pd.DataFrame(comp_data).sort_values("MAE") if comp_data else pd.DataFrame()
                    if not comp_df.empty:
                        comp_df = comp_df.set_index("Model")

                    non_naive = comp_df[~comp_df.index.str.startswith("Naive")]
                    if "R-squared" in non_naive.columns:
                        non_naive = non_naive[non_naive["R-squared"] > 0]
                    if not non_naive.empty:
                        best_name = non_naive.index[0]
                        best_mae = non_naive.loc[best_name, "MAE"]
                    # Use detrended residual stats from the trained model (accurate)
                    if best_name in trained_models:
                        bt = trained_models[best_name]
                        best_std = bt[4] if len(bt) > 4 else best_std
                        best_q10 = bt[5] if len(bt) > 5 else -1.28 * best_std
                        best_q90 = bt[6] if len(bt) > 6 else 1.28 * best_std
                    else:
                        best_std = original_metrics.get(best_name, {}).get("std_res", 0)
                        best_q10 = original_metrics.get(best_name, {}).get("residual_q10", -1.28 * best_std)
                        best_q90 = original_metrics.get(best_name, {}).get("residual_q90", 1.28 * best_std)

                    # Model comparison plot
                    fig_comp = go.Figure()
                    metrics_display = ["MAE", "RMSE", "R-squared", "MAPE%"]
                    colors = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6"]
                    for i, metric in enumerate(metrics_display):
                        if metric in comp_df.columns:
                            fig_comp.add_trace(go.Bar(
                                x=comp_df.index, y=comp_df[metric],
                                name=metric, marker_color=colors[i % len(colors)]
                            ))
                    fig_comp.update_layout(
                        title="Model Comparison — Lower is Better (except R²)",
                        template="plotly_dark", barmode="group", height=400
                    )

                    # Feature importance plot
                    fig_feat = None
                    best_model_obj = trained_models.get(best_name, (None,))[0]
                    if best_model_obj is not None and len(external_cols_present) > 0:
                        importance_split = feature_importance_by_horizon(
                            best_model_obj, all_feature_cols, external_cols_present
                        )
                        own = importance_split.get("own_graph", [])[:15]
                        ext = importance_split.get("external", [])[:15]
                        if own or ext:
                            fig_feat = go.Figure()
                            if own:
                                fig_feat.add_trace(go.Bar(
                                    x=[o["feature"] for o in own],
                                    y=[o["importance"] for o in own],
                                    name="Own-Graph Features", marker_color="#10b981"
                                ))
                            if ext:
                                fig_feat.add_trace(go.Bar(
                                    x=[e["feature"] for e in ext],
                                    y=[e["importance"] for e in ext],
                                    name="External Features", marker_color="#ef4444"
                                ))
                            fig_feat.update_layout(
                                title="Feature Importance: Own-Graph vs External",
                                template="plotly_dark", barmode="group", height=400
                            )
                    elif best_model_obj is not None:
                        if hasattr(best_model_obj, "feature_importances_"):
                            imp = best_model_obj.feature_importances_
                            imp_df = pd.DataFrame({
                                "feature": all_feature_cols[:len(imp)],
                                "importance": imp
                            }).sort_values("importance", ascending=False).head(15)
                            fig_feat = px.bar(
                                imp_df, x="importance", y="feature", orientation="h",
                                title="Top 15 Feature Importances",
                                template="plotly_dark", height=400
                            )

                    # Update state
                    state.update({
                        "model": best_model_obj,
                        "scaler": best_scaler,
                        "feature_cols": all_feature_cols,
                        "target_col": target_name,
                        "detrended_target": detrended_target,
                        "trend_coeffs": coeffs,
                        "ref_date": ref_date,
                        "std_residual": best_std,
                        "residual_q10": best_q10,
                        "residual_q90": best_q90,
                        "best_model_name": best_name,
                        "lags": lags, "windows": windows,
                        "best_mae": best_mae, "freq": freq_val,
                        "trained_models": trained_models,
                        "detrend_method": detrend_meth,
                        "seasonal_cols": seasonal_cols,
                        "external_cols": external_cols_present,
                        "smooth_w": smooth_w,
                    })

                    dataset_info_text = (
                        f"Product: {target_name}\n"
                        f"Training rows: {len(df_feat):,}\n"
                        f"Total features: {len(all_feature_cols)}\n"
                        f"External features: {len(external_cols_present)}\n"
                        f"External status: {external_status}\n"
                        f"Best model: {best_name}\n"
                        f"MAE: {best_mae:,.0f}"
                    )

                    status_msg = (
                        f"✓ Trained {len(results)} models | "
                        f"Best: {best_name} | "
                        f"MAE: {best_mae:,.0f} | "
                        f"External: {external_status}"
                    )

                    for line in log:
                        entries, _ = _log(entries, line)
                    entries, log_text = _log(entries, f"Training complete — Best: {best_name} (MAE={best_mae:,.0f})")

                    return (status_msg, dataset_info_text, fig_comp, fig_feat, comp_df,
                            state, entries, log_text)

                except Exception as e:
                    entries, log_text = _log_err(entries, "Training crashed", e)
                    for line in log:
                        entries, _ = _log(entries, line)
                    return (f"✗ ERROR: {e}", "No dataset", None, None, pd.DataFrame(),
                            state, entries, log_text)

            def auto_params_fn(df, target_name, freq_val, entries):
                try:
                    if df is None or df.empty:
                        entries, log_text = _log(entries, "No data for auto params", ok=False)
                        return ("No data", "1,7,14,30", "7,14,30,60", entries, log_text)
                    lags, windows = auto_params(df, target_name, freq_val)
                    entries, log_text = _log(entries, f"Auto params: lags={lags}, windows={windows}")
                    return ("✓ Auto params set", ",".join(str(x) for x in lags),
                            ",".join(str(x) for x in windows), entries, log_text)
                except Exception as e:
                    entries, log_text = _log_err(entries, "Auto params failed", e)
                    return (f"✗ ERROR: {e}", "1,7,14,30", "7,14,30,60", entries, log_text)

            def save_models_fn(state, entries):
                try:
                    trained = state.get("trained_models", {})
                    product = state.get("target_col", "unknown")
                    if not trained:
                        entries, log_text = _log(entries, "No trained models to save", ok=False)
                        return ("No models to save", entries, log_text)
                    saved = save_trained_models(trained, product)
                    entries, log_text = _log(entries, f"Saved {len(saved)} model(s) for {product}")
                    return (f"✓ Saved {len(saved)} model(s)", entries, log_text)
                except Exception as e:
                    entries, log_text = _log_err(entries, "Save failed", e)
                    return (f"✗ ERROR: {e}", entries, log_text)

            auto_params_btn.click(
                fn=auto_params_fn,
                inputs=[data_state, target, freq, log_entries],
                outputs=[train_status, lags_input, windows_input, log_entries, log_display]
            )

            train_btn.click(
                fn=train_fn,
                inputs=[data_state, target, lags_input, windows_input, fast_mode, add_seasonal,
                        auto_feature_select, detrend_method, models_check, freq, use_ensemble, ensemble_type,
                        model_state, log_entries],
                outputs=[train_status, dataset_info_display, comparison_plot, feature_plot,
                        comparison_table, model_state, log_entries, log_display]
            )

            save_models_btn.click(
                fn=save_models_fn,
                inputs=[model_state, log_entries],
                outputs=[save_status, log_entries, log_display]
            )

        # ═══════════════════════════════════════════════════════════
        # TAB 4: FORECAST
        # ═══════════════════════════════════════════════════════════
        with gr.Tab("🔮 Forecast"):
            gr.Markdown("### 🔮 Generate Forecast from Trained Model")

            with gr.Row():
                with gr.Column():
                    horizon_input = gr.Number(value=14, label="Forecast Horizon (days)", precision=0, minimum=1)
                    model_selector = gr.Dropdown(
                        choices=[], label="Trained Model",
                        info="Auto-populated after training — select the model to use"
                    )
                    refresh_models_btn = gr.Button("🔄 Refresh Model List")
                with gr.Column():
                    forecast_hybrid = gr.Checkbox(
                        value=True, label="Blend with Simple Methods",
                        info="ON: blends ML with moving average/trend for better accuracy. OFF: pure ML."
                    )
                    anomaly_threshold = gr.Slider(
                        value=2, minimum=1, maximum=5, step=0.5,
                        label="Anomaly Sensitivity",
                        info="Lower values flag more forecast steps as potentially unusual"
                    )
                    forecast_btn = gr.Button("📈 Generate Forecast", variant="primary", size="lg")

            forecast_status = gr.Textbox(label="📊 Forecast Result")
            forecast_alerts = gr.Textbox(label="⚠️ Anomaly Alerts", lines=3, elem_classes=["alert-box"])
            forecast_chart = gr.Plot(label="📉 Forecast Chart")
            with gr.Row():
                forecast_table = gr.Dataframe(label="📋 Forecast Values")
                download_btn = gr.File(label="💾 Download CSV")

            def refresh_model_list_fn(state):
                try:
                    trained = state.get("trained_models", {})
                    names = list(trained.keys())
                    best = state.get("best_model_name")
                    default = best if best and best in names else (names[0] if names else None)
                    return gr.update(choices=names, value=default), default
                except Exception:
                    return gr.update(), None

            refresh_models_btn.click(
                fn=refresh_model_list_fn,
                inputs=[model_state],
                outputs=[model_selector, selected_model_name_state]
            )

            def forecast_fn(df, target_name, horizon, selected_model_name, state, thresh, hybrid, entries):
                try:
                    print(f"[FORECAST] Starting: target={target_name}, horizon={horizon}, model={selected_model_name}")
                    entries, _ = _log(entries, f"Generating forecast: {int(horizon)}d | Hybrid={hybrid}")

                    if df is None or df.empty:
                        entries, log_text = _log(entries, "No data loaded", ok=False)
                        return ("✗ No data loaded", "", None, pd.DataFrame(), None, entries, log_text)

                    trained = state.get("trained_models", {})
                    if not trained:
                        entries, log_text = _log(entries, "No trained models available", ok=False)
                        return ("✗ No trained models — train first", "", None, pd.DataFrame(), None, entries, log_text)

                    if selected_model_name is None or selected_model_name not in trained:
                        entries, log_text = _log(entries, "Please select a valid model", ok=False)
                        return ("✗ Select a valid model", "", None, pd.DataFrame(), None, entries, log_text)

                    model_tup = trained[selected_model_name]
                    model, scaler = model_tup[0], model_tup[1]

                    df = df.copy()
                    df["date"] = pd.to_datetime(df["date"])
                    
                    # Resolve price column - use trained target_col from state, not the dropdown
                    price_col = state.get("target_col", target_name)
                    if price_col not in df.columns:
                        if "price" in df.columns:
                            price_col = "price"
                        else:
                            for c in df.columns:
                                if c != "date" and not c.startswith("ext_") and pd.api.types.is_numeric_dtype(df[c]):
                                    price_col = c
                                    break
                    
                    print(f"[FORECAST] Using price column: '{price_col}'")

                    external_cols = state.get("external_cols") or None
                    detrended_target = state.get("detrended_target", "price_detrended")
                    feature_cols = state.get("feature_cols", [])
                    lags = state.get("lags", [1, 7, 14, 30])
                    windows = state.get("windows", [7, 14, 30, 60])
                    freq_val = state.get("freq", "D")
                    coeffs = state.get("trend_coeffs")
                    ref_date = state.get("ref_date")
                    smooth_w = state.get("smooth_w")
                    std_res = state.get("std_residual", 0)
                    res_q10 = state.get("residual_q10", -1.28 * std_res)
                    res_q90 = state.get("residual_q90", 1.28 * std_res)

                    pred_df = iterative_forecast(
                        df, price_col, detrended_target, feature_cols,
                        model, scaler, std_res, int(horizon),
                        lags, windows, freq_val,
                        coeffs, ref_date,
                        residual_q10=res_q10, residual_q90=res_q90,
                        external_cols=external_cols,
                        smooth_w=smooth_w,
                    )

                    if pred_df.empty:
                        entries, log_text = _log(entries, "Forecast generation failed", ok=False)
                        return ("✗ Forecast failed", "", None, pd.DataFrame(), None, entries, log_text)

                    print(f"[FORECAST] Generated {len(pred_df)} forecast points")

                    # Hybrid blending
                    blend_note = ""
                    if hybrid and price_col in df.columns:
                        series = df[price_col].dropna().values
                        if len(series) >= 10:
                            dummy = SmartHybridPredictor(
                                ml_func=lambda *a, **kw: (pred_df, "", "", {}, {}, pd.DataFrame(), pd.DataFrame()),
                                test_window=60
                            )
                            methods = dummy._simple_methods()
                            best_simple_name = None
                            best_simple_score = float("inf")
                            for name, fn in methods.items():
                                score = dummy._evaluate_method(series, int(horizon), fn)
                                if score < best_simple_score:
                                    best_simple_score = score
                                    best_simple_name = name
                            if best_simple_name:
                                simple_pred = methods[best_simple_name](series, int(horizon))
                                ml_pred = pred_df["forecast"].values
                                if best_simple_score > 0 and std_res > 0:
                                    ml_weight = best_simple_score / (best_simple_score + std_res)
                                    ml_weight = max(0.3, min(0.7, ml_weight))
                                else:
                                    ml_weight = 0.6
                                simple_weight = 1.0 - ml_weight
                                blended = ml_weight * ml_pred + simple_weight * simple_pred
                                last_actual = float(series[-1])
                                blended = blended - blended[0] + last_actual
                                if "lower" in pred_df.columns and "upper" in pred_df.columns:
                                    shift = blended - ml_pred
                                    pred_df["lower"] = pred_df["lower"] + shift
                                    pred_df["upper"] = pred_df["upper"] + shift
                                pred_df["forecast"] = blended
                                blend_note = f" | Blended: ML {ml_weight:.0%} + {best_simple_name} {simple_weight:.0%}"
                                print(f"[FORECAST] Blended with {best_simple_name}, ML weight={ml_weight:.0%}")

                    fig = create_forecast_plot(df, price_col, pred_df)

                    # Anomaly detection
                    alert_text = "✓ No unusually large jumps detected in this forecast."
                    if not pred_df.empty and std_res and price_col in df.columns:
                        last_actual = float(df[price_col].dropna().iloc[-1])
                        vals = pred_df["forecast"].values
                        prev = np.concatenate([[last_actual], vals[:-1]])
                        step_moves = vals - prev
                        z = np.abs(step_moves) / max(std_res, 1e-9)
                        flagged = np.where(z > float(thresh))[0]
                        if len(flagged) > 0:
                            lines = [f"⚠️ {len(flagged)} forecast step(s) show larger-than-usual moves:"]
                            for idx in flagged[:10]:
                                d = pd.to_datetime(pred_df["date"].iloc[idx]).strftime("%Y-%m-%d")
                                lines.append(f"  • {d}: move = {step_moves[idx]:,.0f} (z = {z[idx]:.1f})")
                            if len(flagged) > 10:
                                lines.append(f"  ... and {len(flagged) - 10} more")
                            alert_text = "\n".join(lines)

                    csv_path = os.path.join(CACHE_DIR, "forecast_output.csv")
                    pred_df.to_csv(csv_path, index=False)

                    ext_note = f" | External: {len(external_cols)}" if external_cols else ""
                    status = f"✓ Forecast: {selected_model_name}{ext_note}{blend_note}"
                    
                    entries, log_text = _log(entries, "Forecast generated successfully")
                    return (status, alert_text, fig, pred_df, csv_path, entries, log_text)

                except Exception as e:
                    entries, log_text = _log_err(entries, "Forecast crashed", e)
                    return (f"✗ ERROR: {e}", "", None, pd.DataFrame(), None, entries, log_text)

            forecast_btn.click(
                fn=forecast_fn,
                inputs=[data_state, target, horizon_input, model_selector, model_state,
                       anomaly_threshold, forecast_hybrid, log_entries],
                outputs=[forecast_status, forecast_alerts, forecast_chart, forecast_table,
                        download_btn, log_entries, log_display]
            )

        # ═══════════════════════════════════════════════════════════
        # TAB 5: BACKTEST
        # ═══════════════════════════════════════════════════════════
        with gr.Tab("🧪 Backtest"):
            gr.Markdown("### 🧪 Historical Accuracy Test — Validate Before Trusting")

            with gr.Row():
                bt_models = gr.Dropdown(
                    choices=["All Models", "Ridge", "Lasso", "ElasticNet",
                             "Random Forest", "Gradient Boosting"] +
                            (["XGBoost"] if XGB_AVAILABLE else []) +
                            (["LightGBM"] if LGBM_AVAILABLE else []),
                    value="All Models", label="Model to Test"
                )
                bt_horizon = gr.Number(value=30, label="Test Window (days)", precision=0, minimum=7)
                bt_btn = gr.Button("🧪 Run Backtest", variant="primary", size="lg")

            bt_status = gr.Textbox(label="📊 Backtest Result")
            with gr.Row():
                bt_actual_vs_pred = gr.Plot(label="📈 Actual vs Predicted")
                bt_errors = gr.Plot(label="📊 Prediction Errors")
            bt_metrics = gr.Dataframe(label="📋 Summary Metrics")
            gr.Markdown("_💡 Tip: CI Coverage near 80% means confidence intervals are well-calibrated._")
            bt_details = gr.Dataframe(label="📋 Day-by-Day Results", max_height=250)

            def run_backtest_fn(df, target_name, model_name, horizon_val, lags_str, windows_str,
                            detrend_meth, freq_val, add_seasonal_val, use_ens, ens_type, fast, entries):
                try:
                    entries, _ = _log(entries, f"Running backtest: {model_name}, {int(horizon_val)}d window")

                    if df is None or df.empty:
                        entries, log_text = _log(entries, "No data loaded", ok=False)
                        return ("✗ No data loaded", None, None, pd.DataFrame(), pd.DataFrame(), entries, log_text)

                    try:
                        lags = [int(x.strip()) for x in lags_str.split(",") if x.strip().isdigit()]
                        windows = [int(x.strip()) for x in windows_str.split(",") if x.strip().isdigit()]
                    except:
                        lags, windows = [1, 7, 14, 30], [7, 14, 30]

                    models = [m for m in ["Ridge", "Lasso", "ElasticNet", "Random Forest",
                                        "Gradient Boosting", "XGBoost", "LightGBM"]
                            if model_name == "All Models" or m == model_name]

                    bt_df = backtest(
                        df, target_name, lags, windows, freq_val, detrend_meth,
                        models, horizon=int(horizon_val), n_folds=3, fast=fast
                    )

                    if bt_df.empty:
                        entries, log_text = _log(entries, "Not enough data for backtest", ok=False)
                        return ("✗ Not enough data", None, None, pd.DataFrame(), pd.DataFrame(), entries, log_text)

                    # backtest() returns: fold, best_model, mae, rmse, n_test
                    avg_mae = bt_df["mae"].mean()
                    avg_rmse = bt_df["rmse"].mean()
                    total_days = bt_df["n_test"].sum() if "n_test" in bt_df.columns else 0

                    # ── Plot 1: MAE per fold ──
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Bar(
                        x=[f"Fold {int(f)}" for f in bt_df["fold"]],
                        y=bt_df["mae"],
                        name="MAE",
                        marker_color="#ef4444",
                        text=[f"{v:,.0f}" for v in bt_df["mae"]],
                        textposition="outside"
                    ))
                    if "best_model" in bt_df.columns:
                        fig_compare.add_trace(go.Scatter(
                            x=[f"Fold {int(f)}" for f in bt_df["fold"]],
                            y=bt_df["mae"] * 1.05,
                            mode="text",
                            text=bt_df["best_model"],
                            textposition="top center",
                            textfont=dict(size=11, color="white"),
                            showlegend=False
                        ))
                    fig_compare.update_layout(
                        title=f"Backtest MAE per Fold (Avg: {avg_mae:,.0f})",
                        template="plotly_dark", height=400,
                        yaxis_title="MAE"
                    )

                    # ── Plot 2: RMSE per fold ──
                    fig_err = go.Figure()
                    fig_err.add_trace(go.Bar(
                        x=[f"Fold {int(f)}" for f in bt_df["fold"]],
                        y=bt_df["rmse"],
                        name="RMSE",
                        marker_color="#3b82f6",
                        text=[f"{v:,.0f}" for v in bt_df["rmse"]],
                        textposition="outside"
                    ))
                    fig_err.update_layout(
                        title=f"Backtest RMSE per Fold (Avg: {avg_rmse:,.0f})",
                        template="plotly_dark", height=300
                    )

                    # ── Metrics table ──
                    best_fold = bt_df.loc[bt_df["mae"].idxmin()]
                    metrics_df = pd.DataFrame({
                        "Metric": ["Avg MAE", "Avg RMSE", "Best MAE", "Worst MAE", "Best Model", "Test Days", "Folds"],
                        "Value": [
                            f"{avg_mae:,.0f}",
                            f"{avg_rmse:,.0f}",
                            f"{bt_df['mae'].min():,.0f}",
                            f"{bt_df['mae'].max():,.0f}",
                            str(best_fold.get("best_model", "N/A")),
                            str(total_days),
                            str(len(bt_df))
                        ]
                    })

                    # ── Details table ──
                    detail_cols = [c for c in ["fold", "best_model", "mae", "rmse", "n_test"] if c in bt_df.columns]
                    details = bt_df[detail_cols].round(2)
                    details.columns = [c.replace("_", " ").title() for c in details.columns]

                    status_msg = (
                        f"✓ Backtest: {len(bt_df)} folds, {total_days} test days | "
                        f"Avg MAE = {avg_mae:,.0f} | "
                        f"Best MAE = {bt_df['mae'].min():,.0f}"
                    )
                    
                    entries, log_text = _log(entries, f"Backtest done — Avg MAE={avg_mae:,.0f}")
                    return (status_msg, fig_compare, fig_err, metrics_df, details, entries, log_text)

                except Exception as e:
                    entries, log_text = _log_err(entries, "Backtest crashed", e)
                    import traceback
                    traceback.print_exc()
                    return (f"✗ ERROR: {e}", None, None, pd.DataFrame(), pd.DataFrame(), entries, log_text) 

            bt_btn.click(
                fn=run_backtest_fn,
                inputs=[data_state, target, bt_models, bt_horizon, lags_input, windows_input,
                       detrend_method, freq, add_seasonal, use_ensemble, ensemble_type, fast_mode, log_entries],
                outputs=[bt_status, bt_actual_vs_pred, bt_errors, bt_metrics, bt_details,
                         log_entries, log_display]
            )

        # ═══════════════════════════════════════════════════════════
        # GLOBAL: Refresh Products + Sync Dropdowns
        # ═══════════════════════════════════════════════════════════

        def refresh_products_with_log(entries):
            try:
                products, source = load_products()
                df = pd.DataFrame(list(products.items()), columns=["Product", "ID"])
                names = sorted(products.keys())
                first = names[0] if names else None
                entries, log_text = _log(entries, f"Products refreshed: {len(products)} from {source}")
                return (products, f"✓ Loaded {len(products)} products from {source}", df,
                        gr.update(choices=names, value=first),
                        gr.update(choices=names),
                        gr.update(choices=names, value=first),
                        first, entries, log_text)
            except Exception as e:
                entries, log_text = _log_err(entries, "Product refresh failed", e)
                return ({}, f"✗ ERROR: {e}", pd.DataFrame(),
                        gr.update(), gr.update(), gr.update(), None,
                        entries, log_text)

        refresh_products_btn.click(
            fn=refresh_products_with_log,
            inputs=[log_entries],
            outputs=[products_state, products_status, products_table, target, extras,
                    qp_product, current_product_state, log_entries, log_display]
        )

        target.change(
            fn=sync_dropdowns,
            inputs=[target, products_state],
            outputs=[target, extras, qp_product, current_product_state]
        )

    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(
        share=False, 
        server_name="127.0.0.1", 
        server_port=7861,
        theme=gr.themes.Soft(),
        css="""
        .log-box textarea { font-family: 'Consolas', 'Courier New', monospace !important; font-size: 11px !important; }
        .alert-box textarea { font-family: 'Consolas', 'Courier New', monospace !important; }
        """
    )