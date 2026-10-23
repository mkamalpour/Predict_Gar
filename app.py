"""PredictGar: local commodity forecasting interface and workspace callbacks."""

import os, re, unicodedata
import inspect
from html import escape
from datetime import datetime
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px

# Data Pipeline imports
from data_pipeline import (
    Fetcher,
    Cache,
    preprocess,
    create_features,
    detrend_series,
    apply_trend,
    merge_external_features,
    fetch_related_prices,
    get_product_profile,
    check_data_quality,
    auto_params,
    select_features,
    CACHE_DIR,
    MODELS_DIR,
    FORECASTS_DIR,
    EXTERNAL_FEATURE_REGISTRY,
)

# Model Pipeline imports
from model_pipeline import (
    train_all,
    compute_original_scale_metrics,
    iterative_forecast,
    create_forecast_plot,
    save_trained_models,
    load_trained_model,
    list_saved_models,
    backtest,
    quick_predict,
    quick_predict_hybrid,
    detect_anomalies,
    format_anomaly_report,
    feature_importance_by_horizon,
    compute_split_date,
    XGB_AVAILABLE,
    LGBM_AVAILABLE,
    DEFAULT_MODELS,
    SmartHybridPredictor,
    analyze_product_relationships,
    EXTERNAL_LABELS,
    smart_feature_selection,
)


# LOGGING HELPERS


def _ts():
    return datetime.now().strftime("%H:%M:%S")


def _log(entries, msg, ok=True):
    prefix = "✓" if ok else "✗"
    entries = list(entries or [])[-199:]
    entries.append(f"[{_ts()}] {prefix} {msg}")
    return entries, "\n".join(entries)


def _log_err(entries, msg, exc):
    entries = list(entries or [])[-199:]
    entries.append(f"[{_ts()}] ✗ {msg}")
    entries.append(f"  {type(exc).__name__}: {exc}")
    return entries, "\n".join(entries)


# CREATE APP


def prepare_workspace(df, target_name, freq, fetcher, cache):
    if df is not None and not df.empty:
        if target_name not in df.columns:
            raise ValueError(
                "The selected product differs from the loaded dataset. Load its data first."
            )
        return preprocess(df, target_name, freq=freq)
    products, _ = fetcher.discover_products()
    if target_name not in products:
        raise ValueError("Select a valid product or upload a CSV.")
    raw, source = cache.update(products[target_name], force=False)
    if raw is None or raw.empty:
        raise ValueError("Market data unavailable. Upload a real CSV or retry later.")
    raw = raw.rename(columns={"price": target_name})
    frame = preprocess(raw, target_name, freq=freq)
    frame = merge_external_features(frame, target_name, freq=freq)
    return fetch_related_prices(frame, target_name, fetcher, cache, freq=freq)


APP_CSS = """
.gradio-container { width: 100% !important; min-width: 0 !important; box-sizing: border-box; }
.gradio-container .main, .gradio-container main, .gradio-container .wrap, .gradio-container .column, .gradio-container .row { min-width: 0 !important; max-width: 100%; }
@media(max-width: 600px) { .gradio-container .row { flex-wrap: wrap !important; } .gradio-container .row > .column { flex-basis: 100% !important; width: 100% !important; } .gradio-container .main { padding: 0 !important; } .gradio-container [role="tablist"] { flex-wrap: wrap !important; } }

.gradio-container [role="tablist"] { flex-wrap: wrap !important; gap: 4px; }
.gradio-container [role="tab"] { border-radius: 4px; padding: 10px 15px; color: #294956 !important; }
.gradio-container [role="tab"][aria-selected="true"] { background: #173d49 !important; color: #fff !important; }
.gradio-container [data-testid="block-label"], .gradio-container [data-testid="block-label"] span { color: #294956 !important; background: #edf4f2 !important; }
.gradio-container [data-testid="block-info"], .gradio-container .info-text, .gradio-container .subdued, .gradio-container [data-testid="upload-text"], .gradio-container .or, .gradio-container .label p { color: #475e66 !important; }

.gradio-container { max-width: 1380px !important; margin: auto; padding: 24px !important; }
body { background: #f4f5f2; }
.pg-header { border-top: 4px solid #147d79; padding: 25px 0 23px; border-bottom: 1px solid #cbd5d7; margin-bottom: 10px; }
.pg-eyebrow, .pg-meta { color: #43616b; font-size: 11px; letter-spacing: .13em; font-weight: 650; }
.pg-header h1 { color: #173340; font-size: clamp(26px, 3vw, 39px); line-height: 1.2; margin: 10px 0; font-weight: 650; letter-spacing: -.035em; }
.pg-header h1 span { color: #16817b; font-weight: 400; }
.pg-header p { color: #475e66; font-size: 15px; line-height: 1.65; margin: 0 0 18px; }
.tab-nav { gap: 4px; border-bottom: 2px solid #d4dedd !important; padding: 8px 0 !important; flex-wrap: wrap !important; }
.tab-nav button { padding: 10px 16px !important; font-weight: 600 !important; border-radius: 5px !important; }
.tab-nav button.selected { background: #173d49 !important; color: #fff !important; }
button:focus-visible, input:focus-visible, textarea:focus-visible { outline: 3px solid #16817b !important; outline-offset: 3px; }
.log-box textarea { font-family: Consolas, monospace !important; font-size: 12px !important; line-height: 1.6 !important; }
textarea, input { unicode-bidi: plaintext; }
.gradio-container .plot-container { min-height: 220px; }
.gradio-container .chatbot.md p, .gradio-container footer a, .gradio-container footer button { color: #475e66 !important; }
.gradio-container .tabitem { padding-top: 18px; }
.plain-table-scroll { max-height: 360px; overflow: auto; background: #fff; color: #173340; border: 1px solid #cbd5d7; border-radius: 6px; }
.plain-table-scroll:focus-visible { outline: 3px solid #16817b; outline-offset: 3px; }
.plain-results { width: 100%; border-collapse: collapse; font-size: 13px; }
.plain-results caption { text-align: left; padding: 12px; font-weight: 650; color: #173340; }
.plain-results th { background: #edf4f2; color: #294956; text-align: left; position: sticky; top: 0; }
.plain-results th, .plain-results td { padding: 9px 12px; border-bottom: 1px solid #dce4e3; white-space: nowrap; }
.plain-table-note { color: #475e66; font-size: 12px; }
@media(max-width: 600px) { .gradio-container { padding: 12px !important; } .pg-header { padding-top: 18px; } .pg-meta { line-height: 1.8; letter-spacing: .07em; } .tab-nav button { padding: 9px 11px !important; } }
@media(prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; } }
"""


def app_theme():
    return gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="slate",
        neutral_hue="slate",
        font=["Segoe UI", "Tahoma", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    ).set(
        body_background_fill="#f4f5f2",
        block_background_fill="#ffffff",
        button_primary_background_fill="#176c68",
        button_primary_background_fill_hover="#105650",
        button_primary_text_color="#ffffff",
        block_radius="6px",
    )


def _presentation_options(callable_):
    """Gradio 5 styles Blocks; Gradio 6 styles launch instead."""
    parameters = inspect.signature(callable_).parameters
    return {key: value for key, value in {"theme": app_theme(), "css": APP_CSS}.items()
            if key in parameters}


def launch_app(app, **overrides):
    """Launch locally with only options supported by the installed Gradio."""
    options = {"share": False, "server_name": "127.0.0.1", "server_port": 7861,
               **_presentation_options(app.launch)}
    if "footer_links" in inspect.signature(app.launch).parameters:
        options["footer_links"] = []
    options.update(overrides)
    return app.launch(**options)


def _table_html(value, label, max_rows=200):
    """Escaped, bounded semantic table; the original grid retains all rows."""
    title = escape(str(label))
    if value is None:
        return f"<p><strong>{title}</strong>: No results yet. Run this workflow to populate the table.</p>"
    if isinstance(value, dict):
        value = pd.DataFrame(value.get("data", []), columns=value.get("headers"))
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    if frame.empty:
        return f"<p><strong>{title}</strong>: No results yet. Run this workflow to populate the table.</p>"
    view = frame.head(max_rows)
    table = view.to_html(index=False, border=0, escape=True, na_rep="Not available", classes="plain-results")
    table = table.replace("<thead>", f"<caption>{title}</caption><thead>", 1)
    note = (f"Showing the first {len(view)} of {len(frame)} rows. Use the interactive table or available CSV export for all rows."
            if len(frame) > max_rows else f"{len(frame)} rows.")
    return (f'<div class="plain-table-scroll" role="region" tabindex="0" aria-label="{title} table">'
            + table + f'</div><p class="plain-table-note">{note}</p>')


def _private_event_options(method):
    return ({"api_visibility": "private"} if "api_visibility" in inspect.signature(method).parameters
            else {"api_name": False})


def create_app():

    fetcher = Fetcher()
    cache = Cache(fetcher)

    def load_products():
        return fetcher.discover_products(allow_network=False)

    initial_products, initial_source = load_products()
    product_names = (
        sorted(initial_products.keys()) if initial_products else ["(no products found)"]
    )
    default_target = product_names[0] if product_names else None

    # Available models (with XGBoost/LightGBM if installed)
    models_available = [
        "Ridge",
        "Lasso",
        "ElasticNet",
        "Random Forest",
        "Gradient Boosting",
    ]
    if XGB_AVAILABLE:
        models_available.append("XGBoost")
    if LGBM_AVAILABLE:
        models_available.append("LightGBM")

    # Retain sortable grids as an option, with readable semantic tables by default.
    result_tables = []
    def result_table(**kwargs):
        label = kwargs.get("label", "Results")
        initial = kwargs.get("value")
        with gr.Group():
            readable = gr.HTML(value=_table_html(initial, label))
            grid = gr.Dataframe(**kwargs, visible=False)
        def render_table(value):
            return _table_html(value, label)
        grid.change(render_table, grid, readable, queue=False, **_private_event_options(grid.change))
        result_tables.append((grid, readable))
        return grid

    with gr.Blocks(
        title="PredictGar | Commodity forecasting", analytics_enabled=False,
        **_presentation_options(gr.Blocks)
    ) as demo:
        # HEADER
        gr.HTML(
            """<header class="pg-header"><div class="pg-eyebrow">COMMODITY RESEARCH WORKBENCH</div><h1>PredictGar<span> / </span>Price intelligence</h1><p>Explore Iranian commodity prices, compare forecasting models, and test them against history.</p><div class="pg-meta">LOCAL WORKSPACE &nbsp; / &nbsp; HISTORICAL DATA &nbsp; / &nbsp; MODEL VALIDATION</div></header>"""
        )
        with gr.Accordion("Activity log", open=False):
            log_display = gr.Textbox(
                label="Processing details",
                lines=5,
                max_lines=15,
                interactive=False,
                elem_classes=["log-box"],
                value=f"[{_ts()}] Ready: {len(initial_products or {})} products loaded from {initial_source}",
            )

        # GETTING STARTED
        with gr.Accordion("Start here: choose your workflow", open=False):
            gr.Markdown(
                "**Quick Predict** fetches a product's history and compares available models. For your own CSV,\nopen **Data & Charts**, upload `date` and `price` columns, then use **Training** and **Forecast**.\n\n**Backtest** evaluates historical forecasts. **Analyze** explores associations with external factors;\ncorrelation does not establish causation. D = daily, W = weekly, M = monthly. Horizons,\nlags, and rolling windows count periods at the selected frequency.\n\nForecasts and estimated ranges are model outputs, not guaranteed prices. Review data quality,\nvalidation errors, and historical performance before using a result."
            )

        with gr.Accordion("Table display", open=False):
            interactive_tables = gr.Checkbox(
                value=False, label="Use interactive tables",
                info="Default tables use standard readable HTML. Enable sortable grids for large results; CSV exports remain available.",
            )

        # STATE
        products_state = gr.State(initial_products)
        data_state = gr.State(pd.DataFrame())
        model_state = gr.State(
            {
                "model": None,
                "scaler": None,
                "feature_cols": [],
                "target_col": None,
                "detrended_target": None,
                "trend_coeffs": None,
                "ref_date": None,
                "std_residual": None,
                "residual_q10": None,
                "residual_q90": None,
                "best_model_name": None,
                "lags": [1, 7, 14, 30],
                "windows": [7, 14, 30, 60],
                "best_mae": None,
                "freq": "D",
                "trained_models": {},
                "detrend_method": "linear",
                "seasonal_cols": [],
                "external_cols": [],
                "smooth_w": None,
                "other_cols": [],  # ← ADDED for Forecast tab
            }
        )
        current_product_state = gr.State(default_target)
        log_entries = gr.State([])

        def sync_dropdowns(product_name, products):
            try:
                if not product_name or not products:
                    return gr.update(), gr.update(), gr.update(), product_name
                name_nfc = unicodedata.normalize("NFC", product_name)
                for k in products:
                    if unicodedata.normalize("NFC", k) == name_nfc:
                        return (gr.update(value=k), gr.update(), gr.update(value=k), k)
                return gr.update(), gr.update(), gr.update(), product_name
            except Exception:
                return gr.update(), gr.update(), gr.update(), product_name

        # TAB 1: QUICK PREDICT
        with gr.Tab("Quick Predict"):
            gr.Markdown("###  Forecast a product")

            with gr.Row():
                with gr.Column(scale=1):
                    qp_product = gr.Dropdown(
                        choices=product_names,
                        value=default_target,
                        label="Product to Forecast",
                    )

                    qp_product_type = gr.Dropdown(
                        choices=["Stable", "Volatile", "Auto-detect"],
                        value="Auto-detect",
                        label="Product Type",
                        info="Volatile products use log transform. Auto-detect recommended.",
                    )

                    qp_horizon = gr.Number(
                        value=30,
                        label="Forecast horizon (periods)",
                        precision=0,
                        minimum=1,
                        maximum=365,
                    )

                    with gr.Row():
                        qp_hybrid = gr.Checkbox(
                            value=True,
                            label="Hybrid Mode",
                            info="Blends ML with simple statistical methods",
                        )
                        qp_fast = gr.Checkbox(
                            value=True,
                            label="Fast Mode",
                            info="Quicker training with slightly less tuning",
                        )
                        qp_force = gr.Checkbox(
                            value=False,
                            label="Force Refresh",
                            info="Ignore cache and re-download all data",
                        )

                    qp_btn = gr.Button(
                        "Run Quick Predict", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):
                    qp_status = gr.Textbox(
                        interactive=False,
                        label="Result",
                        lines=3,
                        value="Select a product and click predict",
                    )
                    qp_quality = gr.Textbox(
                        interactive=False, label="Data Quality", lines=2
                    )
                    qp_alerts = gr.Textbox(
                        interactive=False,
                        label="Validation notes",
                        lines=3,
                        elem_classes=["alert-box"],
                        info="Flags unusual prediction errors",
                    )
                    qp_metrics = result_table(
                        interactive=False,
                        label="Validation performance",
                        max_height=180,
                    )

            qp_chart = gr.Plot(label="Forecast Chart")
            with gr.Row():
                qp_table = result_table(
                    interactive=False, label="Forecast Values", max_height=200
                )
                qp_download = gr.File(interactive=False, label="Download CSV")

            def qp_fn(
                product_name,
                product_type,
                horizon,
                hybrid,
                fast,
                force,
                products,
                entries,
            ):
                try:
                    entries, _ = _log(
                        entries,
                        f"Quick Predict started: {product_name} ({int(horizon)}d) | Hybrid={hybrid} | Type={product_type}",
                    )

                    if not products or product_name not in products:
                        entries, log_text = _log(
                            entries, "No product selected", ok=False
                        )
                        return (
                            "No product selected",
                            "",
                            "",
                            pd.DataFrame(),
                            None,
                            pd.DataFrame(),
                            None,
                            entries,
                            log_text,
                        )

                    pid = products[product_name]

                    train_log_capture = []

                    if hybrid:
                        result = quick_predict_hybrid(
                            product_name,
                            pid,
                            fetcher,
                            cache,
                            int(horizon),
                            force_refresh=force,
                            fast=fast,
                            train_log=lambda msg: train_log_capture.append(msg),
                            product_type=product_type,
                        )
                    else:
                        result = quick_predict(
                            product_name,
                            pid,
                            fetcher,
                            cache,
                            int(horizon),
                            force_refresh=force,
                            fast=fast,
                            train_log=lambda msg: train_log_capture.append(msg),
                            product_type=product_type,
                        )

                    if (
                        result is None
                        or not isinstance(result, tuple)
                        or len(result) != 7
                    ):
                        entries, log_text = _log(
                            entries, "Prediction returned unexpected result", ok=False
                        )
                        return (
                            "Prediction failed",
                            "",
                            "",
                            pd.DataFrame(),
                            None,
                            pd.DataFrame(),
                            None,
                            entries,
                            log_text,
                        )

                    (
                        pred_df,
                        best_name,
                        status,
                        metrics,
                        trained,
                        test_preds,
                        comp_df,
                    ) = result

                    for line in train_log_capture:
                        entries, _ = _log(entries, line)

                    if pred_df is None:
                        entries, log_text = _log(
                            entries, f"Prediction failed: {status}", ok=False
                        )
                        return (
                            status,
                            "",
                            "",
                            pd.DataFrame(),
                            None,
                            pd.DataFrame(),
                            None,
                            entries,
                            log_text,
                        )

                    # Load historical data for chart
                    hist_df = cache.load(pid)
                    if "price" in hist_df.columns:
                        hist_df = preprocess(hist_df, "price", "D")
                        hist_df = hist_df.rename(columns={"price": product_name})
                    else:
                        hist_df = preprocess(hist_df, product_name, "D")

                    try:
                        fig = create_forecast_plot(
                            hist_df, product_name, pred_df, test_preds
                        )
                    except Exception:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=hist_df["date"],
                                y=hist_df[product_name],
                                mode="lines",
                                name="Historical",
                            )
                        )
                        if not pred_df.empty:
                            bd = pd.concat(
                                [pd.Series([hist_df["date"].iloc[-1]]), pred_df["date"]]
                            ).values
                            bf = pd.concat(
                                [
                                    pd.Series([hist_df[product_name].iloc[-1]]),
                                    pred_df["forecast"],
                                ]
                            ).values
                            fig.add_trace(
                                go.Scatter(x=bd, y=bf, mode="lines", name="Forecast")
                            )
                        fig.update_layout(
                            template="plotly_white", title="Forecast (fallback view)"
                        )

                    warnings_list = check_data_quality(hist_df, product_name)
                    quality_text = (
                        "✓ No data quality issues detected."
                        if not warnings_list
                        else " | ".join(warnings_list)
                    )

                    alert_text = "✓ No unusual price movements detected."
                    if (
                        test_preds is not None
                        and not test_preds.empty
                        and "actual" in test_preds.columns
                    ):
                        # ═══ FIX: Compute std on ORIGINAL scale ═══
                        residuals = (
                            test_preds["actual"].values - test_preds["predicted"].values
                        )
                        std_res = float(np.nanstd(residuals))

                        # Only run anomaly detection if std is meaningful (not log-scale tiny)
                        if std_res > 1000:  # Price scale threshold
                            anomalies = detect_anomalies(
                                test_preds["actual"].values,
                                test_preds["predicted"].values,
                                std_res,
                                threshold=2.5,
                            )
                            if anomalies:
                                dates_list = (
                                    pd.to_datetime(test_preds["date"])
                                    .dt.strftime("%Y-%m-%d")
                                    .tolist()
                                )
                                alert_text = format_anomaly_report(
                                    anomalies, dates=dates_list
                                )
                        else:
                            # std_res is tiny (log scale), skip anomaly detection
                            alert_text = "✓ Volatile product (log transform used) — anomaly detection skipped."

                    safe_name = re.sub(r"[^\w]", "_", product_name)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_path = os.path.join(
                        FORECASTS_DIR, f"quick_forecast_{safe_name}_{ts}.csv"
                    )
                    pred_df.to_csv(csv_path, index=False)

                    entries, log_text = _log(
                        entries, f"Quick Predict complete — Best model: {best_name}"
                    )
                    return (
                        status,
                        quality_text,
                        alert_text,
                        comp_df,
                        fig,
                        pred_df,
                        csv_path,
                        entries,
                        log_text,
                    )

                except Exception as e:
                    entries, log_text = _log_err(
                        entries, "Quick Predict could not complete", e
                    )
                    return (
                        f"✗ ERROR: {e}",
                        "",
                        "",
                        pd.DataFrame(),
                        None,
                        pd.DataFrame(),
                        None,
                        entries,
                        log_text,
                    )

            qp_btn.click(
                fn=qp_fn,
                inputs=[
                    qp_product,
                    qp_product_type,
                    qp_horizon,
                    qp_hybrid,
                    qp_fast,
                    qp_force,
                    products_state,
                    log_entries,
                ],
                outputs=[
                    qp_status,
                    qp_quality,
                    qp_alerts,
                    qp_metrics,
                    qp_chart,
                    qp_table,
                    qp_download,
                    log_entries,
                    log_display,
                ],
            )

        # TAB 2: DATA & CHARTS
        with gr.Tab("Data & Charts"):
            with gr.Accordion("Available Products", open=False):
                refresh_products_btn = gr.Button("Refresh Product List")
                products_status = gr.Textbox(
                    label="Status",
                    value=f"✓ {len(initial_products)} products loaded from {initial_source}",
                )
                products_table = result_table(
                    interactive=False,
                    value=pd.DataFrame(
                        list(initial_products.items()), columns=["Product", "ID"]
                    ),
                    label="All Available Products",
                    max_height=400,
                )

            with gr.Accordion("Fetch or Upload Data", open=True):
                with gr.Row():
                    with gr.Column():
                        target = gr.Dropdown(
                            choices=product_names,
                            value=default_target,
                            label="Target Product",
                        )
                        extras = gr.Dropdown(
                            choices=product_names,
                            label="Additional Products to Merge",
                            multiselect=True,
                            info="Add related products as extra feature columns",
                        )
                        freq = gr.Dropdown(
                            choices=["D", "W", "M"],
                            value="D",
                            label="Frequency (D: daily / W: weekly / M: monthly)",
                        )
                        force_full = gr.Checkbox(
                            value=False, label="Force full re-download"
                        )
                        fetch_btn = gr.Button(
                            "Fetch Data", variant="primary", size="lg"
                        )
                    with gr.Column():
                        file_upload = gr.File(
                            label="Or upload your own CSV (columns: date, price)",
                            file_types=[".csv"],
                        )
                        gr.Markdown(
                            "_Uploaded CSV replaces fetched data. Must have `date` and `price` columns._"
                        )

                profile_indicator = gr.Textbox(
                    label="Product Profile",
                    interactive=False,
                    info="Shows which external features this product uses",
                )

                status = gr.Textbox(interactive=False, label="Status")
                dataset_info = gr.Textbox(
                    interactive=False, label="Dataset Summary", lines=4
                )
                preview = result_table(
                    interactive=False, label="Full Dataset", wrap=True
                )
                history_chart = gr.Plot(label="Historical Price Chart")
                with gr.Row():
                    data_download = gr.File(
                        interactive=False, label="Download Full Dataset"
                    )

                def show_profile_fn(product_name):
                    try:
                        profile = get_product_profile(product_name)
                        ext = profile.get("external_features", [])
                        labels = [
                            EXTERNAL_FEATURE_REGISTRY.get(e, {}).get("label", e)
                            for e in ext
                        ]
                        related = profile.get("related_products", True)
                        if ext:
                            return f"External: {', '.join(labels)} | Related Products: {related}"
                        return f"Own graph only | Related Products: {related}"
                    except Exception:
                        return "Profile unavailable"

                def fetch_data_fn(
                    target_name, extra_names, freq_val, force, products, entries
                ):
                    try:
                        entries, _ = _log(entries, f"Fetching data: {target_name}")
                        target_name = unicodedata.normalize("NFC", target_name)
                        norm_products = {
                            unicodedata.normalize("NFC", k): v
                            for k, v in products.items()
                        }

                        if not norm_products or target_name not in norm_products:
                            entries, log_text = _log(
                                entries, "No product selected", ok=False
                            )
                            return (
                                "No product selected",
                                "No data",
                                pd.DataFrame(),
                                None,
                                pd.DataFrame(),
                                None,
                                entries,
                                log_text,
                            )

                        selected = [target_name] + [
                            unicodedata.normalize("NFC", x) for x in (extra_names or [])
                        ]
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
                                raise ValueError(
                                    "No market data available. Upload a CSV or retry later."
                                )
                            df = df.rename(columns={"price": name})
                            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
                            merged = (
                                df
                                if merged is None
                                else pd.merge(merged, df, on="date", how="outer")
                            )
                        merged = merged.loc[:, ~merged.columns.duplicated()]

                        if merged is None or merged.empty:
                            raise ValueError(
                                "No market data available. Upload a CSV or retry later."
                            )

                        merged = preprocess(merged, target_name, freq=freq_val)
                        merged = merge_external_features(merged, target_name, freq_val)

                        ext_cols = [c for c in merged.columns if c.startswith("ext_")]
                        ext_note = (
                            f"\nExternal features: {len(ext_cols)} merged"
                            if ext_cols
                            else ""
                        )

                        info = (
                            f"Product: {target_name}\n"
                            f"Rows: {len(merged):,}\n"
                            f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}\n"
                            f"Source: {cache_status}{ext_note}"
                        )

                        fig = go.Figure()
                        col_name = (
                            merged.columns[1]
                            if merged.columns[1] != "date"
                            else merged.columns[2]
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=merged["date"],
                                y=merged[col_name],
                                mode="lines",
                                name=target_name,
                                line=dict(color="#10b981", width=2),
                            )
                        )
                        fig.update_layout(
                            title=f"{target_name} — Historical Prices",
                            template="plotly_white",
                            height=400,
                            hovermode="x unified",
                        )

                        entries, log_text = _log(
                            entries, f"Fetched {len(merged):,} rows for {target_name}"
                        )
                        safe_name = re.sub(r"[^\w]", "_", target_name)[:30]
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_path = os.path.join(
                            CACHE_DIR, f"dataset_{safe_name}_{ts}.csv"
                        )
                        merged.to_csv(csv_path, index=False)

                        return (
                            f"✓ Ready: {len(merged):,} rows",
                            info,
                            merged,
                            fig,
                            merged,
                            csv_path,
                            entries,
                            log_text,
                        )

                    except Exception as e:
                        entries, log_text = _log_err(entries, "Data fetch failed", e)
                        return (
                            f"✗ ERROR: {e}",
                            "No data",
                            pd.DataFrame(),
                            None,
                            pd.DataFrame(),
                            None,
                            entries,
                            log_text,
                        )

                def upload_csv_fn(file_obj, target_name, freq_val, entries):
                    try:
                        if file_obj is None:
                            entries, log_text = _log(
                                entries, "No file selected", ok=False
                            )
                            return (
                                "No file selected",
                                "No data",
                                pd.DataFrame(),
                                None,
                                pd.DataFrame(),
                                None,
                                entries,
                                log_text,
                            )

                        entries, _ = _log(
                            entries,
                            f"Uploading CSV: {os.path.basename((str(file_obj) if isinstance(file_obj, str) else file_obj.name))}",
                        )
                        if (
                            os.path.getsize(
                                str(file_obj)
                                if isinstance(file_obj, str)
                                else file_obj.name
                            )
                            > 50 * 1024 * 1024
                        ):
                            raise ValueError("CSV limit is 50 MB.")
                        raw = pd.read_csv(
                            (
                                str(file_obj)
                                if isinstance(file_obj, str)
                                else file_obj.name
                            )
                        )
                        raw.columns = [c.strip().lower() for c in raw.columns]

                        if "date" not in raw.columns:
                            entries, log_text = _log(
                                entries, "CSV must have a 'date' column", ok=False
                            )
                            return (
                                "CSV must have a 'date' column",
                                "No data",
                                pd.DataFrame(),
                                None,
                                pd.DataFrame(),
                                None,
                                entries,
                                log_text,
                            )

                        price_col = "price" if "price" in raw.columns else None
                        if price_col is None:
                            numeric_cols = [
                                c
                                for c in raw.columns
                                if c != "date" and pd.api.types.is_numeric_dtype(raw[c])
                            ]
                            if not numeric_cols:
                                entries, log_text = _log(
                                    entries,
                                    "CSV must have a numeric price column",
                                    ok=False,
                                )
                                return (
                                    "CSV must have a numeric price column",
                                    "No data",
                                    pd.DataFrame(),
                                    None,
                                    pd.DataFrame(),
                                    None,
                                    entries,
                                    log_text,
                                )
                            price_col = numeric_cols[0]

                        use_name = target_name if target_name else price_col
                        raw = raw.rename(columns={price_col: use_name})
                        raw["date"] = pd.to_datetime(
                            raw["date"], errors="coerce"
                        ).dt.normalize()
                        raw = raw.dropna(subset=["date", use_name])
                        merged = preprocess(
                            raw[["date", use_name]], use_name, freq=freq_val
                        )

                        info = (
                            f"Source: uploaded CSV\n"
                            f"Column: {use_name}\n"
                            f"Rows: {len(merged):,}\n"
                            f"Date range: {merged['date'].min().date()} to {merged['date'].max().date()}"
                        )

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=merged["date"],
                                y=merged[use_name],
                                mode="lines",
                                name=use_name,
                                line=dict(color="#10b981", width=2),
                            )
                        )
                        fig.update_layout(
                            title=f"{use_name} — Uploaded Data",
                            template="plotly_white",
                            height=400,
                        )

                        safe_name = re.sub(r"[^\w]", "_", use_name)[:30]
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_path = os.path.join(
                            CACHE_DIR, f"dataset_{safe_name}_{ts}.csv"
                        )
                        merged.to_csv(csv_path, index=False)

                        entries, log_text = _log(
                            entries, f"CSV uploaded: {len(merged):,} rows"
                        )
                        return (
                            f"✓ Ready: {len(merged):,} rows (from upload)",
                            info,
                            merged,
                            fig,
                            merged,
                            csv_path,
                            entries,
                            log_text,
                        )

                    except Exception as e:
                        entries, log_text = _log_err(entries, "CSV upload failed", e)
                        return (
                            f"✗ ERROR: {e}",
                            "No data",
                            pd.DataFrame(),
                            None,
                            pd.DataFrame(),
                            None,
                            entries,
                            log_text,
                        )

                target.change(
                    fn=show_profile_fn, inputs=[target], outputs=[profile_indicator]
                )
                fetch_btn.click(
                    fn=fetch_data_fn,
                    inputs=[
                        target,
                        extras,
                        freq,
                        force_full,
                        products_state,
                        log_entries,
                    ],
                    outputs=[
                        status,
                        dataset_info,
                        preview,
                        history_chart,
                        data_state,
                        data_download,
                        log_entries,
                        log_display,
                    ],
                )
                file_upload.upload(
                    fn=upload_csv_fn,
                    inputs=[file_upload, target, freq, log_entries],
                    outputs=[
                        status,
                        dataset_info,
                        preview,
                        history_chart,
                        data_state,
                        data_download,
                        log_entries,
                        log_display,
                    ],
                )

        # TAB 3: TRAINING — FULLY FIXED
        with gr.Tab("Training"):
            gr.Markdown("###  Compare models on your dataset")

            with gr.Row():
                with gr.Column():
                    lags_input = gr.Textbox(
                        value="1,7,14,30", label="Look-back lags (periods)"
                    )
                    windows_input = gr.Textbox(
                        value="7,14,30,60", label="Rolling windows (periods)"
                    )
                    with gr.Row():
                        auto_params_btn = gr.Button("Auto Params", size="sm")
                        fast_mode = gr.Checkbox(value=False, label="Fast Mode")
                    add_seasonal = gr.Checkbox(
                        value=True, label="Include Seasonal Patterns"
                    )
                    auto_feature_select = gr.Checkbox(
                        value=True, label="Smart Feature Selection"
                    )
                    detrend_method = gr.Dropdown(
                        choices=[
                            "linear",
                            "log",
                            "quadratic",
                            "diff",
                            "smooth_diff",
                            "none",
                        ],
                        value="linear",
                        label="Detrend Method",
                        info="'log' helps with volatile products. 'linear' works with ALL models.",
                    )
                with gr.Column():
                    models_check = gr.CheckboxGroup(
                        choices=models_available,
                        value=DEFAULT_MODELS,
                        label="Algorithms to Train",
                    )
                    use_ensemble = gr.Checkbox(
                        value=False, label="Also Train an Ensemble"
                    )
                    ensemble_type = gr.Dropdown(
                        choices=["Voting"], value="Voting", label="Ensemble Method"
                    )
                    train_btn = gr.Button("Train models", variant="primary", size="lg")

            with gr.Row():
                train_status = gr.Textbox(
                    interactive=False, label="Training Result", scale=2
                )
                dataset_info_display = gr.Textbox(
                    interactive=False, label="Dataset Used", scale=1, lines=5
                )

            with gr.Row():
                comparison_plot = gr.Plot(label="Model Comparison", scale=2)
                feature_plot = gr.Plot(label="Feature Importance", scale=1)

            comparison_table = result_table(
                interactive=False, label="Validation model ranking", max_height=250
            )

            with gr.Row():
                save_models_btn = gr.Button("Save Trained Models")
                save_status = gr.Textbox(
                    interactive=False, label="Save Status", scale=1
                )

            def train_fn(
                df,
                target_name,
                lags_str,
                windows_str,
                fast,
                add_seasonal_feat,
                auto_select,
                detrend_meth,
                selected_models,
                freq_val,
                use_ens,
                ens_type,
                state,
                entries,
                progress=gr.Progress(),
            ):
                log = []
                try:
                    entries, _ = _log(entries, f"Training started: {target_name}")

                    # ═══ FETCH FRESH DATA (IGNORE df) ═══
                    df = prepare_workspace(df, target_name, freq_val, fetcher, cache)
                    state = {}
                    if target_name not in df.columns:
                        entries, log_text = _log(
                            entries, f"Target '{target_name}' not found", ok=False
                        )
                        return (
                            f"✗ Target '{target_name}' not found",
                            "No dataset",
                            None,
                            None,
                            pd.DataFrame(),
                            state,
                            entries,
                            log_text,
                        )

                    try:
                        lags = [
                            int(x.strip())
                            for x in lags_str.split(",")
                            if x.strip().isdigit()
                        ]
                        windows = [
                            int(x.strip())
                            for x in windows_str.split(",")
                            if x.strip().isdigit()
                        ]
                    except Exception:
                        lags, windows = [1, 7, 14, 30], [7, 14, 30, 60]
                        entries, log_text = _log(
                            entries,
                            f"lags/windows parse failed, using defaults {lags}/{windows}",
                        )

                    external_cols_present = [
                        c for c in df.columns if c.startswith("ext_")
                    ]

                    progress(0.05, desc="Splitting data...")
                    split_date = compute_split_date(df)
                    entries, _ = _log(
                        entries, f"🔍 split_date: {split_date.strftime('%Y-%m-%d')}"
                    )
                    entries, _ = _log(
                        entries, f"🔍 Training rows: {len(df[df['date'] < split_date])}"
                    )

                    progress(0.10, desc="Detrending...")
                    detrend_result = detrend_series(
                        df[df["date"] < split_date],
                        target_name,
                        freq=freq_val,
                        method=detrend_meth,
                        split_date=split_date,
                    )
                    smooth_w = None
                    if isinstance(detrend_result, tuple) and len(detrend_result) == 3:
                        coeffs, ref_date = None, None
                        smooth_w = detrend_result[2]
                    else:
                        coeffs, ref_date = detrend_result

                    price_col = (
                        target_name
                        if target_name in df.columns
                        else [c for c in df.columns if c != "date"][0]
                    )
                    detrended_target = "detrended_target"

                    if coeffs is None and smooth_w is not None:
                        df["smooth"] = (
                            df[price_col].rolling(smooth_w, min_periods=1).mean()
                        )
                        df[detrended_target] = df["smooth"].diff()
                    elif coeffs is not None:
                        trend = apply_trend(df["date"], coeffs, ref_date, freq_val)
                        if detrend_meth == "log":
                            df[detrended_target] = (
                                np.log(np.clip(df[price_col].values, 1, None)) - trend
                            )
                        else:
                            df[detrended_target] = df[price_col] - trend
                    elif ref_date is not None:
                        df[detrended_target] = df[price_col]
                    else:
                        df[detrended_target] = df[price_col].diff()

                    progress(0.20, desc="Engineering features...")
                    other_cols = [
                        c
                        for c in df.columns
                        if c
                        not in [
                            target_name,
                            "date",
                            detrended_target,
                            "smooth",
                            "is_holiday",
                        ]
                        and np.issubdtype(df[c].dtype, np.number)
                        and df[c].nunique() > 10
                    ]

                    df_feat, feature_cols, seasonal_cols = create_features(
                        df,
                        detrended_target,
                        lags,
                        windows,
                        freq_val,
                        other_cols=other_cols,
                        external_cols=external_cols_present or None,
                        add_seasonal=add_seasonal_feat,
                    )

                    all_feature_cols = list(dict.fromkeys(feature_cols))
                    own_graph_cols = [
                        c
                        for c in all_feature_cols
                        if not c.startswith("ext_") and c in df_feat.columns
                    ]
                    external_feature_cols = [
                        c
                        for c in all_feature_cols
                        if c.startswith("ext_") and c in df_feat.columns
                    ]

                    # Only drop NaN for target + own-graph (lag/window padding)
                    df_feat = df_feat.dropna(
                        subset=[detrended_target]
                        + [c for c in own_graph_cols if "_lag_" in c]
                    )

                    # Fill external NaN with 0 — safe constant (not a future value), preserves all rows
                    for col in external_feature_cols:
                        if col in df_feat.columns and df_feat[col].isna().any():
                            df_feat[col] = df_feat[col].ffill()

                    if auto_select and len(all_feature_cols) > 5:
                        progress(0.30, desc="Selecting features...")
                        all_feature_cols = select_features(
                            df_feat,
                            all_feature_cols,
                            split_date=split_date,
                            max_corr=0.92,
                        )

                    entries, _ = _log(
                        entries,
                        f"Features engineered: {len(all_feature_cols)} total "
                        f"({len([c for c in all_feature_cols if c.startswith('ext_')])} external)",
                    )

                    progress(0.40, desc="Training models...")

                    # ═══ SMART SELECTION ═══
                    entries, _ = _log(entries, "🧠 Running smart feature selection...")
                    final_feature_cols = smart_feature_selection(
                        df_feat,
                        detrended_target,
                        all_feature_cols,
                        selected_models,
                        split_date=split_date,
                        train_log=log,
                    )

                    (
                        results,
                        trained_models,
                        best_model,
                        best_scaler,
                        best_name,
                        best_std,
                        best_mae,
                    ) = train_all(
                        df_feat,
                        detrended_target,
                        final_feature_cols,
                        selected_models,
                        split_date=split_date,
                        use_ensemble=use_ens,
                        ensemble_type=ens_type,
                        fast=fast,
                        progress=progress,
                        train_log=lambda msg: log.append(msg),
                    )

                    if not results:
                        entries, log_text = _log(
                            entries, "Training failed — all models had R² ≤ 0", ok=False
                        )
                        for line in log:
                            entries, _ = _log(entries, line)
                        return (
                            "✗ Training failed — all models had R² ≤ 0",
                            "No dataset",
                            None,
                            None,
                            pd.DataFrame(),
                            state,
                            entries,
                            log_text,
                        )

                    progress(0.90, desc="Computing final metrics...")
                    original_metrics, _ = compute_original_scale_metrics(
                        trained_models,
                        df_feat,
                        final_feature_cols,
                        target_name,
                        coeffs,
                        ref_date,
                        freq_val,
                        split_date,
                        smooth_w=smooth_w or 0,
                        pred_col=target_name,  # ═══ FIXED: Use target_name ═══
                    )

                    # ─── Build comparison table (detrended metrics only) ───
                    comp_data = []
                    seen = set()
                    for name in trained_models:
                        if name in seen:
                            continue
                        seen.add(name)
                        tup = trained_models[name]
                        mae_det = tup[2] if len(tup) > 2 else None
                        r2_det = tup[3] if len(tup) > 3 else None
                        comp_data.append({"Model": name, "MAE": mae_det, "R²": r2_det})
                    comp_df = pd.DataFrame(comp_data)
                    if not comp_df.empty:
                        comp_df = comp_df.sort_values("MAE").set_index("Model")

                    # ─── Pick best model from ORIGINAL-SCALE metrics ───
                    if original_metrics:
                        best_orig_mae = float("inf")
                        best_orig_name = best_name or (
                            list(trained_models.keys())[0] if trained_models else None
                        )
                        for model_name, metrics in original_metrics.items():
                            if "Naive" in model_name or "Smooth" in model_name:
                                continue
                            mae = metrics.get("MAE", float("inf"))
                            r2 = metrics.get("R-squared", -999.0)
                            if r2 > 0.3 and mae < best_orig_mae:
                                best_orig_name = model_name
                                best_orig_mae = mae
                        if best_orig_name and best_orig_mae < float("inf"):
                            best_name = best_orig_name

                    # ─── Safe defaults from train_all (guaranteed available) ───
                    best_model_obj = best_model
                    best_q10 = -1.28 * best_std if best_std else -2.0
                    best_q90 = 1.28 * best_std if best_std else 2.0

                    # ─── Sync model/scaler/std to match best_name ───
                    if best_name in trained_models:
                        bt = trained_models[best_name]
                        best_model_obj = bt[0]
                        best_scaler = bt[1]
                        best_std = bt[4] if len(bt) > 4 else best_std
                        best_q10 = bt[5] if len(bt) > 5 else -1.28 * best_std
                        best_q90 = bt[6] if len(bt) > 6 else 1.28 * best_std
                        best_mae = original_metrics.get(best_name, {}).get(
                            "MAE", bt[2] if len(bt) > 2 else best_mae
                        )

                    # ─── Comparison plot ───
                    fig_comp = go.Figure()
                    if "MAE" in comp_df.columns:
                        fig_comp.add_trace(
                            go.Bar(
                                x=comp_df.index,
                                y=comp_df["MAE"],
                                name="MAE",
                                marker_color="#10b981",
                            )
                        )
                    fig_comp.update_layout(
                        title="Model Comparison — Lower MAE is Better",
                        template="plotly_white",
                        height=400,
                        yaxis_title="MAE (detrended)",
                    )

                    # ─── Feature importance ───
                    fig_feat = None
                    if best_model_obj is not None:
                        external_cols = [
                            c for c in final_feature_cols if c.startswith("ext_")
                        ]
                        if external_cols:
                            importance_split = feature_importance_by_horizon(
                                best_model_obj, final_feature_cols, external_cols
                            )
                            own = importance_split.get("own_graph", [])[:15]
                            ext = importance_split.get("external", [])[:15]
                            if own or ext:
                                fig_feat = go.Figure()
                                if own:
                                    fig_feat.add_trace(
                                        go.Bar(
                                            x=[o["feature"] for o in own],
                                            y=[o["importance"] for o in own],
                                            name="Own-Graph Features",
                                            marker_color="#10b981",
                                        )
                                    )
                                if ext:
                                    fig_feat.add_trace(
                                        go.Bar(
                                            x=[e["feature"] for e in ext],
                                            y=[e["importance"] for e in ext],
                                            name="External Features",
                                            marker_color="#ef4444",
                                        )
                                    )
                                fig_feat.update_layout(
                                    title="Feature Importance: Own-Graph vs External",
                                    template="plotly_white",
                                    barmode="group",
                                    height=400,
                                )
                        elif hasattr(best_model_obj, "feature_importances_"):
                            imp = best_model_obj.feature_importances_
                            imp_df = (
                                pd.DataFrame(
                                    {
                                        "feature": final_feature_cols[: len(imp)],
                                        "importance": imp,
                                    }
                                )
                                .sort_values("importance", ascending=False)
                                .head(15)
                            )
                            fig_feat = px.bar(
                                imp_df,
                                x="importance",
                                y="feature",
                                orientation="h",
                                title="Top 15 Feature Importances",
                                template="plotly_white",
                                height=400,
                            )

                    # ─── Update state with CONSISTENT best model ───
                    state.update(
                        {
                            "history": df.copy(),
                            "model": best_model_obj,
                            "scaler": best_scaler,
                            "feature_cols": final_feature_cols,
                            "target_col": target_name,
                            "detrended_target": detrended_target,
                            "trend_coeffs": coeffs,
                            "ref_date": ref_date,
                            "std_residual": best_std,
                            "residual_q10": best_q10,
                            "residual_q90": best_q90,
                            "best_model_name": best_name,
                            "lags": lags,
                            "windows": windows,
                            "best_mae": best_mae,
                            "best_mae_r2": original_metrics.get(best_name, {}).get(
                                "R-squared", 0.0
                            )
                            if original_metrics
                            else 0.0,
                            "freq": freq_val,
                            "trained_models": trained_models,
                            "detrend_method": detrend_meth,
                            "seasonal_cols": seasonal_cols,
                            "external_cols": [
                                c for c in final_feature_cols if c.startswith("ext_")
                            ],
                            "smooth_w": smooth_w,
                            "other_cols": other_cols,
                        }
                    )

                    # ─── Display metrics (always original-scale) ───
                    _display_mae = original_metrics.get(best_name, {}).get(
                        "MAE", best_mae if best_mae else 0.0
                    )
                    _display_r2 = original_metrics.get(best_name, {}).get(
                        "R-squared", 0.0
                    )

                    train_mask = pd.to_datetime(df_feat["date"]) < pd.to_datetime(
                        split_date
                    )
                    train_rows_actual = train_mask.sum()
                    total_rows_used = len(df_feat)

                    seasonal_set = {
                        "month_sin",
                        "month_cos",
                        "dow_sin",
                        "dow_cos",
                        "is_holiday",
                    }
                    seasonal_extra = [
                        c for c in final_feature_cols if c in seasonal_set
                    ]
                    cross_features = [
                        c
                        for c in final_feature_cols
                        if ("_ratio_" in c or "_margin_" in c or "_corr_" in c)
                    ]
                    direct_external = [
                        c for c in final_feature_cols if c.startswith("ext_")
                    ]
                    own_graph = [
                        c
                        for c in final_feature_cols
                        if c not in seasonal_extra
                        and c not in cross_features
                        and c not in direct_external
                    ]

                    extra_holiday = 1 if "is_holiday" in seasonal_extra else 0
                    extra_seasonal = len(
                        [c for c in seasonal_extra if c != "is_holiday"]
                    )
                    extra_related_raw = set()
                    for c in final_feature_cols:
                        if c.startswith("ext_feed_"):
                            extra_related_raw.add(c.split("_")[2])
                    extra_related_count = len(extra_related_raw)
                    extra_macro_raw = set()
                    for c in final_feature_cols:
                        if c.startswith("ext_") and not c.startswith("ext_feed_"):
                            extra_macro_raw.add(c.split("_")[1])
                    extra_macro_count = len(extra_macro_raw)
                    extra_total = (
                        extra_holiday
                        + extra_seasonal
                        + extra_related_count
                        + extra_macro_count
                        + len(cross_features)
                    )

                    extra_parts = []
                    if extra_seasonal:
                        extra_parts.append(f"{extra_seasonal} seasonal")
                    if extra_holiday:
                        extra_parts.append("holiday")
                    if extra_related_count:
                        extra_parts.append(f"{extra_related_count} related")
                    if extra_macro_count:
                        extra_parts.append(f"{extra_macro_count} external")
                    if cross_features:
                        extra_parts.append(f"{len(cross_features)} cross")
                    extra_detail = ", ".join(extra_parts) if extra_parts else "none"

                    dataset_info_text = (
                        f"Product: {target_name}\n"
                        f"Training rows: {train_rows_actual:,} (of {total_rows_used:,} after feature eng)\n"
                        f"Total features: {len(final_feature_cols)} (own: {len(own_graph)}, extra: {extra_total} — {extra_detail})\n"
                        f"Best model: {best_name}\n"
                        f"MAE: {_display_mae:,.0f}\n"
                        f"R²: {_display_r2:.3f}"
                    )

                    status_msg = (
                        f"✓ Trained {len(results)} models | "
                        f"Best: {best_name} | "
                        f"MAE: {_display_mae:,.0f} | "
                        f"R²: {_display_r2:.3f}"
                    )

                    for line in log:
                        entries, _ = _log(entries, line)
                    entries, log_text = _log(
                        entries,
                        f"Training complete — Best: {best_name} (MAE={_display_mae:,.0f})",
                    )

                    return (
                        status_msg,
                        dataset_info_text,
                        fig_comp,
                        fig_feat,
                        comp_df,
                        state,
                        entries,
                        log_text,
                    )

                except Exception as e:
                    entries, log_text = _log_err(
                        entries, "Training could not complete", e
                    )
                    for line in log:
                        entries, _ = _log(entries, line)
                    return (
                        f"✗ ERROR: {e}",
                        "No dataset",
                        None,
                        None,
                        pd.DataFrame(),
                        state,
                        entries,
                        log_text,
                    )

            def auto_params_fn(df, target_name, freq_val, entries):
                try:
                    # ═══ FIX: Fetch fresh data for auto params ═══
                    df = prepare_workspace(df, target_name, freq_val, fetcher, cache)
                    lags, windows = auto_params(df, target_name, freq_val)
                    entries, log_text = _log(
                        entries, f"Auto params: lags={lags}, windows={windows}"
                    )
                    return (
                        "✓ Auto params set",
                        ",".join(str(x) for x in lags),
                        ",".join(str(x) for x in windows),
                        entries,
                        log_text,
                    )
                except Exception as e:
                    entries, log_text = _log_err(entries, "Auto params failed", e)
                    return (
                        f"✗ ERROR: {e}",
                        "1,7,14,30",
                        "7,14,30,60",
                        entries,
                        log_text,
                    )

            def save_models_fn(state, entries):
                try:
                    trained = state.get("trained_models", {})
                    product = state.get("target_col", "unknown")
                    if not trained:
                        entries, log_text = _log(
                            entries, "No trained models to save", ok=False
                        )
                        return ("No models to save", entries, log_text)
                    metadata = {
                        k: v
                        for k, v in state.items()
                        if k not in {"model", "scaler", "trained_models"}
                    }
                    saved = save_trained_models(trained, product, metadata)
                    entries, log_text = _log(
                        entries, f"Saved {len(saved)} model(s) for {product}"
                    )
                    return (f"✓ Saved {len(saved)} model(s)", entries, log_text)
                except Exception as e:
                    entries, log_text = _log_err(entries, "Save failed", e)
                    return (f"✗ ERROR: {e}", entries, log_text)

            auto_params_btn.click(
                fn=auto_params_fn,
                inputs=[data_state, target, freq, log_entries],
                outputs=[
                    train_status,
                    lags_input,
                    windows_input,
                    log_entries,
                    log_display,
                ],
            )

            train_btn.click(
                fn=train_fn,
                inputs=[
                    data_state,
                    target,
                    lags_input,
                    windows_input,
                    fast_mode,
                    add_seasonal,
                    auto_feature_select,
                    detrend_method,
                    models_check,
                    freq,
                    use_ensemble,
                    ensemble_type,
                    model_state,
                    log_entries,
                ],
                outputs=[
                    train_status,
                    dataset_info_display,
                    comparison_plot,
                    feature_plot,
                    comparison_table,
                    model_state,
                    log_entries,
                    log_display,
                ],
            )

            save_models_btn.click(
                fn=save_models_fn,
                inputs=[model_state, log_entries],
                outputs=[save_status, log_entries, log_display],
            )

        # TAB 4: FORECAST
        with gr.Tab("Forecast"):
            gr.Markdown("###  Forecast with a trained model")

            with gr.Row():
                with gr.Column():
                    horizon_input = gr.Number(
                        value=14,
                        label="Forecast horizon (periods)",
                        precision=0,
                        minimum=1,
                    )
                    model_selector = gr.Dropdown(
                        choices=[],
                        label="Trained Model",
                        info="Auto-populated after training",
                    )
                    refresh_models_btn = gr.Button("Refresh Model List")
                with gr.Column():
                    forecast_hybrid = gr.Checkbox(
                        value=True,
                        label="Blend with Simple Methods",
                        info="Blends ML with moving average/trend; compare backtests before using",
                    )
                    anomaly_threshold = gr.Slider(
                        value=2,
                        minimum=1,
                        maximum=5,
                        step=0.5,
                        label="Anomaly Sensitivity",
                    )
                    forecast_btn = gr.Button(
                        "Generate Forecast", variant="primary", size="lg"
                    )

            forecast_status = gr.Textbox(interactive=False, label="Forecast Result")
            forecast_alerts = gr.Textbox(
                interactive=False,
                label="Anomaly Alerts",
                lines=3,
                elem_classes=["alert-box"],
            )
            forecast_chart = gr.Plot(label="Forecast Chart")
            with gr.Row():
                forecast_table = result_table(
                    interactive=False, label="Forecast Values"
                )
                download_btn = gr.File(interactive=False, label="Download CSV")

            def refresh_model_list_fn(state):
                try:
                    trained = state.get("trained_models", {})
                    names = list(trained.keys())
                    best = state.get("best_model_name")
                    default = (
                        best
                        if best and best in names
                        else (names[0] if names else None)
                    )
                    return gr.update(choices=names, value=default)
                except Exception:
                    return gr.update()

            refresh_models_btn.click(
                fn=refresh_model_list_fn, inputs=[model_state], outputs=[model_selector]
            )

            with gr.Accordion("Restore a saved model", open=False):
                saved_package = gr.Dropdown(
                    choices=list_saved_models(), label="Local model package"
                )
                trusted_package = gr.Checkbox(
                    value=False, label="I trust this local model file"
                )
                saved_refresh = gr.Button("Refresh saved packages")
                saved_restore = gr.Button("Restore model and history")
                saved_status = gr.Textbox(label="Restore status", interactive=False)

            def restore_saved(filename, trusted):
                if not trusted:
                    raise gr.Error(
                        "Only load trusted model files: they can execute code."
                    )
                if filename not in list_saved_models():
                    raise gr.Error("Choose a saved package from the local registry.")
                package = load_trained_model(os.path.join(MODELS_DIR, filename))
                restored = dict(package["metadata"])
                name = restored["model_name"]
                restored["model"] = package["model"]
                restored["scaler"] = package["scaler"]
                restored["best_model_name"] = name
                restored["trained_models"] = {
                    name: (
                        package["model"],
                        package["scaler"],
                        restored["mae"],
                        restored["r2"],
                        restored["std_res"],
                        restored["residual_q10"],
                        restored["residual_q90"],
                    )
                }
                product = restored["target_col"]
                choices = sorted(set(product_names + [product]))
                return (
                    restored,
                    gr.update(choices=[name], value=name),
                    gr.update(choices=choices, value=product),
                    f"Restored {name} for {product}",
                )

            saved_refresh.click(
                lambda: gr.update(choices=list_saved_models(), value=None),
                outputs=[saved_package],
            )
            saved_restore.click(
                restore_saved,
                inputs=[saved_package, trusted_package],
                outputs=[model_state, model_selector, target, saved_status],
                **({"api_visibility": "private"} if "api_visibility" in inspect.signature(saved_restore.click).parameters else {"api_name": False}),
            )

            def forecast_fn(
                df,
                target_name,
                horizon,
                selected_model_name,
                state,
                thresh,
                hybrid,
                entries,
            ):
                try:
                    entries, _ = _log(
                        entries,
                        f"Generating forecast: {int(horizon)}d | Hybrid={hybrid}",
                    )

                    # ═══ FETCH FRESH DATA ═══
                    if state.get("target_col") != target_name:
                        raise ValueError(
                            "Train a model for the selected product first."
                        )
                    history = state.get("history")
                    if history is None or history.empty:
                        raise ValueError(
                            "Training history is missing. Retrain this model."
                        )
                    df = history.copy()
                    trained = state.get("trained_models", {})
                    if not trained:
                        entries, log_text = _log(
                            entries, "No trained models available", ok=False
                        )
                        return (
                            "✗ No trained models — train first",
                            "",
                            None,
                            pd.DataFrame(),
                            None,
                            entries,
                            log_text,
                        )

                    if (
                        selected_model_name is None
                        or selected_model_name not in trained
                    ):
                        entries, log_text = _log(
                            entries, "Please select a valid model", ok=False
                        )
                        return (
                            "✗ Select a valid model",
                            "",
                            None,
                            pd.DataFrame(),
                            None,
                            entries,
                            log_text,
                        )

                    model_tup = trained[selected_model_name]
                    model, scaler = model_tup[0], model_tup[1]

                    price_col = state.get("target_col", target_name)
                    if price_col not in df.columns:
                        if "price" in df.columns:
                            price_col = "price"
                        else:
                            for c in df.columns:
                                if (
                                    c != "date"
                                    and not c.startswith("ext_")
                                    and pd.api.types.is_numeric_dtype(df[c])
                                ):
                                    price_col = c
                                    break

                    external_cols = [
                        c for c in df.columns if c.startswith("ext_")
                    ] or None
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

                    other_cols = state.get("other_cols", None)
                    if other_cols is None:
                        other_cols = [
                            c for c in feature_cols if c.startswith("ext_feed_")
                        ]

                    from sklearn.base import clone

                    working = df.copy()
                    if coeffs is not None:
                        values = (
                            np.log(working[price_col])
                            if isinstance(coeffs, dict)
                            and coeffs.get("method") == "log"
                            else working[price_col]
                        )
                        working[detrended_target] = values - apply_trend(
                            working["date"], coeffs, ref_date, freq_val
                        )
                    elif ref_date is None:
                        values = (
                            working[price_col].rolling(smooth_w, min_periods=1).mean()
                            if smooth_w
                            else working[price_col]
                        )
                        working[detrended_target] = values.diff()
                    else:
                        working[detrended_target] = working[price_col]
                    fit_frame, _, _ = create_features(
                        working,
                        detrended_target,
                        lags,
                        windows,
                        freq_val,
                        other_cols=other_cols,
                        external_cols=external_cols,
                    )
                    fit_frame = fit_frame.dropna(subset=[detrended_target])
                    scaler = clone(scaler)
                    model = clone(model).fit(
                        scaler.fit_transform(
                            fit_frame[feature_cols].to_numpy(dtype=float)
                        ),
                        fit_frame[detrended_target].to_numpy(dtype=float),
                    )
                    pred_df = iterative_forecast(
                        df,
                        price_col,
                        detrended_target,
                        feature_cols,
                        model,
                        scaler,
                        std_res,
                        int(horizon),
                        lags,
                        windows,
                        freq_val,
                        coeffs,
                        ref_date,
                        residual_q10=res_q10,
                        residual_q90=res_q90,
                        external_cols=external_cols,
                        smooth_w=smooth_w,
                        other_cols=other_cols,
                        add_texture=False,
                    )

                    if pred_df.empty:
                        entries, log_text = _log(
                            entries, "Forecast generation failed", ok=False
                        )
                        return (
                            "✗ Forecast failed",
                            "",
                            None,
                            pd.DataFrame(),
                            None,
                            entries,
                            log_text,
                        )

                    blend_note = ""
                    if hybrid and price_col in df.columns:
                        series = df[price_col].dropna().values
                        if len(series) >= 10:
                            dummy = SmartHybridPredictor(
                                ml_func=lambda *a, **kw: (
                                    pred_df,
                                    "",
                                    "",
                                    {},
                                    {},
                                    pd.DataFrame(),
                                    pd.DataFrame(),
                                ),
                                test_window=60,
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
                                simple_pred = methods[best_simple_name](
                                    series, int(horizon)
                                )
                                ml_pred = pred_df["forecast"].values
                                r2_boost = 0.0
                                if best_simple_score > 0 and std_res > 0:
                                    ml_weight = best_simple_score / (
                                        best_simple_score + std_res
                                    )
                                    ml_weight = max(0.3, min(0.7, ml_weight))
                                else:
                                    ml_weight = 0.6
                                ml_weight = min(ml_weight + r2_boost, 0.95)
                                simple_weight = 1.0 - ml_weight
                                blended = (
                                    ml_weight * ml_pred + simple_weight * simple_pred
                                )
                                blended = np.maximum(blended, 0)
                                if (
                                    "lower" in pred_df.columns
                                    and "upper" in pred_df.columns
                                ):
                                    shift = blended - ml_pred
                                    pred_df["lower"] = pred_df["lower"] + shift
                                    pred_df["upper"] = pred_df["upper"] + shift
                                pred_df["forecast"] = blended
                                pred_df["lower"] = np.minimum(
                                    pred_df["lower"].clip(lower=0), blended
                                )
                                pred_df["upper"] = np.maximum(pred_df["upper"], blended)
                                blend_note = f" | Blended: ML {ml_weight:.0%} + {best_simple_name} {simple_weight:.0%}"

                    try:
                        fig = create_forecast_plot(df, price_col, pred_df)
                    except Exception:
                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=df["date"],
                                y=df[price_col],
                                mode="lines",
                                name="Historical",
                            )
                        )
                        if not pred_df.empty:
                            bd = pd.concat(
                                [pd.Series([df["date"].iloc[-1]]), pred_df["date"]]
                            ).values
                            bf = pd.concat(
                                [
                                    pd.Series([df[price_col].iloc[-1]]),
                                    pred_df["forecast"],
                                ]
                            ).values
                            fig.add_trace(
                                go.Scatter(x=bd, y=bf, mode="lines", name="Forecast")
                            )
                        fig.update_layout(
                            template="plotly_white", title="Forecast (fallback view)"
                        )

                    alert_text = "✓ No unusually large jumps detected."
                    if not pred_df.empty and std_res and price_col in df.columns:
                        last_actual = float(df[price_col].dropna().iloc[-1])
                        vals = pred_df["forecast"].values
                        prev = np.concatenate([[last_actual], vals[:-1]])
                        step_moves = vals - prev
                        z = np.abs(step_moves) / max(std_res, 1e-9)
                        flagged = np.where(z > float(thresh))[0]
                        if len(flagged) > 0:
                            lines = [
                                f"⚠️ {len(flagged)} forecast step(s) show larger-than-usual moves:"
                            ]
                            for idx in flagged[:10]:
                                d = pd.to_datetime(pred_df["date"].iloc[idx]).strftime(
                                    "%Y-%m-%d"
                                )
                                lines.append(
                                    f"  • {d}: move = {step_moves[idx]:,.0f} (z = {z[idx]:.1f})"
                                )
                            if len(flagged) > 10:
                                lines.append(f"  ... and {len(flagged) - 10} more")
                            alert_text = "\n".join(lines)

                    safe_name = re.sub(r"[^\w]", "_", price_col)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    csv_path = os.path.join(
                        FORECASTS_DIR, f"forecast_output_{safe_name}_{ts}.csv"
                    )
                    pred_df.to_csv(csv_path, index=False)

                    ext_note = (
                        f" | External: {len(external_cols)}" if external_cols else ""
                    )
                    status = f"✓ Forecast: {selected_model_name}{ext_note}{blend_note}"

                    entries, log_text = _log(entries, "Forecast generated successfully")
                    return (
                        status,
                        alert_text,
                        fig,
                        pred_df,
                        csv_path,
                        entries,
                        log_text,
                    )

                except Exception as e:
                    entries, log_text = _log_err(
                        entries, "Forecast could not complete", e
                    )
                    return (
                        f"✗ ERROR: {e}",
                        "",
                        None,
                        pd.DataFrame(),
                        None,
                        entries,
                        log_text,
                    )

            forecast_btn.click(
                fn=forecast_fn,
                inputs=[
                    data_state,
                    target,
                    horizon_input,
                    model_selector,
                    model_state,
                    anomaly_threshold,
                    forecast_hybrid,
                    log_entries,
                ],
                outputs=[
                    forecast_status,
                    forecast_alerts,
                    forecast_chart,
                    forecast_table,
                    download_btn,
                    log_entries,
                    log_display,
                ],
            )

        # TAB 5: BACKTEST
        with gr.Tab("Backtest"):
            gr.Markdown("###  Historical Accuracy Test — Validate Before Trusting")

            with gr.Row():
                bt_models = gr.Dropdown(
                    choices=[
                        "All Models",
                        "Ridge",
                        "Lasso",
                        "ElasticNet",
                        "Random Forest",
                        "Gradient Boosting",
                    ]
                    + (["XGBoost"] if XGB_AVAILABLE else [])
                    + (["LightGBM"] if LGBM_AVAILABLE else []),
                    value="All Models",
                    label="Model to Test",
                )
                bt_horizon = gr.Number(
                    value=30, label="Backtest window (periods)", precision=0, minimum=7
                )
                bt_btn = gr.Button("Run Backtest", variant="primary", size="lg")

            bt_status = gr.Textbox(interactive=False, label="Backtest Result")
            with gr.Row():
                bt_actual_vs_pred = gr.Plot(label="Actual vs Predicted")
                bt_errors = gr.Plot(label="Prediction Errors")
            bt_metrics = result_table(interactive=False, label="Summary Metrics")
            gr.Markdown(
                "Estimated ranges are not calibrated guarantees. Compare forecast errors across historical folds."
            )
            bt_details = result_table(
                interactive=False, label="Backtest fold results", max_height=250
            )

            def run_backtest_fn(
                df,
                target_name,
                model_name,
                horizon_val,
                lags_str,
                windows_str,
                detrend_meth,
                freq_val,
                add_seasonal_val,
                use_ens,
                ens_type,
                fast,
                entries,
            ):
                try:
                    entries, _ = _log(
                        entries,
                        f"Running backtest: {model_name}, {int(horizon_val)}d window",
                    )

                    if df is None or df.empty:
                        entries, log_text = _log(entries, "No data loaded", ok=False)
                        return (
                            "✗ No data loaded",
                            None,
                            None,
                            pd.DataFrame(),
                            pd.DataFrame(),
                            entries,
                            log_text,
                        )

                    try:
                        lags = [
                            int(x.strip())
                            for x in lags_str.split(",")
                            if x.strip().isdigit()
                        ]
                        windows = [
                            int(x.strip())
                            for x in windows_str.split(",")
                            if x.strip().isdigit()
                        ]
                    except Exception:
                        lags, windows = [1, 7, 14, 30], [7, 14, 30]
                        entries, _ = _log(
                            entries,
                            f"lags/windows parse failed, using defaults {lags}/{windows}",
                        )

                    models = [
                        m
                        for m in [
                            "Ridge",
                            "Lasso",
                            "ElasticNet",
                            "Random Forest",
                            "Gradient Boosting",
                            "XGBoost",
                            "LightGBM",
                        ]
                        if (model_name == "All Models" or m == model_name)
                        and (m != "XGBoost" or XGB_AVAILABLE)
                        and (m != "LightGBM" or LGBM_AVAILABLE)
                    ]

                    bt_df = backtest(
                        df,
                        target_name,
                        lags,
                        windows,
                        freq_val,
                        detrend_meth,
                        models,
                        horizon=int(horizon_val),
                        n_folds=3,
                        fast=fast,
                        add_seasonal=add_seasonal_val,
                    )

                    if bt_df.empty:
                        entries, log_text = _log(
                            entries, "Not enough data for backtest", ok=False
                        )
                        return (
                            "✗ Not enough data",
                            None,
                            None,
                            pd.DataFrame(),
                            pd.DataFrame(),
                            entries,
                            log_text,
                        )

                    avg_mae = bt_df["mae"].mean()
                    avg_rmse = bt_df["rmse"].mean()
                    total_days = (
                        bt_df["n_test"].sum() if "n_test" in bt_df.columns else 0
                    )

                    fig_compare = go.Figure()
                    fig_compare.add_trace(
                        go.Bar(
                            x=[f"Fold {int(f)}" for f in bt_df["fold"]],
                            y=bt_df["mae"],
                            name="MAE",
                            marker_color="#ef4444",
                            text=[f"{v:,.0f}" for v in bt_df["mae"]],
                            textposition="outside",
                        )
                    )
                    if "best_model" in bt_df.columns:
                        fig_compare.add_trace(
                            go.Scatter(
                                x=[f"Fold {int(f)}" for f in bt_df["fold"]],
                                y=bt_df["mae"] * 1.05,
                                mode="text",
                                text=bt_df["best_model"],
                                textposition="top center",
                                textfont=dict(size=11, color="white"),
                                showlegend=False,
                            )
                        )
                    fig_compare.update_layout(
                        title=f"Backtest MAE per Fold (Avg: {avg_mae:,.0f})",
                        template="plotly_white",
                        height=400,
                        yaxis_title="MAE",
                    )

                    fig_err = go.Figure()
                    fig_err.add_trace(
                        go.Bar(
                            x=[f"Fold {int(f)}" for f in bt_df["fold"]],
                            y=bt_df["rmse"],
                            name="RMSE",
                            marker_color="#3b82f6",
                            text=[f"{v:,.0f}" for v in bt_df["rmse"]],
                            textposition="outside",
                        )
                    )
                    fig_err.update_layout(
                        title=f"Backtest RMSE per Fold (Avg: {avg_rmse:,.0f})",
                        template="plotly_white",
                        height=300,
                    )

                    best_fold = bt_df.loc[bt_df["mae"].idxmin()]
                    metrics_df = pd.DataFrame(
                        {
                            "Metric": [
                                "Avg MAE",
                                "Avg RMSE",
                                "Best MAE",
                                "Worst MAE",
                                "Best Model",
                                "Test Days",
                                "Folds",
                            ],
                            "Value": [
                                f"{avg_mae:,.0f}",
                                f"{avg_rmse:,.0f}",
                                f"{bt_df['mae'].min():,.0f}",
                                f"{bt_df['mae'].max():,.0f}",
                                str(best_fold.get("best_model", "N/A")),
                                str(total_days),
                                str(len(bt_df)),
                            ],
                        }
                    )

                    detail_cols = [
                        c
                        for c in ["fold", "best_model", "mae", "rmse", "n_test"]
                        if c in bt_df.columns
                    ]
                    details = bt_df[detail_cols].round(2)
                    details.columns = [
                        c.replace("_", " ").title() for c in details.columns
                    ]

                    status_msg = (
                        f"✓ Backtest: {len(bt_df)} folds, {total_days} test days | "
                        f"Avg MAE = {avg_mae:,.0f} | "
                        f"Best MAE = {bt_df['mae'].min():,.0f}"
                    )

                    entries, log_text = _log(
                        entries, f"Backtest done — Avg MAE={avg_mae:,.0f}"
                    )
                    return (
                        status_msg,
                        fig_compare,
                        fig_err,
                        metrics_df,
                        details,
                        entries,
                        log_text,
                    )

                except Exception as e:
                    entries, log_text = _log_err(
                        entries, "Backtest could not complete", e
                    )
                    import traceback

                    traceback.print_exc()
                    return (
                        f"✗ ERROR: {e}",
                        None,
                        None,
                        pd.DataFrame(),
                        pd.DataFrame(),
                        entries,
                        log_text,
                    )

            bt_btn.click(
                fn=run_backtest_fn,
                inputs=[
                    data_state,
                    target,
                    bt_models,
                    bt_horizon,
                    lags_input,
                    windows_input,
                    detrend_method,
                    freq,
                    add_seasonal,
                    use_ensemble,
                    ensemble_type,
                    fast_mode,
                    log_entries,
                ],
                outputs=[
                    bt_status,
                    bt_actual_vs_pred,
                    bt_errors,
                    bt_metrics,
                    bt_details,
                    log_entries,
                    log_display,
                ],
            )

        # TAB 6: ANALYZE
        with gr.Tab("Analyze"):
            gr.Markdown("###  Explore price relationships")

            with gr.Accordion("Controls", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        an_product = gr.Dropdown(
                            choices=product_names,
                            value=default_target,
                            label="Product to Analyze",
                        )
                    with gr.Column(scale=1):
                        an_freq = gr.Radio(
                            choices=["D", "W", "M"],
                            value="D",
                            label="Frequency (D: daily / W: weekly / M: monthly)",
                        )
                    with gr.Column(scale=1):
                        an_btn = gr.Button("Analyze", variant="primary", size="lg")
                an_status = gr.Textbox(
                    label="Status",
                    lines=2,
                    interactive=False,
                    value="Select a product and click Analyze",
                )

            with gr.Tabs():
                with gr.Tab("Overview"):
                    an_summary = gr.Markdown(value="")
                    an_overlay = gr.Plot(label="Price")

                with gr.Tab("Correlations"):
                    an_corr_table = result_table(
                        interactive=False,
                        label="Correlation Table",
                        headers=[
                            "Feature",
                            "Pearson R",
                            "R²",
                            "p-value",
                            "Significance",
                            "Peak Lag (d)",
                            "R at Peak",
                        ],
                        max_height=350,
                        wrap=True,
                    )
                    an_cross_corr = gr.Plot(label="Cross-Correlation: Lag vs Pearson R")

                with gr.Tab("Deep Dive"):
                    an_rolling = gr.Plot(label="Rolling 60-period correlation")
                    an_scatter = gr.Plot(label="Price vs Top Feature (with Regression)")

            def _an_label(col: str) -> str:
                if col in EXTERNAL_LABELS:
                    return EXTERNAL_LABELS[col]
                if col.startswith("ext_feed_"):
                    return col.replace("ext_feed_", "خوراک ")
                return col.replace("ext_", "")

            def analyze_fn(product_name, freq, products, entries):
                try:
                    entries, _ = _log(
                        entries, f"Analyze started: {product_name} (freq={freq})"
                    )

                    if not products or product_name not in products:
                        entries, log_text = _log(
                            entries, "No product selected", ok=False
                        )
                        return (
                            "No product selected",
                            "",
                            None,
                            pd.DataFrame(),
                            None,
                            None,
                            None,
                            entries,
                            log_text,
                        )

                    pid = products[product_name]
                    hist_df, _ = cache.update(pid, force=False)

                    if hist_df.empty or "price" not in hist_df.columns:
                        entries, log_text = _log(
                            entries, "No price data found", ok=False
                        )
                        return (
                            "No price data found",
                            "",
                            None,
                            pd.DataFrame(),
                            None,
                            None,
                            None,
                            entries,
                            log_text,
                        )

                    hist_df = preprocess(hist_df, "price", freq)
                    hist_df = merge_external_features(hist_df, product_name, freq)
                    hist_df = fetch_related_prices(
                        hist_df, product_name, fetcher, cache, freq
                    )

                    all_factors = [
                        c
                        for c in hist_df.columns
                        if c.startswith("ext_") and hist_df[c].notna().sum() > 20
                    ]
                    if not all_factors:
                        entries, log_text = _log(
                            entries,
                            "No external features or related products available",
                        )
                        return (
                            "No factors available",
                            "No external feature data or related products found.",
                            None,
                            pd.DataFrame(),
                            None,
                            None,
                            None,
                            entries,
                            log_text,
                        )

                    result = analyze_product_relationships(
                        hist_df,
                        target="price",
                        external_cols=all_factors,
                        max_lag=30,
                        rolling_window=60,
                    )

                    ext_count = sum(
                        1 for c in all_factors if not c.startswith("ext_feed_")
                    )
                    feed_count = sum(
                        1 for c in all_factors if c.startswith("ext_feed_")
                    )
                    entries, _ = _log(
                        entries,
                        f"Computed correlations: {ext_count} external + {feed_count} related products",
                    )

                    colors = [
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",
                        "#9467bd",
                        "#8c564b",
                    ]
                    hover_template = "<b>%{x|%Y-%m-%d}</b><br>%{y:.1f}<extra>%{fullData.name}</extra>"

                    norm_df = result["normalized_df"]
                    fig_overlay = go.Figure()
                    fig_overlay.add_trace(
                        go.Scatter(
                            x=norm_df["date"],
                            y=norm_df["price"],
                            name=product_name,
                            line=dict(width=3, color=colors[0]),
                            hovertemplate=hover_template,
                        )
                    )
                    for i, col in enumerate(all_factors):
                        label = _an_label(col)
                        fig_overlay.add_trace(
                            go.Scatter(
                                x=norm_df["date"],
                                y=norm_df[col],
                                name=label,
                                line=dict(
                                    width=1.5,
                                    dash="dot",
                                    color=colors[(i + 1) % len(colors)],
                                ),
                                hovertemplate=hover_template,
                            )
                        )
                    fig_overlay.update_layout(
                        yaxis_title="Normalised (0–100)",
                        xaxis_title="Date",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                        height=420,
                        margin=dict(l=60, r=30, t=30, b=50),
                        hovermode="x unified",
                    )

                    lags = result["lags"]
                    fig_cross = go.Figure()
                    for i, col in enumerate(all_factors):
                        label = _an_label(col)
                        fig_cross.add_trace(
                            go.Scatter(
                                x=lags,
                                y=result["cross_corrs"][col],
                                name=label,
                                line=dict(width=2, color=colors[(i + 1) % len(colors)]),
                            )
                        )
                    fig_cross.add_hline(
                        y=0, line_dash="dash", line_color="gray", opacity=0.4
                    )
                    fig_cross.update_layout(
                        xaxis_title="Lag (days) — positive = factor leads price",
                        yaxis_title="Pearson R",
                        height=400,
                        margin=dict(l=60, r=30, t=30, b=50),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                    )

                    fig_rolling = go.Figure()
                    for i, col in enumerate(all_factors):
                        label = _an_label(col)
                        rc = result["rolling_corrs"][col].dropna()
                        fig_rolling.add_trace(
                            go.Scatter(
                                x=hist_df["date"].iloc[rc.index].values,
                                y=rc.values,
                                name=label,
                                line=dict(
                                    width=1.5, color=colors[(i + 1) % len(colors)]
                                ),
                                hovertemplate=hover_template,
                            )
                        )
                    fig_rolling.add_hline(
                        y=0, line_dash="dash", line_color="gray", opacity=0.4
                    )
                    fig_rolling.add_hrect(
                        y0=-1, y1=-0.5, fillcolor="red", opacity=0.05, line_width=0
                    )
                    fig_rolling.add_hrect(
                        y0=0.5, y1=1, fillcolor="green", opacity=0.05, line_width=0
                    )
                    fig_rolling.update_layout(
                        yaxis_title="Pearson R",
                        xaxis_title="Date",
                        height=400,
                        margin=dict(l=60, r=30, t=30, b=50),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                        hovermode="x unified",
                    )

                    fig_scatter = go.Figure()
                    if result["correlations"]:
                        top_col = result["correlations"][0]["col"]
                        top_label = result["correlations"][0]["label"]
                        top_r = result["correlations"][0]["r"]
                        valid = hist_df[["price", top_col]].dropna()
                        fig_scatter.add_trace(
                            go.Scatter(
                                x=valid[top_col],
                                y=valid["price"],
                                mode="markers",
                                marker=dict(size=5, opacity=0.45, color=colors[1]),
                                name=f"{product_name} vs {top_label}",
                            )
                        )
                        if len(valid) > 2:
                            z = np.polyfit(
                                valid[top_col].values, valid["price"].values, 1
                            )
                            p = np.poly1d(z)
                            x_line = np.linspace(
                                valid[top_col].min(), valid[top_col].max(), 100
                            )
                            fig_scatter.add_trace(
                                go.Scatter(
                                    x=x_line,
                                    y=p(x_line),
                                    mode="lines",
                                    name=f"Regression (R={top_r:.3f})",
                                    line=dict(color="red", width=2, dash="dash"),
                                )
                            )
                        fig_scatter.update_layout(
                            xaxis_title=top_label,
                            yaxis_title=f"{product_name} Price (IRR)",
                            height=420,
                            margin=dict(l=60, r=30, t=30, b=50),
                        )

                    corr_rows = []
                    for c in result["correlations"]:
                        p = c["p_value"]
                        if p < 0.001:
                            sig = "★★★"
                        elif p < 0.01:
                            sig = "★★"
                        elif p < 0.05:
                            sig = "★"
                        else:
                            sig = "—"
                        corr_rows.append(
                            [
                                c["label"],
                                f"{c['r']:.4f}",
                                f"{c['r'] ** 2:.4f}",
                                f"{p:.2e}",
                                sig,
                                c["lag_at_max"],
                                f"{c['r_at_max']:.4f}",
                            ]
                        )
                    corr_df = pd.DataFrame(
                        corr_rows,
                        columns=[
                            "Feature",
                            "Pearson R",
                            "R²",
                            "p-value",
                            "Significance",
                            "Peak Lag (d)",
                            "R at Peak",
                        ],
                    )

                    entries, log_text = _log(
                        entries,
                        f"Analyze complete — {len(all_factors)} factors analysed",
                    )
                    return (
                        f"Analysed {product_name}: {ext_count} external + {feed_count} related products",
                        result["summary_md"],
                        fig_overlay,
                        corr_df,
                        fig_cross,
                        fig_rolling,
                        fig_scatter,
                        entries,
                        log_text,
                    )

                except Exception as e:
                    import traceback

                    traceback.print_exc()
                    entries, log_text = _log_err(
                        entries, "Analysis could not complete", e
                    )
                    return (
                        f"✗ ERROR: {e}",
                        "",
                        None,
                        pd.DataFrame(),
                        None,
                        None,
                        None,
                        entries,
                        log_text,
                    )

            an_btn.click(
                fn=analyze_fn,
                inputs=[an_product, an_freq, products_state, log_entries],
                outputs=[
                    an_status,
                    an_summary,
                    an_overlay,
                    an_corr_table,
                    an_cross_corr,
                    an_rolling,
                    an_scatter,
                    log_entries,
                    log_display,
                ],
            )

        # GLOBAL: Refresh Products + Sync Dropdowns

        def refresh_products_with_log(entries):
            try:
                products, source = load_products()
                df = pd.DataFrame(list(products.items()), columns=["Product", "ID"])
                names = sorted(products.keys())
                first = names[0] if names else None
                entries, log_text = _log(
                    entries, f"Products refreshed: {len(products)} from {source}"
                )
                return (
                    products,
                    f"✓ Loaded {len(products)} products from {source}",
                    df,
                    gr.update(choices=names, value=first),
                    gr.update(choices=names),
                    gr.update(choices=names, value=first),
                    gr.update(choices=names, value=first),
                    gr.update(choices=[], value=None),
                    first,
                    entries,
                    log_text,
                )
            except Exception as e:
                entries, log_text = _log_err(entries, "Product refresh failed", e)
                return (
                    {},
                    f"✗ ERROR: {e}",
                    pd.DataFrame(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    None,
                    entries,
                    log_text,
                )

        refresh_products_btn.click(
            fn=refresh_products_with_log,
            inputs=[log_entries],
            outputs=[
                products_state,
                products_status,
                products_table,
                target,
                extras,
                qp_product,
                an_product,
                model_selector,
                current_product_state,
                log_entries,
                log_display,
            ],
        )

        target.change(
            fn=sync_dropdowns,
            inputs=[target, products_state],
            outputs=[target, extras, qp_product, current_product_state],
        )

        def switch_table_display(interactive):
            updates = []
            for grid, readable in result_tables:
                updates.extend([gr.update(visible=interactive), gr.update(visible=not interactive)])
            return updates
        interactive_tables.change(
            switch_table_display, interactive_tables,
            [component for pair in result_tables for component in pair],
            queue=False, **_private_event_options(interactive_tables.change),
        )

    return demo


# LAUNCH

if __name__ == "__main__":
    app = create_app()
    launch_app(app)
