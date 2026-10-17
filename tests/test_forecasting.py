"""Synthetic temporal regression tests; no live market requests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
import data_pipeline as data
import model_pipeline as models


def series(n=240):
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n),
            "price": 1000 + np.arange(n) * 5 + np.sin(np.arange(n)),
        }
    )


def test_preprocessing_is_prefix_invariant():
    frame = series()
    prefix = data.preprocess(frame.iloc[:120], "price")
    frame.loc[120:, "price"] *= 1000
    full = data.preprocess(frame, "price")
    pd.testing.assert_frame_equal(prefix, full.iloc[:120])


@pytest.mark.parametrize("freq,rule", [("D", "D"), ("W", "W-SUN"), ("M", "ME")])
def test_completed_periods_and_future_alignment(freq, rule):
    clean = data.preprocess(series(), "price", freq)
    assert clean.date.max() <= series().date.max()
    future = models.future_periods(clean.date.max(), 5, freq)
    assert future[0] > clean.date.max()
    assert len(future) == 5
    if freq == "W":
        assert all(future.dayofweek == 6)
    if freq == "M":
        assert all(future.is_month_end)


def test_features_do_not_read_future_or_current_external_values():
    frame = series()
    frame["ext_signal"] = np.arange(len(frame))
    before, columns, _ = data.create_features(
        frame, "price", [1, 7], [7, 14], external_cols=["ext_signal"]
    )
    frame.loc[100:, ["price", "ext_signal"]] = 999999
    after, _, _ = data.create_features(
        frame, "price", [1, 7], [7, 14], external_cols=["ext_signal"]
    )
    pd.testing.assert_frame_equal(before.loc[:100, columns], after.loc[:100, columns])
    assert pd.isna(before.loc[0, "price_ma_7"])


def test_target_scaling_is_preserved_and_predictions_ignore_test_labels():
    x = np.arange(120).reshape(-1, 1)
    y = x[:, 0] * 200 + 5000
    first = models.train_model(x[:90], y[:90], x[90:], y[90:], Ridge(alpha=0.001))
    second = models.train_model(
        x[:90], y[:90], x[90:], y[90:] * 100, Ridge(alpha=0.001)
    )
    a = first["model"].predict(first["scaler"].transform([[120]]))
    b = second["model"].predict(second["scaler"].transform([[120]]))
    assert a[0] == pytest.approx(29000, rel=0.001)
    np.testing.assert_allclose(a, b)


def test_log_forecast_inverse_and_model_roundtrip(tmp_path, monkeypatch):
    frame = series(200)
    frame["price"] = np.exp(4 + np.arange(200) * 0.003)
    coeff, ref = data.detrend_series(frame.iloc[:150], "price", method="log")
    frame["detrended"] = np.log(frame.price) - data.apply_trend(frame.date, coeff, ref)
    feat, columns, _ = data.create_features(frame, "detrended", [1, 7], [7, 14])
    result = models.train_model(
        feat[columns].iloc[20:150],
        feat.detrended.iloc[20:150],
        feat[columns].iloc[150:],
        feat.detrended.iloc[150:],
        Ridge(),
    )
    args = (
        frame,
        "price",
        "detrended",
        columns,
        result["model"],
        result["scaler"],
        1,
        5,
        [1, 7],
        [7, 14],
        "D",
        coeff,
        ref,
        -1,
        1,
    )
    predicted = models.iterative_forecast(*args)
    np.testing.assert_allclose(
        predicted.forecast, np.exp(4 + np.arange(200, 205) * 0.003), rtol=0.001
    )
    assert (predicted.lower <= predicted.forecast).all()
    assert (predicted.upper >= predicted.forecast).all()
    monkeypatch.setattr(models, "MODELS_DIR", str(tmp_path))
    metadata = {
        "feature_cols": columns,
        "lags": [1, 7],
        "windows": [7, 14],
        "freq": "D",
        "detrended_target": "detrended",
        "target_col": "price",
        "trend_coeffs": coeff,
        "ref_date": ref,
    }
    trained = {
        "Ridge": (
            result["model"],
            result["scaler"],
            result["mae"],
            result["r2"],
            1,
            -1,
            1,
        )
    }
    path = models.save_trained_models(trained, "sample", metadata)[0]
    restored = models.load_trained_model(path)
    np.testing.assert_allclose(
        restored["model"].predict(restored["scaler"].transform(feat[columns].tail())),
        result["model"].predict(result["scaler"].transform(feat[columns].tail())),
    )


def test_unavailable_data_never_becomes_synthetic():
    class EmptyCache:
        def update(self, *args, **kwargs):
            return pd.DataFrame(), "no_data"

    with pytest.raises(ValueError, match="unavailable"):
        models.quick_predict("sample", "1", None, EmptyCache())


def test_optional_model_is_not_silently_replaced():
    with pytest.raises(ValueError):
        models.get_model("not-a-model")


def test_saved_model_requires_forecast_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODELS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="metadata"):
        models.save_trained_models({}, "sample")


def test_zero_lags_rejected():
    with pytest.raises(ValueError, match="Lags"):
        data.create_features(series(), "price", [0], [7])


def test_quick_forecast_and_walk_forward(monkeypatch):
    frame = series(260)
    monkeypatch.setattr(data, "merge_external_features", lambda df, *a, **kw: df)
    monkeypatch.setattr(data, "fetch_related_prices", lambda df, *a, **kw: df)

    class FixtureCache:
        def update(self, *a, **kw):
            return frame.copy(), "fixture"

        def load(self, *a):
            return frame.copy()

    result = models.quick_predict(
        "fixture",
        "fixture",
        None,
        FixtureCache(),
        horizon=5,
        lags=[1, 7],
        windows=[7, 14],
        selected_models=["Ridge"],
        detrend_method="log",
    )
    assert len(result[0]) == 5
    assert np.isfinite(result[0].forecast).all()
    folds = models.backtest(
        frame, "price", [1, 7], [7, 14], "D", "linear", ["Ridge"], horizon=10, n_folds=2
    )
    assert len(folds) == 2
    assert folds.n_test.sum() == 20


def test_gradio_upload_train_forecast_and_restore(tmp_path, monkeypatch):
    import app

    monkeypatch.setattr(
        app.Fetcher, "discover_products", lambda self, **kw: ({"fixture": 1}, "fixture")
    )
    monkeypatch.setattr(app, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(app, "FORECASTS_DIR", str(tmp_path))
    monkeypatch.setattr(app, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(models, "MODELS_DIR", str(tmp_path))
    ui = app.create_app()
    callbacks = {v.fn.__name__: v.fn for v in ui.fns.values() if v.fn}
    path = tmp_path / "upload.csv"
    series().to_csv(path, index=False)
    upload = callbacks["upload_csv_fn"](str(path), "fixture", "D", [])
    frame = upload[4]
    assert len(frame) == 240
    trained = callbacks["train_fn"](
        frame,
        "fixture",
        "1,7",
        "7,14",
        True,
        True,
        True,
        "log",
        ["Ridge"],
        "D",
        False,
        "Voting",
        {},
        [],
    )
    state = trained[5]
    assert state.get("trained_models"), trained[0]
    result = callbacks["forecast_fn"](frame, "fixture", 5, "Ridge", state, 2, False, [])
    assert len(result[3]) == 5, result[0]
    assert np.isfinite(result[3].forecast).all()
    saved = callbacks["save_models_fn"](state, [])
    files = models.list_saved_models()
    assert len(files) == 1, saved[0]
    restored = callbacks["restore_saved"](files[0], True)[0]
    result = callbacks["forecast_fn"](
        frame, "fixture", 5, "Ridge", restored, 2, False, []
    )
    assert len(result[3]) == 5, result[0]
    ui.close()
