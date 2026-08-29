# save as test_data_pipeline.py and run: python test_data_pipeline.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print("DATA PIPELINE TEST – گوساله کشتار")
print("=" * 60)

from data_pipeline import (
    Fetcher, Cache, preprocess, merge_external_features,
    fetch_related_prices, create_features, detrend_series, apply_trend,
    select_features, generate_fallback, auto_params,
)
from model_pipeline import compute_split_date, train_model, get_model

# ── Setup ──
fetcher = Fetcher()
cache = Cache(fetcher)
products, source = fetcher.discover_products()
pn = "گوساله کشتار"
pid = products.get(pn, list(products.values())[0])
target = "price"
print(f"Product: {pn} (ID: {pid}), source: {source}")

# 1. FETCH & PREPROCESS
raw_df, status = cache.update(pid, force=False)
if raw_df.empty:
    raw_df = generate_fallback()
    status = "fallback"
df = preprocess(raw_df, target, "D")
print(f"Cache: {status} | Rows: {len(df)} | Holidays: {df['is_holiday'].sum()} ({(df['is_holiday'].mean()*100):.0f}%)")
print(f"Columns: {list(df.columns)}")
print(f"Price range: {df[target].min():,.0f} → {df[target].max():,.0f}")

# 2. EXTERNAL FEATURES
df = merge_external_features(df, pn, "D")
ext_cols = [c for c in df.columns if c.startswith("ext_")]
print(f"\nExternal columns: {len(ext_cols)} — {ext_cols}")
for ec in ext_cols:
    print(f"  {ec}: {df[ec].notna().sum()}/{len(df)} non‑null, range [{df[ec].min():,.0f}, {df[ec].max():,.0f}]")

# 3. RELATED PRODUCTS
df = fetch_related_prices(df, pn, fetcher, cache, "D")
feed_cols = [c for c in df.columns if c.startswith("ext_feed_")]
print(f"\nFeed columns: {len(feed_cols)} — {feed_cols}")

# 4. FEATURE ENGINEERING
split_date = compute_split_date(df)
lags, windows = auto_params(df, target, "D")
print(f"\nSplit: {split_date.date()}, Lags: {lags}, Windows: {windows}")

detrended = f"{target}_detrended"
detrend_result = detrend_series(df[df["date"] < split_date], target, "D", "linear", split_date=split_date)
if len(detrend_result) == 3:
    coeffs, ref_date = None, None
else:
    coeffs, ref_date = detrend_result

if coeffs is not None:
    df[detrended] = df[target] - apply_trend(df["date"], coeffs, ref_date, "D")
else:
    df[detrended] = df[target].diff()

ext_present = [c for c in df.columns if c.startswith("ext_")]
other_cols = [c for c in df.columns if c not in [target, "date", detrended, "smooth", "is_holiday"]
              and np.issubdtype(df[c].dtype, np.number) and df[c].nunique() > 10]

df_feat, feature_cols, _ = create_features(df, detrended, lags, windows, "D",
                                           other_cols=other_cols, external_cols=ext_present or None,
                                           add_seasonal=True)
for lag in lags:
    col = f"{detrended}_lag_{lag}"
    df_feat[col] = df_feat[detrended].shift(lag)
    feature_cols.append(col)

feature_cols = list(dict.fromkeys([c for c in feature_cols if c in df_feat.columns]))
feature_cols = select_features(df_feat, feature_cols, split_date, max_corr=0.95)
df_clean = df_feat.dropna(subset=[detrended] + [c for c in feature_cols if c in df_feat.columns])

print(f"\nFinal dataset: {len(df_clean)} rows × {len(feature_cols)} features")
print(f"  Own‑graph features: {len([c for c in feature_cols if not c.startswith('ext_')])}")
print(f"  External features: {len([c for c in feature_cols if c.startswith('ext_')])}")

# 5. QUICK MODEL SANITY CHECK (just Ridge & Lasso)
X = df_clean[feature_cols].values
y = df_clean[detrended].values
dates = pd.to_datetime(df_clean["date"]).values
train_mask = dates < np.datetime64(pd.to_datetime(split_date))
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

for name in ["Ridge", "Lasso"]:
    model, params = get_model(name)
    result = train_model(X_train, y_train, X_test, y_test, model, params, fast=True)
    print(f"\n{name}: MAE={result['mae']:,.0f}, R²={result['r2']:.3f}")

print("\n✓ Data pipeline works correctly. Ready for full model training.")