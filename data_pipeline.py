"""Market data adapters, caching, causal preprocessing and lag features."""

from __future__ import annotations

from pathlib import Path
import json
import logging
import os
import random
import re
import time
import unicodedata
from datetime import datetime, timedelta
from typing import Optional

import jdatetime
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


logger = logging.getLogger("itpnews.data_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# 1. UTILITIES


def _nfc(s):
    """Normalize a Persian/Unicode string to NFC form."""
    if s is None:
        return s
    return unicodedata.normalize("NFC", str(s))


def _nfc_keys(d: dict) -> dict:
    """Return a copy of d with all string keys NFC-normalized recursively."""
    result = {}
    for k, v in d.items():
        key = _nfc(k) if isinstance(k, str) else k
        if isinstance(v, dict):
            result[key] = _nfc_keys(v)
        elif isinstance(v, list):
            result[key] = [_nfc(x) if isinstance(x, str) else x for x in v]
        else:
            result[key] = v
    return result


_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

_LAST_REQUEST = 0.0
_MIN_INTERVAL = 0.3
_MAX_INTERVAL = 0.8
_polite_lock = __import__("threading").Lock()


def _polite_wait():
    """Thread-safe rate limiting for web requests."""
    global _LAST_REQUEST
    with _polite_lock:
        now = time.time()
        elapsed = now - _LAST_REQUEST
        delay = random.uniform(_MIN_INTERVAL, _MAX_INTERVAL)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        _LAST_REQUEST = time.time()


def _rotate_ua():
    return random.choice(_USER_AGENTS)


# 2. CONSTANTS

BASE_URL = "https://price.itpnews.com"
LIST_URL = f"{BASE_URL}/dashboard/Domestic-prices"
ITPNEWS_CHART_URL = "https://itpnews.com/fa/engine/ajax/get_prices.php"
MONTH_URL = f"{BASE_URL}/dashboard/market/month"
LAST_YEAR_URL = f"{BASE_URL}/dashboard/market/lastYear"

PERSIAN_MONTHS = {
    "فروردین": 1,
    "اردیبهشت": 2,
    "خرداد": 3,
    "تیر": 4,
    "مرداد": 5,
    "شهریور": 6,
    "مهر": 7,
    "آبان": 8,
    "آذر": 9,
    "دی": 10,
    "بهمن": 11,
    "اسفند": 12,
}

CACHE_DIR = str(Path(__file__).resolve().parent / "data" / "cache")
EXTERNAL_CACHE_DIR = str(Path(__file__).resolve().parent / "data" / "external")
MODELS_DIR = str(Path(__file__).resolve().parent / "data" / "models")
FORECASTS_DIR = str(Path(__file__).resolve().parent / "data" / "forecasts")
PRODUCTS_CACHE_FILE = str(Path(__file__).resolve().parent / "data" / "products.json")

MAX_CACHE_DAYS = 1825
DEFAULT_SMOOTH_WINDOW = 7

for _d in (CACHE_DIR, EXTERNAL_CACHE_DIR, MODELS_DIR, FORECASTS_DIR):
    os.makedirs(_d, exist_ok=True)


# 3. ALL 40 PRODUCTS (from v14)

ALL_PRODUCTS = {
    "کنجاله سویا": 1,
    "ذرت": 13,
    "تخم مرغ": 19,
    "جوجه یکروزه گوشتی": 29,
    "پودر ماهی": 34,
    "گندم": 39,
    "مرغ زنده": 42,
    "کنجاله کلزا": 55,
    "مرغ کشتار": 67,
    "متیونین": 77,
    "لیزین": 78,
    "کنجاله آفتابگردان": 82,
    "فسفات": 93,
    "جو": 95,
    "یونجه": 96,
    "کنجاله تخم پنبه": 97,
    "جوش شیرین": 98,
    "سبوس گندم": 99,
    "گوشت مرغ مادر": 100,
    "جوجه یکروزه تخمگذار": 101,
    "پودر چربی": 200,
    "کربنات کلسیم": 201,
    "کاه": 203,
    "باگاس": 204,
    "سبوس برنج": 205,
    "گوشت منجمد گوسفند": 309,
    "گوشت منجمد گوساله": 310,
    "جوجه یکروزه بوقلمون": 311,
    "بوقلمون زنده": 312,
    "بوقلمون کشتار": 313,
    "ترئونین": 316,
    "کولین": 317,
    "پولت": 318,
    "شترمرغ زنده": 320,
    "شترمرغ کشتار": 321,
    "جوجه شترمرغ": 323,
    "گوسفند زنده": 505,
    "گوساله زنده": 506,
    "گوسفند کشتار": 507,
    "گوساله کشتار": 508,
}

EMERGENCY_FALLBACK = _nfc_keys(ALL_PRODUCTS.copy())


# 4. IRANIAN HOLIDAYS

IRAN_HOLIDAYS_JALALI = [
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 12),
    (1, 13),
    (3, 14),
    (3, 15),
    (11, 22),
    (12, 29),
]


# 5. RELATED PRODUCTS — FULL GRAPH (from v14)

RELATED_PRODUCTS = _nfc_keys(
    {
        # Tier 1 - Flagship Meats
        "گوساله کشتار": [
            "ذرت",
            "کنجاله سویا",
            "جو",
            "گوساله زنده",
            "مرغ کشتار",
            "گوسفند کشتار",
        ],
        "گوسفند کشتار": [
            "ذرت",
            "کنجاله سویا",
            "گندم",
            "گوسفند زنده",
            "گوساله کشتار",
            "مرغ کشتار",
        ],
        "مرغ کشتار": [
            "ذرت",
            "کنجاله سویا",
            "تخم مرغ",
            "مرغ زنده",
            "گوساله کشتار",
            "گوسفند کشتار",
        ],
        # Tier 2 - Live Animals
        "گوساله زنده": ["ذرت", "کنجاله سویا", "جو", "گوساله کشتار"],
        "گوسفند زنده": ["ذرت", "گندم", "گوسفند کشتار"],
        "مرغ زنده": ["ذرت", "کنجاله سویا", "مرغ کشتار", "تخم مرغ"],
        "تخم مرغ": ["ذرت", "کنجاله سویا", "مرغ کشتار", "مرغ زنده"],
        # Tier 2 - Frozen Meats
        "گوشت منجمد گوساله": ["گوساله کشتار", "ذرت", "کنجاله سویا", "گوساله زنده"],
        "گوشت منجمد گوسفند": ["گوسفند کشتار", "ذرت", "گوسفند زنده"],
        "گوشت مرغ مادر": ["مرغ کشتار", "ذرت", "کنجاله سویا"],
        # Tier 3 - Poultry
        "بوقلمون کشتار": ["ذرت", "کنجاله سویا", "مرغ کشتار", "گوساله کشتار"],
        "بوقلمون زنده": ["ذرت", "کنجاله سویا", "بوقلمون کشتار"],
        "شترمرغ کشتار": ["ذرت", "گوساله کشتار", "گوسفند کشتار"],
        "شترمرغ زنده": ["ذرت", "شترمرغ کشتار"],
        # Tier 3 - Day-Old Chicks
        "جوجه یکروزه گوشتی": ["مرغ زنده", "ذرت", "کنجاله سویا", "تخم مرغ"],
        "جوجه یکروزه تخمگذار": ["مرغ زنده", "ذرت", "تخم مرغ"],
        "جوجه یکروزه بوقلمون": ["بوقلمون زنده", "ذرت", "کنجاله سویا"],
        "جوجه شترمرغ": ["شترمرغ زنده", "ذرت"],
        "پولت": ["مرغ زنده", "ذرت", "تخم مرغ"],
        # Tier 3 - Grains & Feed Core
        "ذرت": ["کنجاله سویا", "جو", "گندم"],
        "کنجاله سویا": ["ذرت", "کنجاله آفتابگردان", "کنجاله تخم پنبه", "کنجاله کلزا"],
        "جو": ["ذرت", "گندم", "کاه", "یونجه"],
        "گندم": ["ذرت", "جو", "سبوس گندم"],
        "یونجه": ["جو", "کاه", "ذرت"],
        # Tier 3 - Feed Ingredients
        "کنجاله آفتابگردان": ["کنجاله سویا", "ذرت"],
        "کنجاله تخم پنبه": ["کنجاله سویا", "ذرت"],
        "کنجاله کلزا": ["کنجاله سویا", "ذرت"],
        "پودر ماهی": ["کنجاله سویا", "ذرت"],
        "پودر چربی": ["ذرت", "کنجاله سویا"],
        # Tier 3 - Supplements
        "سبوس گندم": ["گندم", "ذرت", "جو"],
        "سبوس برنج": ["ذرت"],
        "کاه": ["جو", "یونجه"],
        "باگاس": ["ذرت", "کنجاله سویا"],
        "جوش شیرین": ["ذرت"],
        "فسفات": ["ذرت", "کنجاله سویا"],
        "کربنات کلسیم": ["ذرت", "کنجاله سویا"],
        # Tier 3 - Amino Acids
        "متیونین": ["کنجاله سویا", "ذرت"],
        "لیزین": ["کنجاله سویا", "ذرت"],
        "ترئونین": ["کنجاله سویا", "ذرت"],
        "کولین": ["کنجاله سویا", "ذرت"],
    }
)


# 6. EXTERNAL FEATURES — ONLY IRANIAN (6 Features, ALL from tgju.org)

EXTERNAL_FEATURE_REGISTRY = {
    # 1. USD/IRR Free Market - MOST IMPORTANT (✅ WORKS)
    "usd_irr_free": {
        "fetcher": "fetch_usd_irr_free",
        "freq": "D",
        "lag_days": 1,
        "label": "USD/IRR (Free Market)",
        "source": "tgju.org",
        "symbol": "price_dollar_rl",
    },
    # 2. Gold Price (18k/gram) - Inflation signal (✅ WORKS)
    "gold_price": {
        "fetcher": "fetch_gold_price",
        "freq": "D",
        "lag_days": 1,
        "label": "Gold (18k/gram)",
        "source": "tgju.org",
        "symbol": "geram18",
    },
    # 3. EUR/IRR - Euro exchange rate (✅ WORKS)
    "eur_irr": {
        "fetcher": "fetch_eur_irr",
        "freq": "D",
        "lag_days": 1,
        "label": "EUR/IRR",
        "source": "tgju.org",
        "symbol": "price_eur",
    },
    # 4. TRY/IRR - Turkish Lira (regional trade) (✅ WORKS)
    "try_irr": {
        "fetcher": "fetch_try_irr",
        "freq": "D",
        "lag_days": 1,
        "label": "TRY/IRR",
        "source": "tgju.org",
        "symbol": "price_try",
    },
    # 5. CAD/IRR - Canadian Dollar (✅ WORKS, replaces gold_ounce)
    "cad_irr": {
        "fetcher": "fetch_cad_irr",
        "freq": "D",
        "lag_days": 1,
        "label": "CAD/IRR",
        "source": "tgju.org",
        "symbol": "price_cad",
    },
    # 6. GBP/IRR - British Pound (✅ WORKS, new)
    "gbp_irr": {
        "fetcher": "fetch_gbp_irr",
        "freq": "D",
        "lag_days": 1,
        "label": "GBP/IRR",
        "source": "tgju.org",
        "symbol": "price_gbp",
    },
}

# ALL products get access to ALL 6 external features (smart selection will keep only helpful ones)
ALL_EXTERNAL_FEATURES = list(EXTERNAL_FEATURE_REGISTRY.keys())

PRODUCT_PROFILES = _nfc_keys(
    {
        "_default": {
            "related_products": True,
            "external_features": ALL_EXTERNAL_FEATURES,
        }
    }
)


def get_product_profile(name: str) -> dict:
    """Return the profile dict for a product, falling back to _default."""
    key = _nfc(name)
    profile = PRODUCT_PROFILES.get(key, PRODUCT_PROFILES["_default"])
    return dict(profile)


def list_external_features(name: str) -> list:
    """Return list of external feature keys configured for this product."""
    return list(get_product_profile(name).get("external_features", []))


# 7. DATE UTILITIES


def jalali_to_gregorian(jy: int, jm: int, jd: int):
    """Convert Jalali date to Gregorian date string."""
    try:
        d = jdatetime.date(int(jy), int(jm), int(jd)).togregorian()
        return d.isoformat()
    except Exception as exc:
        logger.warning("jalali_to_gregorian failed: %s", exc)
        return None


def gregorian_to_jalali_tuple(dt):
    """Convert Gregorian datetime to (jy, jm, jd) tuple using jdatetime library."""
    try:
        ts = pd.Timestamp(dt)
        jd = jdatetime.date.fromgregorian(date=ts.date())
        return (jd.year, jd.month, jd.day)
    except Exception as exc:
        logger.warning("gregorian_to_jalali_tuple failed: %s", exc)
        return (None, None, None)


PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def clean_number(text):
    """Parse Persian/formatted number string into int or float."""
    if text is None:
        return None
    if isinstance(text, (int, float)):
        return text
    s = _nfc(str(text)).strip()
    if not s:
        return None
    for i, d in enumerate(PERSIAN_DIGITS):
        s = s.replace(d, str(i))
    for i, d in enumerate(ARABIC_DIGITS):
        s = s.replace(d, str(i))
    s = s.replace(",", "").replace("٬", "").replace(" ", "")
    s = re.sub(r"[^\d.\-]", "", s)
    if not s or s in ("-", "."):
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return None


def clean_text(text):
    """Trim, NFC-normalize, collapse whitespace."""
    if text is None:
        return ""
    s = _nfc(str(text))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_iran_holiday(date_val) -> int:
    """Return 1 if date is Iranian holiday (Thu/Fri + fixed solar holidays)."""
    try:
        dt = pd.Timestamp(date_val)
        # Thursday (3) & Friday (4) - Iranian weekend
        if dt.dayofweek in [3, 4]:
            return 1
        # Fixed solar holidays ±1 day
        jy, jm, jd = gregorian_to_jalali_tuple(dt)
        if jm is None:
            return 0
        for hm, hd in IRAN_HOLIDAYS_JALALI:
            if hm == jm and abs(hd - jd) <= 1:
                return 1
        return 0
    except Exception:
        return 0


def _add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_holiday column."""
    df = df.copy()
    if "date" in df.columns:
        dates = df["date"]
    else:
        dates = df.index.to_series()
    df["is_holiday"] = dates.apply(_is_iran_holiday).values
    return df


# 8. FETCHER (ITPNews)


def _base_headers():
    return {
        "User-Agent": _rotate_ua(),
        "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Cache-Control": "max-age=0",
    }


def _ajax_headers():
    h = _base_headers()
    h["X-Requested-With"] = "XMLHttpRequest"
    h["Content-Type"] = "application/x-www-form-urlencoded; charset=UTF-8"
    return h


class Fetcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(_base_headers())
        self._csrf = None

    def _get_csrf(self):
        if self._csrf:
            return self._csrf
        try:
            _polite_wait()
            resp = self.session.get(LIST_URL, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            tag = soup.find("meta", {"name": "csrf-token"})
            if tag and tag.get("content"):
                self._csrf = tag["content"]
                return self._csrf
            for script in soup.find_all("script"):
                if script.string:
                    m = re.search(r'csrf_token[\s"]+[\'"]?([a-f0-9]+)', script.string)
                    if m:
                        self._csrf = m.group(1)
                        return self._csrf
        except Exception as exc:
            logger.warning("csrf fetch failed: %s", exc)
        return None

    def _post_json(self, url, pid, retries=3, **extra):
        token = self._get_csrf()
        payload = {"id": str(pid)}
        payload.update(extra)
        headers = _ajax_headers()
        headers["Referer"] = f"{BASE_URL}/dashboard/market/{pid}/chart"
        if token:
            headers["X-CSRF-TOKEN"] = token
        for attempt in range(retries):
            try:
                _polite_wait()
                headers["User-Agent"] = _rotate_ua()
                resp = self.session.post(url, data=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        return None

    def discover_products(self, force=False):
        cached = self._load_products_cache()
        if cached and not force:
            return cached, "cache"
        try:
            _polite_wait()
            resp = self.session.get(LIST_URL, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            products = {}
            select = soup.find("select", attrs={"name": "p_id"})
            if select:
                for opt in select.find_all("option"):
                    val = opt.get("value")
                    name = opt.get_text(strip=True)
                    if val and val.isdigit() and name:
                        products[name] = int(val)
            if products:
                self._save_products_cache(products)
                return products, "live"
        except Exception as exc:
            logger.warning("discover_products live fetch failed: %s", exc)

        cached = self._load_products_cache()
        if cached:
            return cached, "cache"

        return dict(EMERGENCY_FALLBACK), "fallback"

    def _save_products_cache(self, products):
        try:
            with open(PRODUCTS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(products, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_products_cache(self):
        try:
            if os.path.exists(PRODUCTS_CACHE_FILE):
                with open(PRODUCTS_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def fetch_chart(self, pid, fy, fm, fd, ty, tm, td, sub=0, prd="d"):
        from_year = int(fy)
        to_year = int(ty) + 1
        params = {
            "type": "chart",
            "p_id": str(pid),
            "format": prd or "d",
            "from_d": str(from_year),
            "to_d": str(to_year),
        }
        try:
            _polite_wait()
            self.session.headers.update({"User-Agent": _rotate_ua()})
            resp = self.session.get(ITPNEWS_CHART_URL, params=params, timeout=60)
            resp.raise_for_status()
            raw = json.loads(resp.text)
            return self._parse_itpnews_chart(raw)
        except Exception as exc:
            logger.warning("fetch_chart failed for %s: %s", pid, exc)
            return pd.DataFrame(columns=["date", "price"])

    @staticmethod
    def _parse_itpnews_chart(raw):
        if not isinstance(raw, dict) or "cats" not in raw:
            return pd.DataFrame(columns=["date", "price"])
        records = []
        cats = raw["cats"]
        avg_prices = raw.get("avg_price", [])
        min_prices = raw.get("min_price", [])
        max_prices = raw.get("max_price", [])
        for i in range(min(len(cats), len(avg_prices))):
            cat = cats[i]
            parts = cat.split("-")
            if len(parts) != 3:
                continue
            month_name = parts[0]
            day = parts[1]
            year_str = parts[2]
            month = PERSIAN_MONTHS.get(month_name)
            if month is None:
                continue
            date_str = jalali_to_gregorian(int(year_str), month, int(day))
            if not date_str:
                continue
            price = clean_number(avg_prices[i])
            if price is None:
                continue
            rec = {"date": date_str, "price": price}
            if i < len(min_prices):
                min_p = clean_number(min_prices[i])
                if min_p is not None:
                    rec["min_price"] = min_p
            if i < len(max_prices):
                max_p = clean_number(max_prices[i])
                if max_p is not None:
                    rec["max_price"] = max_p
            records.append(rec)
        if not records:
            return pd.DataFrame(columns=["date", "price"])
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        return (
            df.dropna(subset=["date"])
            .drop_duplicates("date")
            .sort_values("date")
            .reset_index(drop=True)
        )


# 9. EXTERNAL DATA SOURCE (ONLY tgju.org — RELIABLE, NO API KEYS)


class ExternalDataSource:
    def __init__(self, cache_dir: str = EXTERNAL_CACHE_DIR):
        self.session = requests.Session()
        self.session.headers.update(_base_headers())
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.csv")

    def _load_cache(self, key: str):
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            if not df.empty:
                return df
        except Exception as exc:
            logger.warning("external cache load failed for %s: %s", key, exc)
        return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            df.to_csv(self._cache_path(key), index=False)
        except Exception as exc:
            logger.warning("external cache save failed for %s: %s", key, exc)

    def _fetch_tgju_series(self, symbol: str):
        """Core fetcher for ALL Iranian external features."""
        try:
            url = (
                f"https://api.tgju.org/v1/market/indicator/summary-table-data/{symbol}"
            )
            _polite_wait()
            self.session.headers.update({"User-Agent": _rotate_ua()})
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get("data", [])
            recs = []
            for row in rows:
                if len(row) < 7:
                    continue
                close_txt = re.sub(r"[^\d.]", "", str(row[3]))
                if not close_txt:
                    continue
                dt = pd.to_datetime(str(row[6]), errors="coerce")
                if pd.isna(dt):
                    continue
                recs.append({"date": dt.normalize(), "value": float(close_txt)})
            if not recs:
                return None
            df = pd.DataFrame(recs).dropna().sort_values("date")
            df = df.drop_duplicates(subset="date", keep="last")
            return df
        except Exception as exc:
            logger.warning("tgju fetch failed for %s: %s", symbol, exc)
            return None

    def get(self, key: str, force_refresh: bool = False):
        """Fetch external feature by key with cache fallback."""
        info = EXTERNAL_FEATURE_REGISTRY.get(key)
        if info is None:
            logger.warning("unknown external feature key: %s", key)
            return None

        cached = None
        if not force_refresh:
            cached = self._load_cache(key)
            if cached is not None:
                return cached

        # Fetch using the mapped symbol
        symbol = info.get("symbol")
        if not symbol:
            return None

        df = self._fetch_tgju_series(symbol)
        if df is not None and not df.empty:
            self._save_cache(key, df)
            return df

        return cached

    # ─── Individual fetchers (all use _fetch_tgju_series) ───
    def fetch_usd_irr_free(self, force_refresh: bool = False):
        return self.get("usd_irr_free", force_refresh)

    def fetch_gold_price(self, force_refresh: bool = False):
        return self.get("gold_price", force_refresh)

    def fetch_eur_irr(self, force_refresh: bool = False):
        return self.get("eur_irr", force_refresh)

    def fetch_try_irr(self, force_refresh: bool = False):
        return self.get("try_irr", force_refresh)

    def fetch_cad_irr(self, force_refresh: bool = False):
        return self.get("cad_irr", force_refresh)

    def fetch_gbp_irr(self, force_refresh: bool = False):
        return self.get("gbp_irr", force_refresh)


external_data = ExternalDataSource()


# 10. CACHE (ITPNews Product Data)


class Cache:
    def __init__(self, fetcher: Fetcher):
        self.fetcher = fetcher

    def _path(self, pid):
        safe = re.sub(r"[^\w\-]", "_", str(pid))
        return os.path.join(CACHE_DIR, f"product_{safe}.csv")

    def load(self, pid) -> pd.DataFrame:
        path = self._path(pid)
        if not os.path.exists(path):
            return pd.DataFrame(columns=["date", "price"])
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            return df.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            logger.warning("cache load failed for %s: %s", pid, exc)
            return pd.DataFrame(columns=["date", "price"])

    def save(self, pid, df: pd.DataFrame):
        try:
            df.sort_values("date").drop_duplicates("date").to_csv(
                self._path(pid), index=False
            )
        except Exception as exc:
            logger.warning("cache save failed for %s: %s", pid, exc)

    def update(self, pid, force=False):
        """Incrementally fetch new data since last cache."""
        existing = self.load(pid)
        today = datetime.today()

        path = self._path(pid)
        if (
            not force
            and not existing.empty
            and time.time() - os.path.getmtime(path) < 12 * 3600
        ):
            return existing, "cached"
        if not force and not existing.empty:
            last = existing["date"].max()
            start = last - timedelta(days=5)
            fy, fm, fd = gregorian_to_jalali_tuple(start)
        else:
            fy, fm, fd = 1390, 1, 1

        ty, tm, td = gregorian_to_jalali_tuple(today)

        try:
            if fy is None or ty is None:
                raise ValueError("invalid jalali conversion")
            new_df = self.fetcher.fetch_chart(pid, fy, fm, fd, ty, tm, td)
        except Exception as exc:
            logger.warning("cache.update fetch failed: %s", exc)
            new_df = pd.DataFrame(columns=["date", "price"])

        if new_df.empty and existing.empty:
            return existing, "no_data"
        if new_df.empty:
            return existing, "cached"

        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = (
            merged.drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        merged = merged[["date", "price"]]
        if len(merged) > MAX_CACHE_DAYS:
            merged = merged.tail(MAX_CACHE_DAYS).reset_index(drop=True)
        self.save(pid, merged)
        return merged, "updated"


# 11. DATA PROCESSING FUNCTIONS


def preprocess(
    df: pd.DataFrame,
    target: str,
    freq: str = "D",
    iqr_k: Optional[float] = None,
    adaptive: bool = True,
) -> pd.DataFrame:
    """Normalize positive prices to completed periods using past observations only."""
    if freq not in {"D", "W", "M"}:
        raise ValueError("Frequency must be D, W, or M.")
    if df is None or df.empty or "date" not in df:
        raise ValueError("Provide a nonempty table with a date column.")
    df = df.copy().rename(columns={c: _nfc(c) for c in df.columns})
    target = _nfc(target)
    if target not in df:
        raise ValueError(f"Target column not found: {target}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df[target] = pd.to_numeric(df[target], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    df = df.dropna(subset=["date", target])
    df = df[df[target] > 0].drop_duplicates("date", keep="last").sort_values("date")
    if df.empty:
        raise ValueError("No valid dates and positive prices found.")
    if (df["date"].max() - df["date"].min()).days > 36525:
        raise ValueError("Date range exceeds 100 years.")
    last_date = df["date"].max()
    df["_observed"] = df["date"]
    rule = {"D": "D", "W": "W-SUN", "M": "ME"}[freq]
    df = df.set_index("date").resample(rule).last().ffill()
    df = df.loc[df.index <= last_date].reset_index()
    if df.empty:
        raise ValueError("No completed periods available for this frequency.")
    df["gap_days"] = (df["date"] - df.pop("_observed")).dt.days
    # Optional clipping uses past values only; default preserves actual market moves.
    if iqr_k is not None:
        if iqr_k <= 0:
            raise ValueError("IQR multiplier must be positive.")
        history = df[target].shift(1).expanding(min_periods=20)
        q1, q3 = history.quantile(0.25), history.quantile(0.75)
        width = q3 - q1
        df[target] = df[target].clip(q1 - iqr_k * width, q3 + iqr_k * width)
    return _add_holiday_feature(df)


def merge_external_features(
    df: pd.DataFrame, product_name: str, freq: str = "D"
) -> pd.DataFrame:
    """Merge profile-driven external features into df."""
    features = list_external_features(product_name)
    if not features:
        return df.copy()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])

    for key in features:
        info = EXTERNAL_FEATURE_REGISTRY.get(key)
        if info is None:
            continue
        try:
            ext_df = external_data.get(key)
            if ext_df is None or ext_df.empty:
                logger.warning(f"skipping external feature {key}: unavailable")
                continue
            ext_df = ext_df.copy()
            ext_df["date"] = pd.to_datetime(ext_df["date"])
            lag_days = info.get("lag_days", 0)
            if lag_days:
                ext_df["date"] = ext_df["date"] + pd.Timedelta(days=lag_days)

            col_name = f"ext_{key}"
            ext_df = ext_df.rename(columns={"value": col_name})[["date", col_name]]
            ext_df = ext_df.drop_duplicates(subset="date", keep="last").sort_values(
                "date"
            )

            out = out.sort_values("date")
            out = pd.merge_asof(out, ext_df, on="date", direction="backward")
            out[col_name] = out[col_name].ffill()
            if out[col_name].isna().any():
                out[col_name] = out[col_name].ffill()
        except Exception as exc:
            logger.warning("merge_external_features failed for %s: %s", key, exc)
            continue

    return out


def fetch_related_prices(
    df: pd.DataFrame, product_name: str, fetcher, cache, freq: str = "D"
) -> pd.DataFrame:
    """Fetch RELATED_PRODUCTS price data and merge as ext_feed_ columns."""
    related = RELATED_PRODUCTS.get(_nfc(product_name), [])
    if not related:
        return df.copy()

    products, _ = fetcher.discover_products()
    norm_products = {_nfc(k): v for k, v in products.items()}
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])

    for rp in related:
        rp_nfc = _nfc(rp)
        rp_id = norm_products.get(rp_nfc)
        if not rp_id:
            continue
        try:
            _polite_wait()
            rp_df, _ = cache.update(rp_id, force=False)
            if rp_df.empty or "price" not in rp_df.columns:
                continue
            col = f"ext_feed_{rp}"
            rp_df = rp_df[["date", "price"]].rename(columns={"price": col})
            rp_df["date"] = pd.to_datetime(rp_df["date"])
            result = pd.merge_asof(
                result.sort_values("date"),
                rp_df.sort_values("date"),
                on="date",
                direction="backward",
            )
            result[col] = result[col].ffill()
            if result[col].isna().any():
                result[col] = result[col].ffill()
        except Exception as exc:
            logger.warning("fetch_related_prices failed for %s: %s", rp, exc)
            continue

    return result


def create_features(
    df,
    target_col,
    lags,
    windows,
    freq="D",
    other_cols=None,
    external_cols=None,
    add_seasonal=True,
):
    """
    Engineer features for ML training.
    ═══ CRITICAL: ALL features use shift(1) to prevent future leakage ═══
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if not lags or any(type(v) is not int or v < 1 or v > 730 for v in lags):
        raise ValueError("Lags must be integers from 1 to 730.")
    if not windows or any(type(v) is not int or v < 2 or v > 730 for v in windows):
        raise ValueError("Rolling windows must be integers from 2 to 730.")
    feature_cols = []
    seasonal_cols = []

    # Seasonality
    if add_seasonal:
        month = df["date"].dt.month
        dow = df["date"].dt.dayofweek
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        if "is_holiday" not in df.columns:
            df = _add_holiday_feature(df)
        seasonal_cols = ["month_sin", "month_cos", "dow_sin", "dow_cos", "is_holiday"]
        feature_cols.extend([c for c in seasonal_cols if c in df.columns])

    # Gap-days: days since last trade (computed in preprocess, known at prediction time)
    if "gap_days" in df.columns:
        feature_cols.append("gap_days")

    # ═══ CRITICAL: shift(1) to prevent leakage ═══
    shifted = df[target_col].shift(1)

    # Lags
    for lag in lags:
        if len(df) > lag:
            col = f"{target_col}_lag_{lag}"
            df[col] = df[target_col].shift(lag)
            feature_cols.append(col)

    # Rolling windows
    for w in windows:
        if len(df) >= w:
            df[f"{target_col}_ma_{w}"] = shifted.rolling(w).mean().ffill()
            df[f"{target_col}_std_{w}"] = shifted.rolling(w).std().ffill()
            feature_cols.extend([f"{target_col}_ma_{w}", f"{target_col}_std_{w}"])

    # ROC (Rate of Change)
    for period in windows:
        if len(df) > period:
            df[f"{target_col}_roc_{period}"] = (
                shifted.pct_change(period, fill_method=None) * 100
            )
            feature_cols.append(f"{target_col}_roc_{period}")

    # Position & CV
    for w in windows:
        if len(df) >= w:
            roll_min = shifted.rolling(w).min().ffill()
            roll_max = shifted.rolling(w).max().ffill()
            roll_range = roll_max - roll_min
            df[f"{target_col}_position_{w}"] = (
                (shifted - roll_min) / roll_range.replace(0, 1.0)
            ).clip(0, 1)
            roll_ma = shifted.rolling(w).mean().ffill()
            roll_std = shifted.rolling(w).std().ffill()
            df[f"{target_col}_cv_{w}"] = roll_std / roll_ma.abs().clip(lower=1e-9)
            feature_cols.extend([f"{target_col}_position_{w}", f"{target_col}_cv_{w}"])

    # Oscillator
    short_w = [w for w in windows if w <= 14]
    long_w = [w for w in windows if w >= 21]
    for sw in short_w:
        for lw in long_w:
            if sw < lw and len(df) >= lw:
                sma = shifted.rolling(sw).mean().ffill()
                lma = shifted.rolling(lw).mean().ffill()
                df[f"{target_col}_osc_{sw}_{lw}"] = (sma - lma) / lma.abs().clip(
                    lower=1e-9
                )
                feature_cols.append(f"{target_col}_osc_{sw}_{lw}")

    # Other columns (related products)
    if other_cols:
        for other in other_cols:
            if other == target_col or other == "date" or other not in df.columns:
                continue
            safe = re.sub(r"[^\w]", "_", other)
            other_shifted = df[other].shift(1)
            df[f"{target_col}_ratio_{safe}"] = shifted / other_shifted.replace(
                0, np.nan
            )
            df[f"{target_col}_margin_{safe}"] = shifted - other_shifted
            feature_cols.extend(
                [f"{target_col}_ratio_{safe}", f"{target_col}_margin_{safe}"]
            )
            for w in windows:
                if len(df) >= w:
                    col_corr = f"{target_col}_corr_{safe}_{w}"
                    df[col_corr] = shifted.rolling(w).corr(other_shifted).fillna(0)
                    feature_cols.append(col_corr)

    # External columns (macro features) — windows capped at 14
    if external_cols:
        for ec in external_cols:
            if ec not in df.columns:
                continue
            feature_cols.append(ec)
            ec_shifted = df[ec].shift(1)
            df[ec] = ec_shifted
            for w in [w for w in windows if w <= 14]:
                if len(df) >= w:
                    df[f"{ec}_ma_{w}"] = ec_shifted.rolling(w).mean().ffill()
                    df[f"{ec}_std_{w}"] = ec_shifted.rolling(w).std().ffill()
                    feature_cols.extend([f"{ec}_ma_{w}", f"{ec}_std_{w}"])

    # Clean up
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [
        c
        for c in feature_cols
        if c in df.columns and np.issubdtype(df[c].dtype, np.number)
    ]
    # Drop columns with too many NaN (prevents training crashes)

    df = df.replace([np.inf, -np.inf], np.nan)

    return df, feature_cols, seasonal_cols


def rank_features(model, feature_cols):
    """Rank features by importance. Works with sklearn models and ensembles."""
    model = getattr(model, "regressor_", model)
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importances = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    elif hasattr(model, "estimators_"):
        sub_importances = []
        for est in model.estimators_:
            est = getattr(est, "regressor_", est)
            if hasattr(est, "feature_importances_"):
                sub_importances.append(
                    np.asarray(est.feature_importances_, dtype=float)
                )
            elif hasattr(est, "coef_"):
                sub_importances.append(
                    np.abs(np.asarray(est.coef_, dtype=float)).ravel()
                )
        importances = (
            np.mean(sub_importances, axis=0)
            if sub_importances
            else np.zeros(len(feature_cols))
        )
    else:
        importances = np.zeros(len(feature_cols))

    if len(importances) != len(feature_cols):
        importances = np.zeros(len(feature_cols))

    pairs = list(zip(feature_cols, importances.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs


def check_data_quality(df: pd.DataFrame, target_col: str) -> list:
    """Return warning strings about data quality."""
    warnings_list = []
    if df is None or df.empty:
        warnings_list.append("Dataset is empty.")
        return warnings_list

    series = df[target_col]
    n = len(series)
    zeros = (series == 0).sum()
    nans = series.isna().sum()

    if zeros > 0:
        warnings_list.append(f"{zeros} zero prices ({zeros / n * 100:.1f}%)")
    if nans > 0:
        warnings_list.append(f"{nans} missing values ({nans / n * 100:.1f}%)")
    if n < 60:
        warnings_list.append(f"Only {n} rows (need 60+ for reliable training)")

    diffs = df["date"].diff().dropna()
    if len(diffs) > 0:
        max_gap = diffs.max()
        if max_gap > pd.Timedelta(days=14):
            warnings_list.append(f"Data gap of {max_gap.days} days detected")

    if len(series) > 10:
        q1, q99 = series.quantile(0.01), series.quantile(0.99)
        if q1 > 0:
            ratio = q99 / q1
            if ratio > 10:
                warnings_list.append(
                    f"Large price range: {q1:,.0f} to {q99:,.0f} ({ratio:.0f}x)"
                )

    return warnings_list


def detrend_series(
    train_df, target, freq="D", method="linear", smooth_w=None, split_date=None
):
    """
    Fit a trend on the TRAIN portion only.

    Methods:
    - "linear": Linear trend on raw prices
    - "log": Linear trend on log(prices) → for volatile products
    - "quadratic": Quadratic trend
    - "diff": First difference
    - "smooth_diff": Smooth first difference
    - "none": No detrending

    Returns:
    - (coeffs, ref_date) for linear/quadratic/log
    - (None, None, w) for smooth_diff
    - (None, None) for diff
    - (None, Timestamp) for none
    """
    df = train_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if split_date is not None:
        df = df[df["date"] < split_date]
    df = df.dropna(subset=[target])

    # ═══ LOG transform for volatile products ═══
    if method == "log":
        df[target] = np.log(np.clip(df[target].values, 1, None))
        ref_date = df["date"].min()
        x = (df["date"] - ref_date).dt.days.values.astype(float)
        y = df[target].values.astype(float)
        coeffs = (
            np.polyfit(x, y, 1) if len(x) > 1 else np.array([0.0, float(np.mean(y))])
        )
        return {"method": "log", "coefficients": coeffs.tolist()}, ref_date

    if method == "smooth_diff":
        w = smooth_w or DEFAULT_SMOOTH_WINDOW
        return None, None, w
    if method == "diff":
        return None, None
    if method == "none":
        return None, pd.Timestamp.today()

    if df.empty:
        ref_date = pd.Timestamp.today()
        coeffs = (
            np.array([0.0, 0.0, 0.0]) if method == "quadratic" else np.array([0.0, 0.0])
        )
        return coeffs, ref_date

    ref_date = df["date"].min()
    x = (df["date"] - ref_date).dt.days.values.astype(float)
    y = df[target].values.astype(float)

    if method == "quadratic":
        coeffs = (
            np.polyfit(x, y, 2)
            if len(x) > 2
            else np.array([0.0, 0.0, float(np.mean(y))])
        )
    else:
        coeffs = (
            np.polyfit(x, y, 1) if len(x) > 1 else np.array([0.0, float(np.mean(y))])
        )

    return coeffs, ref_date


def apply_trend(dates, coeffs, ref_date, freq="D"):
    """Evaluate the fitted trend polynomial at the given dates."""
    if coeffs is None:
        return np.zeros(len(dates))
    dates = pd.to_datetime(pd.Series(dates))
    x = (dates - ref_date).dt.days.values.astype(float)
    values = coeffs["coefficients"] if isinstance(coeffs, dict) else coeffs
    return np.polyval(values, x)


def select_features(df, feature_cols, split_date=None, max_corr=0.95):
    """Drop highly collinear features using correlation on train rows only."""
    if len(feature_cols) <= 5:
        return list(feature_cols)

    train_df = df
    if split_date is not None and "date" in df.columns:
        train_df = df[pd.to_datetime(df["date"]) < split_date]
    if len(train_df) < 10:
        return list(feature_cols)

    cols = list(dict.fromkeys([c for c in feature_cols if c in train_df.columns]))
    if len(cols) < 2:
        return list(cols)

    corr = train_df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        high_corr = upper.index[upper[col] > max_corr].tolist()
        to_drop.update(high_corr)

    return [c for c in cols if c not in to_drop]


def auto_params(df, target_col, freq="D"):
    """Heuristically choose lags/windows based on dataset length and volatility."""
    n = len(df) if df is not None else 0
    series = df[target_col] if target_col in df.columns else df.iloc[:, 1]
    returns = series.pct_change().dropna()
    vol = float(returns.std()) if len(returns) > 5 else 0.02

    if freq == "D":
        if n > 1500:
            lags, windows = [1, 7, 14, 30], [7, 14, 30]
        elif n > 700:
            lags, windows = [1, 7, 14, 30], [7, 14, 30]
        elif n > 300:
            lags, windows = [1, 7, 14], [7, 14, 30]
        elif n > 100:
            lags, windows = [1, 3, 7, 14], [7, 14]
        else:
            lags, windows = [1, 3, 7], [7, 14]
    elif freq == "W":
        if n > 100:
            lags, windows = [1, 4, 12, 24], [4, 12, 24]
        else:
            lags, windows = [1, 4, 8], [4, 8]
    else:
        if n > 36:
            lags, windows = [1, 3, 6, 12], [3, 6, 12]
        else:
            lags, windows = [1, 2, 3], [3, 6]

    # Adjust for volatility
    if vol > 0.05:
        windows = [w for w in windows if w >= 14]
    if vol < 0.005:
        lags, windows = lags[:4], windows[:4]

    return lags, windows


def generate_fallback(n_days=730, base=8_000_000) -> pd.DataFrame:
    """Generate an explicitly synthetic fixture; never used as a market-data fallback."""
    dates = pd.date_range(
        end=pd.Timestamp.today().normalize(), periods=n_days, freq="D"
    )
    rng = np.random.default_rng(42)
    trend = np.linspace(0, 0.15, n_days)
    seasonal = 0.03 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = rng.normal(0, 0.03, n_days)
    prices = base * (1 + trend + seasonal + np.cumsum(noise) * 0.1)
    prices = np.maximum(prices, base * 0.3)
    return pd.DataFrame({"date": dates, "price": prices.astype(int)})
