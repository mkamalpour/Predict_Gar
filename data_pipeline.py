"""
ITPNews Price Intelligence v13 - data_pipeline.py

Handles: web scraping (ITPNews + external sources), caching, cleaning,
date utilities, and feature engineering.

RULES ENFORCED IN THIS FILE:
- NEVER imports sklearn / xgboost / lightgbm.
- All features that could leak the future use shift(1).
- Unicode NFC normalization on all Persian strings.
- No product-name branches in logic: only PRODUCT_PROFILES controls behavior.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
import unicodedata
from datetime import datetime, timedelta

import jdatetime
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("itpnews.data_pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def _nfc(s):
    """Normalize a Persian/Unicode string to NFC form."""
    if s is None:
        return s
    return unicodedata.normalize("NFC", str(s))


def _nfc_keys(d: dict) -> dict:
    """Return a copy of d with all string keys NFC-normalized."""
    return {(_nfc(k) if isinstance(k, str) else k): v for k, v in d.items()}

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36 Edg/126.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

_LAST_REQUEST = 0.0
_MIN_INTERVAL = 0.8
_MAX_INTERVAL = 2.5
_polite_lock = __import__("threading").Lock()


def _polite_wait():
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

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

BASE_URL = "https://price.itpnews.com"
LIST_URL = f"{BASE_URL}/dashboard/Domestic-prices"
CHART_FILTER_URL = f"{BASE_URL}/dashboard/chart/filter"
MONTH_URL = f"{BASE_URL}/dashboard/market/month"
WEEK_URL = f"{BASE_URL}/dashboard/market/week"
PRICE_LAST_YEARS_URL = f"{BASE_URL}/dashboard/market/priceLastYears"
LAST_YEAR_URL = f"{BASE_URL}/dashboard/market/lastYear"
ITPNEWS_CHART_URL = "https://itpnews.com/fa/engine/ajax/get_prices.php"

PERSIAN_MONTHS = {
    "\u0641\u0631\u0648\u0631\u062f\u06cc\u0646": 1, "\u0627\u0631\u062f\u06cc\u0628\u0647\u0634\u062a": 2, "\u062e\u0631\u062f\u0627\u062f": 3,
    "\u062a\u06cc\u0631": 4, "\u0645\u0631\u062f\u0627\u062f": 5, "\u0634\u0647\u0631\u06cc\u0648\u0631": 6,
    "\u0645\u0647\u0631": 7, "\u0622\u0628\u0627\u0646": 8, "\u0622\u0630\u0631": 9,
    "\u062f\u06cc": 10, "\u0628\u0647\u0645\u0646": 11, "\u0627\u0633\u0641\u0646\u062f": 12,
}

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

CACHE_DIR = "data/cache"
EXTERNAL_CACHE_DIR = "data/external"
MODELS_DIR = "data/models"
FORECASTS_DIR = "data/forecasts"

MAX_CACHE_DAYS = 1825
DEFAULT_SMOOTH_WINDOW = 7
PRODUCTS_CACHE_FILE = "data/products.json"

for _d in (CACHE_DIR, EXTERNAL_CACHE_DIR, MODELS_DIR, FORECASTS_DIR):
    os.makedirs(_d, exist_ok=True)

EMERGENCY_FALLBACK = _nfc_keys({
    "گوساله کشتار": 508, "گوساله زنده": 506, "گوسفند کشتار": 507,
    "گوسفند زنده": 505, "مرغ کشتار": 67, "مرغ زنده": 42,
    "تخم مرغ": 19, "ذرت": 13, "کنجاله سویا": 1, "جو": 95, "گندم": 39,
})

IRAN_HOLIDAYS_JALALI = [
    # Nowruz (Farvardin 1-4) + Islamic Republic Day (12) + Sizdah Bedar (13)
    (1, 1), (1, 2), (1, 3), (1, 4), (1, 12), (1, 13),
    # Imam Khomeini Death (Khordad 14) + 15 Khordad Uprising (15)
    (3, 14), (3, 15),
    # Islamic Revolution (Bahman 22)
    (11, 22),
    # Oil Nationalization (Esfand 29)
    (12, 29),
]


    

RELATED_PRODUCTS = _nfc_keys({
    "گوساله کشتار": ["ذرت", "کنجاله سویا", "جو", "گوساله زنده"],
    "گوسفند کشتار": ["ذرت", "کنجاله سویا", "گندم", "گوسفند زنده"],
    "مرغ کشتار": ["ذرت", "کنجاله سویا", "تخم مرغ"],
    "گوساله زنده": ["ذرت", "کنجاله سویا"],
    "گوسفند زنده": ["ذرت", "گندم"],
    "مرغ زنده": ["ذرت", "کنجاله سویا"],
    "تخم مرغ": ["ذرت", "کنجاله سویا"],
})

# --------------------------------------------------------------------------
# Product Profiles
# --------------------------------------------------------------------------

EXTERNAL_FEATURE_REGISTRY = {
    "usd_irr_free": {"fetcher": "fetch_usd_irr_free", "freq": "D", "lag_days": 1},
    "gold_price": {"fetcher": "fetch_gold_price", "freq": "D", "lag_days": 1},
    "cpi_inflation": {"fetcher": "fetch_cpi_inflation", "freq": "M", "lag_days": 0},
}

PRODUCT_PROFILES = _nfc_keys({
    "_default": {"related_products": True, "external_features": []},
    "گوساله کشتار": {
        "related_products": True,
        "external_features": ["usd_irr_free", "gold_price", "cpi_inflation"],
    },
})


def get_product_profile(name: str) -> dict:
    """Return the profile dict for a product, falling back to _default."""
    key = _nfc(name)
    profile = PRODUCT_PROFILES.get(key, PRODUCT_PROFILES["_default"])
    return dict(profile)


def list_external_features(name: str) -> list:
    """Return list of external feature keys configured for this product."""
    return list(get_product_profile(name).get("external_features", []))


# --------------------------------------------------------------------------
# ExternalDataSource: unified external scraper with graceful degradation
# --------------------------------------------------------------------------

class ExternalDataSource:
    """
    Fetches macro/external series (USD/IRR, gold, CPI) with a fallback chain:
      1. primary source
      2. secondary source (if any)
      3. last successful cache
      4. None (caller must handle gracefully)

    This class NEVER raises: every public method catches all exceptions
    and returns None on total failure.
    """

    def __init__(self, cache_dir: str = EXTERNAL_CACHE_DIR):
        self.session = requests.Session()
        self.session.headers.update(_base_headers())
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    # -- cache helpers -----------------------------------------------------

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.csv")

    def _load_cache(self, key: str):
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            if df.empty:
                return None
            return df
        except Exception as exc:
            logger.warning("external cache load failed for %s: %s", key, exc)
            return None

    def _save_cache(self, key: str, df: pd.DataFrame) -> None:
        try:
            df.to_csv(self._cache_path(key), index=False)
        except Exception as exc:
            logger.warning("external cache save failed for %s: %s", key, exc)

    # -- public dispatcher ---------------------------------------------------

    def get(self, key: str, force_refresh: bool = False):
        """
        Return a DataFrame with columns ['date', 'value'] for the given
        external feature key, or None if unavailable via any route.
        """
        info = EXTERNAL_FEATURE_REGISTRY.get(key)
        if info is None:
            logger.warning("unknown external feature key: %s", key)
            return None

        if not force_refresh:
            cached = self._load_cache(key)
        else:
            cached = None

        fetch_fn = getattr(self, info["fetcher"], None)
        if fetch_fn is None:
            return cached

        try:
            df = fetch_fn(force_refresh=force_refresh)
        except Exception as exc:
            logger.warning("external fetch raised for %s: %s", key, exc)
            df = None

        if df is not None and not df.empty:
            self._save_cache(key, df)
            return df

        if cached is not None:
            logger.warning("external fetch failed for %s, using cache", key)
            return cached

        logger.warning("external feature %s unavailable (no cache, no fetch)", key)
        return None

    # -- individual fetchers -------------------------------------------------

    def fetch_usd_irr_free(self, force_refresh: bool = False):
        """USD/IRR free-market rate: tgju.org -> navasan.tech -> cache -> None."""
        df = self._fetch_tgju_series("price_dollar_rl")
        if df is not None and not df.empty:
            return df
        df = self._fetch_navasan_series("usd_sell")
        if df is not None and not df.empty:
            return df
        return None

    def fetch_gold_price(self, force_refresh: bool = False):
        """Gold price (18k, per gram): tgju.org -> cache -> None."""
        df = self._fetch_tgju_series("geram18")
        if df is not None and not df.empty:
            return df
        return None

    def fetch_cpi_inflation(self, force_refresh: bool = False):
        """Annual CPI index: World Bank API -> cache -> None."""
        try:
            _polite_wait()
            self.session.headers.update({"User-Agent": _rotate_ua()})
            url = "http://api.worldbank.org/v2/country/IR/indicator/FP.CPI.TOTL?format=json&per_page=500"
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
            if not isinstance(payload, list) or len(payload) < 2 or not payload[1]:
                return None
            rows = []
            for item in payload[1]:
                value = item.get("value")
                year = item.get("date")
                if value is not None and year:
                    rows.append({"date": pd.Timestamp(f"{year}-07-01"), "value": float(value)})
            if not rows:
                return None
            df = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date")
            df = df.drop_duplicates(subset="date", keep="last")
            return df
        except Exception as exc:
            logger.warning("cpi_inflation fetch failed: %s", exc)
            return None

    # -- shared scraping helpers ---------------------------------------------

    def _fetch_tgju_series(self, symbol: str):
        try:
            url = f"https://api.tgju.org/v1/market/indicator/summary-table-data/{symbol}"
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

    def _fetch_navasan_series(self, symbol: str):
        api_key = os.environ.get("NAVASAN_API_KEY", "")
        if not api_key:
            logger.warning("navasan skipped: NAVASAN_API_KEY not set")
            return None
        try:
            url = "http://api.navasan.tech/latest/"
            _polite_wait()
            self.session.headers.update({"User-Agent": _rotate_ua()})
            resp = self.session.get(url, timeout=15, params={"api_key": api_key})
            resp.raise_for_status()
            payload = resp.json()
            item = payload.get(symbol)
            if not item:
                return None
            value = float(item.get("value", 0))
            date = pd.to_datetime(item.get("date", datetime.now()), errors="coerce")
            if pd.isna(date) or value <= 0:
                return None
            return pd.DataFrame([{"date": date.normalize(), "value": value}])
        except Exception as exc:
            logger.warning("navasan fetch failed for %s: %s", symbol, exc)
            return None


external_data = ExternalDataSource()  # module-level singleton


# --------------------------------------------------------------------------
# Date utilities
# --------------------------------------------------------------------------

def jalali_to_gregorian(jy: int, jm: int, jd: int):
    """Convert Jalali (Persian) date to an ISO gregorian date string, or None."""
    try:
        d = jdatetime.date(int(jy), int(jm), int(jd)).togregorian()
        return d.isoformat()
    except Exception as exc:
        logger.warning("jalali_to_gregorian failed for %s-%s-%s: %s", jy, jm, jd, exc)
        return None


def gregorian_to_jalali_tuple(dt):
    """Convert a datetime/Timestamp to a (jy, jm, jd) tuple."""
    try:
        gy, gm, gd = dt.year, dt.month, dt.day
        g_days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if (gy % 4 == 0 and (gy % 100 != 0 or gy % 400 == 0)):
            g_days_in_month[1] = 29
        gy2 = gy - 1600
        gm2 = gm - 1
        gd2 = gd - 1
        g_day_no = 365 * gy2 + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400
        for i in range(gm2):
            g_day_no += g_days_in_month[i]
        g_day_no += gd2
        j_day_no = g_day_no - 79
        j_np = j_day_no // 12053
        j_day_no %= 12053
        jy = 979 + 33 * j_np + 4 * (j_day_no // 1461)
        j_day_no %= 1461
        if j_day_no >= 366:
            jy += (j_day_no - 1) // 365
            j_day_no = (j_day_no - 1) % 365
        j_days_in_month = [31, 31, 31, 31, 31, 31, 30, 30, 30, 30, 30, 29]
        jm = 0
        while jm < 11 and j_day_no >= j_days_in_month[jm]:
            j_day_no -= j_days_in_month[jm]
            jm += 1
        jm += 1
        jd = j_day_no + 1
        return (jy, jm, jd)
    except Exception as exc:
        logger.warning("gregorian_to_jalali_tuple failed: %s", exc)
        return (None, None, None)


PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"


def clean_number(text):
    """Parse a Persian/formatted number string into int or float. None on failure."""
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
    """Trim, NFC-normalize, and collapse whitespace in a text string."""
    if text is None:
        return ""
    s = _nfc(str(text))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_iran_holiday(date_val) -> int:
    try:
        dt = pd.Timestamp(date_val)
        
        # Thursday (3) & Friday (4) - Iranian weekend
        if dt.dayofweek in [3, 4]:
            return 1
        
        # Fixed solar holidays ±1 day
        jy, jm, jd = gregorian_to_jalali_tuple(dt)
        if jm is None:
            return 0
        for (hm, hd) in IRAN_HOLIDAYS_JALALI:
            if hm == jm and abs(hd - jd) <= 1:
                return 1
        return 0
    except Exception:
        return 0


def _add_holiday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add an is_holiday column derived from the date index/column."""
    df = df.copy()
    if "date" in df.columns:
        dates = df["date"]
    else:
        dates = df.index.to_series()
    df["is_holiday"] = dates.apply(_is_iran_holiday).values
    return df


# --------------------------------------------------------------------------
# Fetcher (ITPNews scraper)
# --------------------------------------------------------------------------

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
            self.session.headers.update({"User-Agent": _rotate_ua()})
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
        last_exc = None
        for attempt in range(retries):
            try:
                _polite_wait()
                headers["User-Agent"] = _rotate_ua()
                resp = self.session.post(url, data=payload, headers=headers, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        logger.warning("_post_json failed for %s after %d retries: %s", url, retries, last_exc)
        return None

    def discover_products(self):
        """Return (products_dict {name: id}, source_str)."""
        try:
            _polite_wait()
            self.session.headers.update({"User-Agent": _rotate_ua()})
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

        fallback = dict(EMERGENCY_FALLBACK)
        return fallback, "fallback"

    def _save_products_cache(self, products):
        try:
            with open(PRODUCTS_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(products, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("products cache save failed: %s", exc)

    def _load_products_cache(self):
        try:
            if os.path.exists(PRODUCTS_CACHE_FILE):
                with open(PRODUCTS_CACHE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as exc:
            logger.warning("products cache load failed: %s", exc)
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
            try:
                date_str = jalali_to_gregorian(int(year_str), month, int(day))
            except Exception:
                continue
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
        return df.dropna(subset=["date"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)

    def fetch_monthly(self, pid):
        data = self._post_json(MONTH_URL, pid)
        if not data:
            return pd.DataFrame(columns=["date", "price"])
        rows = []
        for item in data:
            y, m = item.get("y"), item.get("m")
            if not y or not m:
                continue
            try:
                date_str = f"{int(y)}-{int(m):02d}-01"
            except (TypeError, ValueError):
                continue
            rows.append({"date": date_str, "price": clean_number(item.get("avg"))})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "price"])

    def fetch_yearly(self, pid):
        data = self._post_json(LAST_YEAR_URL, pid)
        if not data:
            return pd.DataFrame(columns=["date", "price"])
        rows = []
        for item in data:
            y = item.get("y")
            if not y:
                continue
            try:
                date_str = f"{int(y)}-01-01"
            except (TypeError, ValueError):
                continue
            rows.append({"date": date_str, "price": clean_number(item.get("avg"))})
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "price"])

    def fetch_week_compare(self, pid):
        data = self._post_json(WEEK_URL, pid)
        return data or {}

    def fetch_year_compare(self, pid):
        data = self._post_json(PRICE_LAST_YEARS_URL, pid)
        return data or {}

    def fetch_detail(self, pid):
        try:
            url = f"{BASE_URL}/dashboard/market/{pid}/chart"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            title_tag = soup.find("h1")
            title = clean_text(title_tag.get_text()) if title_tag else ""
            return {"product_id": pid, "title": title}
        except Exception as exc:
            logger.warning("fetch_detail failed for %s: %s", pid, exc)
            return {"product_id": pid, "title": ""}


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

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
            df.sort_values("date").drop_duplicates("date").to_csv(self._path(pid), index=False)
        except Exception as exc:
            logger.warning("cache save failed for %s: %s", pid, exc)

    def update(self, pid, force=False):
        """Incrementally fetch new data since last cache; return (df, source_str)."""
        existing = self.load(pid)
        today = datetime.today()

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
            logger.warning("cache.update fetch failed for %s: %s", pid, exc)
            new_df = pd.DataFrame(columns=["date", "price"])

        if new_df.empty and existing.empty:
            return existing, "no_data"
        if new_df.empty:
            return existing, "cached"
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        # ═══ Keep only date and price ═══
        merged = merged[["date", "price"]]
        if len(merged) > MAX_CACHE_DAYS:
            merged = merged.tail(MAX_CACHE_DAYS).reset_index(drop=True)
        self.save(pid, merged)
        return merged, "updated"


# --------------------------------------------------------------------------
# Data processing
# --------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, target: str, freq: str = "D", iqr_k: float = 3.0) -> pd.DataFrame:
    """Clean and prepare price data. freq is accepted for API compat but not used
    (we intentionally keep actual trading days without forcing a frequency)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    col_map = {c: unicodedata.normalize("NFC", c) for c in df.columns if c != "date"}
    df = df.rename(columns=col_map)
    target = unicodedata.normalize("NFC", target)
    df = df.drop_duplicates("date").sort_values("date")
    
    # ═══ FIX: Don't force daily frequency - keep only actual trading days ═══
    df = df.reset_index(drop=True)
    
    for col in df.columns:
        if col == "date":
            continue
        # Unlimited forward/backward fill to keep all rows, then interpolate any remaining gaps
        df[col] = df[col].ffill().bfill().interpolate("linear", limit_direction="both")
        if col == target:
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - iqr_k * iqr
            upper = q3 + iqr_k * iqr
            df[col] = df[col].clip(lower, upper)

    df = _add_holiday_feature(df)
    return df


def merge_external_features(df: pd.DataFrame, product_name: str, freq: str = "D") -> pd.DataFrame:
    """
    Merge profile-driven external features into df. Fast path: if no external
    features configured, return df unchanged immediately.
    """
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
                logger.warning("skipping external feature %s: unavailable", key)
                continue
            ext_df = ext_df.copy()
            ext_df["date"] = pd.to_datetime(ext_df["date"])
            lag_days = info.get("lag_days", 0)
            if lag_days:
                ext_df["date"] = ext_df["date"] + pd.Timedelta(days=lag_days)

            col_name = f"ext_{key}"
            ext_df = ext_df.rename(columns={"value": col_name})[["date", col_name]]
            ext_df = ext_df.drop_duplicates(subset="date", keep="last").sort_values("date")

            out = out.sort_values("date")
            out = pd.merge_asof(out, ext_df, on="date", direction="backward")
            # Fill missing values completely: forward then backward
            out[col_name] = out[col_name].ffill().bfill()
        except Exception as exc:
            logger.warning("merge_external_features failed for %s: %s", key, exc)
            continue

    return out


def fetch_related_prices(df: pd.DataFrame, product_name: str, fetcher, cache, freq: str = "D") -> pd.DataFrame:
    """Fetch RELATED_PRODUCTS price data from ITPNews and merge as ext_feed_ columns."""
    related = RELATED_PRODUCTS.get(_nfc(product_name), [])
    if not related:
        return df.copy()
    products, _ = fetcher.discover_products()
    norm_products = {unicodedata.normalize("NFC", k): v for k, v in products.items()}
    result = df.copy()
    result["date"] = pd.to_datetime(result["date"])
    for rp in related:
        rp_id = norm_products.get(unicodedata.normalize("NFC", rp))
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
            result = pd.merge_asof(result.sort_values("date"), rp_df.sort_values("date"), on="date", direction="backward")
            # Fill missing values completely: forward then backward
            result[col] = result[col].ffill().bfill()
        except Exception as exc:
            logger.warning("fetch_related_prices failed for %s: %s", rp, exc)
            continue
    return result


def create_features(df, target_col, lags, windows, freq="D", other_cols=None,
                     external_cols=None, add_seasonal=True):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = []
    seasonal_cols = []

    if add_seasonal:
        month = df["date"].dt.month
        dow = df["date"].dt.dayofweek
        df["month"] = month
        df["quarter"] = df["date"].dt.quarter
        df["dayofweek"] = dow
        df["dayofyear"] = df["date"].dt.dayofyear
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        if "is_holiday" not in df.columns:
            df = _add_holiday_feature(df)
        seasonal_cols = ["month_sin", "month_cos", "dow_sin", "dow_cos",
                          "month", "quarter", "dayofweek", "dayofyear", "is_holiday"]
        feature_cols.extend([c for c in seasonal_cols if c in df.columns])

    shifted = df[target_col].shift(1)

    for lag in lags:
        if len(df) > lag:
            col = f"{target_col}_lag_{lag}"
            df[col] = df[target_col].shift(lag)
            feature_cols.append(col)

    for w in windows:
        if len(df) >= w:
            mean_col = f"{target_col}_ma_{w}"
            std_col = f"{target_col}_std_{w}"
            df[mean_col] = shifted.rolling(w).mean()
            df[std_col] = shifted.rolling(w).std()
            feature_cols.extend([mean_col, std_col])

    for period in windows:
        if len(df) > period:
            roc_col = f"{target_col}_roc_{period}"
            df[roc_col] = shifted.pct_change(period) * 100
            feature_cols.append(roc_col)

    for w in windows:
        if len(df) >= w:
            roll_min = shifted.rolling(w).min()
            roll_max = shifted.rolling(w).max()
            roll_range = roll_max - roll_min
            pos_col = f"{target_col}_position_{w}"
            df[pos_col] = ((shifted - roll_min) / roll_range.replace(0, np.nan)).clip(0, 1)
            roll_ma = shifted.rolling(w).mean()
            roll_std = shifted.rolling(w).std()
            cv_col = f"{target_col}_cv_{w}"
            df[cv_col] = roll_std / roll_ma.replace(0, np.nan)
            feature_cols.extend([pos_col, cv_col])

    if len(df) > 3:
        d1 = shifted.diff(1)
        accel_col = f"{target_col}_accel"
        df[accel_col] = d1.diff(1)
        feature_cols.append(accel_col)

    short_w = [w for w in windows if w <= 14]
    long_w = [w for w in windows if w >= 21]
    for sw in short_w:
        for lw in long_w:
            if sw < lw and len(df) >= lw:
                sma = shifted.rolling(sw).mean()
                lma = shifted.rolling(lw).mean()
                osc_col = f"{target_col}_osc_{sw}_{lw}"
                df[osc_col] = ((sma - lma) / lma.replace(0, np.nan))
                feature_cols.append(osc_col)

    if other_cols:
        for other in other_cols:
            if other == target_col or other == "date" or other not in df.columns:
                continue
            safe = re.sub(r'[^\w]', '_', other)
            other_shifted = df[other].shift(1)
            denom = other_shifted.replace(0, np.nan)
            ratio_col = f"{target_col}_ratio_{safe}"
            margin_col = f"{target_col}_margin_{safe}"
            df[ratio_col] = shifted / denom
            df[margin_col] = shifted - other_shifted
            feature_cols.extend([ratio_col, margin_col])
            for w in windows:
                if len(df) >= w:
                    corr_col = f"{target_col}_corr_{safe}_{w}"
                    df[corr_col] = shifted.rolling(w).corr(other_shifted)
                    feature_cols.append(corr_col)

    if external_cols:
        for ec in external_cols:
            if ec not in df.columns:
                continue
            feature_cols.append(ec)
            ec_shifted = df[ec].shift(1)
            small_windows = [w for w in windows if w <= 14]   # limit external windows
            for w in small_windows:
                if len(df) >= w:
                    mean_col = f"{ec}_ma_{w}"
                    std_col = f"{ec}_std_{w}"
                    df[mean_col] = ec_shifted.rolling(w).mean()
                    df[std_col] = ec_shifted.rolling(w).std()
                    feature_cols.extend([mean_col, std_col])

    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    df = df.replace([np.inf, -np.inf], np.nan)

    return df, feature_cols, seasonal_cols


def check_data_quality(df: pd.DataFrame, target_col: str) -> list:
    """Return a list of human-readable warning strings about data quality."""
    warnings = []
    if df is None or df.empty:
        warnings.append("Dataset is empty.")
        return warnings

    series = df[target_col]
    n = len(series)
    zeros = (series == 0).sum()
    nans = series.isna().sum()

    if zeros > 0:
        warnings.append(f"{zeros} zero prices ({zeros/n*100:.1f}%)")
    if nans > 0:
        warnings.append(f"{nans} missing values ({nans/n*100:.1f}%)")
    if n < 60:
        warnings.append(f"Only {n} rows (need 60+ for reliable training)")

    diffs = df["date"].diff().dropna()
    if len(diffs) > 0:
        max_gap = diffs.max()
        if max_gap > pd.Timedelta(days=14):
            warnings.append(f"Data gap of {max_gap.days} days detected")

    if len(series) > 10:
        q1, q99 = series.quantile(0.01), series.quantile(0.99)
        if q1 > 0:
            ratio = q99 / q1
            if ratio > 10:
                warnings.append(f"Large price range: {q1:,.0f} to {q99:,.0f} ({ratio:.0f}x)")

    return warnings


def detrend_series(train_df, target, freq="D", method="linear", smooth_w=None, split_date=None):
    """
    Fit a trend on the TRAIN portion only. split_date must be supplied by caller.
    Returns (coeffs, ref_date) or (None, None, smooth_w) for smooth_diff.
    """
    df = train_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if split_date is not None:
        df = df[df["date"] <= split_date]

    df = df.dropna(subset=[target])

    # These three branches don't need actual data rows to produce a
    # correctly-shaped result, so handle them before the empty check -
    # otherwise an empty train slice would silently collapse smooth_diff's
    # 3-tuple and none's Timestamp-marked 2-tuple into a generic (None, None).
    if method == "smooth_diff":
        w = smooth_w or DEFAULT_SMOOTH_WINDOW
        return None, None, w
    if method == "diff":
        return None, None
    if method == "none":
        return None, pd.Timestamp.today()

    if df.empty:
        # Not enough data to fit a trend: fall back to a flat (zero-slope)
        # trend anchored at "today" so callers still get a valid Timestamp
        # and a coeffs array of the right degree.
        ref_date = pd.Timestamp.today()
        coeffs = np.array([0.0, 0.0, 0.0]) if method == "quadratic" else np.array([0.0, 0.0])
        return coeffs, ref_date

    ref_date = df["date"].min()
    x = (df["date"] - ref_date).dt.days.values.astype(float)
    y = df[target].values.astype(float)

    if method == "quadratic":
        try:
            coeffs = np.polyfit(x, y, 2)
        except Exception:
            coeffs = np.array([0.0, 0.0, float(np.mean(y))])
        return coeffs, ref_date

    try:
        coeffs = np.polyfit(x, y, 1)
    except Exception:
        coeffs = np.array([0.0, float(np.mean(y))])
    return coeffs, ref_date


def apply_trend(dates, coeffs, ref_date, freq="D"):
    """Evaluate the fitted trend polynomial at the given dates."""
    if coeffs is None:
        return np.zeros(len(dates))
    dates = pd.to_datetime(pd.Series(dates))
    x = (dates - ref_date).dt.days.values.astype(float)
    return np.polyval(coeffs, x)


def select_features(df, feature_cols, split_date=None, max_corr=0.95):
    """Drop highly collinear features using correlation computed on train rows only."""
    if len(feature_cols) <= 5:
        return list(feature_cols)

    train_df = df
    if split_date is not None and "date" in df.columns:
        train_df = df[pd.to_datetime(df["date"]) <= split_date]
    if len(train_df) < 10:
        return list(feature_cols)

    cols = list(dict.fromkeys([c for c in feature_cols if c in train_df.columns]))
    corr = train_df[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        high_corr = upper.index[upper[col] > max_corr].tolist()
        to_drop.update(high_corr)

    return [c for c in cols if c not in to_drop]


def rank_features(model, feature_cols):
    """Return list of (feature_name, importance) sorted descending."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        importances = np.abs(coef)

    if importances is None or len(importances) != len(feature_cols):
        return [(f, 0.0) for f in feature_cols]

    return sorted(zip(feature_cols, importances), key=lambda p: p[1], reverse=True)


def remove_low_importance_features(df, feature_cols, model, threshold_quantile=0.05):
    """Drop features whose importance falls below the given quantile threshold."""
    ranked = rank_features(model, feature_cols)
    if not ranked or all(v == 0 for _, v in ranked):
        return list(feature_cols)
    values = np.array([v for _, v in ranked])
    threshold = np.quantile(values, threshold_quantile)
    kept = [f for f, v in ranked if v >= threshold]
    return kept if kept else list(feature_cols)


def auto_params(df, target_col, freq="D"):
    """Heuristically choose lags/windows based on dataset length and frequency."""
    n = len(df) if df is not None else 0
    series = df[target_col] if target_col in df.columns else df.iloc[:, 1]
    returns = series.pct_change().dropna()
    vol = float(returns.std()) if len(returns) > 5 else 0.02

    if freq == "D":
        if n > 1500:
            lags, windows = [1, 7, 14, 30], [7, 14, 30]        # cap windows at 30
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

    if vol > 0.05:
        windows = [w for w in windows if w >= 14]
    if vol < 0.005:
        lags, windows = lags[:4], windows[:4]

    return lags, windows


def generate_fallback(n_days=730, base=8_000_000) -> pd.DataFrame:
    """Generate a synthetic price series used when live/cached data is unavailable."""
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="D")
    rng = np.random.default_rng(42)
    trend = np.linspace(0, 0.15, n_days)
    seasonal = 0.03 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = rng.normal(0, 0.03, n_days)
    prices = base * (1 + trend + seasonal + np.cumsum(noise) * 0.1)
    prices = np.maximum(prices, base * 0.3)
    return pd.DataFrame({"date": dates, "price": prices.astype(int)})