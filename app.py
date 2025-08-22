# app.py
# ---------------------------
# Crypto Streamlit App + 7D Daily Forecast + Top Coin + Ensemble + Indicators
# + Alerts + Portfolio + Watchlist + Info + Backtest + Daily + Candles + FGI + Multi-lang
# Converted from Telegram Bot version to Streamlit UI (tanpa menghilangkan fitur)
# VERSI DIPERBARUI: UI input koin disederhanakan agar bisa menganalisis SEMUA koin dari CoinGecko.
# VERSI FINAL V2: Analisis "Top Coin" sekarang otomatis menganalisis Top N koin berdasarkan Market Cap.
# ---------------------------

# pip install streamlit requests nltk prophet statsmodels pandas scikit-learn matplotlib mplfinance transformers torch ta

import io
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Prophet optional
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False
    st.warning("Prophet tidak tersedia. Install dengan: pip install prophet")

# NLTK VADER
try:
    nltk.download("vader_lexicon", quiet=True)
    sia = SentimentIntensityAnalyzer()
except Exception as e:
    st.warning(f"NLTK VADER setup gagal: {e}")
    sia = None

# ---------------------------
# CONFIG (bisa diubah via sidebar)
# ---------------------------
VS_CURRENCY = "usd"
HISTORY_DAYS = 365               # lebih panjang agar model belajar tren
PRED_DAYS = 7
ALPHA = 0.10                     # pengaruh sentimen (eksponensial)

# Aman jika tidak ada secrets
DEFAULT_CRYPTOPANIC_KEY = (st.secrets.get("api", {}) if hasattr(st, "secrets") else {}).get("crypto_key", "")
DEFAULT_NEWSAPI_KEY = (st.secrets.get("api", {}) if hasattr(st, "secrets") else {}).get("news_key", "")

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RETRY_SLEEP = 1.5

STATE_PATH = "bot_state.json"    # file persistensi sederhana

logging.basicConfig(level=logging.INFO)

# ---------------------------
# SIMPLE CACHE (kurangi rate limit)
# ---------------------------
_cache = {
    "history": {},     # key: coin -> (ts, df)
    "news": {},        # key: coin -> (ts, list)
    "price": (0, {}),  # (ts, dict)
    "info": {},        # key: coin -> (ts, dict),
    "ohlc": {},        # key: (coin,days) -> (ts, df),
    "top_coins": (0, []) # (ts, list of coin ids)
}
CACHE_TTL_SEC = 300


def _get_cached(cache_key, subkey):
    bucket = _cache.get(cache_key, {})
    item = bucket.get(subkey)
    if not item:
        return None
    ts, data = item
    if time.time() - ts > CACHE_TTL_SEC:
        return None
    return data


def _set_cached(cache_key, subkey, data):
    if cache_key not in _cache:
        _cache[cache_key] = {}
    _cache[cache_key][subkey] = (time.time(), data)


# ---------------------------
# PERSISTENCE (alerts, portfolio, watchlist, language, keys)
# ---------------------------
STATE = {
    "alerts": {},      # single-user mode: "local" -> list of {"coin","op", "target"}
    "portfolio": {},   # "local" -> {coin: amount}
    "watchlist": {},   # "local" -> set/list of coins
    "lang": {"local": "id"},  # default id
    "keys": { "cryptopanic": DEFAULT_CRYPTOPANIC_KEY, "newsapi": DEFAULT_NEWSAPI_KEY}
}


def load_state():
    global STATE
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, "r") as f:
                data = json.load(f)
                STATE = {
                    "alerts": data.get("alerts", {}),
                    "portfolio": data.get("portfolio", {}),
                    "watchlist": {k: set(v) for k, v in data.get("watchlist", {}).items()},
                    "lang": data.get("lang", {"local": "id"}),
                    "keys": data.get(
                        "keys",
                        {
                            
                            "cryptopanic": DEFAULT_CRYPTOPANIC_KEY,
                            "newsapi": DEFAULT_NEWSAPI_KEY,
                        },
                    ),
                }
        except Exception as e:
            st.warning(f"Gagal load state: {e}")


def save_state():
    try:
        serializable = {
            "alerts": STATE.get("alerts", {}),
            "portfolio": STATE.get("portfolio", {}),
            "watchlist": {k: list(v) for k, v in STATE.get("watchlist", {}).items()},
            "lang": STATE.get("lang", {"local": "id"}),
            "keys": STATE.get(
                "keys",
                {
                    
                    "cryptopanic": DEFAULT_CRYPTOPANIC_KEY,
                    "newsapi": DEFAULT_NEWSAPI_KEY,
                },
            ),
        }
        with open(STATE_PATH, "w") as f:
            json.dump(serializable, f)
    except Exception as e:
        st.warning(f"Gagal save state: {e}")


load_state()


def get_lang():
    return STATE["lang"].get("local", "id")


def set_lang(l):
    STATE["lang"]["local"] = l
    save_state()


def get_keys():
    return STATE.get(
        "keys",
        { "cryptopanic": DEFAULT_CRYPTOPANIC_KEY, "newsapi": DEFAULT_NEWSAPI_KEY},
    )


# ---------------------------
# NETWORK HELPERS (retry)
# ---------------------------
def _get_json(url, params=None, timeout=REQUEST_TIMEOUT):
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                sleep_time = RETRY_SLEEP * (i + 1) * 2
                logging.warning(f"Rate limit exceeded. Waiting for {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(RETRY_SLEEP * (i + 1))
    logging.warning(f"GET json failed: {url} err={last_err}")
    return None


# ---------------------------
# DATA FETCH
# ---------------------------
def get_top_n_coins(n=100):
    ts, cached_list = _cache.get("top_coins", (0, []))
    if time.time() - ts <= CACHE_TTL_SEC and len(cached_list) >= n:
        return [c['id'] for c in cached_list[:n]]

    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": VS_CURRENCY,
        "order": "market_cap_desc",
        "per_page": min(n, 250), # Max per page is 250
        "page": 1,
        "sparkline": "false"
    }
    data = _get_json(url, params=params)
    if data:
        _cache["top_coins"] = (time.time(), data)
        return [c['id'] for c in data]
    return []


def get_current_prices(coins, with_change=False):
    ts, cached = _cache.get("price", (0, {}))
    need_change = with_change
    have_all = all(c in cached for c in coins) and (not need_change or all(f"{c}_24h" in cached for c in coins))
    if time.time() - ts <= CACHE_TTL_SEC and have_all:
        if not with_change:
            return {c: cached.get(c) for c in coins}
        else:
            return {c: (cached.get(c), cached.get(f"{c}_24h")) for c in coins}

    url = "https://api.coingecko.com/api/v3/simple/price"
    # Batasi jumlah koin per request untuk menghindari URL yang terlalu panjang
    coin_chunks = [coins[i:i + 100] for i in range(0, len(coins), 100)]
    full_result = {}

    for chunk in coin_chunks:
        params = {"ids": ",".join(chunk), "vs_currencies": VS_CURRENCY}
        if with_change:
            params["include_24hr_change"] = "true"
        resp = _get_json(url, params=params)
        
        chunk_result = {}
        if resp:
            for c in chunk:
                chunk_result[c] = (resp.get(c, {}) or {}).get(VS_CURRENCY)
                if with_change:
                    chunk_result[f"{c}_24h"] = (resp.get(c, {}) or {}).get(f"{VS_CURRENCY}_24h_change")
        full_result.update(chunk_result)

    _cache["price"] = (time.time(), {**cached, **full_result})
    if not with_change:
        return {c: full_result.get(c) for c in coins}
    else:
        return {c: (full_result.get(c), full_result.get(f"{c}_24h")) for c in coins}


def get_historical_prices(coin_id, days=HISTORY_DAYS):
    cached = _get_cached("history", coin_id)
    if cached is not None:
        return cached.copy()

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": VS_CURRENCY, "days": days, "interval": "daily"}
    rj = _get_json(url, params=params)
    if not rj:
        return pd.DataFrame(columns=["ds", "y"])
    prices = rj.get("prices", [])
    if not prices:
        return pd.DataFrame(columns=["ds", "y"])
    df = pd.DataFrame(prices, columns=["ts", "price"])
    df["ds"] = pd.to_datetime(df["ts"], unit="ms").dt.date
    df["y"] = df["price"].astype(float)
    df = df[["ds", "y"]].dropna().reset_index(drop=True)
    _set_cached("history", coin_id, df)
    return df.copy()


def get_ohlc(coin_id, days=90):
    cached = _get_cached("ohlc", (coin_id, days))
    if cached is not None:
        return cached.copy()
    if days not in [1, 7, 14, 30, 90, 180, 365]:
        days = 90
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
    params = {"vs_currency": VS_CURRENCY, "days": days}
    rj = _get_json(url, params=params)
    if not rj:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
    df = pd.DataFrame(rj, columns=["ts", "Open", "High", "Low", "Close"])
    df["Date"] = pd.to_datetime(df["ts"], unit="ms")
    df = df[["Date", "Open", "High", "Low", "Close"]].dropna().reset_index(drop=True)
    # Volume (best-effort)
    try:
        market_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        market_params = {"vs_currency": VS_CURRENCY, "days": days}
        market_data = _get_json(market_url, params=market_params)
        if market_data and "total_volumes" in market_data:
            volumes = market_data["total_volumes"]
            if len(volumes) == len(df):
                df["Volume"] = [v[1] for v in volumes]
    except Exception as e:
        logging.debug(f"Volume fetch fail {coin_id}: {e}")
    _set_cached("ohlc", (coin_id, days), df)
    return df.copy()


def get_coin_info(coin_id):
    cached = _get_cached("info", coin_id)
    if cached is not None:
        return cached
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }
    rj = _get_json(url, params=params)
    if not rj:
        return {}
    info = {
        "name": rj.get("name"),
        "symbol": rj.get("symbol", "").upper(),
        "rank": rj.get("market_cap_rank"),
        "market_cap": ((rj.get("market_data") or {}).get("market_cap") or {}).get(VS_CURRENCY),
        "volume": ((rj.get("market_data") or {}).get("total_volume") or {}).get(VS_CURRENCY),
        "supply": (rj.get("market_data") or {}).get("circulating_supply"),
        "homepage": (rj.get("links") or {}).get("homepage", [""])[0],
    }
    _set_cached("info", coin_id, info)
    return info


def fetch_cryptopanic(coin_slug, limit=10):
    keys = get_keys()
    CRYPTOPANIC_KEY = keys.get("cryptopanic", "")
    if not CRYPTOPANIC_KEY:
        return []
    cache_key = f"cp:{coin_slug}"
    cached = _get_cached("news", cache_key)
    if cached is not None:
        return cached[:limit]
    base = "https://cryptopanic.com/api/v1/posts/"
    params = {
        "auth_token": CRYPTOPANIC_KEY,
        "currencies": coin_slug,
        "public": "true",
        "kind": "news",
        "order": "published_at",
    }
    rj = _get_json(base, params=params)
    posts = (rj or {}).get("results", [])[:limit]
    items = [
        {
            "title": p.get("title", ""),
            "url": p.get("url", ""),
            "source": (p.get("source") or {}).get("title", "CryptoPanic"),
            "description": p.get("description", ""),
        }
        for p in posts
    ]
    _set_cached("news", cache_key, items)
    return items


def fetch_newsapi(coin_name, limit=10):
    keys = get_keys()
    NEWSAPI_KEY = keys.get("newsapi", "")
    if not NEWSAPI_KEY:
        return []
    cache_key = f"newsapi:{coin_name}"
    cached = _get_cached("news", cache_key)
    if cached is not None:
        return cached[:limit]
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{coin_name} cryptocurrency OR {coin_name}",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(limit, 100),
        "apiKey": NEWSAPI_KEY,
    }
    rj = _get_json(url, params=params)
    articles = (rj or {}).get("articles", [])[:limit]
    items = [
        {
            "title": a.get("title", ""),
            "url": a.get("url", ""),
            "source": (a.get("source") or {}).get("name", "NewsAPI"),
            "description": a.get("description", ""),
        }
        for a in articles
    ]
    _set_cached("news", cache_key, items)
    return items


def fetch_fgi():
    url = "https://api.alternative.me/fng/"
    rj = _get_json(url)
    try:
        v = rj["data"][0]
        return {
            "value": int(v["value"]),
            "classification": v["value_classification"],
            "timestamp": int(v["timestamp"]),
        }
    except Exception:
        return None


# ---------------------------
# SENTIMENT (FinBERT/CryptoBERT)
# ---------------------------
# Transformers optional load
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    @st.cache_resource(show_spinner=False)
    def load_sentiment_model():
        # ProsusAI/finbert widely available; Crypto-specific models can be swapped
        crypto_model_name = "ProsusAI/finbert"
        tokenizer_ = AutoTokenizer.from_pretrained(crypto_model_name)
        model_ = AutoModelForSequenceClassification.from_pretrained(crypto_model_name)
        return tokenizer_, model_

    tokenizer, model = load_sentiment_model()
except Exception as e:
    tokenizer, model = None, None
    st.warning(f"Model Transformers tidak tersedia ({e}). Fallback ke VADER jika ada.")


def compute_avg_sentiment(articles, coin="bitcoin"):
    """
    Analisis sentimen pakai FinBERT jika tersedia, kalau tidak fallback ke NLTK VADER.
    Output: skor sentimen rata-rata antara -1 (negatif) sampai +1 (positif).
    """
    scores = []

    for a in articles:
        text = ((a.get("title") or "") + " " + (a.get("description") or "")).strip()
        if not text:
            continue

        if tokenizer is not None and model is not None:
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                # Label: 0=negative, 1=neutral, 2=positive (FinBERT)
                score = probs[2].item() - probs[0].item()
                scores.append(score)
                continue
            except Exception:
                pass  # fallback ke VADER

        if sia is not None:
            try:
                score = float(sia.polarity_scores(text)["compound"])  # -1..+1
                scores.append(score)
            except Exception:
                continue

    if not scores:
        return 0.0, 0
    avg_sent = float(sum(scores) / len(scores))
    return avg_sent, len(scores)


# ---------------------------
# TECHNICAL INDICATORS
# ---------------------------
def compute_indicators(df):
    """ df: columns ds (date), y (price) """
    s = pd.Series(df["y"].values, dtype=float)
    out = pd.DataFrame({"ds": df["ds"], "close": s})

    # SMA / EMA
    out["sma7"] = s.rolling(7).mean()
    out["sma21"] = s.rolling(21).mean()
    out["ema12"] = s.ewm(span=12, adjust=False).mean()
    out["ema26"] = s.ewm(span=26, adjust=False).mean()

    # RSI 14
    delta = s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    out["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    macd = out["ema12"] - out["ema26"]
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd
    out["macd_signal"] = signal
    out["macd_hist"] = macd - signal

    # Bollinger %B
    sma20 = s.rolling(20).mean()
    std20 = s.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    out["bb_percent_b"] = (s - lower) / (upper - lower)

    return out


def current_actions_from_indicators(ind):
    last = ind.dropna().iloc[-1] if not ind.dropna().empty else None
    if last is None:
        return "⚖️ HOLD (data indikator belum cukup)"
    rsi = last["rsi14"]
    macd_cross_up = last["macd"] > last["macd_signal"]
    price_above_sma21 = last["close"] > last["sma21"]
    bb_b = last["bb_percent_b"]
    score = 0
    if not math.isnan(rsi):
        if rsi < 30:
            score += 2
        elif rsi > 70:
            score -= 2
    if macd_cross_up:
        score += 1
    if price_above_sma21:
        score += 1
    if not math.isnan(bb_b):
        if bb_b < 0:
            score += 1
        elif bb_b > 1:
            score -= 1
    if score >= 2:
        return "🚀 BUY (momentum & value mendukung)"
    if score <= -2:
        return "🔻 SELL (overbought/tekanan turun)"
    return "⚖️ HOLD"


# ---------------------------
# MODEL UTILITIES (ensemble)
# ---------------------------
def _mae(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.mean(np.abs(a - b)))


def _evaluate_model_on_holdout(model_fn, df_history, holdout_days=14):
    if len(df_history) < (holdout_days + 10):
        return None
    try:
        train = df_history.iloc[:-holdout_days].reset_index(drop=True)
        true = df_history.iloc[-holdout_days:]["y"].values
        pred_df = model_fn(train, days=holdout_days)
        if pred_df is None or "yhat" not in pred_df.columns:
            return None
        preds = np.array(pred_df["yhat"].values[:holdout_days])
        if len(preds) != len(true):
            return None
        return _mae(true, preds)
    except Exception as e:
        logging.debug(f"Model evaluation failed: {e}")
        return None


def forecast_with_prophet(df_history, days=PRED_DAYS):
    return _prophet_wrapper(df_history, days)


def forecast_with_arima(df_history, days=PRED_DAYS):
    return _arima_wrapper(df_history, days)


def forecast_with_linear(df_history, days=PRED_DAYS):
    return _linear_wrapper(df_history, days)


def _prophet_wrapper(train_df, days=PRED_DAYS):
    if not PROPHET_AVAILABLE:
        return None
    try:
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        tmp = train_df.copy()
        tmp["ds"] = pd.to_datetime(tmp["ds"])
        tmp.rename(columns={"y": "y"}, inplace=True)
        m.fit(tmp[["ds", "y"]])
        future = m.make_future_dataframe(periods=days)
        fc = m.predict(future)
        out = fc[["ds", "yhat"]].tail(days).copy()
        out["ds"] = out["ds"].dt.date
        return out
    except Exception:
        return None


def _linear_wrapper(train_df, days=PRED_DAYS):
    try:
        y = train_df["y"].values
        if len(y) < 5:
            return None
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        future_idx = np.arange(len(y), len(y) + days).reshape(-1, 1)
        preds = model.predict(future_idx)
        start = pd.to_datetime(train_df["ds"].iloc[-1])
        return pd.DataFrame({"ds": [(start + timedelta(i)).date() for i in range(1, days + 1)], "yhat": preds})
    except Exception:
        return None


def _arima_wrapper(train_df, days=PRED_DAYS):
    try:
        y = train_df["y"].values
        if len(y) < 20:
            return None
        fit = ARIMA(y, order=(2, 1, 2)).fit()
        fc = fit.forecast(steps=days)
        start = pd.to_datetime(train_df["ds"].iloc[-1])
        return pd.DataFrame({"ds": [(start + timedelta(i)).date() for i in range(1, days + 1)], "yhat": fc})
    except Exception:
        return None


def combine_models(df_history, days=PRED_DAYS):
    preds_values = {}
    available_models = []
    default_weights = {"prophet": 0.5, "arima": 0.3, "linear": 0.2}
    errors = {}
    try:
        if PROPHET_AVAILABLE:
            e = _evaluate_model_on_holdout(
                _prophet_wrapper, df_history, holdout_days=min(14, max(7, int(len(df_history) * 0.1)))
            )
            if e is not None:
                errors["prophet"] = e
        e = _evaluate_model_on_holdout(_arima_wrapper, df_history, holdout_days=min(14, max(7, int(len(df_history) * 0.1))))
        if e is not None:
            errors["arima"] = e
        e = _evaluate_model_on_holdout(_linear_wrapper, df_history, holdout_days=min(14, max(7, int(len(df_history) * 0.1))))
        if e is not None:
            errors["linear"] = e
    except Exception as ex:
        logging.debug(f"Error during evaluation: {ex}")

    if PROPHET_AVAILABLE:
        try:
            p_pred = forecast_with_prophet(df_history, days)
            if p_pred is not None:
                preds_values["prophet"] = p_pred["yhat"].values
                available_models.append("prophet")
        except Exception:
            pass
    try:
        a_pred = forecast_with_arima(df_history, days)
        if a_pred is not None:
            preds_values["arima"] = a_pred["yhat"].values
            available_models.append("arima")
    except Exception:
        pass
    try:
        l_pred = forecast_with_linear(df_history, days)
        if l_pred is not None:
            preds_values["linear"] = l_pred["yhat"].values
            available_models.append("linear")
    except Exception:
        pass

    if not available_models:
        return None

    weights = {}
    if errors and any(k in errors for k in available_models):
        inv = {}
        for m in available_models:
            if m in errors and errors[m] is not None and errors[m] > 0:
                inv[m] = 1.0 / (errors[m] + 1e-9)
        if inv:
            total = sum(inv.values())
            for m in available_models:
                if m in inv:
                    weights[m] = inv[m] / total
                else:
                    weights[m] = 1e-6
            s = sum(weights.values())
            if s <= 0:
                weights = {m: default_weights.get(m, 1.0 / len(available_models)) for m in available_models}
            else:
                for m in weights:
                    weights[m] = weights[m] / s
        else:
            weights = {m: default_weights.get(m, 1.0 / len(available_models)) for m in available_models}
    else:
        total_default = sum(default_weights.get(m, 0) for m in available_models)
        if total_default <= 0:
            weights = {m: 1.0 / len(available_models) for m in available_models}
        else:
            weights = {m: default_weights.get(m, 0) / total_default for m in available_models}

    try:
        stack = np.vstack([preds_values[m] for m in available_models])
        w = np.array([weights.get(m, 0.0) for m in available_models])
        w = w / (w.sum() if w.sum() > 0 else 1.0)
        preds_array = np.average(stack, axis=0, weights=w)
    except Exception as e:
        logging.debug(f"Combine models averaging failed: {e}")
        try:
            stack = np.vstack([preds_values[m] for m in available_models])
            preds_array = np.mean(stack, axis=0)
        except Exception as e2:
            logging.debug(f"Fallback averaging also failed: {e2}")
            return None

    start = pd.to_datetime(df_history["ds"].iloc[-1])
    return pd.DataFrame({"ds": [(start + timedelta(i)).date() for i in range(1, days + 1)], "yhat": preds_array})


def apply_sentiment_adjustment(preds_df, sentiment_score, alpha=ALPHA):
    adj = float(np.exp(alpha * sentiment_score))
    df = preds_df.copy()
    df["yhat_adj"] = df["yhat"] * adj
    df["adj_factor"] = adj
    return df


# ---------------------------
# RECOMMENDATION / DECISION
# ---------------------------
def _indicator_score_from_ind(ind):
    last = ind.dropna().iloc[-1] if not ind.dropna().empty else None
    if last is None:
        return 0.0
    score = 0.0
    try:
        rsi = last["rsi14"]
        macd_cross_up = last["macd"] > last["macd_signal"]
        price_above_sma21 = last["close"] > last["sma21"]
        bb_b = last["bb_percent_b"]

        if not math.isnan(rsi):
            if rsi < 30:
                score += 1.5
            elif rsi > 70:
                score -= 1.5
        if macd_cross_up:
            score += 1.0
        if price_above_sma21:
            score += 1.0
        if not math.isnan(bb_b):
            if bb_b < 0:
                score += 0.8
            elif bb_b > 1:
                score -= 0.8
    except Exception:
        pass
    return score


def determine_recommendation(ind, adj_df, last_price, sentiment_score):
    try:
        future_price = float(adj_df["yhat_adj"].iloc[-1])
        pct_change = (future_price - last_price) / last_price * 100.0
    except Exception:
        pct_change = 0.0
    pct_score = np.tanh(pct_change / 5.0) * 2.0
    ind_score = _indicator_score_from_ind(ind)
    sent_score = np.tanh(sentiment_score) * 1.5
    combined = 0.5 * pct_score + 0.35 * ind_score + 0.15 * sent_score
    if combined >= 1.5:
        label = "🚀 STRONG BUY"
    elif combined >= 0.6:
        label = "🚀 BUY"
    elif combined <= -1.5:
        label = "🔻 STRONG SELL"
    elif combined <= -0.6:
        label = "🔻 SELL"
    else:
        label = "⚖️ HOLD"
    reason = f"score={combined:.2f} (pct={pct_change:+.2f}%, ind={ind_score:.2f}, sent={sent_score:.2f})"
    return label, reason, pct_change


# ---------------------------
# PLOT
# ---------------------------
def plot_forecast(hist, raw_preds, adj_df, coin):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(pd.to_datetime(hist["ds"]), hist["y"], label="Historical", marker="o", linewidth=2, color="#2E8B57", markersize=3)
    ax.plot(pd.to_datetime(raw_preds["ds"]), raw_preds["yhat"], label="Raw forecast", marker="x", linestyle="--", linewidth=2, color="#FF6347", markersize=4)
    ax.plot(pd.to_datetime(adj_df["ds"]), adj_df["yhat_adj"], label="Sentiment-adjusted forecast", marker="^", linestyle="-", linewidth=2.5, color="#4169E1", markersize=4)
    ax.axvline(pd.to_datetime(hist["ds"].iloc[-1]), color="gray", linestyle=":", alpha=0.7, linewidth=1)
    ax.set_title(f"{coin.upper()} Price Forecast & Analysis", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price (USD)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.2f}"))
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_candles(ohlc_df, coin):
    df = ohlc_df.copy()
    df.set_index("Date", inplace=True)
    has_volume = "Volume" in df.columns
    style = mpf.make_mpf_style(
        base_mpl_style="seaborn-v0_8",
        marketcolors=mpf.make_marketcolors(
            up="g",
            down="r",
            edge="inherit",
            wick={"up": "green", "down": "red"},
            volume="in",
        ),
    )
    fig, axlist = mpf.plot(
        df,
        type="candle",
        mav=(20, 50),
        volume=has_volume,
        returnfig=True,
        figscale=1.0,
        title=f"{coin.upper()} – Candlestick (90D)",
        style=style,
        tight_layout=True,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf


# ---------------------------
# BACKTEST
# ---------------------------
def backtest_rsi(df, buy_th=30, sell_th=70):
    ind = compute_indicators(df)
    prices = ind["close"].values
    rsi = ind["rsi14"].values
    pos = 0
    entry = 0.0
    pnl = 0.0
    trades = 0
    for i in range(1, len(prices)):
        if np.isnan(rsi[i]):
            continue
        if pos == 0 and rsi[i] < buy_th:
            pos = 1
            entry = prices[i]
            trades += 1
        elif pos == 1 and rsi[i] > sell_th:
            pnl += (prices[i] - entry) / entry
            pos = 0
    if pos == 1:
        pnl += (prices[-1] - entry) / entry
    return pnl * 100, trades


def backtest_macd_cross(df):
    ind = compute_indicators(df).dropna()
    prices = ind["close"].values
    macd = ind["macd"].values
    signal = ind["macd_signal"].values
    pos = 0
    entry = 0.0
    pnl = 0.0
    trades = 0
    for i in range(1, len(prices)):
        if pos == 0 and macd[i - 1] <= signal[i - 1] and macd[i] > signal[i]:
            pos = 1
            entry = prices[i]
            trades += 1
        elif pos == 1 and macd[i - 1] >= signal[i - 1] and macd[i] < signal[i]:
            pnl += (prices[i] - entry) / entry
            pos = 0
    if pos == 1:
        pnl += (prices[-1] - entry) / entry
    return pnl * 100, trades


# ---------------------------
# LANG (ID/EN minimal)
# ---------------------------
TEXTS = {
    "id": {
        "welcome": "🚀 Selamat datang di CryptoApp (Streamlit)!\nPilih menu di sidebar.",
        "alert_set": "🔔 Alert ditambahkan: {coin} {op} {price}",
        "alert_list_empty": "Tidak ada alert aktif.",
        "alert_list": "🔔 Daftar alert kamu:",
        "alert_removed": "✅ Alert dihapus.",
        "invalid_cmd": "Format perintah kurang tepat.",
        "portfolio_empty": "Portfoliomu masih kosong. Tambah di tab Portofolio.",
        "portfolio": "📊 Portofolio (estimasi USD):\n{rows}\nTotal nilai: ${total:,.2f}",
        "added": "✅ Ditambahkan: {amount} {coin}.",
        "removed": "✅ Dikurangi: {amount} {coin}.",
        "no_amount": "Jumlah tidak valid.",
        "watch_added": "📌 {coin} ditambahkan ke watchlist.",
        "watch_removed": "🗑️ {coin} dihapus dari watchlist.",
        "watch_empty": "Watchlist kosong.",
        "watch_list": "📌 Watchlist kamu: {coins}",
        "info": "ℹ️ {name} ({symbol})\nRank: {rank}\nMarket Cap: ${mc:,.0f}\nVolume 24h: ${vol:,.0f}\nSupply Beredar: {sup:,.0f}\nWebsite: {url}",
        "info_na": "Maaf, info koin tidak tersedia.",
        "backtest": "🧪 Backtest {coin} – strategi {strat}\nReturn total: {ret:.2f}%\nJumlah transaksi: {trades}",
        "daily": "📰 Ringkasan Harian:\n{lines}",
        "fgi": "😨 Fear & Greed Index: {value} ({cls})",
        "lang_set": "Bahasa diatur ke Indonesia.",
        "alert_trigger": "🔔 Alert {coin} terpenuhi: harga ${price:,.2f} {relation} {target:,.2f}",
        "chart_fail": "Gagal membuat chart.",
    },
    "en": {
        "welcome": "🚀 Welcome to CryptoApp (Streamlit)!\nUse the sidebar.",
        "alert_set": "🔔 Alert added: {coin} {op} {price}",
        "alert_list_empty": "No active alerts.",
        "alert_list": "🔔 Your alerts:",
        "alert_removed": "✅ Alert removed.",
        "invalid_cmd": "Invalid command format.",
        "portfolio_empty": "Your portfolio is empty. Add in Portfolio tab.",
        "portfolio": "📊 Portfolio (USD est.):\n{rows}\nTotal value: ${total:,.2f}",
        "added": "✅ Added: {amount} {coin}.",
        "removed": "✅ Removed: {amount} {coin}.",
        "no_amount": "Invalid amount.",
        "watch_added": "📌 {coin} added to watchlist.",
        "watch_removed": "🗑️ {coin} removed from watchlist.",
        "watch_empty": "Watchlist is empty.",
        "watch_list": "📌 Your watchlist: {coins}",
        "info": "ℹ️ {name} ({symbol})\nRank: {rank}\nMarket Cap: ${mc:,.0f}\n24h Volume: ${vol:,.0f}\nCirculating Supply: {sup:,.0f}\nWebsite: {url}",
        "info_na": "Sorry, coin info unavailable.",
        "backtest": "🧪 Backtest {coin} – {strat} strategy\nTotal return: {ret:.2f}%\nTrades: {trades}",
        "daily": "📰 Daily Summary:\n{lines}",
        "fgi": "😨 Fear & Greed Index: {value} ({cls})",
        "lang_set": "Language set to English.",
        "alert_trigger": "🔔 Alert {coin} met: price ${price:,.2f} {relation} {target:,.2f}",
        "chart_fail": "Failed to render chart.",
    },
}


def T(key, **kw):
    lang = get_lang()
    return TEXTS.get(lang, TEXTS["id"]).get(key, key).format(**kw)


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="Crypto Forecast App", page_icon="🪙", layout="wide")

# Sidebar: settings & keys
with st.sidebar:
    st.header("⚙️ Pengaturan")
    lang = st.radio("Language / Bahasa", options=["id", "en"], index=0 if get_lang() == "id" else 1)
    if lang != get_lang():
        set_lang(lang)
        st.success(T("lang_set"))

    st.markdown("---")
    st.subheader("🔑 API Keys")
    keys = get_keys()
    new_cp = st.text_input("CryptoPanic Key", value=keys.get("cryptopanic", ""), type="password")
    new_news = st.text_input("NewsAPI Key", value=keys.get("newsapi", ""), type="password")
    # Keep bot token for completeness (not used)
    new_bot = st.text_input("Telegram Bot Token (not used here)", value=keys.get("bot_token", ""), type="password")

    if st.button("Simpan Keys"):
        STATE["keys"]["cryptopanic"] = new_cp.strip()
        STATE["keys"]["newsapi"] = new_news.strip()
        STATE["keys"]["bot_token"] = new_bot.strip()
        save_state()
        st.success("Keys disimpan.")

st.title("🪙 Crypto Forecast & Sentiment (Streamlit)")
st.info(T("welcome"))

tab_forecast, tab_top, tab_price, tab_daily, tab_chart, tab_info, tab_backtest, tab_port, tab_watch, tab_fgi = st.tabs(
    ["📊 Forecast 7D", "🚀 Top Coin", "💰 Price", "📰 Daily", "📈 Candles", "ℹ️ Info", "🧪 Backtest", "📂 Portfolio", "📌 Watchlist", "😨 FGI"]
)


# ---------------------------
# FORECAST
# ---------------------------
with tab_forecast:
    st.subheader("Prediksi 7 Hari + Sentimen + Aksi")

    # --- Pilihan coin ---
    coin_sel = st.text_input("Masukkan ID Coin (e.g., bitcoin, ethereum, solana)", value="bitcoin", key="forecast_coin").lower().strip()

    # --- Forecast ---
    if st.button("Run Forecast", key="run_forecast"):
        if not coin_sel:
            st.warning("Silakan masukkan ID coin untuk dianalisis.")
        else:
            try:
                with st.spinner(f"Menganalisis {coin_sel.upper()}..."):
                    hist = get_historical_prices(coin_sel)

                    if hist.empty or len(hist) < 30:
                        st.error(f"Tidak cukup data historis untuk {coin_sel.upper()}. Pastikan ID coin benar.")
                    else:
                        # Hitung indikator teknikal
                        ind = compute_indicators(hist)

                        # Forecast model ensemble
                        raw_preds = combine_models(hist)
                        if raw_preds is None:
                            st.error("Gagal membuat prediksi.")
                        else:
                            # Ambil berita + sentimen
                            cp = fetch_cryptopanic(coin_sel)
                            na = fetch_newsapi(coin_sel)
                            all_articles = cp + na

                            sentiment_score, n_articles = compute_avg_sentiment(all_articles)
                            adj_df = apply_sentiment_adjustment(raw_preds, sentiment_score)
                            last_price = float(hist["y"].iloc[-1])

                            # Tentukan rekomendasi
                            rec_label, rec_reason, pct_change = determine_recommendation(
                                ind, adj_df, last_price, sentiment_score
                            )

                            # Tampilkan info utama
                            st.markdown(f"**Harga sekarang**: ${last_price:,.2f}")
                            st.markdown(
                                f"**Sentimen berita**: {sentiment_score:+.2f} (n={n_articles}) "
                                f"→ adj × **{adj_df['adj_factor'].iloc[0]:.3f}**"
                            )
                            st.markdown(f"### Rekomendasi Akhir: {rec_label}")
                            st.caption(rec_reason)

                            # Tabel prediksi
                            show = adj_df.copy()
                            show["pct_vs_now_%"] = (show["yhat_adj"] - last_price) / last_price * 100
                            st.dataframe(show)

                            # Snapshot indikator
                            snap = ind.dropna().iloc[-1] if not ind.dropna().empty else None
                            if snap is not None:
                                macd_state = "Bullish" if snap["macd"] > snap["macd_signal"] else "Bearish"
                                trend_state = "Uptrend" if snap["close"] > snap["sma21"] else "Downtrend"
                                rsi_note = "Oversold" if snap["rsi14"] < 30 else ("Overbought" if snap["rsi14"] > 70 else "Neutral")
                                st.markdown(
                                    f"**Snapshot**: RSI14={snap['rsi14']:.1f} ({rsi_note}) "
                                    f"| MACD={macd_state} | Trend(SMA21)={trend_state}"
                                )

                            # Chart forecast
                            try:
                                buf = plot_forecast(hist, raw_preds, adj_df, coin_sel)
                                st.image(buf, caption=f"{coin_sel.upper()} Forecast", use_container_width=True)
                            except Exception as e:
                                st.warning(f"Chart gagal: {e}")

                            # Headline berita
                            if all_articles:
                                st.markdown("#### Headline Terbaru")
                                for a in all_articles[:10]:
                                    ttl = a.get("title") or "(no title)"
                                    src = a.get("source") or "-"
                                    url = a.get("url") or ""
                                    st.write(f"- **{src}**: [{ttl}]({url})")

            except Exception as e:
                st.error(f"Gagal memproses forecast untuk {coin_sel}: {e}")


# ---------------------------
# TOP COIN
# ---------------------------
with tab_top:
    st.subheader("Rekomendasi Koin Terbaik (Berdasarkan Market Cap)")
    st.info("Aplikasi ini akan menganalisis koin-koin teratas dari CoinGecko untuk menemukan potensi terbaik.")

    num_coins_to_analyze = st.slider(
        "Pilih jumlah koin teratas untuk dianalisis",
        min_value=10,
        max_value=100,
        value=20,
        step=10
    )

    if st.button("Cari Top Coin", key="run_top"):
        with st.spinner(f"Mengambil daftar {num_coins_to_analyze} koin teratas..."):
            coins_to_analyze = get_top_n_coins(num_coins_to_analyze)
        
        if not coins_to_analyze:
            st.error("Gagal mengambil daftar koin teratas dari CoinGecko. Coba lagi nanti.")
        else:
            with st.spinner(f"Menganalisis {len(coins_to_analyze)} koin... Ini mungkin butuh waktu."):
                results = []
                progress_bar = st.progress(0, text=f"Menganalisis {len(coins_to_analyze)} koin...")
                
                for i, coin in enumerate(coins_to_analyze):
                    progress_bar.progress((i + 1) / len(coins_to_analyze), text=f"Menganalisis {coin.upper()} ({i+1}/{len(coins_to_analyze)})...")
                    try:
                        hist = get_historical_prices(coin)
                        if hist.empty or len(hist) < 30:
                            logging.warning(f"Skipping {coin}, not enough data.")
                            continue
                        
                        ind = compute_indicators(hist)
                        raw_preds = combine_models(hist)
                        if raw_preds is None:
                            logging.warning(f"Skipping {coin}, forecast failed.")
                            continue
                        
                        cp = fetch_cryptopanic(coin)
                        na = fetch_newsapi(coin)
                        all_articles = cp + na
                        sentiment_score, _ = compute_avg_sentiment(all_articles)
                        adj_df = apply_sentiment_adjustment(raw_preds, sentiment_score)
                        
                        last_price = float(hist["y"].iloc[-1])
                        day7_price = float(adj_df["yhat_adj"].iloc[-1])
                        pct_change = (day7_price - last_price) / last_price * 100

                        last_ind = ind.dropna().iloc[-1] if not ind.dropna().empty else None
                        mom = 0.0
                        if last_ind is not None:
                            if last_ind["close"] > last_ind["sma21"]:
                                mom += 0.5
                            if last_ind["macd"] > last_ind["macd_signal"]:
                                mom += 0.5
                            if not math.isnan(last_ind["rsi14"]):
                                if last_ind["rsi14"] < 30:
                                    mom += 0.4
                                elif last_ind["rsi14"] > 70:
                                    mom -= 0.4

                        score = pct_change * (1 + max(min(sentiment_score, 0.5), -0.5)) + (mom * 2)
                        results.append(
                            {
                                "coin": coin,
                                "last_price": last_price,
                                "pred_7d": day7_price,
                                "pct_change_%": pct_change,
                                "sentiment": sentiment_score,
                                "momentum": mom,
                                "score": score,
                            }
                        )
                    except Exception as e:
                        logging.warning(f"Gagal analisis {coin}: {e}")
                
                progress_bar.empty()

            if not results:
                st.error("Tidak ada koin yang bisa dianalisis. Mungkin ada masalah dengan API atau data historis tidak cukup.")
            else:
                dfres = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
                st.dataframe(dfres, use_container_width=True)
                top = dfres.iloc[0]
                st.success(
                    f"**TOP BUY**: {top['coin'].upper()} — harga sekarang ${top['last_price']:.2f}, "
                    f"pred 7D ${top['pred_7d']:.2f} ({top['pct_change_%']:+.2f}%), "
                    f"sentimen {top['sentiment']:+.2f}, momentum {top['momentum']:.2f}, skor {top['score']:.2f}"
                )


# ---------------------------
# PRICE
# ---------------------------
with tab_price:
    st.subheader("Harga Koin & Perubahan 24h")
    coins_price_input = st.text_input("Masukkan ID coin, dipisahkan koma", "bitcoin, ethereum, dogecoin, solana")
    
    if st.button("Cek Harga", key="check_price"):
        coins_for_price = [coin.strip().lower() for coin in coins_price_input.split(',') if coin.strip()]
        if not coins_for_price:
            st.warning("Masukkan setidaknya satu ID coin untuk menampilkan harga.")
        else:
            with st.spinner("Mengambil harga..."):
                prices = get_current_prices(coins_for_price, with_change=True)
                table = []
                for c in coins_for_price:
                    p, chg = prices.get(c, (None, None))
                    table.append({"coin": c.upper(), "price": p, "change_24h_%": chg})
                st.dataframe(pd.DataFrame(table), use_container_width=True)


# ---------------------------
# DAILY
# ---------------------------
with tab_daily:
    st.subheader("Ringkasan Harian")
    st.info("Menampilkan data untuk koin di Watchlist Anda.")
    wl = list(STATE["watchlist"].get("local", []))
    target_coins = wl
    
    if not target_coins:
        st.warning("Watchlist Anda kosong. Tambahkan koin di tab 'Watchlist' untuk melihat ringkasan harian.")
    else:
        data = get_current_prices(target_coins, with_change=True)
        rows = []
        for c, (p, chg) in data.items():
            if p is None:
                continue
            rows.append({"Coin": c.upper(), "Price": p, "24h %": chg})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.write("Tidak dapat mengambil data harga untuk koin yang dipilih.")


# ---------------------------
# CANDLES
# ---------------------------
with tab_chart:
    st.subheader("Candlestick 90D")
    coin_c = st.text_input("Masukkan ID Coin", value="bitcoin", key="chart_coin").lower().strip()
    if st.button("Render Chart"):
        if not coin_c:
            st.warning("Masukkan ID coin.")
        else:
            with st.spinner(f"Membuat chart untuk {coin_c.upper()}..."):
                ohlc = get_ohlc(coin_c.lower(), days=90)
                if ohlc.empty:
                    st.error(T("chart_fail") + f" untuk {coin_c.upper()}")
                else:
                    try:
                        buf = plot_candles(ohlc, coin_c)
                        st.image(buf, caption=f"{coin_c.upper()} Candlestick", use_container_width=True)
                    except Exception as e:
                        st.warning(f"Chart error: {e}")
                        st.error(T("chart_fail"))


# ---------------------------
# INFO
# ---------------------------
with tab_info:
    st.subheader("Info Coin")
    coin_info_id = st.text_input("Masukkan ID Coin", value="bitcoin", key="info_coin").lower().strip()
    if st.button("Ambil Info Coin"):
        if not coin_info_id:
            st.warning("Masukkan ID coin.")
        else:
            info = get_coin_info(coin_info_id)
            if not info or not info.get("name"):
                st.error(T("info_na"))
            else:
                st.success(
                    T(
                        "info",
                        name=info["name"],
                        symbol=info["symbol"],
                        rank=info.get("rank"),
                        mc=(info.get("market_cap") or 0),
                        vol=(info.get("volume") or 0),
                        sup=(info.get("supply") or 0),
                        url=(info.get("homepage") or ""),
                    )
                )


# ---------------------------
# BACKTEST
# ---------------------------
with tab_backtest:
    st.subheader("Backtest")
    coin_bt = st.text_input("Masukkan ID Coin", value="bitcoin", key="bt_coin").lower().strip()
    strat = st.selectbox("Strategi", ["rsi", "macd"])
    if st.button("Run Backtest"):
        if not coin_bt:
            st.warning("Masukkan ID coin.")
        else:
            hist = get_historical_prices(coin_bt, days=HISTORY_DAYS)
            if hist.empty or len(hist) < 60:
                st.error("Data historis kurang untuk backtest.")
            else:
                if strat == "rsi":
                    ret, trades = backtest_rsi(hist)
                else:
                    ret, trades = backtest_macd_cross(hist)
                st.success(T("backtest", coin=coin_bt.upper(), strat=strat.upper(), ret=ret, trades=trades))


# ---------------------------
# PORTFOLIO
# ---------------------------
with tab_port:
    st.subheader("Portofolio")
    pos = STATE["portfolio"].get("local", {})
    if not pos:
        st.info(T("portfolio_empty"))
    else:
        coins_in_pos = list(pos.keys())
        prices = get_current_prices(coins_in_pos)
        rows = []
        total = 0.0
        for c, amt in pos.items():
            price = prices.get(c) or 0.0
            val = (amt or 0.0) * (price or 0.0)
            total += val
            rows.append([c.upper(), amt, price, val])
        dfp = pd.DataFrame(rows, columns=["Coin", "Amount", "Price", "Value"])
        st.dataframe(dfp, use_container_width=True)
        st.info(f"Total: ${total:,.2f}")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        coin_add = st.text_input("Tambah Coin (id coingecko)", value="bitcoin")
        amt_add = st.number_input("Jumlah", value=0.0, min_value=0.0, step=0.0001, format="%.6f")
        if st.button("Tambah ke Portofolio"):
            STATE["portfolio"].setdefault("local", {})
            STATE["portfolio"]["local"][coin_add.lower()] = STATE["portfolio"]["local"].get(coin_add.lower(), 0.0) + float(amt_add)
            save_state()
            st.success(T("added", amount=amt_add, coin=coin_add.upper()))
            st.rerun()
    with col2:
        coin_rm = st.text_input("Kurangi Coin (id coingecko)", value="bitcoin")
        amt_rm = st.number_input("Jumlah (kurangi)", value=0.0, min_value=0.0, step=0.0001, format="%.6f", key="rm_amt")
        if st.button("Kurangi dari Portofolio"):
            pos = STATE["portfolio"].get("local", {})
            cur = pos.get(coin_rm.lower(), 0.0)
            newv = max(0.0, cur - float(amt_rm))
            pos[coin_rm.lower()] = newv
            save_state()
            st.success(T("removed", amount=amt_rm, coin=coin_rm.upper()))
            st.rerun()


# ---------------------------
# WATCHLIST
# ---------------------------
with tab_watch:
    st.subheader("Watchlist")
    wl = STATE["watchlist"].get("local", set())
    if wl:
        st.write("Daftar:", ", ".join(sorted(list(wl))))
    else:
        st.info(T("watch_empty"))
    st.markdown("---")
    colw1, colw2 = st.columns(2)
    with colw1:
        w_add = st.text_input("Tambah Coin ke Watchlist", value="bitcoin")
        if st.button("Tambah"):
            STATE["watchlist"].setdefault("local", set())
            wl = STATE["watchlist"]["local"]
            wl.add(w_add.lower())
            save_state()
            st.success(T("watch_added", coin=w_add.upper()))
            st.rerun()
    with colw2:
        w_rm = st.text_input("Hapus Coin dari Watchlist", value="bitcoin", key="w_rm")
        if st.button("Hapus"):
            wl = STATE["watchlist"].get("local", set())
            if w_rm.lower() in wl:
                wl.remove(w_rm.lower())
                save_state()
                st.success(T("watch_removed", coin=w_rm.upper()))
                st.rerun()


# ---------------------------
# FGI
# ---------------------------
with tab_fgi:
    st.subheader("Fear & Greed Index")
    if st.button("Ambil FGI"):
        fgi = fetch_fgi()
        if not fgi:
            st.error("Gagal mengambil Fear & Greed Index.")
        else:
            ts = datetime.utcfromtimestamp(fgi["timestamp"]).strftime("%Y-%m-%d %H:%M")
            st.success(T("fgi", value=fgi["value"], cls=fgi["classification"]) + f"\n⏱ {ts} UTC")


# ---------------------------
# ALERTS (manual check)
# ---------------------------
st.markdown("---")
st.header("🔔 Alerts (Manual Check in Streamlit)")
colA, colB, colC = st.columns(3)
with colA:
    a_coin = st.text_input("Coin", value="bitcoin", key="alert_coin")
with colB:
    a_op = st.selectbox("Operator", [">", "<", ">=", "<="], index=0)
with colC:
    a_price = st.number_input("Target Price (USD)", min_value=0.0, value=50000.0, step=100.0)
if st.button("Tambah Alert"):
    STATE["alerts"].setdefault("local", [])
    STATE["alerts"]["local"].append({"coin": a_coin.lower(), "op": a_op, "target": float(a_price)})
    save_state()
    st.success(T("alert_set", coin=a_coin.upper(), op=a_op, price=a_price))
    st.rerun()

alerts = STATE["alerts"].get("local", [])
if alerts:
    st.write(T("alert_list"))
    alert_df_data = []
    for i, a in enumerate(alerts, 1):
        alert_df_data.append({"index": i, "coin": a['coin'].upper(), "condition": f"{a['op']} {a['target']}"})
    st.dataframe(pd.DataFrame(alert_df_data), use_container_width=True)

else:
    st.info(T("alert_list_empty"))

colD, colE = st.columns(2)
with colD:
    if st.button("Cek Alert Sekarang"):
        if not alerts:
            st.info(T("alert_list_empty"))
        else:
            with st.spinner("Mengecek harga untuk alert..."):
                all_coins = list(set([a["coin"] for a in alerts]))
                prices = get_current_prices(all_coins)
                to_remove_idx = []
                triggered = False
                for i, a in enumerate(alerts):
                    price = prices.get(a["coin"])
                    if price is None:
                        continue
                    target = a["target"]
                    op = a["op"]
                    ok = (
                        (op == ">" and price > target)
                        or (op == "<" and price < target)
                        or (op == ">=" and price >= target)
                        or (op == "<=" and price <= target)
                    )
                    if ok:
                        st.success(T("alert_trigger", coin=a["coin"].upper(), price=price, relation=op, target=target))
                        to_remove_idx.append(i)
                        triggered = True

                if not triggered:
                    st.info("Tidak ada alert yang terpenuhi.")

                if to_remove_idx:
                    for idx in reversed(to_remove_idx):
                        try:
                            alerts.pop(idx)
                        except Exception:
                            pass
                    save_state()
                    st.rerun()

with colE:
    rm_idx = st.number_input("Hapus Alert (index)", min_value=0, step=1, value=0)
    if st.button("Hapus Alert by Index"):
        if rm_idx > 0 and rm_idx <= len(alerts):
            alerts.pop(rm_idx - 1)
            save_state()
            st.success(T("alert_removed"))
            st.rerun()
        elif rm_idx > 0:
            st.warning("Indeks tidak valid.")

st.caption("© Streamlit conversion of original Telegram bot. Semua fitur inti dipertahankan.")