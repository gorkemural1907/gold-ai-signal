import os
import time
import json
from io import StringIO
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ============================================================
# TELEGRAM
# ============================================================
DEBUG_NO_TELEGRAM = False
SEND_NO_TRADE = True
SEND_DAILY_SIGNAL_ON_EVERY_RUN = True

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def send_telegram(text: str) -> None:
    if DEBUG_NO_TELEGRAM:
        print("[TG][DEBUG]")
        print(text)
        return

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TG] Telegram secrets missing.")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=30)
    print("[TG] status:", r.status_code)
    if r.status_code != 200:
        print("[TG] response:", r.text[:500])
        raise RuntimeError(f"Telegram HTTP {r.status_code}")


# ============================================================
# STATE
# ============================================================
STATE_DIR = Path(".bot_state")
STATE_PATH = STATE_DIR / "state.json"


def default_state() -> dict:
    return {
        "trade": {},
        "journal": [],
        "stats": {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "exit_now": 0,
            "cancelled": 0,
            "sum_r": 0.0,
            "avg_r": 0.0,
            "winrate": 0.0,
            "expectancy_r": 0.0,
        },
        "meta": {
            "last_daily_signal_date": None,
            "last_intraday_check_ts": None,
        },
    }


def load_state() -> dict:
    if not STATE_PATH.exists():
        return default_state()

    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            return default_state()
        state.setdefault("trade", {})
        state.setdefault("journal", [])
        state.setdefault("stats", default_state()["stats"])
        state.setdefault("meta", default_state()["meta"])
        return state
    except Exception:
        return default_state()


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# HTTP SESSION
# ============================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})
FETCH_PAUSE_S = 0.20


# ============================================================
# CONFIG
# ============================================================
TRAIN_DAYS = 252 * 5

TH_LONG = 0.58
TH_SHORT = 0.42
MIN_EDGE = 0.08
REAL_BIAS = 0.015

ATR_LEN = 14
ATR_SLOW_LEN = 20
STOP_ATR_MULT = 1.2
RR = 2.0

SPREAD_BPS = 10
SPREAD = SPREAD_BPS / 10000.0
HALF_SPREAD = SPREAD / 2.0

STRUCT_LOOKBACK = 5
SQUEEZE_LOOKBACK = 20
SQUEEZE_RATIO_MAX = 0.90
VOL_EXPANSION_MIN_DELTA = 0.02
VOL_EXPANSION_RATIO_MIN = 0.92
MIN_CHART_SCORE_TO_TRADE = 2

EXIT_P_ADJ_LONG = 0.50
EXIT_P_ADJ_SHORT = 0.50

# Retest
ENTRY_MODE_DEFAULT = "RETEST"
LATE_ENTRY_ATR_FRACTION = 0.25
RETEST_TOL_PCT = 0.0010
RETEST_REJECTION_BUFFER_PCT = 0.0003

# Momentum
ENABLE_MOMENTUM_LAYER = True
MOMENTUM_P_LONG = 0.56
MOMENTUM_P_SHORT = 0.44
MOMENTUM_MIN_CLOSE_POS_LONG = 0.70
MOMENTUM_MAX_CLOSE_POS_SHORT = 0.30
MOMENTUM_BREAKOUT_BUFFER_ATR = 0.05
MOMENTUM_STOP_ATR_MULT = 1.0

# Smart late-entry allowance
ENABLE_MOMENTUM_LATE_ALLOWANCE = True
MOMENTUM_LATE_MAX_ATR = 0.80
MOMENTUM_LATE_MIN_P_LONG = 0.59
MOMENTUM_LATE_MAX_P_SHORT = 0.41
MOMENTUM_LATE_MIN_CLOSE_POS_LONG = 0.78
MOMENTUM_LATE_MAX_CLOSE_POS_SHORT = 0.22

# Trend override
EMA_FAST = 20
EMA_SLOW = 50
ENABLE_TREND_OVERRIDE = True
TREND_OVERRIDE_MIN_CLOSE_POS_LONG = 0.72
TREND_OVERRIDE_MAX_CLOSE_POS_SHORT = 0.28
TREND_OVERRIDE_MIN_P_LONG = 0.55
TREND_OVERRIDE_MAX_P_SHORT = 0.45

# NEW: stronger long turn override
TURN_LONG_MIN_P = 0.66
TURN_LONG_MIN_CLOSE_POS = 0.52
TURN_LONG_MIN_REAL_MOM = 0.02

# Continuation override
ENABLE_CONTINUATION_OVERRIDE = True
CONT_SHORT_MAX_P = 0.56
CONT_LONG_MIN_P = 0.46
CONT_BREAK_MOVE_ATR = 0.05
CONT_SHORT_MAX_CLOSE_POS = 0.65
CONT_LONG_MIN_CLOSE_POS = 0.55

# Live continuation entry
ENABLE_LIVE_CONTINUATION_ENTRY = True
LIVE_ENTRY_MAX_ATR = 0.25

# Intraday execution
ENABLE_INTRADAY_EXECUTION = True
YAHOO_INTRADAY_SYMBOL = "GC=F"
INTRADAY_INTERVAL = "5m"
INTRADAY_RANGE = "5d"
MAX_INTRADAY_BARS = 1500

# Trade management
ENABLE_AUTO_MANAGEMENT = True
BE_AT_R = 0.75
PARTIAL_AT_R = 1.00
PARTIAL_CLOSE_FRACTION = 0.50
TRAIL_ENABLE_AT_R = 1.50
TRAIL_LOCK_R = 0.80

# Symbols
XAU_PROVIDERS = [("yahoo", "GC=F"), ("stooq", "xauusd")]
DXY_PROVIDERS = [("yahoo", "DX-Y.NYB"), ("stooq", "usd_i"), ("stooq", "usdidx")]
US10Y_PROVIDERS = [("yahoo", "^TNX"), ("stooq", "10yusy.b"), ("stooq", "us10y")]
VIX_PROVIDERS = [("yahoo", "^VIX"), ("stooq", "vi.f"), ("stooq", "vix")]
SPX_PROVIDERS = [("yahoo", "^GSPC"), ("stooq", "^spx")]
TIPS_PROVIDERS = [("yahoo", "TIP"), ("stooq", "tip.us")]
IEF_PROVIDERS = [("yahoo", "IEF"), ("stooq", "ief.us")]


# ============================================================
# HELPERS
# ============================================================
def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def round4(v):
    if v is None:
        return None
    try:
        return round(float(v), 4)
    except Exception:
        return None


def clamp01(x: float) -> float:
    return float(min(0.999, max(0.001, x)))


def normalize_ohlc_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def compute_atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            (df["High"] - df["Low"]).rename("hl"),
            (df["High"] - prev_close).abs().rename("hc"),
            (df["Low"] - prev_close).abs().rename("lc"),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()


def calc_mfe_mae(side: str, entry: float, high: float, low: float) -> tuple[float, float]:
    if side == "SHORT":
        mfe = max(0.0, entry - low)
        mae = max(0.0, high - entry)
    else:
        mfe = max(0.0, high - entry)
        mae = max(0.0, entry - low)
    return mfe, mae


def r_multiple(side: str, entry: float, stop_dist: float, price: float) -> float:
    if stop_dist <= 0:
        return 0.0
    if side == "LONG":
        return (price - entry) / stop_dist
    return (entry - price) / stop_dist


def compute_r_result(result: str, entry: float, stop_dist: float, close_price: float | None = None) -> float:
    if stop_dist <= 0:
        return 0.0
    if result == "TAKE_PROFIT":
        return RR
    if result == "STOP":
        return -1.0
    if result == "BREAKEVEN":
        return 0.0
    if result == "CANCELLED":
        return 0.0
    if result == "EXIT_NOW" and close_price is not None:
        return (close_price - entry) / stop_dist
    return 0.0


# ============================================================
# FETCHERS
# ============================================================
def fetch_ohlc_stooq(symbol: str, retries: int = 2, sleep_s: float = 1.0) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            print(f"[DATA][STOOQ] {symbol} attempt {attempt}")
            time.sleep(FETCH_PAUSE_S)
            r = SESSION.get(url, timeout=25)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")

            txt = (r.text or "").strip()
            if "Exceeded the daily hits limit" in txt:
                raise RuntimeError("Stooq quota exceeded")
            if len(txt) < 40:
                raise RuntimeError("Empty CSV")

            df = pd.read_csv(StringIO(txt))
            needed = {"Date", "Open", "High", "Low", "Close"}
            if not needed.issubset(df.columns):
                raise RuntimeError(f"Missing columns: {list(df.columns)}")

            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
            df = df.sort_values("Date").set_index("Date")
            df = df[["Open", "High", "Low", "Close"]].dropna()
            df = df[~df.index.duplicated(keep="last")]
            if len(df) < 200:
                raise RuntimeError(f"Too few rows: {len(df)}")
            return normalize_ohlc_index(df)

        except Exception as e:
            last_err = e
            print(f"[DATA][STOOQ] {symbol} failed: {type(e).__name__}: {str(e)[:160]}")
            time.sleep(sleep_s)

    raise RuntimeError(f"Stooq failed for {symbol}: {type(last_err).__name__}: {last_err}")


def fetch_ohlc_yahoo(symbol: str, retries: int = 3, sleep_s: float = 1.0) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=10y&interval=1d&includePrePost=false"
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            print(f"[DATA][YAHOO] {symbol} attempt {attempt}")
            time.sleep(FETCH_PAUSE_S)
            r = SESSION.get(url, timeout=25)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")

            data = r.json()
            result = data.get("chart", {}).get("result")
            if not result:
                raise RuntimeError("No Yahoo result")

            result0 = result[0]
            ts = result0["timestamp"]
            q = result0["indicators"]["quote"][0]

            df = pd.DataFrame({
                "Date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize(),
                "Open": q["open"],
                "High": q["high"],
                "Low": q["low"],
                "Close": q["close"],
            }).dropna()

            df = df.sort_values("Date").set_index("Date")
            df = df[["Open", "High", "Low", "Close"]]
            df = df[~df.index.duplicated(keep="last")]
            if len(df) < 200:
                raise RuntimeError(f"Too few rows: {len(df)}")
            return normalize_ohlc_index(df)

        except Exception as e:
            last_err = e
            print(f"[DATA][YAHOO] {symbol} failed: {type(e).__name__}: {str(e)[:160]}")
            time.sleep(sleep_s)

    raise RuntimeError(f"Yahoo failed for {symbol}: {type(last_err).__name__}: {last_err}")


def fetch_intraday_yahoo(symbol: str, interval: str = INTRADAY_INTERVAL, range_: str = INTRADAY_RANGE) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range={range_}&interval={interval}&includePrePost=false"
    print(f"[DATA][YAHOO][INTRADAY] {symbol} {range_} {interval}")
    time.sleep(FETCH_PAUSE_S)

    r = SESSION.get(url, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")

    data = r.json()
    result = data.get("chart", {}).get("result")
    if not result:
        raise RuntimeError("No Yahoo intraday result")

    result0 = result[0]
    ts = result0.get("timestamp") or []
    q = result0["indicators"]["quote"][0]
    if not ts:
        raise RuntimeError("Empty intraday timestamps")

    df = pd.DataFrame({
        "Datetime": pd.to_datetime(ts, unit="s", utc=True).tz_convert(None),
        "Open": q["open"],
        "High": q["high"],
        "Low": q["low"],
        "Close": q["close"],
    }).dropna()

    if df.empty:
        raise RuntimeError("Empty intraday DataFrame")

    df = df.sort_values("Datetime").set_index("Datetime")
    df = df[["Open", "High", "Low", "Close"]]
    df = df[~df.index.duplicated(keep="last")]

    if len(df) > MAX_INTRADAY_BARS:
        df = df.iloc[-MAX_INTRADAY_BARS:]

    return df


def fetch_ohlc_with_fallback(name: str, providers: list[tuple[str, str]]) -> tuple[pd.DataFrame, str]:
    errors = []
    for provider, symbol in providers:
        try:
            if provider == "yahoo":
                return fetch_ohlc_yahoo(symbol), f"{provider}:{symbol}"
            if provider == "stooq":
                return fetch_ohlc_stooq(symbol), f"{provider}:{symbol}"
        except Exception as e:
            err = f"{provider}:{symbol} -> {type(e).__name__}: {str(e)[:120]}"
            print(f"[DATA][FALLBACK] {name} {err}")
            errors.append(err)
    raise RuntimeError(f"{name} fetch failed. " + " | ".join(errors))


# ============================================================
# FEATURES / ENGINE
# ============================================================
def make_features(
    xau_ohlc: pd.DataFrame,
    dxy: pd.DataFrame | None,
    us10y: pd.DataFrame | None,
    vix: pd.DataFrame | None,
    spx: pd.DataFrame | None,
    tips: pd.DataFrame | None,
    ief: pd.DataFrame | None,
) -> pd.DataFrame:
    xau_ohlc = normalize_ohlc_index(xau_ohlc)

    series = [xau_ohlc["Close"].rename("xau")]
    if dxy is not None:
        series.append(normalize_ohlc_index(dxy)["Close"].rename("dxy"))
    if us10y is not None:
        series.append(normalize_ohlc_index(us10y)["Close"].rename("us10y"))
    if vix is not None:
        series.append(normalize_ohlc_index(vix)["Close"].rename("vix"))
    if spx is not None:
        series.append(normalize_ohlc_index(spx)["Close"].rename("spx"))
    if tips is not None:
        series.append(normalize_ohlc_index(tips)["Close"].rename("tips"))
    if ief is not None:
        series.append(normalize_ohlc_index(ief)["Close"].rename("ief"))

    df = pd.concat(series, axis=1).dropna()
    df = df[~df.index.duplicated(keep="last")]

    df = df.join(
        xau_ohlc[["Open", "High", "Low"]].rename(
            columns={"Open": "open_xau", "High": "high_xau", "Low": "low_xau"}
        ),
        how="left",
    )

    df["x_ret1"] = np.log(df["xau"]).diff()
    df["x_ret3"] = np.log(df["xau"]).diff(3)
    df["x_ret5"] = np.log(df["xau"]).diff(5)

    if "dxy" in df.columns:
        df["dxy_ret1"] = np.log(df["dxy"]).diff()
        df["dxy_ret5"] = np.log(df["dxy"]).diff(5)

    if "us10y" in df.columns:
        df["us10y_chg1"] = df["us10y"].diff()
        df["us10y_chg5"] = df["us10y"].diff(5)
        df["gold_real_spread"] = np.log(df["xau"] / df["us10y"].replace(0, np.nan))
        df["gold_real_mom"] = df["gold_real_spread"].diff(5)

    if "vix" in df.columns:
        df["vix_ret1"] = np.log(df["vix"]).diff()
        df["vix_ret5"] = np.log(df["vix"]).diff(5)

    if "spx" in df.columns:
        df["spx_ret5"] = np.log(df["spx"]).diff(5)
        df["gold_spx_spread"] = np.log(df["xau"] / df["spx"].replace(0, np.nan))
        df["gold_spx_mom"] = df["gold_spx_spread"].diff(5)

    if "tips" in df.columns and "ief" in df.columns:
        df["real_factor"] = np.log(df["tips"] / df["ief"])
        df["real_f_chg5"] = df["real_factor"].diff(5)

    df["atr14"] = compute_atr(xau_ohlc, ATR_LEN)
    df["atr20"] = compute_atr(xau_ohlc, ATR_SLOW_LEN)

    df["ma50"] = df["xau"].rolling(50).mean()
    df["trend_up"] = (df["xau"] > df["ma50"]).astype(int)

    h = xau_ohlc["High"].reindex(df.index)
    l = xau_ohlc["Low"].reindex(df.index)
    o = xau_ohlc["Open"].reindex(df.index)
    c = xau_ohlc["Close"].reindex(df.index)

    prev_high_max = h.shift(1).rolling(STRUCT_LOOKBACK).max()
    prev_low_min = l.shift(1).rolling(STRUCT_LOOKBACK).min()

    df["hh"] = (h > prev_high_max).astype(int)
    df["hl"] = (l > prev_low_min).astype(int)
    df["ll"] = (l < prev_low_min).astype(int)
    df["lh"] = (h < prev_high_max).astype(int)

    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

    df["close_pos"] = ((c - l) / rng).clip(0, 1)
    df["upper_wick_body"] = (upper_wick / body.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    df["lower_wick_body"] = (lower_wick / body.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    atr20_med = df["atr20"].rolling(SQUEEZE_LOOKBACK).median()
    df["atr_ratio"] = df["atr14"] / atr20_med
    df["squeeze_on"] = (df["atr_ratio"] < SQUEEZE_RATIO_MAX).astype(int)
    df["vol_expansion"] = (
        (df["atr_ratio"] > VOL_EXPANSION_RATIO_MIN) &
        ((df["atr_ratio"] - df["atr_ratio"].shift(1)) > VOL_EXPANSION_MIN_DELTA)
    ).astype(int)

    df["y"] = (df["xau"].shift(-1) > df["xau"]).astype(int)

    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    return df


def compute_regime_flags(feats: pd.DataFrame) -> pd.DataFrame:
    df = feats.copy()
    df["ema20"] = df["xau"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema50"] = df["xau"].ewm(span=EMA_SLOW, adjust=False).mean()
    df["ema20_up"] = (df["ema20"] > df["ema20"].shift(1)).astype(int)

    df["trend_override_up"] = (
        (df["vol_expansion"] == 1) &
        (df["close_pos"] >= TREND_OVERRIDE_MIN_CLOSE_POS_LONG) &
        ((df["hh"] == 1) | (df["hl"] == 1) | (df["ema20_up"] == 1))
    ).astype(int)

    df["trend_override_down"] = (
        (df["vol_expansion"] == 1) &
        (df["close_pos"] <= TREND_OVERRIDE_MAX_CLOSE_POS_SHORT) &
        ((df["ll"] == 1) | (df["lh"] == 1) | (df["ema20_up"] == 0))
    ).astype(int)

    return df


def detect_effective_trend(r: pd.Series, p_adj: float) -> tuple[str, str]:
    base_trend = "UP" if int(r["trend_up"]) == 1 else "DOWN"

    if not ENABLE_TREND_OVERRIDE:
        return base_trend, "ma50"

    # strong continuation overrides
    if int(r.get("trend_override_up", 0)) == 1 and p_adj >= TREND_OVERRIDE_MIN_P_LONG:
        return "UP", "override_up"

    if int(r.get("trend_override_down", 0)) == 1 and p_adj <= TREND_OVERRIDE_MAX_P_SHORT:
        return "DOWN", "override_down"

    # NEW: strong long turn override even if MA50 still down
    gold_real_mom = float(r.get("gold_real_mom", 0.0))
    close_pos = float(r.get("close_pos", 0.0))
    vol_expansion = int(r.get("vol_expansion", 0))

    if (
        p_adj >= TURN_LONG_MIN_P
        and vol_expansion == 1
        and gold_real_mom >= TURN_LONG_MIN_REAL_MOM
        and close_pos >= TURN_LONG_MIN_CLOSE_POS
    ):
        return "UP", "turn_long_override"

    return base_trend, "ma50"


def chart_filters(feats: pd.DataFrame, idx: pd.Timestamp, side: str) -> tuple[bool, str, int]:
    r = feats.loc[idx]
    score = 0
    reasons = []

    if side == "LONG":
        if int(r.get("hh", 0)) == 1:
            score += 1
        else:
            reasons.append("no higher high")
        if int(r.get("hl", 0)) == 1:
            score += 1
        else:
            reasons.append("no higher low")
        if float(r.get("close_pos", 0.0)) >= 0.60:
            score += 1
        else:
            reasons.append("weak close position")
    else:
        if int(r.get("ll", 0)) == 1:
            score += 1
        else:
            reasons.append("no lower low")
        if int(r.get("lh", 0)) == 1:
            score += 1
        else:
            reasons.append("no lower high")
        if float(r.get("close_pos", 1.0)) <= 0.40:
            score += 1
        else:
            reasons.append("weak close position")

    passed = score >= MIN_CHART_SCORE_TO_TRADE
    reason = "ok" if passed else "; ".join(reasons[:3]) if reasons else "score too low"
    return passed, reason, score


def build_trade_setup(side: str, y_high: float, y_low: float, atr: float, mode: str) -> tuple[float, float, float, float]:
    if mode == "RETEST":
        if side == "LONG":
            break_level = y_high
            entry = y_high * (1.0 + HALF_SPREAD)
            sl = entry - STOP_ATR_MULT * atr
            tp = entry + RR * (STOP_ATR_MULT * atr)
        else:
            break_level = y_low
            entry = y_low * (1.0 - HALF_SPREAD)
            sl = entry + STOP_ATR_MULT * atr
            tp = entry - RR * (STOP_ATR_MULT * atr)
        return entry, sl, tp, break_level

    if side == "LONG":
        break_level = y_high
        entry = y_high + (MOMENTUM_BREAKOUT_BUFFER_ATR * atr)
        sl = entry - MOMENTUM_STOP_ATR_MULT * atr
        tp = entry + RR * (MOMENTUM_STOP_ATR_MULT * atr)
    else:
        break_level = y_low
        entry = y_low - (MOMENTUM_BREAKOUT_BUFFER_ATR * atr)
        sl = entry + MOMENTUM_STOP_ATR_MULT * atr
        tp = entry - RR * (MOMENTUM_STOP_ATR_MULT * atr)

    return entry, sl, tp, break_level


def recompute_live_trade_setup(side: str, live_entry: float, atr: float) -> tuple[float, float]:
    if side == "LONG":
        sl = live_entry - MOMENTUM_STOP_ATR_MULT * atr
        tp = live_entry + RR * (MOMENTUM_STOP_ATR_MULT * atr)
    else:
        sl = live_entry + MOMENTUM_STOP_ATR_MULT * atr
        tp = live_entry - RR * (MOMENTUM_STOP_ATR_MULT * atr)
    return sl, tp


def late_entry_filter(side: str, price: float, level: float, atr: float) -> tuple[bool, str, float]:
    if not np.isfinite(price) or not np.isfinite(level) or not np.isfinite(atr) or atr <= 0:
        return False, "", 0.0
    move = abs(price - level)
    is_late = move > (atr * LATE_ENTRY_ATR_FRACTION)
    if is_late:
        return True, f"late entry: move {move:.2f} > {LATE_ENTRY_ATR_FRACTION:.2f}*ATR ({atr * LATE_ENTRY_ATR_FRACTION:.2f})", move
    return False, "", move


def rejection_filter(side: str, o: float, h: float, l: float, c: float, level: float, atr: float) -> tuple[bool, str]:
    if not np.isfinite(atr) or atr <= 0:
        return False, "invalid ATR"
    body = abs(c - o)
    if body < atr * 0.01:
        return False, "weak body"

    if side == "LONG":
        touched = l <= level * (1.0 + RETEST_TOL_PCT)
        bullish_close = c >= level * (1.0 + RETEST_REJECTION_BUFFER_PCT)
        ok = touched and bullish_close
        return ok, "ok" if ok else f"touched={touched}, bullish_close={bullish_close}"
    else:
        touched = h >= level * (1.0 - RETEST_TOL_PCT)
        bearish_close = c <= level * (1.0 - RETEST_REJECTION_BUFFER_PCT)
        ok = touched and bearish_close
        return ok, "ok" if ok else f"touched={touched}, bearish_close={bearish_close}"


def momentum_filter(side: str, p_adj: float, close_pos: float, vol_expansion: int, close_now: float, y_high: float, y_low: float, atr: float) -> tuple[bool, str]:
    if side == "LONG":
        conds = {
            "p_adj": p_adj >= MOMENTUM_P_LONG,
            "close_pos": close_pos >= MOMENTUM_MIN_CLOSE_POS_LONG,
            "vol_expansion": vol_expansion == 1,
            "breakout": close_now >= y_high + (MOMENTUM_BREAKOUT_BUFFER_ATR * atr),
        }
    else:
        conds = {
            "p_adj": p_adj <= MOMENTUM_P_SHORT,
            "close_pos": close_pos <= MOMENTUM_MAX_CLOSE_POS_SHORT,
            "vol_expansion": vol_expansion == 1,
            "breakout": close_now <= y_low - (MOMENTUM_BREAKOUT_BUFFER_ATR * atr),
        }
    return all(conds.values()), ", ".join([f"{k}={v}" for k, v in conds.items()])


def momentum_late_allowance(side: str, p_adj: float, close_pos: float, vol_expansion: int, late_move: float, atr: float) -> tuple[bool, str]:
    if not ENABLE_MOMENTUM_LATE_ALLOWANCE:
        return False, "disabled"
    if not np.isfinite(atr) or atr <= 0:
        return False, "invalid ATR"

    late_atr = late_move / atr
    if side == "LONG":
        ok = (
            vol_expansion == 1 and
            p_adj >= MOMENTUM_LATE_MIN_P_LONG and
            close_pos >= MOMENTUM_LATE_MIN_CLOSE_POS_LONG and
            late_atr <= MOMENTUM_LATE_MAX_ATR
        )
    else:
        ok = (
            vol_expansion == 1 and
            p_adj <= MOMENTUM_LATE_MAX_P_SHORT and
            close_pos <= MOMENTUM_LATE_MAX_CLOSE_POS_SHORT and
            late_atr <= MOMENTUM_LATE_MAX_ATR
        )
    return ok, f"late_move_atr={late_atr:.2f}, vol_expansion={vol_expansion}, p_adj={p_adj:.4f}, close_pos={close_pos:.2f}"


def update_trade_extremes(trade: dict, side: str, high: float, low: float, entry: float) -> dict:
    mfe, mae = calc_mfe_mae(side, entry, high, low)
    trade["max_favorable"] = max(float(trade.get("max_favorable", 0.0) or 0.0), mfe)
    trade["max_adverse"] = max(float(trade.get("max_adverse", 0.0) or 0.0), mae)
    return trade


def continuation_override(
    effective_trend: str,
    p_adj: float,
    close_now: float,
    y_high: float,
    y_low: float,
    atr: float,
    close_pos: float,
    vol_expansion: int,
) -> tuple[str, str]:
    if not ENABLE_CONTINUATION_OVERRIDE or not np.isfinite(atr) or atr <= 0:
        return "NO-TRADE", ""

    below_y_low = y_low - close_now
    above_y_high = close_now - y_high

    if (
        effective_trend == "DOWN"
        and vol_expansion == 1
        and close_pos <= CONT_SHORT_MAX_CLOSE_POS
        and below_y_low >= CONT_BREAK_MOVE_ATR * atr
        and p_adj <= CONT_SHORT_MAX_P
    ):
        return "SHORT", f"breakdown continuation override (below_y_low={below_y_low:.2f})"

    if (
        effective_trend == "UP"
        and vol_expansion == 1
        and close_pos >= CONT_LONG_MIN_CLOSE_POS
        and above_y_high >= CONT_BREAK_MOVE_ATR * atr
        and p_adj >= CONT_LONG_MIN_P
    ):
        return "LONG", f"breakout continuation override (above_y_high={above_y_high:.2f})"

    return "NO-TRADE", ""


def maybe_use_live_continuation_entry(side: str, entry: float, close_now: float, atr: float) -> tuple[bool, bool, float, str]:
    if not ENABLE_LIVE_CONTINUATION_ENTRY or not np.isfinite(entry) or not np.isfinite(close_now) or not np.isfinite(atr) or atr <= 0:
        return False, False, entry, ""

    max_move = LIVE_ENTRY_MAX_ATR * atr

    if side == "SHORT" and close_now <= entry:
        move = entry - close_now
        if move <= max_move:
            return True, False, close_now, f"live short entry used (move={move:.2f})"
        return False, True, entry, f"too stretched for live short (move={move:.2f} > {max_move:.2f})"

    if side == "LONG" and close_now >= entry:
        move = close_now - entry
        if move <= max_move:
            return True, False, close_now, f"live long entry used (move={move:.2f})"
        return False, True, entry, f"too stretched for live long (move={move:.2f} > {max_move:.2f})"

    return False, False, entry, ""


# ============================================================
# GRADE / STATS
# ============================================================
def compute_grade(
    side: str,
    p_adj: float,
    chart_score: int,
    late_filter_reason: str,
    rejection_ok: bool,
    effective_trend: str,
    real_chg5: float | None,
    entry_mode: str,
    momentum_ok: bool,
    late_allow_ok: bool,
    override_reason: str = "",
    live_entry_reason: str = "",
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    score = 0

    if side == "NO-TRADE":
        return "N/A", ["no valid setup"]

    if side == "LONG":
        if effective_trend == "UP":
            score += 1
            reasons.append("trend aligned")
        if p_adj >= 0.64:
            score += 2
            reasons.append("strong AI edge")
        elif p_adj >= 0.60:
            score += 1
            reasons.append("good AI edge")
        elif override_reason:
            score += 1
            reasons.append("continuation override used")
        if real_chg5 is not None and np.isfinite(real_chg5) and real_chg5 > 0:
            score += 1
            reasons.append("real factor aligned")
    else:
        if effective_trend == "DOWN":
            score += 1
            reasons.append("trend aligned")
        if p_adj <= 0.36:
            score += 2
            reasons.append("strong AI edge")
        elif p_adj <= 0.40:
            score += 1
            reasons.append("good AI edge")
        elif override_reason:
            score += 1
            reasons.append("continuation override used")
        if real_chg5 is not None and np.isfinite(real_chg5) and real_chg5 < 0:
            score += 1
            reasons.append("real factor aligned")

    if entry_mode == "RETEST":
        if chart_score >= 4:
            score += 2
            reasons.append("strong chart quality")
        elif chart_score >= 3:
            score += 1
            reasons.append("good chart quality")
        if rejection_ok:
            score += 1
            reasons.append("clean rejection setup")
        if not late_filter_reason:
            score += 1
            reasons.append("not late")
    else:
        if momentum_ok or override_reason:
            score += 2
            reasons.append("momentum continuation setup")
        if late_allow_ok:
            score += 1
            reasons.append("smart late allowance")
        if chart_score >= 1:
            score += 1
            reasons.append("minimum chart confirmation")
        reasons.append("trend continuation mode")
        if live_entry_reason:
            reasons.append("live continuation entry")

    if score >= 6:
        return "A", reasons
    if score >= 4:
        return "B", reasons
    return "C", reasons


def add_journal_entry(state: dict, trade: dict, result: str, close_date: str, close_price: float | None = None) -> None:
    journal = state.setdefault("journal", [])
    entry = safe_float(trade.get("entry"), 0.0)
    stop_dist = safe_float(trade.get("stop_dist"), 0.0)
    r_mult = compute_r_result(result, entry, stop_dist, close_price)

    row = {
        "created_date": trade.get("created_date"),
        "open_date": trade.get("open_date"),
        "close_date": close_date,
        "side": trade.get("side"),
        "entry_mode": trade.get("entry_mode"),
        "grade": trade.get("grade"),
        "entry": round4(entry),
        "sl": round4(trade.get("sl_initial", trade.get("sl"))),
        "tp": round4(trade.get("tp")),
        "stop_dist": round4(stop_dist),
        "result": result,
        "close_price": round4(close_price),
        "r_mult": round4(r_mult),
        "max_favorable": round4(trade.get("max_favorable", 0.0)),
        "max_adverse": round4(trade.get("max_adverse", 0.0)),
        "partial_taken": bool(trade.get("partial_taken", False)),
        "partial_price": round4(trade.get("partial_price")),
    }

    journal.append(row)
    state["journal"] = journal[-500:]
    refresh_stats(state)


def refresh_stats(state: dict) -> None:
    journal = state.get("journal", [])
    total = wins = losses = breakeven = exit_now = cancelled = 0
    sum_r = 0.0

    for j in journal:
        result = j.get("result")
        r_mult = float(j.get("r_mult", 0.0) or 0.0)
        if result == "TAKE_PROFIT":
            wins += 1
            total += 1
            sum_r += r_mult
        elif result == "STOP":
            losses += 1
            total += 1
            sum_r += r_mult
        elif result == "BREAKEVEN":
            breakeven += 1
            total += 1
            sum_r += r_mult
        elif result == "EXIT_NOW":
            exit_now += 1
            total += 1
            sum_r += r_mult
        elif result == "CANCELLED":
            cancelled += 1

    avg_r = (sum_r / total) if total > 0 else 0.0
    winrate = (wins / total * 100.0) if total > 0 else 0.0

    state["stats"] = {
        "total": total,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "exit_now": exit_now,
        "cancelled": cancelled,
        "sum_r": round(sum_r, 4),
        "avg_r": round(avg_r, 4),
        "winrate": round(winrate, 2),
        "expectancy_r": round(avg_r, 4),
    }


def stats_text(state: dict) -> str:
    s = state.get("stats", {})
    return (
        f"Trades: {s.get('total', 0)}\n"
        f"Wins: {s.get('wins', 0)} | Losses: {s.get('losses', 0)} | BE: {s.get('breakeven', 0)} | ExitNow: {s.get('exit_now', 0)}\n"
        f"WinRate: {float(s.get('winrate', 0.0)):.2f}%\n"
        f"Avg R: {float(s.get('avg_r', 0.0)):.2f}\n"
        f"Expectancy: {float(s.get('expectancy_r', 0.0)):.2f}R"
    )


# ============================================================
# DAILY SIGNAL ENGINE
# ============================================================
def build_daily_signal(state: dict, feats: pd.DataFrame, xau: pd.DataFrame, used_sources: dict) -> tuple[dict, str]:
    last_day = feats.index[-1]
    r = feats.loc[last_day]

    p_up = float(used_sources["p_up"])
    p_adj = float(used_sources["p_adj"])
    real_chg5 = used_sources["real_chg5"]
    real_available = used_sources["real_available"]
    bias_reason = used_sources["bias_reason"]
    used_vix = used_sources["used_vix"]

    trade = state.get("trade", {})
    has_active_trade = trade.get("status") in ("PENDING", "OPEN")

    base_trend = "UP" if int(r["trend_up"]) == 1 else "DOWN"
    effective_trend, trend_source = detect_effective_trend(r, p_adj)

    close_now = float(xau["Close"].iloc[-1])
    today_bar = xau.iloc[-1]
    today_o = float(today_bar["Open"])
    today_h = float(today_bar["High"])
    today_l = float(today_bar["Low"])
    today_c = float(today_bar["Close"])

    atr = float(r["atr14"])
    close_pos = float(r["close_pos"])
    vol_expansion = int(r["vol_expansion"])

    ybar = xau.iloc[-2]
    y_high = float(ybar["High"])
    y_low = float(ybar["Low"])

    base_side = "NO-TRADE"
    if (p_adj > TH_LONG) and ((p_adj - 0.5) >= MIN_EDGE):
        base_side = "LONG"
    elif (p_adj < TH_SHORT) and ((0.5 - p_adj) >= MIN_EDGE):
        base_side = "SHORT"

    if base_side == "LONG" and effective_trend != "UP":
        if not (ENABLE_MOMENTUM_LAYER and vol_expansion == 1 and p_adj >= MOMENTUM_P_LONG):
            base_side = "NO-TRADE"
    if base_side == "SHORT" and effective_trend != "DOWN":
        if not (ENABLE_MOMENTUM_LAYER and vol_expansion == 1 and p_adj <= MOMENTUM_P_SHORT):
            base_side = "NO-TRADE"

    side = "NO-TRADE"
    entry_mode = ENTRY_MODE_DEFAULT
    entry = float(close_now)
    sl = float("nan")
    tp = float("nan")
    break_level = float("nan")
    late_filter_reason = ""
    late_move = 0.0
    rejection_ok = False
    momentum_ok = False
    late_allow_ok = False
    late_allow_reason = ""
    override_reason = ""
    live_entry_reason = ""
    stale_reason = ""
    grade = "N/A"
    grade_reasons: list[str] = []
    chart_score = 0
    chart_reason = ""
    momentum_reason = ""

    if base_side in ("LONG", "SHORT"):
        chart_ok, chart_reason, chart_score = chart_filters(feats, last_day, base_side)

        if chart_ok:
            c_entry, c_sl, c_tp, c_break = build_trade_setup(base_side, y_high, y_low, atr, "RETEST")
            if base_side == "LONG":
                is_late, late_filter_reason, late_move = late_entry_filter("LONG", close_now, y_high, atr)
                rejection_ok, _ = rejection_filter("LONG", today_o, today_h, today_l, today_c, c_break, atr)
            else:
                is_late, late_filter_reason, late_move = late_entry_filter("SHORT", close_now, y_low, atr)
                rejection_ok, _ = rejection_filter("SHORT", today_o, today_h, today_l, today_c, c_break, atr)

            if not is_late:
                side = base_side
                entry_mode = "RETEST"
                entry, sl, tp, break_level = c_entry, c_sl, c_tp, c_break

    if side == "NO-TRADE" and ENABLE_MOMENTUM_LAYER and base_side in ("LONG", "SHORT"):
        momentum_ok, momentum_reason = momentum_filter(
            side=base_side,
            p_adj=p_adj,
            close_pos=close_pos,
            vol_expansion=vol_expansion,
            close_now=close_now,
            y_high=y_high,
            y_low=y_low,
            atr=atr,
        )

        if momentum_ok:
            if base_side == "LONG":
                _, lf_reason, lm = late_entry_filter("LONG", close_now, y_high, atr)
            else:
                _, lf_reason, lm = late_entry_filter("SHORT", close_now, y_low, atr)

            late_filter_reason = lf_reason
            late_move = lm

            late_allow_ok, late_allow_reason = momentum_late_allowance(
                side=base_side,
                p_adj=p_adj,
                close_pos=close_pos,
                vol_expansion=vol_expansion,
                late_move=late_move,
                atr=atr,
            )

            if (not late_filter_reason) or late_allow_ok:
                side = base_side
                entry_mode = "MOMENTUM"
                entry, sl, tp, break_level = build_trade_setup(base_side, y_high, y_low, atr, "MOMENTUM")

    if side == "NO-TRADE":
        over_side, over_reason = continuation_override(
            effective_trend=effective_trend,
            p_adj=p_adj,
            close_now=close_now,
            y_high=y_high,
            y_low=y_low,
            atr=atr,
            close_pos=close_pos,
            vol_expansion=vol_expansion,
        )
        if over_side in ("LONG", "SHORT"):
            side = over_side
            entry_mode = "MOMENTUM"
            override_reason = over_reason
            entry, sl, tp, break_level = build_trade_setup(over_side, y_high, y_low, atr, "MOMENTUM")

    if side in ("LONG", "SHORT") and entry_mode == "MOMENTUM":
        use_live, too_stretched, live_entry, live_reason = maybe_use_live_continuation_entry(
            side=side,
            entry=entry,
            close_now=close_now,
            atr=atr,
        )

        if too_stretched:
            stale_reason = live_reason
            side = "NO-TRADE"
            grade = "N/A"
            entry = float(close_now)
            sl = float("nan")
            tp = float("nan")
            break_level = float("nan")
        elif use_live:
            entry_mode = "MOMENTUM_LIVE"
            entry = live_entry
            sl, tp = recompute_live_trade_setup(side, entry, atr)
            live_entry_reason = live_reason

    if side in ("LONG", "SHORT"):
        grade, grade_reasons = compute_grade(
            side=side,
            p_adj=p_adj,
            chart_score=chart_score,
            late_filter_reason=late_filter_reason,
            rejection_ok=rejection_ok,
            effective_trend=effective_trend,
            real_chg5=(real_chg5 if real_available else None),
            entry_mode=entry_mode,
            momentum_ok=momentum_ok,
            late_allow_ok=late_allow_ok,
            override_reason=override_reason,
            live_entry_reason=live_entry_reason,
        )

    stop_dist = abs(entry - sl) if np.isfinite(entry) and np.isfinite(sl) else STOP_ATR_MULT * atr
    rr_text = f"1:{RR:.1f}"
    risk_gate = "VIX" if used_vix is not None else "SPX_PROXY"

    msg = (
        f"GOLD AI SIGNAL (XAUUSD)\n\n"
        f"Date: {last_day.date()}\n\n"
        f"Sources:\n"
        f"XAU: {used_sources['used_xau']}\n"
        f"DXY: {used_sources['used_dxy']}\n"
        f"US10Y: {used_sources['used_us10y']}\n"
        f"VIX: {used_vix if used_vix is not None else 'NONE'}\n"
        f"SPX: {used_sources['used_spx']}\n"
        f"TIPS: {used_sources['used_tips']}\n"
        f"IEF: {used_sources['used_ief']}\n\n"
        f"P(up): {p_up:.4f}\n"
        f"P_adj: {p_adj:.4f} ({bias_reason})\n"
        f"Trend(>MA50): {base_trend}\n"
        f"EffectiveTrend: {effective_trend} ({trend_source})\n"
        f"RiskGate: {risk_gate}\n"
        f"ChartScore: {chart_score}/5\n"
        f"EntryMode: {entry_mode}\n"
        f"HasActiveTrade: {has_active_trade}\n"
    )

    if real_available:
        msg += f"RealFactor chg5: {real_chg5:+.4f}\n"
    if "gold_real_mom" in feats.columns:
        msg += f"GoldRealMom5: {float(r['gold_real_mom']):+.4f}\n"
    if "gold_spx_mom" in feats.columns:
        msg += f"GoldSPXMom5: {float(r['gold_spx_mom']):+.4f}\n"
    msg += f"VolExpansion: {int(r['vol_expansion'])}\n"

    if chart_reason:
        msg += f"ChartFilter: {chart_reason}\n"
    if late_filter_reason:
        msg += f"LateFilter: {late_filter_reason}\n"
    if override_reason:
        msg += f"ContinuationOverride: {override_reason}\n"
    if live_entry_reason:
        msg += f"LiveEntry: {live_entry_reason}\n"
    if stale_reason:
        msg += f"StaleEntryFilter: {stale_reason}\n"
    if base_side in ("LONG", "SHORT") or override_reason:
        msg += f"RejectionFilter: {'ok' if rejection_ok else 'weak'}\n"
        msg += f"MomentumFilter: {'ok' if momentum_ok else 'weak'}\n"
        if late_allow_reason:
            msg += f"LateAllowance: {'ok' if late_allow_ok else 'weak'} ({late_allow_reason})\n"

    msg += (
        f"\nSignal: {side}\n"
        f"Grade: {grade}\n\n"
        f"Breakout Entry: {entry:.2f}\n"
        f"BreakLevel: {break_level:.2f}\n"
        f"Yesterday High: {y_high:.2f}\n"
        f"Yesterday Low: {y_low:.2f}\n"
        f"Close Now: {close_now:.2f}\n"
        f"LateMove: {late_move:.2f}\n"
        f"Stop Loss: {sl:.2f}\n"
        f"Take Profit: {tp:.2f}\n"
        f"ATR({ATR_LEN}): {atr:.2f}\n"
        f"StopDist: {stop_dist:.2f}\n"
        f"RR: {rr_text}\n"
    )

    if grade_reasons:
        msg += "\nQuality:\n"
        for reason in grade_reasons:
            msg += f"- {reason}\n"

    if momentum_reason and side == "NO-TRADE":
        msg += f"\nMomentumReason: {momentum_reason}\n"

    msg += f"\nPerformance Snapshot:\n{stats_text(state)}\n"

    if has_active_trade:
        msg += "\nNote: Active trade exists, new setup will NOT be stored.\n"

    if side in ("LONG", "SHORT") and not has_active_trade:
        state["trade"] = {
            "status": "PENDING",
            "created_date": str(last_day.date()),
            "valid_until": str(last_day.date()),
            "side": side,
            "grade": grade,
            "entry_mode": entry_mode,
            "entry": float(entry),
            "break_level": float(break_level),
            "sl": float(sl),
            "sl_initial": float(sl),
            "tp": float(tp),
            "stop_dist": float(stop_dist),
            "be_moved": False,
            "partial_taken": False,
            "partial_fraction": PARTIAL_CLOSE_FRACTION,
            "partial_price": None,
            "trail_active": False,
            "trail_sl": None,
            "max_favorable": 0.0,
            "max_adverse": 0.0,
            "last_processed_date": trade.get("last_processed_date"),
            "last_intraday_bar_ts": trade.get("last_intraday_bar_ts"),
            "opened_via": None,
            "remaining_size": 1.0,
        }

    state["meta"]["last_daily_signal_date"] = str(last_day.date())
    return state, msg


# ============================================================
# INTRADAY EXECUTION / MANAGEMENT
# ============================================================
def process_intraday_bars(state: dict, intraday: pd.DataFrame) -> tuple[dict, list[str]]:
    msgs: list[str] = []
    trade = state.get("trade", {})
    if not trade or trade.get("status") not in ("PENDING", "OPEN"):
        return state, msgs

    side = trade["side"]
    entry = float(trade["entry"])
    tp = float(trade["tp"])
    stop_dist = float(trade["stop_dist"])
    entry_mode = str(trade.get("entry_mode", ENTRY_MODE_DEFAULT))
    last_bar_ts = trade.get("last_intraday_bar_ts")

    bars = intraday.copy()
    if last_bar_ts:
        bars = bars[bars.index > pd.to_datetime(last_bar_ts)]
    if bars.empty:
        return state, msgs

    for ts, row in bars.iterrows():
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])

        trade = update_trade_extremes(trade, side, h, l, entry)

        if trade["status"] == "PENDING":
            triggered = False
            if side == "LONG" and h >= entry:
                triggered = True
            elif side == "SHORT" and l <= entry:
                triggered = True

            if triggered:
                trade["status"] = "OPEN"
                trade["open_date"] = str(ts.date())
                trade["open_ts"] = ts.isoformat()
                trade["opened_via"] = "INTRADAY"
                msgs.append(
                    "✅ TRADE OPENED\n"
                    f"Time: {ts}\n"
                    f"Side: {side}\n"
                    f"Grade: {trade.get('grade','N/A')}\n"
                    f"Mode: {entry_mode}\n"
                    f"Entry: {entry:.2f}\n"
                    f"SL: {trade['sl']:.2f}\n"
                    f"TP: {trade['tp']:.2f}"
                )

        if trade["status"] == "OPEN" and ENABLE_AUTO_MANAGEMENT:
            current_r = r_multiple(side, entry, stop_dist, h if side == "LONG" else l)

            if (not trade.get("be_moved", False)) and current_r >= BE_AT_R:
                trade["sl"] = entry
                trade["be_moved"] = True
                msgs.append(
                    "🟦 MOVE STOP TO BREAK-EVEN\n"
                    f"Time: {ts}\n"
                    f"Side: {side}\n"
                    f"New SL: {entry:.2f}\n"
                    f"Reached: {current_r:.2f}R"
                )

            if (not trade.get("partial_taken", False)) and current_r >= PARTIAL_AT_R:
                trade["partial_taken"] = True
                trade["partial_price"] = c
                trade["remaining_size"] = max(0.0, 1.0 - PARTIAL_CLOSE_FRACTION)
                msgs.append(
                    "🟨 PARTIAL TAKE PROFIT\n"
                    f"Time: {ts}\n"
                    f"Side: {side}\n"
                    f"Price: {c:.2f}\n"
                    f"Closed: {PARTIAL_CLOSE_FRACTION:.2f}\n"
                    f"Remaining: {trade['remaining_size']:.2f}\n"
                    f"Reached: {current_r:.2f}R"
                )

            if current_r >= TRAIL_ENABLE_AT_R:
                trade["trail_active"] = True

            if trade.get("trail_active", False):
                if side == "LONG":
                    candidate_trail = entry + (TRAIL_LOCK_R * stop_dist)
                    if candidate_trail > float(trade["sl"]):
                        trade["sl"] = candidate_trail
                        msgs.append(
                            "🟪 TRAILING STOP UPDATED\n"
                            f"Time: {ts}\n"
                            f"Side: {side}\n"
                            f"New SL: {trade['sl']:.2f}"
                        )
                else:
                    candidate_trail = entry - (TRAIL_LOCK_R * stop_dist)
                    if candidate_trail < float(trade["sl"]):
                        trade["sl"] = candidate_trail
                        msgs.append(
                            "🟪 TRAILING STOP UPDATED\n"
                            f"Time: {ts}\n"
                            f"Side: {side}\n"
                            f"New SL: {trade['sl']:.2f}"
                        )

        if trade["status"] == "OPEN":
            current_sl = float(trade["sl"])

            if side == "LONG":
                hit_sl = l <= current_sl
                hit_tp = h >= tp
            else:
                hit_sl = h >= current_sl
                hit_tp = l <= tp

            if hit_sl or hit_tp:
                if hit_sl and hit_tp:
                    result = "STOP"
                elif hit_tp:
                    result = "TAKE_PROFIT"
                else:
                    result = "BREAKEVEN" if abs(current_sl - entry) < 1e-9 else "STOP"

                trade["status"] = "CLOSED"
                trade["result"] = result
                trade["close_date"] = str(ts.date())
                trade["close_ts"] = ts.isoformat()

                close_price = tp if result == "TAKE_PROFIT" else current_sl
                add_journal_entry(state, trade, result, str(ts.date()), close_price=close_price)

                emoji = "🟩" if result == "TAKE_PROFIT" else ("⬜" if result == "BREAKEVEN" else "🟥")
                label = "TAKE PROFIT" if result == "TAKE_PROFIT" else ("BREAK-EVEN" if result == "BREAKEVEN" else "STOP")

                msgs.append(
                    f"{emoji} TRADE CLOSED ({label})\n"
                    f"Time: {ts}\n"
                    f"Side: {side}\n"
                    f"Grade: {trade.get('grade','N/A')}\n"
                    f"Entry: {entry:.2f}\n"
                    f"SL: {current_sl:.2f}\n"
                    f"TP: {tp:.2f}\n"
                    f"MFE(max): {float(trade.get('max_favorable',0.0)):.2f}\n"
                    f"MAE(max): {float(trade.get('max_adverse',0.0)):.2f}\n\n"
                    + stats_text(state)
                )
                break

        trade["last_intraday_bar_ts"] = ts.isoformat()

    if trade.get("status") in ("CLOSED", "CANCELLED"):
        state["trade"] = {}
    else:
        state["trade"] = trade

    state["meta"]["last_intraday_check_ts"] = now_utc_iso()
    return state, msgs


# ============================================================
# DAILY REVIEW OF OPEN TRADE
# ============================================================
def evaluate_daily_open_trade(state: dict, xau_ohlc: pd.DataFrame, feats: pd.DataFrame | None = None, p_up_adj: float | None = None) -> tuple[dict, list[str]]:
    msgs: list[str] = []
    if len(xau_ohlc) < 3:
        return state, msgs

    trade = state.get("trade")
    if not isinstance(trade, dict) or not trade:
        return state, msgs
    if trade.get("status") not in ("PENDING", "OPEN"):
        return state, msgs

    last_bar_date = xau_ohlc.index[-1]
    if trade.get("last_processed_date") == str(last_bar_date.date()):
        return state, msgs

    bar = xau_ohlc.iloc[-1]
    h = float(bar["High"])
    l = float(bar["Low"])
    c = float(bar["Close"])

    side = trade["side"]
    entry = float(trade["entry"])
    entry_mode = str(trade.get("entry_mode", ENTRY_MODE_DEFAULT))

    trade = update_trade_extremes(trade, side, h, l, entry)

    msgs.append(
        "📊 DAILY UPDATE\n"
        f"Date: {last_bar_date.date()}\n"
        f"Side: {side}\n"
        f"Grade: {trade.get('grade','N/A')}\n"
        f"Status: {trade.get('status')}\n"
        f"Mode: {entry_mode}\n"
        f"Entry: {trade['entry']:.2f}\n"
        f"SL: {trade['sl']:.2f}\n"
        f"TP: {trade['tp']:.2f}\n"
        f"Today High: {h:.2f}\n"
        f"Today Low: {l:.2f}\n"
        f"Today Close: {c:.2f}\n"
        f"MFE(max): {float(trade.get('max_favorable', 0.0)):.2f}\n"
        f"MAE(max): {float(trade.get('max_adverse', 0.0)):.2f}"
    )

    if trade.get("status") == "PENDING":
        valid_until = trade.get("valid_until")
        if valid_until and str(last_bar_date.date()) >= valid_until:
            trade["status"] = "CANCELLED"
            trade["result"] = "CANCELLED"
            trade["close_date"] = str(last_bar_date.date())
            add_journal_entry(state, trade, "CANCELLED", str(last_bar_date.date()), close_price=c)
            msgs.append(
                "⏹️ SETUP CANCELLED\n"
                f"Date: {last_bar_date.date()}\n"
                f"Side: {side}\n"
                f"Grade: {trade.get('grade','N/A')}\n"
                f"Mode: {entry_mode}\n"
                f"Entry: {entry:.2f}\n\n"
                + stats_text(state)
            )
            state["trade"] = {}
            return state, msgs

    if trade.get("status") == "OPEN" and feats is not None and p_up_adj is not None and last_bar_date in feats.index:
        _, chart_reason, chart_score = chart_filters(feats, last_bar_date, side)
        exit_now = False
        exit_reason = ""

        if side == "LONG":
            if p_up_adj < EXIT_P_ADJ_LONG:
                exit_now = True
                exit_reason = f"AI weakened (P_adj={p_up_adj:.4f})"
            elif chart_score < 1 and trade.get("entry_mode") == "RETEST":
                exit_now = True
                exit_reason = f"Chart weakened ({chart_reason})"
        else:
            if p_up_adj > EXIT_P_ADJ_SHORT:
                exit_now = True
                exit_reason = f"AI weakened (P_adj={p_up_adj:.4f})"
            elif chart_score < 1 and trade.get("entry_mode") == "RETEST":
                exit_now = True
                exit_reason = f"Chart weakened ({chart_reason})"

        if exit_now:
            trade["status"] = "CLOSED"
            trade["result"] = "EXIT_NOW"
            trade["close_date"] = str(last_bar_date.date())
            add_journal_entry(state, trade, "EXIT_NOW", str(last_bar_date.date()), close_price=c)
            msgs.append(
                "🟨 EXIT NOW\n"
                f"Date: {last_bar_date.date()}\n"
                f"Side: {side}\n"
                f"Grade: {trade.get('grade','N/A')}\n"
                f"Reason: {exit_reason}\n"
                f"Close: {c:.2f}\n\n"
                + stats_text(state)
            )
            state["trade"] = {}
            return state, msgs

    trade["last_processed_date"] = str(last_bar_date.date())
    state["trade"] = trade
    return state, msgs


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    state = load_state()

    xau, used_xau = fetch_ohlc_with_fallback("XAU", XAU_PROVIDERS)
    dxy, used_dxy = fetch_ohlc_with_fallback("DXY", DXY_PROVIDERS)
    us10y, used_us10y = fetch_ohlc_with_fallback("US10Y", US10Y_PROVIDERS)
    spx, used_spx = fetch_ohlc_with_fallback("SPX", SPX_PROVIDERS)
    tips, used_tips = fetch_ohlc_with_fallback("TIPS", TIPS_PROVIDERS)
    ief, used_ief = fetch_ohlc_with_fallback("IEF", IEF_PROVIDERS)

    try:
        vix, used_vix = fetch_ohlc_with_fallback("VIX", VIX_PROVIDERS)
        print("[RISK] using VIX")
    except Exception as e:
        print("[WARN] VIX unavailable:", str(e)[:250])
        vix, used_vix = None, None

    feats = make_features(xau, dxy, us10y, vix, spx, tips, ief)
    feats = compute_regime_flags(feats)

    if len(feats) < TRAIN_DAYS + 10:
        raise RuntimeError(f"Not enough aligned rows: {len(feats)} need~{TRAIN_DAYS}")

    feature_cols = ["x_ret1", "x_ret3", "x_ret5"]
    if "dxy_ret1" in feats.columns and "dxy_ret5" in feats.columns:
        feature_cols += ["dxy_ret1", "dxy_ret5"]
    if "us10y_chg1" in feats.columns and "us10y_chg5" in feats.columns:
        feature_cols += ["us10y_chg1", "us10y_chg5"]
    if "gold_real_mom" in feats.columns:
        feature_cols += ["gold_real_mom"]
    if "vix_ret1" in feats.columns and "vix_ret5" in feats.columns:
        feature_cols += ["vix_ret1", "vix_ret5"]
    if "spx_ret5" in feats.columns:
        feature_cols += ["spx_ret5"]
    if "gold_spx_mom" in feats.columns:
        feature_cols += ["gold_spx_mom"]
    if "real_f_chg5" in feats.columns:
        feature_cols += ["real_f_chg5"]
    feature_cols += ["trend_up"]

    X = feats[feature_cols]
    y = feats["y"].astype(int)

    last_day = feats.index[-1]
    X_train = X.iloc[-TRAIN_DAYS - 1:-1]
    y_train = y.iloc[-TRAIN_DAYS - 1:-1]
    X_last = X.loc[[last_day]]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    model.fit(X_train, y_train)
    p_up = float(model.predict_proba(X_last)[:, 1][0])

    real_available = "real_f_chg5" in feats.columns
    real_chg5 = float(feats.loc[last_day, "real_f_chg5"]) if real_available else float("nan")

    bias = 0.0
    bias_reason = "RealBias: N/A"
    if np.isfinite(real_chg5):
        if real_chg5 > 0:
            bias = +REAL_BIAS
            bias_reason = f"RealBias: +{REAL_BIAS:.3f} (real up)"
        elif real_chg5 < 0:
            bias = -REAL_BIAS
            bias_reason = f"RealBias: -{REAL_BIAS:.3f} (real down)"
        else:
            bias_reason = "RealBias: +0.000 (flat)"

    p_adj = clamp01(p_up + bias)

    state, daily_track_msgs = evaluate_daily_open_trade(state, xau, feats=feats, p_up_adj=p_adj)
    for m in daily_track_msgs:
        send_telegram(m)

    if ENABLE_INTRADAY_EXECUTION and state.get("trade", {}).get("status") in ("PENDING", "OPEN"):
        try:
            intraday = fetch_intraday_yahoo(YAHOO_INTRADAY_SYMBOL)
            state, intraday_msgs = process_intraday_bars(state, intraday)
            for m in intraday_msgs:
                send_telegram(m)
        except Exception as e:
            print("[INTRADAY] failed:", type(e).__name__, str(e)[:250])

    used_sources = {
        "used_xau": used_xau,
        "used_dxy": used_dxy,
        "used_us10y": used_us10y,
        "used_vix": used_vix,
        "used_spx": used_spx,
        "used_tips": used_tips,
        "used_ief": used_ief,
        "p_up": p_up,
        "p_adj": p_adj,
        "real_chg5": real_chg5,
        "real_available": real_available,
        "bias_reason": bias_reason,
    }

    state, signal_msg = build_daily_signal(state, feats, xau, used_sources)
    if SEND_NO_TRADE and SEND_DAILY_SIGNAL_ON_EVERY_RUN:
        send_telegram(signal_msg)

    save_state(state)


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:800]}"
        print("FATAL ERROR:", err)
        traceback.print_exc()
        try:
            send_telegram(f"❌ BOT ERROR\n{err}")
        except Exception as tg_err:
            print("Telegram error send failed:", type(tg_err).__name__, str(tg_err)[:250])
        raise
