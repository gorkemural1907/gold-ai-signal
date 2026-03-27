import os
import math
import time
import json
from io import StringIO
from pathlib import Path

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

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def send_telegram(text: str) -> None:
    if DEBUG_NO_TELEGRAM:
        print("[TG] DEBUG_NO_TELEGRAM=True")
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
    }


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                return default_state()
            state.setdefault("trade", {})
            state.setdefault("journal", [])
            state.setdefault("stats", default_state()["stats"])
            return state
        except Exception:
            return default_state()
    return default_state()


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# DATA FETCH
# ============================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})
FETCH_PAUSE_S = 0.5


def normalize_ohlc_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_ohlc_stooq(symbol: str, retries: int = 3, sleep_s: float = 1.5) -> pd.DataFrame:
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

            print(f"[DATA][STOOQ] {symbol} OK rows={len(df)}")
            return normalize_ohlc_index(df)

        except Exception as e:
            last_err = e
            print(f"[DATA][STOOQ] {symbol} failed: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(sleep_s)

    raise RuntimeError(f"Stooq failed for {symbol}: {type(last_err).__name__}: {last_err}")


def fetch_ohlc_yahoo(symbol: str, retries: int = 3, sleep_s: float = 1.5) -> pd.DataFrame:
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

            print(f"[DATA][YAHOO] {symbol} OK rows={len(df)}")
            return normalize_ohlc_index(df)

        except Exception as e:
            last_err = e
            print(f"[DATA][YAHOO] {symbol} failed: {type(e).__name__}: {str(e)[:200]}")
            time.sleep(sleep_s)

    raise RuntimeError(f"Yahoo failed for {symbol}: {type(last_err).__name__}: {last_err}")


def fetch_ohlc_with_fallback(name: str, providers: list[tuple[str, str]]) -> tuple[pd.DataFrame, str]:
    errors = []

    for provider, symbol in providers:
        try:
            if provider == "stooq":
                df = fetch_ohlc_stooq(symbol)
                return df, f"{provider}:{symbol}"
            if provider == "yahoo":
                df = fetch_ohlc_yahoo(symbol)
                return df, f"{provider}:{symbol}"
            raise RuntimeError(f"Unknown provider {provider}")

        except Exception as e:
            err = f"{provider}:{symbol} -> {type(e).__name__}: {str(e)[:120]}"
            print(f"[DATA][FALLBACK] {name} {err}")
            errors.append(err)

    raise RuntimeError(f"{name} fetch failed. " + " | ".join(errors))


# ============================================================
# SYMBOL MAPS
# ============================================================
XAU_PROVIDERS = [("stooq", "xauusd"), ("yahoo", "GC=F")]
DXY_PROVIDERS = [("stooq", "usd_i"), ("stooq", "usdidx"), ("yahoo", "DX-Y.NYB")]
US10Y_PROVIDERS = [("stooq", "10yusy.b"), ("stooq", "us10y"), ("yahoo", "^TNX")]
VIX_PROVIDERS = [("stooq", "vi.f"), ("stooq", "vix"), ("yahoo", "^VIX")]
SPX_PROVIDERS = [("stooq", "^spx"), ("yahoo", "^GSPC")]
TIPS_PROVIDERS = [("stooq", "tip.us"), ("yahoo", "TIP")]
IEF_PROVIDERS = [("stooq", "ief.us"), ("yahoo", "IEF")]

# ============================================================
# STRATEGY CONFIG
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

ACCOUNT_USD = 1000.0
MAX_RISK_PCT = 2.0
MAX_RISK_USD = ACCOUNT_USD * (MAX_RISK_PCT / 100.0)

LOT_OZ = 100.0
MIN_LOT = 0.01
LOT_STEP = 0.01

SPREAD_BPS = 10
SPREAD = SPREAD_BPS / 10000.0
HALF_SPREAD = SPREAD / 2.0

STRUCT_LOOKBACK = 5
CLOSE_NEAR_EXTREME_PCT = 0.40
WICK_TO_BODY_FAKE = 1.8
SQUEEZE_LOOKBACK = 20
SQUEEZE_RATIO_MAX = 0.90
VOL_EXPANSION_MIN_DELTA = 0.02
VOL_EXPANSION_RATIO_MIN = 0.92
MIN_CHART_SCORE_TO_TRADE = 2

EXIT_P_ADJ_LONG = 0.50
EXIT_P_ADJ_SHORT = 0.50

# ============================================================
# ENTRY UPGRADE
# ============================================================
ENTRY_MODE = "RETEST"   # "BREAKOUT" or "RETEST"

LATE_ENTRY_ATR_FRACTION = 0.25
RETEST_TOL_PCT = 0.0010
RETEST_REJECTION_BUFFER_PCT = 0.0003

# Yeni: bounce / rejection kalitesi
MIN_REJECTION_BODY_PCT = 0.0015      # %0.15
MIN_BOUNCE_FROM_LEVEL_ATR = 0.10     # en az 0.10 ATR rejection
MAX_CLOSE_AWAY_FROM_LEVEL_ATR = 0.35 # close çok uzak kaçtıysa chase say

# ============================================================
# HELPERS
# ============================================================
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


def clamp01(x: float) -> float:
    return float(min(0.999, max(0.001, x)))


def round_down(x: float, step: float) -> float:
    return math.floor(x / step) * step if step > 0 else x


def lot_from_risk(stop_distance: float, risk_usd: float) -> float:
    if stop_distance <= 0:
        return 0.0
    return risk_usd / (stop_distance * LOT_OZ)


def late_entry_filter(
    side: str,
    close_price: float,
    breakout_level: float,
    atr: float,
) -> tuple[bool, str, float]:
    if atr <= 0 or not np.isfinite(atr):
        return False, "", 0.0

    if side == "LONG":
        move = max(0.0, close_price - breakout_level)
    elif side == "SHORT":
        move = max(0.0, breakout_level - close_price)
    else:
        return False, "", 0.0

    limit = LATE_ENTRY_ATR_FRACTION * atr
    is_late = move > limit
    reason = (
        f"late entry: move {move:.2f} > {LATE_ENTRY_ATR_FRACTION:.2f}*ATR ({limit:.2f})"
        if is_late else ""
    )
    return is_late, reason, move


def build_trade_setup(
    side: str,
    y_high: float,
    y_low: float,
    atr: float,
) -> tuple[float, float, float, float]:
    if side == "LONG":
        break_level = y_high
        entry = y_high * (1.0 + HALF_SPREAD)
        sl = entry - STOP_ATR_MULT * atr
        tp = entry + RR * (STOP_ATR_MULT * atr)
    elif side == "SHORT":
        break_level = y_low
        entry = y_low * (1.0 - HALF_SPREAD)
        sl = entry + STOP_ATR_MULT * atr
        tp = entry - RR * (STOP_ATR_MULT * atr)
    else:
        break_level = float("nan")
        entry = float("nan")
        sl = float("nan")
        tp = float("nan")

    return entry, sl, tp, break_level


def calc_mfe_mae(side: str, entry: float, high: float, low: float) -> tuple[float, float]:
    if side == "SHORT":
        mfe = max(0.0, entry - low)
        mae = max(0.0, high - entry)
    else:
        mfe = max(0.0, high - entry)
        mae = max(0.0, entry - low)
    return mfe, mae


def update_running_extremes(trade: dict, side: str, high: float, low: float, entry: float) -> dict:
    mfe_today, mae_today = calc_mfe_mae(side, entry, high, low)
    trade["max_favorable"] = max(float(trade.get("max_favorable", 0.0) or 0.0), mfe_today)
    trade["max_adverse"] = max(float(trade.get("max_adverse", 0.0) or 0.0), mae_today)
    return trade


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


def refresh_stats(state: dict) -> None:
    journal = state.get("journal", [])
    total = 0
    wins = 0
    losses = 0
    breakeven = 0
    exit_now = 0
    cancelled = 0
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


def add_journal_entry(
    state: dict,
    trade: dict,
    result: str,
    close_date: str,
    close_price: float | None = None,
) -> None:
    journal = state.setdefault("journal", [])

    entry = float(trade.get("entry", 0.0) or 0.0)
    stop_dist = float(trade.get("stop_dist", 0.0) or 0.0)
    r_mult = compute_r_result(result, entry, stop_dist, close_price)

    row = {
        "created_date": trade.get("created_date"),
        "open_date": trade.get("open_date"),
        "close_date": close_date,
        "side": trade.get("side"),
        "entry_mode": trade.get("entry_mode"),
        "entry": round(entry, 4),
        "sl": round(float(trade.get("sl_initial", trade.get("sl", 0.0)) or 0.0), 4),
        "tp": round(float(trade.get("tp", 0.0) or 0.0), 4),
        "stop_dist": round(stop_dist, 4),
        "result": result,
        "close_price": round(float(close_price), 4) if close_price is not None else None,
        "r_mult": round(r_mult, 4),
        "max_favorable": round(float(trade.get("max_favorable", 0.0) or 0.0), 4),
        "max_adverse": round(float(trade.get("max_adverse", 0.0) or 0.0), 4),
    }

    journal.append(row)
    state["journal"] = journal[-500:]
    refresh_stats(state)


def stats_text(state: dict) -> str:
    s = state.get("stats", {})
    return (
        f"Trades: {s.get('total', 0)}\n"
        f"Wins: {s.get('wins', 0)} | Losses: {s.get('losses', 0)} | BE: {s.get('breakeven', 0)} | ExitNow: {s.get('exit_now', 0)}\n"
        f"WinRate: {float(s.get('winrate', 0.0)):.2f}%\n"
        f"Avg R: {float(s.get('avg_r', 0.0)):.2f}\n"
        f"Expectancy: {float(s.get('expectancy_r', 0.0)):.2f}R"
    )


def rejection_filter(
    side: str,
    o: float,
    h: float,
    l: float,
    c: float,
    break_level: float,
    atr: float,
) -> tuple[bool, str]:
    """
    Gecikmiş breakdown chase'i azaltmak için:
    - level'e yakın temas
    - rejection yönlü close
    - yeterli body / bounce
    - close levelden aşırı uzak değil
    """
    if atr <= 0 or not np.isfinite(atr):
        return False, "invalid ATR"

    body_abs = abs(c - o)
    min_body_abs = MIN_REJECTION_BODY_PCT * break_level
    max_close_away = MAX_CLOSE_AWAY_FROM_LEVEL_ATR * atr
    min_bounce_abs = MIN_BOUNCE_FROM_LEVEL_ATR * atr

    if side == "SHORT":
        touched = h >= break_level * (1.0 - RETEST_TOL_PCT)
        bearish_close = c <= break_level * (1.0 - RETEST_REJECTION_BUFFER_PCT)
        close_not_too_far = (break_level - c) <= max_close_away
        bounce_size = max(0.0, h - c)
        enough_bounce = bounce_size >= min_bounce_abs
        enough_body = body_abs >= min_body_abs

        ok = touched and bearish_close and close_not_too_far and enough_bounce and enough_body
        reason = (
            f"touched={touched}, bearish_close={bearish_close}, close_not_too_far={close_not_too_far}, "
            f"enough_bounce={enough_bounce}, enough_body={enough_body}"
        )
        return ok, reason

    if side == "LONG":
        touched = l <= break_level * (1.0 + RETEST_TOL_PCT)
        bullish_close = c >= break_level * (1.0 + RETEST_REJECTION_BUFFER_PCT)
        close_not_too_far = (c - break_level) <= max_close_away
        bounce_size = max(0.0, c - l)
        enough_bounce = bounce_size >= min_bounce_abs
        enough_body = body_abs >= min_body_abs

        ok = touched and bullish_close and close_not_too_far and enough_bounce and enough_body
        reason = (
            f"touched={touched}, bullish_close={bullish_close}, close_not_too_far={close_not_too_far}, "
            f"enough_bounce={enough_bounce}, enough_body={enough_body}"
        )
        return ok, reason

    return False, "unknown side"


# ============================================================
# FEATURES
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
    df = df[~df.index.duplicated(keep="last")]

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

    atr14 = compute_atr(xau_ohlc, ATR_LEN).rename("atr14")
    atr20 = compute_atr(xau_ohlc, ATR_SLOW_LEN).rename("atr20")
    df = df.join(atr14, how="left")
    df = df.join(atr20, how="left")
    df = df[~df.index.duplicated(keep="last")]

    df["ma50"] = df["xau"].rolling(50).mean()
    df["trend_up"] = (df["xau"] > df["ma50"]).astype(int)

    h = df["high_xau"]
    l = df["low_xau"]
    o = df["open_xau"]
    c = df["xau"]

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


def chart_filters(feats: pd.DataFrame, dt: pd.Timestamp, side: str) -> tuple[bool, str, int]:
    r = feats.loc[dt]
    score = 0
    reasons = []

    if side == "LONG":
        structure_ok = (int(r["trend_up"]) == 1) and (int(r["hh"]) == 1 or int(r["hl"]) == 1)
        close_quality = float(r["close_pos"]) >= (1.0 - CLOSE_NEAR_EXTREME_PCT)
        fake_breakout = np.isfinite(r["upper_wick_body"]) and float(r["upper_wick_body"]) > WICK_TO_BODY_FAKE
        wick_ok = (not np.isfinite(r["upper_wick_body"])) or float(r["upper_wick_body"]) <= WICK_TO_BODY_FAKE
        squeeze_bonus = int(r["squeeze_on"]) == 1
        vol_expansion_bonus = int(r["vol_expansion"]) == 1

        if structure_ok:
            score += 1
        else:
            reasons.append("weak market structure")
        if close_quality:
            score += 1
        else:
            reasons.append("close not near high")
        if wick_ok:
            score += 1
        else:
            reasons.append("upper wick too large / fake breakout")
        if squeeze_bonus:
            score += 1
        if vol_expansion_bonus:
            score += 1

        passed = (score >= MIN_CHART_SCORE_TO_TRADE) and (not fake_breakout)

    else:
        structure_ok = (int(r["trend_up"]) == 0) and (int(r["ll"]) == 1 or int(r["lh"]) == 1)
        close_quality = float(r["close_pos"]) <= CLOSE_NEAR_EXTREME_PCT
        fake_breakout = np.isfinite(r["lower_wick_body"]) and float(r["lower_wick_body"]) > WICK_TO_BODY_FAKE
        wick_ok = (not np.isfinite(r["lower_wick_body"])) or float(r["lower_wick_body"]) <= WICK_TO_BODY_FAKE
        squeeze_bonus = int(r["squeeze_on"]) == 1
        vol_expansion_bonus = int(r["vol_expansion"]) == 1

        if structure_ok:
            score += 1
        else:
            reasons.append("weak market structure")
        if close_quality:
            score += 1
        else:
            reasons.append("close not near low")
        if wick_ok:
            score += 1
        else:
            reasons.append("lower wick too large / fake breakout")
        if squeeze_bonus:
            score += 1
        if vol_expansion_bonus:
            score += 1

        passed = (score >= MIN_CHART_SCORE_TO_TRADE) and (not fake_breakout)

    reason = "ok" if passed else "; ".join(reasons[:3]) if reasons else "score too low"
    return passed, reason, score


# ============================================================
# TRACKING / MANAGEMENT
# ============================================================
def evaluate_tracking(
    state: dict,
    xau_ohlc: pd.DataFrame,
    feats: pd.DataFrame | None = None,
    p_up_adj: float | None = None,
) -> tuple[dict, list[str]]:
    msgs: list[str] = []

    if len(xau_ohlc) < 3:
        return state, msgs

    last_bar_date = xau_ohlc.index[-1]
    bar = xau_ohlc.iloc[-1]
    o = float(bar["Open"])
    h = float(bar["High"])
    l = float(bar["Low"])
    c = float(bar["Close"])

    t = state.get("trade")
    if not isinstance(t, dict) or not t:
        return state, msgs

    if t.get("last_processed_date") == str(last_bar_date.date()):
        return state, msgs

    status = t.get("status")
    side = t.get("side")
    entry = float(t.get("entry", 0.0) or 0.0)
    sl = float(t.get("sl", 0.0) or 0.0)
    tp = float(t.get("tp", 0.0) or 0.0)
    valid_until = t.get("valid_until")
    stop_dist = float(t.get("stop_dist", 0.0) or 0.0)
    break_level = float(t.get("break_level", 0.0) or 0.0)
    entry_mode = str(t.get("entry_mode", "BREAKOUT") or "BREAKOUT")

    t = update_running_extremes(t, side, h, l, entry)
    mfe_today, mae_today = calc_mfe_mae(side, entry, h, l)

    if status in ("PENDING", "OPEN"):
        msgs.append(
            "📊 DAILY UPDATE\n"
            f"Date: {last_bar_date.date()}\n"
            f"Side: {side}\n"
            f"Status: {status}\n"
            f"Mode: {entry_mode}\n"
            f"Entry: {entry:.2f}\n"
            f"SL: {sl:.2f}\n"
            f"TP: {tp:.2f}\n"
            f"Today High: {h:.2f}\n"
            f"Today Low: {l:.2f}\n"
            f"Today Close: {c:.2f}\n"
            f"MFE(today): {mfe_today:.2f}\n"
            f"MAE(today): {mae_today:.2f}\n"
            f"MFE(max): {float(t.get('max_favorable', 0.0)):.2f}\n"
            f"MAE(max): {float(t.get('max_adverse', 0.0)):.2f}"
        )

    if status == "PENDING":
        triggered = False
        trigger_reason = ""

        if entry_mode == "BREAKOUT":
            if side == "LONG" and h >= entry:
                triggered = True
                trigger_reason = "breakout touched entry"
            elif side == "SHORT" and l <= entry:
                triggered = True
                trigger_reason = "breakout touched entry"

        elif entry_mode == "RETEST":
            atr_for_trigger = stop_dist / STOP_ATR_MULT if STOP_ATR_MULT > 0 else 0.0
            reject_ok, reject_reason = rejection_filter(
                side=side,
                o=o,
                h=h,
                l=l,
                c=c,
                break_level=break_level,
                atr=atr_for_trigger,
            )

            if side == "LONG":
                touched_entry = h >= entry
                if reject_ok and touched_entry:
                    triggered = True
                    trigger_reason = f"retest + bullish rejection | {reject_reason}"

            elif side == "SHORT":
                touched_entry = l <= entry
                if reject_ok and touched_entry:
                    triggered = True
                    trigger_reason = f"retest + bearish rejection | {reject_reason}"

        if triggered:
            t["status"] = "OPEN"
            t["open_date"] = str(last_bar_date.date())
            t["be_moved"] = False
            msgs.append(
                "✅ TRADE TRIGGERED\n"
                f"Date: {last_bar_date.date()}\n"
                f"Side: {side}\n"
                f"Mode: {entry_mode}\n"
                f"Trigger: {trigger_reason}\n"
                f"Entry: {entry:.2f}\n"
                f"SL: {sl:.2f}\n"
                f"TP: {tp:.2f}"
            )
        elif valid_until and str(last_bar_date.date()) >= valid_until:
            t["status"] = "CANCELLED"
            t["result"] = "CANCELLED"
            t["close_date"] = str(last_bar_date.date())
            add_journal_entry(state, t, "CANCELLED", str(last_bar_date.date()), close_price=c)
            msgs.append(
                "⏹️ SETUP CANCELLED\n"
                f"Date: {last_bar_date.date()}\n"
                f"Reason: Not triggered before next NY close\n"
                f"Side: {side}\n"
                f"Mode: {entry_mode}\n"
                f"Entry: {entry:.2f}\n\n"
                + stats_text(state)
            )

    if t.get("status") == "OPEN":
        be_moved = bool(t.get("be_moved", False))
        current_sl = float(t.get("sl", sl) or sl)

        if (not be_moved) and stop_dist > 0:
            if side == "LONG" and h >= entry + stop_dist:
                t["sl"] = entry
                t["be_moved"] = True
                current_sl = entry
                msgs.append(
                    "🟦 MOVE STOP TO BREAK-EVEN\n"
                    f"Date: {last_bar_date.date()}\n"
                    f"Side: {side}\n"
                    f"New SL: {entry:.2f}"
                )
            elif side == "SHORT" and l <= entry - stop_dist:
                t["sl"] = entry
                t["be_moved"] = True
                current_sl = entry
                msgs.append(
                    "🟦 MOVE STOP TO BREAK-EVEN\n"
                    f"Date: {last_bar_date.date()}\n"
                    f"Side: {side}\n"
                    f"New SL: {entry:.2f}"
                )

        if feats is not None and p_up_adj is not None and last_bar_date in feats.index:
            chart_ok, chart_reason, chart_score = chart_filters(feats, last_bar_date, side)
            exit_now = False
            exit_reason = ""

            if side == "LONG":
                if p_up_adj < EXIT_P_ADJ_LONG:
                    exit_now = True
                    exit_reason = f"AI weakened (P_adj={p_up_adj:.4f})"
                elif chart_score < 1:
                    exit_now = True
                    exit_reason = f"Chart weakened ({chart_reason})"
            elif side == "SHORT":
                if p_up_adj > EXIT_P_ADJ_SHORT:
                    exit_now = True
                    exit_reason = f"AI weakened (P_adj={p_up_adj:.4f})"
                elif chart_score < 1:
                    exit_now = True
                    exit_reason = f"Chart weakened ({chart_reason})"

            if exit_now:
                t["status"] = "CLOSED"
                t["result"] = "EXIT_NOW"
                t["close_date"] = str(last_bar_date.date())
                add_journal_entry(state, t, "EXIT_NOW", str(last_bar_date.date()), close_price=c)
                msgs.append(
                    "🟨 EXIT NOW\n"
                    f"Date: {last_bar_date.date()}\n"
                    f"Side: {side}\n"
                    f"Reason: {exit_reason}\n"
                    f"Close: {c:.2f}\n\n"
                    + stats_text(state)
                )

        if t.get("status") == "OPEN":
            current_sl = float(t.get("sl", sl) or sl)

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

                t["status"] = "CLOSED"
                t["result"] = result
                t["close_date"] = str(last_bar_date.date())

                close_price_for_journal = tp if result == "TAKE_PROFIT" else current_sl
                add_journal_entry(state, t, result, str(last_bar_date.date()), close_price=close_price_for_journal)

                emoji = "🟩" if result == "TAKE_PROFIT" else ("⬜" if result == "BREAKEVEN" else "🟥")
                label = "TAKE PROFIT" if result == "TAKE_PROFIT" else ("BREAK-EVEN" if result == "BREAKEVEN" else "STOP")

                msgs.append(
                    f"{emoji} TRADE CLOSED ({label})\n"
                    f"Date: {last_bar_date.date()}\n"
                    f"Side: {side}\n"
                    f"Entry: {entry:.2f}\n"
                    f"SL: {current_sl:.2f}\n"
                    f"TP: {tp:.2f}\n"
                    f"MFE(max): {float(t.get('max_favorable', 0.0)):.2f}\n"
                    f"MAE(max): {float(t.get('max_adverse', 0.0)):.2f}\n\n"
                    + stats_text(state)
                )

    if t.get("status") in ("CLOSED", "CANCELLED"):
        state["trade"] = {}
    else:
        t["last_processed_date"] = str(last_bar_date.date())
        state["trade"] = t

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
        print("[WARN] VIX unavailable:", str(e)[:300])
        vix, used_vix = None, None

    feats = make_features(xau, dxy, us10y, vix, spx, tips, ief)

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
        ("clf", LogisticRegression(max_iter=2000))
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

    state, track_msgs = evaluate_tracking(state, xau, feats=feats, p_up_adj=p_adj)
    for m in track_msgs:
        send_telegram(m)

    trade = state.get("trade", {})
    has_active_trade = trade.get("status") in ("PENDING", "OPEN")

    trend_up = int(feats.loc[last_day, "trend_up"]) == 1
    trend_txt = "UP" if trend_up else "DOWN"

    side = "NO-TRADE"
    if (p_adj > TH_LONG) and ((p_adj - 0.5) >= MIN_EDGE) and trend_up:
        side = "LONG"
    elif (p_adj < TH_SHORT) and ((0.5 - p_adj) >= MIN_EDGE) and (not trend_up):
        side = "SHORT"

    chart_score = 0
    chart_reason = ""
    if side in ("LONG", "SHORT"):
        chart_ok, chart_reason, chart_score = chart_filters(feats, last_day, side)
        if not chart_ok:
            side = "NO-TRADE"

    atr = float(feats.loc[last_day, "atr14"])
    ybar = xau.iloc[-2]
    y_high = float(ybar["High"])
    y_low = float(ybar["Low"])
    close_now = float(xau["Close"].iloc[-1])

    late_filter_reason = ""
    late_move = 0.0

    if side == "LONG":
        entry, sl, tp, break_level = build_trade_setup("LONG", y_high, y_low, atr)
        is_late, late_filter_reason, late_move = late_entry_filter("LONG", close_now, y_high, atr)
        if is_late:
            side = "NO-TRADE"

    elif side == "SHORT":
        entry, sl, tp, break_level = build_trade_setup("SHORT", y_high, y_low, atr)
        is_late, late_filter_reason, late_move = late_entry_filter("SHORT", close_now, y_low, atr)
        if is_late:
            side = "NO-TRADE"

    else:
        entry = float(close_now)
        sl = float("nan")
        tp = float("nan")
        break_level = float("nan")

    stop_dist = STOP_ATR_MULT * atr
    lot_raw = lot_from_risk(stop_dist, MAX_RISK_USD)
    lot = round_down(lot_raw, LOT_STEP)

    trade_skipped = False
    skip_reason = ""
    if side in ("LONG", "SHORT") and lot < MIN_LOT:
        real_risk_usd = MIN_LOT * stop_dist * LOT_OZ
        trade_skipped = True
        skip_reason = f"Risk too large for account. Min lot risk=${real_risk_usd:.2f} > max ${MAX_RISK_USD:.2f}"
        lot = MIN_LOT

    risk_gate = "VIX" if used_vix is not None else "SPX_PROXY"

    msg = (
        f"GOLD AI SIGNAL (XAUUSD)\n\n"
        f"Date: {last_day.date()}\n\n"
        f"Sources:\n"
        f"XAU: {used_xau}\n"
        f"DXY: {used_dxy}\n"
        f"US10Y: {used_us10y}\n"
        f"VIX: {used_vix if used_vix is not None else 'NONE'}\n"
        f"SPX: {used_spx}\n"
        f"TIPS: {used_tips}\n"
        f"IEF: {used_ief}\n\n"
        f"P(up): {p_up:.4f}\n"
        f"P_adj: {p_adj:.4f} ({bias_reason})\n"
        f"Trend(>MA50): {trend_txt}\n"
        f"RiskGate: {risk_gate}\n"
        f"ChartScore: {chart_score}/5\n"
        f"EntryMode: {ENTRY_MODE}\n"
        f"HasActiveTrade: {has_active_trade}\n"
    )

    if real_available:
        msg += f"RealFactor chg5: {real_chg5:+.4f}\n"
    if "gold_real_mom" in feats.columns:
        msg += f"GoldRealMom5: {float(feats.loc[last_day, 'gold_real_mom']):+.4f}\n"
    if "gold_spx_mom" in feats.columns:
        msg += f"GoldSPXMom5: {float(feats.loc[last_day, 'gold_spx_mom']):+.4f}\n"
    if "vol_expansion" in feats.columns:
        msg += f"VolExpansion: {int(feats.loc[last_day, 'vol_expansion'])}\n"
    if chart_reason:
        msg += f"ChartFilter: {chart_reason}\n"
    if late_filter_reason:
        msg += f"LateFilter: {late_filter_reason}\n"

    msg += (
        f"\nSignal: {side}\n\n"
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
        f"Suggested lot: {lot:.2f}\n\n"
        f"Performance Snapshot:\n{stats_text(state)}\n"
    )

    if has_active_trade:
        msg += "\nNote: Active trade exists, new setup will NOT be stored.\n"

    if trade_skipped:
        msg += f"\nTrade: SKIPPED\nReason: {skip_reason}\n"

    print(msg)

    if side != "NO-TRADE" or SEND_NO_TRADE:
        send_telegram(msg)

    if side in ("LONG", "SHORT") and not trade_skipped and not has_active_trade:
        state["trade"] = {
            "status": "PENDING",
            "created_date": str(last_day.date()),
            "valid_until": str(last_day.date()),
            "side": side,
            "entry_mode": ENTRY_MODE,
            "entry": float(entry),
            "break_level": float(break_level),
            "sl": float(sl),
            "sl_initial": float(sl),
            "tp": float(tp),
            "lot": float(lot),
            "stop_dist": float(stop_dist),
            "be_moved": False,
            "max_favorable": 0.0,
            "max_adverse": 0.0,
            "last_processed_date": trade.get("last_processed_date"),
        }

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
            print("Telegram error message sent.")
        except Exception as tg_err:
            print("Telegram error send failed:", type(tg_err).__name__, str(tg_err)[:300])

        raise
