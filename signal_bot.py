import os
import math
import time
import json
import requests
import pandas as pd
import numpy as np
from io import StringIO
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# Telegram
# ============================================================
DEBUG_NO_TELEGRAM = False
SEND_NO_TRADE = True

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

def send_telegram(text: str) -> None:
    if DEBUG_NO_TELEGRAM:
        print(text)
        return
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"Telegram HTTP {r.status_code}")

def safe_symbol(s: str | None) -> str:
    return s.replace(".", "_") if s else "NONE"

# ============================================================
# Persistent state
# ============================================================
STATE_DIR = Path(".bot_state")
STATE_PATH = STATE_DIR / "state.json"

def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

# ============================================================
# Data fetch (Stooq + Yahoo fallback)
# ============================================================
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; GoldAISignal/1.0)"
})

FETCH_PAUSE_S = 0.6

def stooq_url(symbol: str) -> str:
    return f"https://stooq.com/q/d/l/?s={symbol}&i=d"

def fetch_ohlc_stooq(symbol: str, retries: int = 3, sleep_s: float = 2.0) -> pd.DataFrame:
    url = stooq_url(symbol)
    last_err = None

    for _ in range(retries):
        try:
            time.sleep(FETCH_PAUSE_S)
            resp = SESSION.get(url, timeout=25)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")

            txt = resp.text or ""
            if "Exceeded the daily hits limit" in txt:
                raise RuntimeError("Stooq quota exceeded")
            if len(txt) < 50:
                raise RuntimeError("Empty/short CSV")

            df = pd.read_csv(StringIO(txt))
            needed = {"Date", "Open", "High", "Low", "Close"}
            if not needed.issubset(set(df.columns)):
                raise RuntimeError(f"Missing columns: got={list(df.columns)[:20]}")

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").set_index("Date")
            df = df[["Open", "High", "Low", "Close"]].dropna()

            if len(df) < 300:
                raise RuntimeError(f"Too few rows: {len(df)}")

            return df

        except Exception as e:
            last_err = e
            time.sleep(sleep_s)

    raise RuntimeError(f"Stooq fetch failed for {symbol}: {type(last_err).__name__}: {str(last_err)[:200]}")

def fetch_ohlc_yahoo(symbol: str, retries: int = 3, sleep_s: float = 2.0) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=10y&interval=1d&includePrePost=false"
    last_err = None

    for _ in range(retries):
        try:
            time.sleep(FETCH_PAUSE_S)
            resp = SESSION.get(url, timeout=25)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")

            data = resp.json()
            result = data["chart"]["result"][0]
            ts = result["timestamp"]
            q = result["indicators"]["quote"][0]

            df = pd.DataFrame({
                "Date": pd.to_datetime(ts, unit="s"),
                "Open": q["open"],
                "High": q["high"],
                "Low": q["low"],
                "Close": q["close"],
            })
            df = df.dropna()
            df = df.sort_values("Date").set_index("Date")

            if len(df) < 300:
                raise RuntimeError(f"Too few rows: {len(df)}")

            return df[["Open", "High", "Low", "Close"]]

        except Exception as e:
            last_err = e
            time.sleep(sleep_s)

    raise RuntimeError(f"Yahoo fetch failed for {symbol}: {type(last_err).__name__}: {str(last_err)[:200]}")

def fetch_ohlc(symbol: str, yahoo_symbol: str | None = None) -> pd.DataFrame:
    try:
        return fetch_ohlc_stooq(symbol)
    except Exception as e:
        msg = str(e).lower()
        if yahoo_symbol and (
            "quota exceeded" in msg
            or "empty/short csv" in msg
            or "http" in msg
            or "missing columns" in msg
        ):
            return fetch_ohlc_yahoo(yahoo_symbol)
        raise

def fetch_optional(symbols: list[tuple[str, str | None]], label: str) -> tuple[pd.DataFrame | None, str | None]:
    for stooq_sym, yahoo_sym in symbols:
        try:
            df = fetch_ohlc(stooq_sym, yahoo_sym)
            used = stooq_sym if yahoo_sym is None else f"{stooq_sym}|{yahoo_sym}"
            return df, used
        except Exception:
            pass
    print(f"[WARN] {label} missing")
    return None, None

def fetch_required_xau() -> pd.DataFrame:
    df = fetch_ohlc("xauusd", "XAUUSD=X")

    closes = df["Close"].astype(float).dropna()
    if len(closes) < 180:
        raise RuntimeError(f"XAU sanity needs more history: {len(closes)}")

    last_close = float(closes.iloc[-1])
    prev_close = float(closes.iloc[-2])

    med60 = float(closes.iloc[-60:].median())
    lo = 0.50 * med60
    hi = 2.00 * med60
    if not (lo <= last_close <= hi):
        raise RuntimeError(
            f"XAU sanity fail: last_close={last_close:.2f} not in [{lo:.2f},{hi:.2f}]"
        )

    if prev_close > 0:
        jump = abs(last_close / prev_close - 1.0)
        if jump > 0.25:
            raise RuntimeError(f"XAU sanity fail: 1d jump={jump:.2%}")

    return df

# ============================================================
# Symbols
# ============================================================
XAU_SYMBOL = "xauusd|XAUUSD=X"
DXY_CANDIDATES = [("dx.f", None), ("usd_i", "DX-Y.NYB"), ("usdidx", "DX-Y.NYB")]
US10Y_CANDIDATES = [("10yusy.b", "^TNX"), ("us10y", "^TNX")]
VIX_CANDIDATES = [("vi.f", "^VIX"), ("vix", "^VIX"), ("^vix", "^VIX"), ("vix.f", "^VIX")]
SPX_CANDIDATES = [("^spx", "^GSPC")]
TIPS_CANDIDATES = [("tip.us", "TIP")]
IEF_CANDIDATES = [("ief.us", "IEF")]

# ============================================================
# Strategy config
# ============================================================
TRAIN_DAYS = 252 * 5

TH_LONG = 0.62
TH_SHORT = 0.38
MIN_EDGE = 0.12

REQUIRE_VIX_FOR_TRADE = False
SPX_RISK_PROXY_ON = True
SPX_RET5_ABS_MAX = 0.03

REAL_FACTOR_LOOKBACK = 5
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

# chart / vol thresholds
STRUCT_LOOKBACK = 5
CLOSE_NEAR_EXTREME_PCT = 0.30
WICK_TO_BODY_FAKE = 1.2
SQUEEZE_LOOKBACK = 20
SQUEEZE_RATIO_MAX = 0.90
VOL_EXPANSION_MIN_DELTA = 0.03
VOL_EXPANSION_RATIO_MIN = 0.95

# management thresholds
EXIT_P_ADJ_LONG = 0.52
EXIT_P_ADJ_SHORT = 0.48

# ============================================================
# Helpers
# ============================================================
def compute_atr(ohlc: pd.DataFrame, n: int) -> pd.Series:
    prev_close = ohlc["Close"].shift(1)
    tr = pd.concat(
        [
            (ohlc["High"] - ohlc["Low"]).rename("hl"),
            (ohlc["High"] - prev_close).abs().rename("hc"),
            (ohlc["Low"] - prev_close).abs().rename("lc"),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean()

def round_down(x: float, step: float) -> float:
    return math.floor(x / step) * step if step > 0 else x

def lot_from_risk(stop_distance_usd: float, risk_usd: float) -> float:
    if stop_distance_usd <= 0:
        return 0.0
    return risk_usd / (stop_distance_usd * LOT_OZ)

def clamp01(x: float) -> float:
    return float(min(0.999, max(0.001, x)))

# ============================================================
# Features
# ============================================================
def make_features(
    xau_ohlc: pd.DataFrame,
    dxy_close: pd.Series | None,
    us10y_close: pd.Series | None,
    vix_close: pd.Series | None,
    spx_close: pd.Series | None,
    tips_close: pd.Series | None,
    ief_close: pd.Series | None,
) -> pd.DataFrame:
    base = [xau_ohlc["Close"].rename("xau")]
    if dxy_close is not None:
        base.append(dxy_close.rename("dxy"))
    if us10y_close is not None:
        base.append(us10y_close.rename("us10y"))
    if vix_close is not None:
        base.append(vix_close.rename("vix"))
    if spx_close is not None:
        base.append(spx_close.rename("spx"))
    if tips_close is not None:
        base.append(tips_close.rename("tips"))
    if ief_close is not None:
        base.append(ief_close.rename("ief"))

    df = pd.concat(base, axis=1).dropna()

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
        df["real_f_chg5"] = df["real_factor"].diff(REAL_FACTOR_LOOKBACK)

    atr14 = compute_atr(xau_ohlc, ATR_LEN).rename("atr14")
    atr20 = compute_atr(xau_ohlc, ATR_SLOW_LEN).rename("atr20")
    df = df.join(atr14, how="left")
    df = df.join(atr20, how="left")

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

        passed = structure_ok and close_quality and (not fake_breakout)

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

        passed = structure_ok and close_quality and (not fake_breakout)

    reason = "ok" if passed else "; ".join(reasons[:3])
    return passed, reason, score

# ============================================================
# Tracking / management
# ============================================================
def evaluate_tracking(state: dict, xau_ohlc: pd.DataFrame, feats: pd.DataFrame | None = None,
                      p_up_adj: float | None = None) -> tuple[dict, list[str]]:
    msgs: list[str] = []
    if len(xau_ohlc) < 3:
        return state, msgs

    last_bar_date = xau_ohlc.index[-1]
    bar = xau_ohlc.iloc[-1]
    h = float(bar["High"])
    l = float(bar["Low"])
    c = float(bar["Close"])

    t = state.get("trade")
    if not isinstance(t, dict):
        return state, msgs

    if t.get("last_processed_date") == str(last_bar_date.date()):
        return state, msgs

    status = t.get("status")
    side = t.get("side")
    entry = float(t.get("entry", 0) or 0)
    sl = float(t.get("sl", 0) or 0)
    tp = float(t.get("tp", 0) or 0)
    valid_until = t.get("valid_until")
    stop_dist = float(t.get("stop_dist", 0) or 0)

    if status == "PENDING":
        triggered = False
        if side == "LONG" and h >= entry:
            triggered = True
        elif side == "SHORT" and l <= entry:
            triggered = True

        if triggered:
            t["status"] = "OPEN"
            t["open_date"] = str(last_bar_date.date())
            t["be_moved"] = False
            msgs.append(
                "✅ TRADE TRIGGERED\n"
                f"Date: {last_bar_date.date()}\n"
                f"Side: {side}\n"
                f"Entry: {entry:.2f}\n"
                f"SL: {sl:.2f}\n"
                f"TP: {tp:.2f}"
            )
        elif valid_until and str(last_bar_date.date()) >= valid_until:
            t["status"] = "CANCELLED"
            msgs.append(
                "⏹️ SETUP CANCELLED\n"
                f"Date: {last_bar_date.date()}\n"
                f"Reason: Not triggered before next NY close\n"
                f"Side: {side}\n"
                f"Entry: {entry:.2f}"
            )

    if t.get("status") == "OPEN":
        be_moved = bool(t.get("be_moved", False))
        if (not be_moved) and stop_dist > 0:
            if side == "LONG" and h >= entry + stop_dist:
                t["sl"] = entry
                t["be_moved"] = True
                sl = entry
                msgs.append(
                    "🟦 MOVE STOP TO BREAK-EVEN\n"
                    f"Date: {last_bar_date.date()}\n"
                    f"Side: {side}\n"
                    f"New SL: {entry:.2f}"
                )
            elif side == "SHORT" and l <= entry - stop_dist:
                t["sl"] = entry
                t["be_moved"] = True
                sl = entry
                msgs.append(
                    "🟦 MOVE STOP TO BREAK-EVEN\n"
                    f"Date: {last_bar_date.date()}\n"
                    f"Side: {side}\n"
                    f"New SL: {entry:.2f}"
                )

        if feats is not None and p_up_adj is not None and last_bar_date in feats.index:
            chart_ok, chart_reason, _ = chart_filters(feats, last_bar_date, side)
            exit_now = False
            exit_reason = ""

            if side == "LONG":
                if p_up_adj < EXIT_P_ADJ_LONG:
                    exit_now = True
                    exit_reason = f"AI weakened (P_adj={p_up_adj:.4f})"
                elif not chart_ok:
                    exit_now = True
                    exit_reason = f"Chart weakened ({chart_reason})"
            elif side == "SHORT":
                if p_up_adj > EXIT_P_ADJ_SHORT:
                    exit_now = True
                    exit_reason = f"AI weakened (P_adj={p_up_adj:.4f})"
                elif not chart_ok:
                    exit_now = True
                    exit_reason = f"Chart weakened ({chart_reason})"

            if exit_now:
                t["status"] = "CLOSED"
                t["result"] = "EXIT_NOW"
                t["close_date"] = str(last_bar_date.date())
                msgs.append(
                    "🟨 EXIT NOW\n"
                    f"Date: {last_bar_date.date()}\n"
                    f"Side: {side}\n"
                    f"Reason: {exit_reason}\n"
                    f"Close: {c:.2f}"
                )

        if t.get("status") == "OPEN":
            hit_sl = hit_tp = False
            current_sl = float(t.get("sl", sl) or sl)
            if side == "LONG":
                hit_sl = l <= current_sl
                hit_tp = h >= tp
            elif side == "SHORT":
                hit_sl = h >= current_sl
                hit_tp = l <= tp

            if hit_sl or hit_tp:
                result = "STOP" if hit_sl else "TAKE_PROFIT"
                if hit_sl and hit_tp:
                    result = "STOP"
                t["status"] = "CLOSED"
                t["result"] = result
                t["close_date"] = str(last_bar_date.date())

                msgs.append(
                    ("🟥 TRADE CLOSED (STOP)\n" if result == "STOP" else "🟩 TRADE CLOSED (TAKE PROFIT)\n")
                    + f"Date: {last_bar_date.date()}\n"
                    + f"Side: {side}\n"
                    + f"Entry: {entry:.2f}\n"
                    + f"SL: {current_sl:.2f}\n"
                    + f"TP: {tp:.2f}"
                )

    t["last_processed_date"] = str(last_bar_date.date())
    state["trade"] = t
    return state, msgs

# ============================================================
# Main
# ============================================================
def main():
    state = load_state()
    xau = fetch_required_xau()

    dxy, used_dxy = fetch_optional(DXY_CANDIDATES, "DXY")
    us10y, used_us10y = fetch_optional(US10Y_CANDIDATES, "US10Y")
    vix, used_vix = fetch_optional(VIX_CANDIDATES, "VIX")
    spx, used_spx = fetch_optional(SPX_CANDIDATES, "SPX")
    tips, used_tips = fetch_optional(TIPS_CANDIDATES, "TIPS")
    ief, used_ief = fetch_optional(IEF_CANDIDATES, "IEF")

    feats = make_features(
        xau_ohlc=xau,
        dxy_close=dxy["Close"] if dxy is not None else None,
        us10y_close=us10y["Close"] if us10y is not None else None,
        vix_close=vix["Close"] if vix is not None else None,
        spx_close=spx["Close"] if spx is not None else None,
        tips_close=tips["Close"] if tips is not None else None,
        ief_close=ief["Close"] if ief is not None else None,
    )

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

    if len(feats) < TRAIN_DAYS + 10:
        raise RuntimeError(f"Not enough aligned rows: {len(feats)} need~{TRAIN_DAYS}")

    X = feats[feature_cols]
    y = feats["y"].astype(int)

    last_day = feats.index[-1]
    X_train = X.iloc[-TRAIN_DAYS-1:-1]
    y_train = y.iloc[-TRAIN_DAYS-1:-1]
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
    if real_available and np.isfinite(real_chg5):
        if real_chg5 > 0:
            bias = +REAL_BIAS
            bias_reason = f"RealBias: +{REAL_BIAS:.3f} (real up)"
        elif real_chg5 < 0:
            bias = -REAL_BIAS
            bias_reason = f"RealBias: -{REAL_BIAS:.3f} (real down)"
        else:
            bias_reason = "RealBias: +0.000 (flat)"
    p_up_adj = clamp01(p_up + bias)

    state, track_msgs = evaluate_tracking(state, xau, feats=feats, p_up_adj=p_up_adj)
    for m in track_msgs:
        send_telegram(m)

    trade = state.get("trade", {})
    if trade.get("status") in ("PENDING", "OPEN"):
        save_state(state)
        return

    in_uptrend = int(feats.loc[last_day, "trend_up"]) == 1
    trend_txt = "UP" if in_uptrend else "DOWN"

    vix_available = used_vix is not None
    risk_gate = "VIX" if vix_available else "SPX_PROXY"
    reason = ""

    side = None
    if REQUIRE_VIX_FOR_TRADE and not vix_available:
        side = "NO-TRADE"
        reason = "VIX missing -> quality filter"
    else:
        if (not vix_available) and SPX_RISK_PROXY_ON:
            if "spx_ret5" in feats.columns:
                spx_ret5 = float(feats.loc[last_day, "spx_ret5"])
                if np.isfinite(spx_ret5) and abs(spx_ret5) > SPX_RET5_ABS_MAX:
                    side = "NO-TRADE"
                    reason = f"SPX proxy risk: abs(spx_ret5)>{SPX_RET5_ABS_MAX:.2%}"
            else:
                side = "NO-TRADE"
                reason = "Risk data missing (no VIX, no SPX)"

    if side is None:
        side = "NO-TRADE"
        if (p_up_adj > TH_LONG) and ((p_up_adj - 0.5) >= MIN_EDGE) and in_uptrend:
            side = "LONG"
        elif (p_up_adj < TH_SHORT) and ((0.5 - p_up_adj) >= MIN_EDGE) and (not in_uptrend):
            side = "SHORT"

    chart_ok = False
    chart_reason = ""
    chart_score = 0
    if side in ("LONG", "SHORT"):
        chart_ok, chart_reason, chart_score = chart_filters(feats, last_day, side)
        if not chart_ok:
            side = "NO-TRADE"
            reason = f"Chart filter: {chart_reason}"

    ybar = xau.iloc[-2]
    y_high = float(ybar["High"])
    y_low = float(ybar["Low"])

    spread_long = y_high * HALF_SPREAD
    spread_short = y_low * HALF_SPREAD

    if side == "LONG":
        entry = y_high + spread_long
    elif side == "SHORT":
        entry = y_low - spread_short
    else:
        entry = float(xau["Close"].iloc[-1])

    atr = float(feats.loc[last_day, "atr14"])
    stop_dist = STOP_ATR_MULT * atr

    if side == "LONG":
        sl = entry - stop_dist
        tp = entry + RR * stop_dist
    elif side == "SHORT":
        sl = entry + stop_dist
        tp = entry - RR * stop_dist
    else:
        sl = tp = float("nan")

    lot_raw = lot_from_risk(stop_dist, MAX_RISK_USD)
    lot = round_down(lot_raw, LOT_STEP)

    trade_skipped = False
    skip_reason = ""
    if side in ("LONG", "SHORT") and lot < MIN_LOT:
        real_risk_usd = MIN_LOT * stop_dist * LOT_OZ
        trade_skipped = True
        skip_reason = f"Risk too large for account. Min lot risk=${real_risk_usd:.2f} > max ${MAX_RISK_USD:.2f}"
        lot = MIN_LOT

    valid_until = str(last_day.date())

    src = (
        f"Sources:\n"
        f"XAU: {safe_symbol(XAU_SYMBOL)}\n"
        f"DXY: {safe_symbol(used_dxy)}\n"
        f"US10Y: {safe_symbol(used_us10y)}\n"
        f"VIX: {safe_symbol(used_vix)}\n"
        f"SPX: {safe_symbol(used_spx)}\n"
        f"TIPS: {safe_symbol(used_tips)}\n"
        f"IEF: {safe_symbol(used_ief)}\n"
    )

    header = f"GOLD AI SIGNAL (XAUUSD)\nDate: {last_day.date()}\n"
    info = (
        src
        + f"P(up): {p_up:.4f}\n"
        + f"P_adj: {p_up_adj:.4f} ({bias_reason})\n"
        + f"Trend(>MA50): {trend_txt}\n"
        + f"RiskGate: {risk_gate}\n"
        + f"ChartScore: {chart_score}/5\n"
        + f"Spread: {SPREAD_BPS} bps (±{SPREAD_BPS/2:.1f} bps)\n"
    )
    if real_available:
        info += f"RealFactor chg{REAL_FACTOR_LOOKBACK}: {real_chg5:+.4f}\n"
    else:
        info += f"RealFactor chg{REAL_FACTOR_LOOKBACK}: N/A\n"
    if "gold_real_mom" in feats.columns:
        info += f"GoldRealMom5: {float(feats.loc[last_day, 'gold_real_mom']):+.4f}\n"
    if "gold_spx_mom" in feats.columns:
        info += f"GoldSPXMom5: {float(feats.loc[last_day, 'gold_spx_mom']):+.4f}\n"
    if "vol_expansion" in feats.columns:
        info += f"VolExpansion: {int(feats.loc[last_day, 'vol_expansion'])}\n"
    if reason:
        info += f"Filter: {reason}\n"

    if side == "NO-TRADE":
        msg = header + "\n" + info + "\nSignal: NO-TRADE"
        print(msg)
        if SEND_NO_TRADE:
            send_telegram(msg)
        save_state(state)
        return

    if trade_skipped:
        msg = (
            header + "\n"
            + info + "\n"
            + f"Signal: {side}\n"
            + "Trade: SKIPPED\n"
            + f"Reason: {skip_reason}\n\n"
            + f"Breakout Entry: {entry:.2f}\n"
            + f"Stop Loss: {sl:.2f}\n"
            + f"Take Profit: {tp:.2f}\n"
            + f"Yesterday High: {y_high:.2f}\n"
            + f"Yesterday Low: {y_low:.2f}\n"
            + f"ATR({ATR_LEN}): {atr:.2f}\n"
            + f"StopDist: {stop_dist:.2f} ({STOP_ATR_MULT:.1f}x ATR)\n"
            + f"Max risk: ${MAX_RISK_USD:.2f} ({MAX_RISK_PCT:.1f}%)\n"
            + f"Order valid until next NY close: {valid_until}\n"
        )
        print(msg)
        send_telegram(msg)
        save_state(state)
        return

    msg = (
        header + "\n"
        + info + "\n"
        + f"Signal: {side}\n\n"
        + f"Breakout Entry: {entry:.2f}\n"
        + f"Yesterday High: {y_high:.2f}\n"
        + f"Yesterday Low: {y_low:.2f}\n"
        + f"Stop Loss: {sl:.2f}\n"
        + f"Take Profit: {tp:.2f}\n"
        + f"ATR({ATR_LEN}): {atr:.2f}\n"
        + f"StopDist: {stop_dist:.2f} ({STOP_ATR_MULT:.1f}x ATR)\n"
        + f"RR: {RR:.1f}R\n\n"
        + f"Account: ${ACCOUNT_USD:.0f}\n"
        + f"Max risk: ${MAX_RISK_USD:.2f} ({MAX_RISK_PCT:.1f}%)\n"
        + f"Suggested lot: {lot:.2f}\n"
        + f"Order valid until next NY close: {valid_until}\n"
    )
    print(msg)
    send_telegram(msg)

    state["trade"] = {
        "status": "PENDING",
        "created_date": str(last_day.date()),
        "valid_until": valid_until,
        "side": side,
        "entry": float(entry),
        "sl": float(sl),
        "tp": float(tp),
        "lot": float(lot),
        "stop_dist": float(stop_dist),
        "be_moved": False,
        "last_processed_date": state.get("trade", {}).get("last_processed_date")
    }
    save_state(state)

if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        err_type = type(e).__name__
        err_msg = str(e)
        if len(err_msg) > 800:
            err_msg = err_msg[:800] + "..."
        try:
            send_telegram(f"❌ BOT ERROR\n{err_type}: {err_msg}")
        except Exception:
            pass
        print("FATAL ERROR:", err_type)
        traceback.print_exc()
        raise
