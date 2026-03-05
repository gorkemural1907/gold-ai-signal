import os
import math
import time
import requests
import pandas as pd
import numpy as np
from io import StringIO

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# Telegram & behavior
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
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=25)
        if r.status_code != 200:
            raise RuntimeError(f"Telegram HTTP {r.status_code}")
    except Exception:
        raise RuntimeError("Telegram send failed (network/api).")

# ============================================================
# Stooq download (rate-limit friendly)
# ============================================================
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (compatible; GoldAISignal/1.0; +https://github.com/)"
})

FETCH_PAUSE_S = 0.6

def stooq_url(symbol: str) -> str:
    return f"https://stooq.com/q/d/l/?s={symbol}&i=d"

def fetch_ohlc_stooq(symbol: str, retries: int = 5, sleep_s: float = 2.0) -> pd.DataFrame:
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

def fetch_optional(symbols: list[str], label: str) -> tuple[pd.DataFrame | None, str | None]:
    for sym in symbols:
        try:
            df = fetch_ohlc_stooq(sym)
            return df, sym
        except Exception:
            pass

    send_telegram(f"⚠️ {label} verisi alınamadı ({'/'.join(symbols)}). Bugün {label} features devre dışı.")
    return None, None

# ============================================================
# Symbols (fallback lists)
# ============================================================
XAU = "xauusd"  # required

DXY_CANDIDATES    = ["dx.f", "usd_i", "usdidx"]
US10Y_CANDIDATES  = ["10yusy.b", "us10y"]
SPX_CANDIDATES    = ["^spx"]
TIPS_CANDIDATES   = ["tip.us"]
IEF_CANDIDATES    = ["ief.us"]

# VIX için daha geniş fallback (Stooq bazen boş dönüyor)
VIX_CANDIDATES    = ["vi.f", "vix", "^vix", "vix.f"]

# ============================================================
# Strategy / risk config (MORE FREQUENT but still quality)
# ============================================================
TRAIN_DAYS = 252 * 5

TH_LONG = 0.62
TH_SHORT = 0.38
MIN_EDGE = 0.12

# VIX varsa kullan; yoksa SPX risk proxy ile devam
REQUIRE_VIX_FOR_TRADE = False

# SPX risk proxy eşiği (VIX yoksa devreye girer)
# abs(SPX 5g return) > 3% ise NO-TRADE
SPX_RISK_PROXY_ON = True
SPX_RET5_ABS_MAX = 0.03

# Real-rate proxy (TIP/IEF) -> score adjustment
REAL_FACTOR_LOOKBACK = 5
REAL_BIAS = 0.015

ATR_LEN = 14
STOP_ATR_MULT = 2.0
RR = 2.0

ACCOUNT_USD = 1000.0
RISK_PCT = 1.0
RISK_USD = ACCOUNT_USD * (RISK_PCT / 100.0)

LOT_OZ = 100.0
MIN_LOT = 0.01
LOT_STEP = 0.01

# Spread (10 bps)
SPREAD_BPS = 10
SPREAD = SPREAD_BPS / 10000.0
HALF_SPREAD = SPREAD / 2.0

# ============================================================
# Indicators
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
    xau_close: pd.Series,
    dxy_close: pd.Series | None,
    us10y_close: pd.Series | None,
    vix_close: pd.Series | None,
    spx_close: pd.Series | None,
    tips_close: pd.Series | None,
    ief_close: pd.Series | None,
) -> pd.DataFrame:

    series = [xau_close.rename("xau")]
    if dxy_close is not None:
        series.append(dxy_close.rename("dxy"))
    if us10y_close is not None:
        series.append(us10y_close.rename("us10y"))
    if vix_close is not None:
        series.append(vix_close.rename("vix"))
    if spx_close is not None:
        series.append(spx_close.rename("spx"))
    if tips_close is not None:
        series.append(tips_close.rename("tips"))
    if ief_close is not None:
        series.append(ief_close.rename("ief"))

    df = pd.concat(series, axis=1).dropna()

    df["x_ret1"] = np.log(df["xau"]).diff()
    df["x_ret3"] = np.log(df["xau"]).diff(3)
    df["x_ret5"] = np.log(df["xau"]).diff(5)

    if "dxy" in df.columns:
        df["dxy_ret1"] = np.log(df["dxy"]).diff()
        df["dxy_ret5"] = np.log(df["dxy"]).diff(5)

    if "us10y" in df.columns:
        df["us10y_chg1"] = df["us10y"].diff()
        df["us10y_chg5"] = df["us10y"].diff(5)

    if "vix" in df.columns:
        df["vix_ret1"] = np.log(df["vix"]).diff()
        df["vix_ret5"] = np.log(df["vix"]).diff(5)

    if "spx" in df.columns:
        df["spx_ret1"] = np.log(df["spx"]).diff()
        df["spx_ret5"] = np.log(df["spx"]).diff(5)

    if "tips" in df.columns:
        df["tips_ret1"] = np.log(df["tips"]).diff()
        df["tips_ret5"] = np.log(df["tips"]).diff(5)

    # TIP/IEF real-rate proxy
    if "tips" in df.columns and "ief" in df.columns:
        df["real_factor"] = np.log(df["tips"] / df["ief"])
        df["real_f_chg5"] = df["real_factor"].diff(REAL_FACTOR_LOOKBACK)

    df["ma50"] = df["xau"].rolling(50).mean()
    df["trend_up"] = (df["xau"] > df["ma50"]).astype(int)

    return df

# ============================================================
# Main
# ============================================================
def main():
    print("Downloading data...")

    xau = fetch_ohlc_stooq(XAU)

    dxy, used_dxy = fetch_optional(DXY_CANDIDATES, "DXY")
    us10y, used_us10y = fetch_optional(US10Y_CANDIDATES, "US10Y")
    spx, used_spx = fetch_optional(SPX_CANDIDATES, "SPX")
    tips, used_tips = fetch_optional(TIPS_CANDIDATES, "TIPS")
    ief, used_ief = fetch_optional(IEF_CANDIDATES, "IEF")
    vix, used_vix = fetch_optional(VIX_CANDIDATES, "VIX")

    xau2 = xau.copy()
    xau2["ATR"] = compute_atr(xau2, ATR_LEN)

    feats = make_features(
        xau2["Close"],
        dxy["Close"] if dxy is not None else None,
        us10y["Close"] if us10y is not None else None,
        vix["Close"] if vix is not None else None,
        spx["Close"] if spx is not None else None,
        tips["Close"] if tips is not None else None,
        ief["Close"] if ief is not None else None,
    )

    feats = feats.join(xau2["ATR"].rename("atr"), how="left").dropna()
    feats["y"] = (feats["xau"].shift(-1) > feats["xau"]).astype(int)
    feats = feats.dropna()

    feature_cols = ["x_ret1", "x_ret3", "x_ret5"]

    if "dxy_ret1" in feats.columns and "dxy_ret5" in feats.columns:
        feature_cols += ["dxy_ret1", "dxy_ret5"]
    if "us10y_chg1" in feats.columns and "us10y_chg5" in feats.columns:
        feature_cols += ["us10y_chg1", "us10y_chg5"]
    if "vix_ret1" in feats.columns and "vix_ret5" in feats.columns:
        feature_cols += ["vix_ret1", "vix_ret5"]
    if "spx_ret1" in feats.columns and "spx_ret5" in feats.columns:
        feature_cols += ["spx_ret1", "spx_ret5"]
    if "tips_ret1" in feats.columns and "tips_ret5" in feats.columns:
        feature_cols += ["tips_ret1", "tips_ret5"]
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

    in_uptrend = int(feats.loc[last_day, "trend_up"]) == 1
    trend_txt = "UP" if in_uptrend else "DOWN"

    # Real factor bias
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

    # Risk gate selection
    vix_available = used_vix is not None
    risk_gate = "VIX" if vix_available else "SPX_PROXY"
    reason = ""

    # If VIX is required (not in this version), block when missing
    if REQUIRE_VIX_FOR_TRADE and not vix_available:
        side = "NO-TRADE"
        reason = "VIX missing -> quality filter"
    else:
        # If VIX missing, apply SPX proxy guardrail (if possible)
        if (not vix_available) and SPX_RISK_PROXY_ON:
            if "spx_ret5" in feats.columns:
                spx_ret5 = float(feats.loc[last_day, "spx_ret5"])
                if np.isfinite(spx_ret5) and abs(spx_ret5) > SPX_RET5_ABS_MAX:
                    side = "NO-TRADE"
                    reason = f"SPX proxy risk: abs(spx_ret5)>{SPX_RET5_ABS_MAX:.2%}"
                else:
                    side = None  # continue to signal rules
            else:
                side = None  # no proxy available
        else:
            side = None  # continue

        if side is None:
            side = "NO-TRADE"
            if (p_up_adj > TH_LONG) and ((p_up_adj - 0.5) >= MIN_EDGE) and in_uptrend:
                side = "LONG"
            elif (p_up_adj < TH_SHORT) and ((0.5 - p_up_adj) >= MIN_EDGE) and (not in_uptrend):
                side = "SHORT"

    mid = float(feats.loc[last_day, "xau"])
    atr = float(feats.loc[last_day, "atr"])
    stop_dist = STOP_ATR_MULT * atr

    if side == "LONG":
        entry_exec = mid * (1.0 + HALF_SPREAD)
    elif side == "SHORT":
        entry_exec = mid * (1.0 - HALF_SPREAD)
    else:
        entry_exec = mid

    src = (
        f"Sources:\n"
        f"DXY: {used_dxy or 'NONE'}\n"
        f"US10Y: {used_us10y or 'NONE'}\n"
        f"VIX: {used_vix or 'NONE'}\n"
        f"SPX: {used_spx or 'NONE'}\n"
        f"TIPS: {used_tips or 'NONE'}\n"
        f"IEF: {used_ief or 'NONE'}\n"
    )

    header = f"GOLD AI SIGNAL (XAUUSD)\nDate: {last_day.date()}\n"
    info = (
        src
        + f"P(up): {p_up:.4f}\n"
        + f"P_adj: {p_up_adj:.4f} ({bias_reason})\n"
        + f"Trend(>MA50): {trend_txt}\n"
        + f"RiskGate: {risk_gate}\n"
        + f"Spread: {SPREAD_BPS} bps (±{SPREAD_BPS/2:.1f} bps)\n"
    )
    if real_available:
        info += f"RealFactor chg{REAL_FACTOR_LOOKBACK}: {real_chg5:+.4f}\n"
    else:
        info += f"RealFactor chg{REAL_FACTOR_LOOKBACK}: N/A\n"
    if reason:
        info += f"Filter: {reason}\n"

    if side == "NO-TRADE":
        msg = header + "\n" + info + "\nSignal: NO-TRADE"
        print(msg)
        if SEND_NO_TRADE:
            send_telegram(msg)
        return

    if side == "LONG":
        sl = entry_exec - stop_dist
        tp = entry_exec + RR * stop_dist
    else:
        sl = entry_exec + stop_dist
        tp = entry_exec - RR * stop_dist

    lot_raw = lot_from_risk(stop_dist, RISK_USD)
    lot = round_down(lot_raw, LOT_STEP)
    if lot < MIN_LOT:
        lot = MIN_LOT

    msg = (
        header + "\n"
        + info + "\n"
        + f"Signal: {side}\n\n"
        + f"Mid (Close): {mid:.2f}\n"
        + f"Entry (spread adj): {entry_exec:.2f}\n"
        + f"Stop Loss: {sl:.2f}\n"
        + f"Take Profit: {tp:.2f}\n"
        + f"ATR({ATR_LEN}): {atr:.2f}\n"
        + f"StopDist: {stop_dist:.2f} ({STOP_ATR_MULT:.1f}x ATR)\n"
        + f"RR: {RR:.1f}R\n\n"
        + f"Account: ${ACCOUNT_USD:.0f}\n"
        + f"Risk: ${RISK_USD:.2f} ({RISK_PCT:.1f}%)\n"
        + f"Suggested lot: {lot:.2f}\n"
    )

    print(msg)
    send_telegram(msg)

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
