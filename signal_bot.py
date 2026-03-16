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
# DATA FETCH
# ============================================================
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0"})
FETCH_PAUSE_S = 0.5


def normalize_ohlc_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index).normalize()
    df = df.sort_index()
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
# STRATEGY
# ============================================================
TRAIN_DAYS = 252 * 5

TH_LONG = 0.62
TH_SHORT = 0.38
MIN_EDGE = 0.12
REAL_BIAS = 0.015

ATR_LEN = 14
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


# ============================================================
# FEATURES
# ============================================================
def make_features(
    xau: pd.DataFrame,
    dxy: pd.DataFrame | None,
    us10y: pd.DataFrame | None,
    vix: pd.DataFrame | None,
    spx: pd.DataFrame | None,
    tips: pd.DataFrame | None,
    ief: pd.DataFrame | None,
) -> pd.DataFrame:
    series = [xau["Close"].rename("xau")]
    if dxy is not None:
        series.append(dxy["Close"].rename("dxy"))
    if us10y is not None:
        series.append(us10y["Close"].rename("us10y"))
    if vix is not None:
        series.append(vix["Close"].rename("vix"))
    if spx is not None:
        series.append(spx["Close"].rename("spx"))
    if tips is not None:
        series.append(tips["Close"].rename("tips"))
    if ief is not None:
        series.append(ief["Close"].rename("ief"))

    df = pd.concat(series, axis=1).dropna()
    print(f"[FEATS] aligned rows before feature engineering: {len(df)}")

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

    df["ma50"] = df["xau"].rolling(50).mean()
    df["trend_up"] = (df["xau"] > df["ma50"]).astype(int)

    atr14 = compute_atr(xau, ATR_LEN).rename("atr14")
    df = df.join(atr14, how="left")

    df["y"] = (df["xau"].shift(-1) > df["xau"]).astype(int)

    df = df.dropna()
    print(f"[FEATS] rows after feature engineering: {len(df)}")
    return df


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    state = load_state()

    xau, used_xau = fetch_ohlc_with_fallback("XAU", XAU_PROVIDERS)
    dxy, used_dxy = fetch_ohlc_with_fallback("DXY", DXY_PROVIDERS)
    us10y, used_us10y = fetch_ohlc_with_fallback("US10Y", US10Y_PROVIDERS)
    vix, used_vix = fetch_ohlc_with_fallback("VIX", VIX_PROVIDERS)
    spx, used_spx = fetch_ohlc_with_fallback("SPX", SPX_PROVIDERS)
    tips, used_tips = fetch_ohlc_with_fallback("TIPS", TIPS_PROVIDERS)
    ief, used_ief = fetch_ohlc_with_fallback("IEF", IEF_PROVIDERS)

    feats = make_features(xau, dxy, us10y, vix, spx, tips, ief)

    if len(feats) < TRAIN_DAYS + 10:
        raise RuntimeError(f"Not enough aligned rows: {len(feats)} need~{TRAIN_DAYS}")

    feature_cols = ["x_ret1", "x_ret3", "x_ret5", "trend_up"]

    for col in [
        "dxy_ret1", "dxy_ret5",
        "us10y_chg1", "us10y_chg5",
        "gold_real_mom",
        "vix_ret1", "vix_ret5",
        "spx_ret5",
        "gold_spx_mom",
        "real_f_chg5",
    ]:
        if col in feats.columns:
            feature_cols.append(col)

    print("[MODEL] feature cols:", feature_cols)

    X = feats[feature_cols]
    y = feats["y"].astype(int)

    last_day = feats.index[-1]
    X_train = X.iloc[-TRAIN_DAYS-1:-1]
    y_train = y.iloc[-TRAIN_DAYS-1:-1]
    X_last = X.loc[[last_day]]

    print(f"[MODEL] train rows={len(X_train)} infer_date={last_day.date()}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)

    p_up = float(model.predict_proba(X_last)[:, 1][0])

    real_chg5 = float(feats.loc[last_day, "real_f_chg5"]) if "real_f_chg5" in feats.columns else float("nan")
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

    trend_up = int(feats.loc[last_day, "trend_up"]) == 1
    trend_txt = "UP" if trend_up else "DOWN"

    side = "NO-TRADE"
    if (p_adj > TH_LONG) and ((p_adj - 0.5) >= MIN_EDGE) and trend_up:
        side = "LONG"
    elif (p_adj < TH_SHORT) and ((0.5 - p_adj) >= MIN_EDGE) and (not trend_up):
        side = "SHORT"

    atr = float(feats.loc[last_day, "atr14"])
    ybar = xau.iloc[-2]
    y_high = float(ybar["High"])
    y_low = float(ybar["Low"])

    if side == "LONG":
        entry = y_high * (1.0 + HALF_SPREAD)
        sl = entry - STOP_ATR_MULT * atr
        tp = entry + RR * (STOP_ATR_MULT * atr)
    elif side == "SHORT":
        entry = y_low * (1.0 - HALF_SPREAD)
        sl = entry + STOP_ATR_MULT * atr
        tp = entry - RR * (STOP_ATR_MULT * atr)
    else:
        entry = float(xau["Close"].iloc[-1])
        sl = float("nan")
        tp = float("nan")

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

    msg = (
        f"GOLD AI SIGNAL (XAUUSD)\n\n"
        f"Date: {last_day.date()}\n\n"
        f"Sources:\n"
        f"XAU: {used_xau}\n"
        f"DXY: {used_dxy}\n"
        f"US10Y: {used_us10y}\n"
        f"VIX: {used_vix}\n"
        f"SPX: {used_spx}\n"
        f"TIPS: {used_tips}\n"
        f"IEF: {used_ief}\n\n"
        f"P(up): {p_up:.4f}\n"
        f"P_adj: {p_adj:.4f} ({bias_reason})\n"
        f"Trend(>MA50): {trend_txt}\n\n"
        f"Signal: {side}\n\n"
        f"Breakout Entry: {entry:.2f}\n"
        f"Yesterday High: {y_high:.2f}\n"
        f"Yesterday Low: {y_low:.2f}\n"
        f"Stop Loss: {sl:.2f}\n"
        f"Take Profit: {tp:.2f}\n"
        f"ATR({ATR_LEN}): {atr:.2f}\n"
        f"StopDist: {stop_dist:.2f}\n"
        f"Suggested lot: {lot:.2f}\n"
    )

    if "gold_real_mom" in feats.columns:
        msg += f"GoldRealMom5: {float(feats.loc[last_day, 'gold_real_mom']):+.4f}\n"
    if "gold_spx_mom" in feats.columns:
        msg += f"GoldSPXMom5: {float(feats.loc[last_day, 'gold_spx_mom']):+.4f}\n"

    if trade_skipped:
        msg += f"\nTrade: SKIPPED\nReason: {skip_reason}\n"

    print(msg)

    if side != "NO-TRADE" or SEND_NO_TRADE:
        send_telegram(msg)

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
