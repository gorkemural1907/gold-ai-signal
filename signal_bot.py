import os
import math
import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# DEBUG
# True  => Telegram gönderme kapalı (Actions loguna basar)
# False => Telegram gönderir
# ============================================================
DEBUG_NO_TELEGRAM = True

# ---------- Stooq symbols ----------
def stooq_csv(symbol: str) -> str:
    return f"https://stooq.com/q/d/l/?s={symbol}&i=d"

XAU = "xauusd"
DXY = "dx.f"
US10Y = "10yusy.b"
VIX = "vi.f"

# ---------- Trading config ----------
TRAIN_DAYS = 252 * 5          # 5 yıl train (daha stabil)
TH_LONG = 0.65                # az ama kaliteli
TH_SHORT = 0.35

ATR_LEN = 14
STOP_ATR_MULT = 2.0
RR = 2.0

ACCOUNT_USD = 1000.0
RISK_PCT = 1.0
RISK_USD = ACCOUNT_USD * (RISK_PCT / 100.0)

# XAUUSD lot varsayımı: 1.00 lot = 100 oz (çoğu FX broker)
LOT_OZ = 100.0
MIN_LOT = 0.01
LOT_STEP = 0.01

# ---------- Telegram secrets ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# ============================================================
# Telegram (token/URL kesinlikle loga basılmaz)
# ============================================================
def send_telegram(text: str) -> None:
    if DEBUG_NO_TELEGRAM:
        print("DEBUG_NO_TELEGRAM=True -> Telegram gönderimi kapalı. Mesaj aşağıda:\n")
        print(text)
        return

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets missing (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID). Mesaj aşağıda:\n")
        print(text)
        return

    # Token içeren URL'yi asla print etmiyoruz
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=25)
        if r.status_code != 200:
            # r.text içinde token olmaz, ama gene de kısaltıyoruz
            raise RuntimeError(f"Telegram send failed: HTTP {r.status_code}, body={r.text[:200]}")
    except Exception as e:
        # Token/URL loga düşmesin
        raise RuntimeError(f"Telegram send exception: {type(e).__name__}: {str(e)[:200]}")

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
    # 1$ fiyat hareketi => 1 lot PnL ~ 100$ (100 oz)
    if stop_distance_usd <= 0:
        return 0.0
    return risk_usd / (stop_distance_usd * LOT_OZ)

def fetch_ohlc(symbol: str) -> pd.DataFrame:
    url = stooq_csv(symbol)
    try:
        df = pd.read_csv(url)
    except Exception as e:
        raise RuntimeError(f"Stooq read failed for {symbol}: {type(e).__name__}: {str(e)[:200]}")

    # Beklenen kolonlar yoksa net hata verelim
    needed = {"Date", "Open", "High", "Low", "Close"}
    if not needed.issubset(set(df.columns)):
        raise RuntimeError(f"Stooq columns missing for {symbol}. Got columns={list(df.columns)[:20]}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    df = df[["Open", "High", "Low", "Close"]].dropna()

    # Çok kısa dönüyorsa (boş/servis anomali) hata
    if len(df) < 200:
        raise RuntimeError(f"Stooq returned too few rows for {symbol}: rows={len(df)}")

    return df

def make_features(xau_close: pd.Series, dxy_close: pd.Series, us10y_close: pd.Series, vix_close: pd.Series) -> pd.DataFrame:
    df = pd.concat(
        [
            xau_close.rename("xau"),
            dxy_close.rename("dxy"),
            us10y_close.rename("us10y"),
            vix_close.rename("vix"),
        ],
        axis=1,
    ).dropna()

    # Returns (log where sensible)
    df["x_ret1"] = np.log(df["xau"]).diff()
    df["x_ret3"] = np.log(df["xau"]).diff(3)
    df["x_ret5"] = np.log(df["xau"]).diff(5)

    df["dxy_ret1"] = np.log(df["dxy"]).diff()
    df["dxy_ret5"] = np.log(df["dxy"]).diff(5)

    # yield farkı (log değil)
    df["us10y_chg1"] = df["us10y"].diff()
    df["us10y_chg5"] = df["us10y"].diff(5)

    df["vix_ret1"] = np.log(df["vix"]).diff()
    df["vix_ret5"] = np.log(df["vix"]).diff(5)

    # Trend filter
    df["ma50"] = df["xau"].rolling(50).mean()
    df["trend_up"] = (df["xau"] > df["ma50"]).astype(int)  # 1 uptrend

    return df

# ============================================================
# MAIN
# ============================================================
def main():
    print("Downloading data from Stooq...")
    xau = fetch_ohlc(XAU)
    dxy = fetch_ohlc(DXY)
    us10y = fetch_ohlc(US10Y)
    vix = fetch_ohlc(VIX)

    # ATR on XAU
    xau_atr = xau.copy()
    xau_atr["ATR"] = compute_atr(xau_atr, ATR_LEN)

    feats = make_features(
        xau_atr["Close"],
        dxy["Close"],
        us10y["Close"],
        vix["Close"],
    )

    # Align ATR
    feats = feats.join(xau_atr["ATR"].rename("atr"), how="left").dropna()

    # Label: tomorrow up?
    feats["y"] = (feats["xau"].shift(-1) > feats["xau"]).astype(int)
    feats = feats.dropna()

    feature_cols = [
        "x_ret1","x_ret3","x_ret5",
        "dxy_ret1","dxy_ret5",
        "us10y_chg1","us10y_chg5",
        "vix_ret1","vix_ret5",
        "trend_up",
    ]

    if len(feats) < TRAIN_DAYS + 10:
        raise RuntimeError(f"Not enough aligned rows: {len(feats)} need~{TRAIN_DAYS}")

    X = feats[feature_cols]
    y = feats["y"]

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

    # Az ama kaliteli + trend filtresi
    side = "NO-TRADE"
    if p_up > TH_LONG and in_uptrend:
        side = "LONG"
    elif p_up < TH_SHORT and (not in_uptrend):
        side = "SHORT"

    entry = float(feats.loc[last_day, "xau"])
    atr = float(feats.loc[last_day, "atr"])
    stop_dist = STOP_ATR_MULT * atr

    header = f"GOLD AI SIGNAL (XAUUSD)\nSignal date: {last_day.date()}\n"
    info = f"P(up): {p_up:.4f}\nTrend(>MA50): {trend_txt}\n"

    if side == "NO-TRADE":
        msg = header + "\n" + info + "\nSignal: NO-TRADE"
        print(msg)
        # NO-TRADE'de telegram göndermiyoruz (az ama kaliteli)
        return

    if side == "LONG":
        sl = entry - stop_dist
        tp = entry + RR * stop_dist
    else:
        sl = entry + stop_dist
        tp = entry - RR * stop_dist

    lot_raw = lot_from_risk(stop_dist, RISK_USD)
    lot = round_down(lot_raw, LOT_STEP)

    min_lot_risk = MIN_LOT * stop_dist * LOT_OZ
    lot_note = ""
    if lot < MIN_LOT:
        # İstersen burada NO-TRADE yapabiliriz; şimdilik uyarı basıp min lot yazıyoruz
        lot_note = f"⚠️ Suggested lot < min lot. Min lot {MIN_LOT:.2f} risk≈${min_lot_risk:.2f} (target ${RISK_USD:.2f})."
        lot = MIN_LOT

    msg = (
        header + "\n"
        + info + "\n"
        + f"Signal: {side}\n\n"
        + f"Entry (ref=Close): {entry:.2f}\n"
        + f"ATR({ATR_LEN}): {atr:.2f}\n"
        + f"Stop Loss: {sl:.2f}  ({STOP_ATR_MULT:.1f} ATR)\n"
        + f"Take Profit: {tp:.2f}  ({RR:.1f}R)\n\n"
        + f"Account: ${ACCOUNT_USD:.0f}\n"
        + f"Risk: ${RISK_USD:.2f} ({RISK_PCT:.1f}%)\n"
        + f"Suggested lot: {lot:.2f}\n"
    )
    if lot_note:
        msg += f"\n{lot_note}\n"

    print(msg)
    send_telegram(msg)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("FATAL ERROR:", repr(e))
        traceback.print_exc()
        raise
