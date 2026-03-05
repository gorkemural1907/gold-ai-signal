import os
import math
import requests
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

LOT_OZ = 100.0                # 1.00 lot = 100 oz varsayımı
MIN_LOT = 0.01
LOT_STEP = 0.01

# ---------- Telegram ----------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# Debug amaçlı: sinyal yoksa da mesaj atmak istersen true yap.
SEND_NO_TRADE = False

def send_telegram(text: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram secrets missing (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID). Printing only:\n")
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=20)

    # Hata varsa workflow kırmızıya dönsün diye raise ediyoruz
    if r.status_code != 200:
        raise RuntimeError(f"Telegram send failed: {r.status_code} {r.text}")

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
    df = pd.read_csv(stooq_csv(symbol))
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df[["Open", "High", "Low", "Close"]]

def make_features(xau: pd.DataFrame, dxy: pd.DataFrame, us10y: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    # Close serileri
    x = xau["Close"].rename("xau")
    d = dxy["Close"].rename("dxy")
    y = us10y["Close"].rename("us10y")
    v = vix["Close"].rename("vix")

    df = pd.concat([x, d, y, v], axis=1).dropna()

    # Returns
    df["x_ret1"] = np.log(df["xau"]).diff()
    df["x_ret3"] = np.log(df["xau"]).diff(3)
    df["x_ret5"] = np.log(df["xau"]).diff(5)

    df["dxy_ret1"] = np.log(df["dxy"]).diff()
    df["dxy_ret5"] = np.log(df["dxy"]).diff(5)

    df["us10y_chg1"] = df["us10y"].diff()
    df["us10y_chg5"] = df["us10y"].diff(5)

    df["vix_ret1"] = np.log(df["vix"]).diff()
    df["vix_ret5"] = np.log(df["vix"]).diff(5)

    # Trend filter (MA50)
    df["ma50"] = df["xau"].rolling(50).mean()
    df["trend"] = (df["xau"] > df["ma50"]).astype(int)  # 1=uptrend, 0=downtrend

    return df

def main():
    print("Downloading data...")
    xau = fetch_ohlc(XAU)
    dxy = fetch_ohlc(DXY)
    us10y = fetch_ohlc(US10Y)
    vix = fetch_ohlc(VIX)

    # ATR only on XAU
    xau_atr = xau.copy()
    xau_atr["ATR"] = compute_atr(xau_atr, ATR_LEN)

    feats = make_features(xau_atr, dxy, us10y, vix)
    # Align ATR back
    feats = feats.join(xau_atr["ATR"].rename("atr"), how="left").dropna()

    # Label: tomorrow up?
    feats["y"] = (feats["xau"].shift(-1) > feats["xau"]).astype(int)
    feats = feats.dropna()

    feature_cols = [
        "x_ret1","x_ret3","x_ret5",
        "dxy_ret1","dxy_ret5",
        "us10y_chg1","us10y_chg5",
        "vix_ret1","vix_ret5",
        "trend",
    ]

    if len(feats) < TRAIN_DAYS + 10:
        raise RuntimeError(f"Not enough rows: {len(feats)} (need ~{TRAIN_DAYS})")

    # Train on last TRAIN_DAYS ending at yesterday of last row
    X = feats[feature_cols]
    y = feats["y"]

    # We want signal for last available day
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

    # Decide signal with trend filter:
    # Uptrend => only LONG allowed; Downtrend => only SHORT allowed
    in_uptrend = int(feats.loc[last_day, "trend"]) == 1

    side = "NO-TRADE"
    if p_up > TH_LONG and in_uptrend:
        side = "LONG"
    elif p_up < TH_SHORT and (not in_uptrend):
        side = "SHORT"

    entry = float(feats.loc[last_day, "xau"])
    atr = float(feats.loc[last_day, "atr"])
    stop_dist = STOP_ATR_MULT * atr

    # Compose message (even if no-trade, log it)
    header = f"GOLD AI SIGNAL (XAUUSD)\nSignal date: {last_day.date()}\n"
    info = f"P(up): {p_up:.4f}\nTrend(>MA50): {'UP' if in_uptrend else 'DOWN'}\n"

    if side == "NO-TRADE":
        msg = header + "\n" + info + "\nSignal: NO-TRADE"
        print(msg)
        if SEND_NO_TRADE:
            send_telegram(msg)
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
        lot_note = f"⚠️ Suggested lot < min lot. Min lot {MIN_LOT:.2f} risk≈${min_lot_risk:.2f} (target ${RISK_USD:.2f})."
        # İstersen burada “NO-TRADE” yapabiliriz. Şimdilik uyarı verip min lotu yazıyoruz:
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
    main()
