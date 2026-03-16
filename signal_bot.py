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
# TELEGRAM
# ============================================================

DEBUG_NO_TELEGRAM = False
SEND_NO_TRADE = True

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN","").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID","").strip()


def send_telegram(text: str):

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
        "disable_web_page_preview": True
    }

    r = requests.post(url,json=payload,timeout=30)

    if r.status_code != 200:
        raise RuntimeError(f"Telegram HTTP {r.status_code}")


# ============================================================
# STATE
# ============================================================

STATE_DIR = Path(".bot_state")
STATE_PATH = STATE_DIR / "state.json"


def load_state():

    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except:
            return {}

    return {}


def save_state(state):

    STATE_DIR.mkdir(exist_ok=True)

    STATE_PATH.write_text(
        json.dumps(state,indent=2)
    )


# ============================================================
# DATA
# ============================================================

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent":"Mozilla/5.0"
})


def stooq_url(symbol):

    return f"https://stooq.com/q/d/l/?s={symbol}&i=d"


def fetch_ohlc_stooq(symbol):

    url = stooq_url(symbol)

    r = SESSION.get(url,timeout=30)

    if r.status_code != 200:
        raise RuntimeError("HTTP error")

    txt = r.text

    if len(txt) < 50:
        raise RuntimeError("Empty CSV")

    df = pd.read_csv(StringIO(txt))

    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    df = df.sort_values("Date")

    df = df.set_index("Date")

    df = df[["Open","High","Low","Close"]]

    return df.dropna()


def fetch_ohlc_yahoo(symbol):

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=10y&interval=1d"

    r = SESSION.get(url,timeout=30)

    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")

    data = r.json()

    result = data["chart"]["result"][0]

    ts = result["timestamp"]

    q = result["indicators"]["quote"][0]

    df = pd.DataFrame({

        "Date":pd.to_datetime(ts,unit="s",utc=True).tz_convert(None).normalize(),

        "Open":q["open"],
        "High":q["high"],
        "Low":q["low"],
        "Close":q["close"]

    })

    df = df.dropna()

    df = df.sort_values("Date")

    df = df.set_index("Date")

    return df


def fetch_ohlc(symbol, yahoo_symbol=None):

    try:

        return fetch_ohlc_stooq(symbol)

    except:

        if yahoo_symbol:

            return fetch_ohlc_yahoo(yahoo_symbol)

        raise


# ============================================================
# SYMBOLS
# ============================================================

XAU_SYMBOL = "xauusd|GC=F"

DXY_SYMBOL = "usd_i"
US10Y_SYMBOL = "10yusy_b"
VIX_SYMBOL = "vi_f"

SPX_SYMBOL = "^spx"

TIPS_SYMBOL = "tip_us"
IEF_SYMBOL = "ief_us"


# ============================================================
# STRATEGY
# ============================================================

TRAIN_DAYS = 252*5

TH_LONG = 0.62
TH_SHORT = 0.38

REAL_BIAS = 0.015

ATR_LEN = 14

STOP_ATR_MULT = 1.2

RR = 2.0

ACCOUNT = 1000

RISK = 0.02

LOT_OZ = 100


# ============================================================
# HELPERS
# ============================================================

def compute_atr(df,n):

    prev = df["Close"].shift(1)

    tr = pd.concat([

        df["High"]-df["Low"],
        abs(df["High"]-prev),
        abs(df["Low"]-prev)

    ],axis=1).max(axis=1)

    return tr.rolling(n).mean()


def lot_from_risk(stop):

    risk_usd = ACCOUNT*RISK

    if stop<=0:
        return 0

    return risk_usd/(stop*LOT_OZ)


# ============================================================
# FEATURES
# ============================================================

def make_features(xau,dxy,us10y,vix):

    df = pd.concat([

        xau["Close"].rename("xau"),
        dxy["Close"].rename("dxy"),
        us10y["Close"].rename("us10y"),
        vix["Close"].rename("vix")

    ],axis=1).dropna()

    df["x_ret1"] = np.log(df["xau"]).diff()
    df["x_ret5"] = np.log(df["xau"]).diff(5)

    df["dxy_ret1"] = np.log(df["dxy"]).diff()

    df["us10y_chg1"] = df["us10y"].diff()

    df["vix_ret1"] = np.log(df["vix"]).diff()

    df["ma50"] = df["xau"].rolling(50).mean()

    df["trend_up"] = (df["xau"]>df["ma50"]).astype(int)

    df["y"] = (df["xau"].shift(-1)>df["xau"]).astype(int)

    return df.dropna()


# ============================================================
# MAIN
# ============================================================

def main():

    state = load_state()

    xau = fetch_ohlc("xauusd","GC=F")

    dxy = fetch_ohlc("usd_i")

    us10y = fetch_ohlc("10yusy_b")

    vix = fetch_ohlc("vi_f")

    feats = make_features(xau,dxy,us10y,vix)

    if len(feats)<TRAIN_DAYS+10:
        raise RuntimeError("Not enough aligned rows")

    X = feats[["x_ret1","x_ret5","dxy_ret1","us10y_chg1","vix_ret1","trend_up"]]

    y = feats["y"]

    last = feats.index[-1]

    X_train = X.iloc[-TRAIN_DAYS-1:-1]

    y_train = y.iloc[-TRAIN_DAYS-1:-1]

    X_last = X.loc[[last]]

    model = Pipeline([

        ("scaler",StandardScaler()),
        ("clf",LogisticRegression(max_iter=2000))

    ])

    model.fit(X_train,y_train)

    p_up = float(model.predict_proba(X_last)[:,1][0])

    p_adj = min(0.999,max(0.001,p_up+REAL_BIAS))

    trend = int(feats.loc[last,"trend_up"])==1

    side="NO-TRADE"

    if p_adj>TH_LONG and trend:

        side="LONG"

    elif p_adj<TH_SHORT and not trend:

        side="SHORT"

    entry=float(xau["Close"].iloc[-1])

    atr=float(compute_atr(xau,ATR_LEN).iloc[-1])

    stop=STOP_ATR_MULT*atr

    if side=="LONG":

        sl=entry-stop
        tp=entry+RR*stop

    elif side=="SHORT":

        sl=entry+stop
        tp=entry-RR*stop

    else:

        sl=tp=float("nan")

    lot=lot_from_risk(stop)

    msg=(

        f"GOLD AI SIGNAL (XAUUSD)\n\n"

        f"Date: {last.date()}\n\n"

        f"P(up): {p_up:.4f}\n"

        f"P_adj: {p_adj:.4f}\n\n"

        f"Signal: {side}\n\n"

        f"Entry: {entry:.2f}\n"

        f"Stop Loss: {sl:.2f}\n"

        f"Take Profit: {tp:.2f}\n\n"

        f"ATR: {atr:.2f}\n"

        f"Suggested lot: {lot:.2f}"

    )

    print(msg)

    if side!="NO-TRADE":

        send_telegram(msg)

    elif SEND_NO_TRADE:

        send_telegram(msg)

    save_state(state)


if __name__=="__main__":

    try:

        main()

    except Exception as e:

        send_telegram(f"❌ BOT ERROR\n{type(e).__name__}: {str(e)}")

        raise
