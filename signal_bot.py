import pandas as pd
import numpy as np
import requests
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

URL = "https://stooq.com/q/d/l/?s=xauusd&i=d"

TH_LONG = 0.65
TH_SHORT = 0.35

ATR_LEN = 14
STOP_MULT = 2

ACCOUNT = 1000
RISK = 10

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(msg):

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    requests.post(url,json={
        "chat_id": CHAT_ID,
        "text": msg
    })


df = pd.read_csv(URL)

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").set_index("Date")

prev_close = df["Close"].shift(1)

tr = pd.concat([
(df["High"] - df["Low"]),
(df["High"] - prev_close).abs(),
(df["Low"] - prev_close).abs()
], axis=1).max(axis=1)

df["ATR"] = tr.rolling(ATR_LEN).mean()

close = df["Close"]

ret = np.log(close).diff()

feat = pd.DataFrame(index=df.index)

feat["ret1"] = ret
feat["mom5"] = close / close.rolling(5).mean()
feat["mom10"] = close / close.rolling(10).mean()

y = (close.shift(-1) > close).astype(int)

data = feat.join(y.rename("y")).join(df[["Close","ATR"]]).dropna()

X = data.drop(columns=["y","Close","ATR"])
y = data["y"]

model = Pipeline([
("scaler", StandardScaler()),
("clf", LogisticRegression(max_iter=2000))
])

model.fit(X,y)

last = X.index[-1]

prob = model.predict_proba(X.loc[[last]])[:,1][0]

price = df.loc[last,"Close"]
atr = df.loc[last,"ATR"]

signal = "NO TRADE"

if prob > TH_LONG:
    signal = "LONG"

if prob < TH_SHORT:
    signal = "SHORT"

if signal == "NO TRADE":
    exit()

stop = atr * STOP_MULT

if signal == "LONG":

    sl = price - stop
    tp = price + stop*2

else:

    sl = price + stop
    tp = price - stop*2

lot = RISK / (stop*100)

msg = f"""
GOLD AI SIGNAL

Signal: {signal}

Probability: {round(prob,3)}

Entry: {round(price,2)}
Stop Loss: {round(sl,2)}
Take Profit: {round(tp,2)}

Lot Size: {round(lot,3)}
Risk: ${RISK}
"""

send_telegram(msg)

print(msg)