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
# Symbols
# ============================================================
XAU = "xauusd"  # required

DXY_CANDIDATES    = ["dx.f", "usd_i", "usdidx"]
VIX_CANDIDATES    = ["vi.f", "vix"]
US10Y_CANDIDATES  = ["10yusy.b", "us10y"]
SPX_CANDIDATES    = ["^spx"]
TIPS_CANDIDATES   = ["tip.us"]
IEF_CANDIDATES    = ["ief.us"]

# ============================================================
# Core config (Swing)
# ============================================================
TRAIN_DAYS = 252 * 5

ATR_LEN = 14
STOP_ATR_MULT = 2.0
RR = 2.0
MAX_HOLD_DAYS = 5

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

# En kaliteli gate
REQUIRE_VIX_FOR_TRADE = True

# Real-rate proxy (TIP/IEF) score bias
REAL_FACTOR_LOOKBACK = 5

# Grid candidates (optimize)
GRID_TH_LONG  = [0.65, 0.67]
GRID_MIN_EDGE = [0.17, 0.20]
GRID_REAL_BIAS = [0.01, 0.02, 0.03]

MIN_TRADES_FOR_RANK = 12  # çok az trade çıkarsa Sharpe yanıltır

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
    # base + optional
    series = [xau_ohlc["Close"].rename("xau")]
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

    # XAU returns
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

    # TIP/IEF real proxy
    if "tips" in df.columns and "ief" in df.columns:
        df["real_factor"] = np.log(df["tips"] / df["ief"])
        df["real_f_chg5"] = df["real_factor"].diff(REAL_FACTOR_LOOKBACK)

    # Trend filter (MA50)
    df["ma50"] = df["xau"].rolling(50).mean()
    df["trend_up"] = (df["xau"] > df["ma50"]).astype(int)

    # ATR join (from xau ohlc index)
    atr = compute_atr(xau_ohlc, ATR_LEN).rename("atr")
    df = df.join(atr, how="left")

    # next-day label (for rolling training)
    df["y"] = (df["xau"].shift(-1) > df["xau"]).astype(int)

    df = df.dropna()
    return df

def build_feature_cols(feats: pd.DataFrame) -> list[str]:
    cols = ["x_ret1", "x_ret3", "x_ret5"]
    for a, b in [("dxy_ret1","dxy_ret5"), ("us10y_chg1","us10y_chg5"),
                 ("vix_ret1","vix_ret5"), ("spx_ret1","spx_ret5"),
                 ("tips_ret1","tips_ret5")]:
        if a in feats.columns and b in feats.columns:
            cols += [a, b]
    if "real_f_chg5" in feats.columns:
        cols += ["real_f_chg5"]
    cols += ["trend_up"]
    return cols

# ============================================================
# Swing backtest engine (barrier within MAX_HOLD_DAYS)
# Conservative: if SL & TP both touched same day, assume SL first.
# Returns R-multiple per trade.
# ============================================================
def simulate_trade_R(
    xau_ohlc: pd.DataFrame,
    entry_date: pd.Timestamp,
    side: str,
    entry_exec: float,
    stop_dist: float,
    rr: float,
    max_hold: int
) -> float:
    if side not in ("LONG", "SHORT"):
        return 0.0

    if side == "LONG":
        sl = entry_exec - stop_dist
        tp = entry_exec + rr * stop_dist
    else:
        sl = entry_exec + stop_dist
        tp = entry_exec - rr * stop_dist

    # iterate forward days
    idx = xau_ohlc.index
    try:
        start_pos = idx.get_loc(entry_date)
    except KeyError:
        return 0.0

    # look ahead 1..max_hold
    for k in range(1, max_hold + 1):
        if start_pos + k >= len(idx):
            break
        d = idx[start_pos + k]
        hi = float(xau_ohlc.loc[d, "High"])
        lo = float(xau_ohlc.loc[d, "Low"])

        if side == "LONG":
            hit_sl = lo <= sl
            hit_tp = hi >= tp
            if hit_sl and hit_tp:
                return -1.0  # conservative
            if hit_sl:
                return -1.0
            if hit_tp:
                return +rr
        else:
            hit_sl = hi >= sl
            hit_tp = lo <= tp
            if hit_sl and hit_tp:
                return -1.0
            if hit_sl:
                return -1.0
            if hit_tp:
                return +rr

    # exit at close of last hold day (or last available)
    end_pos = min(start_pos + max_hold, len(idx) - 1)
    exit_date = idx[end_pos]
    exit_px = float(xau_ohlc.loc[exit_date, "Close"])

    if side == "LONG":
        pnl = (exit_px - entry_exec)
    else:
        pnl = (entry_exec - exit_px)

    return pnl / stop_dist if stop_dist > 0 else 0.0

# ============================================================
# Rolling probability series (walk-forward)
# ============================================================
def rolling_probs(feats: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    # p_up for each day using prior TRAIN_DAYS
    p = pd.Series(index=feats.index, dtype=float)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    for i in range(TRAIN_DAYS, len(feats) - 1):
        train_slice = feats.iloc[i-TRAIN_DAYS:i]
        test_row = feats.iloc[i:i+1]

        X_train = train_slice[feature_cols]
        y_train = train_slice["y"].astype(int)

        X_test = test_row[feature_cols]

        model.fit(X_train, y_train)
        p.iloc[i] = float(model.predict_proba(X_test)[:, 1][0])

    return p

# ============================================================
# Grid evaluation
# ============================================================
def evaluate_params(
    feats: pd.DataFrame,
    xau_ohlc: pd.DataFrame,
    p_up_series: pd.Series,
    th_long: float,
    min_edge: float,
    real_bias: float
) -> dict:
    th_short = 1.0 - th_long  # keep symmetric around 0.5 (0.65->0.35, 0.67->0.33)

    Rs = []
    trades = 0

    for dt in feats.index:
        p0 = p_up_series.get(dt, np.nan)
        if not np.isfinite(p0):
            continue

        # VIX gate
        if REQUIRE_VIX_FOR_TRADE:
            if ("vix_ret1" not in feats.columns) or ("vix_ret5" not in feats.columns):
                # no VIX features => treat as missing
                continue

        trend_up = int(feats.loc[dt, "trend_up"]) == 1

        # real bias
        bias = 0.0
        if "real_f_chg5" in feats.columns:
            rf = feats.loc[dt, "real_f_chg5"]
            if np.isfinite(rf):
                if rf > 0:
                    bias = +real_bias
                elif rf < 0:
                    bias = -real_bias

        p_adj = clamp01(p0 + bias)

        side = "NO-TRADE"
        if (p_adj > th_long) and ((p_adj - 0.5) >= min_edge) and trend_up:
            side = "LONG"
        elif (p_adj < th_short) and ((0.5 - p_adj) >= min_edge) and (not trend_up):
            side = "SHORT"

        if side == "NO-TRADE":
            continue

        # entry at "mid close" + spread
        mid = float(feats.loc[dt, "xau"])
        atr = float(feats.loc[dt, "atr"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        stop_dist = STOP_ATR_MULT * atr

        if side == "LONG":
            entry_exec = mid * (1.0 + HALF_SPREAD)
        else:
            entry_exec = mid * (1.0 - HALF_SPREAD)

        R = simulate_trade_R(
            xau_ohlc=xau_ohlc,
            entry_date=dt,
            side=side,
            entry_exec=entry_exec,
            stop_dist=stop_dist,
            rr=RR,
            max_hold=MAX_HOLD_DAYS
        )
        Rs.append(R)
        trades += 1

    if trades == 0:
        return {
            "th_long": th_long,
            "min_edge": min_edge,
            "real_bias": real_bias,
            "trades": 0,
            "avg_R": np.nan,
            "hit_rate": np.nan,
            "sharpe": np.nan,
            "max_dd": np.nan,
            "total_R": 0.0,
        }

    R_arr = np.array(Rs, dtype=float)
    total_R = float(np.nansum(R_arr))
    avg_R = float(np.nanmean(R_arr))
    hit_rate = float(np.nanmean(R_arr > 0))

    # Sharpe on trade returns (not annualized)
    std = float(np.nanstd(R_arr, ddof=1)) if trades > 1 else 0.0
    sharpe = float(avg_R / std) if std > 1e-12 else np.nan

    # max drawdown on equity curve in R units
    equity = np.nancumsum(R_arr)
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    max_dd = float(np.min(dd)) if len(dd) else np.nan

    return {
        "th_long": th_long,
        "min_edge": min_edge,
        "real_bias": real_bias,
        "trades": trades,
        "avg_R": avg_R,
        "hit_rate": hit_rate,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_R": total_R,
    }

def run_grid_backtest(feats: pd.DataFrame, xau_ohlc: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    feature_cols = build_feature_cols(feats)

    # last ~3 years window
    end_dt = feats.index[-1]
    start_dt = end_dt - pd.Timedelta(days=365*3 + 10)
    feats_bt = feats.loc[feats.index >= start_dt].copy()

    # Need enough history for rolling training; keep earlier buffer
    feats_for_probs = feats.loc[feats.index >= (start_dt - pd.Timedelta(days=365*6))].copy()
    # rolling probabilities over the larger slice, then align
    p_series = rolling_probs(feats_for_probs, feature_cols)
    p_series = p_series.reindex(feats_bt.index)

    rows = []
    for th in GRID_TH_LONG:
        for me in GRID_MIN_EDGE:
            for rb in GRID_REAL_BIAS:
                res = evaluate_params(
                    feats=feats_bt,
                    xau_ohlc=xau_ohlc,
                    p_up_series=p_series,
                    th_long=th,
                    min_edge=me,
                    real_bias=rb
                )
                rows.append(res)

    df = pd.DataFrame(rows)

    # ranking: Sharpe desc, then total_R desc, with min trades constraint
    cand = df[df["trades"] >= MIN_TRADES_FOR_RANK].copy()
    best = None
    if len(cand) > 0:
        cand = cand.sort_values(by=["sharpe", "total_R"], ascending=[False, False])
        best = cand.iloc[0].to_dict()

    return df.sort_values(by=["sharpe", "total_R"], ascending=[False, False]), best

# ============================================================
# Today signal with chosen params
# ============================================================
def today_signal(feats: pd.DataFrame, best_params: dict | None) -> str:
    feature_cols = build_feature_cols(feats)

    # train once for today signal on latest TRAIN_DAYS
    if len(feats) < TRAIN_DAYS + 10:
        raise RuntimeError(f"Not enough aligned rows: {len(feats)} need~{TRAIN_DAYS}")

    last_day = feats.index[-1]
    X = feats[feature_cols]
    y = feats["y"].astype(int)

    X_train = X.iloc[-TRAIN_DAYS-1:-1]
    y_train = y.iloc[-TRAIN_DAYS-1:-1]
    X_last = X.loc[[last_day]]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000))
    ])
    model.fit(X_train, y_train)

    p_up = float(model.predict_proba(X_last)[:, 1][0])

    # params
    if best_params is None:
        th_long = 0.65
        min_edge = 0.17
        real_bias = 0.02
        opt_note = "Params: default"
    else:
        th_long = float(best_params["th_long"])
        min_edge = float(best_params["min_edge"])
        real_bias = float(best_params["real_bias"])
        opt_note = f"Params(opt): TH_LONG={th_long:.2f} MIN_EDGE={min_edge:.2f} REAL_BIAS={real_bias:.2f}"

    th_short = 1.0 - th_long

    trend_up = int(feats.loc[last_day, "trend_up"]) == 1
    trend_txt = "UP" if trend_up else "DOWN"

    # VIX gate
    if REQUIRE_VIX_FOR_TRADE and (("vix_ret1" not in feats.columns) or ("vix_ret5" not in feats.columns)):
        side = "NO-TRADE"
        reason = "VIX missing -> quality filter"
    else:
        # bias
        bias = 0.0
        bias_reason = "RealBias: N/A"
        if "real_f_chg5" in feats.columns:
            rf = float(feats.loc[last_day, "real_f_chg5"])
            if np.isfinite(rf):
                if rf > 0:
                    bias = +real_bias
                    bias_reason = f"RealBias: +{real_bias:.2f} (real up)"
                elif rf < 0:
                    bias = -real_bias
                    bias_reason = f"RealBias: -{real_bias:.2f} (real down)"
                else:
                    bias_reason = "RealBias: +0.00 (flat)"
        p_adj = clamp01(p_up + bias)

        side = "NO-TRADE"
        reason = ""
        if (p_adj > th_long) and ((p_adj - 0.5) >= min_edge) and trend_up:
            side = "LONG"
        elif (p_adj < th_short) and ((0.5 - p_adj) >= min_edge) and (not trend_up):
            side = "SHORT"

    mid = float(feats.loc[last_day, "xau"])
    atr = float(feats.loc[last_day, "atr"])
    stop_dist = STOP_ATR_MULT * atr

    if side == "LONG":
        entry_exec = mid * (1.0 + HALF_SPREAD)
        sl = entry_exec - stop_dist
        tp = entry_exec + RR * stop_dist
    elif side == "SHORT":
        entry_exec = mid * (1.0 - HALF_SPREAD)
        sl = entry_exec + stop_dist
        tp = entry_exec - RR * stop_dist
    else:
        entry_exec = mid
        sl = np.nan
        tp = np.nan

    lot_raw = lot_from_risk(stop_dist, RISK_USD) if np.isfinite(stop_dist) and stop_dist > 0 else 0.0
    lot = round_down(lot_raw, LOT_STEP)
    if lot < MIN_LOT:
        lot = MIN_LOT

    msg = (
        f"GOLD AI SIGNAL (XAUUSD)\n"
        f"Date: {last_day.date()}\n\n"
        f"{opt_note}\n"
        f"P(up): {p_up:.4f}\n"
    )
    if "real_f_chg5" in feats.columns:
        rf = float(feats.loc[last_day, "real_f_chg5"])
        msg += f"RealFactor chg{REAL_FACTOR_LOOKBACK}: {rf:+.4f}\n"
    msg += (
        f"Trend(>MA50): {trend_txt}\n"
        f"Spread: {SPREAD_BPS} bps\n"
    )
    if reason:
        msg += f"Filter: {reason}\n"

    msg += f"\nSignal: {side}\n"

    if side in ("LONG", "SHORT"):
        msg += (
            f"\nMid (Close): {mid:.2f}\n"
            f"Entry (spread adj): {entry_exec:.2f}\n"
            f"Stop Loss: {sl:.2f}\n"
            f"Take Profit: {tp:.2f}\n"
            f"ATR({ATR_LEN}): {atr:.2f}\n"
            f"StopDist: {stop_dist:.2f} ({STOP_ATR_MULT:.1f}x ATR)\n"
            f"RR: {RR:.1f}R\n\n"
            f"Account: ${ACCOUNT_USD:.0f}\n"
            f"Risk: ${RISK_USD:.2f} ({RISK_PCT:.1f}%)\n"
            f"Suggested lot: {lot:.2f}\n"
        )

    return msg

# ============================================================
# Main
# ============================================================
def main():
    print("Downloading data...")

    xau = fetch_ohlc_stooq(XAU)

    dxy, _ = fetch_optional(DXY_CANDIDATES, "DXY")
    us10y, _ = fetch_optional(US10Y_CANDIDATES, "US10Y")
    vix, _ = fetch_optional(VIX_CANDIDATES, "VIX")
    spx, _ = fetch_optional(SPX_CANDIDATES, "SPX")
    tips, _ = fetch_optional(TIPS_CANDIDATES, "TIPS")
    ief, _ = fetch_optional(IEF_CANDIDATES, "IEF")

    feats = make_features(
        xau_ohlc=xau,
        dxy_close=dxy["Close"] if dxy is not None else None,
        us10y_close=us10y["Close"] if us10y is not None else None,
        vix_close=vix["Close"] if vix is not None else None,
        spx_close=spx["Close"] if spx is not None else None,
        tips_close=tips["Close"] if tips is not None else None,
        ief_close=ief["Close"] if ief is not None else None,
    )

    print(f"Aligned rows: {len(feats)} | last date: {feats.index[-1].date()}")

    # Grid optimize + summary
    print("\n===== GRID BACKTEST (last ~3 years, swing barriers) =====")
    grid_df, best = run_grid_backtest(feats, xau)

    # pretty print top 12
    show = grid_df.copy()
    show["hit_rate"] = (show["hit_rate"] * 100).round(1)
    show["avg_R"] = show["avg_R"].round(3)
    show["sharpe"] = show["sharpe"].round(3)
    show["max_dd"] = show["max_dd"].round(3)
    show["total_R"] = show["total_R"].round(2)
    show = show.rename(columns={"hit_rate":"hit_%", "avg_R":"avgR", "total_R":"totR", "max_dd":"maxDD"})
    print(show.head(12).to_string(index=False))

    if best is None:
        print("\nNo parameter set met min trade requirement. Using defaults for today's signal.")
    else:
        print("\nBEST PARAMS:",
              f"TH_LONG={best['th_long']:.2f}",
              f"MIN_EDGE={best['min_edge']:.2f}",
              f"REAL_BIAS={best['real_bias']:.2f}",
              f"Trades={int(best['trades'])}",
              f"Sharpe={best['sharpe']:.3f}",
              f"totR={best['total_R']:.2f}",
              f"maxDD={best['max_dd']:.3f}",
              sep=" | ")

    # Today signal (using best params if available)
    print("\n===== TODAY SIGNAL =====")
    msg = today_signal(feats, best)
    print(msg)

    if SEND_NO_TRADE or ("Signal: NO-TRADE" not in msg):
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
