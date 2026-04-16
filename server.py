import os, json, time, math, logging, pickle
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import httpx
from pydantic import BaseModel
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("scanner")

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
ALPACA_KEY    = os.environ.get("ALPACA_API_KEY","")
ALPACA_SECRET = os.environ.get("ALPACA_API_SECRET","")
ALPACA_URL    = os.environ.get("ALPACA_DATA_URL","https://data.alpaca.markets")
DATA_DIR      = Path("/data") if Path("/data").exists() else Path(__file__).parent / ".data"
MODEL_DIR     = DATA_DIR / "models"
OUTCOME_DIR   = DATA_DIR / "outcomes"
SCAN_DIR      = DATA_DIR / "scans"
PORT          = int(os.environ.get("PORT", 10000))

ET = ZoneInfo("America/New_York")
SCAN_HOURS = [10, 11, 12, 13, 14, 15]
TP_PCT = 0.0095   # +0.95% take profit
SL_PCT = 0.0150   # -1.50% stop loss (asymmetric)
FORCED_CLOSE_MIN = 15*60+55  # 15:55 ET in minutes

TICKERS = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK.B","UNH","JNJ",
    "V","XOM","JPM","PG","MA","HD","CVX","MRK","ABBV","LLY",
    "PEP","KO","COST","AVGO","MCD","WMT","TMO","CSCO","ACN","ABT",
    "DHR","CRM","NEE","LIN","TXN","PM","UNP","BMY","QCOM","RTX",
    "AMGN","HON","LOW","INTC","SCHW","INTU","ELV","AMD","CAT","DE",
    "GS","BLK","ADP","SYK","MDLZ","ISRG","GILD","ADI","REGN","VRTX",
    "CB","MMC","PLD","AMAT","NOW","PYPL","CI","USB","DUK","SO",
    "SLB","CL","MO","CME","ICE","AON","EQIX","SHW","APD","FCX",
    "NSC","PNC","MCK","ORLY","AZO","KLAC","ROP","MCHP","ADSK","SNPS",
    "CDNS","FTNT","MSCI","TTD","CRWD","PANW","DDOG","ZS","NET","ABNB"
]
SECTORS = {
    "AAPL":"Tech","MSFT":"Tech","AMZN":"Consumer","NVDA":"Tech","GOOGL":"Tech",
    "META":"Tech","TSLA":"Consumer","BRK.B":"Financial","UNH":"Health","JNJ":"Health",
    "V":"Financial","XOM":"Energy","JPM":"Financial","PG":"Staples","MA":"Financial",
    "HD":"Consumer","CVX":"Energy","MRK":"Health","ABBV":"Health","LLY":"Health",
    "PEP":"Staples","KO":"Staples","COST":"Staples","AVGO":"Tech","MCD":"Consumer",
    "WMT":"Staples","TMO":"Health","CSCO":"Tech","ACN":"Tech","ABT":"Health",
    "DHR":"Health","CRM":"Tech","NEE":"Utilities","LIN":"Materials","TXN":"Tech",
    "PM":"Staples","UNP":"Industrial","BMY":"Health","QCOM":"Tech","RTX":"Industrial",
    "AMGN":"Health","HON":"Industrial","LOW":"Consumer","INTC":"Tech","SCHW":"Financial",
    "INTU":"Tech","ELV":"Health","AMD":"Tech","CAT":"Industrial","DE":"Industrial",
    "GS":"Financial","BLK":"Financial","ADP":"Tech","SYK":"Health","MDLZ":"Staples",
    "ISRG":"Health","GILD":"Health","ADI":"Tech","REGN":"Health","VRTX":"Health",
    "CB":"Financial","MMC":"Financial","PLD":"RealEstate","AMAT":"Tech","NOW":"Tech",
    "PYPL":"Tech","CI":"Health","USB":"Financial","DUK":"Utilities","SO":"Utilities",
    "SLB":"Energy","CL":"Staples","MO":"Staples","CME":"Financial","ICE":"Financial",
    "AON":"Financial","EQIX":"RealEstate","SHW":"Materials","APD":"Materials","FCX":"Materials",
    "NSC":"Industrial","PNC":"Financial","MCK":"Health","ORLY":"Consumer","AZO":"Consumer",
    "KLAC":"Tech","ROP":"Industrial","MCHP":"Tech","ADSK":"Tech","SNPS":"Tech",
    "CDNS":"Tech","FTNT":"Tech","MSCI":"Financial","TTD":"Tech","CRWD":"Tech",
    "PANW":"Tech","DDOG":"Tech","ZS":"Tech","NET":"Tech","ABNB":"Consumer"
}

FEATURE_NAMES = [
    "momentum","ret_from_open","rel_volume","vwap_dist","vwap_slope",
    "orb_strength","atr_reach","realized_vol","trend_str","rsi","range_expansion",
    "hours_left",
    "rank_momentum","rank_ret","rank_volume","rank_vwap","rank_slope",
    "rank_orb","rank_atr_inv","rank_vol","rank_trend","rank_rsi","rank_range"
]

for d in [DATA_DIR, MODEL_DIR, OUTCOME_DIR, SCAN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════
models = {}
calibrators = {}
model_meta = {}
last_scans = {}
training_in_progress = False
training_progress = {"phase":"idle","pct":0,"message":""}
sweep_in_progress = False
sweep_progress = {"phase":"idle","current":0,"total":0,"message":"","currentTP":None,"currentSL":None}

STATUS_PATH = DATA_DIR / "status.json"
SWEEP_RESULTS_PATH = DATA_DIR / "sweep_results.json"

def load_sweep_results():
    try: return json.loads(SWEEP_RESULTS_PATH.read_text())
    except: return {"grid":[],"startedAt":None,"completedAt":None}
def save_sweep_results(r): SWEEP_RESULTS_PATH.write_text(json.dumps(r,indent=2))
def load_status():
    try: return json.loads(STATUS_PATH.read_text())
    except: return {"trained":False,"trainDate":None,"outcomeDays":0,"daysSinceRetrain":0}
def save_status(s): STATUS_PATH.write_text(json.dumps(s,indent=2))
status = load_status()

def load_models():
    global models, calibrators, model_meta
    for h in SCAN_HOURS:
        mp = MODEL_DIR / f"model_{h}.txt"
        cp = MODEL_DIR / f"calibrator_{h}.pkl"
        mtp = MODEL_DIR / f"meta_{h}.json"
        if mp.exists():
            models[h] = lgb.Booster(model_file=str(mp))
            log.info(f"Loaded model {h}:00")
        if cp.exists():
            calibrators[h] = pickle.loads(cp.read_bytes())
        if mtp.exists():
            model_meta[h] = json.loads(mtp.read_text())
load_models()

LAST_SCAN_PATH = DATA_DIR / "last_scans.json"
try: last_scans = json.loads(LAST_SCAN_PATH.read_text())
except: last_scans = {}

# ═══════════════════════════════════════════════════════════════════
# TIME / ALPACA HELPERS
# ═══════════════════════════════════════════════════════════════════
def now_et(): return datetime.now(ET)
def today_et(): return now_et().strftime("%Y-%m-%d")
def hour_et(): return now_et().hour
def market_open():
    n = now_et()
    if n.weekday() >= 5: return False
    return 570 <= n.hour*60+n.minute <= 960
def has_creds():
    return bool(ALPACA_KEY and ALPACA_SECRET and ALPACA_KEY != "your_alpaca_api_key_here")

def sleep(ms): import time as t; t.sleep(ms)
def chunk(a,n):
    o=[]
    for i in range(0,len(a),n): o.append(a[i:i+n])
    return o

def alpaca_client():
    return httpx.Client(base_url=ALPACA_URL,
        headers={"APCA-API-KEY-ID":ALPACA_KEY,"APCA-API-SECRET-KEY":ALPACA_SECRET},
        timeout=30.0)

def fetch_bars(client, symbols, timeframe, start, end):
    all_bars = defaultdict(list)
    for batch in chunk(symbols, 50):
        syms = ",".join(batch)
        pt = None; pg = 0
        while True:
            params = {"symbols":syms,"timeframe":timeframe,"start":start,"end":end,
                      "limit":"10000","adjustment":"split","feed":"sip","sort":"asc"}
            if pt: params["page_token"] = pt
            r = client.get("/v2/stocks/bars", params=params)
            if r.status_code == 429: time.sleep(3); continue
            r.raise_for_status()
            data = r.json()
            for sym, bars in (data.get("bars") or {}).items():
                all_bars[sym].extend(bars)
            pt = data.get("next_page_token"); pg += 1
            if not pt or pg > 100: break
            time.sleep(0.25)
        time.sleep(0.3)
    return dict(all_bars)

def fetch_snapshots(client, symbols):
    snaps = {}
    for batch in chunk(symbols, 100):
        r = client.get("/v2/stocks/snapshots", params={"symbols":",".join(batch),"feed":"sip"})
        if r.status_code == 200: snaps.update(r.json())
        time.sleep(0.3)
    return snaps

def bar_to_et_minutes(b):
    """Convert bar timestamp to ET minutes-since-midnight."""
    try:
        dt = datetime.fromisoformat(b["t"].replace("Z","+00:00")).astimezone(ET)
        return dt.hour * 60 + dt.minute
    except:
        return None

# ═══════════════════════════════════════════════════════════════════
# FIRST-PASSAGE LABEL: does price hit TP before SL?
# ═══════════════════════════════════════════════════════════════════
def compute_trade_outcome(entry_price, bars_after_entry, tp_pct=None, sl_pct=None):
    """
    Walk bars in order. Check each bar against TP/SL barriers.
    tp_pct/sl_pct default to global TP_PCT/SL_PCT if not provided.

    Returns: (outcome, pnl_pct, exit_reason)
    """
    if tp_pct is None: tp_pct = TP_PCT
    if sl_pct is None: sl_pct = SL_PCT
    tp_price = entry_price * (1 + tp_pct)
    sl_price = entry_price * (1 - sl_pct)

    for b in bars_after_entry:
        bmin = bar_to_et_minutes(b)

        # Force close at 15:55
        if bmin is not None and bmin >= FORCED_CLOSE_MIN:
            pnl = (b["c"] - entry_price) / entry_price
            return (1 if pnl > 0 else 0, round(pnl * 100, 3), "close_15:55")

        hit_tp = b["h"] >= tp_price
        hit_sl = b["l"] <= sl_price

        if hit_tp and hit_sl:
            # Both barriers in same bar — use open to disambiguate
            if b["o"] >= entry_price:
                return (1, round(tp_pct * 100, 3), "tp")
            else:
                return (0, round(-sl_pct * 100, 3), "sl")
        elif hit_tp:
            return (1, round(tp_pct * 100, 3), "tp")
        elif hit_sl:
            return (0, round(-sl_pct * 100, 3), "sl")

    # No barrier hit, no 15:55 bar — use last bar close
    if bars_after_entry:
        pnl = (bars_after_entry[-1]["c"] - entry_price) / entry_price
        return (1 if pnl > 0 else 0, round(pnl * 100, 3), "eod")
    return (0, 0.0, "no_data")

# ═══════════════════════════════════════════════════════════════════
# FEATURE COMPUTATION
# ═══════════════════════════════════════════════════════════════════
def compute_features(bars, daily_bars, current_price, open_price, scan_hour):
    if len(bars) < 3: return None
    hours_left = 16 - scan_hour

    tail = bars[-3:]
    momentum = (tail[-1]["c"] - tail[0]["o"]) / tail[0]["o"] if tail[0]["o"] > 0 else 0
    ret_from_open = (current_price - open_price) / open_price if open_price > 0 else 0

    avg_bv = sum(b["v"] for b in bars) / len(bars)
    rel_volume = 1.0
    if daily_bars and len(daily_bars) >= 2:
        adv = sum(d["v"] for d in daily_bars[-5:]) / min(5, len(daily_bars))
        exp = adv / 78
        if exp > 0: rel_volume = avg_bv / exp

    vn = sum((b["h"]+b["l"]+b["c"])/3 * b["v"] for b in bars)
    vd = sum(b["v"] for b in bars)
    vwap = vn/vd if vd > 0 else current_price
    vwap_dist = (current_price - vwap) / vwap if vwap > 0 else 0

    vwap_slope = 0.0
    if len(bars) >= 6:
        t = len(bars)//3
        n1 = sum((b["h"]+b["l"]+b["c"])/3*b["v"] for b in bars[:t])
        d1 = sum(b["v"] for b in bars[:t])
        n2 = sum((b["h"]+b["l"]+b["c"])/3*b["v"] for b in bars[:t*2])
        d2 = sum(b["v"] for b in bars[:t*2])
        v1, v2 = (n1/d1 if d1>0 else current_price), (n2/d2 if d2>0 else current_price)
        vwap_slope = (v2-v1)/v1 if v1>0 else 0

    orb = bars[:min(6,len(bars))]
    orb_h, orb_l = max(b["h"] for b in orb), min(b["l"] for b in orb)
    orb_range = orb_h - orb_l
    orb_strength = (current_price - orb_h)/orb_range if orb_range > 0 else 0

    atr = current_price * 0.015
    if daily_bars and len(daily_bars) >= 5:
        trs = [max(daily_bars[i]["h"]-daily_bars[i]["l"],
                    abs(daily_bars[i]["h"]-daily_bars[i-1]["c"]),
                    abs(daily_bars[i]["l"]-daily_bars[i-1]["c"]))
               for i in range(1, len(daily_bars))]
        atr = np.mean(trs[-5:])
    target = current_price * TP_PCT
    atr_scaled = atr * math.sqrt(hours_left/6.5) if hours_left > 0 else atr*0.1
    atr_reach = target/atr_scaled if atr_scaled > 0 else 2.0

    rets = [math.log(bars[i]["c"]/bars[i-1]["c"]) for i in range(1,len(bars)) if bars[i-1]["c"]>0]
    realized_vol = np.std(rets)*math.sqrt(78) if len(rets)>1 else 0

    trend_str = 0.0
    if len(bars) >= 10:
        half = len(bars)//2
        trend_str = (np.mean([b["c"] for b in bars[-half:]]) / np.mean([b["c"] for b in bars[:half]]) - 1)

    rsi = 50.0
    if len(bars) >= 15:
        gains = [max(0, bars[i]["c"]-bars[i-1]["c"]) for i in range(len(bars)-14, len(bars))]
        losses = [max(0, bars[i-1]["c"]-bars[i]["c"]) for i in range(len(bars)-14, len(bars))]
        ag, al = np.mean(gains), np.mean(losses)
        rsi = 100 - (100/(1+ag/al)) if al > 0 else 100

    last_r = (bars[-1]["h"]-bars[-1]["l"])/bars[-1]["c"] if bars[-1]["c"]>0 else 0
    avg_r = np.mean([(b["h"]-b["l"])/b["c"] for b in bars[-10:] if b["c"]>0]) or 1
    range_expansion = last_r/avg_r if avg_r > 0 else 1

    return {
        "momentum":momentum,"ret_from_open":ret_from_open,"rel_volume":rel_volume,
        "vwap_dist":vwap_dist,"vwap_slope":vwap_slope,"orb_strength":orb_strength,
        "atr_reach":atr_reach,"realized_vol":realized_vol,"trend_str":trend_str,
        "rsi":rsi,"range_expansion":range_expansion,"hours_left":hours_left
    }

def add_ranks(features_list):
    n = len(features_list)
    if n < 2: return features_list
    def pr(vals):
        arr = np.array(vals); o = arr.argsort().argsort()
        return o / (n-1)
    ranks = {
        "rank_momentum": pr([f["momentum"] for f in features_list]),
        "rank_ret":      pr([f["ret_from_open"] for f in features_list]),
        "rank_volume":   pr([f["rel_volume"] for f in features_list]),
        "rank_vwap":     pr([f["vwap_dist"] for f in features_list]),
        "rank_slope":    pr([f["vwap_slope"] for f in features_list]),
        "rank_orb":      pr([f["orb_strength"] for f in features_list]),
        "rank_atr_inv":  pr([-f["atr_reach"] for f in features_list]),
        "rank_vol":      pr([f["realized_vol"] for f in features_list]),
        "rank_trend":    pr([f["trend_str"] for f in features_list]),
        "rank_rsi":      pr([50-abs(f["rsi"]-55) for f in features_list]),
        "rank_range":    pr([f["range_expansion"] for f in features_list]),
    }
    for i in range(n):
        for k, v in ranks.items(): features_list[i][k] = float(v[i])
    return features_list

def feat_to_arr(f):
    return np.array([f.get(n, 0) for n in FEATURE_NAMES])

# ═══════════════════════════════════════════════════════════════════
# TRAINING — FIRST-PASSAGE LABELS
# ═══════════════════════════════════════════════════════════════════
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
BARS_DAILY_CACHE = CACHE_DIR / "bars_daily.pkl"
BARS_INTRADAY_CACHE = CACHE_DIR / "bars_intraday.pkl"
CACHE_MAX_AGE_HOURS = 24

def cache_age_hours(path):
    if not path.exists(): return 999
    age_sec = time.time() - path.stat().st_mtime
    return age_sec / 3600

def run_training(tp_pct=None, sl_pct=None):
    """
    Train models with optional override TP/SL. If tp_pct/sl_pct are provided,
    labels are recomputed with those barriers. Bar data is cached on disk so
    only the first training fetches from Alpaca; subsequent trainings with
    different TP/SL skip the fetch.
    """
    global models, calibrators, model_meta, training_in_progress, training_progress, status
    if training_in_progress: return
    training_in_progress = True
    training_progress = {"phase":"starting","pct":0,"message":"Starting..."}

    # Use globals as defaults; allow override
    use_tp = tp_pct if tp_pct is not None else TP_PCT
    use_sl = sl_pct if sl_pct is not None else SL_PCT

    try:
        # ─── Load or fetch bar data ──────────────────────────────────
        daily_age = cache_age_hours(BARS_DAILY_CACHE)
        intra_age = cache_age_hours(BARS_INTRADAY_CACHE)
        cache_fresh = daily_age < CACHE_MAX_AGE_HOURS and intra_age < CACHE_MAX_AGE_HOURS

        if cache_fresh:
            training_progress = {"phase":"loading_cache","pct":5,
                "message":f"Loading cached bars (age {intra_age:.1f}h)..."}
            log.info(f"Using cached bars (daily age {daily_age:.1f}h, intraday age {intra_age:.1f}h)")
            daily_bars = pickle.loads(BARS_DAILY_CACHE.read_bytes())
            intraday = pickle.loads(BARS_INTRADAY_CACHE.read_bytes())
        else:
            client = alpaca_client()
            end_date = today_et()
            start_obj = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=380)
            start_date = start_obj.strftime("%Y-%m-%d")

            training_progress = {"phase":"fetch_daily","pct":3,"message":"Fetching daily bars..."}
            daily_bars = fetch_bars(client, TICKERS, "1Day", start_date, end_date)

            training_progress = {"phase":"fetch_intraday","pct":8,"message":"Fetching 12 months of 5-min bars (10-15 min)..."}
            intraday = fetch_bars(client, TICKERS, "5Min",
                                  f"{start_date}T09:30:00-04:00", f"{end_date}T16:00:00-04:00")
            client.close()

            training_progress = {"phase":"caching","pct":44,"message":"Caching bars to disk..."}
            BARS_DAILY_CACHE.write_bytes(pickle.dumps(daily_bars))
            BARS_INTRADAY_CACHE.write_bytes(pickle.dumps(intraday))
            log.info(f"Cached {sum(len(v) for v in intraday.values())} intraday bars to disk")

        # ─── Group bars by ticker+date ───────────────────────────────
        training_progress = {"phase":"grouping","pct":45,"message":"Grouping bars by date..."}
        by_td = defaultdict(lambda: defaultdict(list))
        for ticker in TICKERS:
            for b in intraday.get(ticker, []):
                by_td[ticker][b["t"][:10]].append(b)

        all_dates = sorted(set(d for t in by_td for d in by_td[t]))
        log.info(f"Training: {len(all_dates)} dates, TP={use_tp*100:.2f}% / SL={use_sl*100:.2f}%")

        # ─── Build training dataset with the specified TP/SL ─────────
        training_progress = {"phase":"features","pct":50,
            "message":f"Computing outcomes (TP {use_tp*100:.2f}% / SL {use_sl*100:.2f}%)..."}
        rows_per_hour = defaultdict(list)

        for di, date in enumerate(all_dates):
            for scan_hour in SCAN_HOURS:
                scan_min = scan_hour * 60
                date_features, date_meta = [], []

                for ticker in TICKERS:
                    day_bars = by_td[ticker].get(date, [])
                    if len(day_bars) < 12: continue

                    before, after = [], []
                    for b in day_bars:
                        bm = bar_to_et_minutes(b)
                        if bm is None: continue
                        if bm < scan_min: before.append(b)
                        else: after.append(b)

                    if len(before) < 3 or len(after) < 2: continue

                    entry_price = after[0]["o"]
                    feature_price = before[-1]["c"]
                    open_price = day_bars[0]["o"]
                    daily_up_to = [d for d in daily_bars.get(ticker,[]) if d["t"][:10] < date][-10:]

                    feat = compute_features(before, daily_up_to, feature_price, open_price, scan_hour)
                    if feat is None: continue

                    # First-passage outcome with the requested TP/SL
                    outcome, pnl, reason = compute_trade_outcome(entry_price, after[1:], tp_pct=use_tp, sl_pct=use_sl)

                    date_features.append(feat)
                    date_meta.append({"ticker":ticker,"label":outcome,"pnl":pnl,"reason":reason,"date":date})

                if len(date_features) >= 10:
                    add_ranks(date_features)
                    for j in range(len(date_features)):
                        date_features[j]["label"] = date_meta[j]["label"]
                        date_features[j]["date"] = date_meta[j]["date"]
                        date_features[j]["pnl"] = date_meta[j]["pnl"]
                        date_features[j]["reason"] = date_meta[j]["reason"]
                        rows_per_hour[scan_hour].append(date_features[j])

            if (di+1) % 10 == 0:
                training_progress = {"phase":"features","pct":50+int((di/len(all_dates))*35),
                    "message":f"Processed {di+1}/{len(all_dates)} days..."}

        training_progress = {"phase":"training","pct":87,"message":"Training LightGBM models..."}
        new_models, new_cals, new_meta = {}, {}, {}

        for h in SCAN_HOURS:
            rows = rows_per_hour[h]
            if len(rows) < 200:
                log.warning(f"{h}:00 only {len(rows)} samples, skip"); continue

            df = pd.DataFrame(rows)
            dates = sorted(df["date"].unique())
            split = int(len(dates) * 0.8)
            train_dates, val_dates = set(dates[:split]), set(dates[split:])

            train_df = df[df["date"].isin(train_dates)]
            val_df = df[df["date"].isin(val_dates)]

            X_tr, y_tr = train_df[FEATURE_NAMES].values, train_df["label"].values
            X_va, y_va = val_df[FEATURE_NAMES].values, val_df["label"].values

            win_rate_train = y_tr.mean()
            win_rate_val = y_va.mean()

            log.info(f"{h}:00 — train {len(train_df)} (WR {win_rate_train:.3f}), val {len(val_df)} (WR {win_rate_val:.3f})")

            ts = lgb.Dataset(X_tr, y_tr, feature_name=FEATURE_NAMES)
            vs = lgb.Dataset(X_va, y_va, feature_name=FEATURE_NAMES, reference=ts)

            params = {
                "objective":"binary","metric":"binary_logloss",
                "boosting_type":"gbdt","num_leaves":31,"learning_rate":0.05,
                "feature_fraction":0.8,"bagging_fraction":0.8,"bagging_freq":5,
                "min_child_samples":20,"verbose":-1
            }
            model = lgb.train(params, ts, num_boost_round=500, valid_sets=[vs],
                              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

            val_probs = model.predict(X_va)
            auc = roc_auc_score(y_va, val_probs) if len(set(y_va)) > 1 else 0

            # Precision at top-10 per val day (win rate of top 10 ranked)
            val_df = val_df.copy()
            val_df["prob"] = val_probs
            p10_list = []
            for d in val_dates:
                day = val_df[val_df["date"]==d].nlargest(10,"prob")
                if len(day) >= 10: p10_list.append(day["label"].mean())
            avg_p10 = np.mean(p10_list) if p10_list else 0

            # Average P&L of top-10 per day
            val_df_pnl = val_df.copy()
            pnl10_list = []
            for d in val_dates:
                day = val_df_pnl[val_df_pnl["date"]==d].nlargest(10,"prob")
                if len(day) >= 10: pnl10_list.append(day["pnl"].mean())
            avg_pnl10 = np.mean(pnl10_list) if pnl10_list else 0

            # EV at various thresholds
            # Break-even threshold for asymmetric barriers: TP*P = SL*(1-P) → P = SL/(SL+TP)
            breakeven_p = use_sl / (use_sl + use_tp)

            # EV at various probability thresholds
            def ev_at(thresh):
                subset = val_df[val_df["prob"] >= thresh]
                if len(subset) == 0: return 0, 0
                return subset["pnl"].mean(), len(subset)
            ev_at_be, n_at_be = ev_at(breakeven_p)
            ev_at_be5, n_at_be5 = ev_at(breakeven_p + 0.05)
            ev_at_50 = val_df[val_df["prob"]>=0.5]["pnl"].mean() if len(val_df[val_df["prob"]>=0.5])>0 else 0
            n_above_50 = len(val_df[val_df["prob"]>=0.5])
            ev_at_55 = val_df[val_df["prob"]>=0.55]["pnl"].mean() if len(val_df[val_df["prob"]>=0.55])>0 else 0
            n_above_55 = len(val_df[val_df["prob"]>=0.55])

            # Exit reason breakdown (val)
            val_reasons = val_df["reason"].value_counts().to_dict() if "reason" in val_df.columns else {}

            # Isotonic calibration
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.95)
            cal.fit(val_probs, y_va)

            # Feature importance
            imp = dict(zip(FEATURE_NAMES, model.feature_importance("gain").tolist()))
            ti = sum(imp.values()) or 1
            imp = {k: round(v/ti, 4) for k,v in imp.items()}

            model.save_model(str(MODEL_DIR / f"model_{h}.txt"))
            (MODEL_DIR / f"calibrator_{h}.pkl").write_bytes(pickle.dumps(cal))

            meta = {
                "scan_hour":h,
                "train_samples":len(train_df),"val_samples":len(val_df),
                "train_dates":len(train_dates),"val_dates":len(val_dates),
                "train_win_rate":round(float(win_rate_train),4),
                "val_win_rate":round(float(win_rate_val),4),
                "auc":round(auc,4),
                "avg_win_rate_top10":round(float(avg_p10),4),
                "avg_pnl_top10":round(float(avg_pnl10),3),
                "ev_above_50pct":round(float(ev_at_50),3),
                "n_above_50pct":int(n_above_50),
                "ev_above_55pct":round(float(ev_at_55),3),
                "n_above_55pct":int(n_above_55),
                "breakeven_threshold":round(breakeven_p,3),
                "ev_above_breakeven":round(float(ev_at_be),3),
                "n_above_breakeven":int(n_at_be),
                "ev_above_breakeven_plus5":round(float(ev_at_be5),3),
                "n_above_breakeven_plus5":int(n_at_be5),
                "val_exit_reasons":val_reasons,
                "importance":imp,
                "trained_at":datetime.now(ET).isoformat(),
                "best_iteration":model.best_iteration,
                "tp_pct":use_tp*100, "sl_pct":use_sl*100
            }
            (MODEL_DIR / f"meta_{h}.json").write_text(json.dumps(meta, indent=2))

            new_models[h] = model
            new_cals[h] = cal
            new_meta[h] = meta
            log.info(f"{h}:00 — AUC {auc:.3f}, Top10 WR {avg_p10:.3f} (base {win_rate_val:.3f}), Top10 PnL {avg_pnl10:.3f}%, EV@50%: {ev_at_50:.3f}%")

        models.update(new_models)
        calibrators.update(new_cals)
        model_meta.update(new_meta)
        status["trained"] = True
        status["trainDate"] = datetime.now(ET).isoformat()
        status["daysSinceRetrain"] = 0
        status["activeTP"] = use_tp * 100
        status["activeSL"] = use_sl * 100
        save_status(status)

        training_progress = {"phase":"done","pct":100,
            "message":f"Done. {len(new_models)} models trained (TP {use_tp*100:.2f}% / SL {use_sl*100:.2f}%)."}
        log.info(f"Training complete. Active TP/SL: {use_tp*100:.2f}% / {use_sl*100:.2f}%")

    except Exception as e:
        log.error(f"Training failed: {e}", exc_info=True)
        training_progress = {"phase":"error","pct":0,"message":str(e)}
    finally:
        training_in_progress = False

# ═══════════════════════════════════════════════════════════════════
# SWEEP: grid search over TP/SL combinations
# ═══════════════════════════════════════════════════════════════════
# Coarse grid: 3 TP × 5 SL = 15 combinations
SWEEP_TP_VALUES = [0.50, 0.75, 1.00]   # percent
SWEEP_SL_VALUES = [0.50, 0.75, 1.00, 1.25, 1.50]  # percent

def summarize_models_for_sweep(tp, sl):
    """After a training completes, extract sweep-relevant summary from model_meta."""
    summary = {
        "tp_pct": tp, "sl_pct": sl,
        "breakeven": round(sl / (sl + tp) * 100, 2),
        "hours": {},
        "avg_top10_wr": None, "avg_top10_pnl": None, "avg_auc": None,
        "avg_base_wr": None, "avg_edge": None,
        "completedAt": datetime.now(ET).isoformat()
    }
    top10_wrs, top10_pnls, aucs, base_wrs = [], [], [], []
    for h in SCAN_HOURS:
        m = model_meta.get(h)
        if not m: continue
        wr10 = m.get("avg_win_rate_top10", 0) * 100
        pnl10 = m.get("avg_pnl_top10", 0)
        auc = m.get("auc", 0)
        base = m.get("val_win_rate", 0) * 100
        edge = wr10 - summary["breakeven"]
        summary["hours"][str(h)] = {
            "top10_wr": round(wr10, 2),
            "top10_pnl": round(pnl10, 3),
            "auc": round(auc, 4),
            "base_wr": round(base, 2),
            "edge": round(edge, 2)
        }
        top10_wrs.append(wr10); top10_pnls.append(pnl10)
        aucs.append(auc); base_wrs.append(base)

    if top10_wrs:
        summary["avg_top10_wr"] = round(float(np.mean(top10_wrs)), 2)
        summary["avg_top10_pnl"] = round(float(np.mean(top10_pnls)), 3)
        summary["avg_auc"] = round(float(np.mean(aucs)), 4)
        summary["avg_base_wr"] = round(float(np.mean(base_wrs)), 2)
        summary["avg_edge"] = round(summary["avg_top10_wr"] - summary["breakeven"], 2)
    return summary

def run_sweep(resume=True):
    """
    Grid search over TP × SL combinations. Trains a full model suite for each
    combination, records summary metrics, and saves progress to disk after
    each cell so it can resume after crashes.
    """
    global sweep_in_progress, sweep_progress, training_in_progress
    if sweep_in_progress: return
    if training_in_progress: return
    sweep_in_progress = True

    # Build full grid
    grid_cells = [(tp, sl) for tp in SWEEP_TP_VALUES for sl in SWEEP_SL_VALUES]
    total = len(grid_cells)

    # Load existing results for resume
    existing = load_sweep_results() if resume else {"grid":[],"startedAt":None,"completedAt":None}
    completed_keys = {f"{r['tp_pct']}_{r['sl_pct']}" for r in existing.get("grid",[])}
    if not existing.get("startedAt") or not resume:
        existing = {"grid":[], "startedAt":datetime.now(ET).isoformat(), "completedAt":None,
                    "gridShape":{"tpValues":SWEEP_TP_VALUES, "slValues":SWEEP_SL_VALUES}}
        completed_keys = set()

    log.info(f"Sweep: {total} cells, {len(completed_keys)} already complete, {total-len(completed_keys)} to run")

    try:
        for idx, (tp, sl) in enumerate(grid_cells):
            key = f"{tp}_{sl}"
            if key in completed_keys:
                log.info(f"Sweep cell {idx+1}/{total}: TP {tp}% / SL {sl}% — skipping (cached)")
                continue

            sweep_progress = {
                "phase":"running","current":idx+1,"total":total,
                "currentTP":tp,"currentSL":sl,
                "message":f"Cell {idx+1}/{total}: TP {tp}% / SL {sl}% (break-even {sl/(sl+tp)*100:.1f}%)"
            }

            # Call run_training synchronously. It sets training_in_progress=True
            # during its run, so we wait for it to finish.
            log.info(f"Sweep cell {idx+1}/{total}: starting TP={tp}% SL={sl}%")
            run_training(tp_pct=tp/100.0, sl_pct=sl/100.0)

            # After training, extract summary from model_meta
            cell_summary = summarize_models_for_sweep(tp, sl)
            existing["grid"].append(cell_summary)
            save_sweep_results(existing)
            log.info(f"Sweep cell {idx+1}/{total} done: avg top10 WR {cell_summary['avg_top10_wr']}%, "
                     f"breakeven {cell_summary['breakeven']}%, edge {cell_summary['avg_edge']}%")

        existing["completedAt"] = datetime.now(ET).isoformat()
        save_sweep_results(existing)
        sweep_progress = {"phase":"done","current":total,"total":total,
            "message":f"Sweep complete. {total} cells evaluated.",
            "currentTP":None,"currentSL":None}
        log.info("Sweep complete.")

    except Exception as e:
        log.error(f"Sweep failed: {e}", exc_info=True)
        sweep_progress = {"phase":"error","current":0,"total":total,"message":str(e),
                         "currentTP":None,"currentSL":None}
    finally:
        sweep_in_progress = False

# ═══════════════════════════════════════════════════════════════════
# LIVE SCAN
# ═══════════════════════════════════════════════════════════════════
def run_live_scan(scan_hour):
    if scan_hour not in models: raise ValueError(f"No model for {scan_hour}:00")
    t0 = time.time()
    client = alpaca_client()
    today = today_et()

    intra = fetch_bars(client, TICKERS, "5Min", f"{today}T09:30:00-04:00", datetime.now(timezone.utc).isoformat())
    snaps = fetch_snapshots(client, TICKERS)
    sd = (datetime.strptime(today,"%Y-%m-%d")-timedelta(days=20)).strftime("%Y-%m-%d")
    daily = fetch_bars(client, TICKERS, "1Day", sd, today)
    client.close()

    raw_feats, stock_info = [], []
    for ticker in TICKERS:
        bars = intra.get(ticker, [])
        snap = snaps.get(ticker, {})
        if len(bars) < 3: continue
        cp = snap.get("latestTrade",{}).get("p") or bars[-1]["c"]
        op = bars[0]["o"]
        feat = compute_features(bars, daily.get(ticker,[]), cp, op, scan_hour)
        if feat is None: continue
        raw_feats.append(feat)
        stock_info.append({"ticker":ticker,"sector":SECTORS.get(ticker,"?"),"price":cp,"open":op})

    if len(raw_feats) < 5: raise ValueError(f"Only {len(raw_feats)} stocks")
    add_ranks(raw_feats)

    # Use active TP/SL from the trained model's meta (falls back to globals)
    meta = model_meta.get(scan_hour, {})
    active_tp = (meta.get("tp_pct", TP_PCT*100)) / 100
    active_sl = (meta.get("sl_pct", SL_PCT*100)) / 100

    X = np.array([feat_to_arr(f) for f in raw_feats])
    raw_probs = models[scan_hour].predict(X)
    cal_probs = calibrators[scan_hour].predict(raw_probs) if scan_hour in calibrators else raw_probs

    results = []
    for i in range(len(raw_feats)):
        si, rf = stock_info[i], raw_feats[i]
        wp = float(cal_probs[i])
        ev = (wp * active_tp - (1 - wp) * active_sl) * 100  # EV per trade in % (asymmetric)
        results.append({
            "rank":0,"ticker":si["ticker"],"sector":si["sector"],
            "price":f"{si['price']:.2f}",
            "changeFromOpen":f"{((si['price']-si['open'])/si['open']*100):.2f}",
            "winProb":round(wp,4),
            "ev":round(ev,3),
            "rawScore":round(float(raw_probs[i]),4),
            "features":{
                "momentum":f"{rf['momentum']:.4f}","relVolume":f"{rf['rel_volume']:.2f}",
                "vwapDist":f"{rf['vwap_dist']*100:.2f}","vwapSlope":f"{rf['vwap_slope']:.4f}",
                "orbStrength":f"{rf['orb_strength']:.3f}","atrReach":f"{rf['atr_reach']:.2f}",
                "realizedVol":f"{rf['realized_vol']:.4f}","trendStr":f"{rf['trend_str']:.4f}",
                "rsi":f"{rf['rsi']:.1f}"
            }
        })

    results.sort(key=lambda x: x["winProb"], reverse=True)
    for i,r in enumerate(results): r["rank"] = i+1

    elapsed = int((time.time()-t0)*1000)

    scan_result = {
        "data":results,"timestamp":datetime.now(ET).isoformat(),"source":"live",
        "elapsed":elapsed,"scanHour":scan_hour,
        "modelAUC":meta.get("auc"),"modelWR10":meta.get("avg_win_rate_top10"),
        "modelPnL10":meta.get("avg_pnl_top10"),
        "scoreRange":{"min":results[-1]["rawScore"],"max":results[0]["rawScore"]} if results else None,
        "tp_pct":active_tp*100,"sl_pct":active_sl*100,
        "breakeven":round(active_sl/(active_sl+active_tp)*100,1)
    }

    sp = SCAN_DIR / f"{today}.json"
    try: saved = json.loads(sp.read_text())
    except: saved = {}
    saved[str(scan_hour)] = results
    sp.write_text(json.dumps(saved))

    last_scans[str(scan_hour)] = scan_result
    LAST_SCAN_PATH.write_text(json.dumps(last_scans, default=str))

    log.info(f"Scan {scan_hour}:00: {len(results)} stocks, {elapsed}ms, "
             f"top5 EV: {[r['ev'] for r in results[:5]]}")
    return scan_result

# ═══════════════════════════════════════════════════════════════════
# OUTCOME RECORDING — FIRST-PASSAGE
# ═══════════════════════════════════════════════════════════════════
def record_outcomes():
    if not has_creds(): return
    today = today_et()
    out_path = OUTCOME_DIR / f"{today}.json"
    if out_path.exists(): log.info(f"Outcomes {today} done."); return

    log.info(f"Recording outcomes {today} (first-passage)...")
    client = alpaca_client()
    try:
        all_bars = fetch_bars(client, TICKERS, "5Min",
                              f"{today}T09:30:00-04:00", f"{today}T16:05:00-04:00")
        client.close()
    except Exception as e:
        log.error(f"Outcome fetch: {e}"); client.close(); return

    sp = SCAN_DIR / f"{today}.json"
    try: today_scans = json.loads(sp.read_text())
    except: today_scans = {}

    outcomes = {}
    for h in SCAN_HOURS:
        outcomes[str(h)] = []
        scan_min = h * 60
        for ticker in TICKERS:
            bars = all_bars.get(ticker, [])
            if len(bars) < 6: continue
            before, after = [], []
            for b in bars:
                bm = bar_to_et_minutes(b)
                if bm is None: continue
                if bm < scan_min: before.append(b)
                else: after.append(b)
            if not before or len(after) < 2: continue

            entry_price = after[0]["o"]
            outcome, pnl, reason = compute_trade_outcome(entry_price, after[1:])

            raw_score = None
            scanned = today_scans.get(str(h), [])
            for s in scanned:
                if s["ticker"] == ticker:
                    raw_score = s.get("rawScore")
                    break

            outcomes[str(h)].append({
                "ticker":ticker,"entryPrice":entry_price,"outcome":outcome,
                "pnl":pnl,"reason":reason,"rawScore":raw_score
            })

    out_path.write_text(json.dumps({"date":today,"outcomes":outcomes,
        "tp_pct":TP_PCT*100,"sl_pct":SL_PCT*100,
        "recordedAt":datetime.now(ET).isoformat()}, indent=2))

    n_files = len(list(OUTCOME_DIR.glob("*.json")))
    status["outcomeDays"] = n_files
    status["daysSinceRetrain"] = status.get("daysSinceRetrain",0) + 1
    save_status(status)
    log.info(f"Outcomes saved. {n_files} days total.")

# ═══════════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════════
app = FastAPI()

@app.get("/api/health")
def health():
    # Prefer active TP/SL from last training; fall back to globals
    active_tp = status.get("activeTP", TP_PCT*100)
    active_sl = status.get("activeSL", SL_PCT*100)
    return {
        "status":"ok","hasCredentials":has_creds(),"marketOpen":market_open(),
        "currentHourET":hour_et(),
        "trained":status.get("trained",False),"trainDate":status.get("trainDate"),
        "outcomeDays":status.get("outcomeDays",0),
        "daysSinceRetrain":status.get("daysSinceRetrain",0),
        "modelsLoaded":list(models.keys()),
        "hasLastScan":bool(last_scans),
        "lastScanHours":list(last_scans.keys()),
        "tp_pct":active_tp,"sl_pct":active_sl,
        "breakeven":round(active_sl/(active_sl+active_tp)*100,1) if active_tp>0 else 50.0
    }

@app.get("/api/scan/{hour}")
def get_scan(hour: int):
    if hour not in SCAN_HOURS: return JSONResponse({"error":"Invalid"},400)
    if market_open() and has_creds() and hour in models:
        try: return run_live_scan(hour)
        except Exception as e: log.error(f"Scan: {e}")
    cached = last_scans.get(str(hour))
    if cached: return {**cached,"source":"cached"}
    return {"data":[],"source":"offline","timestamp":datetime.now(ET).isoformat(),
            "message":"Train model first, then scan during market hours."}

@app.post("/api/scan/{hour}/refresh")
def refresh(hour: int):
    if hour not in SCAN_HOURS: return JSONResponse({"error":"Invalid"},400)
    if not market_open(): return JSONResponse({"error":"Market closed"},400)
    if hour not in models: return JSONResponse({"error":"No model"},400)
    return run_live_scan(hour)

class TrainRequest(BaseModel):
    tp_pct: Optional[float] = None  # as percentage, e.g. 0.95 for +0.95%
    sl_pct: Optional[float] = None  # as percentage, e.g. 1.5 for -1.5%

@app.post("/api/train")
def trigger_train(bg: BackgroundTasks, req: Optional[TrainRequest] = None):
    if training_in_progress: return {"status":"already_running"}
    # Accept empty body (use globals) or JSON body with tp_pct/sl_pct in percentage units
    tp_pct = req.tp_pct if req and req.tp_pct is not None else None
    sl_pct = req.sl_pct if req and req.sl_pct is not None else None
    # Convert from percentage (0.95) to decimal (0.0095)
    tp = tp_pct / 100.0 if tp_pct is not None else None
    sl = sl_pct / 100.0 if sl_pct is not None else None
    # Bounds check: prevent absurd values
    if tp is not None and not (0.001 <= tp <= 0.05): return JSONResponse({"error":"tp_pct must be 0.1-5.0"},400)
    if sl is not None and not (0.001 <= sl <= 0.05): return JSONResponse({"error":"sl_pct must be 0.1-5.0"},400)
    bg.add_task(run_training, tp, sl)
    return {"status":"started","tp_pct":tp_pct or TP_PCT*100,"sl_pct":sl_pct or SL_PCT*100}

@app.post("/api/cache/clear")
def clear_cache():
    """Delete cached bar data to force fresh fetch on next training."""
    if training_in_progress: return JSONResponse({"error":"Cannot clear during training"},400)
    deleted = []
    for f in [BARS_DAILY_CACHE, BARS_INTRADAY_CACHE]:
        if f.exists():
            f.unlink()
            deleted.append(f.name)
    return {"status":"ok","deleted":deleted}

@app.get("/api/cache/status")
def cache_status():
    return {
        "daily": {"exists":BARS_DAILY_CACHE.exists(),"age_hours":round(cache_age_hours(BARS_DAILY_CACHE),1)},
        "intraday": {"exists":BARS_INTRADAY_CACHE.exists(),"age_hours":round(cache_age_hours(BARS_INTRADAY_CACHE),1)},
        "max_age_hours":CACHE_MAX_AGE_HOURS
    }

@app.get("/api/training/progress")
def progress():
    return {"inProgress":training_in_progress,**training_progress,
            "meta":{str(h):model_meta[h] for h in model_meta}}

# ─── SWEEP endpoints ──────────────────────────────────────────────
@app.post("/api/sweep")
def trigger_sweep(bg: BackgroundTasks):
    if sweep_in_progress: return {"status":"already_running"}
    if training_in_progress: return JSONResponse({"error":"Training in progress; wait for it to finish"},400)
    bg.add_task(run_sweep, True)  # resume=True
    total = len(SWEEP_TP_VALUES) * len(SWEEP_SL_VALUES)
    return {"status":"started","total_cells":total,
            "grid":{"tp":SWEEP_TP_VALUES,"sl":SWEEP_SL_VALUES}}

@app.post("/api/sweep/reset")
def reset_sweep():
    if sweep_in_progress: return JSONResponse({"error":"Cannot reset during sweep"},400)
    if SWEEP_RESULTS_PATH.exists(): SWEEP_RESULTS_PATH.unlink()
    return {"status":"ok"}

@app.get("/api/sweep/status")
def sweep_status():
    return {"inProgress":sweep_in_progress, **sweep_progress,
            "grid":{"tp":SWEEP_TP_VALUES,"sl":SWEEP_SL_VALUES}}

@app.get("/api/sweep/results")
def sweep_results():
    return load_sweep_results()

@app.get("/api/outcomes/summary")
def outcome_summary():
    files = sorted(OUTCOME_DIR.glob("*.json"))
    if not files: return {"totalDays":0,"recent":[]}
    recent = []
    for f in files[-20:]:
        try: d = json.loads(f.read_text())
        except: continue
        hs = {}
        for h in SCAN_HOURS:
            entries = d.get("outcomes",{}).get(str(h),[])
            scored = sorted([e for e in entries if e.get("rawScore") is not None], key=lambda e:-e["rawScore"])
            top10 = scored[:10]
            wins = sum(1 for e in top10 if e["outcome"]==1)
            avg_pnl = np.mean([e["pnl"] for e in top10]) if top10 else 0
            base_wr = np.mean([e["outcome"] for e in entries]) if entries else 0
            reasons = {}
            for e in entries:
                r = e.get("reason","?")
                reasons[r] = reasons.get(r,0)+1
            hs[str(h)] = {"total":len(entries),"top10wins":wins,
                      "top10pnl":round(avg_pnl,3),"baseWR":round(base_wr*100,1),
                      "reasons":reasons}
        recent.append({"date":d["date"],"hours":hs})
    return {"totalDays":len(files),"recent":recent}

@app.get("/api/diagnostic")
def diagnostic():
    outcome_files = sorted(OUTCOME_DIR.glob("*.json"))
    outcomes = []
    for f in outcome_files[-20:]:
        try: d = json.loads(f.read_text())
        except: continue
        hd = {}
        for h in SCAN_HOURS:
            entries = d.get("outcomes",{}).get(str(h),[])
            scored = sorted([e for e in entries if e.get("rawScore") is not None], key=lambda e:-e["rawScore"])
            t10 = scored[:10]
            hd[str(h)] = {
                "totalStocks":len(entries),
                "baseWinRate":round(np.mean([e["outcome"] for e in entries])*100,1) if entries else None,
                "top10":[{"ticker":e["ticker"],"score":e["rawScore"],"outcome":e["outcome"],"pnl":e["pnl"],"reason":e["reason"]} for e in t10],
                "top10wins":sum(1 for e in t10 if e["outcome"]==1),
                "top10pnl":round(np.mean([e["pnl"] for e in t10]),3) if t10 else 0,
                "reasons":{r:sum(1 for e in entries if e.get("reason")==r) for r in set(e.get("reason","?") for e in entries)}
            }
        outcomes.append({"date":d["date"],"hours":hd})

    scans = {}
    for h_str, scan in last_scans.items():
        scans[h_str] = {
            "timestamp":scan.get("timestamp"),"source":scan.get("source"),
            "scoreRange":scan.get("scoreRange"),
            "top20":(scan.get("data") or [])[:20]
        }

    return JSONResponse({
        "_type":"sp500_scanner_diagnostic","_version":"4.0_first_passage",
        "generatedAt":datetime.now(ET).isoformat(),
        "strategy":{"tp_pct":TP_PCT*100,"sl_pct":SL_PCT*100,"forced_close":"15:55 ET","entry_delay":"1 bar"},
        "server":{
            "hasCredentials":has_creds(),"marketOpen":market_open(),"currentHourET":hour_et(),
            "trained":status.get("trained",False),"trainDate":status.get("trainDate"),
            "outcomeDays":status.get("outcomeDays",0),"daysSinceRetrain":status.get("daysSinceRetrain",0)
        },
        "modelMeta":{str(h):model_meta[h] for h in model_meta},
        "lastScans":scans,
        "outcomes":outcomes,
        "outcomeSummary":{"totalDays":len(outcome_files),
            "dateRange":{"first":outcome_files[0].stem,"last":outcome_files[-1].stem} if outcome_files else None}
    }, headers={"Content-Disposition":f'attachment; filename="scanner_diagnostic_{today_et()}.json"'})

# SPA fallback
dist_path = Path(__file__).parent / "dist"
if dist_path.exists():
    app.mount("/assets", StaticFiles(directory=dist_path/"assets"), name="assets")
    @app.get("/{full_path:path}")
    def spa(full_path: str):
        fp = dist_path / full_path
        if fp.is_file(): return FileResponse(fp)
        return FileResponse(dist_path / "index.html")

# ═══════════════════════════════════════════════════════════════════
# SCHEDULER
# ═══════════════════════════════════════════════════════════════════
scheduler = BackgroundScheduler(timezone=ET)
if has_creds():
    def cron_scan():
        h = hour_et()
        if h in SCAN_HOURS and market_open() and h in models:
            try: run_live_scan(h)
            except Exception as e: log.error(f"Cron scan: {e}")
    scheduler.add_job(cron_scan, "cron", hour="10,11,12,13,14,15", minute=2, day_of_week="mon-fri")
    scheduler.add_job(record_outcomes, "cron", hour=16, minute=12, day_of_week="mon-fri")
    scheduler.start()
    log.info("Scheduler: scans :02, outcomes 16:12 ET")
