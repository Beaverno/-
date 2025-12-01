#!/usr/bin/env python3
# monthly_momentum_alert_tiingo_final.py
# Tiingo + Advanced monthly momentum strategy (fixed & final)

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import ssl
import io
import base64
import traceback
import random
import time

# matplotlib safe default
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# -----------------------------
# Parameters (can be overridden by env)
# -----------------------------
TICKERS = os.environ.get("TICKERS", "SPY,QQQ,VOO,VT,EFA,EEM,TLT,GLD").split(",")
RISK_FREE = os.environ.get("RISK_FREE", "SHY").split(",")
NUM_HOLD = int(os.environ.get("NUM_HOLD", 3))
FEE_RATE = float(os.environ.get("FEE_RATE", 0.001))
MIN_FEE = float(os.environ.get("MIN_FEE", 0.0))
SLIPPAGE_BASE = float(os.environ.get("SLIPPAGE_BASE", 0.0005))
TRANSACTION_COST = FEE_RATE + SLIPPAGE_BASE
MIN_TRADE_USD = float(os.environ.get("MIN_TRADE_USD", 50))
ALLOW_LIMIT_ORDERS = os.environ.get("ALLOW_LIMIT_ORDERS", "True").lower() in ("1","true","yes")
LIMIT_FILL_PROB = float(os.environ.get("LIMIT_FILL_PROB", 0.7))
LIMIT_ADVANTAGE = float(os.environ.get("LIMIT_ADVANTAGE", 0.0002))
TARGET_VOL = float(os.environ.get("TARGET_VOL", 0.10))
MAX_LEVER = float(os.environ.get("MAX_LEVER", 2.0))
SMA_WINDOW = int(os.environ.get("SMA_WINDOW", 200))
START_DATE = os.environ.get("START_DATE", "2005-01-01")
END_DATE = os.environ.get("END_DATE", None)
MONTE_CARLO_RUNS = int(os.environ.get("MONTE_CARLO_RUNS", 500))
MONTE_CARLO_HORIZON_YEARS = float(os.environ.get("MONTE_CARLO_HORIZON_YEARS", 5.0))

RMB_CAPITAL = float(os.environ.get("RMB_CAPITAL", 650000))
USD_RMB_RATE = float(os.environ.get("USD_RMB_RATE", 7.1))
START_USD = RMB_CAPITAL / USD_RMB_RATE

# Email / smtp
EMAIL_SENDER = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECIPIENTS = os.environ.get("EMAIL_RECIPIENT", EMAIL_SENDER).split(",") if os.environ.get("EMAIL_RECIPIENT") else ([EMAIL_SENDER] if EMAIL_SENDER else [])
SMTP_PROVIDER = os.environ.get("SMTP_PROVIDER", "qq").lower()
if SMTP_PROVIDER == "qq":
    SMTP_SERVER, SMTP_PORT = "smtp.qq.com", 465
elif SMTP_PROVIDER == "163":
    SMTP_SERVER, SMTP_PORT = "smtp.163.com", 465
elif SMTP_PROVIDER == "gmail":
    SMTP_SERVER, SMTP_PORT = "smtp.gmail.com", 465
else:
    SMTP_SERVER, SMTP_PORT = "smtp.qq.com", 465

TIINGO_TOKEN = os.environ.get("TIINGO_TOKEN")
if not TIINGO_TOKEN:
    raise ValueError("请在环境变量中设置 TIINGO_TOKEN")

# Safety
if EMAIL_SENDER and not EMAIL_PASSWORD:
    print("WARNING: EMAIL_SENDER set but EMAIL_PASSWORD missing — email will fail if attempted.")

# -----------------------------
# Helpers
# -----------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b

def cagr_from_nav(nav):
    if len(nav) < 2:
        return 0.0
    start = nav.iloc[0]
    end = nav.iloc[-1]
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (end / start) ** (1.0/years) - 1

def max_drawdown(nav):
    roll = nav.cummax()
    dd = nav / roll - 1
    return dd.min(), dd

def estimate_slippage(annual_vol):
    k = 0.5
    return SLIPPAGE_BASE + k * annual_vol

def compute_fee(amount_usd):
    return max(amount_usd * FEE_RATE, MIN_FEE)

def simulate_limit_fill(limit_price, market_price, fill_prob):
    if random.random() < fill_prob:
        return limit_price, True
    else:
        # executed later at market with adverse slippage
        return market_price * (1 + estimate_slippage(0.2)), False

# -----------------------------
# Tiingo download with retry (returns Series)
# -----------------------------
def tiingo_price(ticker, start=START_DATE, end=END_DATE, max_retry=4, wait=3):
    for attempt in range(max_retry):
        try:
            url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
            params = {"token": TIINGO_TOKEN}
            r = requests.get(url, params=params, timeout=15)
            r.raise_for_status()
            df = pd.DataFrame(r.json())
            if df.empty:
                raise RuntimeError(f"{ticker} returned empty")
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            s = df["adjClose"].sort_index()
            # optionally trim by start/end if present in env
            if start:
                s = s[s.index >= pd.to_datetime(start)]
            if end:
                s = s[s.index <= pd.to_datetime(end)]
            return s
        except Exception as e:
            print(f"Tiingo {ticker} download failed (attempt {attempt+1}/{max_retry}): {e}")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {ticker} after {max_retry} attempts")

# -----------------------------
# Build monthly_prices DataFrame
# -----------------------------
price_data = pd.DataFrame()
for tk in TICKERS:
    try:
        s = tiingo_price(tk)
        price_data[tk] = s
    except Exception as e:
        print(e)

if price_data.empty:
    raise RuntimeError("No price data downloaded. Check TIINGO_TOKEN and network.")

# forward/backfill then monthly resample to month-end
price_data = price_data.sort_index().ffill().bfill()
monthly_prices = price_data.resample("M").last()
if monthly_prices.isnull().all().all():
    raise RuntimeError("monthly_prices empty after resample.")

# -----------------------------
# Prepare backtest variables
# -----------------------------
dates = monthly_prices.index
n_months = len(dates)
nav = pd.Series(index=dates, dtype=float)
nav.iloc[0] = START_USD
holdings_history = pd.DataFrame(0.0, index=dates, columns=TICKERS)
cash_history = pd.Series(0.0, index=dates)
trade_log = []

# monthly asset returns series
asset_ret_monthly = monthly_prices.pct_change().fillna(0)
vol_rolling = asset_ret_monthly.rolling(12).std() * np.sqrt(12)

portfolio_returns_hist = []
last_weights = pd.Series(0.0, index=TICKERS)
final_exposure = 1.0  # default, will be set in loop
weights = pd.Series(0.0, index=TICKERS)
selected = []

# -----------------------------
# Backtest loop
# -----------------------------
for i in range(1, n_months):
    date = dates[i]
    prev_date = dates[i-1]
    prices_now = monthly_prices.loc[date]
    prices_prev = monthly_prices.loc[prev_date]

    # momentum calculations (1/3/6/12 months)
    mom1 = (prices_now - monthly_prices.shift(1).loc[date]) / monthly_prices.shift(1).loc[date]
    mom3 = (prices_now - monthly_prices.shift(3).loc[date]) / monthly_prices.shift(3).loc[date]
    mom6 = (prices_now - monthly_prices.shift(6).loc[date]) / monthly_prices.shift(6).loc[date]
    mom12 = (prices_now - monthly_prices.shift(12).loc[date]) / monthly_prices.shift(12).loc[date]
    acc = mom1 - mom3
    mom1 = mom1.fillna(-np.inf); mom3 = mom3.fillna(-np.inf); mom6 = mom6.fillna(-np.inf); mom12 = mom12.fillna(-np.inf); acc = acc.fillna(-np.inf)

    # volatility
    asset_vol = vol_rolling.loc[date].fillna(vol_rolling.mean().mean())

    # score (ranked + volatility adjust)
    rank_m1 = mom1.rank(ascending=False, method="average")
    rank_m3 = mom3.rank(ascending=False, method="average")
    rank_m6 = mom6.rank(ascending=False, method="average")
    rank_m12 = mom12.rank(ascending=False, method="average")
    rank_acc = acc.rank(ascending=False, method="average")
    raw_score = 0.35*rank_m1 + 0.25*rank_m3 + 0.15*rank_m6 + 0.15*rank_m12 + 0.10*rank_acc
    score = (raw_score / (asset_vol + 1e-9)).sort_values(ascending=False)

    # trend filter SMA
    sma = monthly_prices.rolling(SMA_WINDOW).mean().loc[date]
    trend_ok = prices_now > sma
    eligible = [t for t in score.index if trend_ok.get(t, False)]
    selected = [t for t in score.index if t in eligible][:NUM_HOLD]

    # benchmark momentum for dynamic exposure
    benchmark = "SPY" if "SPY" in monthly_prices.columns else monthly_prices.columns[0]
    bench_mom3 = (prices_now[benchmark] - monthly_prices.shift(3).loc[date][benchmark]) / monthly_prices.shift(3).loc[date][benchmark]
    bench_mom12 = (prices_now[benchmark] - monthly_prices.shift(12).loc[date][benchmark]) / monthly_prices.shift(12).loc[date][benchmark]

    exposure_scale = 1.0
    if (bench_mom3 < 0) and (bench_mom12 < 0):
        exposure_scale = 0.0
        selected = RISK_FREE
    elif bench_mom3 < 0:
        exposure_scale = 0.5

    if len(selected) == 0:
        selected = RISK_FREE

    # dynamic weights by score among selected
    sel_scores = score.loc[selected]
    if sel_scores.sum() <= 0 or sel_scores.clip(lower=0).sum() == 0:
        weights = pd.Series(1.0/len(selected), index=selected)
    else:
        raw = sel_scores.clip(lower=0)
        weights = raw / raw.sum()

    # realized vol of selected (last up to 12 months)
    hist_rets = asset_ret_monthly[selected].loc[:date].tail(12)
    if len(hist_rets) >= 2:
        port_rets = (hist_rets * weights.values).sum(axis=1)
        realized_ann_vol = np.std(port_rets) * np.sqrt(12)
    else:
        realized_ann_vol = 0.0

    scale = 1.0
    if realized_ann_vol > 1e-9 and TARGET_VOL > 0:
        scale = TARGET_VOL / realized_ann_vol
        if scale > MAX_LEVER:
            scale = MAX_LEVER

    final_exposure = exposure_scale * scale
    final_exposure = max(0.0, final_exposure)

    # dollar targets pre-cost
    prev_nav = nav.loc[prev_date]
    target_dollar_total = prev_nav * final_exposure
    target_dollars = {t: float(weights.get(t, 0.0) * target_dollar_total) for t in TICKERS}

    # mark to market previous holdings
    prev_hold_vals = holdings_history.loc[prev_date] if i-1 >= 0 else pd.Series(0.0, index=TICKERS)
    mark_to_market_vals = prev_hold_vals * (prices_now / prices_prev)
    cash_before = cash_history.loc[prev_date] if i-1 >= 0 else 0.0
    portfolio_value_before = mark_to_market_vals.sum() + cash_before

    # trades required
    trade_amounts = {t: target_dollars.get(t, 0.0) - mark_to_market_vals.get(t, 0.0) for t in TICKERS}

    # ignore very small trades
    for t in TICKERS:
        if abs(trade_amounts[t]) < MIN_TRADE_USD:
            trade_amounts[t] = 0.0
            target_dollars[t] = mark_to_market_vals.get(t,0.0)

    # execute sells then buys
    executed_changes = {t: 0.0 for t in TICKERS}
    trade_details = []

    def execute_trade(ticker, dollar_amount):
        market_px = prices_now[ticker]
        annual_vol = asset_vol.get(ticker, 0.2)
        slippage_pct = estimate_slippage(annual_vol)
        if dollar_amount == 0:
            return 0.0, 0.0, market_px, False
        if ALLOW_LIMIT_ORDERS:
            sign = 1 if dollar_amount > 0 else -1
            limit_price = market_px * (1 - sign * LIMIT_ADVANTAGE)
            filled_price, filled = simulate_limit_fill(limit_price, market_px, LIMIT_FILL_PROB)
            if not filled:
                executed_price = market_px * (1 + np.sign(dollar_amount) * slippage_pct)
            else:
                executed_price = filled_price
        else:
            executed_price = market_px * (1 + np.sign(dollar_amount) * slippage_pct)
        shares = abs(dollar_amount) / executed_price if executed_price>0 else 0.0
        executed_amount = shares * executed_price * (1.0 if dollar_amount>0 else -1.0)
        fee = compute_fee(abs(executed_amount))
        return executed_amount, fee, executed_price, True

    # sells
    for t, amt in trade_amounts.items():
        if amt < 0:
            executed_amt, fee_amt, exec_price, filled = execute_trade(t, amt)
            executed_changes[t] += executed_amt
            trade_details.append({"date": date, "ticker": t, "side":"sell", "target": amt, "executed": executed_amt, "fee": fee_amt, "price": exec_price})
    # buys
    for t, amt in trade_amounts.items():
        if amt > 0:
            executed_amt, fee_amt, exec_price, filled = execute_trade(t, amt)
            executed_changes[t] += executed_amt
            trade_details.append({"date": date, "ticker": t, "side":"buy", "target": amt, "executed": executed_amt, "fee": fee_amt, "price": exec_price})

    # update holdings values and cash
    for t in TICKERS:
        mark_to_market_vals[t] += executed_changes[t]
    total_fees = sum([d["fee"] for d in trade_details])
    cash_after = portfolio_value_before - mark_to_market_vals.sum() - total_fees

    holdings_history.loc[date] = mark_to_market_vals
    cash_history.loc[date] = cash_after
    nav.loc[date] = mark_to_market_vals.sum() + cash_after

    for d in trade_details:
        trade_log.append(d)

    prev_nav_val = prev_nav
    cur_nav_val = nav.loc[date]
    realized_ret = (cur_nav_val / prev_nav_val) - 1 if prev_nav_val!=0 else 0.0
    portfolio_returns_hist.append(realized_ret)

    last_weights = (holdings_history.loc[date] / nav.loc[date]).fillna(0.0)

# -----------------------------
# Results & statistics
# -----------------------------
nav = nav.fillna(method="ffill")
nav.iloc[0] = START_USD
cum_returns = nav / nav.iloc[0]
cagr = cagr_from_nav(nav)
ann_vol = np.std(portfolio_returns_hist) * np.sqrt(12) if len(portfolio_returns_hist)>1 else 0.0
sharpe = cagr / ann_vol if ann_vol>0 else np.nan
md, dd_series = max_drawdown(cum_returns)

print("Backtest summary:")
print(f"Period: {nav.index[0].date()} to {nav.index[-1].date()}")
print(f"Start USD: {START_USD:.2f}, End USD: {nav.iloc[-1]:.2f}")
print(f"CAGR: {cagr:.2%}, Ann Vol: {ann_vol:.2%}, Sharpe: {sharpe:.3f}, MaxDD: {md:.2%}")

# subperiod stats
def subperiod_stats(nav_series, window_years=5):
    stats = []
    months_window = int(window_years * 12)
    for start in range(0, len(nav_series)-months_window+1, 12):
        sub = nav_series.iloc[start:start+months_window]
        c = cagr_from_nav(sub)
        md_sub, _ = max_drawdown(sub)
        stats.append({"start": sub.index[0].date(), "end": sub.index[-1].date(), "CAGR": c, "MaxDD": md_sub})
    return pd.DataFrame(stats)
sub_stats = subperiod_stats(nav, window_years=5)

# Monte Carlo bootstrap
monthly_rets = pd.Series(portfolio_returns_hist) if len(portfolio_returns_hist)>0 else pd.Series([0.0])
mc_cagrs, mc_maxdds = [], []
horizon_months = int(MONTE_CARLO_HORIZON_YEARS * 12)
for run in range(MONTE_CARLO_RUNS):
    sample = np.random.choice(monthly_rets.values, size=horizon_months, replace=True)
    nav_mc = np.ones(horizon_months+1)
    for j in range(horizon_months):
        nav_mc[j+1] = nav_mc[j] * (1 + sample[j])
    years = MONTE_CARLO_HORIZON_YEARS
    cagr_mc = nav_mc[-1] ** (1.0/years) - 1
    roll_max = np.maximum.accumulate(nav_mc)
    dd = nav_mc / roll_max - 1
    maxdd_mc = dd.min()
    mc_cagrs.append(cagr_mc)
    mc_maxdds.append(maxdd_mc)

# plotting and email body
fig_nav, ax = plt.subplots(figsize=(10,6))
cum_returns.plot(ax=ax)
ax.set_title("策略累计净值 (normalized)")
img_nav = fig_to_base64(fig_nav)

fig_dd, ax = plt.subplots(figsize=(10,6))
dd_series.plot(ax=ax, color='red')
ax.set_title("策略回撤")
img_dd = fig_to_base64(fig_dd)

fig_mc, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].hist(mc_cagrs, bins=30)
ax[0].set_title("Monte Carlo CAGR Distribution")
ax[1].hist(mc_maxdds, bins=30)
ax[1].set_title("Monte Carlo MaxDD Distribution")
img_mc = fig_to_base64(fig_mc)

trade_df = pd.DataFrame(trade_log)
if not trade_df.empty:
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    trade_df = trade_df.sort_values('date')

holdings_snapshot = holdings_history.loc[dates[-1]]

body_html = f"""
<h3>高级月度动能轮动策略回测报告</h3>
<p>回测区间: {nav.index[0].date()} - {nav.index[-1].date()}<br>
本金: {RMB_CAPITAL:,.0f} RMB (~${START_USD:,.2f} USD)<br>
CAGR: {cagr:.2%} &nbsp; AnnVol: {ann_vol:.2%} &nbsp; Sharpe: {sharpe:.2f} &nbsp; MaxDD: {md:.2%}</p>
<h4>本月建议 (Top-{NUM_HOLD}, exposure scale={final_exposure:.2f})</h4>
<ul>
"""
for t in selected:
    w = float(weights.get(t, 0.0))
    alloc_usd = START_USD * w * final_exposure * (1 - TRANSACTION_COST)
    body_html += f"<li>{t}: weight {w:.3f}, est allocation ${alloc_usd:,.2f}</li>"
body_html += "</ul>"

body_html += f"<h4>最新持仓（美元）</h4><pre>{holdings_snapshot.to_string()}</pre>"
body_html += "<h4>回测图</h4>"
body_html += f"<img src='data:image/png;base64,{img_nav}'><br>"
body_html += f"<img src='data:image/png;base64,{img_dd}'><br>"
body_html += f"<h4>蒙特卡洛样本（{MONTE_CARLO_RUNS} 次）</h4><img src='data:image/png;base64,{img_mc}'><br>"

csv_buf = io.StringIO()
if not trade_df.empty:
    trade_df.to_csv(csv_buf, index=False)
trade_csv = csv_buf.getvalue()

# send email
def send_email(subject, html_body, attachments=None):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("Email sender/password not configured - skipping email send.")
        return
    msg = MIMEMultipart()
    msg["From"] = Header("Momentum Bot", "utf-8")
    msg["To"] = Header(", ".join(EMAIL_RECIPIENTS), "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    if attachments:
        from email.mime.base import MIMEBase
        from email import encoders
        for fn, content in attachments.items():
            part = MIMEBase('application', "octet-stream")
            part.set_payload(content.encode("utf-8"))
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{fn}"')
            msg.attach(part)
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context) as server:
            server.set_debuglevel(1)
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())
        print("Report email sent.")
    except Exception as e:
        print("Email send failed:", e)
        traceback.print_exc()

send_email("Momentum Strategy Monthly Report (Final)", body_html, attachments={"trade_log.csv": trade_csv})

# save outputs
out_dir = os.environ.get("OUTPUT_DIR", ".")
os.makedirs(out_dir, exist_ok=True)
nav.to_csv(os.path.join(out_dir,"nav.csv"))
if not trade_df.empty:
    trade_df.to_csv(os.path.join(out_dir,"trade_log.csv"), index=False)
holdings_history.to_csv(os.path.join(out_dir,"holdings_history.csv"))
pd.DataFrame({"mc_cagr": mc_cagrs, "mc_maxdd": mc_maxdds}).to_csv(os.path.join(out_dir,"monte_carlo.csv"), index=False)

print("Finished. Files saved to", out_dir)
