#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monthly_momentum_alert_yfinance_final.py
- 月度动能轮动策略
- 数据源: yfinance + 代理
- Monte Carlo 压力测试
- 邮件发送 HTML+图+附件
- 输出 CSV
Requirements: pandas, numpy, matplotlib, yfinance
Env vars: EMAIL_ADDRESS, EMAIL_PASSWORD (optional), EMAIL_RECIPIENT (optional)
"""
import os, time, math, random, io, base64, traceback
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ssl, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import yfinance as yf

# -------------------------
# Defaults / params
# -------------------------
TICKERS = os.environ.get("TICKERS", "SPY,QQQ,VOO,VT,EFA,EEM,TLT,GLD").split(",")
RISK_FREE = os.environ.get("RISK_FREE", "SHY").split(",")
NUM_HOLD = int(os.environ.get("NUM_HOLD", 3))
FEE_RATE = float(os.environ.get("FEE_RATE", 0.001))
SLIPPAGE_BASE = float(os.environ.get("SLIPPAGE_BASE", 0.0005))
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

EMAIL_SENDER = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECIPIENTS = os.environ.get("EMAIL_RECIPIENT", EMAIL_SENDER).split(",") if os.environ.get("EMAIL_RECIPIENT") else ([EMAIL_SENDER] if EMAIL_SENDER else [])
SMTP_PROVIDER = os.environ.get("SMTP_PROVIDER", "163").lower()

if SMTP_PROVIDER == "163":
    SMTP_SERVER, SMTP_PORT = "smtp.163.com", 465
elif SMTP_PROVIDER == "gmail":
    SMTP_SERVER, SMTP_PORT = "smtp.gmail.com", 465
else:
    SMTP_SERVER, SMTP_PORT = "smtp.qq.com", 465

# Safety
if EMAIL_SENDER and not EMAIL_PASSWORD:
    print("WARNING: EMAIL_SENDER 设置但未提供 EMAIL_PASSWORD，邮件发送会失败。")

# -------------------------
# Utils
# -------------------------
def fig_to_base64_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    b = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b

def estimate_slippage(annual_vol):
    k = 0.5
    return SLIPPAGE_BASE + k * annual_vol

def compute_fee(amount_usd):
    return max(amount_usd * FEE_RATE, 0.0)

# -------------------------
# yfinance downloader with proxy
# -------------------------
PROXY = "http://127.0.0.1:7890"

def yf_price_series(ticker, start=START_DATE, end=END_DATE):
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            proxies={"http": PROXY, "https": PROXY},
            auto_adjust=True
        )
        if df.empty:
            raise ValueError(f"{ticker} 返回空数据")
        s = df['Adj Close'].copy()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:
        raise RuntimeError(f"{ticker} 下载失败: {e}")

# -------------------------
# build price DataFrame
# -------------------------
price_dfs = {}
for t in TICKERS:
    try:
        s = yf_price_series(t, start=START_DATE, end=END_DATE)
        price_dfs[t] = s
    except Exception as e:
        print(f"Failed to fetch {t}: {e}")

if len(price_dfs) == 0:
    raise RuntimeError("No price data downloaded. Check network or proxy.")

price_df = pd.DataFrame(price_dfs).sort_index().ffill().bfill()
monthly_prices = price_df.resample('ME').last()
monthly_prices.dropna(how='all', inplace=True)
if monthly_prices.empty:
    raise RuntimeError("No monthly data after resample — check price data range")

# -------------------------
# prepare returns/vol
# -------------------------
asset_ret_monthly = monthly_prices.pct_change().fillna(0)
vol_rolling = asset_ret_monthly.rolling(12).std() * np.sqrt(12)

# -------------------------
# Backtest
# -------------------------
dates = monthly_prices.index
n_months = len(dates)
nav = pd.Series(index=dates, dtype=float)
nav.iloc[0] = START_USD
holdings_history = pd.DataFrame(0.0, index=dates, columns=TICKERS)
cash_history = pd.Series(0.0, index=dates)
trade_log = []
portfolio_returns_hist = []
last_weights = pd.Series(0.0, index=TICKERS)

for i in range(1, n_months):
    date = dates[i]
    prev_date = dates[i-1]
    prices_now = monthly_prices.loc[date]
    prices_prev = monthly_prices.loc[prev_date]

    # momentum
    mom1 = (prices_now - monthly_prices.shift(1).loc[date]) / monthly_prices.shift(1).loc[date]
    mom3 = (prices_now - monthly_prices.shift(3).loc[date]) / monthly_prices.shift(3).loc[date]
    mom6 = (prices_now - monthly_prices.shift(6).loc[date]) / monthly_prices.shift(6).loc[date]
    mom12 = (prices_now - monthly_prices.shift(12).loc[date]) / monthly_prices.shift(12).loc[date]
    acc = (mom1 - mom3).fillna(-np.inf)
    mom1 = mom1.fillna(-np.inf); mom3 = mom3.fillna(-np.inf); mom6 = mom6.fillna(-np.inf); mom12 = mom12.fillna(-np.inf)

    asset_vol = vol_rolling.loc[date].fillna(vol_rolling.mean().mean())
    rank_m1 = mom1.rank(ascending=False)
    rank_m3 = mom3.rank(ascending=False)
    rank_m6 = mom6.rank(ascending=False)
    rank_m12 = mom12.rank(ascending=False)
    rank_acc = acc.rank(ascending=False)
    raw_score = 0.35*rank_m1 + 0.25*rank_m3 + 0.15*rank_m6 + 0.15*rank_m12 + 0.10*rank_acc
    score = (raw_score / (asset_vol + 1e-9)).sort_values(ascending=False)

    sma = monthly_prices.rolling(SMA_WINDOW).mean().loc[date]
    trend_ok = prices_now > sma
    eligible = [t for t in score.index if trend_ok.get(t, False)]
    selected = [t for t in score.index if t in eligible][:NUM_HOLD]
    
    benchmark = "SPY" if "SPY" in monthly_prices.columns else monthly_prices.columns[0]
    bench_mom3 = (prices_now[benchmark] - monthly_prices.shift(3).loc[date][benchmark]) / monthly_prices.shift(3).loc[date][benchmark]
    bench_mom12 = (prices_now[benchmark] - monthly_prices.shift(12).loc[date][benchmark]) / monthly_prices.shift(12).loc[date][benchmark]
    exposure_scale = 1.0
    if bench_mom3 < 0 and bench_mom12 < 0:
        exposure_scale = 0.0
        selected = RISK_FREE
    elif bench_mom3 < 0:
        exposure_scale = 0.5
    if len(selected) == 0:
        selected = RISK_FREE

    sel_scores = score.loc[selected].clip(lower=0)
    weights = (sel_scores / sel_scores.sum()) if sel_scores.sum() > 0 else pd.Series(1.0/len(selected), index=selected)

    hist_rets = asset_ret_monthly[selected].loc[:date].tail(12)
    if len(hist_rets) >= 2:
        port_rets = (hist_rets * weights.values).sum(axis=1)
        realized_ann_vol = np.std(port_rets) * np.sqrt(12)
    else:
        realized_ann_vol = 0.0
    scale = TARGET_VOL / realized_ann_vol if realized_ann_vol > 1e-9 else 1.0
    scale = min(scale, MAX_LEVER)
    final_exposure = max(0.0, exposure_scale * scale)

    prev_nav = nav.loc[prev_date]
    target_total = prev_nav * final_exposure
    target_dollars = {t: float(weights.get(t,0.0) * target_total) for t in TICKERS}
    prev_hold_vals = holdings_history.loc[prev_date]
    mark_to_market = prev_hold_vals * (prices_now / prices_prev)
    cash_before = cash_history.loc[prev_date]
    portfolio_before = mark_to_market.sum() + cash_before
    trade_amounts = {t: target_dollars.get(t,0.0) - mark_to_market.get(t,0.0) for t in TICKERS}

    for t in TICKERS:
        if abs(trade_amounts[t]) < MIN_TRADE_USD:
            trade_amounts[t] = 0.0
            target_dollars[t] = mark_to_market.get(t,0.0)

    executed_changes = {t: 0.0 for t in TICKERS}
    trade_details = []

    def simulate_execution(ticker, dollar_amount):
        if dollar_amount == 0:
            return 0.0, 0.0, float(monthly_prices.loc[date, ticker])
        market_price = float(monthly_prices.loc[date, ticker])
        annual_vol = float(asset_vol.get(ticker, 0.2))
        slippage_pct = estimate_slippage(annual_vol)
        if ALLOW_LIMIT_ORDERS:
            sign = 1 if dollar_amount > 0 else -1
            limit_price = market_price * (1 - sign * LIMIT_ADVANTAGE)
            if random.random() < LIMIT_FILL_PROB:
                exec_price = limit_price
            else:
                exec_price = market_price * (1 + sign * slippage_pct)
        else:
            exec_price = market_price * (1 + np.sign(dollar_amount) * slippage_pct)
        shares = abs(dollar_amount) / exec_price if exec_price>0 else 0.0
        executed_amount = shares * exec_price * (1.0 if dollar_amount>0 else -1.0)
        fee = compute_fee(abs(executed_amount))
        return executed_amount, fee, exec_price

    for t, amt in trade_amounts.items():
        executed_amt, fee_amt, price = simulate_execution(t, amt)
        executed_changes[t] += executed_amt
        side = "buy" if amt>0 else "sell"
        trade_details.append({"date": date, "ticker": t, "side":side, "target": amt, "executed": executed_amt, "fee": fee_amt, "price": price})

    for t in TICKERS:
        mark_to_market[t] += executed_changes[t]
    total_fees = sum([d["fee"] for d in trade_details])
    cash_after = portfolio_before - mark_to_market.sum() - total_fees
    holdings_history.loc[date] = mark_to_market
    cash_history.loc[date] = cash_after
    nav.loc[date] = mark_to_market.sum() + cash_after
    for d in trade_details:
        trade_log.append(d)
    prev_nav_val = prev_nav
    cur_nav_val = nav.loc[date]
    realized_ret = (cur_nav_val / prev_nav_val) - 1 if prev_nav_val!=0 else 0.0
    portfolio_returns_hist.append(realized_ret)
    last_weights = (holdings_history.loc[date] / nav.loc[date]).fillna(0.0)

# -------------------------
# Results
# -------------------------
nav = nav.fillna(method='ffill')
nav.iloc[0] = START_USD
cum_returns = nav / nav.iloc[0]
def cagr(nav_series):
    years = (nav_series.index[-1] - nav_series.index[0]).days/365.25
    return (nav_series.iloc[-1]/nav_series.iloc[0])**(1.0/years) - 1 if years>0 else 0.0
cagr_val = cagr(cum_returns)
ann_vol = (np.std(portfolio_returns_hist) * np.sqrt(12)) if len(portfolio_returns_hist)>1 else 0.0
sharpe = cagr_val / ann_vol if ann_vol>0 else np.nan
md, dd_series = (cum_returns.cummax() / cum_returns - 1).min(), (cum_returns / cum_returns.cummax() - 1)

# Monte Carlo bootstrap
monthly_rets = np.array(portfolio_returns_hist) if len(portfolio_returns_hist)>0 else np.array([0.0])
mc_cagrs = []
mc_maxdds = []
horizon_months = int(MONTE_CARLO_HORIZON_YEARS*12)
for _ in range(MONTE_CARLO_RUNS):
    sample = np.random.choice(monthly_rets, size=horizon_months, replace=True)
    nav_mc = np.ones(horizon_months+1)
    for j in range(horizon_months):
        nav_mc[j+1] = nav_mc[j] * (1+sample[j])
    years = MONTE_CARLO_HORIZON_YEARS
    mc_cagrs.append(nav_mc[-1]**(1.0/years)-1)
    roll = np.maximum.accumulate(nav_mc)
    mc_maxdds.append((nav_mc/roll - 1).min())

# plots
fig1, ax1 = plt.subplots(figsize=(10,6))
cum_returns.plot(ax=ax1)
ax1.set_title("策略累计净值 (normalized)")
img_nav = fig_to_base64_bytes(fig1)
fig2, ax2 = plt.subplots(figsize=(10,6))
dd_series.plot(ax=ax2, color='red')
ax2.set_title("回撤")
img_dd = fig_to_base64_bytes(fig2)
fig3, ax3 = plt.subplots(1,2,figsize=(12,5))
ax3[0].hist(mc_cagrs, bins=30); ax3[0].set_title("Monte Carlo CAGR")
ax3[1].hist(mc_maxdds, bins=30); ax3[1].set_title("Monte Carlo MaxDD")
img_mc = fig_to_base64_bytes(fig3)

# trade log
trade_df = pd.DataFrame(trade_log)
if not trade_df.empty:
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    trade_df = trade_df.sort_values('date')
holdings_snapshot = holdings_history.loc[dates[-1]]

# Email
body_html = f"""
<h3>月度动能轮动策略（yfinance final）回测报告</h3>
<p>区间: {nav.index[0].date()} - {nav.index[-1].date()}<br/>
本金: {RMB_CAPITAL:,.0f} RMB (~${START_USD:,.2f} USD)<br/>
CAGR: {cagr_val:.2%} &nbsp; AnnVol: {ann_vol:.2%} &nbsp; Sharpe: {sharpe:.3f}</p>
<h4>最新持仓（Top-{NUM_HOLD}）</h4><pre>{holdings_snapshot.to_string()}</pre>
<h4>回测图</h4>
<img src="data:image/png;base64,{img_nav}"><br>
<img src="data:image/png;base64,{img_dd}"><br>
<h4>Monte Carlo ({MONTE_CARLO_RUNS})</h4><img src="data:image/png;base64,{img_mc}">
"""
attachments = {}
if not trade_df.empty:
    attachments['trade_log.csv'] = trade_df.to_csv(index=False)

def send_email(subject, html_body, attachments=None):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("email not configured; skip sending")
        return
    msg = MIMEMultipart()
    msg['From'] = Header("Momentum Bot", "utf-8")
    msg['To'] = Header(", ".join(EMAIL_RECIPIENTS), "utf-8")
    msg['Subject'] = Header(subject, "utf-8")
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    if attachments:
        from email.mime.base import MIMEBase
        from email import encoders
        for fn, content in attachments.items():
            part = MIMEBase('application','octet-stream')
            part.set_payload(content.encode('utf-8'))
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{fn}"')
            msg.attach(part)
    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=ctx) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())
            print("Email sent successfully")
    except Exception as e:
        print(f"Failed to send email: {e}")

send_email(f"月度动能策略报告 {datetime.now().date()}", body_html, attachments)

# save CSV
nav.to_csv("nav.csv")
trade_df.to_csv("trade_log.csv", index=False)

print(f"完成回测。CAGR={cagr_val:.2%}, Sharpe={sharpe:.3f}, 最大回撤={md:.2%}")
