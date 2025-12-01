#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monthly_momentum_alert_tiingo_final.py
最终整合版本:
- Tiingo API 下载（自动重试、时区修正）
- NUM_HOLD = 3
- 手续费 (默认 0.1%)、基础滑点 (默认 0.05%)，按波动率放大
- 限价单 & 市价单执行模拟
- 波动率目标 (TARGET_VOL)
- Monte Carlo 压力测试
- 邮件发送（HTML + 嵌入图像 + trade_log.csv 附件）
- 输出 CSV 文件
"""
import os, time, math, random, io, base64, traceback
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ssl, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

# -------------------------
# 参数 / 环境变量
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

TIINGO_TOKEN = os.environ.get("TIINGO_TOKEN", "")
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

if not TIINGO_TOKEN:
    raise ValueError("请设置 TIINGO_TOKEN 环境变量（Tiingo API token）")
if EMAIL_SENDER and not EMAIL_PASSWORD:
    print("WARNING: EMAIL_SENDER 设置但未提供 EMAIL_PASSWORD，邮件发送会失败。")

# -------------------------
# 工具函数
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

def tiingo_price_series(ticker, start=START_DATE, end=None, max_retry=4, wait=2):
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    headers = {"Content-Type": "application/json"}
    params = {"token": TIINGO_TOKEN}
    attempt = 0
    while attempt < max_retry:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                raise RuntimeError(f"{ticker} 返回空")
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df['date'] = df['date'].dt.tz_convert(None) if df['date'].dt.tz is not None else df['date']
            df.set_index('date', inplace=True)
            col = 'adjClose' if 'adjClose' in df.columns else ('close' if 'close' in df.columns else None)
            if col is None:
                raise RuntimeError(f"{ticker} 没有 adjClose/close 字段")
            s = df[col].sort_index()
            if start:
                s = s[s.index >= pd.to_datetime(start)]
            if end:
                s = s[s.index <= pd.to_datetime(end)]
            return s
        except Exception as e:
            attempt += 1
            print(f"Tiingo {ticker} download failed (attempt {attempt}/{max_retry}): {e}")
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {ticker} after {max_retry} attempts")

# -------------------------
# 下载价格数据
# -------------------------
price_dfs = {}
for t in TICKERS + RISK_FREE:
    try:
        s = tiingo_price_series(t, start=START_DATE, end=END_DATE)
        price_dfs[t] = s
    except Exception as e:
        print(f"Failed to fetch {t}: {e}")

if len(price_dfs) == 0:
    raise RuntimeError("No price data downloaded. Check TIINGO_TOKEN and network.")

price_df = pd.DataFrame(price_dfs).sort_index()
price_df = price_df.ffill().bfill()
monthly_prices = price_df.resample('ME').last()
monthly_prices.dropna(how='all', inplace=True)
if monthly_prices.empty:
    raise RuntimeError("No monthly data after resample — check price data range")

asset_ret_monthly = monthly_prices.pct_change().fillna(0)
vol_rolling = asset_ret_monthly.rolling(12).std() * np.sqrt(12)

# -------------------------
# 回测初始化
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

# -------------------------
# 回测循环
# -------------------------
for i in range(1, n_months):
    date = dates[i]
    prev_date = dates[i-1]
    prices_now = monthly_prices.loc[date]
    prices_prev = monthly_prices.loc[prev_date]

    # 动能指标
    mom1 = (prices_now - monthly_prices.shift(1).loc[date]) / monthly_prices.shift(1).loc[date]
    mom3 = (prices_now - monthly_prices.shift(3).loc[date]) / monthly_prices.shift(3).loc[date]
    mom6 = (prices_now - monthly_prices.shift(6).loc[date]) / monthly_prices.shift(6).loc[date]
    mom12 = (prices_now - monthly_prices.shift(12).loc[date]) / monthly_prices.shift(12).loc[date]
    acc = (mom1 - mom3).fillna(-np.inf)

    mom1 = mom1.fillna(-np.inf); mom3 = mom3.fillna(-np.inf)
    mom6 = mom6.fillna(-np.inf); mom12 = mom12.fillna(-np.inf)
    asset_vol = vol_rolling.loc[date].fillna(vol_rolling.mean().mean())

    rank_m1 = mom1.rank(ascending=False, method='average')
    rank_m3 = mom3.rank(ascending=False, method='average')
    rank_m6 = mom6.rank(ascending=False, method='average')
    rank_m12 = mom12.rank(ascending=False, method='average')
    rank_acc = acc.rank(ascending=False, method='average')
    raw_score = 0.35*rank_m1 + 0.25*rank_m3 + 0.15*rank_m6 + 0.15*rank_m12 + 0.10*rank_acc
    score = (raw_score / (asset_vol + 1e-9)).sort_values(ascending=False)

    sma = monthly_prices.rolling(SMA_WINDOW).mean().loc[date]
    trend_ok = prices_now > sma
    eligible = [t for t in score.index if trend_ok.get(t, False)]
    selected = [t for t in score.index if t in eligible][:NUM_HOLD]

    # 市场趋势调整
    benchmark = "SPY" if "SPY" in monthly_prices.columns else monthly_prices.columns[0]
    bench_mom3 = (prices_now[benchmark] - monthly_prices.shift(3).loc[date][benchmark]) / monthly_prices.shift(3).loc[date][benchmark]
    bench_mom12 = (prices_now[benchmark] - monthly_prices.shift(12).loc[date][benchmark]) / monthly_prices.shift(12).loc[date][benchmark]

    exposure_scale = 1.0
    if bench_mom3 < 0 and bench_mom12 < 0:
        exposure_scale = 0.0
        selected = [t for t in RISK_FREE if t in monthly_prices.columns]
    elif bench_mom3 < 0:
        exposure_scale = 0.5

    if len(selected) == 0:
        selected = []

    if len(selected) > 0:
        sel_scores = score.loc[selected].clip(lower=0)
        weights = (sel_scores / sel_scores.sum()) if sel_scores.sum() > 0 else pd.Series(1.0/len(selected), index=selected)
    else:
        weights = pd.Series([], index=[])

    # 波动率目标
    hist_rets = asset_ret_monthly[selected].loc[:date].tail(12) if len(selected)>0 else pd.DataFrame()
    if len(hist_rets) >= 2:
        port_rets = (hist_rets * weights.values).sum(axis=1)
        realized_ann_vol = np.std(port_rets) * np.sqrt(12)
    else:
        realized_ann_vol = 0.0

    scale = 1.0
    if realized_ann_vol > 1e-9 and TARGET_VOL > 0:
        scale = TARGET_VOL / realized_ann_vol
        scale = min(scale, MAX_LEVER)
    final_exposure = exposure_scale * scale
    final_exposure = max(0.0, final_exposure)

    prev_nav = nav.loc[prev_date]
    target_total = prev_nav * final_exposure
    target_dollars = {t: float(weights.get(t,0.0) * target_total) for t in TICKERS}

    prev_hold_vals = holdings_history.loc[prev_date] if i-1 >= 0 else pd.Series(0.0, index=TICKERS)
    mark_to_market = prev_hold_vals * (prices_now / prices_prev)
    cash_before = cash_history.loc[prev_date] if i-1 >= 0 else 0.0
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
                exec_price = market_price * (1 + np.sign(dollar_amount) * slippage_pct)
        else:
            exec_price = market_price * (1 + np.sign(dollar_amount) * slippage_pct)
        shares = abs(dollar_amount) / exec_price if exec_price>0 else 0.0
        executed_amount = shares * exec_price * (1.0 if dollar_amount>0 else -1.0)
        fee = compute_fee(abs(executed_amount))
        return executed_amount, fee, exec_price

    for t, amt in trade_amounts.items():
        if amt < 0:
            executed_amt, fee_amt, price = simulate_execution(t, amt)
            executed_changes[t] += executed_amt
            trade_details.append({"date": date, "ticker": t, "side":"sell", "target": amt, "executed": executed_amt, "fee": fee_amt, "price": price})
    for t, amt in trade_amounts.items():
        if amt > 0:
            executed_amt, fee_amt, price = simulate_execution(t, amt)
            executed_changes[t] += executed_amt
            trade_details.append({"date": date, "ticker": t, "side":"buy", "target": amt, "executed": executed_amt, "fee": fee_amt, "price": price})

    for t in TICKERS:
        mark_to_market[t] += executed_changes[t]

    total_fees = sum([d["fee"] for d in trade_details])
    cash_after = portfolio_before - mark_to_market.sum() - total_fees

    holdings_history.loc[date] = mark_to_market
    cash_history.loc[date] = cash_after
    nav.loc[date] = mark_to_market.sum() + cash_after
    trade_log.extend(trade_details)
    portfolio_returns_hist.append((nav.loc[date]/prev_nav)-1 if prev_nav!=0 else 0.0)
    last_weights = (holdings_history.loc[date]/nav.loc[date]).fillna(0.0)

# -------------------------
# 生成报告、Monte Carlo、邮件发送
# -------------------------
nav = nav.fillna(method='ffill')
nav.iloc[0] = START_USD
cum_returns = nav / nav.iloc[0]
years_total = (nav.index[-1]-nav.index[0]).days/365.25
cagr_val = (nav.iloc[-1]/nav.iloc[0])**(1.0/years_total)-1 if years_total>0 else 0.0
ann_vol = np.std(portfolio_returns_hist)*np.sqrt(12) if len(portfolio_returns_hist)>1 else 0.0
sharpe = cagr_val/ann_vol if ann_vol>0 else np.nan
dd_series = cum_returns / cum_returns.cummax() - 1

# Monte Carlo
monthly_rets = np.array(portfolio_returns_hist) if len(portfolio_returns_hist)>0 else np.array([0.0])
mc_cagrs, mc_maxdds = [], []
horizon_months = int(MONTE_CARLO_HORIZON_YEARS*12)
for _ in range(MONTE_CARLO_RUNS):
    sample = np.random.choice(monthly_rets, size=horizon_months, replace=True)
    nav_mc = np.ones(horizon_months+1)
    for j in range(horizon_months):
        nav_mc[j+1] = nav_mc[j]*(1+sample[j])
    mc_cagrs.append(nav_mc[-1]**(1.0/MONTE_CARLO_HORIZON_YEARS)-1)
    roll = np.maximum.accumulate(nav_mc)
    mc_maxdds.append((nav_mc/roll-1).min())

# 图表
fig1, ax1 = plt.subplots(figsize=(10,6))
cum_returns.plot(ax=ax1); ax1.set_title("策略累计净值")
img_nav = fig_to_base64_bytes(fig1)

fig2, ax2 = plt.subplots(figsize=(10,6))
dd_series.plot(ax=ax2, color='red'); ax2.set_title("回撤")
img_dd = fig_to_base64_bytes(fig2)

fig3, ax3 = plt.subplots(1,2,figsize=(12,5))
ax3[0].hist(mc_cagrs, bins=30); ax3[0].set_title("Monte Carlo CAGR")
ax3[1].hist(mc_maxdds, bins=30); ax3[1].set_title("Monte Carlo MaxDD")
img_mc = fig_to_base64_bytes(fig3)

# trade_log
trade_df = pd.DataFrame(trade_log)
if not trade_df.empty:
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    trade_df = trade_df.sort_values('date')

holdings_snapshot = holdings_history.loc[dates[-1]]

# 邮件 HTML
body_html = f"""
<h3>月度动能轮动策略（Tiingo final）回测报告</h3>
<p>开始资金: {START_USD:.2f} USD</p>
<p>CAGR: {cagr_val*100:.2f}% &nbsp;&nbsp; 年化波动: {ann_vol*100:.2f}% &nbsp;&nbsp; Sharpe: {sharpe:.2f}</p>
<p>Monte Carlo {MONTE_CARLO_RUNS} 次模拟:</p>
<ul>
<li>平均 CAGR: {np.mean(mc_cagrs)*100:.2f}%</li>
<li>95% 下分位 MaxDD: {np.percentile(mc_maxdds,5)*100:.2f}%</li>
</ul>
<h4>最终持仓（美元）</h4>
{holdings_snapshot.to_frame().to_html()}
<h4>累计净值图</h4>
<img src="data:image/png;base64,{img_nav}"/>
<h4>回撤图</h4>
<img src="data:image/png;base64,{img_dd}"/>
<h4>Monte Carlo 分布</h4>
<img src="data:image/png;base64,{img_mc}"/>
"""

# 写 CSV
monthly_prices.to_csv("monthly_prices.csv")
trade_df.to_csv("trade_log.csv")

# 发送邮件
if EMAIL_SENDER and EMAIL_PASSWORD and EMAIL_RECIPIENTS:
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = ",".join(EMAIL_RECIPIENTS)
    msg['Subject'] = Header("月度动能轮动策略报告", 'utf-8')
    msg.attach(MIMEText(body_html, 'html', 'utf-8'))

    # 附件 trade_log.csv
    from email.mime.base import MIMEBase
    from email import encoders
    with open("trade_log.csv", "rb") as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition','attachment; filename="trade_log.csv"')
    msg.attach(part)

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=ssl.create_default_context()) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENTS, msg.as_string())
        print("邮件发送成功")
    except Exception as e:
        print(f"邮件发送失败: {e}")

print("回测完成，CSV 和 HTML 图表已生成。")
