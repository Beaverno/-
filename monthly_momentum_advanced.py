#!/usr/bin/env python3
# monthly_momentum_advanced.py
# 说明：高级月度动能轮动策略回测 + 下单/滑点/手续费模拟 + 波动率目标 + 蒙特卡洛压力测试
# 依赖: yfinance, pandas, numpy, matplotlib

import os
import yfinance as yf
import pandas as pd
import numpy as np
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
import math
import random

# matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 用户可调参数（可通过环境变量覆盖）
# -----------------------------
TICKERS = os.environ.get("TICKERS", "SPY,QQQ,VOO,VT,EFA,EEM,TLT,GLD").split(",")
RISK_FREE = os.environ.get("RISK_FREE", "SHY").split(",")
NUM_HOLD = int(os.environ.get("NUM_HOLD", 3))           # 持仓数量
FEE_RATE = float(os.environ.get("FEE_RATE", 0.001))    # 交易手续费比例（整体近似）
MIN_FEE = float(os.environ.get("MIN_FEE", 0.0))        # 最低手续费（绝对值 USD）
SLIPPAGE_BASE = float(os.environ.get("SLIPPAGE_BASE", 0.0005))  # 基础滑点比例
MIN_TRADE_USD = float(os.environ.get("MIN_TRADE_USD", 50))      # 最小下单金额（美元）
ALLOW_LIMIT_ORDERS = os.environ.get("ALLOW_LIMIT_ORDERS", "True").lower() in ("1","true","yes")
LIMIT_FILL_PROB = float(os.environ.get("LIMIT_FILL_PROB", 0.7))  # 限价单初始成交概率
LIMIT_ADVANTAGE = float(os.environ.get("LIMIT_ADVANTAGE", 0.0002)) # 限价优于市价的比例
TARGET_VOL = float(os.environ.get("TARGET_VOL", 0.10))    # 年化目标波动率 (e.g., 0.10 = 10%)
MAX_LEVER = float(os.environ.get("MAX_LEVER", 2.0))       # 最大放大倍数（防止过度放大）
LOOKBACK_DAYS_12M = int(os.environ.get("LOOKBACK_DAYS_12M", 252))
SMA_WINDOW = int(os.environ.get("SMA_WINDOW", 200))
START_DATE = os.environ.get("START_DATE", "2005-01-01")
END_DATE = os.environ.get("END_DATE", None)  # None -> today
MONTE_CARLO_RUNS = int(os.environ.get("MONTE_CARLO_RUNS", 500))
MONTE_CARLO_HORIZON_YEARS = float(os.environ.get("MONTE_CARLO_HORIZON_YEARS", 5.0))

# 资金与汇率
RMB_CAPITAL = float(os.environ.get("RMB_CAPITAL", 650000))   # 人民币
USD_RMB_RATE = float(os.environ.get("USD_RMB_RATE", 7.1))    # 汇率
START_USD = RMB_CAPITAL / USD_RMB_RATE

# 邮件配置（可选）
EMAIL_SENDER = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
EMAIL_RECIPIENTS = os.environ.get("EMAIL_RECIPIENT", EMAIL_SENDER).split(",") if os.environ.get("EMAIL_RECIPIENT") else [EMAIL_SENDER] if EMAIL_SENDER else []
SMTP_PROVIDER = os.environ.get("SMTP_PROVIDER", "qq").lower()

if SMTP_PROVIDER == "qq":
    SMTP_SERVER = "smtp.qq.com"
    SMTP_PORT = 465
elif SMTP_PROVIDER == "163":
    SMTP_SERVER = "smtp.163.com"
    SMTP_PORT = 465
elif SMTP_PROVIDER == "gmail":
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 465
else:
    SMTP_SERVER = "smtp.qq.com"
    SMTP_PORT = 465

# Safety checks
if EMAIL_SENDER and not EMAIL_PASSWORD:
    print("WARNING: EMAIL_SENDER set but EMAIL_PASSWORD not provided. Email will fail if attempted.")

# -----------------------------
# 工具函数
# -----------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    s = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return s

def annualize_vol(returns_series_monthly):
    # returns in decimal monthly
    return returns_series_monthly.std() * np.sqrt(12)

def cagr_from_nav(nav):
    # nav is series indexed by date
    n_periods = len(nav)
    if n_periods <= 1 or nav.iloc[0] <= 0:
        return 0.0
    total_years = (nav.index[-1] - nav.index[0]).days / 365.25
    return nav.iloc[-1] ** (1.0 / total_years) - 1 if nav.iloc[0] == 1.0 else (nav.iloc[-1]/nav.iloc[0])**(1.0/total_years)-1

def max_drawdown(nav):
    roll = nav.cummax()
    dd = nav / roll - 1
    return dd.min(), dd

# 简化滑点模型：滑点比例 = SLIPPAGE_BASE + k * (annual_vol) ; k 可设为 0.5
def estimate_slippage(annual_vol):
    k = 0.5
    return SLIPPAGE_BASE + k * annual_vol

# 计算手续费（按金额）
def compute_fee(amount_usd):
    fee = max(amount_usd * FEE_RATE, MIN_FEE)
    return fee

# 限价单成交模拟：以 fill_prob 成交优价，否则成交市价并成本上升
def simulate_limit_fill(limit_price, market_price, fill_prob):
    if random.random() < fill_prob:
        return limit_price, True
    else:
        # assume eventually executed at next market price with adverse slippage
        adverse = 1 + estimate_slippage(0.2)  # assume moderate vol next period
        return market_price * adverse, False

# -----------------------------
# 下载价格数据
# -----------------------------
print("Downloading price data for:", TICKERS)
data = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
if data.empty:
    raise RuntimeError("No data downloaded -- check tickers/network.")
# data may be single-column price series per ticker (auto_adjust=True)
price = data.ffill().bfill()
# monthly price at month end
monthly_price = price.resample("M").last()

# -----------------------------
# 回测框架状态变量
# -----------------------------
dates = monthly_price.index
n_months = len(dates)
nav = pd.Series(index=dates, dtype=float)
nav.iloc[0] = START_USD
# store holdings value dollars
holdings_history = pd.DataFrame(0.0, index=dates, columns=TICKERS)
cash_history = pd.Series(0.0, index=dates)
trade_log = []  # list of dicts

# for realized returns calculation use monthly returns of assets
asset_ret_monthly = monthly_price.pct_change().fillna(0)

# rolling realized vol (annualized) for volatility targeting window: use 12-months
vol_rolling = asset_ret_monthly.rolling(12).std() * np.sqrt(12)

# historical portfolio returns (for realized vol)
portfolio_returns_hist = []

last_weights = pd.Series(0.0, index=TICKERS)

# -----------------------------
# 回测主循环（按月调仓）
# -----------------------------
for i in range(1, n_months):
    date = dates[i]
    prev_date = dates[i-1]
    prices_now = monthly_price.loc[date]
    prices_prev = monthly_price.loc[prev_date]

    # 1) 计算多周期动能（1/3/6/12）
    mom1 = (prices_now - monthly_price.shift(1).loc[date]) / monthly_price.shift(1).loc[date]
    mom3 = (prices_now - monthly_price.shift(3).loc[date]) / monthly_price.shift(3).loc[date]
    mom6 = (prices_now - monthly_price.shift(6).loc[date]) / monthly_price.shift(6).loc[date]
    mom12 = (prices_now - monthly_price.shift(12).loc[date]) / monthly_price.shift(12).loc[date]
    acc = mom1 - mom3  # 动能加速度

    # avoid NaN by filling with -inf (push to bottom)
    mom1 = mom1.fillna(-np.inf); mom3 = mom3.fillna(-np.inf); mom6 = mom6.fillna(-np.inf); mom12 = mom12.fillna(-np.inf); acc = acc.fillna(-np.inf)

    # 2) volatility estimate for each asset (annualized)
    asset_vol = vol_rolling.loc[date].fillna(vol_rolling.mean().mean())  # fallback to overall avg if NaN

    # 3) compute composite score (higher better)
    # use ranks for robustness, then scale by inverse volatility to penalize high vol
    rank_m1 = mom1.rank(ascending=False, method="average")
    rank_m3 = mom3.rank(ascending=False, method="average")
    rank_m6 = mom6.rank(ascending=False, method="average")
    rank_m12 = mom12.rank(ascending=False, method="average")
    rank_acc = acc.rank(ascending=False, method="average")

    # weighted rank sum
    raw_score = 0.35*rank_m1 + 0.25*rank_m3 + 0.15*rank_m6 + 0.15*rank_m12 + 0.10*rank_acc
    # adjust by volatility (lower vol -> higher adjusted score)
    score = raw_score / (asset_vol + 1e-9)
    score = score.sort_values(ascending=False)

    # 4) trend filter (SMA)
    sma = monthly_price.rolling(SMA_WINDOW).mean().loc[date]
    trend_ok = prices_now > sma

    eligible = [t for t in score.index if trend_ok.get(t, False)]
    selected = [t for t in score.index if t in eligible][:NUM_HOLD]

    # dynamic cash/empty threshold: if market (benchmark) short and medium term both negative => reduce exposure
    benchmark = "SPY" if "SPY" in monthly_price.columns else monthly_price.columns[0]
    bench_mom3 = (prices_now[benchmark] - monthly_price.shift(3).loc[date][benchmark]) / monthly_price.shift(3).loc[date][benchmark]
    bench_mom12 = (prices_now[benchmark] - monthly_price.shift(12).loc[date][benchmark]) / monthly_price.shift(12).loc[date][benchmark]

    exposure_scale = 1.0
    # if both short and medium term negative -> heavy defense
    if bench_mom3 < 0 and bench_mom12 < 0:
        exposure_scale = 0.0  # move to risk free fully
        selected = RISK_FREE
    elif bench_mom3 < 0:
        exposure_scale = 0.5  # half exposure
    # else exposure_scale remains 1.0

    # if no selected, fallback to risk-free
    if len(selected) == 0:
        selected = RISK_FREE

    # 5) dynamic weight allocation among selected: use scores to weight
    sel_scores = score.loc[selected]
    if sel_scores.sum() <= 0:
        weights = pd.Series(1.0/len(selected), index=selected)
    else:
        raw = sel_scores.clip(lower=0)
        if raw.sum() == 0:
            weights = pd.Series(1.0/len(selected), index=selected)
        else:
            weights = raw / raw.sum()

    # 6) volatility targeting: compute recent realized vol of a hypothetical equal-weight portfolio of selected
    # compute monthly returns for selected based on last 12 months
    hist_rets = asset_ret_monthly[selected].loc[:date].tail(12)  # months up to current
    if len(hist_rets) >= 2:
        # approximate portfolio vol if fully invested
        port_rets = (hist_rets * weights.values).sum(axis=1)
        realized_ann_vol = np.std(port_rets) * np.sqrt(12)
    else:
        realized_ann_vol = 0.0

    # compute scaling factor to reach TARGET_VOL
    scale = 1.0
    if realized_ann_vol > 1e-9 and TARGET_VOL > 0:
        scale = TARGET_VOL / realized_ann_vol
        # constrain scale
        if scale > MAX_LEVER:
            scale = MAX_LEVER
    # final exposure after considering dynamic exposure_scale (market condition)
    final_exposure = exposure_scale * scale
    if final_exposure < 0:
        final_exposure = 0.0

    # 7) determine dollar targets before costs
    prev_nav = nav.loc[prev_date]
    target_dollar_total = prev_nav * final_exposure
    # allocate to each asset
    target_dollars = {t: float(weights.get(t, 0.0) * target_dollar_total) for t in TICKERS}

    # 8) simulate trades from last_weights to target_dollars
    # compute current holdings value using previous prices (we track value only, not share counts)
    # For simplicity we maintain holdings value directly and compute PnL from price changes
    # holdings_history stores dollar value of each asset at month-ends AFTER price move and rebalancing.

    # compute pre-rebalance holdings value (mark-to-market using current prices)
    prev_hold_vals = holdings_history.loc[prev_date] if i-1 >= 0 else pd.Series(0.0, index=TICKERS)
    # after price move to today, value changes:
    mark_to_market_vals = prev_hold_vals * (prices_now / prices_prev)
    cash_before = cash_history.loc[prev_date] if i-1 >= 0 else 0.0
    portfolio_value_before = mark_to_market_vals.sum() + cash_before

    # amount to trade = target - mark_to_market_vals
    trade_amounts = {t: target_dollars.get(t, 0.0) - mark_to_market_vals.get(t, 0.0) for t in TICKERS}
    # sum of buys and sells
    total_buy = sum([amt for amt in trade_amounts.values() if amt > 0])
    total_sell = -sum([amt for amt in trade_amounts.values() if amt < 0])

    # enforce min trade size: if allocation to some asset below MIN_TRADE_USD, skip (keep as cash)
    for t in TICKERS:
        if abs(trade_amounts[t]) < MIN_TRADE_USD:
            # small trade ignored; adjust cash/target accordingly by leaving existing holding
            trade_amounts[t] = 0.0
            target_dollars[t] = mark_to_market_vals.get(t,0.0)

    # recompute totals after min trade pruning
    total_buy = sum([amt for amt in trade_amounts.values() if amt > 0])
    total_sell = -sum([amt for amt in trade_amounts.values() if amt < 0])

    # compute transaction costs and simulate execution prices
    executed_changes = {t: 0.0 for t in TICKERS}  # actual dollar change executed (positive=buy)
    trade_details = []

    # Helper to compute execution price & effective amount after fee+slippage
    def execute_trade(ticker, dollar_amount):
        # dollar_amount positive -> buy; negative -> sell
        market_px = prices_now[ticker]
        # estimate annual vol for asset to compute slippage
        annual_vol = asset_vol.get(ticker, 0.2)
        slippage_pct = estimate_slippage(annual_vol)
        executed_amount = 0.0
        fee_total = 0.0
        if dollar_amount == 0:
            return 0.0, 0.0, market_px, False
        # simulate limit order attempts (for buys, limit slightly better price; for sells, limit slightly better)
        if ALLOW_LIMIT_ORDERS:
            # decide limit price advantange: buys try slightly lower price, sells try slightly higher
            sign = 1 if dollar_amount > 0 else -1
            limit_price = market_px * (1 - sign * LIMIT_ADVANTAGE)
            filled_price, filled = simulate_limit_fill(limit_price, market_px, LIMIT_FILL_PROB)
            if not filled:
                # execute at market with adverse slippage applied
                executed_price = market_px * (1 + np.sign(dollar_amount) * slippage_pct)
            else:
                executed_price = filled_price
        else:
            executed_price = market_px * (1 + np.sign(dollar_amount) * slippage_pct)

        # shares executed = abs(dollar_amount) / executed_price
        shares = abs(dollar_amount) / executed_price
        executed_amount = shares * executed_price * (1.0 if dollar_amount>0 else -1.0)
        # fees proportional to traded dollar amount (absolute)
        fee = compute_fee(abs(executed_amount))
        fee_total = fee
        # net cash impact (buy increases negative cash)
        net_cash_impact = executed_amount + (fee_total if dollar_amount>0 else fee_total)  # fee always reduces cash
        # For simplicity treat fee as added on top of trade amount (i.e., buyer pays amount + fee; seller receives amount - fee)
        return executed_amount, fee_total, executed_price, True

    # execute sells first -> free up cash
    for t, amt in trade_amounts.items():
        if amt < 0:
            executed_amt, fee_amt, exec_price, filled = execute_trade(t, amt)
            executed_changes[t] += executed_amt  # negative
            trade_details.append({"date": date, "ticker": t, "side":"sell", "target": amt, "executed": executed_amt, "fee": fee_amt, "price": exec_price})
    # then execute buys
    for t, amt in trade_amounts.items():
        if amt > 0:
            executed_amt, fee_amt, exec_price, filled = execute_trade(t, amt)
            executed_changes[t] += executed_amt  # positive
            trade_details.append({"date": date, "ticker": t, "side":"buy", "target": amt, "executed": executed_amt, "fee": fee_amt, "price": exec_price})

    # Apply executed changes to mark_to_market_vals to get new holdings values
    # sold executed_amt is negative dollar value -> reduce holding; buys increase
    for t in TICKERS:
        mark_to_market_vals[t] += executed_changes[t]

    # cash after trades:
    cash_after = portfolio_value_before - mark_to_market_vals.sum()
    # subtract total fees
    total_fees = sum([d["fee"] for d in trade_details])
    cash_after -= total_fees

    # set histories
    holdings_history.loc[date] = mark_to_market_vals
    cash_history.loc[date] = cash_after
    nav.loc[date] = mark_to_market_vals.sum() + cash_after

    # record trades
    for d in trade_details:
        trade_log.append(d)

    # compute realized monthly return for reporting (net of costs)
    prev_nav_val = prev_nav
    cur_nav_val = nav.loc[date]
    realized_ret = (cur_nav_val / prev_nav_val) - 1
    portfolio_returns_hist.append(realized_ret)

    # update last_weights for next iteration (approx from holdings values)
    last_weights = (holdings_history.loc[date] / nav.loc[date]).fillna(0.0)

# -----------------------------
# 回测结果与统计
# -----------------------------
# fill initial nav if zero
nav = nav.fillna(method="ffill")
nav.iloc[0] = START_USD

cum_returns = nav / nav.iloc[0]
cagr = cagr_from_nav(cum_returns)
ann_vol = np.std(portfolio_returns_hist) * np.sqrt(12) if len(portfolio_returns_hist)>1 else 0.0
sharpe = cagr / ann_vol if ann_vol>0 else np.nan
md, dd_series = max_drawdown(cum_returns)

# print summary
print("Backtest summary:")
print(f"Period: {nav.index[0].date()} to {nav.index[-1].date()}")
print(f"Start USD: {START_USD:.2f}, End USD: {nav.iloc[-1]:.2f}")
print(f"CAGR: {cagr:.2%}, Ann Vol: {ann_vol:.2%}, Sharpe: {sharpe:.3f}, MaxDD: {md:.2%}")

# -----------------------------
# 子区间回测（滚动窗口统计）
# -----------------------------
def subperiod_stats(nav_series, window_years=5):
    stats = []
    months_window = int(window_years * 12)
    for start in range(0, len(nav_series)-months_window+1, 12):  # step 12 months
        sub = nav_series.iloc[start:start+months_window]
        c = cagr_from_nav(sub)
        md_sub, _ = max_drawdown(sub)
        stats.append({"start": sub.index[0].date(), "end": sub.index[-1].date(), "CAGR": c, "MaxDD": md_sub})
    return pd.DataFrame(stats)

sub_stats = subperiod_stats(cum_returns, window_years=5)

# -----------------------------
# 蒙特卡洛压力测试（bootstrap monthly returns）
# -----------------------------
monthly_rets = pd.Series(portfolio_returns_hist) if len(portfolio_returns_hist)>0 else pd.Series([0.0])
mc_cagrs = []
mc_maxdds = []

horizon_months = int(MONTE_CARLO_HORIZON_YEARS * 12)
for run in range(MONTE_CARLO_RUNS):
    # bootstrap sample of monthly returns
    sample = np.random.choice(monthly_rets.values, size=horizon_months, replace=True)
    nav_mc = np.ones(horizon_months+1)
    for j in range(horizon_months):
        nav_mc[j+1] = nav_mc[j] * (1 + sample[j])
    # compute CAGR and MaxDD for this run
    years = MONTE_CARLO_HORIZON_YEARS
    cagr_mc = nav_mc[-1] ** (1.0/years) - 1
    roll_max = np.maximum.accumulate(nav_mc)
    dd = nav_mc / roll_max - 1
    maxdd_mc = dd.min()
    mc_cagrs.append(cagr_mc)
    mc_maxdds.append(maxdd_mc)

# -----------------------------
# 绘图与邮件正文构建
# -----------------------------
# NAV plot
fig_nav, ax = plt.subplots(figsize=(10,6))
cum_returns.plot(ax=ax)
ax.set_title("策略累计净值 (normalized)")
ax.set_ylabel("Normalized NAV")
img_nav = fig_to_base64(fig_nav)

# drawdown plot
fig_dd, ax = plt.subplots(figsize=(10,6))
dd_series.plot(ax=ax, color='red')
ax.set_title("策略回撤")
img_dd = fig_to_base64(fig_dd)

# Monte Carlo histograms
fig_mc, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].hist(mc_cagrs, bins=30)
ax[0].set_title("Monte Carlo CAGR Distribution")
ax[1].hist(mc_maxdds, bins=30)
ax[1].set_title("Monte Carlo MaxDD Distribution")
img_mc = fig_to_base64(fig_mc)

# trade log DataFrame
trade_df = pd.DataFrame(trade_log)
if not trade_df.empty:
    trade_df['date'] = pd.to_datetime(trade_df['date'])
    trade_df = trade_df.sort_values('date')

# holdings snapshot latest
holdings_snapshot = holdings_history.loc[dates[-1]]

# Build HTML email
body_html = f"""
<h3>高级月度动能轮动策略回测报告</h3>
<p>回测区间: {nav.index[0].date()} - {nav.index[-1].date()}<br>
本金: {RMB_CAPITAL:,.0f} RMB (~${START_USD:,.2f} USD)<br>
CAGR: {cagr:.2%} &nbsp; AnnVol: {ann_vol:.2%} &nbsp; Sharpe: {sharpe:.2f} &nbsp; MaxDD: {md:.2%}</p>
<h4>本月建议 (Top-{NUM_HOLD}, exposure scale={final_exposure:.2f})</h4>
<ul>
"""
# note: 'selected' and 'weights' currently reflect last loop values
for t in selected:
    w = weights.get(t, 0.0)
    alloc_usd = (START_USD * (1 - TRANSACTION_COST)) * w * final_exposure
    body_html += f"<li>{t}: weight {w:.3f}, est allocation ${alloc_usd:,.2f}</li>"
body_html += "</ul>"

body_html += f"<h4>最新持仓（美元）</h4><pre>{holdings_snapshot.to_string()}</pre>"
body_html += "<h4>回测图</h4>"
body_html += f"<img src='data:image/png;base64,{img_nav}'><br>"
body_html += f"<img src='data:image/png;base64,{img_dd}'><br>"
body_html += f"<h4>蒙特卡洛样本（{MONTE_CARLO_RUNS} 次）</h4><img src='data:image/png;base64,{img_mc}'><br>"

# attach trade log as CSV string
if not trade_df.empty:
    csv_buf = io.StringIO()
    trade_df.to_csv(csv_buf, index=False)
    trade_csv = csv_buf.getvalue()
else:
    trade_csv = ""

# -----------------------------
# 发送邮件（可选）
# -----------------------------
def send_email(subject, html_body, attachments=None):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        print("Email sender/password not configured - skipping email send.")
        return
    msg = MIMEMultipart()
    msg["From"] = Header("Momentum Bot", "utf-8")
    msg["To"] = Header(", ".join(EMAIL_RECIPIENTS), "utf-8")
    msg["Subject"] = Header(subject, "utf-8")
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    # attachments: dict of filename->content
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

# auto-send if email configured
send_email("Momentum Strategy Monthly Report (Advanced)", body_html, attachments={"trade_log.csv": trade_csv})

# -----------------------------
# 保存输出文件
# -----------------------------
out_dir = os.environ.get("OUTPUT_DIR", ".")
os.makedirs(out_dir, exist_ok=True)
nav.to_csv(os.path.join(out_dir,"nav.csv"))
if not trade_df.empty:
    trade_df.to_csv(os.path.join(out_dir,"trade_log.csv"), index=False)
holdings_history.to_csv(os.path.join(out_dir,"holdings_history.csv"))
pd.DataFrame({"mc_cagr": mc_cagrs, "mc_maxdd": mc_maxdds}).to_csv(os.path.join(out_dir,"monte_carlo.csv"), index=False)

print("Finished. Files saved to", out_dir)