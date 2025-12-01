#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==========================================================
# CONFIG
# ==========================================================
TIINGO_TOKEN = "<YOUR_TIINGO_TOKEN>"     # 必填
SEND_EMAIL = False                        # GitHub Actions 使用建议 True
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_SENDER = "xxx@gmail.com"
EMAIL_PASSWORD = "<APP_PASSWORD>"
EMAIL_RECEIVER = "xxx@gmail.com"

# 交易成本
SLIPPAGE = 0.0005     # 0.05%
COMMISSION = 0.0003   # 0.03%

# 动量参数
LOOKBACK_M = 12
TOP_K = 2

# ==========================================================
# 函数：下载 Tiingo 数据（含重试 + 时区修正）
# ==========================================================
def download_tiingo(symbol, max_retries=4):
    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices"
    params = {
        "token": TIINGO_TOKEN,
        "startDate": "1990-01-01",
        "resampleFreq": "daily"
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()

            if len(data) == 0:
                raise ValueError("Empty Tiingo response")

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None)  # <=== FIX 关键：移除 tzinfo
            df = df.set_index("date")
            df = df[["adjClose"]].rename(columns={"adjClose": "close"})

            return df

        except Exception as e:
            print(f"Tiingo {symbol} download failed (attempt {attempt}/{max_retries}): {e}")

    print(f"Failed to download {symbol} after {max_retries} attempts")
    return None

# ==========================================================
# 函数：月度重采样
# ==========================================================
def resample_monthly(df):
    return df["close"].resample("M").last().dropna()

# ==========================================================
# 函数：计算动量
# ==========================================================
def compute_momentum(monthly_price, lookback=12):
    return monthly_price.pct_change(lookback)

# ==========================================================
# 函数：组合回测（含滑点 + 佣金）
# ==========================================================
def backtest_momentum(momentum_df, prices, top_k):
    holdings = []
    portfolio_value = 1.0
    equity_curve = []

    for date in momentum_df.index:
        scores = momentum_df.loc[date].dropna()
        selected = scores.sort_values(ascending=False).head(top_k).index.tolist()

        # 每次换仓：加入交易成本
        portfolio_value *= (1 - SLIPPAGE - COMMISSION)

        # 投入等权
        weights = {s: 1/len(selected) for s in selected}

        # 下个月收益
        next_month = date + relativedelta(months=1)
        next_prices = prices.get(next_month, None)

        if next_prices is not None:
            ret = 0
            for s in selected:
                if s in next_prices and s in prices[date]:
                    ret += weights[s] * (next_prices[s] / prices[date][s] - 1)
            portfolio_value *= (1 + ret)

        equity_curve.append({"date": date, "value": portfolio_value, "hold": selected})

    return pd.DataFrame(equity_curve).set_index("date")

# ==========================================================
# 函数：Monte Carlo 压力测试
# ==========================================================
def monte_carlo_simulation(returns, runs=2000, horizon=12):
    sims = []
    ar = returns.dropna().values
    for _ in range(runs):
        sampled = np.random.choice(ar, horizon, replace=True)
        sims.append(np.prod(1 + sampled) - 1)
    return np.percentile(sims, [1, 5, 50, 95, 99])

# ==========================================================
# 邮件发送函数
# ==========================================================
def send_email(subject, body):
    if not SEND_EMAIL:
        return
    msg = MIMEMultipart()
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    context = ssl.create_default_context()
    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls(context=context)
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())

# ==========================================================
# 主程序
# ==========================================================
if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "VOO", "VT", "EFA", "EEM", "TLT", "GLD"]
    all_data = {}

    print("=== Downloading Tiingo Data ===")
    for t in tickers:
        df = download_tiingo(t)
        if df is not None:
            all_data[t] = df

    if len(all_data) == 0:
        raise RuntimeError("No price data downloaded. Check TIINGO_TOKEN and network.")

    # 按月
    monthly = pd.DataFrame()
    for t, df in all_data.items():
        monthly[t] = resample_monthly(df)

    # 动量
    momentum = compute_momentum(monthly, LOOKBACK_M)

    # 回测
    monthly_returns = monthly.pct_change()
    prices_dict = {d: monthly.loc[d].to_dict() for d in monthly.index}

    bt = backtest_momentum(momentum, prices_dict, TOP_K)

    # Monte Carlo
    mc = monte_carlo_simulation(monthly_returns.stack())

    # 报告
    latest_hold = bt["hold"].iloc[-1]
    latest_value = bt["value"].iloc[-1]

    report = f"""
    月度动量策略报告（Tiingo）
    ==========================
    当前持仓（Top {TOP_K} 动量）:
    {latest_hold}

    当前组合价值：{latest_value:.2f}

    Monte Carlo（1年分布）:
    1%:  {mc[0]*100:.2f}%
    5%:  {mc[1]*100:.2f}%
    50%: {mc[2]*100:.2f}%
    95%: {mc[3]*100:.2f}%
    99%: {mc[4]*100:.2f}%
    """

    print(report)
    send_email("月度动量策略报告", report)
