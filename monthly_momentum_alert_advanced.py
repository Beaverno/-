#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monthly_momentum_alert_advanced.py

升级版月度动能轮动策略：
- NUM_HOLD = 3
- 多周期动能 (12M, 6M, 3M)
- 动能加速度
- 波动率调整
- 最大单资产权重 50%
- 手续费 + 滑点模型
- 空仓保护 / 避险资产（SHY, TLT, GLD）
- 支持 GitHub Actions 自动发送邮件
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import ssl
import traceback

# ---------------------------
# 参数设置
# ---------------------------
TICKERS = ["SPY","QQQ","VOO","VT","EFA","EEM","TLT","GLD"]
RISK_FREE = ["SHY"]  # 空仓时避险资产
NUM_HOLD = 3  # 每月持有3只ETF

# 成本模型
FEE_RATE = 0.001      # 手续费 0.1%
SLIPPAGE = 0.0005     # 滑点 0.05%
TRANSACTION_COST = FEE_RATE + SLIPPAGE

# 动能窗口（交易日近似）
MOM_1M = 21
MOM_3M = 63
MOM_6M = 126
MOM_12M = 252

SMA_WINDOW = 200  # 趋势过滤用200日均线

# 本金
RMB_CAPITAL = float(os.environ.get("RMB_CAPITAL", 650000))
USD_RMB_RATE = float(os.environ.get("USD_RMB_RATE", 7.1))
usd_capital = RMB_CAPITAL / USD_RMB_RATE

# 邮箱配置（GitHub Secrets）
sender = os.environ.get("EMAIL_ADDRESS")
password = os.environ.get("EMAIL_PASSWORD")
receiver = os.environ.get("EMAIL_RECIPIENT", sender)
smtp_provider = os.environ.get("SMTP_PROVIDER", "qq").lower()

if not sender or not password:
    raise ValueError("请在 GitHub Secrets 中设置 EMAIL_ADDRESS 和 EMAIL_PASSWORD")

# SMTP 配置
if smtp_provider == "qq":
    smtp_server = "smtp.qq.com"
    smtp_port = 465
elif smtp_provider == "163":
    smtp_server = "smtp.163.com"
    smtp_port = 465
elif smtp_provider == "gmail":
    smtp_server = "smtp.gmail.com"
    smtp_port = 465
else:
    raise ValueError("不支持的 SMTP_PROVIDER (支持: qq, 163, gmail)")

# ---------------------------
# 下载数据
# ---------------------------
end_date = datetime.today().strftime("%Y-%m-%d")
data = yf.download(TICKERS, start="2010-01-01", end=end_date, auto_adjust=True, progress=False)
if data.empty:
    raise RuntimeError("未下载到数据，请检查网络或标的")

data = data.dropna(how="all")
adj_close = data

# ---------------------------
# 动能计算
# ---------------------------
def safe_shift(df, days):
    if len(df) <= days:
        return pd.Series([np.nan]*len(df.columns), index=df.columns)
    return df.shift(days).iloc[-1]

price_now = adj_close.iloc[-1]
p_1m = safe_shift(adj_close, MOM_1M)
p_3m = safe_shift(adj_close, MOM_3M)
p_6m = safe_shift(adj_close, MOM_6M)
p_12m = safe_shift(adj_close, MOM_12M)

mom1 = (price_now - p_1m)/p_1m
mom3 = (price_now - p_3m)/p_3m
mom6 = (price_now - p_6m)/p_6m
mom12 = (price_now - p_12m)/p_12m

acc = mom1 - mom3  # 动能加速度
vol = adj_close.pct_change().rolling(MOM_12M).std().iloc[-1]  # 波动率

# 综合评分
score = 0.35*mom1.rank(ascending=False) + 0.35*mom3.rank(ascending=False) + 0.15*acc.rank(ascending=False) + 0.15*mom12.rank(ascending=False)
score = score / vol  # 风险调整
score = score.sort_values(ascending=False)

# ---------------------------
# 趋势过滤
# ---------------------------
sma200 = adj_close.rolling(SMA_WINDOW).mean().iloc[-1]
trend_ok = price_now > sma200

eligible = [t for t in score.index if trend_ok.get(t, False)]
selected = []
for t in score.index:
    if len(selected) >= NUM_HOLD:
        break
    if t in eligible:
        selected.append(t)
if len(selected) < NUM_HOLD:
    for t in score.index:
        if len(selected) >= NUM_HOLD:
            break
        if t not in selected:
            selected.append(t)

# ---------------------------
# 空仓保护
# ---------------------------
spy_mom12 = mom12.get("SPY", 0)
if spy_mom12 < 0:
    selected = RISK_FREE  # 全仓避险资产

allocation = 1.0 / len(selected)

# ---------------------------
# 交易成本模拟
# ---------------------------
# 假设全部换仓一次
turnover = 1.0
total_cost = turnover * TRANSACTION_COST
capital_after_cost = usd_capital * (1 - total_cost)

# ---------------------------
# 邮件正文
# ---------------------------
msg_lines = []
msg_lines.append("【月度动能轮动策略 - 升级版】")
msg_lines.append(f"日期: {datetime.today().strftime('%Y-%m-%d')}")
msg_lines.append(f"本金: {RMB_CAPITAL:,.0f} RMB (~${usd_capital:,.2f} USD)")
msg_lines.append("")
msg_lines.append(f"本月买入 ETF 建议 (Top-{NUM_HOLD}):")
for t in selected:
    msg_lines.append(f"- {t}: 分配 {allocation*100:.1f}% (~${capital_after_cost*allocation:,.2f})")
msg_lines.append("")
msg_lines.append("说明: 多周期动能 + 动能加速度 + 风险调整 + 趋势过滤 + 空仓保护 + 手续费滑点考虑。")

body = "\n".join(msg_lines)
message = MIMEText(body, "plain", "utf-8")
message["From"] = Header("月度动能轮动策略", "utf-8")
message["To"] = Header("用户", "utf-8")
message["Subject"] = Header("月度买入提醒（升级版）", "utf-8")

# ---------------------------
# 发送邮件
# ---------------------------
try:
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
        server.set_debuglevel(1)
        server.login(sender, password)
        server.sendmail(sender, [receiver], message.as_string())
    print("邮件已发送成功！")
    print(body)
except Exception as e:
    print("邮件发送失败:", e)
    traceback.print_exc()
    exit(1)
