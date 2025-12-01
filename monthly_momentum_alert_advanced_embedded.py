#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monthly_momentum_alert_advanced_embedded.py

升级版月度动能轮动策略 + 邮件回测图嵌入正文
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header
import ssl
import io
import base64
import traceback

# ---------------------------
# 参数设置（与之前一致）
# ---------------------------
TICKERS = ["SPY","QQQ","VOO","VT","EFA","EEM","TLT","GLD"]
RISK_FREE = ["SHY"]
NUM_HOLD = 3
FEE_RATE = 0.001
SLIPPAGE = 0.0005
TRANSACTION_COST = FEE_RATE + SLIPPAGE

MOM_1M = 21
MOM_3M = 63
MOM_6M = 126
MOM_12M = 252
SMA_WINDOW = 200

RMB_CAPITAL = float(os.environ.get("RMB_CAPITAL", 650000))
USD_RMB_RATE = float(os.environ.get("USD_RMB_RATE", 7.1))
usd_capital = RMB_CAPITAL / USD_RMB_RATE

sender = os.environ.get("EMAIL_ADDRESS")
password = os.environ.get("EMAIL_PASSWORD")
receiver = os.environ.get("EMAIL_RECIPIENT", sender)
smtp_provider = os.environ.get("SMTP_PROVIDER", "qq").lower()

if not sender or not password:
    raise ValueError("请在 GitHub Secrets 中设置 EMAIL_ADDRESS 和 EMAIL_PASSWORD")

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
# 下载数据，只取 Adj Close
# ---------------------------
end_date = datetime.today().strftime("%Y-%m-%d")
data = yf.download(TICKERS, start="2010-01-01", end=end_date, auto_adjust=True, progress=False)
adj_close = data['Adj Close'].dropna(how='all')

# ---------------------------
# 简单回测
# ---------------------------
monthly_prices = adj_close.resample('M').last()
portfolio_value = pd.Series(index=monthly_prices.index, dtype=float)
portfolio_value.iloc[0] = usd_capital
last_hold = None

for i in range(1, len(monthly_prices)):
    date = monthly_prices.index[i]
    mom1 = (monthly_prices.iloc[i] - monthly_prices.shift(MOM_1M).iloc[i]) / monthly_prices.shift(MOM_1M).iloc[i]
    mom3 = (monthly_prices.iloc[i] - monthly_prices.shift(MOM_3M).iloc[i]) / monthly_prices.shift(MOM_3M).iloc[i]
    mom6 = (monthly_prices.iloc[i] - monthly_prices.shift(MOM_6M).iloc[i]) / monthly_prices.shift(MOM_6M).iloc[i]
    mom12 = (monthly_prices.iloc[i] - monthly_prices.shift(MOM_12M).iloc[i]) / monthly_prices.shift(MOM_12M).iloc[i]
    acc = mom1 - mom3
    vol = monthly_prices.pct_change().rolling(MOM_12M).std().iloc[i]
    
    score = 0.35*mom1.rank(ascending=False) + 0.35*mom3.rank(ascending=False) + 0.15*acc.rank(ascending=False) + 0.15*mom12.rank(ascending=False)
    score = score / vol
    score = score.sort_values(ascending=False)
    
    sma200 = monthly_prices.rolling(SMA_WINDOW).mean().iloc[i]
    trend_ok = monthly_prices.iloc[i] > sma200
    eligible = [t for t in score.index if trend_ok.get(t, False)]
    selected = [t for t in score.index if t in eligible][:NUM_HOLD]
    
    if last_hold is None:
        last_hold = selected
    
    turnover = len(set(selected) ^ set(last_hold)) / NUM_HOLD
    cost = turnover * TRANSACTION_COST
    monthly_ret = monthly_prices.loc[date, selected].pct_change().mean()
    portfolio_value.iloc[i] = portfolio_value.iloc[i-1]*(1 + monthly_ret - cost)
    last_hold = selected

# ---------------------------
# 生成图像为 base64
# ---------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

fig1, ax1 = plt.subplots(figsize=(10,6))
portfolio_value.plot(ax=ax1)
ax1.set_title("策略累计收益曲线")
ax1.set_ylabel("组合价值 (USD)")
img1_base64 = fig_to_base64(fig1)

cummax = portfolio_value.cummax()
drawdown = (portfolio_value - cummax)/cummax
fig2, ax2 = plt.subplots(figsize=(10,6))
drawdown.plot(ax=ax2, color='red')
ax2.set_title("策略回撤曲线")
ax2.set_ylabel("回撤 (%)")
img2_base64 = fig_to_base64(fig2)

# ---------------------------
# 本月买入建议
# ---------------------------
latest_score = score
latest_sma = monthly_prices.rolling(SMA_WINDOW).mean().iloc[-1]
trend_ok = monthly_prices.iloc[-1] > latest_sma
eligible = [t for t in latest_score.index if trend_ok.get(t, False)]
selected = [t for t in latest_score.index if t in eligible][:NUM_HOLD]

if (monthly_prices['SPY'].iloc[-1] - monthly_prices['SPY'].shift(MOM_12M).iloc[-1])/monthly_prices['SPY'].shift(MOM_12M).iloc[-1] < 0:
    selected = RISK_FREE

allocation = 1.0 / len(selected)
capital_after_cost = usd_capital * (1 - TRANSACTION_COST)

# ---------------------------
# 邮件正文（HTML嵌入图片）
# ---------------------------
body_html = f"""
<h3>月度动能轮动策略 - 升级版</h3>
<p>日期: {datetime.today().strftime('%Y-%m-%d')}<br>
本金: {RMB_CAPITAL:,.0f} RMB (~${usd_capital:,.2f} USD)</p >
<p><b>本月买入 ETF 建议 (Top-{NUM_HOLD}):</b></p >
<ul>
"""
for t in selected:
    body_html += f"<li>{t}: 分配 {allocation*100:.1f}% (~${capital_after_cost*allocation:,.2f})</li>"
body_html += "</ul>"
body_html += "<p>说明: 多周期动能 + 动能加速度 + 风险调整 + 趋势过滤 + 空仓保护 + 手续费滑点考虑。</p >"
body_html += f"<h4>策略累计收益曲线</h4>< img src='data:image/png;base64,{img1_base64}'><br>"
body_html += f"<h4>策略回撤曲线</h4>< img src='data:image/png;base64,{img2_base64}'>"

# ---------------------------
# 发送邮件
# ---------------------------
msg = MIMEMultipart("alternative")
msg["From"] = Header("月度动能轮动策略", "utf-8")
msg["To"] = Header("用户", "utf-8")
msg["Subject"] = Header("月度买入提醒（升级版+回测图嵌入正文）", "utf-8")
msg.attach(MIMEText(body_html, "html", "utf-8"))

try:
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
        server.set_debuglevel(1)
        server.login(sender, password)
        server.sendmail(sender, [receiver], msg.as_string())
    print("邮件已发送成功！")
except Exception as e:
    print("邮件发送失败:", e)
    traceback.print_exc()
    exit(1)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
monthly_momentum_alert_advanced_embedded.py

升级版月度动能轮动策略 + 邮件回测图嵌入正文
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.header import Header
import ssl
import io
import base64
import traceback

# ---------------------------
# 参数设置（与之前一致）
# ---------------------------
TICKERS = ["SPY","QQQ","VOO","VT","EFA","EEM","TLT","GLD"]
RISK_FREE = ["SHY"]
NUM_HOLD = 3
FEE_RATE = 0.001
SLIPPAGE = 0.0005
TRANSACTION_COST = FEE_RATE + SLIPPAGE

MOM_1M = 21
MOM_3M = 63
MOM_6M = 126
MOM_12M = 252
SMA_WINDOW = 200

RMB_CAPITAL = float(os.environ.get("RMB_CAPITAL", 650000))
USD_RMB_RATE = float(os.environ.get("USD_RMB_RATE", 7.1))
usd_capital = RMB_CAPITAL / USD_RMB_RATE

sender = os.environ.get("EMAIL_ADDRESS")
password = os.environ.get("EMAIL_PASSWORD")
receiver = os.environ.get("EMAIL_RECIPIENT", sender)
smtp_provider = os.environ.get("SMTP_PROVIDER", "qq").lower()

if not sender or not password:
    raise ValueError("请在 GitHub Secrets 中设置 EMAIL_ADDRESS 和 EMAIL_PASSWORD")

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
# 下载数据，只取 Adj Close
# ---------------------------
end_date = datetime.today().strftime("%Y-%m-%d")
data = yf.download(TICKERS, start="2010-01-01", end=end_date, auto_adjust=True, progress=False)
adj_close = data['Adj Close'].dropna(how='all')

# ---------------------------
# 简单回测
# ---------------------------
monthly_prices = adj_close.resample('M').last()
portfolio_value = pd.Series(index=monthly_prices.index, dtype=float)
portfolio_value.iloc[0] = usd_capital
last_hold = None

for i in range(1, len(monthly_prices)):
    date = monthly_prices.index[i]
    mom1 = (monthly_prices.iloc[i] - monthly_prices.shift(MOM_1M).iloc[i]) / monthly_prices.shift(MOM_1M).iloc[i]
    mom3 = (monthly_prices.iloc[i] - monthly_prices.shift(MOM_3M).iloc[i]) / monthly_prices.shift(MOM_3M).iloc[i]
    mom6 = (monthly_prices.iloc[i] - monthly_prices.shift(MOM_6M).iloc[i]) / monthly_prices.shift(MOM_6M).iloc[i]
    mom12 = (monthly_prices.iloc[i] - monthly_prices.shift(MOM_12M).iloc[i]) / monthly_prices.shift(MOM_12M).iloc[i]
    acc = mom1 - mom3
    vol = monthly_prices.pct_change().rolling(MOM_12M).std().iloc[i]
    
    score = 0.35*mom1.rank(ascending=False) + 0.35*mom3.rank(ascending=False) + 0.15*acc.rank(ascending=False) + 0.15*mom12.rank(ascending=False)
    score = score / vol
    score = score.sort_values(ascending=False)
    
    sma200 = monthly_prices.rolling(SMA_WINDOW).mean().iloc[i]
    trend_ok = monthly_prices.iloc[i] > sma200
    eligible = [t for t in score.index if trend_ok.get(t, False)]
    selected = [t for t in score.index if t in eligible][:NUM_HOLD]
    
    if last_hold is None:
        last_hold = selected
    
    turnover = len(set(selected) ^ set(last_hold)) / NUM_HOLD
    cost = turnover * TRANSACTION_COST
    monthly_ret = monthly_prices.loc[date, selected].pct_change().mean()
    portfolio_value.iloc[i] = portfolio_value.iloc[i-1]*(1 + monthly_ret - cost)
    last_hold = selected

# ---------------------------
# 生成图像为 base64
# ---------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

fig1, ax1 = plt.subplots(figsize=(10,6))
portfolio_value.plot(ax=ax1)
ax1.set_title("策略累计收益曲线")
ax1.set_ylabel("组合价值 (USD)")
img1_base64 = fig_to_base64(fig1)

cummax = portfolio_value.cummax()
drawdown = (portfolio_value - cummax)/cummax
fig2, ax2 = plt.subplots(figsize=(10,6))
drawdown.plot(ax=ax2, color='red')
ax2.set_title("策略回撤曲线")
ax2.set_ylabel("回撤 (%)")
img2_base64 = fig_to_base64(fig2)

# ---------------------------
# 本月买入建议
# ---------------------------
latest_score = score
latest_sma = monthly_prices.rolling(SMA_WINDOW).mean().iloc[-1]
trend_ok = monthly_prices.iloc[-1] > latest_sma
eligible = [t for t in latest_score.index if trend_ok.get(t, False)]
selected = [t for t in latest_score.index if t in eligible][:NUM_HOLD]

if (monthly_prices['SPY'].iloc[-1] - monthly_prices['SPY'].shift(MOM_12M).iloc[-1])/monthly_prices['SPY'].shift(MOM_12M).iloc[-1] < 0:
    selected = RISK_FREE

allocation = 1.0 / len(selected)
capital_after_cost = usd_capital * (1 - TRANSACTION_COST)

# ---------------------------
# 邮件正文（HTML嵌入图片）
# ---------------------------
body_html = f"""
<h3>月度动能轮动策略 - 升级版</h3>
<p>日期: {datetime.today().strftime('%Y-%m-%d')}<br>
本金: {RMB_CAPITAL:,.0f} RMB (~${usd_capital:,.2f} USD)</p >
<p><b>本月买入 ETF 建议 (Top-{NUM_HOLD}):</b></p >
<ul>
"""
for t in selected:
    body_html += f"<li>{t}: 分配 {allocation*100:.1f}% (~${capital_after_cost*allocation:,.2f})</li>"
body_html += "</ul>"
body_html += "<p>说明: 多周期动能 + 动能加速度 + 风险调整 + 趋势过滤 + 空仓保护 + 手续费滑点考虑。</p >"
body_html += f"<h4>策略累计收益曲线</h4>< img src='data:image/png;base64,{img1_base64}'><br>"
body_html += f"<h4>策略回撤曲线</h4>< img src='data:image/png;base64,{img2_base64}'>"

# ---------------------------
# 发送邮件
# ---------------------------
msg = MIMEMultipart("alternative")
msg["From"] = Header("月度动能轮动策略", "utf-8")
msg["To"] = Header("用户", "utf-8")
msg["Subject"] = Header("月度买入提醒（升级版+回测图嵌入正文）", "utf-8")
msg.attach(MIMEText(body_html, "html", "utf-8"))

try:
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
        server.set_debuglevel(1)
        server.login(sender, password)
        server.sendmail(sender, [receiver], msg.as_string())
    print("邮件已发送成功！")
except Exception as e:
    print("邮件发送失败:", e)
    traceback.print_exc()
    exit(1)
