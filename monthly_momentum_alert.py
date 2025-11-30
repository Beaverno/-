import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import traceback
import ssl

try:
    # ---------------------------
    # 用户参数
    # ---------------------------
    tickers = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "BIL"]
    num_hold = 2
    lookback_days = 252
    sma_window = 200
    usd_rmb_rate = 7.1
    RMB_capital = 70000

    # ---------------------------
    # 邮箱配置（从 Secrets 读取）
    # ---------------------------
    sender = os.environ.get("EMAIL_ADDRESS")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = sender
    smtp_provider = os.environ.get("SMTP_PROVIDER", "qq")  # qq / 163 / gmail

    if not sender or not password:
        raise ValueError("请在 GitHub Secrets 中设置 EMAIL_ADDRESS 和 EMAIL_PASSWORD")

    # SMTP 配置
    if smtp_provider.lower() == "qq":
        smtp_server = "smtp.qq.com"
        smtp_port = 465
    elif smtp_provider.lower() == "163":
        smtp_server = "smtp.163.com"
        smtp_port = 465
    elif smtp_provider.lower() == "gmail":
        smtp_server = "smtp.gmail.com"
        smtp_port = 465
    else:
        raise ValueError("不支持的 SMTP_PROVIDER")

    # ---------------------------
    # 下载历史数据
    # ---------------------------
    end_date = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(tickers, start="2010-01-01", end=end_date, auto_adjust=True)
    data = data.dropna(how="all")

    price_now = data.iloc[-1]
    price_prev = data.iloc[-lookback_days]

    ret_12m = (price_now - price_prev) / price_prev
    sma200 = data.rolling(sma_window).mean()
    sma200_now = sma200.iloc[-1]

    trend_ok = price_now > sma200_now
    ret_12m_filtered = ret_12m[trend_ok]

    top_n = ret_12m_filtered.sort_values(ascending=False).head(num_hold)
    allocation = 1.0 / len(top_n)

    # ---------------------------
    # 生成提醒文本
    # ---------------------------
    msg_body = f"【月度动能轮动策略提醒】\n日期: {datetime.today().strftime('%Y-%m-%d')}\n\n"
    msg_body += f"本金: {RMB_capital} RMB (约 {RMB_capital/usd_rmb_rate:.2f} USD)\n\n"
    msg_body += "本月买入 ETF 建议:\n"
    for t in top_n.index:
        usd_alloc = (RMB_capital/usd_rmb_rate)*allocation
        msg_body += f"- {t}: 分配 {allocation*100:.1f}% (~${usd_alloc:.2f})\n"
    msg_body += "\n提醒: 请在券商下单执行。"

    message = MIMEText(msg_body, "plain", "utf-8")
    message["From"] = Header("月度动能轮动策略", "utf-8")
    message["To"] = Header("用户", "utf-8")
    message["Subject"] = Header("月度买入提醒", "utf-8")

    # ---------------------------
    # 发送邮件
    # ---------------------------
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
        server.set_debuglevel(1)  # 输出详细 SMTP 日志
        server.login(sender, password)
        server.sendmail(sender, [receiver], message.as_string())

    print("邮件已发送成功！")

except Exception as e:
    print("脚本执行失败:", e)
    traceback.print_exc()
    exit(1)
