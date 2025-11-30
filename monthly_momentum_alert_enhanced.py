#!/usr/bin/env python3
# monthly_momentum_alert_enhanced.py
import os
import yfinance as yf
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import traceback
import ssl

def get_env_or_default(key, default=None):
    v = os.environ.get(key)
    return default if v is None else v

try:
    # ---------------------------
    # 参数（可通过 GitHub Secrets 覆盖）
    # ---------------------------
    # 资产池
    TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "BIL"]
    NUM_HOLD = int(get_env_or_default("NUM_HOLD", 2))       # Top-N
    LOOKBACK_DAYS_12M = int(get_env_or_default("LOOKBACK_DAYS_12M", 252))
    SMA_WINDOW = int(get_env_or_default("SMA_WINDOW", 200))
    USD_RMB_RATE = float(get_env_or_default("USD_RMB_RATE", 7.1))
    # 默认本金 650000 RMB，如果在 Secrets 中设置 RMB_CAPITAL 会覆盖
    RMB_CAPITAL = float(get_env_or_default("RMB_CAPITAL", 650000))
    # 动能窗口（交易日近似）
    MOM_1M = int(get_env_or_default("MOM_1M", 21))
    MOM_3M = int(get_env_or_default("MOM_3M", 63))
    MOM_6M = int(get_env_or_default("MOM_6M", 126))

    # ---------------------------
    # 邮箱配置（从 Secrets 读取）
    # ---------------------------
    sender = os.environ.get("EMAIL_ADDRESS")
    password = os.environ.get("EMAIL_PASSWORD")
    receiver = os.environ.get("EMAIL_RECIPIENT", sender)  # 可设置接收方，默认是发件人
    smtp_provider = get_env_or_default("SMTP_PROVIDER", "qq").lower()  # qq / 163 / gmail

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
    # 下载数据（兼容新版 yfinance）
    # ---------------------------
    end_date = datetime.today().strftime("%Y-%m-%d")
    data = yf.download(TICKERS, start="2010-01-01", end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError("yfinance 未能下载到任何数据，请稍后重试")

    data = data.dropna(how="all")

    # 最新价与历史差值
    price_now = data.iloc[-1]
    # 抵御少数 ticker 数据不够长的情况：使用 try/except 并过滤掉 NaN
    def safe_price_shift(days):
        if len(data) <= days:
            # 如果数据不足，返回 NaN 系列
            return pd.Series([float('nan')] * len(data.columns), index=data.columns)
        return data.iloc[-days]

    # 动能：1m,3m,6m 收益率（用收盘价近似）
    p_1m = safe_price_shift(MOM_1M)
    p_3m = safe_price_shift(MOM_3M)
    p_6m = safe_price_shift(MOM_6M)

    mom1 = (price_now - p_1m) / p_1m
    mom3 = (price_now - p_3m) / p_3m
    mom6 = (price_now - p_6m) / p_6m

    # 把 NaN 的标记出来并最终过滤掉
    mom_df = pd.DataFrame({"mom1": mom1, "mom3": mom3, "mom6": mom6})

    # 计算动能综合得分（权重可调整）
    # 逻辑：更看重中期和短期动能，同时加入加速度项 (mom1 - mom3)
    # score = w1*mom1_rank + w2*mom3_rank + w3*mom6_rank + w4*acc_rank
    # 我们用直接数值结合排名以降低极端值影响
    acc = mom1 - mom3  # 动能加速度（短期动能减中期动能）
    # 对于 NaN 行剔除
    valid_mask = (~mom_df.isna()).all(axis=1)
    candidates = mom_df[valid_mask].index.tolist()
    if len(candidates) == 0:
        raise RuntimeError("没有足够历史数据来计算动能，请检查资产池或数据源")

    # 为每个候选资产计算分数（这里用排名组合）
    ranks = pd.DataFrame(index=candidates)
    ranks["r_mom1"] = mom1.loc[candidates].rank(ascending=False)
    ranks["r_mom3"] = mom3.loc[candidates].rank(ascending=False)
    ranks["r_mom6"] = mom6.loc[candidates].rank(ascending=False)
    ranks["r_acc"] = acc.loc[candidates].rank(ascending=False)

    # 权重设置：动能短中期为主，加速度适度加分，长期动量较小权重
    score = 0.35 * ranks["r_mom1"] + 0.35 * ranks["r_mom3"] + 0.15 * ranks["r_acc"] + 0.15 * ranks["r_mom6"]
    score = score.sort_values()  # 小的 rank -> better, but we constructed so larger rank = worse; invert
    # Convert to descending score (higher better)
    score = score.rank(ascending=True)  # normalize to ranks (1 best)

    # SMA200 趋势过滤（基于调整后收盘价）
    sma200 = data.rolling(SMA_WINDOW).mean().iloc[-1]
    trend_ok = price_now > sma200

    # 从高到低选择 Top N，且满足趋势 (否则排除)
    eligible = [t for t in score.index if trend_ok.get(t, False)]
    # if eligible fewer than NUM_HOLD, allow assets without trend but best scores (fallback)
    selected = []
    for t in score.index.sort_values():
        if len(selected) >= NUM_HOLD:
            break
        if t in eligible:
            selected.append(t)
    if len(selected) < NUM_HOLD:
        # 补足：加入得分最高的剩余资产（无视趋势）
        for t in score.index.sort_values():
            if len(selected) >= NUM_HOLD:
                break
            if t not in selected:
                selected.append(t)

    # 如果仍为空（极端情况），则全部资金放 BIL
    if len(selected) == 0:
        selected = ["BIL"]

    allocation = 1.0 / len(selected)

    # 生成邮件正文
    usd_capital = RMB_CAPITAL / USD_RMB_RATE
    msg_lines = []
    msg_lines.append(f"【月度动能轮动策略提醒 - 增强动能】")
    msg_lines.append(f"日期: {datetime.today().strftime('%Y-%m-%d')}")
    msg_lines.append(f"本金: {RMB_CAPITAL:,.0f} RMB (约 ${usd_capital:,.2f} USD)")
    msg_lines.append("")
    msg_lines.append("本月买入 ETF 建议 (Top-{}):".format(NUM_HOLD))
    for t in selected:
        msg_lines.append(f"- {t}: 分配 {allocation*100:.1f}% (~${usd_capital * allocation:,.2f})")
    msg_lines.append("")
    msg_lines.append("说明: 使用短期+中期动能结合“动能加速度”增强捕捉加速上涨的品种。若目标 ETF 跌破 SMA200, 优先选择趋势通过的ETF；若没有满足趋势则选择得分最高的ETF。")
    body = "\n".join(msg_lines)

    # Build email
    message = MIMEText(body, "plain", "utf-8")
    message["From"] = Header("月度动能轮动策略", "utf-8")
    message["To"] = Header("用户", "utf-8")
    message["Subject"] = Header("月度买入提醒（增强动能）", "utf-8")

    # 发送邮件并输出详尽 SMTP 调试日志（有助于在 Actions 上定位连接问题）
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, smtp_port, context=context) as server:
        server.set_debuglevel(1)
        server.login(sender, password)
        server.sendmail(sender, [receiver], message.as_string())

    print("邮件已发送成功！")
    print(body)

except Exception as e:
    print("脚本执行失败:", e)
    traceback.print_exc()
    # 保持非零退出码以便 Actions 标记失败
    exit(1)
