# Momentum Monthly Alert

自动化月度动能轮动策略提醒仓库。

## 使用方法

1. Fork 本仓库到你的 GitHub 账号
2. 设置仓库 Secrets:
   - `QQ_EMAIL` = 你的 QQ 邮箱
   - `QQ_PASSWORD` = QQ 邮箱授权码（开启 SMTP 服务后生成）
3. 手动触发一次 workflow 测试邮件是否收到
4. 每月 1 号自动发送 ETF 买入提醒邮件
