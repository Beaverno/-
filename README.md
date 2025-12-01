使用说明

1. 将 monthly_momentum_alert_advanced.py 放在仓库根目录


2. 创建 .github/workflows/monthly_momentum_alert.yml 并粘贴上述 YAML


3. 在仓库 Settings → Secrets → Actions 添加：

EMAIL_ADDRESS = 你的邮箱

EMAIL_PASSWORD = 授权码/应用专用密码

EMAIL_RECIPIENT = 收件人（可填自己邮箱）

SMTP_PROVIDER = qq / 163 / gmail

可选：

RMB_CAPITAL = 650000

USD_RMB_RATE = 7.1




4. 可以手动触发，也会每月 1 号自动执行


5. 邮件中会显示本月 Top-3 ETF 买入建议，已考虑 手续费、滑点、趋势、空仓保护


