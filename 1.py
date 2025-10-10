import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据准备
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30)
market_prices = 1000 + np.cumsum(np.random.normal(0, 2, size=30))
stock_prices = 50 + np.cumsum(np.random.normal(0, 1, size=30)) + 0.2 * (market_prices - 1000)

a = pd.Series(stock_prices, index=dates)
b = pd.Series(market_prices, index=dates)

def estimate_beta_on_window(a: pd.Series, b: pd.Series, use_log_price=True, min_len: int = 30) -> float:
    a = np.log(a.dropna()) if use_log_price else a.dropna().copy()
    b = np.log(b.dropna()) if use_log_price else b.dropna().copy()
    idx = a.index.intersection(b.index)
    a = a.loc[idx]; b = b.loc[idx]
    if len(a) < min_len:
        return 1.0
    # 简化：用OLS的近似斜率 = cov / var
    cov = np.cov(b.values, a.values)[0, 1]
    var = np.var(b.values)
    beta = cov / var if var > 0 else 1.0
    if not np.isfinite(beta):
        beta = 1.0
    return float(np.clip(beta, 0.1, 10.0))

# 计算 Beta
beta = estimate_beta_on_window(a, b)
print(f"估算的 Beta 值为: {beta:.3f}")

# 画图部分
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# 1. 价格走势
axes[0].plot(a.index, a.values, label='股票价格', color='blue')
axes[0].plot(b.index, b.values, label='市场指数', color='orange')
axes[0].set_title("价格走势")
axes[0].legend()
axes[0].set_ylabel("价格")

# 2. 对数收益率散点图与回归线
log_a = np.log(a)
log_b = np.log(b)
axes[1].scatter(log_b, log_a, label='对数价格点', alpha=0.7)
# 回归直线
x_vals = np.linspace(log_b.min(), log_b.max(), 100)
y_vals = log_a.mean() + beta * (x_vals - log_b.mean())
axes[1].plot(x_vals, y_vals, color='red', label=f'回归线 (Beta={beta:.2f})')
axes[1].set_xlabel("市场指数对数价格")
axes[1].set_ylabel("股票对数价格")
axes[1].set_title("对数价格关系与Beta回归线")
axes[1].legend()

plt.tight_layout()
plt.show()
