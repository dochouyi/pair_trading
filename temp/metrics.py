
import numpy as np
import pandas as pd
import math
from typing import Tuple
from .data import intersect_align

def series_returns(s: pd.Series) -> pd.Series:
    return s.pct_change().dropna()

def corr_returns(a: pd.Series, b: pd.Series) -> float:
    ra, rb = series_returns(a), series_returns(b)
    idx = ra.index.intersection(rb.index)
    if len(idx) < 3:
        return np.nan
    return float(ra.loc[idx].corr(rb.loc[idx]))

def estimate_beta_ols(a: pd.Series, b: pd.Series, use_log_price=True, min_len: int = 30) -> float:
    import numpy as np
    a = np.log(a.dropna()) if use_log_price else a.dropna().copy()
    b = np.log(b.dropna()) if use_log_price else b.dropna().copy()
    a, b = intersect_align(a, b)
    if len(a) < min_len:
        return 1.0
    x = b.values
    y = a.values
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx**2).sum()
    if denom <= 0:
        return 1.0
    beta = (vx * vy).sum() / denom
    if not np.isfinite(beta):
        beta = 1.0
    return float(np.clip(beta, 0.1, 10.0))

def unified_scaled_distance(a: pd.Series, b: pd.Series) -> float:
    import numpy as np
    idx = a.dropna().index.intersection(b.dropna().index)
    if len(idx) == 0:
        return np.inf
    A = a.loc[idx].values.reshape(-1, 1)
    B = b.loc[idx].values.reshape(-1, 1)
    X = np.vstack([A, B])
    mu = X.mean()
    sd = X.std() if X.std() > 0 else 1.0
    As = (A - mu) / sd
    Bs = (B - mu) / sd
    return float(np.linalg.norm(As.flatten() - Bs.flatten()))

def adf_test_simple(x: np.ndarray, lags: int = 1) -> float:
    x = x.astype(float)
    dx = np.diff(x)
    x_1 = x[:-1]
    n = len(dx)
    if n - lags - 1 <= 3:
        return np.nan
    Y = dx[lags:]
    X_cols = []
    X_cols.append(np.ones(len(Y)))
    X_cols.append(x_1[lags:])
    for i in range(1, lags + 1):
        X_cols.append(dx[lags - i: -i])
    X = np.vstack(X_cols).T
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return np.nan
    beta_hat = XtX_inv @ (X.T @ Y)
    residuals = Y - X @ beta_hat
    sigma2 = (residuals @ residuals) / (len(Y) - X.shape[1])
    var_beta = sigma2 * XtX_inv
    phi = beta_hat[1]
    se_phi = math.sqrt(var_beta[1, 1]) if var_beta[1, 1] > 0 else np.inf
    t_phi = phi / se_phi if se_phi > 0 else np.nan
    return float(t_phi)

def engle_granger_beta(a: pd.Series, b: pd.Series, use_log_price=True) -> Tuple[float, pd.Series]:
    import numpy as np
    a = np.log(a.dropna()) if use_log_price else a.dropna().copy()
    b = np.log(b.dropna()) if use_log_price else b.dropna().copy()
    a, b = intersect_align(a, b)
    x = b.values; y = a.values
    if len(x) < 30:
        return 1.0, pd.Series(index=a.index, dtype=float)
    X = np.vstack([np.ones(len(x)), x]).T
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        return 1.0, pd.Series(index=a.index, dtype=float)
    beta_hat = XtX_inv @ (X.T @ y)
    alpha, beta = beta_hat[0], beta_hat[1]
    resid = y - (alpha + beta * x)
    return float(np.clip(beta, 0.1, 10.0)), pd.Series(resid, index=a.index)

def bollinger_spread_score(a: pd.Series, b: pd.Series, beta: float, window: int, take_z: float,
                           close_to_sma: bool, fee_bps: float) -> Tuple[float, float]:
    import numpy as np
    a, b = a.dropna(), b.dropna()
    idx = a.index.intersection(b.index)
    if len(idx) < window + 10:
        return -np.inf, np.inf
    A = a.loc[idx].values
    B = b.loc[idx].values
    S = A - beta * B
    sma = pd.Series(S).rolling(window).mean().values
    std = pd.Series(S).rolling(window).std(ddof=0).values
    up = sma + take_z * std
    dn = sma - take_z * std

    pnl = 0.0
    peak = 0.0
    maxdd = 0.0
    pos = 0
    entry = 0.0
    fee = fee_bps / 10000.0

    for t in range(window, len(S)):
        s = S[t]; mu = sma[t]; u = up[t]; d = dn[t]
        if np.isnan(mu) or np.isnan(u) or np.isnan(d):
            continue
        if pos == 0:
            if s > u:
                pos = -1; entry = s; pnl -= abs(s) * fee
            elif s < d:
                pos = 1; entry = s; pnl -= abs(s) * fee
        else:
            close_cond = (abs(s - mu) < 1e-12) if close_to_sma else (d < s < u)
            if close_cond:
                pnl += (s - entry) * pos
                pnl -= abs(s) * fee
                pos = 0
            else:
                float_pnl = pnl + (s - entry) * pos
                peak = max(peak, float_pnl)
                maxdd = max(maxdd, peak - float_pnl)

    if pos != 0 and not np.isnan(sma[-1]):
        pnl += (S[-1] - entry) * pos
        pnl -= abs(S[-1]) * fee
        pos = 0

    std_mean = np.nanmean(std)
    norm = std_mean if (std_mean is not None and std_mean > 1e-12) else 1.0
    score = pnl / norm
    maxdd_n = maxdd / norm if norm > 0 else maxdd
    return float(score), float(maxdd_n)