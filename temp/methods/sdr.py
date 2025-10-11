
from dataclasses import dataclass
from typing import Optional
from ..metrics import series_returns, adf_test_simple, corr_returns, estimate_beta_ols

def _market_series(prices: Dict[str, pd.Series], mode: str, symbol: Optional[str]) -> pd.Series:
    dfs = pd.concat(prices.values(), axis=1, join="inner")
    dfs.columns = list(prices.keys())
    if mode == "symbol" and symbol is not None and symbol in prices:
        base = prices[symbol]
        return series_returns(base)
    px_mean = dfs.mean(axis=1)
    return series_returns(px_mean)

def sdr_gamma_diff(a: pd.Series, b: pd.Series, market_r: pd.Series) -> float:
    ra, rb = series_returns(a), series_returns(b)
    idx = ra.index.intersection(rb.index).intersection(market_r.index)
    if len(idx) < 30:
        return 0.0
    ri_a = ra.loc[idx].values
    ri_b = rb.loc[idx].values
    rm = market_r.loc[idx].values
    vx = rm - rm.mean()
    def ols_beta(y, x_centered):
        vy = y - y.mean()
        denom = (x_centered**2).sum()
        if denom <= 0:
            return 0.0
        return float((x_centered * vy).sum() / denom)
    gamma_a = ols_beta(ri_a, vx)
    gamma_b = ols_beta(ri_b, vx)
    return float(gamma_a - gamma_b)

@dataclass
class SDRConfig:
    select_pairs_per_window: int = 5
    max_candidates: int = 200
    min_form_bars: int = 60
    min_corr: float = 0.2
    use_log_price: bool = False
    bb_window_for_beta: int = 30
    market_mode: str = "mean"  # "mean" 或 "symbol"
    market_symbol: Optional[str] = None

class SDRSelector:
    def __init__(self, **kwargs):
        self.config = SDRConfig(**kwargs)

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:
        import numpy as np
        cfg = self.config
        market_r = _market_series(prices, cfg.market_mode, cfg.market_symbol)
        keys = list(prices.keys())
        scored: List[Tuple[str, str, float]] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                a_s, b_s = intersect_align(prices[a], prices[b])
                if len(a_s) < max(cfg.min_form_bars, 30):
                    continue
                ra, rb = series_returns(a_s), series_returns(b_s)
                idx = ra.index.intersection(rb.index).intersection(market_r.index)
                if len(idx) < 30:
                    continue
                gamma = sdr_gamma_diff(a_s, b_s, market_r)
                gt = ra.loc[idx].values - rb.loc[idx].values - gamma * market_r.loc[idx].values
                t_stat = adf_test_simple(gt, lags=1)
                if not np.isfinite(t_stat):
                    continue
                scored.append((a, b, -t_stat))
        scored.sort(key=lambda x: -x[2])
        used: Set[str] = set()
        out: List[Pair] = []
        for a, b, _ in scored[: cfg.max_candidates]:
            a_s, b_s = intersect_align(prices[a], prices[b])
            c = corr_returns(a_s, b_s)
            if not np.isfinite(c) or c < cfg.min_corr:
                continue
            beta = estimate_beta_ols(a_s, b_s, use_log_price=cfg.use_log_price,
                                     min_len=max(30, cfg.bb_window_for_beta))
            if a in used or b in used:
                continue
            out.append((a, b, beta))
            used.add(a); used.add(b)
            if len(out) >= cfg.select_pairs_per_window:
                break
        return out