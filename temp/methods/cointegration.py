from dataclasses import dataclass
from ..metrics import engle_granger_beta, adf_test_simple, corr_returns, estimate_beta_ols

@dataclass
class CointegrationConfig:
    select_pairs_per_window: int = 5
    min_form_bars: int = 60
    use_log_price: bool = False
    adf_lags: int = 1
    adf_crit: float = -3.3
    min_corr: float = 0.2
    bb_window_for_beta: int = 30
    max_pairs_scan: int = 100000  # 防止极端大规模

class CointegrationSelector:
    def __init__(self, **kwargs):
        self.config = CointegrationConfig(**kwargs)

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:
        import numpy as np
        cfg = self.config
        keys = list(prices.keys())
        candidates: List[Tuple[str, str, float, float]] = []
        scanned = 0
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                scanned += 1
                if scanned > cfg.max_pairs_scan:
                    break
                a, b = keys[i], keys[j]
                a_s, b_s = intersect_align(prices[a], prices[b])
                if len(a_s) < max(cfg.min_form_bars, 30):
                    continue
                beta_coint, resid = engle_granger_beta(a_s, b_s, use_log_price=cfg.use_log_price)
                if resid.empty or resid.isna().all():
                    continue
                t_stat = adf_test_simple(resid.values, lags=cfg.adf_lags)
                if np.isfinite(t_stat) and t_stat < cfg.adf_crit:
                    c = corr_returns(a_s, b_s)
                    if np.isfinite(c) and c >= cfg.min_corr:
                        # 使用协整beta
                        candidates.append((a, b, float(beta_coint), float(c)))
        # 互斥 + 排序（按相关度降序）
        candidates.sort(key=lambda x: -x[3])
        used: Set[str] = set()
        out: List[Pair] = []
        for a, b, beta, _ in candidates:
            if a in used or b in used:
                continue
            out.append((a, b, beta))
            used.add(a); used.add(b)
            if len(out) >= cfg.select_pairs_per_window:
                break
        return out