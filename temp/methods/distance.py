from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd
from ..base import Pair
from ..data import intersect_align
from ..metrics import unified_scaled_distance, corr_returns, estimate_beta_ols

@dataclass
class DistanceConfig:
    select_pairs_per_window: int = 5
    max_candidates: int = 200
    min_form_bars: int = 60
    min_corr: float = 0.2
    use_log_price: bool = False
    bb_window_for_beta: int = 30  # 仅用于 beta 估计的最小窗约束

class DistanceSelector:
    def __init__(self, **kwargs):
        # 允许直接在构造函数中覆盖配置
        self.config = DistanceConfig(**kwargs)

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:
        cfg = self.config
        keys = list(prices.keys())
        scores: List[Tuple[str, str, float]] = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                d = unified_scaled_distance(prices[a], prices[b])
                if np.isfinite(d):
                    scores.append((a, b, d))
        scores.sort(key=lambda x: x[2])
        used: Set[str] = set()
        out: List[Pair] = []
        for a, b, _ in scores[: cfg.max_candidates]:
            a_s, b_s = intersect_align(prices[a], prices[b])
            if len(a_s) < max(cfg.min_form_bars, 30):
                continue
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