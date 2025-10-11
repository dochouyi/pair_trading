import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
from freqtrade.strategy import IStrategy
from pandas import DataFrame

@dataclass
class PairSelectorConfig:
    timeframe: str = "5m"
    form_period: int = 200
    bb_window: int = 30          # 仅用于 beta 估计最小窗长限制
    min_form_bars: int = 60
    recompute_every: int = 20
    min_corr: float = 0.2
    use_log_price: bool = False
    select_pairs_per_window: int = 1
    max_candidates_per_method: int = 20

def estimate_beta_on_window(a: pd.Series, b: pd.Series, use_log_price=True, min_len: int = 30) -> float:
    a = np.log(a.dropna()) if use_log_price else a.dropna().copy()
    b = np.log(b.dropna()) if use_log_price else b.dropna().copy()
    idx = a.index.intersection(b.index)
    a = a.loc[idx]; b = b.loc[idx]
    if len(a) < min_len:
        return 1.0
    cov = np.cov(b.values, a.values)[0, 1]
    var = np.var(b.values)
    beta = cov / var if var > 0 else 1.0
    if not np.isfinite(beta):
        beta = 1.0
    return float(np.clip(beta, 0.1, 10.0))

def unified_scaled_distance(a: pd.Series, b: pd.Series) -> float:
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

def euclidean_distance_method(prices: Dict[str, pd.Series], cfg: PairSelectorConfig, topk: int) -> List[Tuple[str, str, float]]:
    scores = []
    keys = list(prices.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            d = unified_scaled_distance(prices[a], prices[b])
            if not np.isfinite(d):
                continue
            scores.append((a, b, d))
    scores.sort(key=lambda x: x[2])
    pairs = []
    for a, b, _ in scores[:topk]:
        beta = estimate_beta_on_window(prices[a], prices[b], use_log_price=cfg.use_log_price, min_len=max(30, cfg.bb_window))
        pairs.append((a, b, beta))
    return pairs

class PairSelectorOnly(IStrategy):
    """
    仅选择币对，不进行交易/下单/发信号。
    使用方法：
    1) 将本文件放入 user_data/strategies/
    2) 配置文件中启用本策略：--strategy PairSelectorOnly
    3) 观察日志中输出的所选配对及 beta
    """
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 260   # >= form_period
    can_short = False

    # 关闭交易相关
    minimal_roi = {"0": 10}
    stoploss = -0.99

    cfg = PairSelectorConfig()
    _last_recompute_index: int = 0
    _current_pairs: List[Tuple[str, str, float]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeframe = self.cfg.timeframe

    def informative_pairs(self):
        # 允许读取白名单中的所有对
        return [(p, self.timeframe) for p in self.dp.current_whitelist()]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]

        wl = sorted(self.dp.current_whitelist())
        if not wl or pair != wl[0]:
            return dataframe

        # 收集所有pair的收盘价
        df_map: Dict[str, DataFrame] = {}
        for p in wl:
            try:
                dfp = self.dp.get_pair_dataframe(p, self.timeframe)
                df_map[p] = dfp.copy()
            except Exception:
                continue

        closes: Dict[str, pd.Series] = {}
        for p, dfp in df_map.items():
            if "close" not in dfp.columns:
                continue
            s = dfp["close"].dropna()
            if s.shape[0] >= self.cfg.min_form_bars:
                closes[p] = s.iloc[-self.cfg.form_period:]

        if len(closes) < 2:
            return dataframe

        cur_len = len(dataframe)
        if self._last_recompute_index == 0:
            self._last_recompute_index = cur_len

        # 周期性重算候选
        if (cur_len - self._last_recompute_index) >= self.cfg.recompute_every and cur_len > self.cfg.form_period:
            base_pairs = euclidean_distance_method(closes, self.cfg, topk=self.cfg.max_candidates_per_method)

            # 互斥选择 + 二次过滤：相关性与样本量
            used = set()
            filtered: List[Tuple[str, str, float]] = []
            for a, b, beta in base_pairs:
                if a in used or b in used:
                    continue
                sA = closes[a].pct_change().dropna()
                sB = closes[b].pct_change().dropna()
                idx = sA.index.intersection(sB.index)
                if len(idx) < max(self.cfg.min_form_bars, 30):
                    continue
                corr = sA.loc[idx].corr(sB.loc[idx])
                if (corr is None) or (not np.isfinite(corr)) or (corr < self.cfg.min_corr):
                    continue
                filtered.append((a, b, beta))
                used.add(a); used.add(b)
                if len(filtered) >= self.cfg.select_pairs_per_window:
                    break

            self._current_pairs = filtered
            self._last_recompute_index = cur_len
            if filtered:
                self.logger.info(f"[PAIR-SELECT] Selected pairs: {filtered}")
            else:
                self.logger.info("[PAIR-SELECT] No pairs selected this round.")

        return dataframe

    # 不产生任何进出场信号
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        return dataframe
