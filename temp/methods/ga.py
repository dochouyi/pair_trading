from dataclasses import dataclass

@dataclass
class GAConfig:
    pairs_per_chrom: int = 5
    pop: int = 60
    gen: int = 40
    cxp: float = 0.8
    mutp: float = 0.2
    candidate_cap: int = 200
    min_form_bars: int = 60
    use_log_price: bool = False
    bb_window_for_beta: int = 30
    # 评分参数
    bb_window: int = 30
    takeprofit_z: float = 2.0
    close_to_sma: bool = True
    fee_bps: float = 0.0
    seed: int = 42
    # 候选来源
    candidate_source: str = "distance"  # "distance" 或 "sdr"

class GASelector:
    def __init__(self, **kwargs):
        self.config = GAConfig(**kwargs)

    def _build_candidates(self, prices: Dict[str, pd.Series]) -> List[Pair]:
        import numpy as np, random
        from ..metrics import unified_scaled_distance, estimate_beta_ols
        from ..data import intersect_align
        from ..metrics import series_returns, adf_test_simple
        cfg = self.config
        keys = list(prices.keys())
        pool = []
        if cfg.candidate_source == "distance":
            raw = []
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    a, b = keys[i], keys[j]
                    d = unified_scaled_distance(prices[a], prices[b])
                    if np.isfinite(d):
                        raw.append((a, b, d))
            raw.sort(key=lambda x: x[2])
            raw = raw[:cfg.candidate_cap]
            for a, b, _ in raw:
                a_s, b_s = intersect_align(prices[a], prices[b])
                if len(a_s) < max(cfg.min_form_bars, 30):
                    continue
                beta = estimate_beta_ols(a_s, b_s, use_log_price=cfg.use_log_price,
                                         min_len=max(30, cfg.bb_window_for_beta))
                pool.append((a, b, beta))
        else:
            from .sdr import _market_series, sdr_gamma_diff
            market_r = _market_series(prices, "mean", None)
            raw = []
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
                    if np.isfinite(t_stat):
                        raw.append((a, b, -t_stat))
            raw.sort(key=lambda x: -x[2])
            raw = raw[:cfg.candidate_cap]
            for a, b, _ in raw:
                a_s, b_s = intersect_align(prices[a], prices[b])
                if len(a_s) < max(cfg.min_form_bars, 30):
                    continue
                beta = estimate_beta_ols(a_s, b_s, use_log_price=cfg.use_log_price,
                                         min_len=max(30, cfg.bb_window_for_beta))
                pool.append((a, b, beta))
        return pool

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:
        import numpy as np, random
        from ..data import intersect_align
        from ..metrics import bollinger_spread_score
        cfg = self.config
        random.seed(cfg.seed); np.random.seed(cfg.seed)
        candidates = self._build_candidates(prices)
        # 验证样本
        valid = []
        for a, b, beta in candidates:
            a_s, b_s = intersect_align(prices[a], prices[b])
            if len(a_s) >= max(cfg.min_form_bars, 30):
                valid.append((a, b, beta))
        candidates = valid
        if not candidates:
            return []

        K = min(cfg.pairs_per_chrom, max(1, len(candidates)//5))

        def random_chrom():
            used = set(); chrom = []
            idxs = list(range(len(candidates))); random.shuffle(idxs)
            for idx in idxs:
                a, b, beta = candidates[idx]
                if (a not in used) and (b not in used):
                    chrom.append(idx); used.add(a); used.add(b)
                    if len(chrom) >= K:
                        break
            return chrom

        def fitness(ch):
            score_sum = 0.0
            for idx in ch:
                a, b, beta = candidates[idx]
                a_s, b_s = intersect_align(prices[a], prices[b])
                s1, _ = bollinger_spread_score(
                    np.log(a_s) if cfg.use_log_price else a_s,
                    np.log(b_s) if cfg.use_log_price else b_s,
                    beta, cfg.bb_window, cfg.takeprofit_z, cfg.close_to_sma, cfg.fee_bps
                )
                if not np.isfinite(s1): s1 = -1e6
                score_sum += s1
            return score_sum

        pop = [random_chrom() for _ in range(cfg.pop)]
        fit = [fitness(ch) for ch in pop]

        def tournament():
            k = 3
            cand = random.sample(range(len(pop)), k)
            best = max(cand, key=lambda i: fit[i])
            return pop[best]

        def repair(ch):
            used = set(); out = []
            for idx in ch:
                a, b, _ = candidates[idx]
                if a in used or b in used: continue
                out.append(idx); used.add(a); used.add(b)
                if len(out) >= K: break
            if len(out) < K:
                pool = list(range(len(candidates))); random.shuffle(pool)
                for idx in pool:
                    a, b, _ = candidates[idx]
                    if a in used or b in used or idx in out: continue
                    out.append(idx); used.add(a); used.add(b)
                    if len(out) >= K: break
            return out[:K]

        def crossover(p1, p2):
            if random.random() > cfg.cxp: return p1[:], p2[:]
            cut = random.randint(1, max(1, K-1))
            c1 = p1[:cut] + [g for g in p2 if g not in p1[:cut]]
            c2 = p2[:cut] + [g for g in p1 if g not in p2[:cut]]
            return repair(c1), repair(c2)

        def mutate(ch):
            if random.random() > cfg.mutp: return ch
            used_assets = set()
            for idx in ch:
                a, b, _ = candidates[idx]
                used_assets.add(a); used_assets.add(b)
            pool = []
            for i, (a, b, _) in enumerate(candidates):
                if (a not in used_assets) and (b not in used_assets):
                    pool.append(i)
            if not pool: return ch
            pos = random.randrange(len(ch))
            ch2 = ch[:]; ch2[pos] = random.choice(pool)
            return repair(ch2)

        for _ in range(cfg.gen):
            new_pop = []
            while len(new_pop) < cfg.pop:
                p1 = tournament(); p2 = tournament()
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1); c2 = mutate(c2)
                new_pop.extend([c1, c2])
            pop = new_pop[:cfg.pop]
            fit = [fitness(ch) for ch in pop]

        best_idx = max(range(len(pop)), key=lambda i: fit[i])
        best = pop[best_idx]
        out = []
        used = set()
        for idx in best:
            a, b, beta = candidates[idx]
            if a in used or b in used: continue
            out.append((a, b, beta)); used.add(a); used.add(b)
        return out