from dataclasses import dataclass

@dataclass
class NSGA2Config:
    pairs_per_chrom: int = 5
    pop: int = 80
    gen: int = 60
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

class NSGA2Selector:
    def __init__(self, **kwargs):
        self.config = NSGA2Config(**kwargs)

    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:
        import numpy as np, random
        from ..metrics import unified_scaled_distance, estimate_beta_ols, bollinger_spread_score
        from ..data import intersect_align
        cfg = self.config
        random.seed(cfg.seed); np.random.seed(cfg.seed)

        keys = list(prices.keys())
        raw = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                d = unified_scaled_distance(prices[a], prices[b])
                if np.isfinite(d):
                    raw.append((a, b, d))
        raw.sort(key=lambda x: x[2])
        raw = raw[:cfg.candidate_cap]

        candidates = []
        for a, b, _ in raw:
            a_s, b_s = intersect_align(prices[a], prices[b])
            if len(a_s) < max(cfg.min_form_bars, 30): continue
            beta = estimate_beta_ols(a_s, b_s, use_log_price=cfg.use_log_price,
                                     min_len=max(30, cfg.bb_window_for_beta))
            candidates.append((a, b, beta))
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
                    if len(chrom) >= K: break
            return chrom

        def obj_pair(idx):
            a, b, beta = candidates[idx]
            a_s, b_s = intersect_align(prices[a], prices[b])
            s1, dd = bollinger_spread_score(
                np.log(a_s) if cfg.use_log_price else a_s,
                np.log(b_s) if cfg.use_log_price else b_s,
                beta, cfg.bb_window, cfg.takeprofit_z, cfg.close_to_sma, cfg.fee_bps
            )
            if not np.isfinite(s1): s1 = -1e6
            if not np.isfinite(dd): dd = 1e6
            return s1, dd

        cache_obj = [obj_pair(i) for i in range(len(candidates))]

        def objectives(ch):
            tot_score = sum(cache_obj[i][0] for i in ch)
            tot_dd = sum(max(0.0, cache_obj[i][1]) for i in ch)
            return (tot_score, tot_dd)

        def dominates(a_obj, b_obj):
            better_or_equal = (a_obj[0] >= b_obj[0]) and (a_obj[1] <= b_obj[1])
            strictly_better = (a_obj[0] > b_obj[0]) or (a_obj[1] < b_obj[1])
            return better_or_equal and strictly_better

        def fast_nondominated_sort(pop_objs):
            S = [[] for _ in pop_objs]; n = [0] * len(pop_objs)
            fronts = [[]]
            for p in range(len(pop_objs)):
                for q in range(len(pop_objs)):
                    if p == q: continue
                    if dominates(pop_objs[p], pop_objs[q]):
                        S[p].append(q)
                    elif dominates(pop_objs[q], pop_objs[p]):
                        n[p] += 1
                if n[p] == 0:
                    fronts[0].append(p)
            i = 0
            while fronts[i]:
                Q = []
                for p in fronts[i]:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            Q.append(q)
                i += 1
                fronts.append(Q)
            fronts.pop()
            return fronts

        def crowding_distance(front, pop_objs):
            if not front: return []
            distances = [0.0 for _ in front]
            for m in range(2):
                vals = [pop_objs[i][m] for i in front]
                order = np.argsort(vals)
                distances[order[0]] = distances[order[-1]] = float('inf')
                vmin, vmax = vals[order[0]], vals[order[-1]]
                if vmax == vmin: continue
                for k in range(1, len(front) - 1):
                    prev_v = vals[order[k - 1]]
                    next_v = vals[order[k + 1]]
                    distances[order[k]] += (next_v - prev_v) / (vmax - vmin)
            return distances

        def selection(pop, pop_objs, pop_size):
            fronts = fast_nondominated_sort(pop_objs)
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) > pop_size:
                    dist = crowding_distance(front, pop_objs)
                    order = np.argsort([-dist[i] for i in range(len(front))])
                    for idx in order:
                        if len(new_pop) >= pop_size: break
                        new_pop.append(pop[front[idx]])
                    break
                else:
                    for i in front:
                        new_pop.append(pop[i])
            return new_pop

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
            cut = random.randint(1, max(1, K-1))
            c1 = p1[:cut] + [g for g in p2 if g not in p1[:cut]]
            c2 = p2[:cut] + [g for g in p1 if g not in p2[:cut]]
            return repair(c1), repair(c2)

        def mutate(ch):
            if random.random() > 0.2: return ch
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

        pop = [random_chrom() for _ in range(cfg.pop)]
        pop_objs = [objectives(ch) for ch in pop]

        for _ in range(cfg.gen):
            children = []
            while len(children) < cfg.pop:
                p1 = random.choice(pop); p2 = random.choice(pop)
                c1, c2 = crossover(p1, p2)
                c1 = mutate(c1); c2 = mutate(c2)
                children.extend([c1, c2])
            children = children[:cfg.pop]
            children_objs = [objectives(ch) for ch in children]
            pop = pop + children
            pop_objs = pop_objs + children_objs
            pop = selection(pop, pop_objs, cfg.pop)
            pop_objs = [objectives(ch) for ch in pop]

        # 从第一前沿选收益最高
        def fast_nondominated_sort_indices(population):
            return fast_nondominated_sort(pop_objs)
        fronts = fast_nondominated_sort(pop_objs)
        f0 = fronts[0] if fronts else list(range(len(pop_objs)))
        best_idx = max(f0, key=lambda i: pop_objs[i][0])
        best = pop[best_idx]
        out = []
        used = set()
        for idx in best:
            a, b, beta = candidates[idx]
            if a in used or b in used: continue
            out.append((a, b, beta)); used.add(a); used.add(b)
        return out