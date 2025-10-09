# -*- coding: utf-8 -*-
# Monica(gpt-5): Pairs Trading v3 aligned with Ko et al. (2023), keep Distance/SDR/NSGA-II only
# - Methods: Distance, SDR, NSGA-II
# - Trading: Bollinger bands on spread, SMA±2*STD, cross-entry, revert-to-SMA exit
# - Windowing:
#   - Paper-like preset: form=5, trade=1 (very short), bb_window=5 (for Bollinger)
#   - Stable preset (default): form=200, trade=200, bb_window=50
# - Risk controls: time stop, z extreme stop, light vol-weighting
# - NSGA-II: return+, drawdown-, winrate+; detailed debug prints
#
# Reference:
# Ko, P.-C. et al. (2023) Pairs Trading Strategies in Cryptocurrency Markets: A Comparative Study between Statistical Methods and Evolutionary Algorithms.
# https://doi.org/10.3390/engproc2023038074

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler

# Evolutionary algorithms
try:
    from deap import base, creator, tools, algorithms
except Exception:
    base = creator = tools = algorithms = None

@dataclass
class Config:
    # 是否使用论文对齐预设
    use_paper_like_config: bool = False  # True: form=5, trade=1, bb=5; False: 更稳健默认设置

    # 窗口设置（会在__post_init__中根据use_paper_like_config覆盖）
    freq: str = "5min"
    form_period: int = 200   # 形成期
    trade_period: int = 200  # 交易期
    select_pairs_per_window: int = 5
    max_pairs_for_ea: int = 5  # NSGA-II每窗最多选几对（论文为最多5对）

    # 资金与费用
    capital_per_trade: float = 1000.0
    fee_rate: float = 0.0005  # 论文未明确，Binance现货常见 5-10 bps；适度保守

    # 价差与回归
    use_log_price: bool = False
    rebalance_beta: bool = False

    # SDR：使用BTC作为市场
    sdr_use_btc_as_market: bool = True

    # Bollinger参数（论文：SMA ± 2*STD）
    bb_window: int = 50
    bb_k: float = 2.0
    std_clip: float = 1e-6

    # 入场/离场（以z为准；用于触发）
    z_exit_to_sma: bool = True   # 回归SMA即平仓
    z_stop: float = 4.0          # 极端止损，避免长趋势
    bars_in_trade_max: int = 80  # 时间止损

    # 轻量风控
    vol_weight: bool = True  # 以对两边波动给权重

    # NSGA-II
    seed: int = 123
    debug: bool = True  # 为NSGA-II增加详细日志

    # 候选限制
    max_candidates_per_method: int = 50  # 每窗每法最多候选对数

    def __post_init__(self):
        if self.use_paper_like_config:
            # 论文对齐预设：形成=5，交易=1，布林窗口=5
            self.form_period = 5
            self.trade_period = 1
            self.bb_window = 5

def parse_symbol_from_filename(fn: str) -> str:
    base = os.path.basename(fn)
    if base.endswith(".json"):
        base = base[:-5]
    if "-5m-futures" in base:
        return base.split("-5m-futures")[0]
    return base

def load_freqtrade_file(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        raw = json.load(f)
    arr = np.array(raw, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Unexpected data shape in {path}: {arr.shape}")
    df = pd.DataFrame(arr, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df.set_index("ts", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["open","high","low","close","volume"]]

def load_freqtrade_dir(dir_path: str, max_symbols: int = None) -> Dict[str, pd.DataFrame]:
    files = [fn for fn in os.listdir(dir_path) if fn.endswith("-5m-futures.json")]
    files.sort()
    if max_symbols:
        files = files[:max_symbols]
    data = {}
    for fn in files:
        full = os.path.join(dir_path, fn)
        sym = parse_symbol_from_filename(full)
        try:
            df = load_freqtrade_file(full)
            data[sym] = df
        except Exception as e:
            print(f"Skip {fn}: {e}")
    if not data:
        raise FileNotFoundError("No -5m-futures.json found.")
    return data

def estimate_beta_on_window(a_series: pd.Series, b_series: pd.Series, use_log_price=True) -> float:
    a = np.log(a_series.dropna()) if use_log_price else a_series.dropna().copy()
    b = np.log(b_series.dropna()) if use_log_price else b_series.dropna().copy()
    idx = a.index.intersection(b.index)
    a = a.loc[idx]; b = b.loc[idx]
    if len(a) < 30:
        return 1.0
    X = sm.add_constant(b.values)
    try:
        model = sm.OLS(a.values, X).fit()
        beta = float(model.params[1])
    except Exception:
        beta = 1.0
    if not np.isfinite(beta):
        beta = 1.0
    return float(np.clip(beta, 0.1, 10.0))

def robust_std(s: pd.Series, window: int, std_clip: float) -> Tuple[pd.Series, pd.Series]:
    sma = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std().clip(lower=std_clip)
    return sma, std

def pair_vol(a: pd.Series, b: pd.Series, window: int) -> Tuple[float, float]:
    ra = a.pct_change().rolling(window, min_periods=window).std().iloc[-1]
    rb = b.pct_change().rolling(window, min_periods=window).std().iloc[-1]
    ra = float(ra) if np.isfinite(ra) and ra > 0 else 1e-4
    rb = float(rb) if np.isfinite(rb) and rb > 0 else 1e-4
    return ra, rb

@dataclass
class TradeResult:
    pair: Tuple[str, str]
    entries: int
    pnl_list: List[float]
    total_return_pct: float
    avg_return_pct: float
    max_return_pct: float
    min_return_pct: float
    max_drawdown_pct: float
    winrate: float

def trade_pair_bollinger(A: pd.Series, B: pd.Series, cfg: Config, beta: float) -> TradeResult:
    A = A.dropna(); B = B.dropna()
    idx = A.index.intersection(B.index)
    A = A.loc[idx]; B = B.loc[idx]
    if len(A) < max(10, cfg.bb_window):
        return TradeResult(("A","B"), 0, [], 0, 0, 0, 0, 0, 0.0)

    spread = A - beta*B
    sma, std = robust_std(spread, cfg.bb_window, cfg.std_clip)
    upper = sma + cfg.bb_k * std
    lower = sma - cfg.bb_k * std

    if cfg.vol_weight:
        va, vb = pair_vol(A, B, cfg.bb_window)
        w_a = 1.0/max(va,1e-9); w_b = 1.0/max(vb,1e-9)
        norm = w_a + w_b
        w_a /= norm; w_b /= norm
    else:
        w_a = w_b = 0.5

    in_pos = False; pos_type = None
    ea = eb = None; bars = 0

    pnl_list = []; wins = 0; losses = 0; entries = 0
    equity = [1.0]

    for t in range(len(spread)):
        sp = spread.iloc[t]
        ma = sma.iloc[t]
        sd = std.iloc[t]
        if np.isnan(ma) or np.isnan(sd) or sd <= 0:
            continue

        pa = A.iloc[t]; pb = B.iloc[t]
        up = upper.iloc[t]; lo = lower.iloc[t]

        if not in_pos:
            if sp >= up:
                in_pos = True; pos_type = "shortA_longB"; ea, eb = pa, pb; bars = 0; entries += 1
            elif sp <= lo:
                in_pos = True; pos_type = "longA_shortB"; ea, eb = pa, pb; bars = 0; entries += 1
        else:
            bars += 1
            close_now = False
            if cfg.z_exit_to_sma and ((pos_type == "shortA_longB" and sp <= ma) or (pos_type == "longA_shortB" and sp >= ma)):
                close_now = True
            # 极端z止损
            z = (sp - ma) / sd
            if abs(z) >= cfg.z_stop:
                close_now = True
            # 时间止损
            if bars >= cfg.bars_in_trade_max or t == len(spread) - 1:
                close_now = True

            if close_now:
                cap = cfg.capital_per_trade
                qty_a = w_a*cap/max(ea,1e-9)
                qty_b = w_b*cap/max(eb,1e-9)
                if pos_type == "shortA_longB":
                    pnl = qty_a*(ea-pa) + qty_b*(pb-eb)
                else:
                    pnl = qty_a*(pa-ea) + qty_b*(eb-pb)
                fee = cfg.fee_rate*(ea*qty_a+eb*qty_b+pa*qty_a+pb*qty_b)
                pnl -= fee
                pnl_pct = pnl/(2*cfg.capital_per_trade)*100.0
                pnl_list.append(pnl_pct)
                if pnl_pct > 0: wins += 1
                else: losses += 1
                in_pos = False; pos_type = None; ea = eb = None; bars = 0

        equity.append(equity[-1] * (1 + (pnl_list[-1]/100.0 if pnl_list else 0.0)))

    def max_dd(arr):
        arr = np.array(arr)
        roll = np.maximum.accumulate(arr)
        dd = (arr - roll) / roll
        return abs(dd.min()) * 100.0

    if len(pnl_list) == 0:
        return TradeResult(("A","B"), entries, [], 0, 0, 0, 0, 0, 0.0)

    wr = wins / max(1, wins+losses)
    return TradeResult(
        pair=("A","B"),
        entries=entries,
        pnl_list=pnl_list,
        total_return_pct=float(np.sum(pnl_list)),
        avg_return_pct=float(np.mean(pnl_list)),
        max_return_pct=float(np.max(pnl_list)),
        min_return_pct=float(np.min(pnl_list)),
        max_drawdown_pct=float(max_dd(equity)),
        winrate=float(wr)
    )

# 统计型配对挑选：Distance
def euclidean_distance_method(prices: Dict[str, pd.Series], cfg: Config, topk: int) -> List[Tuple[str, str, float]]:
    scaler = StandardScaler()
    normalized={}
    for sym,s in prices.items():
        vals=s.values.reshape(-1,1)
        if len(vals)==0: continue
        normalized[sym]=scaler.fit_transform(vals).ravel()
    scores=[]
    for a,b in combinations(normalized.keys(),2):
        d=np.linalg.norm(normalized[a]-normalized[b])
        scores.append((a,b,d))
    scores.sort(key=lambda x:x[2])
    # 估beta
    pairs=[]
    for a,b,_ in scores[:topk]:
        beta=estimate_beta_on_window(prices[a], prices[b], use_log_price=cfg.use_log_price)
        pairs.append((a,b,beta))
    return pairs

# 统计型配对挑选：SDR
def sdr_method(prices: Dict[str, pd.Series], market: pd.Series, cfg: Config, topk: int) -> List[Tuple[str, str, float]]:
    results=[]
    ret_m=np.log(market).diff().dropna()
    keys=list(prices.keys())
    for a,b in combinations(keys,2):
        ra=np.log(prices[a]).diff().dropna()
        rb=np.log(prices[b]).diff().dropna()
        idx=ret_m.index.intersection(ra.index).intersection(rb.index)
        if len(idx)<max(60, cfg.bb_window):
            continue
        X=sm.add_constant(ret_m.loc[idx].values)
        try:
            beta_a=sm.OLS(ra.loc[idx].values, X).fit().params
            beta_b=sm.OLS(rb.loc[idx].values, X).fit().params
            resid_a=ra.loc[idx].values-(beta_a[0]+beta_a[1]*ret_m.loc[idx].values)
            resid_b=rb.loc[idx].values-(beta_b[0]+beta_b[1]*ret_m.loc[idx].values)
            spread=resid_a-resid_b
            p=adfuller(spread, autolag="AIC")[1]
            results.append((a,b,p))
        except Exception:
            continue
    results.sort(key=lambda x:x[2])
    pairs=[]
    for a,b,_ in results[:topk]:
        beta=estimate_beta_on_window(prices[a], prices[b], use_log_price=cfg.use_log_price)
        pairs.append((a,b,beta))
    return pairs

# 构造候选对（Distance/SDR）
def make_candidate_pairs(form_prices: Dict[str, pd.Series],
                         method: str,
                         market_series: pd.Series,
                         cfg: Config) -> List[Tuple[str,str,float]]:
    topk = cfg.max_candidates_per_method
    if method=="Distance":
        return euclidean_distance_method(form_prices, cfg, topk=topk)
    elif method=="SDR":
        return sdr_method(form_prices, market_series, cfg, topk=topk)
    else:
        raise ValueError("Unknown method for candidate building")

def select_pairs_nsgaii(form_prices: Dict[str, pd.Series],
                        trade_prices: Dict[str, pd.Series],
                        method_seed_pairs: List[Tuple[str,str,float]],
                        cfg: Config) -> List[Tuple[str,str,float]]:
    # 去重
    uniq = []
    seen = set()
    for (a,b,beta) in method_seed_pairs:
        key = (a,b) if a < b else (b,a)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((a,b,beta))
    if len(uniq) == 0:
        return []

    candidates = uniq[:40]  # 控制复杂度
    c_count = len(candidates)
    K = min(cfg.max_pairs_for_ea, max(1, c_count//5))  # 与论文一致：最多5对

    tp = {s: trade_prices[s] for s in trade_prices}

    def eval_subset(idxs):
        total_pnl = 0.0
        eq = [1.0]
        wins = 0; losses = 0; trades = 0
        for i in idxs:
            a,b,beta = candidates[i]
            a_tp = tp.get(a); b_tp = tp.get(b)
            if a_tp is None or b_tp is None:
                continue
            tr = trade_pair_bollinger(a_tp["close"], b_tp["close"], cfg, beta)
            total_pnl += tr.total_return_pct
            trades += tr.entries
            w = int(round(tr.winrate * max(1, tr.entries)))
            wins += w
            losses += max(0, tr.entries - w)
            eq.append(eq[-1] * (1 + tr.total_return_pct/100.0))
        if len(eq) > 1:
            roll = np.maximum.accumulate(np.array(eq))
            dd = abs(((np.array(eq)-roll)/roll).min())*100.0
        else:
            dd = 0.0
        wr = wins / max(1, wins+losses)
        return total_pnl, dd, wr, trades

    # 无DEAP时：退化为贪心（按总收益）
    if base is None or creator is None or tools is None:
        scored = []
        for idx,(a,b,beta) in enumerate(candidates):
            a_tp = tp.get(a); b_tp = tp.get(b)
            if a_tp is None or b_tp is None:
                continue
            tr = trade_pair_bollinger(a_tp["close"], b_tp["close"], cfg, beta)
            scored.append((idx, tr.total_return_pct))
        if not scored:
            return []
        scored.sort(key=lambda x: -x[1])
        best_idxs = [i for i,_ in scored[:K]]
        if cfg.debug:
            print(f"[NSGA-II:Fallback] Selected idxs: {best_idxs} from {len(candidates)} candidates")
        return [candidates[i] for i in best_idxs]

    # 定义DEAP结构
    np.random.seed(cfg.seed)
    # 避免重复创建报错
    try:
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, 1.0))  # return+, drawdown-, winrate+
    except Exception:
        pass
    try:
        creator.create("IndividualNS", list, fitness=creator.FitnessMulti)
    except Exception:
        pass
    toolbox = base.Toolbox()

    # 个体初始化：从候选集中无放回采样K个索引
    def init_individual():
        return creator.IndividualNS(list(np.random.choice(range(c_count), size=K, replace=False)))

    toolbox.register("individual", tools.initIterate, creator.IndividualNS, init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_ns(ind):
        pnl, dd, wr, trades = eval_subset(ind)
        # 对交易数太低的组合添加微弱惩罚，减少偶然性
        pnl_adj = pnl - 0.05*max(0, 2-trades)
        return (pnl_adj, dd, wr)

    toolbox.register("evaluate", eval_ns)

    # 交叉：TwoPoint + 去重补全
    def cx_unique(ind1, ind2):
        c1, c2 = tools.cxTwoPoint(list(ind1), list(ind2))
        def fix(child_list):
            seen_idx = set()
            new = []
            for g in child_list:
                if g not in seen_idx:
                    new.append(int(g)); seen_idx.add(int(g))
            choices = list(set(range(c_count)) - set(new))
            np.random.shuffle(choices)
            while len(new) < K and choices:
                new.append(int(choices.pop()))
            while len(new) < K:
                r = int(np.random.randint(0, c_count))
                if r not in new:
                    new.append(r)
            return new
        c1 = creator.IndividualNS(fix(c1))
        c2 = creator.IndividualNS(fix(c2))
        return (c1, c2)

    # 变异：随机替换一个基因
    def mut_swap(ind):
        pos = np.random.randint(0, K)
        choices = list(set(range(c_count)) - set(ind))
        if choices:
            ind[pos] = int(np.random.choice(choices))
        return (ind,)

    toolbox.register("mate", cx_unique)
    toolbox.register("mutate", mut_swap)
    toolbox.register("select", tools.selNSGA2)

    pop_size = 60
    ngen = 40
    cxpb, mutpb = 0.7, 0.3

    pop = toolbox.population(n=pop_size)
    if cfg.debug:
        print(f"[NSGA-II] Candidates: {c_count}, select K={K}, pop={pop_size}, ngen={ngen}")

    # 进化（mu+lambda）
    pop, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=pop_size, lambda_=pop_size,
        cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False
    )

    # 取帕累托前沿并按合成评分挑一个代表
    front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    scored = []
    for ind in front:
        pnl, dd, wr, trades = eval_subset(ind)
        score = pnl - 0.6*dd + 6.0*wr  # 略偏收益，抑制回撤
        scored.append((score, pnl, dd, wr, trades, ind))
    if not scored:
        return []
    scored.sort(key=lambda x: -x[0])
    best = scored[0]
    best_ind = best[5]
    if cfg.debug:
        print(f"[NSGA-II] Pareto size={len(front)}; top score={best[0]:.3f}, pnl={best[1]:.2f}, dd={best[2]:.2f}, wr={best[3]:.2%}, trades={best[4]}")
        print(f"[NSGA-II] Selected idxs: {list(best_ind)}")
        for idx in list(best_ind):
            a,b,beta = candidates[idx]
            print(f"  - Pair: {a}-{b}, beta={beta:.3f}")

    return [candidates[i] for i in best_ind]

def backtest(data: Dict[str, pd.DataFrame], cfg: Config, method: str, seed_method_for_nsga: str = "Distance") -> pd.DataFrame:
    assert method in ["Distance","SDR","NSGA-II"]
    symbols=list(data.keys())
    closes={s: data[s]["close"].copy() for s in symbols}

    # 统一索引
    all_index=None
    for s in symbols:
        idx=closes[s].index
        all_index=idx if all_index is None else all_index.union(idx)
    all_index=all_index.sort_values()

    # 市场序列（SDR使用）
    if cfg.sdr_use_btc_as_market:
        btc_key=next((k for k in symbols if k.upper().startswith("BTC")), None)
        market_series=closes[btc_key].reindex(all_index).interpolate() if btc_key else pd.DataFrame({k: closes[k].reindex(all_index) for k in symbols}).mean(axis=1)
    else:
        market_series=pd.DataFrame({k: closes[k].reindex(all_index) for k in symbols}).mean(axis=1)

    total_bars=len(all_index)
    win_size=cfg.form_period+cfg.trade_period
    results=[]

    for start in range(0, total_bars - win_size + 1, cfg.trade_period):
        form_range=all_index[start:start+cfg.form_period]
        trade_range=all_index[start+cfg.form_period:start+win_size]

        form_prices={s: data[s].reindex(form_range).dropna() for s in symbols}
        form_prices={k:v for k,v in form_prices.items() if len(v)>=max(10, cfg.bb_window)}
        if len(form_prices)<2:
            continue

        trade_prices={s: data[s].reindex(trade_range).dropna() for s in symbols}
        trade_prices={k:v for k,v in trade_prices.items() if len(v)>=max(10, cfg.bb_window)}
        if len(trade_prices)<2:
            continue

        if method in ["Distance","SDR"]:
            base_pairs = make_candidate_pairs(
                {k:form_prices[k]["close"] for k in form_prices},
                method,
                market_series.loc[form_range],
                cfg
            )
            pairs = base_pairs[:cfg.select_pairs_per_window]
        else:
            # NSGA-II：先用 seed_method_for_nsga 产生候选（默认 Distance），也可改为 "SDR"
            base_pairs = make_candidate_pairs(
                {k:form_prices[k]["close"] for k in form_prices},
                seed_method_for_nsga,
                market_series.loc[form_range],
                cfg
            )
            if len(base_pairs) == 0:
                pairs = []
            else:
                pairs = select_pairs_nsgaii(
                    {k:form_prices[k][["close"]] for k in form_prices},
                    {k:trade_prices[k][["close"]] for k in trade_prices},
                    base_pairs,
                    cfg
                )

        if not pairs:
            continue

        # 逐对推进交易期回测
        for (a,b,beta) in pairs:
            ta_full=trade_prices.get(a); tb_full=trade_prices.get(b)
            if ta_full is None or tb_full is None:
                continue
            ta = ta_full["close"]; tb = tb_full["close"]
            tr = trade_pair_bollinger(ta, tb, cfg, beta=beta)
            results.append({
                "method": method,
                "pair": f"{a}-{b}",
                "start": trade_range[0],
                "end": trade_range[-1],
                "entries": tr.entries,
                "total_return": tr.total_return_pct,
                "avg_return": tr.avg_return_pct,
                "max_return": tr.max_return_pct,
                "min_return": tr.min_return_pct,
                "max_drawdown": tr.max_drawdown_pct,
                "winrate": tr.winrate
            })

    df_res=pd.DataFrame(results)
    if df_res.empty:
        print(f"[{method}] No trades generated.")
        return df_res
    summary=df_res.groupby("method")[["total_return","entries","avg_return","max_return","min_return","max_drawdown","winrate"]].mean()
    print("\n=== Summary (Window-level average) ===")
    print(summary)
    return df_res

if __name__ == "__main__":
    # 数据目录（按需修改）
    data_dir="data/bybit/futures"
    print(f"Loading data from: {data_dir}")
    data=load_freqtrade_dir(data_dir, max_symbols=12)

    # 选择是否使用“论文对齐预设”
    cfg=Config(
        use_paper_like_config=False,  # 若需严格论文参数，改为 True
        freq="5min",
        form_period=400,   # 若 use_paper_like_config=True 会被覆盖为 5
        trade_period=200,  # 若 use_paper_like_config=True 会被覆盖为 1
        select_pairs_per_window=5,
        max_pairs_for_ea=5,
        capital_per_trade=1000.0,
        fee_rate=0.0004,   # 手续费（可根据交易所调整）
        use_log_price=False,
        rebalance_beta=False,
        sdr_use_btc_as_market=True,
        seed=123,
        bb_window=50,      # 若 use_paper_like_config=True 会被覆盖为 5
        bb_k=2.0,
        std_clip=1e-6,
        z_exit_to_sma=True,
        z_stop=4.0,
        bars_in_trade_max=80,
        vol_weight=True,
        debug=True,        # 打开NSGA-II调试输出
        max_candidates_per_method=50
    )

    # 仅运行 Distance / SDR / NSGA-II
    methods=["Distance","SDR","NSGA-II"]
    for method in methods:
        print(f"\nRunning method: {method}")
        if method == "NSGA-II":
            # 你可以将 seed_method_for_nsga 改为 "SDR" 来尝试更强的候选质量
            df_res=backtest(data, cfg, method=method, seed_method_for_nsga="Distance")
        else:
            df_res=backtest(data, cfg, method=method)
        if not df_res.empty:
            print(df_res.tail(5))
