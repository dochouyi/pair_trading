import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from ta.volatility import BollingerBands

# =========================
# 交易记录的数据结构
# =========================
@dataclass
class TradeRecord:
    method: str
    pair: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str
    beta: float
    w_a: float
    w_b: float
    qty_a: float
    qty_b: float
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    fee: float
    pnl: float
    pnl_pct: float
    capital_per_leg: float
    bars_held: int

# =========================
# 全局配置
# =========================
@dataclass
class Config:
    freq: str = "5min"
    form_period: int = 200             # 形成期长度
    select_pairs_per_window: int = 5   # 每次重算候选时保留的TopK
    capital_per_trade: float = 1000.0
    fee_rate: float = 0.0005
    use_log_price: bool = False
    sdr_use_btc_as_market: bool = True
    bb_window: int = 50
    bb_k: float = 2.0
    std_clip: float = 1e-6
    z_exit_to_sma: bool = True
    z_stop: float = 4.0
    vol_weight: bool = True
    seed: int = 123
    debug: bool = True
    max_candidates_per_method: int = 50
    save_dir: Optional[str] = None
    method: str = "Distance"           # Distance 或 SDR
    recompute_candidates_every: int = 20

    # 执行与风险控制
    use_next_bar_price: bool = False   # True 用下一根价格执行（更保守），False 用当前bar价格
    slippage_bps: float = 0.0          # 成交滑点，基点
    max_holding_bars: Optional[int] = 12*24  # 最长持仓bar数，None表示不限制
    cooldown_bars: int = 0             # 平仓后冷却若干bar不再入场

    # 候选过滤
    min_corr: float = 0.2              # 形成期收益相关性阈值
    max_adf_p: float = 0.2             # ADF p值上限
    use_adf_filter: bool = True        # 是否应用ADF过滤
    min_form_bars: int = 60            # 形成期最小样本（收益）

    # 数量分配
    beta_neutral_qty: bool = True      # 用beta做名义对冲
    min_qty_clip: float = 1e-9

    # 新增：利润门槛和平仓缓冲
    min_take_profit_pct: float = 0.0   # 最小净利润门槛（%）。0 表示不要求
    enforce_profit_on_cross: bool = True  # 对“跨中轨”的常规平仓是否强制应用利润门槛
    takeprofit_exit_buffer_z: float = 0.0 # z回到中轨的缓冲，例如0.2；0表示无缓冲

    def __post_init__(self):
        if self.bb_k <= 0:
            raise ValueError("cfg.bb_k must be > 0")

# =========================
# 工具函数
# =========================
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
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"], errors="coerce").astype("Int64"), unit="ms", utc=True)
    df = df.dropna(subset=["ts"])
    df.set_index("ts", inplace=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df[["open","high","low","close","volume"]]

def load_freqtrade_dir(dir_path: str, max_symbols: int = None) -> Dict[str, pd.DataFrame]:
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Data directory not found: {dir_path}")
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
    a = np.log(a_series) if use_log_price else a_series.dropna().copy()
    b = np.log(b_series) if use_log_price else b_series.dropna().copy()
    idx = a.index.intersection(b.index)
    a = a.loc[idx]; b = b.loc[idx]

    X = sm.add_constant(b.values)
    try:
        model = sm.OLS(a.values, X).fit()
        beta = float(model.params[1])
    except Exception:
        beta = 1.0
    if not np.isfinite(beta):
        beta = 1.0
    return float(np.clip(beta, 0.1, 10.0))

def pair_vol(a: pd.Series, b: pd.Series, window: int) -> Tuple[float, float]:
    ra = a.pct_change().rolling(window, min_periods=window).std().iloc[-1]
    rb = b.pct_change().rolling(window, min_periods=window).std().iloc[-1]
    ra = float(ra) if np.isfinite(ra) and ra > 0 else 1e-4
    rb = float(rb) if np.isfinite(rb) and rb > 0 else 1e-4
    return ra, rb

def unified_scaled_distance(a: pd.Series, b: pd.Series) -> float:
    idx = a.dropna().index.intersection(b.dropna().index)
    if len(idx) == 0:
        return np.inf
    A = a.loc[idx].values.reshape(-1,1)
    B = b.loc[idx].values.reshape(-1,1)
    X = np.vstack([A, B])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    As = Xs[:len(idx), 0]
    Bs = Xs[len(idx):, 0]
    return float(np.linalg.norm(As - Bs))

# =========================
# 候选构建方法
# =========================
def euclidean_distance_method(prices: Dict[str, pd.Series], cfg: Config, topk: int) -> List[Tuple[str, str, float]]:
    scores=[]
    keys = list(prices.keys())
    for a,b in combinations(keys,2):
        d = unified_scaled_distance(prices[a], prices[b])
        if not np.isfinite(d):
            continue
        scores.append((a,b,d))
    scores.sort(key=lambda x:x[2])
    pairs=[]
    for a,b,_ in scores[:topk]:
        beta=estimate_beta_on_window(prices[a], prices[b], use_log_price=cfg.use_log_price, min_len=max(30, cfg.bb_window))
        pairs.append((a,b,beta))
    return pairs

def sdr_method(prices: Dict[str, pd.Series], market: pd.Series, cfg: Config, topk: int) -> List[Tuple[str, str, float]]:
    results=[]
    ret_m=np.log(market).diff().dropna()
    keys=list(prices.keys())
    for a,b in combinations(keys,2):
        ra=np.log(prices[a]).diff().dropna()
        rb=np.log(prices[b]).diff().dropna()
        idx=ret_m.index.intersection(ra.index).intersection(rb.index)
        if len(idx)<max(cfg.min_form_bars, cfg.bb_window):
            continue
        X = sm.add_constant(np.asarray(ret_m.loc[idx].values).reshape(-1,1))
        try:
            model_a = sm.OLS(np.asarray(ra.loc[idx].values), X).fit()
            model_b = sm.OLS(np.asarray(rb.loc[idx].values), X).fit()
            fitted_a = model_a.predict(X)
            fitted_b = model_b.predict(X)
            spread=fitted_a - fitted_b
            p=adfuller(spread, autolag="AIC")[1]
            results.append((a,b,p))
        except Exception:
            continue
    results.sort(key=lambda x:x[2])
    pairs=[]
    for a,b,_ in results[:topk]:
        beta=estimate_beta_on_window(prices[a], prices[b], use_log_price=cfg.use_log_price, min_len=max(30, cfg.bb_window))
        pairs.append((a,b,beta))
    return pairs



def make_candidate_pairs(form_prices: Dict[str, pd.Series],
                         method: str,
                         market: pd.Series,
                         cfg: Config) -> List[Tuple[str,str,float]]:
    topk = cfg.max_candidates_per_method
    if method=="Distance":
        base_pairs = euclidean_distance_method(form_prices, cfg, topk=topk)
    elif method=="SDR":
        base_pairs = sdr_method(form_prices, market, cfg, topk=topk)
    else:
        raise ValueError("Unknown method for candidate building")

    # 二次过滤：相关性与ADF（对形成期）
    filtered=[]
    for a,b,beta in base_pairs:
        sA = form_prices[a].pct_change().dropna()
        sB = form_prices[b].pct_change().dropna()
        idx = sA.index.intersection(sB.index)
        if len(idx) < max(cfg.min_form_bars, cfg.bb_window):
            continue
        corr = sA.loc[idx].corr(sB.loc[idx])
        if (corr is None) or (not np.isfinite(corr)) or (corr < cfg.min_corr):
            continue
        if cfg.use_adf_filter:
            if cfg.use_log_price:
                spread = (np.log(form_prices[a].loc[idx]) - beta*np.log(form_prices[b].loc[idx])).dropna()
            else:
                spread = (form_prices[a].loc[idx] - beta*form_prices[b].loc[idx]).dropna()
            if len(spread) >= max(40, cfg.bb_window):
                try:
                    p = adfuller(spread.values, autolag="AIC")[1]
                    if not np.isfinite(p) or p > cfg.max_adf_p:
                        continue
                except Exception:
                    continue
            else:
                continue
        filtered.append((a,b,beta))
        if len(filtered) >= cfg.select_pairs_per_window:
            break
    return filtered

# =========================
# 指标与执行价
# =========================
def compute_bb(spread: pd.Series, cfg: Config):
    bb = BollingerBands(close=spread, window=cfg.bb_window, window_dev=cfg.bb_k, fillna=False)
    sma = bb.bollinger_mavg()
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    sd = ((upper - sma) / cfg.bb_k).clip(lower=cfg.std_clip)
    return sma, upper, lower, sd

def get_exec_price(series: pd.Series, t: int, use_next: bool, slippage_bps: float, min_clip: float=1e-12) -> float:
    if use_next and (t+1) < len(series):
        px = float(series.iloc[t+1])
    else:
        px = float(series.iloc[t])
    px = max(px, min_clip)
    return px * (1.0 + slippage_bps*1e-4)

def compute_position_sizes(ea: float, eb: float, beta: float, cfg: Config,
                           va: Optional[float]=None, vb: Optional[float]=None) -> Tuple[float,float,float,float]:
    cap = cfg.capital_per_trade
    if cfg.vol_weight and (va is not None) and (vb is not None):
        w_a = 1.0/max(va, cfg.min_qty_clip)
        w_b = 1.0/max(vb, cfg.min_qty_clip)
    else:
        w_a = w_b = 0.5
    s = w_a + w_b
    w_a /= s; w_b /= s
    if cfg.beta_neutral_qty:
        denom = (ea**2 + (beta*eb)**2)
        if denom <= 0:
            qty_a = (w_a*cap)/max(ea, cfg.min_qty_clip)
            qty_b = (w_b*cap)/max(eb, cfg.min_qty_clip)
        else:
            N_target = (w_a*cap + beta*w_b*cap) / (1.0 + beta)
            qty_a = N_target / max(ea, cfg.min_qty_clip)
            qty_b = (N_target / max(beta, cfg.min_qty_clip)) / max(eb, cfg.min_qty_clip)
    else:
        qty_a = (w_a*cap)/max(ea, cfg.min_qty_clip)
        qty_b = (w_b*cap)/max(eb, cfg.min_qty_clip)
    return float(w_a), float(w_b), float(qty_a), float(qty_b)

# =========================
# 交易信号与管理
# =========================
def try_open_position(A: pd.Series, B: pd.Series, beta: float, cfg: Config,
                      t_idx: int, label: str):
    spread = A - beta*B
    sma, upper, lower, sd = compute_bb(spread, cfg)
    if t_idx < len(spread):
        sp = spread.iloc[t_idx]
        ma = sma.iloc[t_idx]; up = upper.iloc[t_idx]; lo = lower.iloc[t_idx]
        if np.isnan(ma) or np.isnan(up) or np.isnan(lo):
            return None
        if sp >= up:
            side = "shortA_longB"
        elif sp <= lo:
            side = "longA_shortB"
        else:
            return None

        pa = get_exec_price(A, t_idx, cfg.use_next_bar_price, cfg.slippage_bps)
        pb = get_exec_price(B, t_idx, cfg.use_next_bar_price, cfg.slippage_bps)
        ts = spread.index[t_idx]
        va, vb = pair_vol(A.iloc[:t_idx+1], B.iloc[:t_idx+1], cfg.bb_window)
        w_a, w_b, qty_a, qty_b = compute_position_sizes(pa, pb, beta, cfg, va, vb)

        state = {
            "pair_label": label,
            "side": side,
            "entry_price_a": float(pa),
            "entry_price_b": float(pb),
            "entry_time": ts,
            "w_a": float(w_a),
            "w_b": float(w_b),
            "qty_a": float(qty_a),
            "qty_b": float(qty_b),
            "beta": float(beta),
            "entry_bar_index": int(t_idx)
        }
        return state
    return None

def step_manage_position(A: pd.Series, B: pd.Series, beta: float, cfg: Config,
                         state: Dict, start_bar: int, end_bar: int,
                         method_name: str, window_start: pd.Timestamp, window_end: pd.Timestamp):
    """
    管理已有持仓，从 start_bar 推进到 end_bar（含end_bar）
    平仓规则：
      - 止损：abs(z) >= cfg.z_stop 时无条件平仓；
      - 常规平仓（跨中轨）：side=shortA_longB 要求 spread <= 中轨；side=longA_shortB 要求 spread >= 中轨
        * 若 enforce_profit_on_cross=True，则还需满足净 pnl_pct >= min_take_profit_pct
        * 若 takeprofit_exit_buffer_z>0，还需 z 回到 [-buffer, buffer] 内
      - 最长持仓：超过 cfg.max_holding_bars 则强平
    执行价：按配置使用当前或下一根价格，并加入滑点
    """
    spread = A - beta*B
    sma, upper, lower, sd = compute_bb(spread, cfg)
    in_pos = True
    side = state["side"]
    ea = float(state["entry_price_a"])
    eb = float(state["entry_price_b"])
    et = pd.Timestamp(state["entry_time"])
    w_a = float(state["w_a"])
    w_b = float(state["w_b"])
    qty_a = float(state["qty_a"])
    qty_b = float(state["qty_b"])
    entry_bar_index = int(state.get("entry_bar_index", start_bar))

    closed_records = []
    bars_consumed = start_bar
    for t in range(start_bar, min(end_bar+1, len(spread))):
        sp = spread.iloc[t]; ma = sma.iloc[t]; sdd = sd.iloc[t]
        if np.isnan(ma) or np.isnan(sdd) or sdd <= 0:
            bars_consumed = t
            continue

        z = (sp - ma) / sdd

        # 止损
        stop_flag = abs(z) >= cfg.z_stop

        # 中轨缓冲判定
        cross_ok = False
        if cfg.z_exit_to_sma:
            if side == "shortA_longB":
                if cfg.takeprofit_exit_buffer_z > 0:
                    cross_ok = sp <= ma and abs(z) <= cfg.takeprofit_exit_buffer_z
                else:
                    cross_ok = sp <= ma
            elif side == "longA_shortB":
                if cfg.takeprofit_exit_buffer_z > 0:
                    cross_ok = sp >= ma and abs(z) <= cfg.takeprofit_exit_buffer_z
                else:
                    cross_ok = sp >= ma

        # 最长持仓
        hold_bars = t - entry_bar_index
        timeout_flag = (cfg.max_holding_bars is not None) and (hold_bars >= cfg.max_holding_bars)

        # 若满足跨中轨，需要检查利润门槛（如开启）
        want_close = False
        if cross_ok:
            if cfg.enforce_profit_on_cross and (cfg.min_take_profit_pct is not None) and (cfg.min_take_profit_pct > 0.0):
                # 用当前t的执行价估算净收益，再决定是否平仓
                pa_tmp = get_exec_price(A, t, cfg.use_next_bar_price, cfg.slippage_bps)
                pb_tmp = get_exec_price(B, t, cfg.use_next_bar_price, cfg.slippage_bps)
                if side == "shortA_longB":
                    pnl_tmp = qty_a*(ea-pa_tmp) + qty_b*(pb_tmp-eb)
                else:
                    pnl_tmp = qty_a*(pa_tmp-ea) + qty_b*(eb-pb_tmp)
                fee_tmp = cfg.fee_rate*(ea*qty_a+eb*qty_b+pa_tmp*qty_a+pb_tmp*qty_b)
                pnl_tmp -= fee_tmp
                pnl_pct_tmp = pnl_tmp/(cfg.capital_per_trade)*100.0
                want_close = pnl_pct_tmp >= cfg.min_take_profit_pct
            else:
                want_close = True

        # 综合平仓逻辑：止损或超时无视利润门槛；常规跨中轨需通过门槛
        if stop_flag or timeout_flag or want_close:
            pa = get_exec_price(A, t, cfg.use_next_bar_price, cfg.slippage_bps)
            pb = get_exec_price(B, t, cfg.use_next_bar_price, cfg.slippage_bps)
            ts = spread.index[t]

            if side == "shortA_longB":
                pnl = qty_a*(ea-pa) + qty_b*(pb-eb)
            else:
                pnl = qty_a*(pa-ea) + qty_b*(eb-pb)
            fee = cfg.fee_rate*(ea*qty_a+eb*qty_b+pa*qty_a+pb*qty_b)
            pnl -= fee
            pnl_pct = pnl/(cfg.capital_per_trade)*100.0

            rec = TradeRecord(
                method=method_name or "",
                pair=state.get("pair_label","A-B"),
                window_start=window_start,
                window_end=window_end,
                entry_time=et,
                exit_time=ts,
                side=side,
                beta=float(beta),
                w_a=float(w_a),
                w_b=float(w_b),
                qty_a=float(qty_a),
                qty_b=float(qty_b),
                entry_price_a=float(ea),
                entry_price_b=float(eb),
                exit_price_a=float(pa),
                exit_price_b=float(pb),
                fee=float(fee),
                pnl=float(pnl),
                pnl_pct=float(pnl_pct),
                capital_per_leg=float(cfg.capital_per_trade),
                bars_held=int(hold_bars if hold_bars>=0 else 0)
            )
            closed_records.append(rec)
            in_pos = False
            bars_consumed = t
            return closed_records, None, bars_consumed

        bars_consumed = t

    if in_pos:
        new_state = {
            "pair_label": state["pair_label"],
            "side": side,
            "entry_price_a": ea,
            "entry_price_b": eb,
            "entry_time": et,
            "w_a": w_a,
            "w_b": w_b,
            "qty_a": qty_a,
            "qty_b": qty_b,
            "beta": float(beta),
            "entry_bar_index": entry_bar_index
        }
        return closed_records, new_state, bars_consumed

# =========================
# 回测主流程（单持仓）
# =========================
def backtest_single_position_flow(data: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    assert cfg.method in ["Distance","SDR"]
    symbols=list(data.keys())
    closes={s: data[s]["close"].copy() for s in symbols}
    all_index=None
    for s in symbols:
        idx=closes[s].index
        all_index=idx if all_index is None else all_index.union(idx)
    all_index=all_index.sort_values()

    close_aligned = {s: closes[s].reindex(all_index).ffill().bfill() for s in symbols}

    if cfg.sdr_use_btc_as_market:
        btc_key=next((k for k in symbols if k.upper().startswith("BTC")), None)
        if btc_key is None and cfg.debug:
            print("[WARN] BTC not found in symbols; falling back to equal-weight market index.")
        market_series=close_aligned[btc_key] if btc_key else pd.DataFrame(close_aligned).mean(axis=1)
    else:
        market_series=pd.DataFrame(close_aligned).mean(axis=1)

    if len(all_index) <= cfg.form_period:
        print("Insufficient data for formation period.")
        return pd.DataFrame()

    all_trade_records: List[Dict] = []
    pos_state: Optional[Dict] = None
    method = cfg.method

    cached_pairs: List[Tuple[str,str,float]] = []
    last_recompute_bar = -1
    cooldown_until_bar = -1

    def recompute_candidates(upto_bar: int):
        start = upto_bar - cfg.form_period
        if start < 0:
            return []
        form_index = all_index[start:upto_bar]
        form_prices = {s: close_aligned[s].loc[form_index] for s in symbols}
        form_prices = {k:v.dropna() for k,v in form_prices.items() if v.dropna().shape[0] >= max(cfg.min_form_bars, cfg.bb_window)}
        if len(form_prices) < 2:
            return []
        base_pairs = make_candidate_pairs(
            form_prices,
            method,
            market_series.loc[form_index],
            cfg
        )
        return base_pairs

    i = cfg.form_period
    while i < len(all_index):
        in_cooldown = (i <= cooldown_until_bar)

        if pos_state is None and (not in_cooldown) and (last_recompute_bar < 0 or (i - last_recompute_bar) >= cfg.recompute_candidates_every):
            cached_pairs = recompute_candidates(i)
            last_recompute_bar = i
            if cfg.debug:
                print(f"[{all_index[i]}] Recomputed candidates: {len(cached_pairs)}")

        if pos_state is None and (not in_cooldown):
            for (a,b,beta) in cached_pairs:
                A = close_aligned[a]; B = close_aligned[b]
                label = f"{a}-{b}"
                state = try_open_position(A, B, beta, cfg, i, label)
                if state is not None:
                    pos_state = state
                    if cfg.debug:
                        print(f"[{all_index[i]}] Open {label} {state['side']} at A={state['entry_price_a']:.6f}, B={state['entry_price_b']:.6f}")
                    break
            i += 1
            continue

        if pos_state is not None:
            a, b = pos_state["pair_label"].split("-")
            beta = float(pos_state["beta"])
            A = close_aligned[a]; B = close_aligned[b]
            closed_records, new_state, consumed = step_manage_position(
                A, B, beta, cfg, pos_state, start_bar=i, end_bar=i,
                method_name=method,
                window_start=all_index[max(0, i - cfg.form_period)],
                window_end=all_index[i]
            )
            for rec in closed_records:
                all_trade_records.append(asdict(rec))
                if cfg.debug:
                    print(f"[{rec.exit_time}] Close {rec.pair} {rec.side} pnl={rec.pnl_pct:.4f}% (bars={rec.bars_held})")
                cooldown_until_bar = max(cooldown_until_bar, i + cfg.cooldown_bars)
            pos_state = new_state
            i += 1
            continue

        if pos_state is None and in_cooldown:
            i += 1
            continue

    if pos_state is not None and cfg.debug:
        print(f"End with open position: {pos_state['pair_label']} since {pos_state['entry_time']}")

    trades_df = pd.DataFrame(all_trade_records)
    if trades_df.empty:
        print("[INFO] No completed trades.")
        return trades_df

    save_base = cfg.save_dir or "out"
    os.makedirs(save_base, exist_ok=True)
    ts_str = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(save_base, f"trades_seq_{cfg.method}_{ts_str}.csv")
    json_path = os.path.join(save_base, f"trades_seq_{cfg.method}_{ts_str}.json")

    sort_cols = [c for c in ["window_start", "pair", "entry_time"] if c in trades_df.columns]
    if sort_cols:
        trades_df = trades_df.sort_values(sort_cols).reset_index(drop=True)
    trades_df.to_csv(csv_path, index=False)

    def _json_default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, pd.Timestamp):
            if pd.isna(o):
                return None
            return o.isoformat()
        try:
            if o is pd.NaT:
                return None
        except Exception:
            pass
        return str(o)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(trades_df.to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=_json_default)

    print(f"[{cfg.method}] Trades saved to: {csv_path} and {json_path}")

    trades_df["cum_pnl_pct"] = trades_df["pnl_pct"].cumsum()
    trades_df["ret"] = 1.0 + trades_df["pnl_pct"]/100.0
    trades_df["equity"] = trades_df["ret"].cumprod()
    trades_df["peak"] = trades_df["equity"].cummax()
    trades_df["dd"] = trades_df["equity"]/trades_df["peak"] - 1.0
    max_dd = trades_df["dd"].min() if not trades_df.empty else 0.0

    avg_bars = trades_df["bars_held"].mean() if "bars_held" in trades_df.columns else np.nan
    med_bars = trades_df["bars_held"].median() if "bars_held" in trades_df.columns else np.nan

    wins = (trades_df["pnl_pct"] > 0).sum()
    total_trades = len(trades_df)
    global_winrate = wins / total_trades if total_trades > 0 else float("nan")
    total_return_pct = trades_df["pnl_pct"].sum() if total_trades > 0 else 0.0
    avg_return_pct = trades_df["pnl_pct"].mean() if total_trades > 0 else 0.0
    std_return_pct = trades_df["pnl_pct"].std(ddof=1) if total_trades > 1 else 0.0

    print("\n=== Global trade-level stats (Completed trades only) ===")
    print(f"Method: {cfg.method}")
    print(f"Completed trades: {total_trades}")
    print(f"Winrate: {global_winrate:.2%}" if total_trades > 0 else "Winrate: N/A")
    print(f"Total return (sum of pnl_pct): {total_return_pct:.2f}%")
    print(f"Avg return per trade: {avg_return_pct:.4f}%  |  Std: {std_return_pct:.4f}%")
    print(f"Equity (final): {trades_df['equity'].iloc[-1]:.4f}  |  Max Drawdown: {max_dd:.2%}")
    if np.isfinite(avg_bars):
        print(f"Bars held: avg={avg_bars:.2f}, median={med_bars:.0f}")

    return trades_df

# =========================
# 运行入口
# =========================
if __name__ == "__main__":
    data_dir="/home/houyi/crypto/download_data/robot/data/bybit/futures/"
    print(f"Loading data from: {data_dir}")
    data=load_freqtrade_dir(data_dir, max_symbols=33)
    cfg=Config(
        freq="5min",
        form_period=200,
        select_pairs_per_window=1,
        capital_per_trade=1000.0,
        fee_rate=0.0003,
        use_log_price=False,
        sdr_use_btc_as_market=True,
        bb_window=30,
        bb_k=3,
        std_clip=1e-6,
        z_exit_to_sma=True,
        z_stop=6.0,
        vol_weight=True,
        debug=True,
        max_candidates_per_method=20,
        save_dir="out",
        method="SDR",
        recompute_candidates_every=20,

        use_next_bar_price=True,
        slippage_bps=1.0,
        max_holding_bars=100,
        cooldown_bars=1,

        min_corr=0.2,
        max_adf_p=0.2,
        use_adf_filter=True,
        min_form_bars=60,

        beta_neutral_qty=True,

        # 新增：利润门槛与z缓冲
        min_take_profit_pct=0.2,        # 例如要求至少+0.2% 净利润才因跨中轨平仓
        enforce_profit_on_cross=True,
        takeprofit_exit_buffer_z=0.2    # 可设为0关闭缓冲
    )
    trades_df = backtest_single_position_flow(data, cfg)
    if not trades_df.empty:
        print(trades_df.tail(5))
