from typing import Dict
from .data import load_prices_from_csv
from .methods.distance import DistanceSelector
from .methods.correlation import CorrelationSelector
from .methods.cointegration import CointegrationSelector
from .methods.sdr import SDRSelector
from .methods.ga import GASelector
from .methods.nsga2 import NSGA2Selector

def demo(csv_path: str):
    prices = load_prices_from_csv(csv_path)

    # 1) 不同方法使用各自构造参数（或默认）
    distance = DistanceSelector(select_pairs_per_window=5, max_candidates=300, use_log_price=False)
    corr = CorrelationSelector(select_pairs_per_window=5, min_corr=0.1)
    coint = CointegrationSelector(select_pairs_per_window=5, adf_crit=-3.4, use_log_price=True)
    sdr = SDRSelector(select_pairs_per_window=5, market_mode="mean")
    ga = GASelector(pairs_per_chrom=5, pop=80, gen=60, candidate_source="distance", use_log_price=False)
    nsga = NSGA2Selector(pairs_per_chrom=5, pop=80, gen=60, use_log_price=False)

    methods = {
        "Distance": distance,
        "Correlation": corr,
        "Cointegration": coint,
        "SDR": sdr,
        "GA": ga,
        "NSGA-II": nsga,
    }

    results: Dict[str, list] = {}
    for name, method in methods.items():
        pairs = method.select_pairs(prices)
        results[name] = pairs

    for k, v in results.items():
        print(f"[{k}]")
        for a, b, beta in v:
            print(f"  ({a}, {b}) beta={beta:.3f}")
        print()

if __name__ == "__main__":
    demo("123.csv")