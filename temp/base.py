
from typing import Protocol, Dict, List, Tuple
import pandas as pd

Pair = Tuple[str, str, float]

class PairSelectionMethod(Protocol):
    def select_pairs(self, prices: Dict[str, pd.Series]) -> List[Pair]:
        """
        返回 [(symbol_a, symbol_b, beta)]。方法内部自行使用其配置。
        """
        ...