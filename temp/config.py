from dataclasses import dataclass
from typing import Optional

@dataclass
class GlobalContext:
    seed: int = 42  # 若需要统一随机种子或日志开关，可放这里
    # 你也可以加入全局的交易成本或数据路径等通用信息