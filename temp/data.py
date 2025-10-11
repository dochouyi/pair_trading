import pandas as pd
from typing import Dict, Tuple

def load_prices_from_csv(path: str) -> Dict[str, pd.Series]:
    df = pd.read_csv(path)
    if df.shape[1] >= 2:
        first_col = df.columns[0]
        if not pd.api.types.is_numeric_dtype(df[first_col]):
            ts = pd.to_datetime(df[first_col], errors='coerce')
            if ts.notna().mean() > 0.7:
                df[first_col] = ts
            df = df.set_index(first_col)
    else:
        raise ValueError("CSV 至少需要两列（索引列 + 一个资产列）")
    df = df.dropna(axis=1, how='all')
    value_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(value_cols) < 2:
        raise ValueError("有效的数值资产列不足 2 列")
    df = df[value_cols].astype(float)
    return {col: df[col].dropna() for col in df.columns}

def intersect_align(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    idx = a.dropna().index.intersection(b.dropna().index)
    return a.loc[idx], b.loc[idx]
