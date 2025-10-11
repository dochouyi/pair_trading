import pandas as pd
import numpy as np


data = {"A": [1, 2], "B": [3, 4]}
df = pd.DataFrame(data, index=["row1", "row2"])
print(df)

df_reset = df.reset_index()
print(df_reset)





















