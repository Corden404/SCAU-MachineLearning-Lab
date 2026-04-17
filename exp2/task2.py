import pandas as pd
import numpy as np

# 加载乳腺癌数据集
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")

# 数据集中缺失值用 '?' 表示，将其统一替换为 NumPy 的标准缺失值标识符 np.nan
# 这样做是为了方便后续使用 pandas 内置的缺失值处理函数（如 dropna 或 fillna）
df.replace('?', np.nan, inplace=True)

# 使用 info() 查看数据集的摘要信息
# 重点关注每一列的 Non-Null Count（非空值数量）以及 Dtype（数据类型）
# 观察发现某些数值型列可能因为 '?' 的存在被识别为了 object 类型
print(df.info())
