import pandas as pd
import numpy as np

# 读取数据集
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")

# 缺失值预处理：将 '?' 替换为标准缺失值格式 np.nan
df.replace('?', np.nan, inplace=True)

# 删除所有包含缺失值（NaN）的行
# 这种方法适用于缺失样本比例较小且删除后不影响整体分布的情况
df.dropna(inplace=True)

# 打印信息，确认删除操作后的样本总数和数据状态
print(df.info())
