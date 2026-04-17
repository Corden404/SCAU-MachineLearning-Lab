import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 读取原始数据集
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")

# 数据清洗：将缺失符号 '?' 替换为 NaN，并转为数值型后删除空值
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric) # 将 object 类型转成数值
df.dropna(inplace=True)      # 删除含有缺失值的样本
df = df.astype(int)          # 统一数据类型为整数

# 准备特征 (X) 和标签 (y)
# X: 剔除 ID 列（Sample code number）和 目标列（Class）
X = df.drop(columns=['Sample code number', 'Class'])
# y: 将原始标签 2 (良性) 映射为 0，4 (恶性) 映射为 1
y = df['Class'].map({2: 0, 4: 1})

# 特征标准化
# 使用 StandardScaler 使数据符合标准正态分布 (均值0, 标准差1)
# 这对逻辑回归等基于梯度下降或距离权重的模型非常重要
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

# 为了提高可读性，将标准化后的数据转换回 DataFrame 格式
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("Features (first 5 rows):")
print(X_scaled_df.head())
print("\nLabels (first 5 rows):")
print(y.head())
