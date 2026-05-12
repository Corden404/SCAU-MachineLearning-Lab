import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris_dataset.csv')
X = df.drop(['target', 'species'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 固定随机种子，保证每次生成的缺失值位置一致
np.random.seed(42)
# 复制一份训练集用于人为构造缺失值
X_train_missing = X_train.copy()
# 遍历每一个特征列
for col in X_train_missing.columns:
    # 随机选择 20 个索引作为缺失值的位置
    missing_indices = np.random.choice(X_train_missing.index, size=20, replace=False)
    # 将选中的位置置为 NaN
    X_train_missing.loc[missing_indices, col] = np.nan

print("模拟缺失值后，训练集各特征缺失值统计：")
print(X_train_missing.isnull().sum())
print("\n包含缺失值的训练集前5行：")
print(X_train_missing.head())
