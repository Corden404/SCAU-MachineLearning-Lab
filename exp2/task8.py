import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression

# 1. 准备数据
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
X = df.drop(columns=['Sample code number', 'Class'])
y = df['Class'].map({2: 0, 4: 1})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 设置 10 折交叉验证
# 交叉验证可以减少数据集划分带来的偶然性，更准确地评估模型泛化能力
lr = LogisticRegression(random_state=2005)
kf = KFold(n_splits=10, shuffle=True, random_state=2005)

# 执行交叉验证并获取各折的准确率
scores = cross_val_score(lr, X_scaled, y, cv=kf)

print("10-Fold Cross-Validation Accuracies:")
print(scores)
# 输出平均准确率和标准差，标准差反映了模型表现的稳定性
print(f"Mean Accuracy: {scores.mean():.4f}")
print(f"Standard Deviation: {scores.std():.4f}")
