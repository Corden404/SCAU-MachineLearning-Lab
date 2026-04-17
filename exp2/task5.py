import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

# 1. 数据加载与预处理
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df.dropna(inplace=True)
df = df.astype(int)

# 2. 特征与标签提取及标准化
X = df.drop(columns=['Sample code number', 'Class'])
y = df['Class'].map({2: 0, 4: 1})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分数据集
# 80% 用于训练，20% 用于测试；设置 random_state 保证实验可重复
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2005)

# 4. 训练普通逻辑回归模型
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
correct = (y_pred == y_test).sum()

print("LogisticRegression:")
print(f"Sample test count: {len(y_test)}")
print(f"Correct predictions: {correct}")
print(f"Weights (9 features): {lr.coef_[0]}")

# 5. 训练带有交叉验证的逻辑回归模型
# LogisticRegressionCV 会自动通过交叉验证选择最佳正则化强度
lrcv = LogisticRegressionCV()
lrcv.fit(X_train, y_train)
y_pred_cv = lrcv.predict(X_test)
correct_cv = (y_pred_cv == y_test).sum()

print("\nLogisticRegressionCV:")
print(f"Sample test count: {len(y_test)}")
print(f"Correct predictions: {correct_cv}")
print(f"Weights (9 features): {lrcv.coef_[0]}")
