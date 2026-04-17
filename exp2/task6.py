import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# 数据预处理流水线
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric)
df.dropna(inplace=True)
df = df.astype(int)
X = df.drop(columns=['Sample code number', 'Class'])
y = df['Class'].map({2: 0, 4: 1})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2005)

# 模型初始化与训练
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# 模型评估
# 准确率：分类正确的样本占总样本的比例
accuracy = lr.score(X_test, y_test)
print(f"Accuracy (score): {accuracy}")

# 分类报告：包含各分类的精确率 (Precision)、召回率 (Recall) 和 F1 值
report = metrics.classification_report(y_test, y_pred)
print(f"\nClassification Report:\n{report}")
