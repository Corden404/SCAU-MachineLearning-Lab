import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 基础数据预处理
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
X = df.drop(columns=['Sample code number', 'Class'])
y = df['Class'].map({2: 0, 4: 1})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 观察不同正则化强度参数 C 对模型及其权重的影响
# C 越小，正则化越强，权重会被压缩得更小
C_values = [10, 1, 0.1, 0.01, 0.001]
for C in C_values:
    # solver='saga' 适合处理逻辑回归的大规模数据集且支持各正则化设置
    lr = LogisticRegression(C=C, solver='saga', random_state=2005)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    correct = (y_pred == y_test).sum()
    
    print(f"C={C}:")
    print(f"Sample test count: {len(y_test)}")
    print(f"Correct predictions: {correct}")
    # 查看模型学习到的特征权重（系数）
    print(f"Weights (9 features): {lr.coef_[0]}\n")
