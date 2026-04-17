import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc

# 1. 加载并清洗数据
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
X = df.drop(columns=['Sample code number', 'Class'])
y = df['Class'].map({2: 0, 4: 1})

# 特征标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=2005)

# 要对比的正则化参数列表
C_values = [1, 0.1, 0.01, 0.001]

# 2. 绘制 P-R 曲线 (精确率-召回率曲线)
plt.figure(figsize=(8, 6))
for C in C_values:
    lr = LogisticRegression(C=C, solver='saga', random_state=2005)
    lr.fit(X_train, y_train)
    # 使用 decision_function 获取预测的置信度得分
    y_scores = lr.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    plt.plot(recall, precision, label=f'C={C}')

plt.xlabel('Recall (召回率)')
plt.ylabel('Precision (精确率)')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('exp2/PR_curves.png')
plt.close()

# 3. 绘制 ROC 曲线 (受试者工作特征曲线)
plt.figure(figsize=(8, 6))
for C in C_values:
    lr = LogisticRegression(C=C, solver='saga', random_state=2005)
    lr.fit(X_train, y_train)
    y_scores = lr.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr) # 计算曲线下面积 AUC
    plt.plot(fpr, tpr, label=f'C={C} (AUC = {roc_auc:.2f})')

# 绘制一条从 (0,0) 到 (1,1) 的虚线，代表随机猜测模型
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (假阳性率)')
plt.ylabel('True Positive Rate (真阳性率)')
plt.title('ROC Curve')
plt.legend()
plt.savefig('exp2/ROC_curves.png')
plt.close()

print("Saved exp2/PR_curves.png and exp2/ROC_curves.png")
