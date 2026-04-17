import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, accuracy_score

# 1. 准备数据
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
X = df.drop(columns=['Sample code number', 'Class'])
# 2 为良性(阴性=0), 4 为恶性(阳性=1)
y = df['Class'].map({2: 0, 4: 1}).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. 设置 10 折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=2005)

recall_base_list = []
recall_weight_list = []
recall_thresh_list = []

precision_base_list = []
precision_weight_list = []
precision_thresh_list = []

accuracy_base_list = []
accuracy_weight_list = []
accuracy_thresh_list = []

for train_idx, test_idx in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # ---------- Baseline 模型 ----------
    lr_base = LogisticRegression(random_state=2005)
    lr_base.fit(X_train, y_train)
    y_pred_base = lr_base.predict(X_test)
    recall_base_list.append(recall_score(y_test, y_pred_base))
    precision_base_list.append(precision_score(y_test, y_pred_base))
    accuracy_base_list.append(accuracy_score(y_test, y_pred_base))

    # ---------- 方法 1: 增大对 FN 的惩罚 ----------
    # FN(False Negative) 是指实际为恶性(1)但预测为良性(0)的样本。
    # 我们通过 class_weight 参数调高类别 1(恶性) 的权重，从而增加错分恶性肿瘤模型受到的惩罚。
    lr_weight = LogisticRegression(class_weight={0: 1, 1: 5}, random_state=2005)
    lr_weight.fit(X_train, y_train)
    y_pred_weight = lr_weight.predict(X_test)
    recall_weight_list.append(recall_score(y_test, y_pred_weight))
    precision_weight_list.append(precision_score(y_test, y_pred_weight))
    accuracy_weight_list.append(accuracy_score(y_test, y_pred_weight))

    # ---------- 方法 2: 降低决策阈值 ----------
    # 逻辑回归默认的决策阈值是 0.5。降低该阈值意味着“只要有微小的可能是恶性，就宁可错杀也不放过”。
    # 这里我们还是基于 Baseline 模型，但将决策判断阈值降低（比如 0.2）
    probas = lr_base.predict_proba(X_test)[:, 1] # 取出预测为 1（恶性）的概率
    y_pred_thresh = (probas >= 0.2).astype(int)
    recall_thresh_list.append(recall_score(y_test, y_pred_thresh))
    precision_thresh_list.append(precision_score(y_test, y_pred_thresh))
    accuracy_thresh_list.append(accuracy_score(y_test, y_pred_thresh))

# 3. 打印结果进行对比
print("========== 综合指标提升评估 ==========")
print(f"1. Baseline (默认阈值 0.5, 无额外类别权重)")
print(f"   Recall: {np.mean(recall_base_list):.4f}, Precision: {np.mean(precision_base_list):.4f}, Accuracy: {np.mean(accuracy_base_list):.4f}")
print(f"2. 方法一 (增大对 FN 的惩罚, 调整 class_weight)")
print(f"   Recall: {np.mean(recall_weight_list):.4f}, Precision: {np.mean(precision_weight_list):.4f}, Accuracy: {np.mean(accuracy_weight_list):.4f}")
print(f"3. 方法二 (降低决策阈值至少于 0.5, 本例为 0.2)")
print(f"   Recall: {np.mean(recall_thresh_list):.4f}, Precision: {np.mean(precision_thresh_list):.4f}, Accuracy: {np.mean(accuracy_thresh_list):.4f}")
