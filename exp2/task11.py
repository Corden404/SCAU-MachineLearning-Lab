import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score, precision_score, roc_curve, auc, precision_recall_curve, average_precision_score

# ================= 1. 准备数据 =================
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# 确保全部转为浮点型以防转换过程中数据类型不一致
X = df.drop(columns=['Sample code number', 'Class']).astype(float)
# 阴性为0(良性)，阳性为1(恶性)
y = df['Class'].map({2: 0, 4: 1}).values

kf = KFold(n_splits=10, shuffle=True, random_state=2005)

# ================= 2. 构建模型 =================

# --- Model 1: Baseline 模型 (无多项式特征升维, 逻辑回归默认配置) ---
pipe_base = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=2005, max_iter=1000))
])

# 获取每折中的验证集预测结果，这可以用来统一计算交叉验证下的总体验出概率
probas_base = cross_val_predict(pipe_base, X, y, cv=kf, method='predict_proba')[:, 1]
y_pred_base = (probas_base >= 0.5).astype(int)

# --- Model 2: 特征升维(二维) + L1 正则化 ---
# PolynomialFeatures(degree=2) 会将原始的各个属性做二次组合
# penalty='l1', solver='liblinear' 是使用 L1 正则化的标准组合，它能实现特征稀疏化
pipe_poly_l1 = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(penalty='l1', solver='liblinear', random_state=2005, max_iter=2000))
])

probas_poly = cross_val_predict(pipe_poly_l1, X, y, cv=kf, method='predict_proba')[:, 1]
y_pred_poly = (probas_poly >= 0.5).astype(int)


# ================= 3. 计算各个指标 =================
metrics = {}
for name, y_pred, probas in [("Baseline(原模型)", y_pred_base, probas_base), 
                             ("Poly+L1(特征升维+L1)", y_pred_poly, probas_poly)]:
    
    fpr, tpr, _ = roc_curve(y, probas)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y, probas)
    pr_auc = average_precision_score(y, probas)
    
    metrics[name] = {
        'Recall': recall_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc,
        'fpr': fpr,
        'tpr': tpr,
        'pr_p': precision,
        'pr_r': recall
    }

# ================= 4. 控制台输出对比结果 =================
print("\n" + "="*45)
print("=== Baseline vs 特征升维+L1正则化 对比 ===")
print("="*45)
for name, m in metrics.items():
    print(f"[{name}]")
    print(f"  Recall(召回率)   : {m['Recall']:.4f}")
    print(f"  Precision(精确率): {m['Precision']:.4f}")
    print(f"  ROC AUC         : {m['ROC AUC']:.4f}")
    print(f"  PR AUC (AP)     : {m['PR AUC']:.4f}")
    print("-" * 45)

# 额外训练一次全部数据来看看特征数
pipe_poly_l1.fit(X, y)
coef_ = pipe_poly_l1.named_steps['lr'].coef_[0]
print("[特征维度变化说明]")
print(f"原始特征数为: {X.shape[1]}")
print(f"升至二次特征后总数为: {len(coef_)}")
print(f"受到 L1 正则化(稀疏化)影响, 权重不为 0 (被实际用到) 的特征数量为: {np.sum(coef_ != 0)}")


# ================= 5. 可视化 ROC 和 PR 曲线 =================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体以支持中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示异常的问题

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ---- (1) 绘制 ROC 曲线 ----
ax_roc = axes[0]
for name, m in metrics.items():
    ax_roc.plot(m['fpr'], m['tpr'], label=f"{name} (AUC = {m['ROC AUC']:.4f})")
ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.5)
ax_roc.set_title('ROC 曲线对比 (Receiver Operating Characteristic)', fontsize=14)
ax_roc.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax_roc.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=12)
ax_roc.legend(loc='lower right', fontsize=11)
ax_roc.grid(True, linestyle='--', alpha=0.6)

# ---- (2) 绘制 PR 曲线 ----
ax_pr = axes[1]
for name, m in metrics.items():
    # 注意: PR曲线绘制时 x轴通常是 Recall, y轴是 Precision
    ax_pr.plot(m['pr_r'], m['pr_p'], label=f"{name} (AP = {m['PR AUC']:.4f})")
ax_pr.set_title('PR 曲线对比 (Precision-Recall)', fontsize=14)
ax_pr.set_xlabel('Recall (召回率)', fontsize=12)
ax_pr.set_ylabel('Precision (精确率)', fontsize=12)
ax_pr.legend(loc='lower left', fontsize=11)
ax_pr.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('exp2/task11_curve_comparison.png', dpi=300)
print("\nROC与PR对比图已保存为: exp2/task11_curve_comparison.png")
plt.show()
