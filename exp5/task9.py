import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('iris_dataset.csv')
X = df.drop(['target', 'species'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模拟缺失值
np.random.seed(42)
X_train_missing = X_train.copy()
for col in X_train_missing.columns:
    missing_indices = np.random.choice(X_train_missing.index, size=20, replace=False)
    X_train_missing.loc[missing_indices, col] = np.nan

# 缺失值标记法 (填充为-999)
X_train_indicator = X_train_missing.fillna(-999)
X_test_indicator = X_test.copy()

max_depths = [2, 5, 10, 15, None]
min_samples_leafs = [8, 6, 4, 2, 1]

train_acc_matrix = np.zeros((len(max_depths), len(min_samples_leafs)))
test_acc_matrix = np.zeros((len(max_depths), len(min_samples_leafs)))

for i, depth in enumerate(max_depths):
    for j, leaf in enumerate(min_samples_leafs):
        clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, min_samples_leaf=leaf, random_state=42)
        clf.fit(X_train_indicator, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train_indicator))
        test_acc = accuracy_score(y_test, clf.predict(X_test_indicator))
        
        train_acc_matrix[i, j] = train_acc
        test_acc_matrix[i, j] = test_acc
        
        print(f"max_depth:{depth}, min_samples_leaf:{leaf} | 训练集准确率:{train_acc:.4f}, 测试集准确率:{test_acc:.4f}")

# 可视化结果
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
depth_labels = [str(d) if d is not None else 'None' for d in max_depths]

sns.heatmap(train_acc_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=min_samples_leafs, yticklabels=depth_labels, ax=axes[0])
axes[0].set_title('训练集准确率热力图(缺失值标记法)')
axes[0].set_xlabel('min_samples_leaf')
axes[0].set_ylabel('max_depth')

sns.heatmap(test_acc_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=min_samples_leafs, yticklabels=depth_labels, ax=axes[1])
axes[1].set_title('测试集准确率热力图(缺失值标记法)')
axes[1].set_xlabel('min_samples_leaf')
axes[1].set_ylabel('max_depth')

plt.tight_layout()
plt.savefig('task9_visualization.png', dpi=300)
plt.show()
