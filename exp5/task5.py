import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer, KNNImputer

df = pd.read_csv('iris_dataset.csv')
X = df.drop(['target', 'species'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模拟数据缺失：在训练集中为每个特征随机制造 20 个缺失值
np.random.seed(42)
X_train_missing = X_train.copy()
for col in X_train_missing.columns:
    missing_indices = np.random.choice(X_train_missing.index, size=20, replace=False)
    X_train_missing.loc[missing_indices, col] = np.nan

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义模型评估函数：遍历超参数组合，训练决策树，输出结果并绘制热力图
def evaluate_method(X_tr, y_tr, method_name, filename):
    print(f"=== {method_name} ===")
    max_depths = [2, 5, 10, 15, None]
    min_samples_leafs = [8, 6, 4, 2, 1]
    
    train_acc_matrix = np.zeros((len(max_depths), len(min_samples_leafs)))
    test_acc_matrix = np.zeros((len(max_depths), len(min_samples_leafs)))
    
    for i, depth in enumerate(max_depths):
        for j, leaf in enumerate(min_samples_leafs):
            clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, min_samples_leaf=leaf, random_state=42)
            clf.fit(X_tr, y_tr)
            train_acc = accuracy_score(y_tr, clf.predict(X_tr))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            
            train_acc_matrix[i, j] = train_acc
            test_acc_matrix[i, j] = test_acc
            
            print(f"max_depth:{depth}, min_samples_leaf:{leaf} | 训练集准确率:{train_acc:.4f}, 测试集准确率:{test_acc:.4f}")
    print()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    depth_labels = [str(d) if d is not None else 'None' for d in max_depths]
    
    sns.heatmap(train_acc_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
                xticklabels=min_samples_leafs, yticklabels=depth_labels, ax=axes[0])
    axes[0].set_title(f'训练集准确率热力图 ({method_name})')
    axes[0].set_xlabel('min_samples_leaf')
    axes[0].set_ylabel('max_depth')
    
    sns.heatmap(test_acc_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
                xticklabels=min_samples_leafs, yticklabels=depth_labels, ax=axes[1])
    axes[1].set_title(f'测试集准确率热力图 ({method_name})')
    axes[1].set_xlabel('min_samples_leaf')
    axes[1].set_ylabel('max_depth')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

# 1. 缺失值丢弃法：直接删除包含缺失值的样本行
X_train_drop = X_train_missing.dropna()
# 同步截取标签中对应的非缺失行
y_train_drop = y_train.loc[X_train_drop.index]
evaluate_method(X_train_drop, y_train_drop, "缺失值丢弃法", "task5_drop_visualization.png")

# 2. 均值填充法：使用特征所在列的平均值来填补缺失值
imputer_mean = SimpleImputer(strategy='mean')
X_train_mean = pd.DataFrame(imputer_mean.fit_transform(X_train_missing), columns=X_train_missing.columns)
evaluate_method(X_train_mean, y_train, "均值填充法", "task5_mean_visualization.png")

# 3. KNN填充法：利用 K 近邻算法，根据距离最近的 5 个邻居进行插值填补
imputer_knn = KNNImputer(n_neighbors=5)
X_train_knn = pd.DataFrame(imputer_knn.fit_transform(X_train_missing), columns=X_train_missing.columns)
evaluate_method(X_train_knn, y_train, "KNN填充法", "task5_knn_visualization.png")

plt.show()
