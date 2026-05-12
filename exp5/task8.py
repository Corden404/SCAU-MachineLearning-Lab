import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

df = pd.read_csv('iris_dataset.csv')
X = df.drop(['target', 'species'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.random.seed(42)
X_train_missing = X_train.copy()
for col in X_train_missing.columns:
    missing_indices = np.random.choice(X_train_missing.index, size=20, replace=False)
    X_train_missing.loc[missing_indices, col] = np.nan

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

# 1. 不做归一化的 KNN 填充法：直接在原始带有缺失值的数据上计算距离并填充
imputer_knn_unnormalized = KNNImputer(n_neighbors=5)
X_train_knn_unnormalized = pd.DataFrame(imputer_knn_unnormalized.fit_transform(X_train_missing), columns=X_train_missing.columns)
evaluate_method(X_train_knn_unnormalized, y_train, "KNN填充法 (未归一化)", "task8_knn_unnormalized_visualization.png")

# 2. 做归一化的 KNN 填充法：KNN 依赖于距离计算，消除量纲影响通常会使距离更合理
scaler = MinMaxScaler()
# 第一步：先将包含缺失值的数据进行归一化 (MinMaxScaler 会忽略 NaN 进行计算)
X_train_missing_scaled = pd.DataFrame(scaler.fit_transform(X_train_missing), columns=X_train_missing.columns)

# 第二步：在归一化后的数据上进行 KNN 填充，此时距离度量不受量纲差异干扰
imputer_knn_normalized = KNNImputer(n_neighbors=5)
X_train_knn_scaled = pd.DataFrame(imputer_knn_normalized.fit_transform(X_train_missing_scaled), columns=X_train_missing.columns)

# 第三步：填充完毕后，将数据反归一化回原来的量纲
# (虽然决策树算法本身对特征量纲不敏感，但为了保持输入数据格式一致性，我们将其反转回来)
X_train_knn_normalized = pd.DataFrame(scaler.inverse_transform(X_train_knn_scaled), columns=X_train_missing.columns)
evaluate_method(X_train_knn_normalized, y_train, "KNN填充法 (归一化后)", "task8_knn_normalized_visualization.png")

plt.show()
