import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei'] 
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('iris_dataset.csv')
X = df.drop(['target', 'species'], axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depths = [2, 5, 10, 15, None]
min_samples_leafs = [8, 6, 4, 2, 1]

# 遍历参数寻找在测试集上表现最好的 Entropy 模型
best_entropy_acc = 0
best_entropy_clf = None
best_entropy_params = {}
for depth in max_depths:
    for leaf in min_samples_leafs:
        # 训练当前参数组合的 Entropy 决策树
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, min_samples_leaf=leaf, random_state=42)
        clf.fit(X_train, y_train)
        # 计算测试集准确率
        acc = accuracy_score(y_test, clf.predict(X_test))
        # 记录准确率最高的模型及其参数
        if acc > best_entropy_acc:
            best_entropy_acc = acc
            best_entropy_clf = clf
            best_entropy_params = {'max_depth': depth, 'min_samples_leaf': leaf}

# 遍历参数寻找在测试集上表现最好的 Gini 模型
best_gini_acc = 0
best_gini_clf = None
best_gini_params = {}
for depth in max_depths:
    for leaf in min_samples_leafs:
        # 训练当前参数组合的 Gini 决策树
        clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, min_samples_leaf=leaf, random_state=42)
        clf.fit(X_train, y_train)
        # 计算测试集准确率
        acc = accuracy_score(y_test, clf.predict(X_test))
        # 记录准确率最高的模型及其参数
        if acc > best_gini_acc:
            best_gini_acc = acc
            best_gini_clf = clf
            best_gini_params = {'max_depth': depth, 'min_samples_leaf': leaf}

# 绘制并保存最佳信息熵决策树的结构图
plt.figure(figsize=(12, 8))
plot_tree(best_entropy_clf, feature_names=X.columns.tolist(), class_names=df['species'].unique().tolist(), filled=True)
plt.title(f"最佳信息熵模型 (深度:{best_entropy_params['max_depth']}, 叶节点最小样本:{best_entropy_params['min_samples_leaf']}, 准确率:{best_entropy_acc:.4f})")
plt.savefig('entropy_tree.png')
plt.close()
print("已生成并保存信息熵最佳模型可视化图：entropy_tree.png")

# 绘制并保存最佳基尼系数决策树的结构图
plt.figure(figsize=(12, 8))
plot_tree(best_gini_clf, feature_names=X.columns.tolist(), class_names=df['species'].unique().tolist(), filled=True)
plt.title(f"最佳Gini指数模型 (深度:{best_gini_params['max_depth']}, 叶节点最小样本:{best_gini_params['min_samples_leaf']}, 准确率:{best_gini_acc:.4f})")
plt.savefig('gini_tree.png')
plt.close()
print("已生成并保存Gini指数最佳模型可视化图：gini_tree.png")
