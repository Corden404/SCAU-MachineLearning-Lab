# 导入数据处理与计算库
import pandas as pd
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
import seaborn as sns
# 导入机器学习模块：数据划分、决策树分类器及评估指标
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载鸢尾花数据集
df = pd.read_csv('iris_dataset.csv')
# 提取特征变量 X (丢弃 target 和 species 列) 和目标变量 y
X = df.drop(['target', 'species'], axis=1)
y = df['target']

# 将数据集划分为训练集和测试集，测试集占比 20%，设置随机种子保证结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


max_depths = [2, 5, 10, 15, None]
min_samples_leafs = [8, 6, 4, 2, 1]

train_acc_matrix = np.zeros((len(max_depths), len(min_samples_leafs)))
test_acc_matrix = np.zeros((len(max_depths), len(min_samples_leafs)))

# 遍历不同的最大深度(max_depth)和叶节点最小样本数(min_samples_leaf)的组合
for i, depth in enumerate(max_depths):
    for j, leaf in enumerate(min_samples_leafs):
        # 创建决策树分类器，使用信息熵(entropy)作为划分标准
        clf = DecisionTreeClassifier(criterion='entropy', max_depth=depth, min_samples_leaf=leaf, random_state=42)
        # 在训练集上拟合模型
        clf.fit(X_train, y_train)
        
        # 分别对训练集和测试集进行预测
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        cm = confusion_matrix(y_test, y_test_pred)
        
        train_acc_matrix[i, j] = train_acc
        test_acc_matrix[i, j] = test_acc
        
        print(f"max_depth: {depth}, min_samples_leaf: {leaf}")
        print(f"训练集准确率: {train_acc:.4f}, 测试集准确率: {test_acc:.4f}")
        print(f"测试集混淆矩阵:\n{cm}\n")

# ==================== 可视化结果 ====================
# 设置 matplotlib 参数以正常显示中文字符和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建包含 2 个子图的画布 (1行2列)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

depth_labels = [str(d) if d is not None else 'None' for d in max_depths]

sns.heatmap(train_acc_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=min_samples_leafs, yticklabels=depth_labels, ax=axes[0])
axes[0].set_title('训练集准确率热力图')
axes[0].set_xlabel('min_samples_leaf')
axes[0].set_ylabel('max_depth')

sns.heatmap(test_acc_matrix, annot=True, fmt=".4f", cmap="YlGnBu", 
            xticklabels=min_samples_leafs, yticklabels=depth_labels, ax=axes[1])
axes[1].set_title('测试集准确率热力图')
axes[1].set_xlabel('min_samples_leaf')
axes[1].set_ylabel('max_depth')

plt.tight_layout()
plt.savefig('task2_visualization.png', dpi=300)
plt.show()
