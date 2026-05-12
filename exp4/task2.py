import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('FE_day.csv')

# 划分训练集和测试集
X = df.drop(columns=['instant', 'cnt'])
y = df['cnt']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 仅使用训练集计算相关系数矩阵
df_train = X_train.copy()
df_train['cnt'] = y_train

corr_matrix = df_train.corr()

plt.figure(figsize=(16, 14))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0,
            annot=False, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.title('特征相关系数矩阵 (基于训练集)', fontsize=16)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150)
corr_matrix.to_csv('correlation_matrix.csv', float_format='%.4f')
print("相关系数矩阵已保存至 correlation_matrix.csv")
print("相关性热力图已保存至 correlation_matrix.png")

print("\n与目标变量 (cnt) 的相关性（由高到低排序）：")
cnt_corr = corr_matrix['cnt'].sort_values(ascending=False).round(4)
print(cnt_corr)

