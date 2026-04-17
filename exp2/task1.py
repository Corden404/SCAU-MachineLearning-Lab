import pandas as pd

# 读取数据集
df = pd.read_csv("exp2/breast-cancer-wisconsin.csv")

# 使用 head() 方法预览数据集的前 5 行内容
print(df.head())
