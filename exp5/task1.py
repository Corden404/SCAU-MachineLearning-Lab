# 导入 pandas 库，用于数据的加载与处理
import pandas as pd

# 1. 数据加载与预处理
# 从当前目录读取鸢尾花数据集的 CSV 文件
df = pd.read_csv('iris_dataset.csv')

# 统计并输出数据集中每一列的缺失值数量
print("数据集缺失值统计：")
print(df.isnull().sum())

# 打印数据集的前 5 行，以便初步查看数据的基本格式和内容
print("\n前5个样本：")
print(df.head())
