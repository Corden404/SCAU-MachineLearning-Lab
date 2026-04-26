import pandas as pd
import numpy as np

def main():
    # 读取包含手写数字特征和标签的CSV数据集
    data = pd.read_csv('digits.csv')
    
    # 提取特征数据:iloc 是基于位置的索引。[:, :-1] 的意思是选取所有的行（:），以及除了最后一列之外的所有列（:-1）。
    X = data.iloc[:, :-1].values
    # 提取标签数据:.values 将其转换为 Numpy 矩阵供模型使用。这些提取出来的数据就是图片的像素信息
    y = data['target'].values
    
    print("----- Digits 数据集基本信息 -----")
    # 获取特征矩阵的行数，即样本的总数量
    print(f"样本数量: {X.shape[0]}")
    # 获取特征矩阵的列数，即每个样本的特征维度（此数据集为 8x8=64 像素）
    print(f"特征维度: {X.shape[1]}")
    # 使用 np.unique 获取所有不重复的标签值，计算其长度即可得到类别数量
    print(f"标签数量: {len(np.unique(y))}")
    print(f"标签类别: {np.unique(y)}")

if __name__ == "__main__":
    main()
