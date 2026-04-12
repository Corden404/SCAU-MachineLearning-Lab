import pandas as pd

def main():
    # 1. 加载波士顿房价预测数据集，读取数据
    # 使用相对于脚本文件的路径
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'HousingData.csv')
    df = pd.read_csv(file_path)
    
    # 显示前 5 行
    print("===== 数据集前 5 行 =====")
    print(df.head())

if __name__ == "__main__":
    main()
