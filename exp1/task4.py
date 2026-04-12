import pandas as pd

def main():
    # 读取数据
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'HousingData.csv')
    df = pd.read_csv(file_path)
    
    # 4. 使用 corr 函数计算不同数据序列之间的相关系数
    print("===== 数据集不同序列之间的相关系数 =====")
    print(df.corr())

if __name__ == "__main__":
    main()
