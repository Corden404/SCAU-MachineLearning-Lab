import pandas as pd

def main():
    # 读取数据
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'HousingData.csv')
    df = pd.read_csv(file_path)
    
    # 3. 使用 describe 函数返回数据的统计变量
    print("===== 数据的统计变量 =====")
    print(df.describe())

if __name__ == "__main__":
    main()
