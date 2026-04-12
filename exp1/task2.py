import pandas as pd

def main():
    # 读取数据
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'HousingData.csv')
    df = pd.read_csv(file_path)
    
    # 2. 使用 info 函数展示数据信息，查看是否有缺失值
    print("===== 数据信息与缺失值检查 =====")
    df.info()

if __name__ == "__main__":
    main()
