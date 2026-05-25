from pathlib import Path

import pandas as pd


def main() -> None:
    # 数据文件与脚本放在同一目录，避免运行位置不同导致相对路径失效。
    data_path = Path(__file__).with_name("diabetes.csv")
    df = pd.read_csv(data_path)

    # 先查看样本规模、字段数量和列名，确认后续建模的特征范围。
    print("=== 数据集结构 ===")
    print(f"样本数: {df.shape[0]}")
    print(f"特征/列数: {df.shape[1]}")
    print(f"列名: {list(df.columns)}")

    # head() 用于快速观察每列数值形态，尤其留意医学指标中的 0 值。
    print("\n=== 前五行数据 ===")
    print(df.head())

    # 原始缺失值统计只会识别 NaN；本数据中的异常 0 需要在步骤 2 继续处理。
    print("\n=== 缺失值统计 ===")
    print(df.isnull().sum())

    # 数据类型会影响后续标准化、KNN 填充和模型输入是否需要额外转换。
    print("\n=== 数据类型 ===")
    print(df.dtypes)

    print("\n=== DataFrame 信息 ===")
    df.info()


if __name__ == "__main__":
    main()
