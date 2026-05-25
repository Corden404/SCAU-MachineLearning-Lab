from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ZERO_AS_MISSING_COLUMNS = [
    # 这些医学指标取 0 不符合实际含义，因此按异常缺失值处理。
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_data() -> pd.DataFrame:
    data_path = Path(__file__).with_name("diabetes.csv")
    return pd.read_csv(data_path)


def replace_zero_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    # Pregnancies 和 Outcome 中的 0 有业务含义，只替换医学检测指标里的 0。
    processed[ZERO_AS_MISSING_COLUMNS] = processed[ZERO_AS_MISSING_COLUMNS].replace(
        0, np.nan
    )
    return processed


def knn_impute_after_scaling(
    df: pd.DataFrame, n_neighbors: int = 5
) -> pd.DataFrame:
    features = df.drop(columns="Outcome")
    target = df["Outcome"].astype(int)

    # KNNImputer 基于距离寻找相似样本，先标准化可避免量纲大的特征主导距离。
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # 在标准化空间完成 KNN 缺失值填充，n_neighbors=5 与实验说明保持一致。
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_scaled_features = imputer.fit_transform(scaled_features)

    # 填充完成后还原到原始量纲，便于后续输出和模型解释。
    imputed_features = scaler.inverse_transform(imputed_scaled_features)
    imputed_df = pd.DataFrame(imputed_features, columns=features.columns, index=df.index)
    imputed_df["Outcome"] = target
    return imputed_df


def build_train_test_data(
    test_size: float = 0.2, random_state: int = 42, n_neighbors: int = 5
):
    raw_df = load_data()
    nan_df = replace_zero_with_nan(raw_df)
    imputed_df = knn_impute_after_scaling(nan_df, n_neighbors=n_neighbors)

    # Outcome 是标签列，其余列作为模型特征。
    X = imputed_df.drop(columns="Outcome")
    y = imputed_df["Outcome"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        # 分层抽样保持训练集和测试集中正负样本比例接近，减少小数据集划分偏差。
        stratify=y,
    )


def main() -> None:
    raw_df = load_data()

    print("=== 原始数据中 0 值数量 ===")
    print(raw_df.eq(0).sum())

    nan_df = replace_zero_with_nan(raw_df)
    print("\n=== 将不合理的 0 替换为 NaN 后的缺失值数量 ===")
    print(nan_df.isna().sum())

    imputed_df = knn_impute_after_scaling(nan_df, n_neighbors=5)
    print("\n=== KNN 填充后的缺失值数量 ===")
    print(imputed_df.isna().sum())

    print("\n=== KNN 填充后的前五行数据 ===")
    print(imputed_df.head())

    X_train, X_test, y_train, y_test = build_train_test_data(
        test_size=0.2, random_state=42, n_neighbors=5
    )
    print("\n=== 训练集/测试集划分结果 ===")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    print("\n=== 训练集标签分布 ===")
    print(y_train.value_counts().sort_index())
    print("\n=== 测试集标签分布 ===")
    print(y_test.value_counts().sort_index())


if __name__ == "__main__":
    main()
