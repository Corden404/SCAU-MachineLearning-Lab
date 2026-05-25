from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ZERO_AS_MISSING_COLUMNS = [
    # XGBoost 可以直接处理 NaN，因此这里不再做 KNN 填充。
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def load_xgboost_data(test_size: float = 0.2, random_state: int = 42):
    data_path = Path(__file__).with_name("diabetes.csv")
    df = pd.read_csv(data_path)

    processed = df.copy()
    # 只把不合理医学 0 转成 NaN，让 XGBoost 在分裂时学习缺失值默认方向。
    processed[ZERO_AS_MISSING_COLUMNS] = processed[ZERO_AS_MISSING_COLUMNS].replace(
        0, np.nan
    )

    # 保留 NaN 直接作为 XGBoost 输入，标签 Outcome 单独取出。
    X = processed.drop(columns="Outcome")
    y = processed["Outcome"].astype(int)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        # 与随机森林实验保持一致，使用分层划分保证可比性。
        stratify=y,
    ), processed


def evaluate_model(model, X, y, dataset_name: str) -> None:
    # 默认阈值下评估 XGBoost，后续步骤再单独研究阈值调整。
    y_pred = model.predict(X)

    print(f"\n=== {dataset_name}混淆矩阵 ===")
    print(confusion_matrix(y, y_pred))

    print(f"\n=== {dataset_name}分类报告 ===")
    print(classification_report(y, y_pred, digits=4))


def main() -> None:
    (X_train, X_test, y_train, y_test), processed = load_xgboost_data(
        test_size=0.2,
        random_state=42,
    )

    print("=== 将不合理的 0 替换为 NaN 后的缺失值数量 ===")
    print(processed.isna().sum())

    # 步骤 5 要求默认 XGBoost，参数先不手动指定。
    model = XGBClassifier()
    model.fit(X_train, y_train)

    print("\n=== XGBoost 模型参数 ===")
    pprint(model.get_params())

    evaluate_model(model, X_train, y_train, "训练集")
    evaluate_model(model, X_test, y_test, "测试集")


if __name__ == "__main__":
    main()
