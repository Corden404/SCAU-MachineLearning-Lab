from pathlib import Path

import matplotlib

# 使用非交互后端，脚本在命令行或无图形界面环境下也能保存图片。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from task2 import build_train_test_data
from task5 import load_xgboost_data


RF_PARAMS = {
    # 使用步骤 4 得到的随机森林最优参数做特征重要性对比。
    "n_estimators": 300,
    "max_depth": 7,
    "min_samples_split": 20,
    "random_state": 42,
}

XGB_PARAMS = {
    # 使用步骤 6 得到的 XGBoost 最优参数，保证对比来自已调参模型。
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 2,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "gamma": 0.1,
    "reg_alpha": 0,
    "reg_lambda": 2,
}


def normalize(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    if total == 0:
        return values
    # 归一化后两种模型的重要性总和都为 1，柱状图更容易横向比较。
    return values / total


def main() -> None:
    rf_X_train, _, rf_y_train, _ = build_train_test_data(
        test_size=0.2,
        random_state=42,
        n_neighbors=5,
    )
    (xgb_X_train, _, xgb_y_train, _), _ = load_xgboost_data(
        test_size=0.2,
        random_state=42,
    )

    rf_model = RandomForestClassifier(**RF_PARAMS)
    rf_model.fit(rf_X_train, rf_y_train)

    # 随机森林使用 KNN 填充后的数据，XGBoost 使用保留 NaN 的数据。
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        **XGB_PARAMS,
    )
    xgb_model.fit(xgb_X_train, xgb_y_train)

    importance_df = pd.DataFrame(
        {
            "Feature": rf_X_train.columns,
            # feature_importances_ 表示模型训练后各特征对分裂收益的相对贡献。
            "RandomForest": normalize(rf_model.feature_importances_),
            "XGBoost": normalize(xgb_model.feature_importances_),
        }
    )
    # 用平均重要性排序，突出两个模型共同认为更关键的指标。
    importance_df["Average"] = importance_df[["RandomForest", "XGBoost"]].mean(axis=1)
    importance_df = importance_df.sort_values("Average", ascending=False)

    print("=== Feature importance comparison ===")
    print(importance_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = Path(__file__).with_name("task7_feature_importance.csv")
    importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 横向柱状图按 Average 从小到大绘制，图中最重要的特征会显示在顶部附近。
    plot_df = importance_df.sort_values("Average", ascending=True)
    y = np.arange(len(plot_df))
    bar_height = 0.36

    plt.figure(figsize=(9, 5.6))
    plt.barh(
        y - bar_height / 2,
        plot_df["RandomForest"],
        height=bar_height,
        label="Random Forest",
    )
    plt.barh(
        y + bar_height / 2,
        plot_df["XGBoost"],
        height=bar_height,
        label="XGBoost",
    )
    plt.yticks(y, plot_df["Feature"])
    plt.xlabel("Normalized feature importance")
    plt.title("Feature Importance Comparison")
    plt.grid(axis="x", alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plot_path = Path(__file__).with_name("task7_feature_importance_comparison.png")
    plt.savefig(plot_path, dpi=160)

    print(f"\nCSV saved to: {csv_path.name}")
    print(f"Plot saved to: {plot_path.name}")


if __name__ == "__main__":
    main()
