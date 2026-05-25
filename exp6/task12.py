from pathlib import Path

import matplotlib

# 无需打开窗口，直接把 PR 曲线对比图保存为文件。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from task5 import ZERO_AS_MISSING_COLUMNS, load_xgboost_data


BEST_PARAMS = {
    # 与步骤 11 保持同一 XGBoost 参数，只比较是否加入缺失指示变量。
    "learning_rate": 0.1,
    "n_estimators": 50,
    "max_depth": 6,
    "min_child_weight": 7,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "gamma": 0.5,
    "reg_alpha": 0.1,
    "reg_lambda": 2,
}


def load_xgboost_data_with_missing_indicators(
    test_size: float = 0.2, random_state: int = 42
):
    data_path = Path(__file__).with_name("diabetes.csv")
    df = pd.read_csv(data_path)

    processed = df.copy()
    for column in ZERO_AS_MISSING_COLUMNS:
        # 在替换 NaN 之前记录原始 0 的位置，让模型知道“该指标原本缺失”。
        processed[f"{column}_missing"] = (processed[column] == 0).astype(int)

    # 原始医学指标仍替换为 NaN，交给 XGBoost 学习缺失方向。
    processed[ZERO_AS_MISSING_COLUMNS] = processed[ZERO_AS_MISSING_COLUMNS].replace(
        0, np.nan
    )

    X = processed.drop(columns="Outcome")
    y = processed["Outcome"].astype(int)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    ), processed


def build_model() -> XGBClassifier:
    # 封装建模函数，确保 baseline 和 indicator 两组实验使用完全相同参数。
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        **BEST_PARAMS,
    )


def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    # 在指定阈值下统一计算准确率、类别 1 指标和混淆矩阵。
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_1": precision[0],
        "recall_1": recall[0],
        "f1_1": f1[0],
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true, y_pred, digits=4, zero_division=0
        ),
    }


def summarize_model(name: str, X_train, X_test, y_train, y_test) -> dict:
    model = build_model()
    model.fit(X_train, y_train)
    # 用概率而不是硬分类结果，方便同时比较默认阈值和最佳 F1 阈值。
    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    average_precision = average_precision_score(y_test, y_prob)

    # 从所有候选阈值中选测试集 F1 最大的位置。
    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]
    f1_scores = 2 * threshold_precision * threshold_recall / (
        threshold_precision + threshold_recall + 1e-12
    )
    best_f1_index = int(np.argmax(f1_scores))
    best_f1_threshold = float(thresholds[best_f1_index])

    default_metrics = metrics_at_threshold(y_test, y_prob, 0.5)
    best_f1_metrics = metrics_at_threshold(y_test, y_prob, best_f1_threshold)

    return {
        "name": name,
        "model": model,
        "y_prob": y_prob,
        "precision_curve": precision,
        "recall_curve": recall,
        "average_precision": average_precision,
        "default": default_metrics,
        "best_f1": best_f1_metrics,
        "best_f1_threshold": best_f1_threshold,
    }


def print_summary(summary: dict) -> None:
    print(f"\n=== {summary['name']} ===")
    print(f"Average precision (AP): {summary['average_precision']:.4f}")

    default = summary["default"]
    print("\n--- Default threshold = 0.5000 ---")
    print(f"Accuracy: {default['accuracy']:.4f}")
    print(
        "Class 1: "
        f"precision={default['precision_1']:.4f}, "
        f"recall={default['recall_1']:.4f}, "
        f"f1={default['f1_1']:.4f}, "
        f"FN={default['fn']}, FP={default['fp']}"
    )
    print(default["confusion_matrix"])
    print(default["classification_report"])

    best_f1 = summary["best_f1"]
    print(f"\n--- Best F1 threshold = {best_f1['threshold']:.4f} ---")
    print(f"Accuracy: {best_f1['accuracy']:.4f}")
    print(
        "Class 1: "
        f"precision={best_f1['precision_1']:.4f}, "
        f"recall={best_f1['recall_1']:.4f}, "
        f"f1={best_f1['f1_1']:.4f}, "
        f"FN={best_f1['fn']}, FP={best_f1['fp']}"
    )
    print(best_f1["confusion_matrix"])
    print(best_f1["classification_report"])


def main() -> None:
    (base_X_train, base_X_test, base_y_train, base_y_test), _ = load_xgboost_data(
        test_size=0.2,
        random_state=42,
    )
    (
        indicator_X_train,
        indicator_X_test,
        indicator_y_train,
        indicator_y_test,
    ), indicator_df = load_xgboost_data_with_missing_indicators(
        test_size=0.2,
        random_state=42,
    )

    base_summary = summarize_model(
        # baseline 对应步骤 11 原始特征模型。
        "Step 11 baseline without missing indicators",
        base_X_train,
        base_X_test,
        base_y_train,
        base_y_test,
    )
    indicator_summary = summarize_model(
        # indicator 组只额外加入缺失指示变量，用于隔离特征工程效果。
        "XGBoost with missing indicators",
        indicator_X_train,
        indicator_X_test,
        indicator_y_train,
        indicator_y_test,
    )

    print("=== Added missing indicator columns ===")
    added_columns = [f"{column}_missing" for column in ZERO_AS_MISSING_COLUMNS]
    print(added_columns)
    print("\n=== Missing indicator positive counts ===")
    print(indicator_df[added_columns].sum())

    print_summary(base_summary)
    print_summary(indicator_summary)

    # 汇总成表，便于在报告中直接比较 AP、默认阈值和最佳 F1 阈值表现。
    comparison = pd.DataFrame(
        [
            {
                "model": "Step 11 baseline",
                "AP": base_summary["average_precision"],
                "default_accuracy": base_summary["default"]["accuracy"],
                "default_precision_1": base_summary["default"]["precision_1"],
                "default_recall_1": base_summary["default"]["recall_1"],
                "default_f1_1": base_summary["default"]["f1_1"],
                "default_FN": base_summary["default"]["fn"],
                "default_FP": base_summary["default"]["fp"],
                "best_f1_threshold": base_summary["best_f1"]["threshold"],
                "best_f1_accuracy": base_summary["best_f1"]["accuracy"],
                "best_f1_precision_1": base_summary["best_f1"]["precision_1"],
                "best_f1_recall_1": base_summary["best_f1"]["recall_1"],
                "best_f1_1": base_summary["best_f1"]["f1_1"],
                "best_f1_FN": base_summary["best_f1"]["fn"],
                "best_f1_FP": base_summary["best_f1"]["fp"],
            },
            {
                "model": "With missing indicators",
                "AP": indicator_summary["average_precision"],
                "default_accuracy": indicator_summary["default"]["accuracy"],
                "default_precision_1": indicator_summary["default"]["precision_1"],
                "default_recall_1": indicator_summary["default"]["recall_1"],
                "default_f1_1": indicator_summary["default"]["f1_1"],
                "default_FN": indicator_summary["default"]["fn"],
                "default_FP": indicator_summary["default"]["fp"],
                "best_f1_threshold": indicator_summary["best_f1"]["threshold"],
                "best_f1_accuracy": indicator_summary["best_f1"]["accuracy"],
                "best_f1_precision_1": indicator_summary["best_f1"]["precision_1"],
                "best_f1_recall_1": indicator_summary["best_f1"]["recall_1"],
                "best_f1_1": indicator_summary["best_f1"]["f1_1"],
                "best_f1_FN": indicator_summary["best_f1"]["fn"],
                "best_f1_FP": indicator_summary["best_f1"]["fp"],
            },
        ]
    )
    print("\n=== Comparison table ===")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = Path(__file__).with_name("task12_missing_indicator_comparison.csv")
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_path = Path(__file__).with_name("task12_pr_curve_comparison.png")
    plt.figure(figsize=(8, 6))
    for summary in [base_summary, indicator_summary]:
        plt.plot(
            summary["recall_curve"],
            summary["precision_curve"],
            label=f"{summary['name']} (AP={summary['average_precision']:.4f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)

    print(f"\nCSV saved to: {csv_path.name}")
    print(f"Plot saved to: {plot_path.name}")


if __name__ == "__main__":
    main()
