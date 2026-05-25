from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from task5 import load_xgboost_data


BEST_PARAMS = {
    # 步骤 10 随机搜索得到的最优 XGBoost 参数。
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


def evaluate_at_threshold(y_true, y_prob, threshold: float, name: str) -> None:
    # predict_proba 输出的是患病概率，手动阈值可以改变 precision-recall 取舍。
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n=== {name} threshold = {threshold:.4f} ===")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification report:")
    print(classification_report(y_true, y_pred, digits=4))


def main() -> None:
    (X_train, X_test, y_train, y_test), _ = load_xgboost_data(
        test_size=0.2,
        random_state=42,
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        **BEST_PARAMS,
    )
    model.fit(X_train, y_train)

    # 取类别 1 的预测概率，用于绘制 PR 曲线和尝试不同分类阈值。
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    average_precision = average_precision_score(y_test, y_prob)

    # precision_recall_curve 的最后一个点没有对应阈值，因此计算 F1 时去掉末尾点。
    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]
    f1_scores = 2 * threshold_precision * threshold_recall / (
        threshold_precision + threshold_recall + 1e-12
    )
    best_f1_index = int(np.argmax(f1_scores))
    best_f1_threshold = float(thresholds[best_f1_index])
    best_f1 = float(f1_scores[best_f1_index])
    best_f1_precision = float(threshold_precision[best_f1_index])
    best_f1_recall = float(threshold_recall[best_f1_index])

    default_threshold = 0.5
    default_pred = (y_prob >= default_threshold).astype(int)
    # 手动统计 TP/FP/FN，用来解释默认阈值下 precision 和 recall 的来源。
    default_tp = np.sum((default_pred == 1) & (y_test.to_numpy() == 1))
    default_fp = np.sum((default_pred == 1) & (y_test.to_numpy() == 0))
    default_fn = np.sum((default_pred == 0) & (y_test.to_numpy() == 1))
    default_precision = default_tp / (default_tp + default_fp)
    default_recall = default_tp / (default_tp + default_fn)

    # 在 recall 至少 0.7 的约束下，再寻找 precision 最高的阈值。
    high_recall_mask = threshold_recall >= 0.7
    if np.any(high_recall_mask):
        candidate_indices = np.where(high_recall_mask)[0]
        best_high_recall_index = candidate_indices[
            np.argmax(threshold_precision[candidate_indices])
        ]
        high_recall_threshold = float(thresholds[best_high_recall_index])
        high_recall_precision = float(threshold_precision[best_high_recall_index])
        high_recall_recall = float(threshold_recall[best_high_recall_index])
    else:
        high_recall_threshold = None
        high_recall_precision = None
        high_recall_recall = None

    output_path = Path(__file__).with_name("task11_precision_recall_curve.png")
    # 图中标出默认阈值和最佳 F1 阈值，直观看到阈值移动后的取舍。
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR curve (AP = {average_precision:.4f})")
    plt.scatter(
        [default_recall],
        [default_precision],
        marker="o",
        s=70,
        label=f"threshold=0.50 (P={default_precision:.3f}, R={default_recall:.3f})",
    )
    plt.scatter(
        [best_f1_recall],
        [best_f1_precision],
        marker="s",
        s=70,
        label=(
            f"best F1 threshold={best_f1_threshold:.3f} "
            f"(P={best_f1_precision:.3f}, R={best_f1_recall:.3f})"
        ),
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for XGBoost")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)

    print("=== Precision-Recall summary ===")
    print(f"Average precision (AP): {average_precision:.4f}")
    print(
        "Default threshold 0.5000: "
        f"precision={default_precision:.4f}, recall={default_recall:.4f}"
    )
    print(
        "Best F1 threshold: "
        f"{best_f1_threshold:.4f}, precision={best_f1_precision:.4f}, "
        f"recall={best_f1_recall:.4f}, f1={best_f1:.4f}"
    )
    if high_recall_threshold is not None:
        print(
            "Best precision with recall >= 0.7000: "
            f"threshold={high_recall_threshold:.4f}, "
            f"precision={high_recall_precision:.4f}, "
            f"recall={high_recall_recall:.4f}"
        )
    print(f"Plot saved to: {output_path.name}")

    evaluate_at_threshold(y_test, y_prob, default_threshold, "Default")
    evaluate_at_threshold(y_test, y_prob, best_f1_threshold, "Best F1")
    if high_recall_threshold is not None:
        evaluate_at_threshold(y_test, y_prob, high_recall_threshold, "Recall >= 0.70")


if __name__ == "__main__":
    main()
