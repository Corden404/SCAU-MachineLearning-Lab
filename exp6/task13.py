from pathlib import Path
from pprint import pprint

import matplotlib

# 直接保存图片，避免脚本运行时依赖图形界面。
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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from task5 import load_xgboost_data
from task11 import BEST_PARAMS as STEP11_PARAMS


def median_impute_train_test(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    medians = pd.Series(imputer.statistics_, index=X_train.columns)
    return X_train_imputed, X_test_imputed, medians


def add_degree2_polynomial_features(X: pd.DataFrame) -> pd.DataFrame:
    expanded = X.copy()
    columns = list(X.columns)

    # 手动构造所有二阶项：包含平方项和两两交互项。
    for i, left in enumerate(columns):
        for right in columns[i:]:
            if left == right:
                feature_name = f"{left}__squared"
            else:
                feature_name = f"{left}__mul__{right}"
            expanded[feature_name] = X[left] * X[right]

    return expanded


def build_model(params: dict) -> XGBClassifier:
    # 多项式特征和原始特征都通过同一函数建模，减少参数口径差异。
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
        **params,
    )


def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    # 统一阈值评估逻辑，便于和步骤 11 的原始特征模型对比。
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


def summarize_model(name: str, model, X_test, y_test) -> dict:
    # 用预测概率计算 AP、PR 曲线，并寻找当前模型的最佳 F1 阈值。
    y_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    average_precision = average_precision_score(y_test, y_prob)

    threshold_precision = precision[:-1]
    threshold_recall = recall[:-1]
    f1_scores = 2 * threshold_precision * threshold_recall / (
        threshold_precision + threshold_recall + 1e-12
    )
    best_f1_index = int(np.argmax(f1_scores))
    best_f1_threshold = float(thresholds[best_f1_index])

    return {
        "name": name,
        "precision_curve": precision,
        "recall_curve": recall,
        "average_precision": average_precision,
        "default": metrics_at_threshold(y_test, y_prob, 0.5),
        "best_f1": metrics_at_threshold(y_test, y_prob, best_f1_threshold),
    }


def print_model_summary(summary: dict) -> None:
    print(f"\n=== {summary['name']} ===")
    print(f"Average precision (AP): {summary['average_precision']:.4f}")

    for label in ["default", "best_f1"]:
        metrics = summary[label]
        title = "Default threshold" if label == "default" else "Best F1 threshold"
        print(f"\n--- {title} = {metrics['threshold']:.4f} ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(
            "Class 1: "
            f"precision={metrics['precision_1']:.4f}, "
            f"recall={metrics['recall_1']:.4f}, "
            f"f1={metrics['f1_1']:.4f}, "
            f"FN={metrics['fn']}, FP={metrics['fp']}"
        )
        print(metrics["confusion_matrix"])
        print(metrics["classification_report"])


def main() -> None:
    (X_train, X_test, y_train, y_test), _ = load_xgboost_data(
        test_size=0.2,
        random_state=42,
    )

    X_train_imputed, X_test_imputed, medians = median_impute_train_test(
        X_train, X_test
    )
    X_train_poly = add_degree2_polynomial_features(X_train_imputed)
    X_test_poly = add_degree2_polynomial_features(X_test_imputed)

    # 打印维度变化，说明二阶多项式特征带来的特征数量膨胀。
    print("=== Feature dimensions ===")
    print(f"Original features: {X_train.shape[1]}")
    print(f"Degree-2 polynomial features: {X_train_poly.shape[1]}")
    print("\n=== Median values used for imputation ===")
    print(medians.to_string())

    baseline_model = build_model(STEP11_PARAMS)
    baseline_model.fit(X_train_imputed, y_train)
    # 原始特征模型作为对照组，使用步骤 11 的最优参数。
    baseline_summary = summarize_model(
        "Step 11 params with median-imputed original features",
        baseline_model,
        X_test_imputed,
        y_test,
    )

    param_distributions = {
        # 特征空间变大后重新随机搜索，避免直接沿用原始特征下的参数。
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [1, 2, 3, 4, 5, 6],
        "min_child_weight": [0.5, 1, 2, 3, 5, 7, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8],
        "gamma": [0, 0.1, 0.3, 0.5, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda": [1, 2, 5, 10, 20],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=build_model({}),
        param_distributions=param_distributions,
        # 继续抽样 80 组，在可接受运行时间内寻找更适合多项式特征的参数。
        n_iter=80,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        return_train_score=True,
    )
    random_search.fit(X_train_poly, y_train)

    print("\n=== Degree-2 polynomial RandomizedSearchCV parameter ranges ===")
    pprint(param_distributions)
    print("\nRandom samples: 80")
    print("\n=== Best polynomial model parameters ===")
    pprint(random_search.best_params_)
    print(f"Best cross-validation F1: {random_search.best_score_:.4f}")

    results = pd.DataFrame(random_search.cv_results_)
    # 展示前五名交叉验证结果，辅助判断重搜后的最优参数是否有竞争者。
    top_results = results.sort_values("rank_test_score").head(5)
    print("\n=== Top 5 cross-validation F1 results ===")
    print(
        top_results[
            [
                "rank_test_score",
                "mean_test_score",
                "mean_train_score",
                "param_learning_rate",
                "param_n_estimators",
                "param_max_depth",
                "param_min_child_weight",
                "param_subsample",
                "param_colsample_bytree",
                "param_gamma",
                "param_reg_alpha",
                "param_reg_lambda",
            ]
        ].to_string(index=False)
    )

    polynomial_summary = summarize_model(
        "Median-imputed degree-2 polynomial features with re-search",
        random_search.best_estimator_,
        X_test_poly,
        y_test,
    )

    print_model_summary(baseline_summary)
    print_model_summary(polynomial_summary)

    # 结果表同时列出默认阈值和最佳 F1 阈值，观察多项式特征的真实收益。
    comparison = pd.DataFrame(
        [
            {
                "model": "Median-imputed original features",
                "feature_count": X_train_imputed.shape[1],
                "cv_best_f1": np.nan,
                "AP": baseline_summary["average_precision"],
                "default_accuracy": baseline_summary["default"]["accuracy"],
                "default_precision_1": baseline_summary["default"]["precision_1"],
                "default_recall_1": baseline_summary["default"]["recall_1"],
                "default_f1_1": baseline_summary["default"]["f1_1"],
                "default_FN": baseline_summary["default"]["fn"],
                "default_FP": baseline_summary["default"]["fp"],
                "best_f1_threshold": baseline_summary["best_f1"]["threshold"],
                "best_f1_accuracy": baseline_summary["best_f1"]["accuracy"],
                "best_f1_precision_1": baseline_summary["best_f1"]["precision_1"],
                "best_f1_recall_1": baseline_summary["best_f1"]["recall_1"],
                "best_f1_1": baseline_summary["best_f1"]["f1_1"],
                "best_f1_FN": baseline_summary["best_f1"]["fn"],
                "best_f1_FP": baseline_summary["best_f1"]["fp"],
            },
            {
                "model": "Median-imputed degree-2 polynomial + re-search",
                "feature_count": X_train_poly.shape[1],
                "cv_best_f1": random_search.best_score_,
                "AP": polynomial_summary["average_precision"],
                "default_accuracy": polynomial_summary["default"]["accuracy"],
                "default_precision_1": polynomial_summary["default"]["precision_1"],
                "default_recall_1": polynomial_summary["default"]["recall_1"],
                "default_f1_1": polynomial_summary["default"]["f1_1"],
                "default_FN": polynomial_summary["default"]["fn"],
                "default_FP": polynomial_summary["default"]["fp"],
                "best_f1_threshold": polynomial_summary["best_f1"]["threshold"],
                "best_f1_accuracy": polynomial_summary["best_f1"]["accuracy"],
                "best_f1_precision_1": polynomial_summary["best_f1"]["precision_1"],
                "best_f1_recall_1": polynomial_summary["best_f1"]["recall_1"],
                "best_f1_1": polynomial_summary["best_f1"]["f1_1"],
                "best_f1_FN": polynomial_summary["best_f1"]["fn"],
                "best_f1_FP": polynomial_summary["best_f1"]["fp"],
            },
        ]
    )

    print("\n=== Comparison table ===")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path = Path(__file__).with_name("task13_polynomial_comparison.csv")
    comparison.to_csv(csv_path, index=False, encoding="utf-8-sig")

    plot_path = Path(__file__).with_name("task13_pr_curve_comparison.png")
    # PR 曲线对比能显示排序能力是否提升，而不只看单一阈值下的分类报告。
    plt.figure(figsize=(8, 6))
    for summary in [baseline_summary, polynomial_summary]:
        plt.plot(
            summary["recall_curve"],
            summary["precision_curve"],
            label=f"{summary['name']} (AP={summary['average_precision']:.4f})",
        )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve: Median-Imputed Original vs Degree-2 Polynomial Features")
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)

    print(f"\nCSV saved to: {csv_path.name}")
    print(f"Plot saved to: {plot_path.name}")


if __name__ == "__main__":
    main()
