from pprint import pprint

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from task5 import load_xgboost_data


def evaluate_model(model, X, y, dataset_name: str) -> None:
    # 随机搜索得到的最优模型仍用混淆矩阵和分类报告评价。
    y_pred = model.predict(X)

    print(f"\n=== {dataset_name}混淆矩阵 ===")
    print(confusion_matrix(y, y_pred))

    print(f"\n=== {dataset_name}分类报告 ===")
    print(classification_report(y, y_pred, digits=4))


def main() -> None:
    (X_train, X_test, y_train, y_test), _ = load_xgboost_data(
        test_size=0.2,
        random_state=42,
    )

    param_distributions = {
        # XGBoost 参数较多，全量网格会很大，因此用随机搜索抽样更可控。
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 200, 300, 500],
        "max_depth": [1, 2, 3, 4, 5, 6],
        "min_child_weight": [0.5, 1, 2, 3, 5, 7],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.3, 0.5, 1.0],
        "reg_alpha": [0, 0.01, 0.1, 0.5, 1.0],
        "reg_lambda": [0.5, 1, 2, 5, 10],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        # 内层模型固定单线程，外层 RandomizedSearchCV 再并行，避免线程过度竞争。
        n_jobs=1,
    )
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        # 从大范围中随机抽 80 组，兼顾搜索覆盖面和运行时间。
        n_iter=80,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=42,
        return_train_score=True,
    )
    random_search.fit(X_train, y_train)

    print("=== XGBoost 随机搜索参数范围 ===")
    pprint(param_distributions)
    print("\n随机抽样次数: 80")

    print("\n=== 最优参数 ===")
    pprint(random_search.best_params_)
    print(f"最优交叉验证 F1: {random_search.best_score_:.4f}")

    results = pd.DataFrame(random_search.cv_results_)
    # 输出前五组结果，检查最优参数是否只是偶然领先。
    top_results = results.sort_values("rank_test_score").head(5)
    print("\n=== 交叉验证 F1 前五组参数 ===")
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

    best_model = random_search.best_estimator_
    evaluate_model(best_model, X_train, y_train, "训练集")
    evaluate_model(best_model, X_test, y_test, "测试集")


if __name__ == "__main__":
    main()
