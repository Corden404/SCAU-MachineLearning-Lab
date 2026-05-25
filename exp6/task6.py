from pprint import pprint

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from task5 import load_xgboost_data


def evaluate_model(model, X, y, dataset_name: str) -> None:
    # 与步骤 5 保持同一评估口径，便于观察调参前后的变化。
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

    param_grid = {
        # 学习率、树数量和树深度共同控制模型复杂度与拟合速度。
        "learning_rate": [0.03, 0.1],
        "n_estimators": [100, 300],
        "max_depth": [2, 3, 4],
        # 下面参数控制叶子节点、采样比例和正则化，主要用于抑制过拟合。
        "min_child_weight": [1, 3],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 2],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = XGBClassifier(
        objective="binary:logistic",
        # 显式设置二分类目标和评估指标，避免新版 xgboost 产生默认行为警告。
        eval_metric="logloss",
        random_state=42,
        n_jobs=1,
    )
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        # 继续使用 F1 作为搜索目标，兼顾患病类 precision 和 recall。
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    print("=== XGBoost 网格搜索参数范围 ===")
    pprint(param_grid)

    print("\n=== 最优参数 ===")
    pprint(grid_search.best_params_)
    print(f"最优交叉验证 F1: {grid_search.best_score_:.4f}")

    results = pd.DataFrame(grid_search.cv_results_)
    # 输出前五组参数，便于分析哪些参数组合接近最优。
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

    best_model = grid_search.best_estimator_
    evaluate_model(best_model, X_train, y_train, "训练集")
    evaluate_model(best_model, X_test, y_test, "测试集")


if __name__ == "__main__":
    main()
