from pprint import pprint

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from task2 import build_train_test_data


def evaluate_model(model, X, y, dataset_name: str) -> None:
    # 这里重点看分类报告中类别 1 的 recall 是否提高。
    y_pred = model.predict(X)

    print(f"\n=== {dataset_name}混淆矩阵 ===")
    print(confusion_matrix(y, y_pred))

    print(f"\n=== {dataset_name}分类报告 ===")
    print(classification_report(y, y_pred, digits=4))


def main() -> None:
    X_train, X_test, y_train, y_test = build_train_test_data(
        test_size=0.2,
        random_state=42,
        n_neighbors=5,
    )

    param_grid = {
        # 搜索空间沿用步骤 8，只改变模型选择目标，避免混入其他变量。
        "n_estimators": [200, 300, 500, 800],
        "max_depth": [5, 6, 7, 8, 9, 10],
        "min_samples_split": [10, 20, 30, 50, 80],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        # 医学场景更怕漏判时，可以把搜索目标改为 recall。
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    print("=== 网格搜索参数范围 ===")
    pprint(param_grid)

    print("\n=== 最优参数 ===")
    pprint(grid_search.best_params_)
    print(f"最优交叉验证 Recall: {grid_search.best_score_:.4f}")

    results = pd.DataFrame(grid_search.cv_results_)
    # 查看 Recall 排名前五的参数组合，辅助分析“少漏判”目标的代价。
    top_results = results.sort_values("rank_test_score").head(5)
    print("\n=== 交叉验证 Recall 前五组参数 ===")
    print(
        top_results[
            [
                "rank_test_score",
                "mean_test_score",
                "mean_train_score",
                "param_n_estimators",
                "param_max_depth",
                "param_min_samples_split",
            ]
        ].to_string(index=False)
    )

    best_model = grid_search.best_estimator_
    evaluate_model(best_model, X_train, y_train, "训练集")
    evaluate_model(best_model, X_test, y_test, "测试集")


if __name__ == "__main__":
    main()
