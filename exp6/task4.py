from pprint import pprint

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from task2 import build_train_test_data


def evaluate_model(model, X, y, dataset_name: str) -> None:
    # 输出混淆矩阵和分类报告，重点观察类别 1 的 recall 与 f1。
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
        # 调整树数量和单棵树复杂度，用来缓解默认随机森林过拟合。
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [3, 5, 7, 9, None],
        "min_samples_split": [2, 5, 10, 20],
    }

    # 分层 5 折交叉验证让每折中正负样本比例更稳定。
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        # 糖尿病预测既关心漏判也关心误判，这里用 F1 综合 precision 和 recall。
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    print("=== 网格搜索参数范围 ===")
    pprint(param_grid)

    print("\n=== 最优参数 ===")
    pprint(grid_search.best_params_)
    print(f"最优交叉验证 F1: {grid_search.best_score_:.4f}")

    results = pd.DataFrame(grid_search.cv_results_)
    # 查看前五名参数组合，判断最优结果是否稳定或是否贴近搜索边界。
    top_results = results.sort_values("rank_test_score").head(5)
    print("\n=== 交叉验证 F1 前五组参数 ===")
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

    # best_estimator_ 已经用最优参数在完整训练集上重新拟合。
    best_model = grid_search.best_estimator_
    evaluate_model(best_model, X_train, y_train, "训练集")
    evaluate_model(best_model, X_test, y_test, "测试集")


if __name__ == "__main__":
    main()
