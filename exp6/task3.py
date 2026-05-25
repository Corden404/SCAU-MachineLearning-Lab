from pprint import pprint

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from task2 import build_train_test_data


def evaluate_model(model, X, y, dataset_name: str) -> None:
    # 同一评估函数复用在训练集和测试集，方便直接比较是否过拟合。
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

    # 步骤 3 要求使用默认参数，先建立一个未经调参的随机森林基线模型。
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 打印完整参数，便于解释默认 max_depth、n_estimators 等设置带来的影响。
    print("=== 随机森林模型参数 ===")
    pprint(model.get_params())

    evaluate_model(model, X_train, y_train, "训练集")
    evaluate_model(model, X_test, y_test, "测试集")


if __name__ == "__main__":
    main()
