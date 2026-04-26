import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

def main():
    data = pd.read_csv('digits.csv')
    X = data.iloc[:, :-1].values
    y = data['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 针对多项式核函数(poly)，定义需要搜索的超参数空间：
    # degree: 多项式的最高阶数（取值1到9）
    # C: 正则化参数
    param_grid = {
        'degree': list(range(1, 10)),
        'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10]
    }
    
    # 初始化指定多项式核的 SVM 模型
    model = SVC(kernel='poly')
    # 配置 GridSearchCV 执行针对 degree 和 C 组合的 5 折交叉验证搜索
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    results = grid.cv_results_
    
    print(f"{'degree':<10} {'C':<10} {'Mean CV Accuracy'}")
    print("-" * 40)
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f"{params['degree']:<10} {params['C']:<10} {mean_score:.4f}")
        
    # 提取交叉验证过程中分类准确率最高的参数组合
    print("\n[Best Poly Model Parameters]")
    print(f"Best degree: {grid.best_params_['degree']}")
    print(f"Best C: {grid.best_params_['C']}")
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")
    
    # 验证该最佳多项式模型在测试集上的最终准确率
    best_model = grid.best_estimator_
    test_acc = best_model.score(X_test, y_test)
    print(f"Test Accuracy with best params: {test_acc:.4f}")

if __name__ == "__main__":
    main()
