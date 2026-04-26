import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

def main():
    data = pd.read_csv('digits.csv')
    X = data.iloc[:, :-1].values
    y = data['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 定义需要使用网格搜索进行优化的超参数字典组合：gamma 控制单个样本的作用范围，C 为正则化参数
    param_grid = {
        'gamma': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        'C': [0.01, 0.05, 0.1, 0.5, 1, 10]
    }
    
    # 初始化采用径向基（RBF）核函数的 SVM 模型
    model = SVC(kernel='rbf')
    # 配置 GridSearchCV，使用 5 折交叉验证（cv=5）自动遍历 param_grid 中所有的超参数组合
    grid = GridSearchCV(model, param_grid, cv=5)
    # 在训练集数据上执行穷举式的参数搜索过程
    grid.fit(X_train, y_train)
    
    # 获取包含每次网格参数组合及其对应验证分数的详细结果
    results = grid.cv_results_
    
    print(f"{'gamma':<10} {'C':<10} {'Mean CV Accuracy'}")
    print("-" * 40)
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f"{params['gamma']:<10} {params['C']:<10} {mean_score:.4f}")
        
    # 输出在交叉验证过程中获得最高准确率的最佳超参数组合
    print("\n[Best RBF Model Parameters]")
    print(f"Best gamma: {grid.best_params_['gamma']}")
    print(f"Best C: {grid.best_params_['C']}")
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")
    
    # 从 GridSearchCV 中提取最佳参数对应的模型实例，并在独立的测试集上进行最终验证
    best_model = grid.best_estimator_
    test_acc = best_model.score(X_test, y_test)
    print(f"Test Accuracy with best params: {test_acc:.4f}")

if __name__ == "__main__":
    main()
