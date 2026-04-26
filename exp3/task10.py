import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

def main():
    data = pd.read_csv('digits.csv')
    X = data.iloc[:, :-1].values
    y = data['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 初始化数据归一化器 MinMaxScaler，用于将特征放缩到 [0, 1] 区间
    scaler = MinMaxScaler()
    # 使用训练集的数据拟合归一化器并转换训练集特征
    X_train_scaled = scaler.fit_transform(X_train)
    # 使用由训练集学习到的最值对测试集进行转换
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'gamma': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        'C': [0.01, 0.05, 0.1, 0.5, 1, 10]
    }
    
    model = SVC(kernel='rbf')
    grid = GridSearchCV(model, param_grid, cv=5)
    # 将归一化后的训练集传入进行超参数的网格搜索与交叉验证
    grid.fit(X_train_scaled, y_train)
    
    results = grid.cv_results_
    
    print(f"{'gamma':<10} {'C':<10} {'Mean CV Accuracy (MinMaxScaled)'}")
    print("-" * 45)
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        print(f"{params['gamma']:<10} {params['C']:<10} {mean_score:.4f}")
        
    print("\n[Best MinMaxScaled RBF Model Parameters]")
    print(f"Best gamma: {grid.best_params_['gamma']}")
    print(f"Best C: {grid.best_params_['C']}")
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")
    
    best_model = grid.best_estimator_
    # 使用同样归一化处理过的测试集对获得的最优参数模型进行评估
    test_acc = best_model.score(X_test_scaled, y_test)
    print(f"Test Accuracy with best params: {test_acc:.4f}")

if __name__ == "__main__":
    main()
