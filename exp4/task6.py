import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import time

def main():
    print('=' * 50)
    print('使用全量特征进行模型构建与优化 (对比实验)')
    print('=' * 50)
    
    # 1. 加载数据集
    df = pd.read_csv('FE_day.csv')
    
    # 2. 使用所有特征 (剔除 instant 序号和标签 cnt)
    X = df.drop(columns=['instant', 'cnt'])
    y = df['cnt']
    
    print(f"使用的特征数量: {X.shape[1]}")
    
    # 3. 拆分数据集，测试集占比 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 4. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. 模型构建与超参数优化
    print("\n开始进行 GridSearchCV 超参数搜索 (核函数=rbf)...")
    param_grid = {
        'C': [100, 1000, 5000, 10000],          # 正则化参数
        'gamma': [0.001, 0.01, 0.1, 1],         # rbf 核函数的核系数
        'epsilon': [0.1, 1, 10, 100]            # 损失函数中的 epsilon 管道宽度
    }
    
    svr = SVR(kernel='rbf')
    
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=5,                           # 5折交叉验证
        scoring='r2',                   # 回归问题通常可以以 R^2 作为得分标准
        n_jobs=-1,                      # 使用所有可用的 CPU 核心
        verbose=1                       # 输出进度信息
    )
    
    start_time = time.time()
    # 在训练集上进行搜索和拟合
    grid_search.fit(X_train_scaled, y_train)
    end_time = time.time()
    
    print('\n' + '=' * 50)
    print('超参数优化结果')
    print('=' * 50)
    print(f"最优超参数组合: {grid_search.best_params_}")
    print(f"交叉验证最高得分 (R^2): {grid_search.best_score_:.4f}")
    print(f"网格搜索耗时: {end_time - start_time:.2f} 秒")
    
    from sklearn.metrics import mean_squared_error, r2_score
    best_model = grid_search.best_estimator_
    
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    print('\n' + '=' * 50)
    print('模型评估 (最优模型)')
    print('=' * 50)
    print(f"训练集 R^2: {train_r2:.4f}, MSE: {train_mse:.2f}")
    print(f"测试集 R^2: {test_r2:.4f}, MSE: {test_mse:.2f}")
    
    print('\n' + '=' * 50)
    print('与 Task3 (精简20个特征) 测试集结果对比：')
    print('Task3 最优参数组合: {\'C\': 5000, \'epsilon\': 0.1, \'gamma\': 0.01}')
    print('Task3 测试集 R^2 得分: 0.6222')
    print('Task3 测试集 MSE: 1327591.25')
    print('-' * 30)
    print(f'全量特征 最优参数组合: {grid_search.best_params_}')
    print(f'全量特征 测试集 R^2 得分: {test_r2:.4f}')
    print(f'全量特征 测试集 MSE: {test_mse:.2f}')
    
    diff = test_r2 - 0.6222
    if diff > 0:
        print(f"结论: 使用全量特征后，测试集 R^2 得分上升了 {diff:.4f}")
    else:
        print(f"结论: 使用全量特征后，测试集 R^2 得分下降了 {abs(diff):.4f}")

if __name__ == '__main__':
    main()
