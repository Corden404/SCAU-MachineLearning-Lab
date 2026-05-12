import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import time

def main():
    print('=' * 60)
    print('Task 10: 尝试多项式核 (Polynomial Kernel) 对比实验')
    print('=' * 60)
    
    # 1. 加载数据集并进行正余弦编码 (继承 Task 9 的特征工程)
    df = pd.read_csv('FE_day.csv')
    
    # 类别特征还原与正余弦编码
    for col, count, start in [('month', 12, 1), ('weekday', 7, 0), ('season', 4, 1)]:
        cols = [f'mnth_{i}' if col=='month' else f'{col}_{i}' for i in range(start, start+count)]
        df[col] = df[cols].idxmax(axis=1).apply(lambda x: int(x.split('_')[-1]))
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / count)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / count)

    features = [
        'atemp', 'yr', 'windspeed', 'hum', 'workingday', 'holiday', 
        'weathersit_1', 'weathersit_2', 'weathersit_3',
        'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos', 'season_sin', 'season_cos'
    ]
    
    X = df[features]
    y = df['cnt']
    
    # 2. 划分数据集 (时间顺序)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. 多项式核实验 (Degree 1-5)
    print("开始进行多项式核 Degree (1-5) 的性能测试...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    results = []
    
    for d in range(1, 6):
        print(f"\n正在评估 Degree = {d} ...")
        # 为每个 degree 进行一轮参数微调，确保对比公平
        param_grid = {
            'C': [1000, 10000, 50000],
            'epsilon': [100, 500],
            'gamma': ['scale', 'auto']
        }
        
        svr = SVR(kernel='poly', degree=d)
        grid_search = GridSearchCV(svr, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
        
        start_time = time.time()
        grid_search.fit(X_train_scaled, y_train)
        duration = time.time() - start_time
        
        # 评估最优模型
        best_model = grid_search.best_estimator_
        y_test_pred = best_model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        results.append({
            'degree': d,
            'best_C': grid_search.best_params_['C'],
            'test_r2': test_r2,
            'test_mse': test_mse,
            'time': duration
        })
        print(f"Degree {d} 完成. 测试集 R^2: {test_r2:.4f}, 最优 C: {grid_search.best_params_['C']}")

    # 5. 结果汇总对比
    print('\n' + '=' * 60)
    print(f"{'Degree':<8} | {'最优 C':<10} | {'测试集 R^2':<12} | {'测试集 MSE':<15}")
    print('-' * 60)
    for res in results:
        print(f"{res['degree']:<8} | {res['best_C']:<10} | {res['test_r2']:<12.4f} | {res['test_mse']:<15.2f}")
    
    print('\n' + '=' * 60)
    print(f"Task 9 (RBF核) 参考 R^2: 0.6567")
    
    # 找出 Poly 中的最优值
    best_poly = max(results, key=lambda x: x['test_r2'])
    print(f"多项式核最优结果: Degree {best_poly['degree']} (R^2: {best_poly['test_r2']:.4f})")
    
    if best_poly['test_r2'] > 0.6567:
        print(f"结论: 多项式核 (Degree {best_poly['degree']}) 成功超越了 RBF 核！")
    else:
        print(f"结论: 多项式核在本任务中未能超越 RBF 核。")

if __name__ == '__main__':
    main()
