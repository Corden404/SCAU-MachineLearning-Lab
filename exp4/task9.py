import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import time

def main():
    print('=' * 60)
    print('Task 9: 引入正余弦编码处理周期性特征 (月份、星期、季节)')
    print('=' * 60)
    
    # 1. 加载数据集
    df = pd.read_csv('FE_day.csv')
    
    # 2. 逆向还原类别特征
    # 月份 (1-12)
    month_cols = [f'mnth_{i}' for i in range(1, 13)]
    df['month'] = df[month_cols].idxmax(axis=1).apply(lambda x: int(x.split('_')[1]))
    
    # 星期 (0-6)
    weekday_cols = [f'weekday_{i}' for i in range(0, 7)]
    df['weekday'] = df[weekday_cols].idxmax(axis=1).apply(lambda x: int(x.split('_')[1]))
    
    # 季节 (1-4)
    season_cols = [f'season_{i}' for i in range(1, 5)]
    df['season'] = df[season_cols].idxmax(axis=1).apply(lambda x: int(x.split('_')[1]))
    
    # 3. 正余弦编码 (Sine and Cosine Encoding)
    # 月份编码
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 星期编码
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    
    # 季节编码
    df['season_sin'] = np.sin(2 * np.pi * df['season'] / 4)
    df['season_cos'] = np.cos(2 * np.pi * df['season'] / 4)
    
    # 4. 构建新的特征集
    # 包含基础连续特征/二分类特征，以及正余弦编码特征
    features = [
        'atemp', 'yr', 'windspeed', 'hum', 'workingday', 'holiday', 
        'weathersit_1', 'weathersit_2', 'weathersit_3',
        'month_sin', 'month_cos', 
        'weekday_sin', 'weekday_cos', 
        'season_sin', 'season_cos'
    ]
    
    X = df[features]
    y = df['cnt']
    
    print(f"使用的特征数量: {X.shape[1]}")
    print(f"特征列表: {', '.join(features)}\n")
    
    # 5. 划分数据集 (按时间顺序，无 shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 6. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. 模型构建与超参数优化
    print("开始进行 GridSearchCV 超参数搜索 (正余弦编码 + TimeSeriesSplit)...")
    param_grid = {
        'C': [10000, 50000, 80000, 100000],
        'gamma': [0.0001, 0.001, 0.01, 0.1],
        'epsilon': [100, 500, 800, 1000]
    }
    
    svr = SVR(kernel='rbf')
    # 使用 TimeSeriesSplit 确保内部交叉验证也不存在时间上的“偷看”
    tscv = TimeSeriesSplit(n_splits=5)
    grid_search = GridSearchCV(svr, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
    
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    duration = time.time() - start_time
    
    print(f"\n最优参数组合: {grid_search.best_params_}")
    
    # 8. 预测与评估
    best_model = grid_search.best_estimator_
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    print('\n' + '=' * 60)
    print('正余弦编码模型评估结果')
    print('=' * 60)
    print(f"训练集 R^2: {train_r2:.4f}, MSE: {train_mse:.2f}")
    print(f"测试集 R^2: {test_r2:.4f}, MSE: {test_mse:.2f}")
    print(f"模型训练耗时: {duration:.2f} 秒")
    
    print('\n' + '-' * 60)
    print('与 Task 8 (独热编码) 测试集结果对比')
    print('-' * 60)
    print(f"独热编码 (Task 8)  -> R^2: 0.5937, MSE: 1427765.18")
    print(f"正余弦编码 (Task 9) -> R^2: {test_r2:.4f}, MSE: {test_mse:.2f}")
    
    diff_r2 = test_r2 - 0.5937
    if diff_r2 > 0:
        print(f"\n结论: 引入正余弦编码后，测试集 R^2 得分上升了 {diff_r2:.4f}！模型泛化能力变强。")
    else:
        print(f"\n结论: 引入正余弦编码后，测试集 R^2 得分下降了 {-diff_r2:.4f}。")

if __name__ == '__main__':
    main()
