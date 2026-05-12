import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import time

def run_experiment(name, shuffle_mode):
    print(f"\n>>> 正在运行实验: {name} (shuffle={shuffle_mode})")
    
    # 1. 加载数据
    df = pd.read_csv('FE_day.csv')
    X = df.drop(columns=['instant', 'cnt'])
    y = df['cnt']
    
    # 2. 划分数据集 (80/20)
    if shuffle_mode:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 3. 特征筛选 (使用 Task 3 确定的 20 个特征)
    features = [
        'atemp', 'yr', 'windspeed', 'hum', 'workingday', 'holiday', 
        'weathersit_1', 'weathersit_2', 'weathersit_3',
        'mnth_1', 'mnth_2', 'mnth_3', 'mnth_5', 'mnth_6', 'mnth_7', 
        'mnth_8', 'mnth_9', 'mnth_10', 'mnth_11', 'mnth_12'
    ]
    X_train = X_train[features]
    X_test = X_test[features]
    
    # 4. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. 网格搜索
    param_grid = {
        'C': [1000, 5000, 10000, 50000],
        'gamma': [0.001, 0.01, 0.1, 1],
        'epsilon': [10, 100, 500]
    }
    
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2', n_jobs=-1)
    
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    duration = time.time() - start_time
    
    # 6. 评估
    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test_scaled)
    
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    return {
        'best_params': grid_search.best_params_,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'duration': duration
    }

def main():
    # 运行随机抽样实验
    random_res = run_experiment("随机抽样 (Random Shuffle)", True)
    
    # 运行按时间顺序抽样实验
    temporal_res = run_experiment("按时间抽样 (Temporal Split)", False)
    
    # 输出对比结果
    print("\n" + "="*60)
    print(f"{'对比项':<20} | {'随机抽样 (Random)':<20} | {'按时间抽样 (Temporal)':<20}")
    print("-" * 60)
    
    print(f"{'最优 C':<20} | {random_res['best_params']['C']:<20} | {temporal_res['best_params']['C']:<20}")
    print(f"{'最优 Gamma':<20} | {random_res['best_params']['gamma']:<20} | {temporal_res['best_params']['gamma']:<20}")
    print(f"{'最优 Epsilon':<20} | {random_res['best_params']['epsilon']:<20} | {temporal_res['best_params']['epsilon']:<20}")
    print(f"{'测试集 R^2':<20} | {random_res['test_r2']:<20.4f} | {temporal_res['test_r2']:<20.4f}")
    print(f"{'测试集 MSE':<20} | {random_res['test_mse']:<20.2f} | {temporal_res['test_mse']:<20.2f}")
    print("="*60)
    
    diff_r2 = random_res['test_r2'] - temporal_res['test_r2']
    print(f"\n结果分析：")
    print(f"1. 随机抽样比按时间抽样的 R^2 高出: {diff_r2:.4f}")
    print(f"2. 按时间抽样的 MSE 是随机抽样的: {temporal_res['test_mse'] / random_res['test_mse']:.2f} 倍")
    print(f"3. 这种巨大差异证明了在时间序列数据中使用随机打乱会导致严重的'过拟合假象'或'信息泄漏'。")

if __name__ == '__main__':
    main()
