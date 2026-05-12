import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import time

def main():
    print('=' * 50)
    print('SVM 超参数追加优化 (探索更大的 C 和 epsilon)')
    print('=' * 50)
    
    # 1. 加载数据集
    df = pd.read_csv('FE_day.csv')
    
    # 2. 划分数据集
    X_full = df.drop(columns=['instant', 'cnt'])
    y = df['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, shuffle=False)
    
    # 3. 仅使用训练集筛选特征
    features = [
        'atemp', 'yr', 'windspeed', 'hum', 'workingday', 'holiday', 
        'weathersit_1', 'weathersit_2', 'weathersit_3',
        'mnth_1', 'mnth_2', 'mnth_3', 'mnth_5', 'mnth_6', 'mnth_7', 
        'mnth_8', 'mnth_9', 'mnth_10', 'mnth_11', 'mnth_12'
    ]
    
    X_train = X_train[features]
    X_test = X_test[features]
    
    # 4. 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. 模型构建与追加超参数优化
    print("\n开始进行 GridSearchCV 追加搜索...")
    # 上一轮最优 C=10000, epsilon=100, gamma=0.01
    # 扩大 C 和 epsilon 的搜索范围
    param_grid = {
        'C': [30000, 50000, 80000],
        'epsilon': [300, 500, 800, 1000, 1200],
        'gamma': [0.01]  # 固定为上一轮的最优值
    }
    
    svr = SVR(kernel='rbf')
    
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    end_time = time.time()
    
    print('\n' + '=' * 50)
    print('追加优化结果')
    print('=' * 50)
    print(f"最优超参数组合: {grid_search.best_params_}")
    print(f"交叉验证最高得分 (R^2): {grid_search.best_score_:.4f}")
    print(f"网格搜索耗时: {end_time - start_time:.2f} 秒")
    
    from sklearn.metrics import mean_squared_error, r2_score
    best_model = grid_search.best_estimator_
    
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    print('\n' + '=' * 50)
    print('模型评估 (追加优化后的最优模型)')
    print('=' * 50)
    print(f"训练集 R^2: {r2_score(y_train, y_train_pred):.4f}, MSE: {mean_squared_error(y_train, y_train_pred):.2f}")
    print(f"测试集 R^2: {r2_score(y_test, y_test_pred):.4f}, MSE: {mean_squared_error(y_test, y_test_pred):.2f}")
    
    # 6. 结果可视化 (C vs epsilon 热力图)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    results = pd.DataFrame(grid_search.cv_results_)
    
    # 因为 gamma 固定为 0.01，直接用 C 和 epsilon 构建热力图矩阵
    scores_matrix = results.pivot(index='param_C', columns='param_epsilon', values='mean_test_score')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(scores_matrix, annot=True, cmap='viridis', fmt='.4f', cbar_kws={'label': 'R^2 得分'})
    plt.title('SVM 模型精度热力图 (C vs Epsilon)\n[固定 gamma=0.01]', fontsize=16)
    plt.xlabel('Epsilon', fontsize=14)
    plt.ylabel('C', fontsize=14)
    plt.tight_layout()
    plt.savefig('svm_heatmap_task7.png', dpi=150)
    print("\n热力图已保存至 svm_heatmap_task7.png")

if __name__ == '__main__':
    main()
