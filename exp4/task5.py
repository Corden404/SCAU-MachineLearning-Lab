import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def main():
    print("开始生成模型评估和可视化结果...")
    
    # 1. 加载数据集
    df = pd.read_csv('FE_day.csv')
    
    # 2. 拆分数据集，测试集占比 20%
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
    
    # 5. 模型构建与超参数优化
    param_grid = {
        'C': [100, 1000, 5000, 10000],          
        'gamma': [0.001, 0.01, 0.1, 1],         
        'epsilon': [0.1, 1, 10, 100]            
    }
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=5,                           
        scoring='r2',                   
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    # ================= 步骤 5: 结果可视化 =================
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 提取交叉验证结果
    results = pd.DataFrame(grid_search.cv_results_)
    
    # 固定 epsilon 为最优值
    best_epsilon = grid_search.best_params_['epsilon']
    filtered_results = results[results['param_epsilon'] == best_epsilon]
    
    # 提取 C, gamma 和 得分，构建热力图矩阵
    scores_matrix = filtered_results.pivot(index='param_C', columns='param_gamma', values='mean_test_score')
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(scores_matrix, annot=True, cmap='viridis', fmt='.4f', cbar_kws={'label': 'R^2 得分'})
    plt.title(f'SVM 模型精度热力图 (C vs Gamma)\n[固定 epsilon={best_epsilon}]', fontsize=16)
    plt.xlabel('Gamma', fontsize=14)
    plt.ylabel('C', fontsize=14)
    plt.tight_layout()
    plt.savefig('svm_heatmap.png', dpi=150)
    print("热力图已保存至 svm_heatmap.png")

if __name__ == '__main__':
    main()
