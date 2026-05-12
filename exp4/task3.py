import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def main():
    print('=' * 50)
    print('模型构建与超参数优化')
    print('=' * 50)
    
    # 1. 加载数据集
    df = pd.read_csv('FE_day.csv')
    
    # 2. 划分数据集，测试集占比 20%
    X_full = df.drop(columns=['instant', 'cnt'])
    y = df['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, shuffle=False)
    
    # 3. 仅使用训练集计算相关系数矩阵并筛选特征
    # 剔除了多余/弱相关的特征，去除了共线性特征，保留了20个核心特征
    features = [
        'atemp', 'yr', 'windspeed', 'hum', 'workingday', 'holiday', 
        'weathersit_1', 'weathersit_2', 'weathersit_3',
        'mnth_1', 'mnth_2', 'mnth_3', 'mnth_5', 'mnth_6', 'mnth_7', 
        'mnth_8', 'mnth_9', 'mnth_10', 'mnth_11', 'mnth_12'
    ]
    
    X_train = X_train[features]
    X_test = X_test[features]
    
    print(f"使用的特征数量: {X_train.shape[1]}")
    print(f"包含的特征: {', '.join(features)}")
    
    print(f"\n训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")
    
    # 4. 特征标准化 (SVM 对数值尺度敏感，故需要对特征进行标准化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # 注意测试集只能使用训练集的 scaler 进行 transform
    X_test_scaled = scaler.transform(X_test)
    
    # 5. 模型构建与超参数优化
    print("\n开始进行 GridSearchCV 超参数搜索 (核函数=rbf)...")
    # 设置参数网格
    param_grid = {
        'C': [100, 1000, 5000, 10000],          # 正则化参数
        'gamma': [0.001, 0.01, 0.1, 1],         # rbf 核函数的核系数
        'epsilon': [0.1, 1, 10, 100]            # 损失函数中的 epsilon 管道宽度
    }
    
    # 建模采用 rbf 核
    svr = SVR(kernel='rbf')
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=5,                           # 5折交叉验证
        scoring='r2',                   # 回归问题通常可以以 R^2 作为得分标准
        n_jobs=-1,                      # 使用所有可用的 CPU 核心
        verbose=1                       # 输出进度信息
    )
    
    # 在训练集上进行搜索和拟合
    grid_search.fit(X_train_scaled, y_train)
    
    print('\n' + '=' * 50)
    print('超参数优化结果')
    print('=' * 50)
    print(f"最优超参数组合: {grid_search.best_params_}")
    print(f"交叉验证最高得分 (R^2): {grid_search.best_score_:.4f}")

    from sklearn.metrics import mean_squared_error, r2_score
    best_model = grid_search.best_estimator_
    
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)
    
    print('\n' + '=' * 50)
    print('模型评估 (最优模型)')
    print('=' * 50)
    print(f"训练集 R^2: {r2_score(y_train, y_train_pred):.4f}, MSE: {mean_squared_error(y_train, y_train_pred):.2f}")
    print(f"测试集 R^2: {r2_score(y_test, y_test_pred):.4f}, MSE: {mean_squared_error(y_test, y_test_pred):.2f}")

if __name__ == '__main__':
    main()
