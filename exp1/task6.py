import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'HousingData.csv')
    
    print("====================================================")
    print("实验1 任务6: 基于前置EDA结论的“理论最优”线性回归模型")
    print("理论支撑:")
    print("1. 缺失值: 存在偏态异常值，采用中位数插补(SimpleImputer)更稳健")
    print("2. 偏态与量纲灾难: 采用 PowerTransformer(强制正态化+标准化)")
    print("3. 多重共线性: 采用 ElasticNet(结合L1/L2正则化)动态进行特征筛选平滑")
    print("====================================================\n")
    
    # 1. 载入数据
    try:
        data = pd.read_csv(data_path)
        print(f"成功加载数据集，形状: {data.shape}")
    except FileNotFoundError:
        print(f"数据文件未找到: {data_path}")
        return

    if 'MEDV' not in data.columns:
        print("未找到目标列 'MEDV'")
        return
        
    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    
    # 2. 构建理论最优 Pipeline
    # - 针对 Task 2 & 3: 用中位数填补替代均值填补，避免极大值（如犯罪率）拉偏均值。
    # - 针对 Task 3 & 5: 使用 PowerTransformer 代替 StandardScaler，一方面完成均值0方差1的缩放，另一方面把严重右偏的数据（长尾分布）进行类似对数的变换，使其尽量服从正态分布。
    # - 针对 Task 4: 使用 ElasticNet，网格搜索其 l1_ratio，让算法自己在 Lasso 和 Ridge 之间寻找最佳平衡，解决特征多重共线性问题。
    optimal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', PowerTransformer(method='yeo-johnson', standardize=True)),
        ('model', ElasticNet(random_state=2005))
    ])
    
    # 3. 设定超参数网格并进行 5 折交叉验证寻找最优参数
    cv = KFold(n_splits=5, shuffle=True, random_state=2005)
    
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9] # l1_ratio=0退化为Ridge, l1_ratio=1退化为Lasso
    }
    # 超参数对性能的影响是一定是 U 型的，属于传统的凸优化机器学习
    
    print("正在进行 5折交叉验证 + 暴力网格搜索超参数...")
    grid_search = GridSearchCV(
        estimator=optimal_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='r2', # 评价标准定为 R-square
        refit=True,   # 找到最好的参数后，在所有数据上重新拟合一次以得到 best_model
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    
    # 输出最佳参数
    best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
    print(f"\n=> 寻找到的最优模型超参数: {best_params}")
    
    # 4. 获取整体的评估指标 
    # cv_predict能够模拟在未知数据（测试集）上的预测表现
    print("\n评估模型泛化性能...")
    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1)
    
    mse = mean_squared_error(y, y_pred_cv)
    mae = mean_absolute_error(y, y_pred_cv)
    r2 = r2_score(y, y_pred_cv)
    
    print("\n" + "="*40)
    print("理论最优模型交叉验证(CV=5)最终评估结果:")
    print("="*40)
    print(f"MSE (均方误差)     : {mse:.4f}")
    print(f"MAE (平均绝对误差) : {mae:.4f}")
    print(f"R-square (决定系数): {r2:.4f}")
    print("========================================\n")
    
    # 5. 可视化分析
    print("正在生成分析图表...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('理论最优模型 (ElasticNet + 偏态惩罚优化) 性能可视化', fontsize=16)
    
    # 图 1: 真实值 vs 预测值
    ax1 = axes[0]
    ax1.scatter(y, y_pred_cv, alpha=0.6, color='mediumseagreen', edgecolors='w', s=60)
    min_val = min(y.min(), y_pred_cv.min())
    max_val = max(y.max(), y_pred_cv.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线 (y=x)')
    ax1.set_title(f'真实值 vs 预测值 ($R^2$ = {r2:.4f})')
    ax1.set_xlabel('真实房价 (MEDV)')
    ax1.set_ylabel('交叉验证预测房价')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 图 2: 残差分布图 (高阶检验方式，正规线性回归残差必须趋向于白噪声/正态分布)
    ax2 = axes[1]
    residuals = y - y_pred_cv
    sns.histplot(residuals, kde=True, color='purple', ax=ax2)
    ax2.axvline(0, color='r', linestyle='--', label='零误差线')
    ax2.set_title('残差分布直方图 (衡量模型偏差)')
    ax2.set_xlabel('残差 (真实值 - 预测值)')
    ax2.set_ylabel('频数')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(base_dir, 'task6_optimal_model_analysis.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"=> 评估图表已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
