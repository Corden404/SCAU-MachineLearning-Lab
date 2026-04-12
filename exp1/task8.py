import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    # 获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'HousingData.csv')
    
    print("====================================================")
    print("实验1 任务8: 二次多项式升维 + 理论最优泛化")
    print("实验目的: ")
    print("探讨当基础线性域的挖掘潜力耗尽时，手动让模型引入特征的")
    print("“平方项”与“交叉互乘项”，能否带来预测精度的质的飞跃。")
    print("====================================================\n")
    
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"数据文件未找到: {data_path}")
        return

    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    
    # 构建升维最优 Pipeline
    # 不仅保留了 Task 6 的 Imputer 和 Transformer，还插入了 PolynomialFeatures
    # degree=2 代表它会根据 13 个基础变量，衍生出所有的 x_i * x_j，以及 x_i^2。
    optimal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('poly', PolynomialFeatures(degree=2, include_bias=False)), # 升维
        ('scaler', PowerTransformer(method='yeo-johnson', standardize=True)),
        # 升维后极易难以收敛，因此调高了 max_iter，其余一排照旧
        ('model', ElasticNet(random_state=2005, max_iter=20000))
    ])
    
    cv = KFold(n_splits=5, shuffle=True, random_state=2005)
    
    # 相比 Task6，升维后的变量群极度庞大且存在巨量噪音
    # 因此模型会极其依赖 alpha 来做变量剔除
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    print("正在生成约 104 维的高维数据，并网格寻优...")
    grid_search = GridSearchCV(
        estimator=optimal_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='r2', 
        refit=True,   
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    
    # 打印看看网格搜到了什么参数，以及最终留下了多少个有用特征？
    best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
    
    # 提取特征名称
    imputer = best_model.named_steps['imputer']
    poly = best_model.named_steps['poly']
    scaler = best_model.named_steps['scaler']
    en_model = best_model.named_steps['model']
    
    print(f"\n=> 寻找到的最优模型超参数: {best_params}")
    print(f"=> 原生变量数量: {X.shape[1]}")
    print(f"=> 经过 PolynomialFeatures(degree=2) 升维后的变量总数: {poly.get_feature_names_out().shape[0]}")
    # 筛选出系数不为 0 的特征数量（只有 Lasso / ElasticNet 才能办到这个，LinearRegression 做不到）
    nonzero_coefs = np.sum(en_model.coef_ != 0)
    print(f"=> ElasticNet剔除了无用跨维噪音，保留的核心有效衍生变量数: {nonzero_coefs}")
    
    # 获取整体的评估指标 
    print("\n评估高维降噪后的最优泛化性能...")
    y_pred_cv = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1)
    
    mse = mean_squared_error(y, y_pred_cv)
    mae = mean_absolute_error(y, y_pred_cv)
    r2 = r2_score(y, y_pred_cv)
    
    print("\n" + "="*40)
    print("高维二次空间模型(CV=5)最终评估结果:")
    print("="*40)
    print(f"MSE (均方误差)     : {mse:.4f}")
    print(f"MAE (平均绝对误差) : {mae:.4f}")
    print(f"R-square (决定系数): {r2:.4f}")
    print("========================================\n")
    
    # 可视化分析
    print("正在生成分析图表...")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('任务8 (多项式升维 + ElasticNet筛选) 性能可视化', fontsize=16)
    
    # 图 1: 真实值 vs 预测值
    ax1 = axes[0]
    ax1.scatter(y, y_pred_cv, alpha=0.6, color='coral', edgecolors='w', s=60)
    min_val = min(y.min(), y_pred_cv.min())
    max_val = max(y.max(), y_pred_cv.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线 (y=x)')
    ax1.set_title(f'真实值 vs 预测值 ($R^2$ = {r2:.4f})')
    ax1.set_xlabel('真实房价 (MEDV)')
    ax1.set_ylabel('交叉验证预测房价')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 图 2: 残差分布图
    ax2 = axes[1]
    residuals = y - y_pred_cv
    sns.histplot(residuals, kde=True, color='teal', ax=ax2)
    ax2.axvline(0, color='r', linestyle='--', label='零误差线')
    ax2.set_title('残差分布直方图')
    ax2.set_xlabel('残差 (真实值 - 预测值)')
    ax2.set_ylabel('频数')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = os.path.join(base_dir, 'task8_poly2_model_analysis.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"=> 评估图表已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
