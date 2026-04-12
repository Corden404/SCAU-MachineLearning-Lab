import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'HousingData.csv')
    
    print("====================================================")
    print("实验1 任务7: 默认参数下四种线性模型的基线性能评估")
    print("1. LinearRegression (普通最小二乘法，无正则化)")
    print("2. Ridge (L2 正则化，默认 alpha=1.0)")
    print("3. Lasso (L1 正则化，默认 alpha=1.0)")
    print("4. ElasticNet (L1+L2，默认 alpha=1.0, l1_ratio=0.5)")
    print("====================================================\n")
    
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"找不到数据: {data_path}")
        return

    X = data.drop('MEDV', axis=1)
    y = data['MEDV']
    
    # 保持与 task6 设定的完全一致的验证折分，以便苹果对苹果地对比
    cv = KFold(n_splits=5, shuffle=True, random_state=2005)
    
    # 为了让 Ridge/Lasso 不要因为量纲崩溃，我们依然保留最基础也是最核心的处理
    # 这里用和 Task 6 完全一致的基础 pipeline 架构，以便完全控制变量对比模型本身
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', PowerTransformer(method='yeo-johnson', standardize=True))
    ])
    
    # 四种默认参数模型
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),            
        'Lasso': Lasso(),            
        'ElasticNet': ElasticNet()   
    }
    
    scoring = {
        'MSE': make_scorer(mean_squared_error),
        'MAE': make_scorer(mean_absolute_error),
        'R-square': make_scorer(r2_score)
    }
    
    results = []
    
    print("开始在不同模型中运行...")
    for name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        mse = np.mean(cv_results['test_MSE'])
        mae = np.mean(cv_results['test_MAE'])
        r2 = np.mean(cv_results['test_R-square'])
        
        print(f"[{name:16s}] -> MSE: {mse:7.4f} | MAE: {mae:6.4f} | R-square: {r2:6.4f}")
        results.append({
            'Model': name,
            'MSE': mse,
            'MAE': mae,
            'R-square': r2
        })
        
    results_df = pd.DataFrame(results)
    
    # 可视化对比图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False 
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('任务7：四种默认参数线性模型性能对比', fontsize=16, y=1.05)
    
    sns.barplot(x='Model', y='MSE', data=results_df, ax=axes[0], hue='Model', palette='rocket', legend=False)
    axes[0].set_title('均方误差 MSE (越低越好)')
    
    sns.barplot(x='Model', y='MAE', data=results_df, ax=axes[1], hue='Model', palette='rocket', legend=False)
    axes[1].set_title('平均绝对误差 MAE (越低越好)')
    
    sns.barplot(x='Model', y='R-square', data=results_df, ax=axes[2], hue='Model', palette='mako', legend=False)
    axes[2].set_title('决定系数 R² (越高越好)')
    
    for ax in axes:
        ax.tick_params(axis='x', rotation=15)
        
    plt.tight_layout()
    save_path = os.path.join(base_dir, 'task7_baseline_comparison.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\n=> 对比条形图已保存至: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
