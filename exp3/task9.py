import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

def main():
    # 1. 加载数据
    data = pd.read_csv('digits.csv')
    X = data.iloc[:, :-1].values
    y = data['target'].values
    
    # 2. 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 3. 定义网格搜索参数，按用户要求增加 C=100 和 C=1000
    param_grid = {
        'gamma': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        'C': [0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000]
    }
    
    # 4. 执行网格搜索
    print("正在执行网格搜索，请稍候...")
    model = SVC(kernel='rbf')
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    # 5. 整理结果为表格形式
    results = pd.DataFrame(grid.cv_results_)
    # 提取 mean_test_score 并进行透视
    # 我们希望行是 C，列是 gamma
    viz_data = results.pivot(index='param_C', columns='param_gamma', values='mean_test_score')
    
    # 6. 表格可视化 - 控制台输出
    print("\n[网格搜索结果表格 (Mean CV Accuracy)]")
    print(viz_data)
    
    # 7. 表格可视化 - 热图 (Heatmap)
    plt.figure(figsize=(12, 8))
    sns.heatmap(viz_data, annot=True, fmt=".4f", cmap="YlGnBu", cbar_kws={'label': 'Mean CV Accuracy'})
    plt.title('Grid Search Accuracy: SVM RBF Kernel (C vs Gamma)')
    plt.xlabel('Gamma')
    plt.ylabel('C')
    
    # 保存结果
    save_path = 'task9_grid_search_heatmap.png'
    plt.savefig(save_path)
    print(f"\n可视化热图已保存至: {save_path}")
    
    # 8. 输出最佳参数
    print("\n[最佳模型参数]")
    print(f"Best gamma: {grid.best_params_['gamma']}")
    print(f"Best C: {grid.best_params_['C']}")
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")
    
    # 在测试集上验证
    test_acc = grid.best_estimator_.score(X_test, y_test)
    print(f"Test Accuracy with best params: {test_acc:.4f}")

if __name__ == "__main__":
    main()
