import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    data = pd.read_csv('digits.csv')
    X = data.iloc[:, :-1].values
    y = data['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 首先执行与 task5 相同的过程：对 RBF 核函数进行参数网格搜索以获取表现最优的模型
    param_grid = {
        'gamma': [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
        'C': [0.01, 0.05, 0.1, 0.5, 1, 10]
    }
    model = SVC(kernel='rbf')
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
    
    # 获取最高交叉验证准确率对应的最佳 RBF SVM 实例
    best_model = grid.best_estimator_
    print(f"Best RBF Model: {best_model}")
    
    # 使用这个最佳模型预测测试集，用于生成混淆矩阵的数据
    y_pred = best_model.predict(X_test)
    
    # 计算混淆矩阵：它能详细反映各类标签被正确分类与被错误分类到其他类别的数量分布
    cm = confusion_matrix(y_test, y_pred)
    # 使用 ConfusionMatrixDisplay 传入混淆矩阵与对应的类标签来实例化可视化对象
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    
    # 创建一个 matplotlib 图像并使用特定的色图（Blues，即蓝色渐变）进行热力图绘制
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix (RBF Kernel - Best Accuracy)')
    
    # 将生成的混淆矩阵可视化图表输出保存为本地 PNG 文件
    plt.savefig('task8_confusion_matrix.png')
    # plt.show()

if __name__ == "__main__":
    main()
