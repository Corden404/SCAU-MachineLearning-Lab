import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    # 读取数据并分离出特征矩阵(X)与目标标签(y)
    data = pd.read_csv('digits.csv')
    X = data.iloc[:, :-1].values
    y = data['target'].values
    
    # 划分数据集：70%的数据用于训练，30%的数据用于测试，random_state=42保证每次划分结果一致
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 定义需要进行分类对比的四种支持向量机(SVM)核函数
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    for kernel in kernels:
        # 使用指定的核函数实例化 SVM 分类器，其余参数保持默认设置
        model = SVC(kernel=kernel)
        # 在训练集上拟合(训练)模型
        model.fit(X_train, y_train)
        # 使用训练好的模型对测试集的特征进行分类预测
        y_pred = model.predict(X_test)
        
        # 将测试集的预测结果与真实标签进行比对，计算准确率(Accuracy)
        acc = accuracy_score(y_test, y_pred)
        print(f"===== Kernel: {kernel} =====")
        print(f"Accuracy: {acc:.4f}")
        print(f"True labels (first 20) : {y_test[:20]}")
        print(f"Predicted labels (first 20): {y_pred[:20]}")
        print("-" * 40)

if __name__ == "__main__":
    main()
