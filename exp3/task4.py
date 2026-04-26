import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main():
    # 读取数据并将其划分为 70% 的训练集与 30% 的测试集
    data = pd.read_csv('digits.csv')
    X = data.iloc[:, :-1].values
    y = data['target'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    # 定义不同的正则化惩罚参数 C 的取值列表（C 值越小，正则化约束越强）
    C_values = [0.01, 0.05, 0.1, 0.5, 1, 10]
    
    best_acc = 0
    best_params = {}
    
    print(f"{'Kernel':<10} {'C':<10} {'Accuracy'}")
    print("-" * 35)
    
    # 嵌套循环：遍历所有给定的核函数类型以及对应的参数 C 的组合
    for kernel in kernels:
        for C in C_values:
            # 采用特定的核函数与 C 参数构建 SVM 分类器模型
            model = SVC(kernel=kernel, C=C)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 评估当前参数组合模型在测试集上的准确率
            acc = accuracy_score(y_test, y_pred)
            print(f"{kernel:<10} {C:<10} {acc:.4f}")
            
            # 比较并记录准确率最高的模型得分及其对应的超参数
            if acc > best_acc:
                best_acc = acc
                best_params = {'kernel': kernel, 'C': C}
                
    print("\n[Highest Score Model]")
    print(f"Kernel: {best_params['kernel']}, C: {best_params['C']}, Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
