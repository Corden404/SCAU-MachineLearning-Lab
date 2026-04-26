import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 读取数据集并提取特征矩阵
    data = pd.read_csv('digits.csv')
    X = data.iloc[:, :-1].values
    
    # 创建一个 4 行 5 列的图像子图网格，用于显示前 20 张图片
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    fig.suptitle('First 20 Sample Images', fontsize=16)
    
    for i, ax in enumerate(axes.ravel()):
        if i < 20:
            # 将 64 维的一维特征数组重新塑形为 8x8 的二维矩阵，这样就可以用索引i来访问每个子图
            ax.matshow(X[i].reshape(8, 8), cmap='gray')
            # 关闭坐标轴的显示，使得只展示图像内容
            ax.axis('off')
            
    # 自动调整子图参数，使所有子图紧凑地显示，避免重叠
    plt.tight_layout()
    # 将生成的图像保存为本地的 PNG 文件
    plt.savefig('task2_images.png')
    # plt.show()

if __name__ == "__main__":
    main()
