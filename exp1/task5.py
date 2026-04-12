import pandas as pd
import matplotlib.pyplot as plt

def main():
    # 读取数据
    import os
    file_path = os.path.join(os.path.dirname(__file__), 'HousingData.csv')
    df = pd.read_csv(file_path)
    
    # 5. 画出不同变量在不同值之间的分布直方图（选取前 8 列）
    cols_to_plot = df.columns[:8]
    
    # 调整画布大小并绘制前 8 列的直方图
    df[cols_to_plot].hist(figsize=(10, 8), bins=10, edgecolor='black')
    
    # 自动调整子图间距，防止重叠
    plt.tight_layout()
    
    # 保存图像到本地
    save_path = os.path.join(os.path.dirname(__file__), 'housing_distribution.png')
    plt.savefig(save_path)
    print(f"图像已成功保存至: {save_path}")

if __name__ == "__main__":
    main()
