import pandas as pd

df = pd.read_csv('FE_day.csv')

print('=' * 50)
print('数据集加载与预检')
print('=' * 50)

print(f'样本数量: {df.shape[0]}')

X = df.drop(columns=['instant', 'cnt'])
print(f'特征维度: {X.shape[1]}')

print(f'标签: cnt (回归目标，1个连续值列)')
print(f'标签值范围: [{df["cnt"].min()}, {df["cnt"].max()}]')

print('\n前5个样本:')
print(df.head(5))
