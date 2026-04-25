import numpy as np
import pandas as pd

def shuffle_and_split_data(data, test_ratio, rng):
   # 1. 随机打乱索引
    shuffled_indices = rng.permutation(len(data))
    
    # 2. 计算测试集大小
    test_set_size = int(len(data) * test_ratio)
    
    # 3. 切分索引
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    # 4. 按索引取数据
    return data.iloc[train_indices], data.iloc[test_indices]

df=pd.read_csv('datasets/housing/housing.csv')
rng=np.random.RandomState(42) # 设置随机数种子(保证每次结果都一样)
# 切分数据（80%训练，20%测试）
train_set, test_set = shuffle_and_split_data(df, test_ratio=0.2, rng=rng)

print(f"训练集大小: {len(train_set)}")
print(f"测试集大小: {len(test_set)}")