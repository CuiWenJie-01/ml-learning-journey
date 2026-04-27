import pandas as pd
import numpy as np

# 1. 读取数据
housing = pd.read_csv("datasets/housing/housing.csv")

# 2. 只保留数值型特征（非常重要）
housing_num=housing.select_dtypes(include=[np.number]) 

# 3. 计算相关系数矩阵
corr_matrix = housing_num.corr()

# 4. 查看与房价的相关性排序
print(corr_matrix["median_house_value"].sort_values(ascending=False))