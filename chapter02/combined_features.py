import pandas as pd
import numpy as np

# 1. 读取数据
housing = pd.read_csv("datasets/housing/housing.csv")

# 2. 复制数据（避免污染原始数据）
housing = housing.copy()

# 3. 创建组合特征（Feature Engineering）
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

# 4. 处理可能的缺失值（非常重要！）
housing=housing.dropna()

# 5. 计算相关性矩阵（只对数值特征）
corr_matrix=housing.corr(numeric_only=True)

# 6. 查看与房价的相关性排序
print(corr_matrix["median_house_value"].sort_values(ascending=False))
