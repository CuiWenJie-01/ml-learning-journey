import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# 1. 读取数据
housing = pd.read_csv("datasets/housing/housing.csv")

# 2. 查看文本属性
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))

# 3.使用OneHotEncoder进行独热编码
cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

print("One-hot编码结果（稀疏矩阵）：")
print(housing_cat_1hot[:10]) # 这是一个稀疏矩阵

# 4.查看类别列表
print("类别列表：")
print(cat_encoder.categories_)
