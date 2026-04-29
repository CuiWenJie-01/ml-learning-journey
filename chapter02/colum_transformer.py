# ===============================
# 1. 导入库
# ===============================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin


# ===============================
# 2. 读取数据
# ===============================
housing = pd.read_csv("datasets/housing/housing.csv")


# ===============================
# 3. 划分训练集和测试集（重要！避免数据泄露）
# ===============================
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# ===============================
# 4. 分离标签
# ===============================
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()


# ===============================
# 5. 数值列 & 类别列
# ===============================
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]


# ===============================
# 6. 自定义特征工程
# ===============================
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household,
                         population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household,
                         population_per_household]


# ===============================
# 7. 数值 Pipeline
# ===============================
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])


# ===============================
# 8. ColumnTransformer（核心）
# ===============================
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),     # 数值特征
    ("cat", OneHotEncoder(), cat_attribs),  # 类别特征
])


# ===============================
# 9. 执行转换
# ===============================
housing_prepared = full_pipeline.fit_transform(housing)


# ===============================
# 10. 查看结果
# ===============================
print(housing_prepared.shape)