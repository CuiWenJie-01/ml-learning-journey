import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin


# =========================
# 1. 读取数据
# =========================
housing = pd.read_csv("datasets/housing/housing.csv")


# =========================
# 2. 划分训练/测试集
# =========================
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# =========================
# 3. 分离特征和标签
# =========================
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()


# =========================
# 4. 列分类
# =========================
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]


# =========================
# 5. 自定义特征工程
# =========================
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
            return np.c_[X,
                         rooms_per_household,
                         population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X,
                         rooms_per_household,
                         population_per_household]


# =========================
# 6. 数值 pipeline
# =========================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("scaler", StandardScaler())
])


# =========================
# 7. 完整预处理
# =========================
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])


# =========================
# 8. 数据预处理
# =========================
housing_prepared = full_pipeline.fit_transform(housing)


# =========================================================
# 9. ================== 训练模型 ==========================
# =========================================================

# -------- 9.1 线性回归 --------
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# -------- 9.2 决策树 --------
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# =========================================================
# 10. ================== 模型评估 =========================
# =========================================================

# -------- 10.1 线性回归预测 --------
lin_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, lin_predictions)
lin_rmse = np.sqrt(lin_mse)

print("Linear Regression RMSE:", lin_rmse)


# -------- 10.2 决策树预测 --------
tree_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, tree_predictions)
tree_rmse = np.sqrt(tree_mse)

print("Decision Tree RMSE:", tree_rmse)


# =========================================================
# 11. ================== 看几个样本 =========================
# =========================================================

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print("\nLinear Regression Predictions:")
print(lin_reg.predict(some_data_prepared))

print("True Labels:")
print(list(some_labels))