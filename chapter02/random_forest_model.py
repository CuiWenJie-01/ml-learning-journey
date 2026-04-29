import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin


# =========================
# 1. 读取数据
# =========================
housing = pd.read_csv("datasets/housing/housing.csv")


# =========================
# 2. 划分训练集
# =========================
train_set, test_set = train_test_split(
    housing,
    test_size=0.2,
    random_state=42
)

housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()


# =========================
# 3. 特征分类
# =========================
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]


# =========================
# 4. 自定义特征工程
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
# 5. 数值 Pipeline
# =========================
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("scaler", StandardScaler())
])


# =========================
# 6. ColumnTransformer
# =========================
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])


# =========================
# 7. 数据预处理
# =========================
housing_prepared = full_pipeline.fit_transform(housing)


# =========================================================
# 8. ================== 随机森林模型 ======================
# =========================================================

forest_reg = RandomForestRegressor(
    
    n_estimators=100,
    # 🌳 树的数量（非常重要）
    #
    # 👉 随机森林 = 多棵决策树的集合
    #
    # 这里表示：
    #   训练 100 棵决策树
    #
    # 📌 影响：
    # - 树越多 → 模型越稳定（方差下降）
    # - 但训练越慢
    #
    # 👉 一般经验：
    # 100 ~ 500 常用

    random_state=42,
    # 🎲 随机种子（保证结果可复现）
    #
    # 👉 随机森林内部有很多随机过程：
    # - 随机抽样数据（bootstrap）
    # - 随机选择特征
    #
    # 如果不固定 random_state：
    #   每次训练结果都会不同
    #
    # 👉 42 只是一个“约定俗成的种子值”

    n_jobs=-1
    # ⚙️ 并行计算核心参数（性能关键）
    #
    # 👉 表示使用 CPU 的所有核心来训练
    #
    # n_jobs = 并行线程数
    #
    # -1 = 使用全部 CPU 核心
    #
    # 📌 作用：
    # - 加快训练速度（尤其是 100+ 棵树时）
    #
    # 👉 如果是 8 核 CPU：
    #    会同时训练多棵树
)


# =========================
# 9. 训练模型
# =========================
forest_reg.fit(housing_prepared, housing_labels)


# =========================
# 10. 训练集评估
# =========================
forest_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, forest_predictions)
forest_rmse = np.sqrt(forest_mse)

print("=== Random Forest (Train Set) ===")
print("RMSE:", forest_rmse)


# =========================
# 11. 交叉验证评估
# =========================
forest_scores = cross_val_score(
    forest_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10
)

forest_rmse_scores = np.sqrt(-forest_scores)


# =========================
# 12. 输出交叉验证结果
# =========================
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


print("\n=== Random Forest (Cross Validation) ===")
display_scores(forest_rmse_scores)