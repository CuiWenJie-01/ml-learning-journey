import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin


# =========================
# 1. 读取数据
# =========================
housing = pd.read_csv("datasets/housing/housing.csv")


# =========================
# 2. 划分训练集
# =========================
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()


# =========================
# 3. 列分类
# =========================
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]


# =========================
# 4. 特征工程
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
# 8. ================== 模型定义 ==========================
# =========================================================

tree_reg = DecisionTreeRegressor(random_state=42)
lin_reg = LinearRegression()


# =========================================================
# 9. ================== 交叉验证 ==========================
# =========================================================

# -------------------------
# 9.1 Decision Tree CV
# -------------------------
# 使用交叉验证评估决策树模型的性能
tree_scores = cross_val_score(
    
    tree_reg,  
    # 👆 模型（Estimator）
    # 这里是 DecisionTreeRegressor
    # cross_val_score 会“反复训练这个模型”

    housing_prepared,  
    # 👆 特征数据（X）
    # 已经经过 Pipeline 处理后的“干净数据”
    # shape = (16512, 16)

    housing_labels,  
    # 👆 标签（y）
    # 房价（median_house_value）

    scoring="neg_mean_squared_error",  
    # 👆 评分函数（非常关键）
    # 表示：使用 MSE（均方误差）来评估模型
    #
    # ⚠️ 注意：
    # sklearn 规定“评分必须是越大越好”
    # 但 MSE 是“越小越好”
    #
    # 👉 所以这里返回的是：
    #     score = -MSE（负的均方误差）
    #
    # 👉 后面要手动转回来：
    #     RMSE = sqrt(-score)

    cv=10  
    # 👆 交叉验证折数（K-fold）
    #
    # 表示：把数据分成 10 份（10 folds）
    #
    # 过程：
    # 第1次：第1份做验证，其余9份训练
    # 第2次：第2份做验证，其余9份训练
    # ...
    # 第10次：第10份做验证
    #
    # 👉 最终会训练 10 次模型，并得到 10 个分数
)

tree_rmse_scores = np.sqrt(-tree_scores)


# -------------------------
# 9.2 Linear Regression CV
# -------------------------
lin_scores = cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10
)

lin_rmse_scores = np.sqrt(-lin_scores)


# =========================================================
# 10. ================== 结果输出函数 ======================
# =========================================================

# 定义一个函数来显示交叉验证的分数
def display_scores(scores):
    print("Scores:", scores) # 打印分数
    print("Mean:", scores.mean()) # 打印平均值
    print("Standard deviation:", scores.std()) # 打印标准差


print("=== Decision Tree ===")
display_scores(tree_rmse_scores)

print("\n=== Linear Regression ===")
display_scores(lin_rmse_scores)