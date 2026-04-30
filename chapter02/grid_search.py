import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
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
# 3. 数值 / 类别特征
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
# 8. 随机森林模型
# =========================================================
forest_reg = RandomForestRegressor(random_state=42)


# =========================================================
# 9. 超参数搜索空间（Grid Search）
# =========================================================
param_grid = [

    # =========================================================
    # 🔵 第一组：基础随机森林参数组合（默认 bootstrap=True）
    # =========================================================
    {
        "n_estimators": [3, 10, 30],
        # 🌳 n_estimators = 决策树的数量
        #
        # 👉 表示训练多少棵树组成森林
        #
        # 这里尝试三种情况：
        #   - 3棵树（模型很弱，容易不稳定）
        #   - 10棵树（中等规模）
        #   - 30棵树（更稳定，但更慢）

        "max_features": [2, 4, 6, 8]
        # 🌿 max_features = 每棵树在“分裂节点时”最多使用多少个特征
        #
        # 👉 控制“每棵树看的视野大小”
        #
        # 数值越小：
        #   ✔ 树之间差异更大（更随机）
        #   ✔ 更不容易过拟合
        #
        # 数值越大：
        #   ✔ 每棵树更强
        #   ❌ 但树之间更相似
    },


    # =========================================================
    # 🔴 第二组：关闭 bootstrap 的情况
    # =========================================================
    {
        "bootstrap": [False],
        # 🚫 bootstrap = 是否使用“有放回抽样”
        #
        # False 表示：
        #   👉 每棵树使用“完整训练数据”
        #   👉 不做随机抽样
        #
        # 影响：
        #   ❌ 树之间相似度更高
        #   ❌ 随机性降低
        #   ✔ 可能在某些数据集上表现更好

        "n_estimators": [3, 10],
        # 🌳 树的数量（这里只测试较小规模）
        #
        # 👉 因为 bootstrap=False 时计算更“重”，
        #    所以只测试较少树数（3 / 10）

        "max_features": [2, 3, 4]
        # 🌿 每次分裂使用的特征数量
        #
        # 👉 这里范围比第一组更小
        # 👉 因为已经关闭随机抽样，
        #    需要通过控制特征来增加随机性
    }

]


# =========================================================
# 10. Grid Search
# =========================================================
grid_search = GridSearchCV(
    estimator=forest_reg, # 模型
    param_grid=param_grid, # 参数搜索空间
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True, # 返回训练集得分
    n_jobs=-1
)


# =========================================================
# 11. 训练（核心步骤）
# =========================================================
grid_search.fit(housing_prepared, housing_labels)


# =========================================================
# 12. 最优参数
# =========================================================
#print("Best Parameters:", grid_search.best_params_)


# =========================================================
# 13. 最优模型
# =========================================================
best_model = grid_search.best_estimator_


# =========================================================
# 14. 在训练集上评估最优模型
# =========================================================
final_predictions = best_model.predict(housing_prepared)

final_mse = mean_squared_error(housing_labels, final_predictions)
final_rmse = np.sqrt(final_mse)

#print("Train RMSE (Best Model):", final_rmse)



# 特征重要性
feature_importances = grid_search.best_estimator_.feature_importances_

# =========================
# 数值特征名
# =========================
num_attribs = list(housing.drop("ocean_proximity", axis=1).columns)

# =========================
# 你自己加的组合特征
# =========================
extra_attribs = [
    "rooms_per_household",
    "population_per_household",
    "bedrooms_per_room"
]

# =========================
# One-Hot 编码后的分类特征
# =========================
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(
    cat_encoder.get_feature_names_out(["ocean_proximity"])
)

# =========================
# 所有特征拼接
# =========================
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

# =========================
# 排序输出（从重要到不重要）
# =========================
sorted_features = sorted(
    zip(feature_importances, attributes),
    reverse=True
)

# print("Feature Importances (sorted):")
# for score, name in sorted_features:
#     print(f"{name}: {score:.4f}")

# 从 GridSearchCV 中获取最优模型（在验证集上表现最好的模型）
final_model = grid_search.best_estimator_

# 从测试集提取特征数据（X）
# axis=1 表示按列删除 target 变量
X_test = test_set.drop("median_house_value", axis=1)

# 从测试集提取真实标签（y）
# copy() 防止后续修改影响原始数据
y_test = test_set["median_house_value"].copy()

# 对测试集特征进行预处理（与训练时完全一致）
# ⚠️ 注意：这里只是 transform，不能 fit
# 因为所有统计量（均值、方差等）必须来自训练集，避免数据泄漏
X_test_prepared = full_pipeline.transform(X_test)

# 使用最优模型对测试集进行预测
# 这里输入的是已经预处理后的特征
final_predictions = final_model.predict(X_test_prepared)

# 计算均方误差（MSE）
# 衡量预测值与真实值之间的平均平方差
final_mse = mean_squared_error(y_test, final_predictions)

# 将 MSE 转换为 RMSE（均方根误差）
# RMSE 更直观，因为与原始标签单位一致
final_rmse = np.sqrt(final_mse)

# 输出最终模型在测试集上的表现
print("Test RMSE (Best Model):", final_rmse)