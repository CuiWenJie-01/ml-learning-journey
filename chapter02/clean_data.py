import pandas as pd
from sklearn.impute import SimpleImputer # 处理缺失值

# 1. 读取数据
housing = pd.read_csv("datasets/housing/housing.csv")

# 2. 缺失值检查
print(housing.isnull().sum()) # 统计每列的缺失值数量

# 3. 处理缺失值（这里我们用中位数填充,Scikit-Learn方法）
# 去掉文本特征
housing_num=housing.drop("ocean_proximity", axis=1) # axis=1表示列

# 创建填充器（使用中位数填充）
imputer = SimpleImputer(strategy="median")

# 计算每一列的中位数
imputer.fit(housing_num) #fit只能在训练集上计算

# 查看每列中位数
print("各特征中位数：")
print(imputer.statistics_)

# 4 转换数据（真正填充）
X=imputer.transform(housing_num)

# 转回 DataFrame
housing_tr=pd.DataFrame(X, columns=housing_num.columns)

# 5. 检查填充结果
print(housing_tr.isnull().sum()) # 确认缺失值已被填充   0
print(housing_tr.head())