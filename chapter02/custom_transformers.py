# 导入必要的库
import numpy as np          # 用于数值计算
import pandas as pd         # 用于数据处理和分析
from sklearn.base import BaseEstimator, TransformerMixin  # 用于创建自定义转换器的基础类

# 读取房价数据集
housing = pd.read_csv("datasets/housing/housing.csv")

# 定义列索引，对应数据集中的不同特征
# rooms_ix, bedrooms_ix, population_ix, households_ix 分别是房间数、卧室数、人口数、家庭数的列索引
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

# 自定义特征转换器类，继承自BaseEstimator和TransformerMixin
# BaseEstimator提供get_params和set_params方法
# TransformerMixin提供fit_transform方法
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # 初始化方法，没有*args或**kwargs参数
        # 构造函数，设置是否添加卧室与房间比例的选项
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self, X, y=None):
        # 拟合方法，对于这个转换器来说不需要实际的拟合操作
        # 返回自身实例，符合sklearn转换器的标准接口
        return self  # 不需要执行其他操作
    
    def transform(self, X):
        # 转换方法，对输入数据进行特征工程
        # 计算每户平均房间数：总房间数除以总家庭数
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        
        # 计算每户平均人口数：总人口数除以总家庭数
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        # 如果设置了添加卧室与房间比例的选项
        if self.add_bedrooms_per_room:
            # 计算卧室与房间的比例：卧室数除以房间数
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            # 使用np.c_将原始特征与新计算的特征连接起来
            # 返回包含原始特征及三个新特征的数据集
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            # 如果不添加卧室与房间比例，则只返回原始特征和前两个新特征
            return np.c_[X, rooms_per_household, population_per_household]

# 创建转换器实例，设置不添加卧室与房间比例
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)


# 对房价数据集应用转换器，将pandas DataFrame转换为numpy数组后再进行变换
housing_extra_attribs = attr_adder.transform(housing.values)

# 打印转换后的数据，显示添加新特征后的结果
print(housing_extra_attribs)