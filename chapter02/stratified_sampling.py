# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np   # 用于数值计算，特别是处理无穷大值
from sklearn.model_selection import StratifiedShuffleSplit  # 用于进行分层随机分割数据集

# 从CSV文件中读取房价数据集
housing = pd.read_csv('datasets/housing/housing.csv')

# 创建数据集副本以避免修改原始数据
housing = housing.copy() 

# 基于收入中位数创建收入类别列
# 将收入范围划分为5个区间，并分配标签1-5
housing['income_cat'] = pd.cut(housing['median_income'],
                            bins=[0., 1.5, 3.0, 4.5, 6., np.inf],  # 定义收入区间的边界
                            labels=[1, 2, 3, 4, 5])                # 为每个区间分配标签

# 初始化分层随机分割器
# n_splits=1 表示只进行一次分割
# test_size=0.2 表示测试集占总数据的20%
# random_state=42 确保每次运行结果一致（可重现）
split = StratifiedShuffleSplit(
    n_splits=1, test_size=0.2, random_state=42)

# 执行分层分割，获得训练集和测试集的索引
for train_index, test_index in split.split(housing, housing['income_cat']):
    # 根据索引获取训练集和测试集
    strat_train_set = housing.loc[train_index].copy()  # 训练集
    strat_test_set = housing.loc[test_index].copy()    # 测试集

# 创建对比表，比较总体和分层抽样的分布
compare_props = pd.DataFrame({
    "Overall": housing['income_cat'].value_counts(normalize=True),      # 整体数据集中各收入类别的比例
    "Stratified": strat_test_set['income_cat'].value_counts(normalize=True),  # 分层抽样后测试集中各收入类别的比例
}).sort_index()  # 按类别标签排序

# 打印对比结果，验证分层抽样的效果
print(compare_props)

# 从训练集和测试集中删除临时的收入类别列，因为后续不再需要
for dataset in (strat_train_set, strat_test_set):
    dataset.drop('income_cat', axis=1, inplace=True)  # 删除income_cat列，axis=1表示按列操作，inplace=True表示直接在原数据上修改