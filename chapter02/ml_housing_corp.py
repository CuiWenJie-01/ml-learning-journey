import pandas as pd

df = pd.read_csv('datasets/housing/housing.csv')


#print(df.head()) # 看前5行
#print(df.info()) # 查看数据集的基本信息，包括每列的数据类型和非空值数量

#print(df["ocean_proximity"].value_counts()) # 统计 ocean_proximity 列中每个类别的数量

print(df.describe()) # 计算数值型列的统计信息，包括计数、平均值、标准差、最小值、25%分位数、中位数（50%分位数）、75%分位数和最大值