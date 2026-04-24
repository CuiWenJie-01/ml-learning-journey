import matplotlib.pyplot as plt # plt：绘图库
import numpy as np # np：科学计算库
import pandas as pd # pd：数据分析库
from sklearn.linear_model import LinearRegression # LinearRegression：线性回归模型
#from sklearn.neighbors import KNeighborsRegressor # KNeighborsRegressor：K近邻回归模型

# 数据集
data_root="https://raw.githubusercontent.com/ageron/data/main/"
lifesat=pd.read_csv(data_root+"lifesat/lifesat.csv")

# 查看数据集的列名
print("数据集的列名:")
print(lifesat.columns.tolist())
print("\n数据集的前几行:")
print(lifesat.head())

x=lifesat[["GDP per capita (USD)"]].values # GDP per capita：人均GDP
y=lifesat[["Life satisfaction"]].values # Life satisfaction：生活满意度

# 绘制散点图，展示人均GDP与生活满意度之间的关系
lifesat.plot(kind='scatter',grid=True,x="GDP per capita (USD)",y="Life satisfaction")
# 设置坐标轴范围：X轴从23500到62500（人均GDP），Y轴从4到9（生活满意度）
plt.axis([23_500,62_500,4,9])
# 显示绘制的图形
plt.show()

model=LinearRegression() # 创建线性回归模型对象
#model=KNeighborsRegressor(n_neighbors=3) # 创建K近邻回归模型对象
model.fit(x,y) # 使用数据集中的人均GDP和生活满意度进行模型训练

x_new=[[33_442.8]] # 新的人均GDP数据点
print(model.predict(x_new)) # 使用训练好的模型预测新数据点的生活满意度，并打印结果