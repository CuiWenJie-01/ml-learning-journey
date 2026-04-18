import matplotlib.pyplot as plt # plt：绘图库
import numpy as np # np：科学计算库
import pandas as pd # pd：数据分析库
from sklearn.linear_model import LinearRegression # LinearRegression：线性回归模型

# 数据集
data_root="https://raw.githubusercontent.com/ageron/data/main/lifesat/lifesat.csv"
lifesat=pd.read_csv(data_root+"lifesat/lifesat.csv")
x=lifesat[["GDP per capita"]].values # GDP per capita：人均GDP
y=lifesat[["Life satisfaction"]].values # Life satisfaction：生活满意度