import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取房价数据集
housing = pd.read_csv('datasets/housing/housing.csv')


# 绘制散点图（地理位置 + 房价 + 人口）
housing.plot(
    kind="scatter",
    x="longitude",
    y="latitude",
    alpha=0.4,
    s=housing["population"] / 100,   # 点大小：人口
    label="population",
    figsize=(10, 7),
    c="median_house_value",          # 颜色：房价
    cmap=plt.get_cmap("jet"),        # 颜色映射
    colorbar=True
)

# 图例
plt.legend()

# 标题
plt.title("California Housing Prices Visualization")

# 显示图像
plt.show()