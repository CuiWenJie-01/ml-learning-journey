import numpy as np
import matplotlib.pyplot as plt

# 随机线性数据集
# 生成100个在[0,2]区间内的随机特征值X
# rand(100,1)生成100行1列的矩阵，每个值在[0,1]之间，乘以2后变为[0,2]区间
# X = 2*np.random.rand(100,1)

# # 生成目标值y，遵循线性关系y = 4 + 3*x + 噪声
# # 4是截距(intercept)，3是斜率(coefficient)，randn添加正态分布的噪声
# y = 4+3*X+np.random.randn(100,1)

# # 使用蓝色圆点绘制X和y的散点图
# plt.plot(X,y,"b.")

# # 设置坐标轴的显示范围：x轴[0,2]，y轴[0,15]
# plt.axis([0,2,0,15])

# 显示绘制的图形
# plt.show()

# 构造设计矩阵X_b，在原X矩阵前添加一列全为1的值，用于表示截距项
# 这样可以将截距项和斜率项统一到一个矩阵运算中
X_b = np.c_[np.ones((100,1)),X]

# 使用正规方程计算最优参数theta
# 正规方程公式：θ = (X^T * X)^(-1) * X^T * y
# 其中X_b是添加了截距项的特征矩阵，theta_best是使得成本函数最小的参数向量
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)

# 定义新的输入值点，用于绘制拟合直线
# 这里选择x=0和x=2两个点来绘制预测直线
X_new = np.array([[0],[2]])

# 对新输入值构造设计矩阵，同样添加截距项（全1列）
X_new_b = np.c_[np.ones((2,1)),X_new]

# 计算预测值，使用训练好的参数theta_best进行预测
# y_pred = X_new_b * theta_best，即对新数据进行线性回归预测
y_pred = X_new_b.dot(theta_best)
# print(y_pred)

# 绘制预测直线
# plt.plot(X_new,y_pred,"r-")
# plt.plot(X,y,"b.")
# plt.axis([0,2,0,15])
# plt.show()

# 使用Scikit-Learn 代码可以达到相同的效果(更简便)
from sklearn.linear_model import LinearRegression
# 创建线性回归模型实例
# lin_reg = LinearRegression()

# # 使用fit方法训练模型，拟合X和y之间的关系
# lin_reg.fit(X,y)

# # 输出模型的截距和系数，验证与正规方程结果的一致性
# lin_reg.intercept_, lin_reg.coef_
# print(lin_reg.predict(X_new))

#  批量梯度下降
# from IPython.display import clear_output
# eta = 0.1 # 学习率
# n_iter = 1000
# m = 100
# theta = np.random.randn(2,1)

# plt.figure(figsize=(8,6))
# plt.ion()# 打开交互模式
# plt.axis([0,2,0,15])
# plt.rcParams["font.sans-serif"] = "SimHei"

# for iter in range(n_iter):
#     plt.cla() # 清除原图像
#     gradients = 2/m*X_b.T.dot(X_b.dot(theta)-y)
#     theta = theta - eta*gradients
#     X_new = np.array([[0],[2]])
#     X_new_b = np.c_[np.ones((2,1)),X_new]
#     y_pred = X_new_b.dot(theta)
#     plt.plot(X,y,"b.")
#     plt.plot(X_new,y_pred,"r-")
#     plt.title("学习率：{:.2f}".format(eta))
#     plt.pause(0.1) # 暂停一会
#     clear_output(wait=True)# 刷新图像
# plt.ioff()# 关闭交互模式    
# # plt.show()
# theta

# 随机梯度下降
from sklearn.linear_model import SGDRegressor
# 创建SGD回归器实例，设置最大迭代次数为100，无正则化惩罚项，学习率为0.1
sgd_reg = SGDRegressor(max_iter=100, penalty=None, eta0=0.1)
# 使用随机梯度下降算法训练模型，其中y.ravel()将目标值数组展平成一维
sgd_reg.fit(X,y.ravel())
# 输出模型的截距项和系数，用于比较与正规方程、批量梯度下降等方法的结果
sgd_reg.intercept_, sgd_reg.coef_
# 使用训练好的SGD模型对新数据进行预测，并打印预测结果
# print(sgd_reg.predict(X_new))

#  polynomial regression 多项式回归
m = 100
X = 6*np.random.rand(m,1)-3
y = 0.5*X**2 + X + 2 + np.random.randn(m,1)
# plt.rcParams["axes.unicode_minus"] = False # 显示负号
# plt.plot(X, y, "g.")
# plt.axis([-3,3,0,10])
# plt.show()

# sklearn提供了PolynomialFeatures类，可以自动生成多项式特征
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, include_bias=False)
# help(PolynomialFeatures)
X_ploy = pf.fit_transform(X)
# print(X[0])
# print(X_ploy[0])

# 使用线性回归模型进行训练(拟合)
lin_reg = LinearRegression()
lin_reg.fit(X_ploy, y)
lin_reg.intercept_, lin_reg.coef_
# print("截距和系数:", lin_reg.intercept_, lin_reg.coef_)
plt.plot(X, y, "g.")
x = np.linspace(-3.5, 3.5, 500)
print(x.shape)
y_pred = lin_reg.intercept_ + lin_reg.coef_[0][0]*x + lin_reg.coef_[0][1]*x**2
plt.plot(x, y_pred, 'r-')
plt.show()