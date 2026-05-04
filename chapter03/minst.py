import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml #fetch_openml用于获取开放机器学习数据集。

# 加载 MNIST 数据集
# 784是28×28的图片，784个像素点，每个像素点是一个特征
mnist=fetch_openml('mnist_784', version=1, as_frame=False)

# 拆分数据
# 将数据分为特征矩阵X和标签向量y
X, y = mnist["data"], mnist["target"]

print(X.shape)
print(y.shape)

# 数据类型处理（必须）
# 将标签从字符串类型转换为无符号8位整数(uint8)，便于后续处理
y=y.astype(np.uint8)

# 看其中一个样本
some_digit=X[36000]

# 将 1D → 2D 图像
some_digit_image=some_digit.reshape(28, 28)

# 可视化这个数字
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

# 查看这个数字的标签
print("label:",y[36000])


# 看前 10 张图片（理解数据结构）
def plot_digits(instances, images_per_row=5):
    images = [instance.reshape(28, 28) for instance in instances]
    
    for i in range(len(instances)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i], cmap="binary")
        plt.axis("off")

plt.figure(figsize=(10, 4))
plot_digits(X[:10])
plt.show()