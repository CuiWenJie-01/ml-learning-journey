# 从sklearn库导入生成月牙形数据集的函数
from sklearn.datasets import make_moons
# 从sklearn库导入训练测试集分割函数
from sklearn.model_selection import train_test_split

# 生成月牙形样本数据，包含100个样本，随机打乱，噪声为1.0
X, y = make_moons(n_samples=100, shuffle=True, noise=1.0, random_state=42)

# 将数据分割成训练集和测试集（默认75%训练，25%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 打印训练集大小以确认
print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 导入Bagging分类器和决策树分类器
from sklearn.ensemble import BaggingClassifier 
from sklearn.tree import DecisionTreeClassifier

# 创建Bagging分类器，使用决策树作为基分类器，并启用OOB评估
# n_estimators=500表示使用500个决策树
# max_samples设置为训练集大小或者较小的数值
# oob_score=True启用袋外评估
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), 
    n_estimators=500,        
    max_samples=min(len(X_train), 100),  # 限制在训练集大小范围内
    bootstrap=True, 
    n_jobs=-1,
    oob_score=True,  # 启用袋外评估
    random_state=42
)

# 在训练集上训练Bagging分类器
bag_clf.fit(X_train, y_train) 

# 在测试集上进行预测
y_pred = bag_clf.predict(X_test)

# 添加输出部分，显示模型性能
from sklearn.metrics import accuracy_score
print("Bagging分类器的准确率:", accuracy_score(y_test, y_pred))

# 输出袋外评估分数
print("袋外评估准确率 (OOB Score):", bag_clf.oob_score_)

# 可选：也可以单独训练一个决策树分类器进行对比
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_single = tree_clf.predict(X_test)
print("单个决策树的准确率:", accuracy_score(y_test, y_pred_single))

# 显示预测结果的前几个值（可选）
print("Bagging分类器的预测结果（前10个）:", y_pred[:10])
print("真实标签（前10个）:", y_test[:10])

# 如果需要查看袋外决策函数（每个样本的类别概率）
print("\n袋外决策函数形状:", bag_clf.oob_decision_function_.shape)
print("前5个样本的袋外决策概率:")
print(bag_clf.oob_decision_function_[:5])

# 配置matplotlib以支持中文字体
import matplotlib
import matplotlib.pyplot as plt
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

import numpy as np

# 定义绘制决策边界的函数
def plot_decision_boundary(model, X, y, title):
    """绘制分类器的决策边界"""
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    # 预测网格点
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和散点图
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black', s=30)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("$x_1$", fontsize=12)
    plt.ylabel("$x_2$", fontsize=12)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

# 创建图形（与官方示例风格一致）
plt.figure(figsize=(12, 5))

# 绘制单个决策树的决策边界
plt.subplot(1, 2, 1)
plot_decision_boundary(tree_clf, X, y, "Decision Tree")

# 绘制Bagging分类器的决策边界
plt.subplot(1, 2, 2)
plot_decision_boundary(bag_clf, X, y, "Decision Trees with Bagging")

plt.tight_layout()
plt.show()

# 比较预测结果
plt.figure(figsize=(12, 5))

# 绘制Bagging分类器的预测结果
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.RdYlBu, edgecolors='black', s=30)
plt.title(f"Bagging分类器预测结果 (准确率: {accuracy_score(y_test, y_pred):.2f})", fontsize=12)
plt.xlabel("$x_1$", fontsize=10)
plt.ylabel("$x_2$", fontsize=10)

# 绘制真实标签
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors='black', s=30)
plt.title("真实标签", fontsize=12)
plt.xlabel("$x_1$", fontsize=10)
plt.ylabel("$x_2$", fontsize=10)

plt.tight_layout()
plt.show()