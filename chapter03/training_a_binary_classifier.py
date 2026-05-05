import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml #fetch_openml用于获取开放机器学习数据集。

# 加载 MNIST 数据集
# 784是28×28的图片，784个像素点，每个像素点是一个特征
mnist=fetch_openml('mnist_784', version=1, as_frame=False)

# 拆分数据
# 将数据分为特征矩阵X和标签向量y
X, y = mnist["data"], mnist["target"]

# 标签转换为整数
y=y.astype(np.uint8)

#print(type(y[0]),y[:5])

# 拆分训练集和测试集
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# 打乱训练集
# MNIST 数据是按数字排序的
# 如果不打乱，模型会学偏（比如前面全是0）
shuffle_index=np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# 构造“是否为5”的标签（核心步骤）
'''
SGDClassifier 是什么？
简单理解：
    一个线性分类器
    用**随机梯度下降（SGD）**训练
    非常适合：
    大数据（MNIST 7万条）
    稀疏特征
'''
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# 训练一个分类器（SGDClassifier）
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# 用模型预测一个数字
some_digit = X[36000]
#print(sgd_clf.predict([some_digit]))

# 使用交叉验证计算 Accuracy
from sklearn.model_selection import cross_val_score
scores=cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# print(scores)
# print("Accuracy:", np.mean(scores))
'''
| 方法                | 输出       |
| ----------------- | -------- |
| cross_val_score   | 分数       |
| cross_val_predict | 每个样本的预测值 |
'''
# 使用交叉验证计算 Accuracy
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# 计算混淆矩阵
from sklearn.metrics import confusion_matrix
conf_mx=confusion_matrix(y_train_5, y_train_pred)
# print(conf_mx)

# 可视化混淆矩阵
# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

# Precision & Recall
from sklearn.metrics import precision_score, recall_score
# 精确率
precision=precision_score(y_train_5, y_train_pred)
# 召回率
recall=recall_score(y_train_5, y_train_pred)

# print("Precision:", precision)
# print("Recall:", recall)

# F1 Score
from sklearn.metrics import f1_score
f1=f1_score(y_train_5, y_train_pred)
# print("F1:", f1)

# ROC Curve
from sklearn.metrics import roc_curve
y_scores=sgd_clf.decision_function(X_train)
fpr, tpr, thresholds=roc_curve(y_train_5, y_scores)
# plt.plot(fpr, tpr, linewidth=2,label="ROC curve")
# plt.plot([0,1], [0,1], 'k--', label="Random classifier's ROC curve")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate (Recall)")
# plt.legend()
# plt.show()

# AUC Score（ROC 曲线面积）
from sklearn.metrics import roc_auc_score
roc_auc=roc_auc_score(y_train_5, y_scores)
# print("AUC:", roc_auc)

# Multiclass Classification（多类别分类）
# One-vs-Rest（OvR，一对多）
# from sklearn.linear_model import SGDClassifier

# sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train)
# prediction = sgd_clf.predict([X_train[0]])
# print("Predicted class for first training sample:", prediction)
# print("Actual class for first training sample:", y_train[0])

# One-vs-One（OvO，一对一）
# from sklearn.multiclass import OneVsOneClassifier
# ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
# ovo_clf.fit(X_train, y_train)
# prediction = ovo_clf.predict([X_train[0]])
# print("Predicted class for first training sample:", prediction)
# print("Actual class for first training sample:", y_train[0])


# Error Analysis（误差分析）
## 1. 混淆矩阵
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train, cv=3)

conf_mx = confusion_matrix(y_train, y_train_pred)
# print(conf_mx)

# 可视化混淆矩阵
import matplotlib.pyplot as plt

# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

# 标准化混淆矩阵
import numpy as np

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# 去掉对角线（只看错误）
np.fill_diagonal(norm_conf_mx, 0)

plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()