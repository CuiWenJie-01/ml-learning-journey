# 从sklearn库导入生成月牙形数据集的函数
from sklearn.datasets import make_moons
# 从sklearn库导入训练测试集分割函数
from sklearn.model_selection import train_test_split

# 生成月牙形样本数据，包含100个样本，随机打乱，噪声为10
X, y = make_moons(n_samples=100, shuffle=True, noise=10)

# 将数据分割成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 从sklearn库导入多个机器学习模型
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC

# 初始化三个不同的分类器
log_clf = LogisticRegression()           # 逻辑回归分类器
rnd_clf = RandomForestClassifier()      # 随机森林分类器
svm_clf = SVC()                         # 支持向量机分类器

# 创建投票分类器，使用硬投票方式组合三个基础分类器
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], 
    voting='hard'
)

# 使用训练集数据训练投票分类器
voting_clf.fit(X_train, y_train)

# 导入准确率评估指标
from sklearn.metrics import accuracy_score 

# 对每个分类器进行训练和预测，并输出其在测试集上的准确率
for clf in (log_clf, rnd_clf, svm_clf, voting_clf): 
    # 在训练集上训练分类器
    clf.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = clf.predict(X_test) 
    # 输出分类器名称及其准确率
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


