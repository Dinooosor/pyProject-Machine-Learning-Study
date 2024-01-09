import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
import scienceplots#运行有问题就加这个

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 24

data = sns.load_dataset('titanic')  # 导入泰坦尼克号生还数据
data0 = data.copy()#备份原始数据

data.info()#查看数据特征
#查看年龄数据分布
# with plt.style.context(['science','ieee','grid','no-latex']):
#     plt.figure(figsize=(12,5))
#     plt.subplot(121)
#     data['age'].hist(bins=70)
#     plt.xlabel('Age')
#     plt.ylabel('Num')
#     plt.subplot(122)
#     data.boxplot(column='age', showfliers=False)
    # plt.savefig('fig4.png', dpi=300)



#看是否有缺失数值
for col in data:
    pct_missing = np.mean(data[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing,2)))

data.replace(to_replace=r'^\s*$', value=np.nan, regex=True, inplace=True)   # 把各类缺失类型统一改为NaN的形式
data.isnull().mean()
# let's see if there is any missing data

del data['who']
del data['adult_male']
del data['deck']
del data['class']
del data['alone']


"""
随机森林所有超参数
sklearn.ensemble.RandomForestClassifier (n_estimators=100, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                                         min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                         min_impurity_split=None, class_weight=None, random_state=None, bootstrap=True, oob_score=False, 
                                         n_jobs=None, verbose=0, warm_start=False)
"""
data['age'].fillna(np.mean(data.age), inplace=True)   # 年龄特征使用均值对缺失值进行填补
data['embarked'].fillna(data['embarked'].mode(dropna=False)[0], inplace=True)   # 文本型特征视同众数进行缺失值填补

x = data.drop(['alive','survived','embark_town'],axis=1)# 取出用于建模的特征列X,这三列无了
label = data['survived']#取出标签列

oe = OrdinalEncoder()   # 定义特征转化函数
 # 把需要转化的特征都写进去
x[['sex', 'embarked']] = oe.fit_transform(x[['sex', 'embarked']])

# 划分训练集、测试集
xtrain, xtest, ytrain, ytest = train_test_split(x, label, test_size=0.3)

# 单颗决策树
clf = DecisionTreeClassifier(class_weight='balanced', random_state=37)
clf = clf.fit(xtrain, ytrain)  # 拟合训练集
score_c = clf.score(xtest, ytest)  # 输出测试集准确率

# 随机森林
rfc = RandomForestClassifier(class_weight='balanced', random_state=37)
rfc = rfc.fit(xtrain, ytrain)
score_r = rfc.score(xtest, ytest)


# 决策树 预测测试集
y_test_proba_clf = clf.predict_proba(xtest)
false_positive_rate_clf, recall_clf, thresholds_clf = roc_curve(ytest, y_test_proba_clf[:, 1])
# 决策树 AUC指标
roc_auc_clf = auc(false_positive_rate_clf, recall_clf)

# 随机森林 预测测试集
y_test_proba_rfc = rfc.predict_proba(xtest)
false_positive_rate_rfc, recall_rfc, thresholds_rfc = roc_curve(ytest, y_test_proba_rfc[:, 1])
# 随机森林 AUC指标
roc_auc_rfc = auc(false_positive_rate_rfc, recall_rfc)

# 画图 画出俩模型的ROC曲线
with plt.style.context(['science', 'ieee', 'grid', 'no-latex']):
    plt.plot(false_positive_rate_clf, recall_clf, color='blue', label='AUC_clf=%0.3f' % roc_auc_clf)
    plt.plot(false_positive_rate_rfc, recall_rfc, color='orange', label='AUC_rfc=%0.3f' % roc_auc_rfc)
    plt.legend(loc='best', fontsize=10, frameon=False)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.ylabel('Recall')
    plt.xlabel('Fall-out')
    plt.savefig('figure5.png',dpi=300)

# 定义空列表，用来存放每一个基学习器数量所对应的AUC值
superpa = []
# 循环200次
for i in range(200):
    rfc = ensemble.RandomForestClassifier(n_estimators=i + 1, class_weight='balanced', random_state=37, n_jobs=10)
    rfc = rfc.fit(xtrain, ytrain)  # 拟合模型

    y_test_proba_rfc = rfc.predict_proba(xtest)  # 预测测试集
    false_positive_rate_rfc, recall_rfc, thresholds_rfc = roc_curve(ytest, y_test_proba_rfc[:, 1])
    roc_auc_rfc = auc(false_positive_rate_rfc, recall_rfc)  # 计算模型AUC

    superpa.append(roc_auc_rfc)  # 记录每一轮的AUC值

print(max(superpa), superpa.index(max(superpa)))  # 输出最大的AUC值和其对应的轮数
with plt.style.context(['science', 'ieee', 'grid', 'no-latex']):
    plt.figure(figsize=[20, 5])
    plt.plot(range(1, 201), superpa)
    plt.savefig('figure6',dpi=300)