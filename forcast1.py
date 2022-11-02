import nltk
import re
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import lightgbm as lgb
import scipy.sparse as sp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, classification_report
from urllib.parse import unquote, urlparse, quote
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

# 数据读取与拼接
train = pd.read_csv('dataTrain.csv')
test = pd.read_csv('dataA.csv')
data = pd.concat([train, test]).reset_index(drop=True) #让数据在竖直方向上面拼接，后面的reset_index是让索引不仅是简单的拼接，而是按照顺寻进行数值 变化
data['f3'] = data['f3'].map({'low': 0, 'mid': 1, 'high': 2})#将f3列的low换成0，mid换成1，high换成2

# 暴力Feature 位置这里代表的是构造特征值
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for i in range(len(loc_f)):
    for j in range(i + 1, len(loc_f)):
        data[f'{loc_f[i]}+{loc_f[j]}'] = data[loc_f[i]] + data[loc_f[j]]
        data[f'{loc_f[i]}-{loc_f[j]}'] = data[loc_f[i]] - data[loc_f[j]]
        data[f'{loc_f[i]}*{loc_f[j]}'] = data[loc_f[i]] * data[loc_f[j]]
        data[f'{loc_f[i]}/{loc_f[j]}'] = data[loc_f[i]] / data[loc_f[j]]

# 暴力Feature 通话
com_f = ['f43', 'f44', 'f45', 'f46']
for i in range(len(com_f)):
    for j in range(i + 1, len(com_f)):
        data[f'{com_f[i]}+{com_f[j]}'] = data[com_f[i]] + data[com_f[j]]
        data[f'{com_f[i]}-{com_f[j]}'] = data[com_f[i]] - data[com_f[j]]
        data[f'{com_f[i]}*{com_f[j]}'] = data[com_f[i]] * data[com_f[j]]
        data[f'{com_f[i]}/{com_f[j]}'] = data[com_f[i]] / data[com_f[j]]

# 训练测试分离
train = data[~data['label'].isna()].reset_index(drop=True)   #选取了label值不等于空的数据集
test = data[data['label'].isna()].reset_index(drop=True)    #选取了label值等于空的数据集

features = [i for i in train.columns if i not in ['label',  'id']] #特征在train训练集中非lable，id的下面
y = train['label']   #获取label值不等于空的label列表
#https://www.cnblogs.com/pinard/p/5992719.html --这里是交叉验证，适用于数据集较小的情况
"""
使用sklearn包中的StratifiedKFold可以将数据按规定方式划分成任意块数，然后实现交叉验证。

但这个函数本身是不能返回交叉验证的分数的。他只是切分数据。

切分数据的规则是train set 和 test set每种类别比例和总的样本每种类别比例相同。
"""
#StratifiedKFold方法是根据标签中不同类别占比来进行拆分数据的。
"""
1.每次划分成五个部分
2.每次划分的时候进行洗牌
3.随机数种子是2021
4.注意这里的划分不是均分
5.如果十行数据划分成三个部分，那么最后一个部分可以是4行，且其他部分都是3行
本质上：
for train, test in sKFold.split(X, y):
    print('train index:\n',train)
    print('train X value:\n',X[train])
    print('train y value:\n',y[train])
    print('test index:\n',test)
    print('test X value:\n',X[test])
    print('test y value:\n',y[test])
    print()
如果有这样的话，里面取得是索引，代表的是行数
其中如果是四份会在X中选三份，y中选一份
"""
KF = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
feat_imp_df = pd.DataFrame({'feat': features, 'imp': 0})#去除标签，id的数据集放在这个dataframe中
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'n_jobs': 30,
    'learning_rate': 0.05,
    'num_leaves': 2 ** 6,
    'max_depth': 8,
    'tree_learner': 'serial',
    'colsample_bytree': 0.8,
    'subsample_freq': 1,
    'subsample': 0.8,
    'num_boost_round': 5000,
    'max_bin': 255,
    'verbose': -1,
    'seed': 2021,
    'bagging_seed': 2021,
    'feature_fraction_seed': 2021,
    'early_stopping_rounds': 100,
 }

oof_lgb = np.zeros(len(train))       #创建一个和训练集大小相同的np数组
predictions_lgb = np.zeros((len(test))) #创建一个测试集大小相同的np数组

# 模型训练
for fold_, (trn_idx, val_idx) in enumerate(KF.split(train.values, y.values)):     #这个里面取得是train中没有标签的数据，取了不等于空的lable列表作为y代表测试集
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features], label=y.iloc[val_idx])
    num_round = 3000
    clf = lgb.train(
        params,
        trn_data,
        num_round,
        valid_sets=[trn_data, val_data],
        verbose_eval=100,
        early_stopping_rounds=50,
    )

    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    predictions_lgb[:] += clf.predict(test[features], num_iteration=clf.best_iteration) / 5
    feat_imp_df['imp'] += clf.feature_importance() / 5

print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
print("F1 score: {}".format(f1_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Precision score: {}".format(precision_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))
print("Recall score: {}".format(recall_score(y, [1 if i >= 0.5 else 0 for i in oof_lgb])))

# 提交结果
test['label'] = predictions_lgb
test[['id', 'label']].to_csv('FX_sub_8705.csv', index=False)