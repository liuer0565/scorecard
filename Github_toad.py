import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV as gscv
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import glob
import math
import xgboost as xgb
import toad
import matplotlib.pyplot as plt
plt.show()

#加载数据
f_dev = open('D:/0.学习/python/屁屁和铭仔的数据之路/TrainData.csv')
dev = pd.read_csv(f_dev, sep=',')
# print('dev',dev.head(5))
f_val = open('D:/0.学习/python/屁屁和铭仔的数据之路/TestData.csv')
off = pd.read_csv(f_val, sep=',')
# print('off',off.head(5))

# 描述性分析
a = toad.detector.detect(dev)

dev_slct1, drop_lst= toad.selection.select(dev,dev['SeriousDlqin2yrs'], empty = 0.7, iv = 0.02, corr = 0.7, return_drop=True)
print("keep:",dev_slct1.shape[1],
      "drop empty:",len(drop_lst['empty']),
      "drop iv:",len(drop_lst['iv']),
      "drop corr:",len(drop_lst['corr']))

dev_slct2, drop_lst= toad.selection.select(dev_slct1,dev_slct1['SeriousDlqin2yrs'], empty = 0.6, iv = 0.02, corr = 0.7, return_drop=True)
print("keep:",dev_slct2.shape[1],
      "drop empty:",len(drop_lst['empty']),
      "drop iv:",len(drop_lst['iv']),
      "drop corr:",len(drop_lst['corr']))

#得到切分节点
combiner = toad.transform.Combiner()
combiner.fit(dev_slct2,dev_slct2['SeriousDlqin2yrs'],method='chi',min_samples = 0.05)

#导出箱的节点
bins = combiner.export()

#根据节点实施分箱
dev_slct3 = combiner.transform(dev_slct2)
off3 = combiner.transform(off[dev_slct2.columns])

#分箱后通过画图观察
from toad.plot import  bin_plot,badrate_plot
bin_plot(dev_slct3,x='age',target='SeriousDlqin2yrs')
bin_plot(off3,x='age',target='SeriousDlqin2yrs')

#查看单箱节点
# print(bins['age'])

t=toad.transform.WOETransformer()
dev_slct2_woe = t.fit_transform(dev_slct3.drop(['SeriousDlqin2yrs'],axis=1),dev_slct3['SeriousDlqin2yrs'])
dev_slct2_woe['SeriousDlqin2yrs']=dev_slct3['SeriousDlqin2yrs']
dev_slct2_woe['flag']='train'
# print(dev_slct2_woe.head())
dev_slct2_woe.to_csv('D:/0.学习/python/屁屁和铭仔的数据之路/xjh_toad_train_woe.csv', sep=',', index=None)
off_woe = t.transform(off3.drop(['SeriousDlqin2yrs'],axis=1))
off_woe['SeriousDlqin2yrs']=off3['SeriousDlqin2yrs']
off_woe['flag']='test'
# print(off_woe.head())
off_woe.to_csv('D:/0.学习/python/屁屁和铭仔的数据之路/xjh_toad_test_woe.csv', sep=',', index=None)

data = pd.concat([dev_slct2_woe,off_woe])

# 模型训练
dep = 'SeriousDlqin2yrs'

# 变量名
lis = list(data.iloc[:,0:10].columns)
devv = data[data['flag'] == 'train']
offf = data[data['flag'] == 'test']
x, y = devv[lis], devv[dep]
offx, offy = offf[lis], offf[dep]
lr = LogisticRegression(C=0.1, class_weight='balanced')
lr.fit(x, y)

from toad.metrics import KS, F1, AUC
prob_dev = lr.predict_proba(x)[:,1]
print('训练集')
print('F1:', F1(prob_dev,y))
print('KS:', KS(prob_dev,y))
print('AUC:', AUC(prob_dev,y))

prob_off = lr.predict_proba(offx)[:,1]
print('测试集')
print('F1:', F1(prob_off,offy))
print('KS:', KS(prob_off,offy))
print('AUC:', AUC(prob_off,offy))

print('模型PSI:',toad.metrics.PSI(prob_dev,prob_off))
print('特征PSI:','\n',toad.metrics.PSI(x,offx).sort_values(0))

off_bucket = toad.metrics.KS_bucket(prob_off,offy,bucket=10,method='quantile')
print('off_bucket',off_bucket)

from toad.scorecard import ScoreCard
card = ScoreCard(combiner = combiner, transer  = t,class_weight = 'balanced',C=0.1,base_score = 600,base_odds = 35 ,pdo = 60,rate = 2)
card.fit(x,y)
final_card = card.export(to_frame = True)
final_card.to_csv('D:/0.学习/python/屁屁和铭仔的数据之路/xjh_toad_train_rule.csv', sep=',', index=None)