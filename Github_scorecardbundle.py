# 模块导入
from scorecardbundle.feature_discretization import ChiMerge as cm
from scorecardbundle.feature_encoding import WOE as woe
from scorecardbundle.feature_selection import FeatureSelection as fs
from scorecardbundle.model_training import LogisticRegressionScoreCard as lrsc
from scorecardbundle.model_evaluation import ModelEvaluation as me
import pandas as pd

# 加载数据
f_dev = open('D:/0.学习/python/屁屁和铭仔的数据之路/TrainData.csv')
dev = pd.read_csv(f_dev, sep=',')
# print('dev',dev.head(5))
f_val = open('D:/0.学习/python/屁屁和铭仔的数据之路/TestData.csv')
off = pd.read_csv(f_val, sep=',')
# print('off',off.head(5))

# 变量名
y = dev['SeriousDlqin2yrs']
x = dev.drop(['SeriousDlqin2yrs'], axis=1)
y_off = off['SeriousDlqin2yrs']
x_off = off.drop(['SeriousDlqin2yrs'], axis=1)

# 特征离散化（基于ChiMerge）
trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=2, output_dataframe=True)
result_cm = trans_cm.fit_transform(x, y)
print('每个特征的区间切分',trans_cm.boundaries_) # 每个特征的区间切分


# 特征编码（基于证据权重WOE）
trans_woe = woe.WOE_Encoder(output_dataframe=True)
result_woe = trans_woe.fit_transform(result_cm, y)
print('每个特征的信息值 (iv)',trans_woe.iv_) # 每个特征的信息值 (iv)
print('每个特征的WOE字典和信息值 (iv)',trans_woe.result_dict_) # 每个特征的WOE字典和信息值 (iv)

# 特征选择
fs.selection_with_iv_corr(trans_woe, result_woe)

# 模型训练
model = lrsc.LogisticRegressionScoreCard(trans_woe, PDO=-20, basePoints=100, verbose=True)
model.fit(result_woe, y)
# print('从woe_df_属性中可得评分卡规则',model.woe_df_) # 从woe_df_属性中可得评分卡规则
model.woe_df_.to_csv('D:/0.学习/python/屁屁和铭仔的数据之路/xjh_scorecardbundle_rule.csv', sep=',', index=None)
result = model.predict(x) # 评分卡应该应用在初始的特征上（即未经离散化和WOE编码的特征数据）
result.to_csv('D:/0.学习/python/屁屁和铭仔的数据之路/xjh_scorecardbundle_result.csv', sep=',', index=None) # 输出的csv包含每个变量的得分和总得分，不包含变量的原始值

# 用户可手动调整规则（如下面方法的手动调整、或导出到本地修改好再上传excel文件），并使用predict()的load_scorecard参数传入模型，详见load_scorecard参数的文档。
# sc_table = model.woe_df_.copy()
# sc_table['score'][(sc_table.feature=='longitude') & (sc_table.value=='-122.62~-121.58000000000001')] = 100

# 模型评估
dev_evaluation = me.BinaryTargets(y, result['TotalScore'])
# dev_evaluation.plot_all()
print('train_ks',dev_evaluation.ks_stat())

result_off = model.predict(x_off)
val_evaluation = me.BinaryTargets(y_off, result_off['TotalScore'])
# val_evaluation.plot_all()
print('test_ks',val_evaluation.ks_stat())

