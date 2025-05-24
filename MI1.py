from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import pandas as pd

# 加载数据
fPath = 'D:\\E盘\\数据\\microarray data\\Leukemia4.csv'
dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))
print(np.shape(dataMatrix))
# 分离样本数据和类别
sampleM = dataMatrix[:, :-1]  # 特征数据
classM = dataMatrix[:, -1]    # 类别标签

# **新增：离散化数据**
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
sampleM_discrete = discretizer.fit_transform(sampleM)  # 离散化后的特征数据

# 计算所有特征与类别的互信息分数（基于离散化数据）
mi_scores = mutual_info_classif(sampleM_discrete, classM, discrete_features=True)

# 获取互信息分数及对应的特征索引
sorted_indices = np.argsort(mi_scores)[::-1]  # 按互信息分数从高到低排序
top_200_indices = sorted_indices[:200]  # 取前200个特征的索引
top_50_indices = sorted_indices[:50]
top_500_indices = sorted_indices[:500]
top_1000_indices = sorted_indices[:1000]
top_2000_indices = sorted_indices[:2000]
top_5000_indices = sorted_indices[:5000]
top_10000_indices = sorted_indices[:10000]
top_200_scores = mi_scores[top_200_indices]  # 取前200个特征的互信息分数

print(list(top_50_indices))
print(list(top_200_indices))
print(list(top_500_indices))
print(list(top_1000_indices))
print(list(top_2000_indices))
print(list(top_5000_indices))
print(list(top_10000_indices))

# 输出前200个特征的互信息分数
# print("互信息分数最高的前200个特征（离散化后）：")
# for rank, (index, score) in enumerate(zip(top_200_indices, top_200_scores), start=1):
#     feature_name = f"特征 {index+1}"
#     print(f"第{rank}名 - {feature_name} 的互信息分数: {score:.10f}")
