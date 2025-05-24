from sklearn.impute import SimpleImputer  # 使用 SimpleImputer
from sklearn.preprocessing import MinMaxScaler  # 归一化
from skrebate import ReliefF  # ReliefF 特征选择
import numpy as np
import pandas as pd

# 读取数据
fPath = 'D:\E盘\数据\microarray data\Leukemia4.csv'
dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))
rowNum, colNum = dataMatrix.shape[0], dataMatrix.shape[1]
sampleData = []
sampleClass = []
for i in range(0, rowNum):
    tempList = list(dataMatrix[i, :])
    sampleClass.append(tempList[-1])
    sampleData.append(tempList[:-1])
sampleM = np.array(sampleData)  # 特征矩阵
classM = np.array(sampleClass)  # 类别向量

# 处理缺失值：用均值填补
imputer = SimpleImputer(strategy='mean')  # 使用 SimpleImputer
sampleM = imputer.fit_transform(sampleM)

# 删除常数特征
var = sampleM.var(axis=0)
constant_features = np.where(var == 0)[0]
sampleM = np.delete(sampleM, constant_features, axis=1)

# 对特征进行归一化（Min-Max 归一化到 [0,1]）
scaler = MinMaxScaler()
sampleM = scaler.fit_transform(sampleM)

# 使用 ReliefF 计算特征评分
relief = ReliefF(n_neighbors=10, n_features_to_select=200)  # 选择前 200 个特征
relief.fit(sampleM, classM)

# 获取所有特征的评分和索引
scores = relief.feature_importances_
indices = np.arange(len(scores))

# 对 ReliefF 评分进行排序
sorted_indices = np.argsort(scores)[::-1]  # 按降序排列
sorted_scores = scores[sorted_indices]
sorted_features = indices[sorted_indices]

# 获取前 200 个特征
top_200_scores = sorted_scores[:200]
top_200_features = sorted_features[:200]

# 输出结果
print("Top 200 Features' Indices and ReliefF Scores:")
print(list(top_200_features))