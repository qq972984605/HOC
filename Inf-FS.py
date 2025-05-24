import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from PyIFS import InfFS

# === Step 1: 读取数据集 ===
fPath = r'D:\E盘\数据\microarray data\Breast.csv'  # 修改为实际路径
dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))
X = dataMatrix[:, :-1]  # 特征矩阵
y = dataMatrix[:, -1]   # 类别标签

# 转换标签为 -1 / 1（仅限二分类）
classes = np.unique(y)
if len(classes) == 2:
    y = np.where(y == classes[0], -1, 1)

# === Step 2: 离散化特征（用于计算互信息） ===
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X_discrete = discretizer.fit_transform(X)

# === Step 3: 执行 Inf-FS 特征选择 ===
inf = InfFS()

# 设置参数
alpha = [1/3, 1/3, 1/3]         # 权重
supervision = 1                  # 使用监督信息（1 为有监督，0 为无监督）
verbose = 1                       # 打印详细信息

# 调用 Inf-FS，传入 X (特征矩阵)，y (标签向量)，alpha (权重)，supervision (是否监督)，verbose (打印信息)
RANKED, WEIGHT = inf.infFS(X, y, alpha=alpha, supervision=supervision, verbose=verbose)

# === Step 4: 输出特征选择结果 ===
print("\nTop 50 selected features (by index):")
print(list(RANKED[:200]))  # 输出前50个特征的索引

print(f"\nNumber of features selected: {len(RANKED)}")
print(f"Feature weights: {WEIGHT[:50]}")  # 输出前50个特征的权重
