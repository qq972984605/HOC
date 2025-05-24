import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer

# ========== 读取数据 ==========
fPath = r'D:\E盘\数据\microarray data\Leukemia4.csv'
dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))

X = dataMatrix[:, :-1]
y = dataMatrix[:, -1]

# ========== 连续数据离散化 ==========
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X_discrete = discretizer.fit_transform(X)

# ========== 超快速 Top-k mRMR ==========
def topk_mrmr(X_discrete, y, num_features_to_select=200, top_k=5):
    n_samples, n_features = X_discrete.shape
    selected_features = []
    remaining_features = list(range(n_features))

    # Step 1: 计算每个特征与标签的互信息（relevance）
    relevance = mutual_info_classif(X_discrete, y, discrete_features=True)

    # Step 2: 只在需要的时候计算特征之间的互信息
    MI_cache = dict()

    def get_mi(i, j):
        key = tuple(sorted((i, j)))
        if key not in MI_cache:
            mi = mutual_info_classif(
                X_discrete[:, i].reshape(-1, 1),
                X_discrete[:, j],
                discrete_features=True
            )[0]
            MI_cache[key] = mi
        return MI_cache[key]

    # Step 3: 选第一个特征
    first_feature = np.argmax(relevance)
    selected_features.append(first_feature)
    remaining_features.remove(first_feature)

    # Step 4: 迭代选择剩余的
    while len(selected_features) < num_features_to_select:
        mrmr_scores = []

        for feature in remaining_features:
            # 只取最近 top_k 个已选特征计算冗余
            recent_selected = selected_features[-top_k:] if len(selected_features) >= top_k else selected_features
            redundancy = np.mean([get_mi(feature, sel) for sel in recent_selected])
            mrmr_score = relevance[feature] - redundancy
            mrmr_scores.append(mrmr_score)

        best_feature_idx = np.argmax(mrmr_scores)
        best_feature = remaining_features[best_feature_idx]

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

    return selected_features

# ========== 执行 ==========
# ========== 执行 ==========

selected = topk_mrmr(X_discrete, y, num_features_to_select=200, top_k=5)
feature_data = X[:, selected]
labels = y

# 输出选择的特征索引和形状
print(f"选择的特征索引: {selected}")
print(f"feature_data 形状: {feature_data.shape}")

# ========== 输出互信息分数及其排名 ==========

# 再次计算所有特征与标签的互信息
relevance = mutual_info_classif(X_discrete, y, discrete_features=True)

# 创建 DataFrame 用于排序和展示
mi_df = pd.DataFrame({
    'Feature Index': list(range(X.shape[1])),
    'MI Score': relevance
})
mi_df_sorted = mi_df.sort_values(by='MI Score', ascending=False).reset_index(drop=True)

# 输出前若干高 MI 分数的特征
print("\n互信息 (MI) 分数排名前20的特征：")
print(mi_df_sorted.head(20))

# 可选：输出已选特征在 MI 排名中的位置
print("\n已选特征在 MI 排名中的位置：")
selected_ranks = mi_df_sorted[mi_df_sorted['Feature Index'].isin(selected)].copy()
selected_ranks['MI Rank'] = selected_ranks.index + 1
print('开始')
print(selected_ranks.sort_values(by='MI Rank'))
print('结束')
