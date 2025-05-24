import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import time

# ========== 读取数据 ==========
fPath = r'D:\E盘\数据\microarray data\Leukemia4.csv'  # 请根据实际路径修改
dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))
X = dataMatrix[:, :-1]
y = dataMatrix[:, -1]

# ========== 连续数据离散化 ==========
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
X_discrete = discretizer.fit_transform(X)

# ========== Fast Top-k JMIM ==========
def fast_topk_jmim(X_discrete, y, num_features_to_select=200, top_k=200):
    n_samples, n_features = X_discrete.shape
    selected_features = []
    remaining_features = list(range(n_features))

    # Step 1: 计算每个特征与标签的互信息
    relevance = mutual_info_classif(X_discrete, y, discrete_features=True)

    # Step 2: 选第一个特征（MI 最大）
    first_feature = np.argmax(relevance)
    selected_features.append(first_feature)
    remaining_features.remove(first_feature)

    # Step 3: 缓存条件互信息函数
    def conditional_mi(fx, fy, target):
        """估计 I(fx; target | fy) ≈ I(fx; (target, fy)) - I(fx; fy)"""
        fy_str = fy.astype(str)
        target_str = target.astype(str)

        # 使用 defchararray.add 分两次拼接字符串，确保类型一致
        joint = np.core.defchararray.add(target_str, "_")
        fy_target = np.core.defchararray.add(joint, fy_str)

        return mutual_info_score(fx, fy_target) - mutual_info_score(fx, fy)

    # Step 4: 主循环（K-top JMIM）
    while len(selected_features) < num_features_to_select:
        # 仅保留 MI 排名前 top_k 的候选特征
        topk_candidates = sorted(
            remaining_features,
            key=lambda idx: relevance[idx],
            reverse=True
        )[:top_k]

        best_score = -np.inf
        best_feature = None

        for feature in topk_candidates:
            cmis = [conditional_mi(X_discrete[:, feature], X_discrete[:, sel], y)
                    for sel in selected_features]
            jmim_score = min(cmis)

            if jmim_score > best_score:
                best_score = jmim_score
                best_feature = feature

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)

    return selected_features

# ========== 执行并计时 ==========
start_time = time.time()
selected = fast_topk_jmim(X_discrete, y, num_features_to_select=200, top_k=10)
end_time = time.time()

# ========== 输出 ==========
feature_data = X[:, selected]
labels = y

print(f"选择的特征索引: {selected}")
print(f"选择特征后的数据形状: {feature_data.shape}")
print(f"特征选择耗时: {end_time - start_time:.2f} 秒")

# ========== 输出互信息分数及其排名 ==========
relevance = mutual_info_classif(X_discrete, y, discrete_features=True)

mi_df = pd.DataFrame({
    'Feature Index': list(range(X.shape[1])),
    'MI Score': relevance
})
mi_df_sorted = mi_df.sort_values(by='MI Score', ascending=False).reset_index(drop=True)

print("\n互信息 (MI) 分数排名前20的特征：")
print(mi_df_sorted.head(20))

print("\n已选特征在 MI 排名中的位置：")
selected_ranks = mi_df_sorted[mi_df_sorted['Feature Index'].isin(selected)].copy()
selected_ranks['MI Rank'] = selected_ranks.index + 1
print(selected_ranks.sort_values(by='MI Rank'))
