import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata


def MRMD_iterative_oneclass(X, y, positive_label, k):
    n_samples, n_features = X.shape
    rank_matrix_all = np.apply_along_axis(rankdata, 0, X)
    pos_indices = np.where(y == positive_label)[0]
    rank_matrix_pos = rank_matrix_all[pos_indices, :]

    relevance_scores = np.sum(rank_matrix_pos, axis=0)

    selected = []
    not_selected = set(range(n_features))

    # 第一个特征
    first_feature = np.argmax(relevance_scores)
    selected.append(first_feature)
    not_selected.remove(first_feature)

    while len(selected) < k:
        best_score = -np.inf
        best_feature = None
        print(len(selected), selected)
        for candidate in not_selected:
            R_cand = relevance_scores[candidate]

            diversities = []
            for s in selected:
                diff = np.abs(rank_matrix_pos[:, candidate] - rank_matrix_pos[:, s])
                D_val = np.sum(diff)
                diversities.append(D_val)

            avg_diversity = np.mean(diversities) if diversities else 0
            J_avg = R_cand + avg_diversity

            if J_avg > best_score:
                best_score = J_avg
                best_feature = candidate

        selected.append(best_feature)
        not_selected.remove(best_feature)

    return selected


def MRMD_multiclass_iterative(X, y, k):
    classes = np.unique(y)
    n_features = X.shape[1]
    n_classes = len(classes)

    # 存储每个类别迭代得到的特征列表 (n_classes x k)
    selected_features_per_class = np.zeros((n_classes, k), dtype=int)

    for idx, c in enumerate(classes):
        selected_features = MRMD_iterative_oneclass(X, y, c, k)
        selected_features_per_class[idx, :] = selected_features

    # 统计每个特征在所有类别选中特征中的出现频率和平均排名
    feature_scores = np.zeros(n_features)

    for f in range(n_features):
        ranks = []
        for cls_idx in range(n_classes):
            # 如果特征 f 在该类别的选择序列中，取其排名
            if f in selected_features_per_class[cls_idx, :]:
                rank = np.where(selected_features_per_class[cls_idx, :] == f)[0][0] + 1
                ranks.append(k - rank + 1)  # 选得越早得分越高
        if ranks:
            feature_scores[f] = np.mean(ranks)
        else:
            feature_scores[f] = 0

    # 选出分数最高的k个特征
    final_selected = np.argsort(-feature_scores)[:k]

    return final_selected


if __name__ == "__main__":
    fPath = r'D:\E盘\数据\microarray data\Leukemia4.csv'
    dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))
    print("数据形状:", np.shape(dataMatrix))

    select = range(7129)
    feature_data = dataMatrix[:, select]
    labels = dataMatrix[:, -1]

    scaler = MinMaxScaler()
    feature_data = scaler.fit_transform(feature_data)

    k = 200
    selected_features = MRMD_multiclass_iterative(feature_data, labels, k)
    print("多类迭代版本选出的特征索引:", selected_features)


#  SRBCT MLL 3c 4c Lung