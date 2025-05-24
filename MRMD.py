import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata


def MRMD_positive_ranks(X, y, k, positive_label=1):
    n_samples, n_features = X.shape

    # 1. 对所有样本每个特征列做排名，排名起点为1
    # rank_matrix shape: (n_samples, n_features)
    rank_matrix_all = np.apply_along_axis(rankdata, 0, X)

    # 2. 找出正样本索引
    pos_indices = np.where(y == positive_label)[0]

    # 3. 只保留正样本的排名值用于计算相关性和多样性
    rank_matrix_pos = rank_matrix_all[pos_indices, :]  # shape: (#pos_samples, n_features)

    # 4. 计算相关性R(x_k;y) = 所有正样本排名和
    relevance_scores = np.sum(rank_matrix_pos, axis=0)

    selected = []
    not_selected = set(range(n_features))

    # 5. 选第一个特征，R最大的
    first_feature = np.argmax(relevance_scores)
    selected.append(first_feature)
    not_selected.remove(first_feature)

    # 6. 迭代选剩余特征
    while len(selected) < k:
        best_score = -np.inf
        best_feature = None
        print(len(selected),selected)
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


if __name__ == "__main__":
    fPath = r'D:\E盘\数据\microarray data\Ovary.csv'
    dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))
    print("数据形状:", np.shape(dataMatrix))

    select = range(15154)
    feature_data = dataMatrix[:, select]
    labels = dataMatrix[:, -1]

    scaler = MinMaxScaler()
    feature_data = scaler.fit_transform(feature_data)

    k = 200
    positive_label = 1

    selected_features = MRMD_positive_ranks(feature_data, labels, k, positive_label)
    print("选出的特征索引:", selected_features)
