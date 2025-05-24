import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict

# ========== 读取数据 ==========
fPath = r'D:\E盘\数据\microarray data\Leukemia4.csv'
dataMatrix = np.array(pd.read_csv(fPath, header=None, skiprows=1))
print(np.shape(dataMatrix))
b = 1
c = 3
# 1,2  1,3
select = range(7129)  # 你的 select 列表
feature_data = dataMatrix[:, select]
labels = dataMatrix[:, -1]

# 数据归一化
scaler = MinMaxScaler()
feature_data = scaler.fit_transform(feature_data)

unique_labels = np.unique(labels)
n_samples = feature_data.shape[0]
n_features = len(select)

# ========== 初始化 矩阵 ==========
select_mask_matrix = np.zeros((n_features, n_samples), dtype=np.uint32)

# 用于存储每个特征的选中样本数量
feature_selected_counts = []

# 计算所有特征的重叠面积并存储
overlap_area_cache = {}

cross_points_all = []
all_selected_samples = []
selected_samples_per_feature = {}  # 初始化：每个特征的异常样本集合
skip = []

# range(feature_data.shape[1])
# ========== 循环处理每个特征 ==========
for feat_global_idx in range(feature_data.shape[1]):  # 遍历所有特征
    print(feat_global_idx)
    feature_values = feature_data[:, feat_global_idx]
    if np.std(feature_values) <= 0.1:
        print('跳过', feat_global_idx)
        skip.append(feat_global_idx)
        for i in range(feature_data.shape[0]):
            select_mask_matrix[feat_global_idx, i] = i+1
        feature_selected_counts.append((feat_global_idx, feature_data.shape[1]))
        overlap_area_cache[feat_global_idx] = 1e9
        continue

    # 初始化当前特征的所有交点
    cross_points_current_feature = []

    # 对所有类别进行一次KDE建模
    kde_list = []
    all_area = []
    for label in unique_labels:
        class_values = feature_values[labels == label]
        kde_list.append(gaussian_kde(class_values))
        for i in range(len(kde_list)):
            kde = kde_list[i]
            x_range = np.linspace(min(feature_values), max(feature_values), 1000)
            y = kde(x_range)
            all_area.append(np.trapz(y, x_range))

    # 遍历每对类别
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):  # 对每一对类别进行处理
            kde1 = kde_list[i]
            kde2 = kde_list[j]

            # 生成密度曲线
            x_range = np.linspace(min(feature_values), max(feature_values), 1000)
            y1 = kde1(x_range)
            y2 = kde2(x_range)

            # 找交点
            diff = y1 - y2
            sign_changes = np.where(np.diff(np.sign(diff)))[0]

            if len(sign_changes) == 0:
                print(f"特征 {feat_global_idx} 类别 {unique_labels[i]} 和 类别 {unique_labels[j]} 没有交点，跳过。")
                continue

            # 插值确定交点位置
            for idx in sign_changes:
                x0, x1_ = x_range[idx], x_range[idx + 1]
                y0, y1_ = diff[idx], diff[idx + 1]
                cross_x = x0 - y0 * (x1_ - x0) / (y1_ - y0)  # 插值计算交点
                cross_points_current_feature.append(cross_x)

    # 将当前特征的所有交点添加到所有交点数组中
    cross_points_all.append(sorted(cross_points_current_feature))  # 对交点进行排序

    # 划分区域，并计算每个区域内每个类的面积
    cross_points_sorted = sorted(cross_points_current_feature)  # 按照交点排序
    all_intervals = []

    # 创建区域划分：负无穷到第一个交点, 第一个交点到第二个交点, ... ，最后一个交点到正无穷
    intervals = [(min(feature_values)-0.0000001, cross_points_sorted[0])]  # 第一个区域是负无穷到第一个交点
    for i in range(1, len(cross_points_sorted)):
        intervals.append((cross_points_sorted[i - 1], cross_points_sorted[i]))  # 交点之间的区域
    intervals.append((cross_points_sorted[-1], max(feature_values)+0.0000001))  # 最后一个区域是最后一个交点到正无穷

    dominant_intervals = defaultdict(list)

    # 计算每个区域内每个类的面积，并判断哪个类在该区域占主导地位
    # 合并支配区间：按顺序处理 intervals
    merged_intervals_per_class = defaultdict(list)

    current_class = None
    current_start = None
    current_end = None
    area_dominant_class = []
    area_label = []
    area_max = []
    for interval in intervals:
        left, right = interval
        area_per_class = {}
        for i, kde in enumerate(kde_list):
            x_range = np.linspace(left, right, 500)
            y = kde(x_range)
            area = np.trapz(y, x_range)
            area_per_class[unique_labels[i]] = area
            area_label.append(area)
            area_max.append(max(area_label))

        dominant_class = max(area_per_class, key=area_per_class.get)

        # 如果和前一个区间支配类相同，则扩展当前区间
        if current_class == dominant_class:
            current_end = right
        else:
            if current_class is not None:
                # 保存上一个区间
                merged_intervals_per_class[current_class].append((current_start, current_end))
            # 更新当前区间
            current_class = dominant_class
            current_start = left
            current_end = right

    # 保存最后一个合并区间
    if current_class is not None:
        merged_intervals_per_class[current_class].append((current_start, current_end))

    dominant_intervals = merged_intervals_per_class  # 替换旧变量名，保证后续逻辑不变

    # ========== 根据面积阈值筛除异常区域 ==========
    final_dominant_intervals = defaultdict(list)

    for label, intervals in merged_intervals_per_class.items():
        areas = []
        for interval in intervals:
            left, right = interval
            x_range = np.linspace(left, right, 500)
            y = kde_list[list(unique_labels).index(label)](x_range)
            area = np.trapz(y, x_range)
            areas.append(area)

        if len(areas) == 0:
            continue

        avg_area = np.mean(areas)
        for idx, interval in enumerate(intervals):
            if areas[idx] >= 0.5 * avg_area:
                final_dominant_intervals[label].append(interval)
            else:
                print(f"区域 {interval} 面积过小（{areas[idx]:.3f} < 50% 平均值 {avg_area:.3f}），被视为异常，剔除。")

    # 最终使用更新后的 dominant_intervals
    dominant_intervals = final_dominant_intervals

    # 存储异常样本信息
    all_selected_samples = []

    # 遍历每个类及其样本，检查是否落在该类的支配区域内
    for i, class_label in enumerate(unique_labels):
        class_indices = np.where(labels == class_label)[0]
        class_feat_values = feature_values[class_indices]
        class_dominant_ranges = dominant_intervals[class_label]

        for idx, val in zip(class_indices, class_feat_values):
            in_any_range = any(left <= val <= right for left, right in class_dominant_ranges)

            if not in_any_range:  # 视为异常值
                # 找到最近边界
                boundary_distances = []
                for left, right in class_dominant_ranges:
                    dist = min(abs(val - left), abs(val - right))
                    boundary_distances.append((dist, left, right))
                if boundary_distances:
                    min_dist, near_left, near_right = min(boundary_distances)
                    nearest_boundary = near_left if abs(val - near_left) < abs(val - near_right) else near_right
                else:
                    nearest_boundary = None  # 没有任何支配区域
                all_selected_samples.append((idx, class_label, np.abs(val+1)))

    # 记录选中样本数量
    all_selected_samples = list(dict.fromkeys(all_selected_samples))
    selected_count = len(all_selected_samples)
    feature_selected_counts.append((feat_global_idx, selected_count))
    # 按超出量排序（从小到大）
    all_selected_samples.sort(key=lambda x: x[2])  # 按超出值排序

    # 给排名：超得最少的赋1，第二少的赋2，...
    for rank, (sample_idx, _, _) in enumerate(all_selected_samples, start=1):
        select_mask_matrix[feat_global_idx, sample_idx] = rank

    # 保存当前特征的异常样本
    selected_samples_per_feature[feat_global_idx] = all_selected_samples.copy()

    # 计算重叠面积（通过计算两曲线的交集区域）
    # 使用已有 KDE 列表 kde_list 计算所有类在统一 x_range 上的密度
    x_range = np.linspace(min(feature_values), max(feature_values), 1000)
    kde_densities = np.array([kde(x_range) for kde in kde_list])  # shape: [num_classes, 1000]

    # 逐点取最小值（表示所有类别的共同部分）
    min_density = np.min(kde_densities, axis=0)

    # 计算最小密度的积分作为重叠面积
    overlap_area = np.trapz(min_density, x_range)
    overlap_area_cache[feat_global_idx] = overlap_area
    all_selected_samples.clear()

min_feat_idx = min(overlap_area_cache, key=overlap_area_cache.get)
min_overlap = overlap_area_cache[min_feat_idx]


print(f"重叠面积最小的是特征 {min_feat_idx}，重叠面积为 {min_overlap:.6f}")


# ========== 按选中样本数量排序 ==========
feature_selected_counts.sort(key=lambda x: x[1])  # 按异常样本数量升序排序
sorted_feature_indices = [feat_global_idx for feat_global_idx, _ in feature_selected_counts]

print("\n所有特征按选中样本数量排序：")
for idx, (feat_global_idx, selected_count) in enumerate(feature_selected_counts):
    print(f"排名 {idx + 1}: 特征 {feat_global_idx}，选中样本数为 {selected_count}")

# ========== 输出选中样本最少的特征 ==========

min_selected_feature = feature_selected_counts[0][0]
min_selected_count = feature_selected_counts[0][1]
min_selected_feature_mask = select_mask_matrix[min_selected_feature, :]  # 直接用全局索引即可

print(f"\n选中样本最少的特征是：特征 {min_selected_feature}，选中样本数为 {min_selected_count}。")
print(f"特征 {min_selected_feature} 的选择掩码矩阵（排名从1开始，未选中为0）：")
print(min_selected_feature_mask)

print("\n排序后的特征索引列表（按异常样本数升序）：")
print(sorted_feature_indices)



# ========== 计算与最少选中样本特征相加后的 综合指标（重叠面积 + 标准差） ==========
min_feature_mask = select_mask_matrix[select.index(min_selected_feature), :]
current_mask = select_mask_matrix[select.index(min_selected_feature), :]
# 用于记录每个特征加上后的得分
feature_scores = []

for feat_idx, feat_global_idx in enumerate(select):
    if feat_global_idx == min_selected_feature:
        continue  # 跳过自己和自己加


    other_feature_mask = select_mask_matrix[feat_idx, :]

    # 合并当前组合 + 新特征
    combined_mask = current_mask + other_feature_mask
    # 计算重叠面积
    overlap_area = overlap_area_cache[feat_global_idx]  # 从缓存中读取重叠面积

    # 计算标准差
    std = np.std(combined_mask)

    # 综合得分：重叠面积 + 0.5 * 标准差
    score = overlap_area * b +  std * c
    feature_scores.append((feat_global_idx, score))

# 按得分排序（重叠面积越小，标准差越小，得分越低）
feature_scores.sort(key=lambda x: x[1])

# 输出得分最小的特征
best_match_feature = feature_scores[0][0]
best_match_score = feature_scores[0][1]


print(f"与特征 {min_selected_feature} 组合后（重叠面积+标准差）得分最小的是特征 {best_match_feature}，得分为 {best_match_score:.6f}")

# ========== 迭代合并特征，逐步扩展组合（限制50个特征） ==========
used_features = [min_selected_feature]  # 已选特征，初始是选中最少样本的特征
available_features = set(select) - set(used_features)  # 可选特征集合

# 当前组合的mask（开始时是单个特征）
current_mask = select_mask_matrix[select.index(min_selected_feature), :]

# 第一次特别处理：找第一个最佳组合
feature_scores = []

for feat_global_idx in available_features:

    feat_idx = select.index(feat_global_idx)
    other_feature_mask = select_mask_matrix[feat_idx, :]

    # 合并当前组合 + 新特征
    combined_mask = current_mask + other_feature_mask

    # 计算重叠面积
    overlap_area = overlap_area_cache[feat_global_idx]  # 从缓存中读取重叠面积
    # 计算标准差
    std = np.std(combined_mask)

    # 综合得分：重叠面积 + 0.1 * 标准差
    score = overlap_area * b + std * c

    feature_scores.append((feat_global_idx, score))

# 找得分最小的特征
feature_scores.sort(key=lambda x: x[1])
best_match_feature = feature_scores[0][0]
best_match_score = feature_scores[0][1]

# 更新
used_features.append(best_match_feature)
available_features.remove(best_match_feature)
current_mask = current_mask + select_mask_matrix[select.index(best_match_feature), :]

print(f"与特征 {min_selected_feature} 组合后（重叠面积+标准差）得分最小的是特征 {best_match_feature}，得分为 {best_match_score:.6f}")
q = 0
# 后续正常循环
while len(used_features) < 200:  # 这里限制到50个特征
    feature_scores = []
    q = q + 1
    for feat_global_idx in available_features:
        feat_idx = select.index(feat_global_idx)
        other_feature_mask = select_mask_matrix[feat_idx, :]

        # 合并当前组合 + 新特征
        combined_mask = current_mask + other_feature_mask

        # 计算重叠面积（Overlap Area）
        overlap_area = overlap_area_cache[feat_global_idx]  # 从缓存中读取重叠面积

        # 计算标准差

        std = np.std(combined_mask)
        # 综合得分（重叠面积 + 标准差）
        score = overlap_area * b + std * c
        feature_scores.append((feat_global_idx, score))

    # 找得分最小的特征
    feature_scores.sort(key=lambda x: x[1])
    best_match_feature = feature_scores[0][0]
    best_match_score = feature_scores[0][1]

    # 更新
    used_features.append(best_match_feature)
    available_features.remove(best_match_feature)
    current_mask = current_mask + select_mask_matrix[select.index(best_match_feature), :]

    print(f"添加特征 {best_match_feature}，当前组合得分 {best_match_score:.6f}")

# 最后输出
print("\n最终特征选择顺序（50个特征）：")
print(used_features)
