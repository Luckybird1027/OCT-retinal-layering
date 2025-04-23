import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.stats import pearsonr


# --- 辅助函数 ---

def _get_boundary(mask, connectivity=1):
    """
    获取二值掩码的边界 (内部边界)
    :param mask: 输入的二值掩码 (HxW), NumPy array
    :param connectivity: 连通性 (1 for 4-connectivity, 2 for 8-connectivity)
    :return: 边界掩码 (HxW), NumPy array
    """
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)
    eroded_mask = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    boundary = mask ^ eroded_mask  # XOR operation finds the boundary
    return boundary


def _one_hot_encode(target, num_classes):
    """将类别索引 target 转换为 one-hot 编码"""
    if target.ndim == 4 and target.shape[1] == 1:  # (B, 1, H, W)
        target = target.squeeze(1)  # -> (B, H, W)
    elif target.ndim == 3:  # (B, H, W)
        pass
    elif target.ndim == 2:  # (H, W)
        target = target.unsqueeze(0)  # -> (1, H, W)
    else:
        raise ValueError(f"Unsupported target shape: {target.shape}")

    if isinstance(target, torch.Tensor):
        return F.one_hot(target.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    elif isinstance(target, np.ndarray):
        # NumPy one-hot encoding
        b, h, w = target.shape
        one_hot = np.zeros((b, num_classes, h, w), dtype=np.float32)
        for i in range(num_classes):
            one_hot[:, i, :, :] = (target == i)
        return one_hot
    else:
        raise TypeError("Input must be a PyTorch Tensor or NumPy array")


# --- 主要指标函数 ---

def dice_coefficient(pred_indices, target_indices, num_classes, ignore_index=0, epsilon=1e-6):
    """
    计算多类别 Dice 系数 (忽略指定类别)
    :param pred_indices: 预测的类别索引 (B, H, W) or (H, W), NumPy array or Torch Tensor
    :param target_indices: 真实的类别索引 (B, H, W) or (H, W), NumPy array or Torch Tensor
    :param num_classes: 类别总数
    :param ignore_index: 要忽略的类别索引 (通常是背景)
    :param epsilon: 防止除以零的小值
    :return: 平均 Dice 系数 (float)
    """
    if isinstance(pred_indices, torch.Tensor):
        pred_indices = pred_indices.detach().cpu().numpy()
    if isinstance(target_indices, torch.Tensor):
        target_indices = target_indices.detach().cpu().numpy()

    if pred_indices.ndim == 2:  # Add batch dimension if missing
        pred_indices = np.expand_dims(pred_indices, axis=0)
    if target_indices.ndim == 2:
        target_indices = np.expand_dims(target_indices, axis=0)

    assert pred_indices.shape == target_indices.shape, "Prediction and target shapes must match"
    assert pred_indices.ndim == 3, "Inputs should be (B, H, W)"

    dice_scores = []
    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = (pred_indices == c)
        target_c = (target_indices == c)

        intersection = np.sum(pred_c * target_c, axis=(1, 2))
        cardinality = np.sum(pred_c, axis=(1, 2)) + np.sum(target_c, axis=(1, 2))

        dice = (2. * intersection + epsilon) / (cardinality + epsilon)
        dice_scores.append(dice)  # Shape: [num_valid_classes, B]

    # Average over classes and then over batch
    if not dice_scores:  # Handle case where only ignore_index exists
        warnings.warn("No valid classes found for Dice calculation (excluding ignore_index). Returning 0.0")
        return 0.0
    mean_dice_per_batch = np.mean(dice_scores, axis=0)  # Average over classes for each batch item
    mean_dice = np.mean(mean_dice_per_batch)  # Average over batch

    return float(mean_dice)


def intersection_over_union(pred_indices, target_indices, num_classes, ignore_index=0, epsilon=1e-6):
    """
    计算多类别 IoU (Jaccard Index) (忽略指定类别)
    :param pred_indices: 预测的类别索引 (B, H, W) or (H, W), NumPy array or Torch Tensor
    :param target_indices: 真实的类别索引 (B, H, W) or (H, W), NumPy array or Torch Tensor
    :param num_classes: 类别总数
    :param ignore_index: 要忽略的类别索引 (通常是背景)
    :param epsilon: 防止除以零的小值
    :return: 平均 IoU (float)
    """
    if isinstance(pred_indices, torch.Tensor):
        pred_indices = pred_indices.detach().cpu().numpy()
    if isinstance(target_indices, torch.Tensor):
        target_indices = target_indices.detach().cpu().numpy()

    if pred_indices.ndim == 2:  # Add batch dimension if missing
        pred_indices = np.expand_dims(pred_indices, axis=0)
    if target_indices.ndim == 2:
        target_indices = np.expand_dims(target_indices, axis=0)

    assert pred_indices.shape == target_indices.shape, "Prediction and target shapes must match"
    assert pred_indices.ndim == 3, "Inputs should be (B, H, W)"

    iou_scores = []
    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = (pred_indices == c)
        target_c = (target_indices == c)

        intersection = np.sum(pred_c * target_c, axis=(1, 2))
        union = np.sum(pred_c, axis=(1, 2)) + np.sum(target_c, axis=(1, 2)) - intersection

        iou = (intersection + epsilon) / (union + epsilon)
        iou_scores.append(iou)  # Shape: [num_valid_classes, B]

    # Average over classes and then over batch
    if not iou_scores:
        warnings.warn("No valid classes found for IoU calculation (excluding ignore_index). Returning 0.0")
        return 0.0
    mean_iou_per_batch = np.mean(iou_scores, axis=0)  # Average over classes for each batch item
    mean_iou = np.mean(mean_iou_per_batch)  # Average over batch

    return float(mean_iou)


def mean_absolute_border_location_deviation(pred_indices, target_indices, num_classes, ignore_index=0, connectivity=1):
    """
    计算平均绝对边界位置偏差 (MABLD)
    :param pred_indices: 预测的类别索引 (H, W), NumPy array
    :param target_indices: 真实的类别索引 (H, W), NumPy array
    :param num_classes: 类别总数
    :param ignore_index: 要忽略的类别索引
    :param connectivity: 边界提取的连通性
    :return: 平均 MABLD (float)
    """
    assert pred_indices.shape == target_indices.shape, "Prediction and target shapes must match"
    assert pred_indices.ndim == 2, "Inputs should be (H, W)"

    total_mabld = 0.0
    valid_classes = 0

    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = (pred_indices == c)
        target_c = (target_indices == c)

        # 跳过在 target 中不存在的类别
        if np.sum(target_c) == 0:
            continue

        pred_boundary = _get_boundary(pred_c, connectivity)
        target_boundary = _get_boundary(target_c, connectivity)

        # 如果任一边界为空，则无法计算距离，偏差视为很大 (或跳过)
        # 这里我们选择跳过，避免影响平均值
        if np.sum(pred_boundary) == 0 or np.sum(target_boundary) == 0:
            # warnings.warn(f"Boundary calculation failed or empty for class {c}. Skipping MABLD calculation for this class.")
            continue  # 或者可以赋一个较大的惩罚值

        # 计算从 pred 边界到 target 边界的距离
        dist_transform_target_inv = distance_transform_edt(np.logical_not(target_boundary))
        dist_pred_to_target = dist_transform_target_inv[pred_boundary]

        # 计算从 target 边界到 pred 边界的距离
        dist_transform_pred_inv = distance_transform_edt(np.logical_not(pred_boundary))
        dist_target_to_pred = dist_transform_pred_inv[target_boundary]

        # 计算平均距离
        mean_dist_pred = np.mean(dist_pred_to_target) if len(dist_pred_to_target) > 0 else 0
        mean_dist_target = np.mean(dist_target_to_pred) if len(dist_target_to_pred) > 0 else 0

        # MABLD for class c
        mabld_c = (mean_dist_pred + mean_dist_target) / 2.0
        total_mabld += mabld_c
        valid_classes += 1

    if valid_classes == 0:
        warnings.warn("No valid classes found for MABLD calculation. Returning NaN.")
        return np.nan  # Or a large value like float('inf')
    else:
        return total_mabld / valid_classes


def hausdorff_distance_95(pred_indices, target_indices, num_classes, ignore_index=0, connectivity=1):
    """
    计算 95% 分位数的 Hausdorff 距离
    :param pred_indices: 预测的类别索引 (H, W), NumPy array
    :param target_indices: 真实的类别索引 (H, W), NumPy array
    :param num_classes: 类别总数
    :param ignore_index: 要忽略的类别索引
    :param connectivity: 边界提取的连通性
    :return: 平均 95% Hausdorff 距离 (float)
    """
    assert pred_indices.shape == target_indices.shape, "Prediction and target shapes must match"
    assert pred_indices.ndim == 2, "Inputs should be (H, W)"

    total_hausdorff_95 = 0.0
    valid_classes = 0

    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = (pred_indices == c)
        target_c = (target_indices == c)

        # 跳过在 target 中不存在的类别
        if np.sum(target_c) == 0:
            continue

        pred_boundary = _get_boundary(pred_c, connectivity)
        target_boundary = _get_boundary(target_c, connectivity)

        # 如果任一边界为空，则无法计算距离
        if np.sum(pred_boundary) == 0 or np.sum(target_boundary) == 0:
            # warnings.warn(f"Boundary calculation failed or empty for class {c}. Skipping Hausdorff calculation for this class.")
            continue  # 或者可以赋一个较大的惩罚值

        # 计算从 pred 边界到 target 边界的距离
        dist_transform_target_inv = distance_transform_edt(np.logical_not(target_boundary))
        dist_pred_to_target = dist_transform_target_inv[pred_boundary]

        # 计算从 target 边界到 pred 边界的距离
        dist_transform_pred_inv = distance_transform_edt(np.logical_not(pred_boundary))
        dist_target_to_pred = dist_transform_pred_inv[target_boundary]

        # 合并所有距离并计算 95% 分位数
        all_distances = np.concatenate((dist_pred_to_target, dist_target_to_pred))
        if len(all_distances) == 0:
            hausdorff_95_c = 0  # Or handle as error/warning
        else:
            hausdorff_95_c = np.percentile(all_distances, 95)

        total_hausdorff_95 += hausdorff_95_c
        valid_classes += 1

    if valid_classes == 0:
        warnings.warn("No valid classes found for Hausdorff 95 calculation. Returning NaN.")
        return np.nan  # Or a large value
    else:
        return total_hausdorff_95 / valid_classes


def layer_thickness_correlation(pred_indices, target_indices, num_classes, ignore_index=0):
    """
    计算预测层厚度与真实层厚度之间的 Pearson 相关系数
    :param pred_indices: 预测的类别索引 (H, W), NumPy array
    :param target_indices: 真实的类别索引 (H, W), NumPy array
    :param num_classes: 类别总数 (包括背景)
    :param ignore_index: 背景类别索引
    :return: 平均 Pearson 相关系数 (float)
    """
    assert pred_indices.shape == target_indices.shape, "Prediction and target shapes must match"
    assert pred_indices.ndim == 2, "Inputs should be (H, W)"
    h, w = pred_indices.shape

    all_correlations = []

    # 假设层是从 1 到 num_classes-1 顺序排列的
    # 厚度定义为层 c 的下边界 y 坐标 - 层 c 的上边界 y 坐标
    # 层 c 的上边界是 c 本身的最上方像素
    # 层 c 的下边界是 c+1 层的最上方像素 (或图像底部)

    boundaries_pred = {}
    boundaries_target = {}

    # 找到每个类别（层）的上边界 y 坐标
    for c in range(num_classes):
        if c == ignore_index: continue
        boundaries_pred[c] = np.full(w, h, dtype=int)  # 初始化为图像底部
        boundaries_target[c] = np.full(w, h, dtype=int)
        for col in range(w):
            pred_col = np.where(pred_indices[:, col] == c)[0]
            target_col = np.where(target_indices[:, col] == c)[0]
            if len(pred_col) > 0:
                boundaries_pred[c][col] = np.min(pred_col)
            if len(target_col) > 0:
                boundaries_target[c][col] = np.min(target_col)

    # 计算每层的厚度并计算相关性
    for c in range(1, num_classes - 1):  # 遍历内部层 (假设层按顺序)
        # 上边界是当前层 c 的上边界
        upper_bound_pred = boundaries_pred.get(c)
        upper_bound_target = boundaries_target.get(c)

        # 下边界是下一层 c+1 的上边界
        lower_bound_pred = boundaries_pred.get(c + 1)
        lower_bound_target = boundaries_target.get(c + 1)

        if upper_bound_pred is None or lower_bound_pred is None or \
                upper_bound_target is None or lower_bound_target is None:
            continue  # 如果缺少边界信息，跳过该层

        # 计算厚度 (只在上下边界都有效的地方计算)
        valid_cols = (upper_bound_pred < h) & (lower_bound_pred < h) & \
                     (upper_bound_target < h) & (lower_bound_target < h) & \
                     (lower_bound_pred > upper_bound_pred) & \
                     (lower_bound_target > upper_bound_target)

        if np.sum(valid_cols) < 2:  # 需要至少两个点来计算相关性
            # warnings.warn(f"Not enough valid columns to calculate thickness correlation for class {c}.")
            continue

        thickness_pred = lower_bound_pred[valid_cols] - upper_bound_pred[valid_cols]
        thickness_target = lower_bound_target[valid_cols] - upper_bound_target[valid_cols]

        # 检查标准差是否为零
        if np.std(thickness_pred) < 1e-6 or np.std(thickness_target) < 1e-6:
            # warnings.warn(f"Zero standard deviation in thickness for class {c}. Correlation is undefined, skipping.")
            # 如果厚度恒定，可以认为相关性为1或跳过
            # all_correlations.append(1.0 if np.allclose(thickness_pred, thickness_target) else 0.0)
            continue

        # 计算 Pearson 相关系数
        corr, p_value = pearsonr(thickness_pred, thickness_target)
        if not np.isnan(corr):
            all_correlations.append(corr)
        # else:
        # warnings.warn(f"NaN correlation calculated for class {c}. Skipping.")

    if not all_correlations:
        warnings.warn("No valid layer thickness correlations could be calculated. Returning NaN.")
        return np.nan
    else:
        return np.mean(all_correlations)


# --- 拓扑完整性指标 (简化版 - 仅断点和交叉点) ---
# 注意：这些指标的精确实现可能依赖于具体的层定义和边界跟踪算法。
# 以下是基于像素掩码的简化近似实现。

def breakpoint_density(pred_indices, num_classes, ignore_index=0, connectivity=1):
    """
    计算断裂点密度 (简化版)
    :param pred_indices: 预测的类别索引 (H, W), NumPy array
    :param num_classes: 类别总数
    :param ignore_index: 要忽略的类别索引
    :param connectivity: 连通性
    :return: 平均断裂点密度 (float)
    """
    assert pred_indices.ndim == 2, "Input should be (H, W)"
    total_density = 0.0
    valid_classes = 0

    for c in range(num_classes):
        if c == ignore_index:
            continue

        pred_c = (pred_indices == c)
        if np.sum(pred_c) == 0:
            continue

        # 使用 connected components 查找断开的区域
        # num_labels > 1 表示存在断裂
        num_labels, labels_im = cv2.connectedComponents(pred_c.astype(np.uint8),
                                                        connectivity=4 if connectivity == 1 else 8)  # connectivity 4 or 8

        # 断裂点数量近似为 (组件数 - 1)
        # 归一化：除以该类别的像素总数或周长可能更合理，这里用组件数作为简化指标
        breakpoints = max(0, num_labels - 1 - 1)  # -1 for background label, -1 because 1 component means no breaks

        # 简单的密度：断点数 / 类别像素数 (避免除零)
        # 更复杂的密度可能需要计算边界长度
        class_pixel_count = np.sum(pred_c)
        density_c = breakpoints / class_pixel_count if class_pixel_count > 0 else 0

        total_density += density_c
        valid_classes += 1

    if valid_classes == 0:
        return 0.0  # No valid classes to calculate density for
    else:
        # 返回的是平均每个像素的断点数，值会很小
        # 或者可以返回总断点数 / 总像素数
        return total_density / valid_classes  # Average density across classes


def intersection_count(pred_indices, num_classes, ignore_index=0):
    """
    计算层间交叉点数量 (简化版)
    :param pred_indices: 预测的类别索引 (H, W), NumPy array
    :param num_classes: 类别总数
    :param ignore_index: 要忽略的类别索引
    :return: 交叉点总数 (int)
    """
    assert pred_indices.ndim == 2, "Input should be (H, W)"
    intersection_pixels = 0

    # 查找一个像素被分配给多个非背景类别的边界区域
    # 这通常发生在形态学操作（如膨胀）后，检查重叠区域
    for r in range(pred_indices.shape[0]):
        for col in range(pred_indices.shape[1]):
            pixel_val = pred_indices[r, col]
            if pixel_val == ignore_index: continue

            # 检查邻域（例如3x3）是否存在其他非背景类别
            neighborhood = pred_indices[max(0, r - 1):min(pred_indices.shape[0], r + 2),
                           max(0, col - 1):min(pred_indices.shape[1], col + 2)]
            unique_labels = np.unique(neighborhood)
            non_bg_labels = [label for label in unique_labels if label != ignore_index]

            # 如果邻域包含多于一个非背景类别，则认为可能存在交叉
            if len(non_bg_labels) > 1:
                # 更精确的方法是检查边界像素是否同时接触两个不同类别
                # 简化：如果像素本身属于一个类，但其邻域包含另一个类，计数
                other_labels_in_neighbor = [l for l in non_bg_labels if l != pixel_val]
                if len(other_labels_in_neighbor) > 0:
                    intersection_pixels += 1
                    # 为避免重复计数，可以将此像素标记为已处理或仅计数一次

    # 返回总交叉像素数（这是一个粗略估计）
    return intersection_pixels // 2  # 每个交叉点可能被邻域检查两次


# --- 主评估函数 ---

def calculate_all_metrics(pred_indices, target_indices, num_classes, ignore_index=0):
    """
    计算所有定义的评估指标
    :param pred_indices: 预测的类别索引 (B, H, W) or (H, W), NumPy array or Torch Tensor
    :param target_indices: 真实的类别索引 (B, H, W) or (H, W), NumPy array or Torch Tensor
    :param num_classes: 类别总数
    :param ignore_index: 要忽略的类别索引 (通常是背景)
    :return: 包含所有指标结果的字典
    """
    if isinstance(pred_indices, torch.Tensor):
        pred_indices = pred_indices.detach().cpu().numpy()
    if isinstance(target_indices, torch.Tensor):
        target_indices = target_indices.detach().cpu().numpy()

    # 如果输入是批处理的，我们逐个处理或要求输入是单张图像
    # 当前实现假设输入是单张图像 (H, W) for boundary/topo metrics
    # Dice/IoU 可以处理批处理 (B, H, W)
    is_batch = pred_indices.ndim == 3

    metrics = {}

    # --- Overlap Metrics (can handle batch) ---
    metrics['dice'] = dice_coefficient(pred_indices, target_indices, num_classes, ignore_index)
    metrics['iou'] = intersection_over_union(pred_indices, target_indices, num_classes, ignore_index)

    # --- Boundary, Thickness, Topology Metrics (process per image in batch if needed) ---
    mabld_list, hausdorff_list, thickness_corr_list, breakpoint_list, intersection_list = [], [], [], [], []

    pred_batch = pred_indices if is_batch else np.expand_dims(pred_indices, axis=0)
    target_batch = target_indices if is_batch else np.expand_dims(target_indices, axis=0)

    for i in range(pred_batch.shape[0]):
        pred_single = pred_batch[i]
        target_single = target_batch[i]

        with warnings.catch_warnings():  # Suppress warnings during calculation if desired
            warnings.simplefilter("ignore", category=RuntimeWarning)  # Ignore mean of empty slice etc.
            mabld = mean_absolute_border_location_deviation(pred_single, target_single, num_classes, ignore_index)
            hd95 = hausdorff_distance_95(pred_single, target_single, num_classes, ignore_index)
            thick_corr = layer_thickness_correlation(pred_single, target_single, num_classes, ignore_index)
            # Topological metrics (optional, can be slow/complex)
            # breakpoints = breakpoint_density(pred_single, num_classes, ignore_index)
            # intersections = intersection_count(pred_single, num_classes, ignore_index)

        if not np.isnan(mabld): mabld_list.append(mabld)
        if not np.isnan(hd95): hausdorff_list.append(hd95)
        if not np.isnan(thick_corr): thickness_corr_list.append(thick_corr)
        # breakpoint_list.append(breakpoints)
        # intersection_list.append(intersections)

    # Average metrics over the batch
    metrics['mabld'] = np.mean(mabld_list) if mabld_list else np.nan
    metrics['hausdorff_95'] = np.mean(hausdorff_list) if hausdorff_list else np.nan
    metrics['thickness_correlation'] = np.mean(thickness_corr_list) if thickness_corr_list else np.nan
    # metrics['breakpoint_density'] = np.mean(breakpoint_list) if breakpoint_list else 0.0
    # metrics['intersection_count'] = np.mean(intersection_list) if intersection_list else 0.0 # Mean might not make sense, maybe sum?

    # Filter out NaN values before returning if needed, or keep them to indicate calculation issues
    # metrics = {k: v for k, v in metrics.items() if not np.isnan(v)}

    return metrics
