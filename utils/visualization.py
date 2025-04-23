import cv2
import numpy as np
import torch
import torchvision


def generate_color_map(num_classes=11):
    """
    生成用于可视化的颜色映射 (RGB格式)
    """
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    # 背景类 - 黑色
    color_map[0] = [0, 0, 0]
    # 视网膜不同层次的颜色 - 使用容易区分的颜色
    color_map[1] = [255, 0, 0]  # 红色 - RNFL
    color_map[2] = [0, 255, 0]  # 绿色 - GCL+IPL (合并或根据实际情况调整)
    color_map[3] = [0, 0, 255]  # 蓝色 - INL
    color_map[4] = [255, 255, 0]  # 黄色 - OPL
    color_map[5] = [0, 255, 255]  # 青色 - ONL
    color_map[6] = [255, 0, 255]  # 紫色 - IS/OS (光感受器层)
    color_map[7] = [128, 0, 0]  # 深红色 - RPE (视网膜色素上皮)
    color_map[8] = [0, 128, 0]  # 深绿色 - Choroid (脉络膜)
    color_map[9] = [0, 0, 128]  # 深蓝色 - Vitreous (玻璃体) - 可能需要调整
    color_map[10] = [128, 128, 0]  # 橄榄色 - Sclera (巩膜) - 可能需要调整
    # 注意：这里的类别名称仅为示例，请根据你的数据集标签含义进行匹配
    return color_map


def create_difference_map(original_img_rgb, pred_indices, mask_indices, alpha=0.5):
    """
    创建差异对比图 (绿色正确, 红色错误)
    :param original_img_rgb: 原始 RGB 图像 (H, W, 3)
    :param pred_indices: 预测的类别索引 (H, W)
    :param mask_indices: 真实的类别索引 (H, W)
    :param alpha: 叠加透明度
    :return: 混合后的差异图 (H, W, 3)
    """
    diff_overlay = np.zeros_like(original_img_rgb, dtype=np.uint8)
    # 正确预测的像素 (非背景)
    correct_pixels = (pred_indices == mask_indices) & (mask_indices != 0)
    # 错误预测的像素 (非背景真实标签)
    incorrect_pixels = (pred_indices != mask_indices) & (mask_indices != 0)

    diff_overlay[correct_pixels] = [0, 255, 0]  # 绿色
    diff_overlay[incorrect_pixels] = [255, 0, 0]  # 红色

    # 创建一个掩码，只在有颜色标注的地方进行混合
    colored_mask = (correct_pixels | incorrect_pixels)

    # 混合
    blended_diff = original_img_rgb.copy()
    # 只在有颜色的区域应用混合
    blended_diff[colored_mask] = cv2.addWeighted(
        original_img_rgb[colored_mask], 1 - alpha,
        diff_overlay[colored_mask], alpha, 0
    )
    return blended_diff


def create_visualization_grid(image_tensor, mask_tensor, pred_tensor, num_classes, max_images=4):
    """
    为 TensorBoard 创建可视化网格 (类似 predict.py 中的组合图)
    :param image_tensor: 原始图像张量 (B, 1, H, W), 范围 [0, 1]
    :param mask_tensor: 真实标签张量 (B, H, W), 类别索引
    :param pred_tensor: 预测标签张量 (B, H, W), 类别索引
    :param num_classes: 类别数
    :param max_images: 最多显示多少张图片
    :return: Torch tensor (3, H*N_rows, W*N_cols) 格式的图像网格
    """
    batch_size = image_tensor.size(0)
    num_display = min(batch_size, max_images)

    # 确保张量在 CPU 上并转换为 NumPy
    image_np = image_tensor[:num_display].cpu().numpy().squeeze(1)  # (N, H, W)
    mask_np = mask_tensor[:num_display].cpu().numpy()  # (N, H, W)
    pred_np = pred_tensor[:num_display].cpu().numpy()  # (N, H, W)

    color_map = generate_color_map(num_classes)
    grid_images = []

    for i in range(num_display):
        img_single = (image_np[i] * 255).astype(np.uint8)
        mask_single = mask_np[i]
        pred_single = pred_np[i]

        # 1. 原始图像 (灰度转 RGB)
        original_rgb = cv2.cvtColor(img_single, cv2.COLOR_GRAY2RGB)

        # 2. 真实标签图
        mask_map = np.zeros_like(original_rgb)
        for class_idx in range(num_classes):
            mask_map[mask_single == class_idx] = color_map[class_idx]

        # 3. 预测分割图
        segmentation_map = np.zeros_like(original_rgb)
        for class_idx in range(num_classes):
            segmentation_map[pred_single == class_idx] = color_map[class_idx]

        # 4. 混合图 (原始 + 预测)
        blend_alpha = 0.5
        blend = cv2.addWeighted(original_rgb, 1 - blend_alpha, segmentation_map, blend_alpha, 0)

        # 5. 差异图
        diff_map = create_difference_map(original_rgb, pred_single, mask_single, alpha=0.6)

        # 将所有图像转换为 (3, H, W) 的 Torch Tensor 并添加到列表
        grid_images.append(torch.from_numpy(original_rgb.transpose(2, 0, 1)))
        grid_images.append(torch.from_numpy(mask_map.transpose(2, 0, 1)))
        grid_images.append(torch.from_numpy(segmentation_map.transpose(2, 0, 1)))
        grid_images.append(torch.from_numpy(blend.transpose(2, 0, 1)))
        grid_images.append(torch.from_numpy(diff_map.transpose(2, 0, 1)))
        # 可以添加一个空白图像占位符，如果需要 2x3 布局
        placeholder = torch.zeros_like(torch.from_numpy(original_rgb.transpose(2, 0, 1)))
        grid_images.append(placeholder)

    # 使用 torchvision 创建网格
    # 每行显示 3 张图 (Original, Mask, Pred / Blend, Diff, Placeholder)
    # 如果 num_display=4, 会有 4 行，每行 6 张图
    combined_grid = torchvision.utils.make_grid(grid_images, nrow=6, padding=2, normalize=False)  # 每行6张图

    return combined_grid
