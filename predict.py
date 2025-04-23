import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # 用于处理和显示表格
import torch
from tqdm import tqdm

from model.dataset import create_dataloader
from model.unet import UNet
from utils.metrics import calculate_all_metrics
from utils.visualization import generate_color_map, create_difference_map


# 设置matplotlib支持中文 (如果需要)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


def predict_and_visualize_single(model, image_tensor, mask_tensor, output_path, device, num_classes=11):
    """预测单张图像，计算指标并可视化结果"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        # mask_tensor = mask_tensor.to(device) # mask 仅用于评估，可以在 CPU 上处理

        # 预测
        output = model(image_tensor)  # (1, C, H, W)
        output = torch.softmax(output, dim=1)
        pred_indices = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # (H, W)

    mask_indices = mask_tensor.squeeze().cpu().numpy()  # (H, W)
    # 注意：image_tensor 已经是 [0, 1] 范围的 float tensor
    image_np = image_tensor.squeeze().cpu().numpy()  # (H, W)

    # --- 计算指标 ---
    metrics_results = calculate_all_metrics(pred_indices, mask_indices, num_classes=num_classes, ignore_index=0)

    # --- 可视化 ---
    color_map = generate_color_map(num_classes)

    # 创建彩色分割图 (预测)
    segmentation_map = np.zeros((pred_indices.shape[0], pred_indices.shape[1], 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        segmentation_map[pred_indices == class_idx] = color_map[class_idx]

    # 创建彩色标签图 (真实)
    mask_map = np.zeros((mask_indices.shape[0], mask_indices.shape[1], 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        mask_map[mask_indices == class_idx] = color_map[class_idx]

    # 生成原始图像的彩色版本 (从 [0, 1] float 转换)
    original_img_scaled = (image_np * 255).astype(np.uint8)
    original_rgb = cv2.cvtColor(original_img_scaled, cv2.COLOR_GRAY2RGB)

    # 创建混合图像 (原始图像 + 半透明分割图)
    blend_alpha = 0.5
    blend = cv2.addWeighted(original_rgb, 1 - blend_alpha, segmentation_map, blend_alpha, 0)

    # 创建差异对比图 (使用导入的函数)
    diff_map = create_difference_map(original_rgb, pred_indices, mask_indices, alpha=0.6)

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # --- 保存结果 ---
    original_save_path = os.path.join(output_path, 'original.png')
    mask_save_path = os.path.join(output_path, 'mask_ground_truth.png')
    segmentation_save_path = os.path.join(output_path, 'segmentation_predicted.png')
    blend_save_path = os.path.join(output_path, 'blend_prediction_overlay.png')
    diff_map_save_path = os.path.join(output_path, 'difference_map.png')
    metrics_file_path = os.path.join(output_path, 'metrics.txt')
    combined_plot_path = os.path.join(output_path, 'visualization_summary.png')

    cv2.imwrite(original_save_path, original_img_scaled)
    cv2.imwrite(mask_save_path, mask_map)
    cv2.imwrite(segmentation_save_path, segmentation_map)
    cv2.imwrite(blend_save_path, blend)
    cv2.imwrite(diff_map_save_path, cv2.cvtColor(diff_map, cv2.COLOR_RGB2BGR))

    # 保存指标到文本文件
    with open(metrics_file_path, 'w') as f:
        f.write("Evaluation Metrics:\n")
        f.write("=====================\n")
        # 格式化 NaN 值为 'N/A' 或其他标识符
        for key, value in metrics_results.items():
            f.write(f"{key}: {value:.4f}\n" if isinstance(value, (float, np.float_)) and not np.isnan(
                value) else f"{key}: {value if not np.isnan(value) else 'N/A'}\n")

    # --- 创建组合可视化图 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=200)  # 调整布局为 2x3
    axes = axes.ravel()  # 展平以便索引

    axes[0].imshow(original_img_scaled, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask_map)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')

    axes[2].imshow(segmentation_map)
    axes[2].set_title('Predicted Segmentation')
    axes[2].axis('off')

    axes[3].imshow(blend)
    axes[3].set_title('Prediction Overlay')
    axes[3].axis('off')

    axes[4].imshow(diff_map)
    axes[4].set_title('Difference Map (Correct:Green, Error:Red)')
    axes[4].axis('off')

    # 在最后一个子图中显示指标文本
    axes[5].axis('off')
    metrics_text = "Metrics:\n" + "\n".join([f"{k}: {v:.3f}" if isinstance(v, (float, np.float_)) and not np.isnan(
        v) else f"{k}: {v if not np.isnan(v) else 'N/A'}" for k, v in metrics_results.items()])
    axes[5].text(0.05, 0.95, metrics_text, ha='left', va='top', fontsize=11, wrap=True,
                 bbox=dict(boxstyle='round,pad=0.8', fc='wheat', alpha=0.5))
    axes[5].set_title('Metrics Summary')

    plt.tight_layout()
    plt.savefig(combined_plot_path)
    plt.close(fig)  # 关闭图形，防止内存泄漏

    return metrics_results, metrics_file_path, combined_plot_path


def predict_folder(model, data_loader, output_dir, device, num_classes=11):
    """预测指定路径的所有图像，保存结果和指标，并计算总体指标"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    all_image_metrics = []  # 存储每个图像的指标字典
    image_names_list = []  # 存储图像名称或索引

    model.eval()
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Predicting and Evaluating')
        for i, (image, mask) in enumerate(pbar):
            # 假设 data_loader 返回原始图像名或索引，如果不是，则生成一个
            # image_name = data_loader.dataset.img_files[i] # 如果数据集暴露了这个信息
            image_name = f'image_{i:04d}'  # 否则使用索引
            image_names_list.append(image_name)

            output_path_single = os.path.join(output_dir, image_name)

            # 预测、可视化并获取单图指标和保存路径
            metrics_single, metrics_file_path, combined_plot_path = predict_and_visualize_single(
                model, image, mask, output_path_single, device, num_classes
            )
            metrics_single['image_name'] = image_name  # 添加图像名到字典
            all_image_metrics.append(metrics_single)

            # 使用 pbar.write() 在循环中打印日志，避免干扰进度条
            pbar.write(f"Metrics saved to {metrics_file_path}")
            pbar.write(f"Combined visualization saved to {combined_plot_path}")

    print('\n--- Individual image predictions complete ---')

    # --- 计算并保存总体指标 ---
    if not all_image_metrics:
        print("No images were processed.")
        return

    # 使用 Pandas DataFrame 进行聚合和保存
    metrics_df = pd.DataFrame(all_image_metrics)
    metrics_df.set_index('image_name', inplace=True)

    # 计算平均值 (忽略 NaN)
    average_metrics = metrics_df.mean(axis=0, skipna=True)
    std_metrics = metrics_df.std(axis=0, skipna=True)

    # 创建摘要 DataFrame
    summary_df = pd.DataFrame({'Mean': average_metrics, 'StdDev': std_metrics})

    print("\n--- Overall Evaluation Metrics Summary ---")
    print(summary_df)

    # 保存详细指标到 CSV
    detailed_csv_path = os.path.join(output_dir, 'all_images_metrics.csv')
    metrics_df.to_csv(detailed_csv_path)
    print(f"\nDetailed metrics for all images saved to: {detailed_csv_path}")

    # 保存摘要指标到 CSV
    summary_csv_path = os.path.join(output_dir, 'summary_metrics.csv')
    summary_df.to_csv(summary_csv_path)
    print(f"Summary metrics saved to: {summary_csv_path}")

    print('\nPrediction and evaluation finished!')


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # --- 配置参数 --- (可以替换为 argparse)
    model_path = 'train/checkpoints/best_model.pth'  # 或者 'final_model.pth'
    predict_images_dir = 'data/SJTU/test/img'
    predict_masks_dir = 'data/SJTU/test/mask'
    output_results_dir = 'results/predictions_with_metrics'
    batch_size = 1  # 评估时通常 batch_size=1
    num_classes = 11  # 确保与模型训练时一致
    img_size = (720, 992)  # 确保与模型训练/数据集定义一致
    preprocess_in_loader = True  # 是否在 dataloader 中进行预处理

    # 加载模型
    model = UNet(in_channels=1, out_channels=num_classes).to(device)
    # 兼容旧版和新版保存的模型
    try:
        # 尝试加载 state_dict (推荐)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception:
        # 尝试加载包含字典的旧格式
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 如果只是模型本身
            try:
                model.load_state_dict(checkpoint)
            except TypeError: # Handle case where checkpoint might be the model object itself
                print("Warning: Loaded checkpoint might be the entire model object, not state_dict. This is not recommended.")
                # If you are sure it's the model object and compatible:
                # model = checkpoint
                # Or raise an error if this format is unexpected:
                raise ValueError("Loaded checkpoint is not a state_dict or a recognized dictionary format.")

    print(f"模型已从 {model_path} 加载")

    # 创建数据加载器 (确保使用与训练匹配的设置，特别是 img_size 和 preprocess)
    # 注意：评估时 shuffle 通常为 False
    data_loader = create_dataloader(
        images_dir=predict_images_dir,
        masks_dir=predict_masks_dir,
        img_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        preprocess=preprocess_in_loader  # 确保与训练时一致
    )
    print(f"数据加载器已创建，共 {len(data_loader.dataset)} 张图像")

    # 预测文件夹中的所有图像并进行评估
    predict_folder(model, data_loader, output_results_dir, device, num_classes)

    print('\n预测和评估流程完成!')


if __name__ == '__main__':
    main()
