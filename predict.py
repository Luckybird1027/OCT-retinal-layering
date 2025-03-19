import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from model.unet import UNet
from utils.dataset import create_dataloader

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_color_map(num_classes=11):
    """生成用于可视化的颜色映射"""
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)

    # 设置特定的颜色 (RGB格式)
    # 背景类 - 黑色
    color_map[0] = [0, 0, 0]

    # 视网膜不同层次的颜色 - 使用容易区分的颜色
    color_map[1] = [255, 0, 0]  # 红色 - RNFL
    color_map[2] = [0, 255, 0]  # 绿色 - GCL
    color_map[3] = [0, 0, 255]  # 蓝色 - IPL
    color_map[4] = [255, 255, 0]  # 黄色 - INL
    color_map[5] = [0, 255, 255]  # 青色 - OPL
    color_map[6] = [255, 0, 255]  # 紫色 - ONL
    color_map[7] = [128, 0, 0]  # 深红色 - IS/OS
    color_map[8] = [0, 128, 0]  # 深绿色 - RPE
    color_map[9] = [0, 0, 128]  # 深蓝色 - Choroid
    color_map[10] = [128, 128, 0]  # 深黄色 - VE

    return color_map


def predict_and_visualize_single(model, image, mask, output_path, device):
    """预测单张图像并可视化结果"""
    # 预测
    output = model(image)
    output = torch.softmax(output, dim=1)
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    mask = mask.squeeze().cpu().numpy()
    image_np = image.squeeze().cpu().numpy()  # 获取 NumPy 数组

    # 获取颜色映射
    color_map = generate_color_map()

    # 创建彩色分割图
    segmentation_map = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_idx in range(len(color_map)):
        segmentation_map[pred == class_idx] = color_map[class_idx]

    # 创建彩色标签图
    mask_map = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_idx in range(len(color_map)):
        mask_map[mask == class_idx] = color_map[class_idx]

    # 生成原始图像的彩色版本用于可视化
    # 将图像数据缩放回 0-255 并转换为 uint8
    original_img_scaled = (image_np * 255).astype(np.uint8)
    original_rgb = cv2.cvtColor(original_img_scaled, cv2.COLOR_GRAY2RGB)

    # 创建混合图像 (原始图像 + 半透明分割图)
    alpha = 0.5
    blend = cv2.addWeighted(original_rgb, 1 - alpha, segmentation_map, alpha, 0)

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    # 保存结果
    cv2.imwrite(os.path.join(output_path, 'original.png'), original_img_scaled)
    cv2.imwrite(os.path.join(output_path, 'segmentation.png'), segmentation_map)
    cv2.imwrite(os.path.join(output_path, 'blend.png'), blend)
    cv2.imwrite(os.path.join(output_path, 'mask.png'), mask_map)

    # 可视化结果
    plt.figure(figsize=(14, 10), dpi=300)

    plt.subplot(2, 2, 1)
    plt.imshow(original_img_scaled, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(mask_map)
    plt.title('标准标签')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(segmentation_map)
    plt.title('分割结果')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(blend)
    plt.title('融合结果')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'visualization.png'))
    plt.close()


def predict_folder(model, data_loader, output_dir, device):
    """预测指定路径的所有图像并保存结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Predicting')
        for i, (image, mask) in enumerate(pbar):
            image = image.to(device)
            mask = mask.to(device)  # 虽然mask只用于可视化，但保持一致性

            image_name = f'image_{i}'
            predict_and_visualize_single(model, image, mask, os.path.join(output_dir, image_name), device)

    print('所有图像预测完成!')


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载模型
    model = UNet(in_channels=1, out_channels=11).to(device)
    model.load_state_dict(torch.load('train/checkpoints/best_model.pth', map_location=device, weights_only=True))

    # 设置预测数据路径
    predict_images_dir = 'data/SJTU/test/img'
    predict_masks_dir = 'data/SJTU/test/mask'
    output_results_dir = 'results/predictions'

    # 创建数据加载器
    data_loader = create_dataloader(predict_images_dir, predict_masks_dir, batch_size=1, shuffle=False)

    # 预测指定路径的所有图像
    predict_folder(model, data_loader, output_results_dir, device)

    print('预测完成!')


if __name__ == '__main__':
    main()
