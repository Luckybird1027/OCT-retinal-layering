import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.unet import UNet
from utils.advanced_augmentation import standardize_size
from utils.advanced_preprocess import advanced_denoising, advanced_enhancement
from utils.dataset import OCTDataset

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


def predict_and_visualize(model, image_path, output_dir, device):
    """预测单张图像并可视化结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    original_size = img.shape[:2]

    # 调整图像大小
    img_resized = standardize_size(img)

    # 标准化
    img_norm = img_resized / 255.0

    # 应用预处理
    img_denoised = advanced_denoising(img_norm)
    img_enhanced = advanced_enhancement(img_denoised)

    # 转换为PyTorch张量
    img_tensor = torch.from_numpy(img_enhanced).float().unsqueeze(0).unsqueeze(0).to(device)

    # 预测
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        output = torch.softmax(output, dim=1)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # 获取颜色映射
    color_map = generate_color_map()

    # 创建彩色分割图
    segmentation_map = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_idx in range(len(color_map)):
        segmentation_map[pred == class_idx] = color_map[class_idx]

    # 调整回原始大小
    segmentation_map_resized = cv2.resize(segmentation_map, (original_size[1], original_size[0]),
                                          interpolation=cv2.INTER_NEAREST)

    # 生成原始图像的彩色版本用于可视化
    original_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # 创建混合图像 (原始图像 + 半透明分割图)
    alpha = 0.5
    blend = cv2.addWeighted(original_rgb, 1 - alpha, segmentation_map_resized, alpha, 0)

    # 保存结果
    cv2.imwrite(os.path.join(output_dir, 'original.png'), img)
    cv2.imwrite(os.path.join(output_dir, 'segmentation.png'), segmentation_map_resized)
    cv2.imwrite(os.path.join(output_dir, 'blend.png'), blend)

    # 可视化结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(segmentation_map_resized)
    plt.title('分割结果')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(blend)
    plt.title('融合结果')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'visualization.png'))
    plt.show()

    return segmentation_map_resized, blend


def evaluate_test_set(model, test_loader, device, output_dir='results'):
    """评估测试集并保存结果"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取颜色映射
    color_map = generate_color_map()

    # 计算Dice系数
    dice_scores = []

    model.eval()
    with torch.no_grad():
        for i, (image, mask) in enumerate(tqdm(test_loader, desc='Evaluating')):
            # 将数据移至设备
            image = image.to(device)
            mask = mask.to(device)

            # 预测
            output = model(image)
            pred = torch.argmax(output, dim=1)

            # 计算Dice系数
            dice_score = 0.0
            for class_idx in range(11):
                pred_class = (pred == class_idx).float()
                mask_class = (mask == class_idx).float()

                intersection = (pred_class * mask_class).sum()
                union = pred_class.sum() + mask_class.sum()

                if union > 0:
                    dice_score += (2 * intersection) / (union + 1e-6)

            dice_score /= 11  # 11个类别的平均值
            dice_scores.append(dice_score.item())

            # 保存第一个批次的一些示例结果
            if i == 0:
                for j in range(min(4, image.size(0))):
                    img = image[j].squeeze().cpu().numpy()
                    msk = mask[j].cpu().numpy()
                    prd = pred[j].cpu().numpy()

                    # 创建彩色分割图
                    msk_rgb = np.zeros((msk.shape[0], msk.shape[1], 3), dtype=np.uint8)
                    prd_rgb = np.zeros((prd.shape[0], prd.shape[1], 3), dtype=np.uint8)

                    for class_idx in range(len(color_map)):
                        msk_rgb[msk == class_idx] = color_map[class_idx]
                        prd_rgb[prd == class_idx] = color_map[class_idx]

                    # 创建可视化结果
                    plt.figure(figsize=(15, 5))

                    plt.subplot(1, 3, 1)
                    plt.imshow(img, cmap='gray')
                    plt.title('原始图像')
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(msk_rgb)
                    plt.title('真实标签')
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(prd_rgb)
                    plt.title('预测结果')
                    plt.axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'example_{j + 1}.png'))
                    plt.close()

    # 计算平均Dice系数
    mean_dice = np.mean(dice_scores)
    print(f'平均Dice系数: {mean_dice:.4f}')

    # 保存Dice系数
    with open(os.path.join(output_dir, 'dice_scores.txt'), 'w') as f:
        f.write(f'平均Dice系数: {mean_dice:.4f}\n')
        for i, dice in enumerate(dice_scores):
            f.write(f'样本 {i + 1}: {dice:.4f}\n')

    return mean_dice, dice_scores


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 加载模型
    model = UNet(in_channels=1, out_channels=11).to(device)
    model.load_state_dict(torch.load('train/checkpoints/best_model.pth', map_location=device, weights_only=True))

    # 设置测试数据
    test_images_dir = 'data/SJTU/test/img'
    test_masks_dir = 'data/SJTU/test/mask'
    test_dataset = OCTDataset(test_images_dir, test_masks_dir, transform=None)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # 评估测试集
    evaluate_test_set(model, test_loader, device, output_dir='results')

    # 预测单个样本
    sample_image_path = 'data/RetinalOCT_Dataset/processed/test/images/sample_image.jpg'
    if os.path.exists(sample_image_path):
        predict_and_visualize(model, sample_image_path, 'results/sample', device)

    print('预测完成!')


if __name__ == '__main__':
    main()
