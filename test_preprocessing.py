import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.advanced_preprocess import (
    anisotropic_diffusion,
    bm3d_denoising,
    phase_congruency_enhancement,
    butterworth_highpass_filter,
    advanced_denoising,
    advanced_enhancement
)
from utils.advanced_augmentation import (
    tps_transform,
    perspective_transform,
    dynamic_gamma_correction,
    mixed_noise_injection,
    nsct_enhancement,
    advanced_augmentation
)

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_comparison(images, titles, main_title="OCT图像处理效果对比", figsize=(15, 10)):
    """
    绘制多张图像的对比图
    
    参数:
        images: 图像列表
        titles: 标题列表
        main_title: 主标题
        figsize: 图像大小
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)

    if n == 1:
        axes = [axes]

    for i, (img, title) in enumerate(zip(images, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.suptitle(main_title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


def test_preprocessing(image_path):
    """
    测试OCT图像预处理流程
    
    参数:
        image_path: 图像路径
    """
    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 归一化
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))

    # 1. 测试各向异性扩散
    print("正在应用各向异性扩散...")
    img_ad = anisotropic_diffusion(img_norm)

    # 2. 测试BM3D去噪
    print("正在应用BM3D去噪...")
    img_bm3d = bm3d_denoising(img_ad)

    # 3. 测试相位一致性增强
    print("正在应用相位一致性增强...")
    img_pc = phase_congruency_enhancement(img_bm3d, threshold=0.45, nscale=4)

    # 4. 测试巴特沃斯高通滤波器
    print("正在应用巴特沃斯高通滤波器...")
    img_hp = butterworth_highpass_filter(img_pc, cutoff=0.4, order=5)

    # 5. 测试完整的高级预处理流程
    print("正在应用完整预处理流程...")
    img_denoised = advanced_denoising(img_norm)
    img_enhanced = advanced_enhancement(img_denoised)

    # 显示预处理结果
    print("正在生成预处理对比图...")
    # 去噪步骤对比
    denoising_fig = plot_comparison(
        [img_norm, img_ad, img_bm3d],
        ["原始图像", "各向异性扩散", "BM3D去噪"],
        "OCT图像去噪效果对比"
    )
    denoising_fig.savefig("./chart/denoising_comparison.png", dpi=300)

    # 增强步骤对比
    enhancement_fig = plot_comparison(
        [img_bm3d, img_pc, img_hp],
        ["去噪后图像", "相位一致性增强", "高通滤波增强"],
        "OCT图像增强效果对比"
    )
    enhancement_fig.savefig("./chart/enhancement_comparison.png", dpi=300)

    # 完整流程对比
    complete_fig = plot_comparison(
        [img_norm, img_denoised, img_enhanced],
        ["原始图像", "去噪结果", "增强结果"],
        "OCT图像完整预处理流程对比"
    )
    complete_fig.savefig("./chart/complete_preprocessing_comparison.png", dpi=300)

    return img_norm, img_enhanced


def test_augmentation(img):
    """
    测试OCT图像增强流程
    
    参数:
        img: 输入图像
    """
    # 确保图像在[0,1]范围内
    if img.max() > 1.0:
        img = img / 255.0

    # 1. 测试TPS形变
    print("正在应用TPS形变...")
    img_tps = tps_transform(img, regularization=0.3)

    # 2. 测试透视变换
    print("正在应用透视变换...")
    img_perspective = perspective_transform(img)

    # 3. 测试伽马校正
    print("正在应用伽马校正...")
    img_gamma = dynamic_gamma_correction(img, gamma_range=(0.6, 1.8))

    # 4. 测试混合噪声注入
    print("正在应用混合噪声注入...")
    img_noise = mixed_noise_injection(img)

    # 5. 测试频域增强
    print("正在应用频域增强...")
    img_freq = nsct_enhancement(img)

    # 6. 测试完整的增强流程
    print("正在应用完整增强流程...")
    img_aug, _ = advanced_augmentation(img)

    # 显示增强结果
    print("正在生成增强对比图...")
    # 空间变换对比
    spatial_fig = plot_comparison(
        [img, img_tps, img_perspective],
        ["原始图像", "TPS形变", "透视变换"],
        "OCT图像空间变换效果对比"
    )
    spatial_fig.savefig("./chart/spatial_augmentation_comparison.png", dpi=300)

    # 强度变换对比
    intensity_fig = plot_comparison(
        [img, img_gamma, img_noise],
        ["原始图像", "伽马校正", "混合噪声注入"],
        "OCT图像强度变换效果对比"
    )
    intensity_fig.savefig("./chart/intensity_augmentation_comparison.png", dpi=300)

    # 频域与完整增强对比
    complete_aug_fig = plot_comparison(
        [img, img_freq, img_aug],
        ["原始图像", "频域增强", "完整增强"],
        "OCT图像频域与完整增强效果对比"
    )
    complete_aug_fig.savefig("./chart/complete_augmentation_comparison.png", dpi=300)


def generate_all_stages_comparison(image_path):
    """
    生成所有处理阶段的完整对比图
    
    参数:
        image_path: 图像路径
    """
    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))

    # 分别应用每个处理步骤
    print("正在生成所有处理阶段的对比图...")
    img_ad = anisotropic_diffusion(img_norm)
    img_bm3d = bm3d_denoising(img_ad)
    img_pc = phase_congruency_enhancement(img_bm3d)
    img_hp = butterworth_highpass_filter(img_pc)
    img_aug, _ = advanced_augmentation(img_hp)

    # 生成完整处理流程对比图
    stages = [img_norm, img_ad, img_bm3d, img_pc, img_hp, img_aug]
    titles = ["原始图像", "各向异性扩散", "BM3D去噪",
              "相位一致性增强", "高通滤波增强", "数据增强结果"]

    # 创建3x2网格布局
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (img, title) in enumerate(zip(stages, titles)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.suptitle("OCT图像完整处理流程各阶段效果", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.savefig("./chart/all_stages_comparison.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    # 设置图像路径
    image_path = 'data/RetinalOCT_Dataset/raw/train/DR/dr_train_1001.jpg'

    if not os.path.exists(image_path):
        print(f"错误：找不到图像文件 {image_path}")
        exit(1)

    # 测试预处理
    print("\n===== 开始测试预处理效果 =====")
    img_norm, img_enhanced = test_preprocessing(image_path)

    # 测试数据增强
    print("\n===== 开始测试数据增强效果 =====")
    test_augmentation(img_enhanced)

    # 生成完整流程对比图
    print("\n===== 生成完整处理流程对比图 =====")
    generate_all_stages_comparison(image_path)

    print("\n所有测试完成！结果已保存为PNG图像。")
