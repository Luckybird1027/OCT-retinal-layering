import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt # 不再需要，仅用 cv2 保存

# 导入你的预处理和增强函数
from utils.preprocess import (
    anisotropic_diffusion,
    bm3d_denoising,
    phase_congruency_enhancement,
    butterworth_highpass_filter,
    standardize_size,
)
from utils.augmentation import (
    tps_transform,
    dynamic_gamma_correction,
    mixed_noise_injection,
    rotate_image,
    random_crop,
    horizontal_flip,
    adjust_brightness_contrast,
    spatial_domain_augmentation,
    intensity_domain_augmentation,
)

# --- 配置 ---
# !!! 请将 'path/to/your/sample_oct_image.png' 替换为你的示例OCT图像路径 !!!
# 如果没有图像，可以将 use_synthetic_image 设置为 True
SAMPLE_IMAGE_PATH = './data/SJTU/train/img/3_L_07.bmp'
OUTPUT_DIR_BASE = './chart/output_test_images_structured' # 基础输出目录
USE_SYNTHETIC_IMAGE = False # 如果没有真实图像，设置为True
SYNTHETIC_SIZE = (256, 512) # 合成图像大小 (height, width)

# --- 辅助函数 ---
def create_synthetic_image(size=(256, 512)):
    """创建一个简单的合成灰度图像用于测试"""
    img = np.zeros(size, dtype=np.float32)
    center_y, center_x = size[0] // 2, size[1] // 2
    # 添加一些特征
    cv2.rectangle(img, (center_x - 50, center_y - 30), (center_x + 50, center_y + 30), 0.8, -1)
    cv2.line(img, (20, 20), (size[1] - 20, size[0] - 20), 0.5, 2)
    cv2.circle(img, (center_x, center_y), 60, 0.6, 3)
    # 添加渐变背景
    gradient = np.linspace(0.1, 0.4, size[1])
    img += gradient
    img = np.clip(img, 0, 1)
    # 添加一点噪声
    noise = np.random.normal(0, 0.05, size)
    img = np.clip(img + noise, 0, 1)
    return (img * 255).astype(np.uint8)

def save_image(img_data, method_subdir, filename, base_output_dir):
    """保存图像到指定的基础目录下的方法子目录中"""
    output_subdir = os.path.join(base_output_dir, method_subdir)
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)

    # 转换数据类型并归一化到 [0, 255] uint8
    if img_data.dtype != np.uint8:
        if img_data.max() <= 1.0 and img_data.min() >= 0:
            img_data = (img_data * 255).astype(np.uint8)
        elif img_data.max() > 255 or img_data.min() < 0: # 需要归一化
             min_val, max_val = img_data.min(), img_data.max()
             if max_val > min_val:
                 img_data = ((img_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
             else: # 图像是纯色
                 img_data = np.full_like(img_data, img_data.item(0) if img_data.size > 0 else 0, dtype=np.uint8)
        else: # 已经是类似uint8范围的float，直接转换
             img_data = img_data.astype(np.uint8)


    # 保存图像
    filepath = os.path.join(output_subdir, f"{filename}.png")
    try:
        if len(img_data.shape) == 3 and img_data.shape[2] == 3:
             cv2.imwrite(filepath, cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)) # OpenCV 使用 BGR
        elif len(img_data.shape) == 3 and img_data.shape[2] == 1:
             cv2.imwrite(filepath, img_data[:, :, 0])
        else: # 假设是 2D 灰度图
             cv2.imwrite(filepath, img_data)
        print(f"Saved: {filepath}")
    except Exception as e:
        print(f"Error saving {filepath}: {e}")


# --- 主程序 ---
if __name__ == "__main__":
    # 加载或创建图像
    if USE_SYNTHETIC_IMAGE or not os.path.exists(SAMPLE_IMAGE_PATH) or os.path.isdir(SAMPLE_IMAGE_PATH):
        print("Using synthetic image.")
        original_img = create_synthetic_image(SYNTHETIC_SIZE)
        if original_img is None:
             print("Error: Could not create synthetic image. Exiting.")
             exit()
        sample_path_valid = False
    else:
        print(f"Loading image from: {SAMPLE_IMAGE_PATH}")
        original_img = cv2.imread(SAMPLE_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            print(f"Error: Could not load image at {SAMPLE_IMAGE_PATH}. Trying synthetic image.")
            original_img = create_synthetic_image(SYNTHETIC_SIZE)
            if original_img is None:
                 print("Error: Could not create synthetic image. Exiting.")
                 exit()
            sample_path_valid = False
        else:
            sample_path_valid = True

    print(f"Using image of size: {original_img.shape}")

    # 预处理函数通常需要 [0, 1] 范围的浮点数
    img_float = original_img.astype(np.float32) / 255.0

    # 保存原始图像在基础目录下
    save_image(original_img, "00_original", "original", OUTPUT_DIR_BASE)

    print("\n--- Testing Preprocessing Functions (Saving to Subdirs) ---")
    # 1. 各向异性扩散
    method_name = "01_anisotropic_diffusion"
    print(f"Testing {method_name}...")
    for niter_val in [5, 15]:
        for kappa_val in [0.10, 0.15, 0.20]:
            img_ad = anisotropic_diffusion(img_float.copy(), niter=niter_val, kappa=kappa_val)
            filename = f"ad_niter{niter_val}_kappa{kappa_val:.2f}"
            save_image(img_ad, method_name, filename, OUTPUT_DIR_BASE)

    # 2. BM3D去噪
    method_name = "02_bm3d_denoising"
    print(f"Testing {method_name}...")
    # BM3D 需要输入有噪声的图像效果才明显，这里我们在原图上加一点高斯噪声再处理
    noise_std_dev = 15 / 255.0
    noisy_img_float = np.clip(img_float + np.random.normal(0, noise_std_dev, img_float.shape), 0, 1).astype(np.float32)
    save_image(noisy_img_float, method_name, "input_noisy", OUTPUT_DIR_BASE) # 保存带噪声的输入

    for sigma_val in [10/255.0, 15/255.0, 25/255.0]:
        try:
            # 使用加噪后的图像进行BM3D测试
            img_bm3d = bm3d_denoising(noisy_img_float.copy(), sigma_psd=sigma_val)
            filename = f"bm3d_sigma{sigma_val*255:.0f}"
            save_image(img_bm3d, method_name, filename, OUTPUT_DIR_BASE)
        except Exception as e:
            print(f"BM3D failed for sigma={sigma_val*255:.0f}: {e}. Skipping this value.")

    # 3. 相位一致性增强
    method_name = "03_phase_congruency"
    print(f"Testing {method_name}...")
    for nscale_val in [3, 4, 5]:
        for threshold_val in [0.35, 0.45, 0.55]:
            try:
                img_pc = phase_congruency_enhancement(img_float.copy(), threshold=threshold_val, nscale=nscale_val)
                filename = f"pc_nscale{nscale_val}_thresh{threshold_val:.2f}"
                save_image(img_pc, method_name, filename, OUTPUT_DIR_BASE)
            except Exception as e: # Phasepack might raise errors on some inputs/params
                print(f"Phase Congruency failed for nscale={nscale_val}, thresh={threshold_val:.2f}: {e}. Skipping.")

    # 4. 巴特沃斯高通滤波
    method_name = "04_butterworth_hpf"
    print(f"Testing {method_name}...")
    # 注意：cutoff太高或太低，以及order的影响
    for cutoff_val in [0.1, 0.4, 0.7]: # 截止频率 (0.4 是项目指定值)
        for order_val in [2, 5, 8]: # 滤波器阶数 (5 是项目指定值)
            img_hpf = butterworth_highpass_filter(img_float.copy(), cutoff=cutoff_val, order=order_val)
            filename = f"hpf_cutoff{cutoff_val:.1f}_order{order_val}"
            save_image(img_hpf, method_name, filename, OUTPUT_DIR_BASE)


    print("\n--- Testing Augmentation Functions (Saving to Subdirs) ---")
    # 增强函数操作 uint8 或 float [0, 1]

    # 5. TPS 形变
    method_name = "05_tps_transform"
    print(f"Testing {method_name}...")
    for num_points in [9, 16, 25]: # 控制点数量 (需要是平方数)
        for std_dev_val in [5, 10, 20]: # 控制点移动标准差
            img_tps = tps_transform(original_img.copy(), num_control_points=num_points, std_dev=std_dev_val, regularization=0.3)
            filename = f"tps_points{num_points}_std{std_dev_val}"
            save_image(img_tps, method_name, filename, OUTPUT_DIR_BASE)

    # 6. 动态 Gamma 校正
    method_name = "06_dynamic_gamma"
    print(f"Testing {method_name}...")
    for gamma_val in [0.6, 1.0, 1.8]: # 项目范围 [0.6, 1.8]
        img_gamma = dynamic_gamma_correction(img_float.copy(), gamma_range=(gamma_val, gamma_val)) # 固定gamma值
        filename = f"gamma_{gamma_val:.1f}"
        save_image(img_gamma, method_name, filename, OUTPUT_DIR_BASE)

    # 7. 混合噪声注入
    method_name = "07_mixed_noise"
    print(f"Testing {method_name}...")
    # Gaussian
    for std_val in [0.005, 0.01, 0.02]:
         params = {'gaussian': {'mean': 0, 'std': std_val}}
         img_noise = mixed_noise_injection(img_float.copy(), noise_types=['gaussian'], noise_params=params)
         filename = f"noise_gaussian_std{std_val:.3f}"
         save_image(img_noise, method_name, filename, OUTPUT_DIR_BASE)
    # Speckle
    for var_val in [0.005, 0.01, 0.02]:
         params = {'speckle': {'mean': 0, 'var': var_val}}
         img_noise = mixed_noise_injection(img_float.copy(), noise_types=['speckle'], noise_params=params)
         filename = f"noise_speckle_var{var_val:.3f}"
         save_image(img_noise, method_name, filename, OUTPUT_DIR_BASE)
    # Poisson
    for scale_val in [10, 20, 40]:
         params = {'poisson': {'scale': scale_val}}
         img_noise = mixed_noise_injection(img_float.copy(), noise_types=['poisson'], noise_params=params)
         filename = f"noise_poisson_scale{scale_val}"
         save_image(img_noise, method_name, filename, OUTPUT_DIR_BASE)

    # 8. 旋转
    method_name = "08_rotation"
    print(f"Testing {method_name}...")
    for angle_val in [-10, -5, 5, 10]: # 项目建议 max_angle=10
        height, width = original_img.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_val, 1.0)
        img_rotated_fixed = cv2.warpAffine(original_img.copy(), rotation_matrix, (width, height),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        filename = f"rotate_angle{angle_val}"
        save_image(img_rotated_fixed, method_name, filename, OUTPUT_DIR_BASE)


    # 9. 随机裁剪 (Resized)
    method_name = "09_random_crop_resized"
    print(f"Testing {method_name}...")
    for ratio_val in [0.7, 0.8, 0.9]: # 项目建议 [0.8, 0.95]
        # random_crop 内部随机选择位置，我们只固定比例看效果
        img_cropped, _ = random_crop(original_img.copy(), crop_ratio=(ratio_val, ratio_val)) # 固定比例
        filename = f"crop_ratio{ratio_val:.1f}"
        save_image(img_cropped, method_name, filename, OUTPUT_DIR_BASE)

    # 10. 水平翻转
    method_name = "10_horizontal_flip"
    print(f"Testing {method_name}...")
    img_flipped, _ = horizontal_flip(original_img.copy())
    filename = "flipped"
    save_image(img_flipped, method_name, filename, OUTPUT_DIR_BASE)

    # 11. 亮度对比度调整
    method_name = "11_brightness_contrast"
    print(f"Testing {method_name}...")
    # Vary brightness, keep contrast neutral
    for bright_val in [-0.1, 0.0, 0.1]: # 项目范围 [-0.1, 0.1]
        img_bc = adjust_brightness_contrast(img_float.copy(), brightness_range=(bright_val, bright_val), contrast_range=(1.0, 1.0))
        filename = f"brightness{bright_val:+.1f}_contrast1.0"
        save_image(img_bc, method_name, filename, OUTPUT_DIR_BASE)
    # Vary contrast, keep brightness neutral
    for contrast_val in [0.8, 1.0, 1.2]: # 项目范围 [0.8, 1.2]
        img_bc = adjust_brightness_contrast(img_float.copy(), brightness_range=(0.0, 0.0), contrast_range=(contrast_val, contrast_val))
        filename = f"brightness+0.0_contrast{contrast_val:.1f}"
        save_image(img_bc, method_name, filename, OUTPUT_DIR_BASE)

# 12. 标准化尺寸 (展示中间步骤)
    method_name = "12_standardize_size_steps"
    print(f"Testing {method_name}...")
    # 选择一个与原图尺寸差异较大的目标尺寸以突出效果
    h_orig, w_orig = original_img.shape[:2]
    target_size_test = (int(h_orig * 0.7), int(w_orig * 1.2)) # 示例: 缩小高度, 放大宽度
    if h_orig < 100 or w_orig < 100: # 如果原图太小
        target_size_test = (128, 256)
    target_h, target_w = target_size_test

    # --- 模拟 standardize_size 的步骤 ---
    # a) 计算缩放比例和新尺寸
    scale = max(target_h / h_orig, target_w / w_orig)
    new_h, new_w = int(h_orig * scale), int(w_orig * scale)

    # b) 缩放图像
    resized_img_step = cv2.resize(original_img.copy(), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    filename_resized = f"step1_resized_to_{new_h}x{new_w}"
    save_image(resized_img_step, method_name, filename_resized, OUTPUT_DIR_BASE)

    # c) 计算裁剪起始点
    start_y = max(0, (new_h - target_h) // 2)
    start_x = max(0, (new_w - target_w) // 2)

    # d) 裁剪到目标尺寸 (这是裁剪后的核心内容，未填充)
    # 注意: 这里的裁剪尺寸可能小于target_size, 如果缩放后图像小于目标
    crop_end_y = min(new_h, start_y + target_h)
    crop_end_x = min(new_w, start_x + target_w)
    cropped_img_step = resized_img_step[start_y:crop_end_y, start_x:crop_end_x]
    filename_cropped = f"step2_cropped_part_{cropped_img_step.shape[0]}x{cropped_img_step.shape[1]}"
    save_image(cropped_img_step, method_name, filename_cropped, OUTPUT_DIR_BASE)

    # --- 调用原函数获取最终结果 (包含填充) ---
    final_img, _ = standardize_size(original_img.copy(), target_size=target_size_test)
    filename_final = f"step3_final_padded_{target_h}x{target_w}"
    save_image(final_img, method_name, filename_final, OUTPUT_DIR_BASE)


    print("\n--- Testing Combined Augmentations (Single Random Example) ---")
    # 13. 空间域增强组合
    method_name = "13_spatial_combined"
    print(f"Testing {method_name}...")
    img_spatial, _ = spatial_domain_augmentation(original_img.copy())
    filename = "spatial_random_example"
    save_image(img_spatial, method_name, filename, OUTPUT_DIR_BASE)

    # 14. 灰度域增强组合
    method_name = "14_intensity_combined"
    print(f"Testing {method_name}...")
    img_intensity = intensity_domain_augmentation(img_float.copy())
    filename = "intensity_random_example"
    save_image(img_intensity, method_name, filename, OUTPUT_DIR_BASE)


    print(f"\n--- All tests finished. Images saved in subdirectories under '{OUTPUT_DIR_BASE}'. ---")
