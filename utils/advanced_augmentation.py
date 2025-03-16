import random

import cv2
import numpy as np
import pywt


def tps_transform(img, num_control_points=16, std_dev=10, regularization=0.3):
    """
    Thin Plate Spline形变
    
    参数:
        img: 输入图像
        num_control_points: 控制点数量
        std_dev: 控制点移动的标准差
        regularization: 正则化参数λ
    
    返回:
        形变后的图像
    """
    rows, cols = img.shape
    
    # 创建控制点网格
    src_points = np.zeros((num_control_points, 2), dtype=np.float32)
    dst_points = np.zeros((num_control_points, 2), dtype=np.float32)
    
    # 均匀分布控制点
    grid_size = int(np.sqrt(num_control_points))
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            src_points[idx] = [j * cols / (grid_size-1), i * rows / (grid_size-1)]
            
            # 添加随机位移，但保持在图像边界内
            dx = np.random.normal(0, std_dev)
            dy = np.random.normal(0, std_dev)
            x = np.minimum(np.maximum(src_points[idx][0] + dx, 0), cols - 1)
            y = np.minimum(np.maximum(src_points[idx][1] + dy, 0), rows - 1)
            dst_points[idx] = [x, y]
    
    # 计算TPS形变
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.setRegularizationParameter(regularization)
    
    # 将点reshape为需要的格式
    src_points = src_points.reshape(1, -1, 2)
    dst_points = dst_points.reshape(1, -1, 2)
    
    # 匹配点
    matches = [cv2.DMatch(i, i, 0) for i in range(num_control_points)]
    
    # 设置匹配点
    tps.estimateTransformation(dst_points, src_points, matches)
    
    # 应用变换
    warped_img = tps.warpImage(img)
    
    return warped_img

def dynamic_gamma_correction(img, gamma_range=(0.6, 1.8)):
    """
    动态Gamma校正
    
    参数:
        img: 输入图像
        gamma_range: gamma值范围
    
    返回:
        校正后的图像
    """
    # 随机选择gamma值
    gamma = np.random.uniform(gamma_range[0], gamma_range[1])
    
    # 确保图像在[0,1]范围内
    if img.max() > 1.0:
        img = img / 255.0
    
    # 应用gamma校正
    corrected = np.power(img, gamma)
    
    return corrected

def mixed_noise_injection(img, noise_types=None, noise_params=None):
    """
    混合噪声注入
    
    参数:
        img: 输入图像
        noise_types: 噪声类型列表
        noise_params: 噪声参数字典
    
    返回:
        添加噪声后的图像
    """
    # 确保图像在[0,1]范围内
    if noise_types is None:
        noise_types = ['gaussian', 'speckle', 'poisson']
    if img.max() > 1.0:
        img = img / 255.0
    
    # 默认噪声参数
    if noise_params is None:
        noise_params = {
            'gaussian': {'mean': 0, 'std': 0.01},
            'speckle': {'mean': 0, 'var': 0.01},
            'poisson': {'scale': 10}
        }
    
    # 复制图像
    noisy_img = img.copy()
    
    # 随机选择噪声类型
    noise_type = random.choice(noise_types)
    
    # 应用选择的噪声
    if noise_type == 'gaussian':
        # 高斯噪声
        mean = noise_params['gaussian']['mean']
        std = noise_params['gaussian']['std']
        noise = np.random.normal(mean, std, img.shape)
        noisy_img = img + noise
    
    elif noise_type == 'speckle':
        # 散斑噪声
        mean = noise_params['speckle']['mean']
        var = noise_params['speckle']['var']
        noise = np.random.normal(mean, var**0.5, img.shape)
        noisy_img = img + img * noise
    
    elif noise_type == 'poisson':
        # 泊松噪声
        scale = noise_params['poisson']['scale']
        noisy_img = np.random.poisson(img * scale) / scale
    
    # 裁剪值到[0,1]范围
    noisy_img = np.clip(noisy_img, 0, 1)
    
    return noisy_img

def nsct_enhancement(img, decomp_levels=3, directions=[4, 8, 8]):
    """
    NSCT多尺度分解系统 - 对高频子带选择性增强
    
    参数:
        img: 输入图像
        decomp_levels: 分解级数
        directions: 每级方向数
    
    返回:
        增强后的图像
    """
    # 注意：实际实现需要NSCT库，这里提供伪代码
    # TODO: NSCT库可能需要单独安装或自行实现
    
    # 使用小波变换作为替代演示
    # TODO: 在实际项目中，应使用真正的NSCT变换
    coeffs = pywt.wavedec2(img, 'db1', level=decomp_levels)
    
    # 增强高频子带
    for i in range(1, len(coeffs)):
        # 获取当前级别的水平、垂直和对角线细节系数
        if isinstance(coeffs[i], tuple):
            h, v, d = coeffs[i]
            
            # 增强因子（可调整）
            enhance_factor = 1.5
            
            # 增强高频系数
            coeffs[i] = (h * enhance_factor, v * enhance_factor, d * enhance_factor)
    
    # 重建图像
    enhanced_img = pywt.waverec2(coeffs, 'db1')
    
    # 归一化
    enhanced_img = (enhanced_img - np.min(enhanced_img)) / (np.max(enhanced_img) - np.min(enhanced_img))
    
    return enhanced_img

def rotate_image(img, label=None, max_angle=10):
    """
    图像旋转增强
    
    参数:
        img: 输入图像
        label: 标签图像（如果有）
        max_angle: 最大旋转角度（度）
    
    返回:
        旋转后的图像和标签
    """
    # 随机选择旋转角度
    angle = np.random.uniform(-max_angle, max_angle)
    
    # 获取图像中心点
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 应用旋转变换
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height), 
                                 flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
    
    # 如果有标签，同样旋转标签
    rotated_label = None
    if label is not None:
        rotated_label = cv2.warpAffine(label, rotation_matrix, (width, height),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
    
    return rotated_img, rotated_label


def random_crop(img, label=None, crop_ratio=(0.8, 0.95)):
    """
    随机裁剪增强
    
    参数:
        img: 输入图像
        label: 标签图像（如果有）
        crop_ratio: 裁剪比例范围（相对于原图大小）
    
    返回:
        裁剪后的图像和标签
    """
    height, width = img.shape[:2]
    
    # 随机选择裁剪比例
    ratio = np.random.uniform(crop_ratio[0], crop_ratio[1])
    
    # 计算裁剪尺寸
    crop_height = int(height * ratio)
    crop_width = int(width * ratio)
    
    # 随机选择裁剪起始点
    start_x = np.random.randint(0, width - crop_width + 1)
    start_y = np.random.randint(0, height - crop_height + 1)
    
    # 裁剪图像
    cropped_img = img[start_y:start_y+crop_height, start_x:start_x+crop_width]
    
    # 调整回原始尺寸
    resized_img = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # 如果有标签，同样裁剪标签
    resized_label = None
    if label is not None:
        cropped_label = label[start_y:start_y+crop_height, start_x:start_x+crop_width]
        resized_label = cv2.resize(cropped_label, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return resized_img, resized_label


def horizontal_flip(img, label=None):
    """
    水平翻转增强
    
    参数:
        img: 输入图像
        label: 标签图像（如果有）
    
    返回:
        翻转后的图像和标签
    """
    # 水平翻转图像
    flipped_img = cv2.flip(img, 1)  # 1表示水平翻转
    
    # 如果有标签，同样翻转标签
    flipped_label = None
    if label is not None:
        flipped_label = cv2.flip(label, 1)
    
    return flipped_img, flipped_label


def adjust_brightness_contrast(img, brightness_range=(-0.1, 0.1), contrast_range=(0.8, 1.2)):
    """
    亮度和对比度调整
    
    参数:
        img: 输入图像
        brightness_range: 亮度调整范围
        contrast_range: 对比度调整范围
    
    返回:
        调整后的图像
    """
    # 确保图像在[0,1]范围内
    if img.max() > 1.0:
        img = img / 255.0
    
    # 随机选择亮度和对比度调整值
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    
    # 应用亮度和对比度调整
    adjusted = img * contrast + brightness
    
    # 裁剪值到[0,1]范围
    adjusted = np.clip(adjusted, 0, 1)
    
    return adjusted

def spatial_domain_augmentation(img, label=None):
    """
    空间域数据增强
    
    参数:
        img: 输入图像
        label: 标签图像（如果有）
    
    返回:
        增强后的图像和标签
    """
    # 复制输入图像和标签
    img_aug = img.copy()
    label_aug = label.copy() if label is not None else None
    
    # 1. 随机旋转 (50%概率)
    if np.random.random() < 0.5:
        img_aug, label_aug = rotate_image(img_aug, label_aug, max_angle=10)
    
    # 2. 随机裁剪 (50%概率)
    if np.random.random() < 0.5:
        img_aug, label_aug = random_crop(img_aug, label_aug)
    
    # 3. 随机水平翻转 (50%概率)
    if np.random.random() < 0.5:
        img_aug, label_aug = horizontal_flip(img_aug, label_aug)
    
    # 4. 应用TPS形变 (50%概率)
    if np.random.random() < 0.5:
        if label_aug is not None:
            # 将图像和标签堆叠在一起进行同样的变换
            stacked = np.dstack((img_aug, label_aug))
            stacked_warped = tps_transform(stacked, regularization=0.3)
            img_aug = stacked_warped[:, :, 0]
            label_aug = stacked_warped[:, :, 1]
        else:
            img_aug = tps_transform(img_aug, regularization=0.3)
    
    # 5. 亮度和对比度调整 (50%概率)
    if np.random.random() < 0.5:
        img_aug = adjust_brightness_contrast(img_aug)
    
    return img_aug, label_aug

def intensity_domain_augmentation(img):
    """
    灰度域数据增强
    
    参数:
        img: 输入图像
    
    返回:
        增强后的图像
    """
    # 应用动态Gamma校正
    if np.random.random() < 0.5:
        img = dynamic_gamma_correction(img)
    
    # 应用混合噪声注入
    if np.random.random() < 0.5:
        img = mixed_noise_injection(img)
    
    return img

def frequency_domain_augmentation(img):
    """
    频域数据增强
    
    参数:
        img: 输入图像
    
    返回:
        增强后的图像
    """
    # 应用NSCT增强
    if np.random.random() < 0.5:
        img = nsct_enhancement(img)
    
    return img

def advanced_augmentation(img, label=None):
    """
    三维空间-频域联合增强框架
    
    参数:
        img: 输入图像
        label: 标签图像（如果有）
    
    返回:
        增强后的图像和标签
    """
    # 1. 空间域增强
    img_aug, label_aug = spatial_domain_augmentation(img, label)
    
    # 2. 灰度域增强（仅对输入图像）
    img_aug = intensity_domain_augmentation(img_aug)
    
    # 3. 频域增强（仅对输入图像）
    img_aug = frequency_domain_augmentation(img_aug)
    
    return img_aug, label_aug 