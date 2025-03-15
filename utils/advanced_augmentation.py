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
            x = np.min(np.max(src_points[idx][0] + dx, 0), cols - 1)
            y = np.min(np.max(src_points[idx][1] + dy, 0), rows - 1)
            dst_points[idx] = [x, y]
    
    # 计算TPS形变
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.regularizationParameter = regularization
    
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

def perspective_transform(img, angle_range=3):
    """
    受控透视变换
    
    参数:
        img: 输入图像
        angle_range: 投影矩阵扰动范围（度）
    
    返回:
        透视变换后的图像
    """
    rows, cols = img.shape
    
    # 随机角度（±angle_range度）
    angle_x = np.random.uniform(-angle_range, angle_range) * np.pi / 180
    angle_y = np.random.uniform(-angle_range, angle_range) * np.pi / 180
    angle_z = np.random.uniform(-angle_range, angle_range) * np.pi / 180
    
    # 创建透视变换矩阵
    # 旋转矩阵
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # 创建透视变换矩阵
    d = 500  # 相机距离
    f = 500  # 焦距
    
    K = np.array([
        [f, 0, cols/2],
        [0, f, rows/2],
        [0, 0, 1]
    ])
    
    P = np.dot(K, R)
    
    # 应用透视变换
    warped_img = cv2.warpPerspective(img, P[:2], (cols, rows))
    
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
    # 在实际项目中，应使用真正的NSCT变换
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

def spatial_domain_augmentation(img, label=None):
    """
    空间域数据增强
    
    参数:
        img: 输入图像
        label: 标签图像（如果有）
    
    返回:
        增强后的图像和标签
    """
    # 应用TPS形变
    if np.random.random() < 0.5:
        if label is not None:
            # 将图像和标签堆叠在一起进行同样的变换
            stacked = np.dstack((img, label))
            stacked_warped = tps_transform(stacked, regularization=0.3)
            img_warped = stacked_warped[:, :, 0]
            label_warped = stacked_warped[:, :, 1]
        else:
            img_warped = tps_transform(img, regularization=0.3)
            label_warped = None
    else:
        img_warped = img
        label_warped = label
    
    # 应用受控透视变换
    if np.random.random() < 0.5:
        if label_warped is not None:
            # 确保图像和标签使用相同的变换
            stacked = np.dstack((img_warped, label_warped))
            stacked_warped = perspective_transform(stacked)
            img_warped = stacked_warped[:, :, 0]
            label_warped = stacked_warped[:, :, 1]
        else:
            img_warped = perspective_transform(img_warped)
    
    return img_warped, label_warped

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