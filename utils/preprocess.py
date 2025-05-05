import bm3d
import numpy as np
import phasepack
import cv2
from bm3d import BM3DProfile
from scipy import ndimage
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

def standardize_size(img, label=None, target_size=(720, 992)):
    """
    将图像标准化为固定尺寸（保持长宽比的缩放+中心裁剪）

    参数:
        img: 输入图像
        label: 标签图像（如果有）
        target_size: 目标尺寸，格式为(高度, 宽度)

    返回:
        标准化后的图像和标签
    """
    # 获取原始尺寸
    h, w = img.shape[:2]
    target_h, target_w = target_size

    # 计算缩放比例（保持长宽比）
    scale = max(target_h / h, target_w / w)

    # 计算缩放后的尺寸
    new_h, new_w = int(h * scale), int(w * scale)

    # 缩放图像
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 计算裁剪的起始点（中心裁剪）
    start_y = max(0, (new_h - target_h) // 2)
    start_x = max(0, (new_w - target_w) // 2)

    # 裁剪到目标尺寸
    cropped_img = resized_img[start_y:start_y + target_h, start_x:start_x + target_w]

    # 处理尺寸不足的情况（在边缘填充）
    final_img = np.zeros(target_size, dtype=img.dtype)

    # 计算实际裁剪得到的尺寸（可能小于目标尺寸）
    crop_h, crop_w = cropped_img.shape[:2]

    # 计算在目标图像中的放置位置
    place_y = (target_h - crop_h) // 2
    place_x = (target_w - crop_w) // 2

    # 放置裁剪的图像
    final_img[place_y:place_y + crop_h, place_x:place_x + crop_w] = cropped_img

    # 如果有标签，同样处理标签
    final_label = None
    if label is not None:
        # 缩放标签（使用最近邻插值以保持标签值）
        resized_label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 裁剪标签到目标尺寸
        cropped_label = resized_label[start_y:start_y + target_h, start_x:start_x + target_w]

        # 处理尺寸不足的情况
        final_label = np.zeros(target_size, dtype=label.dtype)
        final_label[place_y:place_y + crop_h, place_x:place_x + crop_w] = cropped_label

        # 转换为整型标签
        if final_label is not None:
            final_label = final_label.astype(np.uint8)

    return final_img, final_label

def anisotropic_diffusion(img, niter=5, kappa=0.15, gamma=0.15, option=1):
    """
    各向异性扩散 - 实现散斑噪声抑制
    
    参数:
        img: 输入图像
        niter: 迭代次数
        kappa: 传导系数
        gamma: 步长
        option: 1或2，选择扩散函数
    
    返回:
        处理后的图像
    """
    # 复制图像为浮点型
    img = img.astype('float32')

    # 初始化输出图像为输入图像
    out = img.copy()

    # 构建Sobel梯度算子
    dx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    dy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    for i in range(niter):
        # 计算梯度
        grad_x = ndimage.convolve(out, dx)
        grad_y = ndimage.convolve(out, dy)

        # 计算梯度幅度
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # 根据选择的扩散函数计算扩散系数
        if option == 1:
            # 使用Perona-Malik扩散函数1
            diff_coef = np.exp(-(grad_mag / kappa) ** 2)
        else:
            # 使用Perona-Malik扩散函数2
            diff_coef = 1 / (1 + (grad_mag / kappa) ** 2)

        # 更新图像
        out = out + gamma * (
                diff_coef * grad_x +
                diff_coef * grad_y
        )

    return out


def bm3d_denoising(img, sigma_psd=10 / 255):
    """
    BM3D去噪 - 三维块匹配滤波
    
    参数:
        img: 输入图像 [0,1]范围
        sigma_psd: 噪声标准差
    
    返回:
        去噪后的图像
    """
    # 确保图像在[0,1]范围内
    if img.max() > 1.0:
        img = img / 255.0

    # BM3D参数设置
    custom_bm3d_profile = BM3DProfile()
    # 硬阈值处理阶段的块大小
    custom_bm3d_profile.bs_ht = 4
    # 维纳滤波阶段的块大小
    custom_bm3d_profile.bs_wiener = 4
    # 硬阈值处理阶段的搜索窗口大小
    custom_bm3d_profile.search_window_ht = 30
    # 维纳滤波阶段的搜索窗口大小
    custom_bm3d_profile.search_window_wiener = 30

    # 应用BM3D去噪
    denoised_img = bm3d.bm3d(img, sigma_psd=sigma_psd,
                             stage_arg=bm3d.BM3DStages.ALL_STAGES,
                             profile=custom_bm3d_profile,
                             )
    return denoised_img


def phase_congruency_enhancement(img, threshold=0.45, nscale=4):
    """
    相位一致性算法 - 增强层间边界
    
    参数:
        img: 输入图像
        threshold: PC阈值
        nscale: 尺度数
    
    返回:
        增强边缘的图像
    """
    # 确保图像在[0,1]范围内
    if img.max() > 1.0:
        img = img / 255.0

    # 计算相位一致性
    pc = phasepack.phasecong(img, nscale=nscale)

    # 提取相位一致性结果
    pc_edge = pc[0]  # 第一个输出是相位一致性度量

    # 阈值处理
    pc_edge = (pc_edge > threshold).astype(float)

    # 增强原始图像的边缘
    enhanced = img + 0.5 * pc_edge

    # 确保值在[0,1]范围内
    enhanced = np.clip(enhanced, 0, 1)

    return enhanced


def butterworth_highpass_filter(img, cutoff=0.08, order=1):
    """
    巴特沃斯高通滤波器 - 提升高频细节响应
    
    参数:
        img: 输入图像
        cutoff: 截止频率（相对于Nyquist频率）
        order: 滤波器阶数
    
    返回:
        高频增强的图像
    """
    # 将图像转换到频域
    img_fft = fftshift(fft2(img))

    # 创建滤波器
    rows, cols = img.shape
    crow, _ = rows // 2, cols // 2

    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    x, y = np.meshgrid(x, y)
    dist = np.sqrt(x ** 2 + y ** 2)

    # 巴特沃斯高通滤波器
    h = 1.0 / (1.0 + (cutoff / dist) ** (2 * order))

    # 应用滤波器
    img_fft_filtered = img_fft * h

    # 转回空间域
    img_filtered = np.real(ifft2(ifftshift(img_fft_filtered)))

    # 归一化
    img_filtered = (img_filtered - np.min(img_filtered)) / (np.max(img_filtered) - np.min(img_filtered))

    # 与原图组合
    alpha = 0.5  # 原图权重
    beta = 0.5  # 高频细节权重
    combined = alpha * img + beta * img_filtered

    # 归一化
    combined = (combined - np.min(combined)) / (np.max(combined) - np.min(combined))

    return combined


def ad_bm3d_image_denoising(img):
    """
    对图像进行基于AD-BM3D的混合去噪
    
    参数:
        img: 输入图像
    
    返回:
        去噪后的图像
    """
    # 归一化到[0,1]
    if img.max() > 1.0:
        img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        img_norm = img

    # 步骤1：各向异性扩散算法抑制散斑噪声
    img_ad = anisotropic_diffusion(img_norm)

    # 步骤2：BM3D多尺度联合去噪
    img_bm3d = bm3d_denoising(img_ad)

    return img_bm3d


def pc_hpf_image_enhancement(img):
    """
    对图像进行基于PC-HPF的混合增强
    
    参数:
        img: 输入图像
    
    返回:
        增强后的图像
    """
    # 步骤1：相位一致性算法增强层间边界
    img_pc = phase_congruency_enhancement(img, threshold=0.45, nscale=4)

    # 步骤2：巴特沃斯高通滤波器提升高频细节响应
    img_enhanced = butterworth_highpass_filter(img_pc)

    return img_enhanced

