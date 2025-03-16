import bm3d
import numpy as np
import phasepack
from bm3d import BM3DProfile
from scipy import ndimage
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift


def anisotropic_diffusion(img, niter=5, kappa=0.15, gamma=0.2, option=1):
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

    # 应用BM3D去噪
    # 参数设置
    custom_bm3d_profile = BM3DProfile()
    custom_bm3d_profile.bs_ht = 4  # 硬阈值处理阶段的块大小
    custom_bm3d_profile.bs_wiener = 4  # 维纳滤波阶段的块大小
    custom_bm3d_profile.search_window_ht = 30  # 硬阈值处理阶段的搜索窗口大小
    custom_bm3d_profile.search_window_wiener = 30  # 维纳滤波阶段的搜索窗口大小

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
    crow, ccol = rows // 2, cols // 2

    x = np.linspace(-1, 1, cols)
    y = np.linspace(-1, 1, rows)
    X, Y = np.meshgrid(x, y)
    dist = np.sqrt(X ** 2 + Y ** 2)

    # 巴特沃斯高通滤波器
    H = 1.0 / (1.0 + (cutoff / dist) ** (2 * order))

    # 应用滤波器
    img_fft_filtered = img_fft * H

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


def advanced_denoising(img):
    """
    基于AD-BM3D的混合去噪框架
    
    参数:
        img: 输入图像
    
    返回:
        去噪后的图像
    """
    # 归一化到[0,1]
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))

    # 步骤1：各向异性扩散算法抑制散斑噪声
    img_ad = anisotropic_diffusion(img_norm)

    # 步骤2：BM3D多尺度联合去噪
    img_bm3d = bm3d_denoising(img_ad)

    return img_bm3d


def advanced_enhancement(img):
    """
    高级图像增强
    
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
