import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置Matplotlib支持中文的字体
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种常见的中文黑体字体
mpl.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

def normalize_image(image, method='minmax'):
    """
    图像归一化
    
    参数:
        image: 输入图像
        method: 'minmax'或'zscore'
    
    返回:
        归一化后的图像
    """
    if method == 'minmax':
        # MinMax归一化到[0,1]
        return (image - np.min(image)) / (np.max(image) - np.min(image))
    elif method == 'zscore':
        # Z-score标准化
        return (image - np.mean(image)) / np.std(image)
    else:
        raise ValueError("不支持的归一化方法")

def denoise_image(image, method='gaussian', kernel_size=5):
    """
    图像去噪
    
    参数:
        image: 输入图像
        method: 'gaussian'或'median'
        kernel_size: 滤波器核大小
    
    返回:
        去噪后的图像
    """
    if method == 'gaussian':
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError("不支持的去噪方法")

def enhance_contrast(image, method='clahe'):
    """
    对比度增强
    
    参数:
        image: 输入图像
        method: 'clahe'或'histeq'
    
    返回:
        对比度增强后的图像
    """
    if method == 'clahe':
        # 确保图像是8位整型
        if image.dtype != np.uint8:
            image = (normalize_image(image) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
        return clahe.apply(image)
    elif method == 'histeq':
        # 直方图均衡化
        return exposure.equalize_hist(image)
    else:
        raise ValueError("不支持的对比度增强方法")

def show_comparison(original, processed, title="图像处理前后对比"):
    """显示处理前后的图像对比"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('原始图像')
    axes[1].imshow(processed, cmap='gray')
    axes[1].set_title('处理后图像')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 加载一张图像
    img_path = "../data/2015_BOE_Chiu/raw/test/test_1.jpg"  # 替换为你的图像路径
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # 预处理示例
    img_normalized = normalize_image(img)
    img_normalized_uint8 = (img_normalized * 255).astype(np.uint8)
    # img_gaussian_denoised = denoise_image(img_normalized_uint8, method='gaussian')
    img_median_denoised = denoise_image(img_normalized_uint8, method='median')
    img_enhanced = enhance_contrast(img_median_denoised, method='clahe')
    
    # 显示结果
    show_comparison(img, img_median_denoised, "原始图像 vs 处理后图像")
