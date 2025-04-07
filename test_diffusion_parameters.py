import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess import anisotropic_diffusion, bm3d_denoising

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def test_anisotropic_diffusion_parameters(image_path, save_dir="./chart/diffusion_tests"):
    """
    测试不同参数下的各向异性扩散效果及后续BM3D去噪效果
    
    参数:
        image_path: 输入图像路径
        save_dir: 保存结果的目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 归一化
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    # 定义要测试的参数组合
    test_params = [
        # niter, kappa, gamma, option
        (5, 0.15, 0.2, 1),    # 默认参数，使用扩散函数1
        (5, 0.15, 0.2, 2),    # 默认参数，使用扩散函数2
        (3, 0.15, 0.2, 1),    # 减少迭代次数
        (10, 0.15, 0.2, 1),   # 增加迭代次数
        (5, 0.10, 0.2, 1),    # 降低传导系数
        (5, 0.25, 0.2, 1),    # 提高传导系数
        (5, 0.15, 0.1, 1),    # 降低步长
        (5, 0.15, 0.3, 1)     # 提高步长
    ]
    
    # 为每组参数应用各向异性扩散和BM3D去噪
    results = []
    param_labels = []
    
    for params in test_params:
        niter, kappa, gamma, option = params
        param_label = f"niter={niter}, κ={kappa}, γ={gamma}, opt={option}"
        print(f"正在处理: {param_label}")
        
        # 应用各向异性扩散
        diff_result = anisotropic_diffusion(img_norm, niter=niter, kappa=kappa, gamma=gamma, option=option)
        
        # 应用BM3D去噪
        bm3d_result = bm3d_denoising(diff_result)
        
        # 保存结果
        results.append((diff_result, bm3d_result))
        param_labels.append(param_label)
    
    # 生成对比图 - 各向异性扩散结果
    fig1, axes1 = plt.subplots(3, 3, figsize=(15, 15))  # 修改为3×3布局
    axes1 = axes1.flatten()
    
    # 先添加原始图像作为参考
    axes1[0].imshow(img_norm, cmap='gray')
    axes1[0].set_title("原始图像")
    axes1[0].axis('off')
    
    # 显示各种参数的各向异性扩散结果
    for i, ((diff_img, _), param_label) in enumerate(zip(results, param_labels)):
        axes1[i+1].imshow(diff_img, cmap='gray')
        axes1[i+1].set_title(param_label)
        axes1[i+1].axis('off')
    
    # 隐藏未使用的子图
    for i in range(len(results)+1, len(axes1)):
        axes1[i].axis('off')
    
    plt.suptitle("不同参数下的各向异性扩散效果对比", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig1.savefig(f"{save_dir}/anisotropic_diffusion_comparison.png", dpi=300)
    
    # 生成对比图 - BM3D去噪后结果
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 15))  # 修改为3×3布局
    axes2 = axes2.flatten()
    
    # 先添加原始图像作为参考
    axes2[0].imshow(img_norm, cmap='gray')
    axes2[0].set_title("原始图像")
    axes2[0].axis('off')
    
    # 显示各种参数的BM3D去噪后结果
    for i, ((_, bm3d_img), param_label) in enumerate(zip(results, param_labels)):
        axes2[i+1].imshow(bm3d_img, cmap='gray')
        axes2[i+1].set_title(f"{param_label} + BM3D")
        axes2[i+1].axis('off')
    
    # 隐藏未使用的子图
    for i in range(len(results)+1, len(axes2)):
        axes2[i].axis('off')
    
    plt.suptitle("不同参数下的各向异性扩散+BM3D去噪效果对比", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig2.savefig(f"{save_dir}/anisotropic_diffusion_bm3d_comparison.png", dpi=300)
    
    # 生成详细的参数对比
    for i, ((diff_img, bm3d_img), param_label) in enumerate(zip(results, param_labels)):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_norm, cmap='gray')
        axes[0].set_title("原始图像")
        axes[0].axis('off')
        
        axes[1].imshow(diff_img, cmap='gray')
        axes[1].set_title(f"各向异性扩散\n{param_label}")
        axes[1].axis('off')
        
        axes[2].imshow(bm3d_img, cmap='gray')
        axes[2].set_title("BM3D去噪后")
        axes[2].axis('off')
        
        plt.suptitle(f"参数组合 {i+1}: {param_label}", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        fig.savefig(f"{save_dir}/params_set_{i+1}_detail.png", dpi=300)
        plt.close(fig)
    
    print(f"所有测试完成，结果已保存到 {save_dir} 目录")
    return fig1, fig2

def test_specific_parameter_range(image_path, parameter='niter', value_range=None, save_dir="./chart/parameter_tests"):
    """
    测试单个参数不同值下的各向异性扩散效果
    
    参数:
        image_path: 输入图像路径
        parameter: 要测试的参数('niter', 'kappa', 'gamma' 或 'option')
        value_range: 参数值范围
        save_dir: 保存结果的目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # 归一化
    img_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    # 设置默认参数
    default_params = {
        'niter': 5,
        'kappa': 0.15,
        'gamma': 0.2,
        'option': 1
    }
    
    # 设置参数范围的默认值
    if value_range is None:
        if parameter == 'niter':
            value_range = [1, 3, 5, 7, 10, 15]
        elif parameter == 'kappa':
            value_range = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        elif parameter == 'gamma':
            value_range = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        elif parameter == 'option':
            value_range = [1, 2]
    
    # 参数名称映射
    param_name_map = {
        'niter': '迭代次数',
        'kappa': '传导系数κ',
        'gamma': '步长γ',
        'option': '扩散函数'
    }
    
    # 为每个参数值应用各向异性扩散和BM3D去噪
    results = []
    param_labels = []
    
    for value in value_range:
        # 更新测试参数
        test_params = default_params.copy()
        test_params[parameter] = value
        
        param_label = f"{param_name_map[parameter]}={value}"
        print(f"正在处理: {param_label}")
        
        # 应用各向异性扩散
        diff_result = anisotropic_diffusion(img_norm, 
                                           niter=test_params['niter'],
                                           kappa=test_params['kappa'],
                                           gamma=test_params['gamma'],
                                           option=test_params['option'])
        
        # 应用BM3D去噪
        bm3d_result = bm3d_denoising(diff_result)
        
        # 保存结果
        results.append((diff_result, bm3d_result))
        param_labels.append(param_label)
    
    # 计算每行放置的图像数
    n_values = len(value_range)
    n_cols = min(3, n_values)
    n_rows = (n_values + n_cols - 1) // n_cols
    
    # 生成各向异性扩散结果对比图
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    if n_rows * n_cols > 1:
        axes1 = axes1.flatten()
    else:
        axes1 = [axes1]
    
    for i, ((diff_img, _), param_label) in enumerate(zip(results, param_labels)):
        axes1[i].imshow(diff_img, cmap='gray')
        axes1[i].set_title(param_label)
        axes1[i].axis('off')
    
    # 如果子图数量多于参数值数量，隐藏多余的子图
    for i in range(n_values, len(axes1)):
        axes1[i].axis('off')
    
    plt.suptitle(f"不同{param_name_map[parameter]}下的各向异性扩散效果对比", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig1.savefig(f"{save_dir}/{parameter}_diffusion_comparison.png", dpi=300)
    
    # 生成BM3D去噪后结果对比图
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
    if n_rows * n_cols > 1:
        axes2 = axes2.flatten()
    else:
        axes2 = [axes2]
    
    for i, ((_, bm3d_img), param_label) in enumerate(zip(results, param_labels)):
        axes2[i].imshow(bm3d_img, cmap='gray')
        axes2[i].set_title(f"{param_label} + BM3D")
        axes2[i].axis('off')
    
    # 如果子图数量多于参数值数量，隐藏多余的子图
    for i in range(n_values, len(axes2)):
        axes2[i].axis('off')
    
    plt.suptitle(f"不同{param_name_map[parameter]}下的各向异性扩散+BM3D去噪效果对比", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig2.savefig(f"{save_dir}/{parameter}_diffusion_bm3d_comparison.png", dpi=300)
    
    # 对比原图与处理后图像
    fig3, axes3 = plt.subplots(n_values, 3, figsize=(15, n_values*5))
    if n_values > 1:
        for i, ((diff_img, bm3d_img), param_label) in enumerate(zip(results, param_labels)):
            axes3[i, 0].imshow(img_norm, cmap='gray')
            axes3[i, 0].set_title("原始图像")
            axes3[i, 0].axis('off')
            
            axes3[i, 1].imshow(diff_img, cmap='gray')
            axes3[i, 1].set_title(f"各向异性扩散\n{param_label}")
            axes3[i, 1].axis('off')
            
            axes3[i, 2].imshow(bm3d_img, cmap='gray')
            axes3[i, 2].set_title("BM3D去噪后")
            axes3[i, 2].axis('off')
    else:
        axes3[0].imshow(img_norm, cmap='gray')
        axes3[0].set_title("原始图像")
        axes3[0].axis('off')
        
        axes3[1].imshow(results[0][0], cmap='gray')
        axes3[1].set_title(f"各向异性扩散\n{param_labels[0]}")
        axes3[1].axis('off')
        
        axes3[2].imshow(results[0][1], cmap='gray')
        axes3[2].set_title("BM3D去噪后")
        axes3[2].axis('off')
    
    plt.suptitle(f"不同{param_name_map[parameter]}对OCT图像处理的影响", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    fig3.savefig(f"{save_dir}/{parameter}_complete_comparison.png", dpi=300)
    
    print(f"参数{parameter}的测试完成，结果已保存到 {save_dir} 目录")
    return fig1, fig2, fig3

if __name__ == "__main__":
    # 设置图像路径
    image_path = 'data/RetinalOCT_Dataset/raw/train/DR/dr_train_1001.jpg'
    
    if not os.path.exists(image_path):
        print(f"错误：找不到图像文件 {image_path}")
        exit(1)
    
    # 测试不同参数组合的效果
    print("\n===== 测试不同参数组合的各向异性扩散效果 =====")
    test_anisotropic_diffusion_parameters(image_path)
    
    # 测试不同迭代次数的效果
    print("\n===== 测试不同迭代次数(niter)的效果 =====")
    test_specific_parameter_range(image_path, parameter='niter')
    
    # 测试不同传导系数的效果
    print("\n===== 测试不同传导系数(kappa)的效果 =====")
    test_specific_parameter_range(image_path, parameter='kappa')
    
    # 测试不同步长的效果
    print("\n===== 测试不同步长(gamma)的效果 =====")
    test_specific_parameter_range(image_path, parameter='gamma')
    
    # 测试不同扩散函数的效果
    print("\n===== 测试不同扩散函数(option)的效果 =====")
    test_specific_parameter_range(image_path, parameter='option')
    
    print("\n所有测试完成！结果已保存为PNG图像。") 