import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# 假设这些模块位于项目的 model/ 和 utils/ 目录下
from model.unet import UNet
from utils.visualization import generate_color_map, create_difference_map
from utils.preprocess import standardize_size, ad_bm3d_image_denoising, pc_hpf_image_enhancement

# 设置matplotlib支持中文 (如果需要，并确保SimHei字体已安装)
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_image(image_path, mask_path=None, img_size=(512, 704), preprocess_filters=True):
    """
    加载并预处理单张图像（和可选的掩码）。

    返回:
        image_tensor: 预处理后的图像张量 (1, 1, H, W)
        true_mask_tensor: 预处理后的真实掩码张量 (1, H, W)，如果提供了mask_path则返回，否则为None
        original_image_std_np: 标准化尺寸后的原始图像 (H, W, uint8)，用于可视化
        true_mask_indices_np: 真实掩码的类别索引 (H, W, uint8)，如果提供了mask_path则返回，否则为None
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像未在路径 {image_path}找到")

    true_mask_cv2 = None
    if mask_path:
        true_mask_cv2 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if true_mask_cv2 is None:
            print(f"警告: 掩码未在路径 {mask_path}找到。将不使用掩码进行比较。")

    # 标准化尺寸
    # standardize_size 会返回 (image_std, mask_std)，mask_std 在 mask_path 为 None 时也是 None
    img_std_np, mask_std_np = standardize_size(img, true_mask_cv2, target_size=img_size)

    # 归一化图像至 [0, 1] 范围的浮点数
    img_norm_np = img_std_np / 255.0

    # 应用高级预处理（如果启用）
    if preprocess_filters:
        # 这些函数期望输入是 [0,1] 范围的浮点数，并返回相同类型的图像
        img_processed_np = ad_bm3d_image_denoising(img_norm_np.copy()) # 使用 .copy() 避免修改原始数组
        img_processed_np = pc_hpf_image_enhancement(img_processed_np)
    else:
        img_processed_np = img_norm_np

    # 将图像转换为 PyTorch 张量: [1, 1, H, W]
    image_tensor = torch.from_numpy(img_processed_np).float().unsqueeze(0).unsqueeze(0)

    true_mask_tensor = None
    true_mask_indices_np = None
    if mask_std_np is not None:
        # 将掩码的灰度值映射到类别索引 (同 OCTDataset 中的逻辑)
        gray_to_label = {
            0: 0, 26: 1, 51: 2, 77: 3, 102: 4, 128: 5,
            153: 6, 179: 7, 204: 8, 230: 9, 255: 10
        }
        lookup_table = np.zeros(256, dtype=np.uint8)
        for gray_value, label in gray_to_label.items():
            lookup_table[gray_value] = label
        
        true_mask_indices_np = lookup_table[mask_std_np]
        true_mask_tensor = torch.from_numpy(true_mask_indices_np).long().unsqueeze(0) # [1, H, W]
    
    return image_tensor, true_mask_tensor, img_std_np, true_mask_indices_np


def predict_and_display_single(model, image_tensor, original_image_np,
                               true_mask_indices_np, # Numpy array of true mask labels, or None
                               device, num_classes=11, image_path_for_title="Image"):
    """
    对单张图像进行预测并使用matplotlib显示结果。
    original_image_np: 原始的、仅经过尺寸标准化的灰度图像 (uint8)
    true_mask_indices_np: 真实掩码的类别索引 (uint8), 或 None
    """
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device) # Shape [1, 1, H, W]

        output = model(image_tensor)  # (1, C, H, W)
        output_probs = torch.softmax(output, dim=1)
        pred_indices_np = torch.argmax(output_probs, dim=1).squeeze().cpu().numpy()  # (H, W)

    # --- 可视化 ---
    color_map = generate_color_map(num_classes)

    # 创建预测分割图 (RGB)
    segmentation_map_rgb = np.zeros((pred_indices_np.shape[0], pred_indices_np.shape[1], 3), dtype=np.uint8)
    for class_idx in range(num_classes):
        segmentation_map_rgb[pred_indices_np == class_idx] = color_map[class_idx]

    # 原始图像转为RGB（用于混合和差异图）
    original_rgb = cv2.cvtColor(original_image_np, cv2.COLOR_GRAY2RGB)

    # 创建混合图像 (原始图像 + 半透明分割图)
    blend_alpha = 0.5
    blend_rgb = cv2.addWeighted(original_rgb, 1 - blend_alpha, segmentation_map_rgb, blend_alpha, 0)

    # --- 绘图 ---
    plot_titles = ['Original Image', 'Predicted Segmentation', 'Prediction Overlay']
    # original_image_np 是灰度图，其他是RGB图
    images_to_plot = [original_image_np, segmentation_map_rgb, blend_rgb] 

    if true_mask_indices_np is not None:
        # 创建真实掩码图 (RGB)
        true_mask_map_rgb = np.zeros((true_mask_indices_np.shape[0], true_mask_indices_np.shape[1], 3), dtype=np.uint8)
        for class_idx in range(num_classes):
            true_mask_map_rgb[true_mask_indices_np == class_idx] = color_map[class_idx]
        
        # 创建差异对比图
        diff_map_rgb = create_difference_map(original_rgb, pred_indices_np, true_mask_indices_np, alpha=0.6)

        # 插入真实掩码和差异图到绘图列表
        plot_titles.insert(1, 'Ground Truth Mask')
        images_to_plot.insert(1, true_mask_map_rgb)
        plot_titles.append('Difference Map (Correct:Green, Error:Red)')
        images_to_plot.append(diff_map_rgb)

    num_subplots = len(images_to_plot)
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5.5), dpi=120)
    if num_subplots == 1: # 确保axes始终是可迭代的
        axes = [axes]

    for i, ax in enumerate(axes):
        if plot_titles[i] == 'Original Image': # 原始图像使用灰度图cmap
            ax.imshow(images_to_plot[i], cmap='gray')
        else: # 其他图像是RGB格式
            ax.imshow(images_to_plot[i])
        ax.set_title(plot_titles[i])
        ax.axis('off')
    
    fig.suptitle(f"Segmentation for: {os.path.basename(image_path_for_title)}", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # 为suptitle留出空间
    plt.show()

    # 如果需要，可以取消注释以下代码来保存组合图像
    # output_dir = "single_prediction_output"
    # os.makedirs(output_dir, exist_ok=True)
    # fig_save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(image_path_for_title))[0]}_summary.png")
    # try:
    #     plt.savefig(fig_save_path)
    #     print(f"组合可视化图像已保存到: {fig_save_path}")
    # except Exception as e:
    #     print(f"保存图像失败: {e}")
    # plt.close(fig)


def main_predict_single():
    """主函数，用于加载模型和图像，并进行预测和可视化。"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # --- 用户配置区域 ---
    # 请根据你的实际情况修改以下路径和参数
    model_path = 'train/checkpoints_512x704_gr64/best_model.pth'  # 训练好的模型路径
    
    # 输入图像路径 (示例)
    image_path = 'data/SJTU/test/img/6_L_08_flip.bmp'
    # 可选的真实掩码路径 (如果不需要比较，设为 None)
    # mask_path = 'data/SJTU/test/mask/6_L_08_flip.bmp'
    mask_path = None # 如果没有真实掩码，取消注释此行

    num_classes = 11  # 确保与模型训练时的类别数一致
    img_size = (512, 704)  # 确保与模型训练/数据集定义一致
    preprocess_filters_enabled = True # 是否在加载时应用AD-BM3D和PC-HPF预处理
    # --- 配置区域结束 ---

    # 加载模型
    model = UNet(in_channels=1, out_channels=num_classes).to(device)
    try:
        # 优先尝试直接加载 state_dict
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e1:
        print(f"直接加载 state_dict 失败: {e1}。尝试加载 checkpoint 字典...")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict): # 如果checkpoint本身就是state_dict
                 model.load_state_dict(checkpoint)
            else:
                raise TypeError("加载的checkpoint文件不是一个可识别的state_dict或字典格式。")
        except Exception as e2:
            raise ValueError(f"从 {model_path} 加载模型失败。详细错误: {e2}")
    
    print(f"模型已从 {model_path} 加载。")

    # 加载和预处理图像 (及可选掩码)
    try:
        image_tensor, _, original_image_np, true_mask_indices_np = \
            load_and_preprocess_image(image_path, mask_path, img_size, preprocess_filters_enabled)
        # _ 对应 true_mask_tensor, 在此函数中我们主要使用其numpy版本 true_mask_indices_np 进行可视化
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    except Exception as e:
        print(f"加载或预处理图像时发生错误: {e}")
        return
    
    print(f"图像 '{os.path.basename(image_path)}' 已加载并预处理。")
    if mask_path and true_mask_indices_np is not None:
        print(f"真实掩码 '{os.path.basename(mask_path)}' 已加载并预处理。")
    elif mask_path and true_mask_indices_np is None: # mask_path提供了，但加载失败
        print(f"警告: 真实掩码 '{os.path.basename(mask_path)}' 未能成功加载，将不显示真值和差异图。")
    elif not mask_path:
        print("未提供真实掩码路径，将不显示真值和差异图。")


    # 进行预测并显示结果
    predict_and_display_single(model, image_tensor, original_image_np,
                               true_mask_indices_np, # 直接传递处理后的Numpy掩码或None
                               device, num_classes,
                               image_path_for_title=image_path)

if __name__ == '__main__':
    # 在某些环境/后端下，显式设置matplotlib后端可能有助于解决显示问题
    # import matplotlib
    # matplotlib.use('TkAgg') # 或 'Qt5Agg', 'Agg' (用于非GUI环境保存文件)等
    main_predict_single()