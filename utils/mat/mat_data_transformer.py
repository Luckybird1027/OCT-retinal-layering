import os

import numpy as np
import scipy.io as sio

# 假设的灰度值映射 (基于 8 条边界线 -> 9 个区域)
# 这些值是从 model/dataset.py 的 gray_to_label 中提取的前9个值
# 你可能需要根据数据集的实际层定义来调整这个映射
# 区域 0 (ILM以上): 0, 区域 1 (ILM-L2): 26, ..., 区域 8 (L8以下): 204
LABEL_GRAY_VALUES = [0, 26, 51, 77, 102, 128, 153, 179, 204]


def create_label_map(layers_data, height, width, gray_values):
    """
    根据边界线数据创建灰度标签图。

    参数:
        layers_data (np.array): 单个切片的边界线数据 (num_boundaries, width)
        height (int): 图像高度
        width (int): 图像宽度
        gray_values (list): 用于填充每个区域的灰度值列表

    返回:
        np.array: 生成的灰度标签图 (height, width)
    """
    label_map = np.zeros((height, width), dtype=np.uint8)
    num_boundaries = layers_data.shape[0]  # 通常是 8

    if len(gray_values) < num_boundaries + 1:
        # 如果灰度值不够覆盖所有区域，则填充为0并打印警告
        print(f"Warning: Not enough gray values ({len(gray_values)}) for {num_boundaries + 1} regions. Filling with 0.")
        return label_map  # 返回全黑图

    for x in range(width):
        # 获取当前列的边界 y 坐标
        y_coords = layers_data[:, x]

        # 处理 NaN 值 (替换为 0) 并转换为整数，限制在图像高度范围内
        y_coords = np.nan_to_num(y_coords, nan=0.0)
        y_coords = np.round(y_coords).astype(int)
        y_coords = np.clip(y_coords, 0, height - 1)  # 确保在 [0, height-1] 范围内

        # 对 y 坐标排序，以处理可能的顺序错误
        # argsort 返回的是排序后的索引
        sorted_indices = np.argsort(y_coords)
        y_coords_sorted = y_coords[sorted_indices]

        # 确保使用的灰度值索引与排序后的边界线对应
        # 注意：这里我们假设 gray_values[0] 对应第一条边界线以上的区域，
        # gray_values[1] 对应第一条和第二条边界线之间的区域，以此类推。
        current_gray_idx = 0

        # 填充第一个区域 (从 0 到第一条边界线)
        if y_coords_sorted[0] > 0:
            label_map[0:y_coords_sorted[0], x] = gray_values[current_gray_idx]
        current_gray_idx += 1

        # 填充中间区域 (边界线之间)
        for i in range(num_boundaries - 1):
            y_start = y_coords_sorted[i]
            y_end = y_coords_sorted[i + 1]
            if y_end > y_start:  # 只有当结束坐标大于起始坐标时才填充
                label_map[y_start:y_end, x] = gray_values[current_gray_idx]
            current_gray_idx += 1

        # 填充最后一个区域 (最后一条边界线到图像底部)
        if y_coords_sorted[-1] < height:
            label_map[y_coords_sorted[-1]:height, x] = gray_values[current_gray_idx]

    return label_map


def load_and_slice_mat_data(directory, image_key='images', layer_key='manualLayers1'):
    """
    读取指定路径的所有mat文件，切分成单张图像和对应的标签图。

    参数:
    directory (str): 存放mat文件的路径
    image_key (str): 图像数据的键名 (默认为 'images')
    layer_key (str): 用于生成标签图的边界线数据的键名 (默认为 'manualLayers1')

    返回:
    tuple: (np.array: 切分后的图像数组 (uint8), np.array: 切分后的标签图数组 (uint8))

    使用示例:
    ```python
    current_dir = os.path.join(os.getcwd(), "..", "data", "2015_BOE_Chiu")
    sliced_images_np, sliced_labels_np = load_and_slice_mat_data(current_dir)
    ```
    """
    file_name_array = os.listdir(directory)
    all_images_data = []
    all_layers_data = []  # 存储边界线数据

    print(f"Scanning directory: {directory}")
    for file_name in file_name_array:
        if file_name.endswith(".mat"):
            mat_path = os.path.join(directory, file_name)
            print(f"Loading {file_name}...")
            try:
                mat_data = sio.loadmat(mat_path)
                if image_key in mat_data and layer_key in mat_data:
                    images = mat_data[image_key]
                    layers = mat_data[layer_key]

                    # 检查数据维度是否匹配 (H, W, Slices) vs (Num_Layers, W, Slices)
                    if images.shape[1:] == layers.shape[1:]:
                        # 确保数据类型是浮点数以便后续处理
                        all_images_data.append(images.astype(np.float64))
                        all_layers_data.append(layers.astype(np.float64))
                        print(f"  - Loaded images: {images.shape}, layers: {layers.shape}")
                    else:
                        print(
                            f"  - Skipping {file_name}: Mismatched dimensions between images {images.shape} and layers {layers.shape}")
                else:
                    print(f"  - Skipping {file_name}: Missing '{image_key}' or '{layer_key}' key.")
            except Exception as e:
                print(f"  - Error loading {file_name}: {e}")
        else:
            print(f"Skipping non-mat file: {file_name}")

    sliced_images = []
    sliced_labels = []

    print("\nSlicing images and generating label maps...")
    for images, layers in zip(all_images_data, all_layers_data):
        height, width, num_slices = images.shape
        num_layer_boundaries = layers.shape[0]

        # 检查是否有足够的灰度值
        if len(LABEL_GRAY_VALUES) < num_layer_boundaries + 1:
            print(
                f"Warning: Not enough gray values ({len(LABEL_GRAY_VALUES)}) for {num_layer_boundaries + 1} regions defined by {num_layer_boundaries} boundaries. Skipping label generation for this file.")
            # 仍然处理图像，但标签将是空的或不生成
            # continue # 如果希望完全跳过此文件

        print(f"  - Processing volume with shape: images={images.shape}, layers={layers.shape}")
        for i in range(num_slices):
            image_slice = images[:, :, i]
            layer_slice = layers[:, :, i]  # Shape: (num_boundaries, width)

            # 将图像数据从 double 转换为 uint8 (0-255)
            # **重要**: 确认原始 'images' 数据的范围。
            # 如果是 0-1，则需要乘以 255.0。这里假设是 0-255 范围。
            # image_slice_uint8 = np.clip(image_slice * 255.0, 0, 255).astype(np.uint8) # 如果范围是 0-1
            image_slice_uint8 = np.clip(image_slice, 0, 255).astype(np.uint8)  # 如果范围是 0-255
            sliced_images.append(image_slice_uint8)

            # 创建标签图
            # 只有在灰度值足够时才创建有效的标签图
            if len(LABEL_GRAY_VALUES) >= num_layer_boundaries + 1:
                label_map = create_label_map(layer_slice, height, width, LABEL_GRAY_VALUES)
                sliced_labels.append(label_map)
            else:
                # 如果灰度值不够，可以添加一个全零的标签图以保持对应关系
                sliced_labels.append(np.zeros((height, width), dtype=np.uint8))

    if not sliced_images:
        print("No images were successfully sliced.")
        return np.array([]), np.array([])

    sliced_images_np = np.array(sliced_images)
    sliced_labels_np = np.array(sliced_labels)
    print(f"\nTotal sliced images: {sliced_images_np.shape[0]}")
    print(f"Sliced images np shape: {sliced_images_np.shape}")
    if sliced_labels_np.size > 0:
        print(f"Sliced labels np shape: {sliced_labels_np.shape}")
    else:
        print("No labels were generated.")
    return sliced_images_np, sliced_labels_np
