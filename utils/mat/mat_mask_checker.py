import os

import numpy as np
import scipy.io as sio

# --- 配置 ---
# 修改为你存放 .mat 文件的实际路径
mat_file_path = os.path.join(os.getcwd(), "data", "2015_BOE_Chiu", "mat", "Subject_01.mat")  # 替换成你的一个 mat 文件名
layer_key = 'manualLayers1'  # 或者 'manualLayers2'
image_key = 'images'
# --- 配置结束 ---
if not os.path.exists(mat_file_path):
    print(f"错误：找不到 MAT 文件: {mat_file_path}")
else:
    try:
        print(f"正在加载: {mat_file_path}")
        mat_data = sio.loadmat(mat_file_path)
        if layer_key in mat_data and image_key in mat_data:
            layers = mat_data[layer_key]  # shape: (num_layers, width, slices)
            images = mat_data[image_key]  # shape: (height, width, slices)
            image_height = images.shape[0]
            print(f"图像高度 (来自 'images' key): {image_height}")
            # 确保 layers 是 numpy 数组
            layers = np.array(layers, dtype=np.float64)  # 转换为 float64 以处理 NaN
            # 计算 Y 坐标的统计信息 (忽略 NaN 值)
            try:
                min_y = np.nanmin(layers)
                max_y = np.nanmax(layers)
                mean_y = np.nanmean(layers)
                median_y = np.nanmedian(layers)
                print(f"\n--- '{layer_key}' Y 坐标统计信息 ---")
                print(f"最小值 (Min Y): {min_y}")
                print(f"最大值 (Max Y): {max_y}")
                print(f"平均值 (Mean Y): {mean_y}")
                print(f"中位数 (Median Y): {median_y}")
                # 打印一些样本值 (例如，第一个切片，第一条边界线的前 10 个 Y 坐标)
                if layers.ndim == 3 and layers.shape[0] > 0 and layers.shape[1] > 10 and layers.shape[2] > 0:
                    print("\n第一个切片，第一条边界线的前 10 个 Y 坐标:")
                    print(layers[0, :10, 0])
                elif layers.ndim == 2 and layers.shape[0] > 0 and layers.shape[1] > 10:  # 如果数据是 2D
                    print("\n第一条边界线的前 10 个 Y 坐标:")
                    print(layers[0, :10])
            except Exception as e_stat:
                print(f"\n计算统计信息时出错: {e_stat}")
                print("可能是因为所有值都是 NaN？尝试打印原始数据形状:")
                print(f"Layers shape: {layers.shape}")
            # --- 解释 ---
            print("\n--- 可能的解释 ---")
            print(f"图像高度约为: {image_height}")
            if min_y is not None and max_y is not None:
                if 0 <= min_y < image_height and 0 < max_y <= image_height:
                    if max_y < image_height * 0.8:  # 如果最大值远小于图像高度
                        print("坐标范围似乎在 [0, 图像高度] 内，但最大值较小。")
                        print("-> 这可能意味着 Y=0 是图像顶部，但标注可能集中在上方，或者存在很多 NaN 值被处理为 0。")
                    else:
                        print("坐标范围似乎在 [0, 图像高度] 内。")
                        print("-> Y=0 很可能代表图像顶部。")
                elif min_y > image_height or max_y > image_height:
                    print("坐标值超出了图像高度！")
                    print("-> 坐标系可能不是从顶部 (0) 开始，或者单位不是像素。需要检查文档。")
                elif min_y < 0:
                    print("坐标值包含负数！")
                    print("-> 坐标系定义非常规，需要检查文档。")
            else:
                print("无法计算有效的最小/最大值，可能数据全是 NaN 或格式有问题。")
        else:
            print(f"错误：在 MAT 文件中找不到 key '{layer_key}' 或 '{image_key}'")
            print(f"可用的 keys: {list(mat_data.keys())}")
    except Exception as e:
        print(f"加载或处理 MAT 文件时出错: {e}")
