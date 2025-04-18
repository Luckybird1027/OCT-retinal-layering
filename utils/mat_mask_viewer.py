import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

mat_file_path = os.path.join(os.getcwd(), "data", "2015_BOE_Chiu", "mat", "Subject_01.mat") # 替换成你的 mat 文件
layer_key = 'manualLayers1'
image_key = 'images'
slice_index = 15

if not os.path.exists(mat_file_path):
    print(f"错误：找不到 MAT 文件: {mat_file_path}")
else:
    try:
        mat_data = sio.loadmat(mat_file_path)
        if image_key in mat_data and layer_key in mat_data:
            images = mat_data[image_key]
            layers = mat_data[layer_key] # shape: (num_layers, width, slices)

            # 检查数据维度和索引
            if images.ndim == 3 and layers.ndim == 3 and images.shape[2] > slice_index and layers.shape[2] > slice_index:
                image_slice = images[:, :, slice_index]
                num_layers = layers.shape[0]
                width = image_slice.shape[1]
                x_coords = np.arange(width)

                plt.figure(figsize=(12, 7)) # 可以适当调整图像大小
                plt.imshow(image_slice, cmap='gray')
                plt.title(f'Image Slice {slice_index} with All Layer Boundaries')
                plt.xlabel('Width (X-coordinate)')
                plt.ylabel('Height (Y-coordinate)')

                # 定义一组颜色用于区分不同的层
                # 你可以根据需要添加更多颜色，或者使用 matplotlib 的 colormap
                colors = plt.cm.get_cmap('tab10', num_layers) # 使用 tab10 colormap 获取 num_layers 种颜色

                print(f"找到 {num_layers} 条边界线，正在绘制...")

                all_y_coords_min = []
                all_y_coords_max = []

                # 循环绘制每一条边界线
                for layer_index in range(num_layers):
                    layer_slice_data = layers[layer_index, :, slice_index] # 获取当前层和切片的 Y 坐标

                    # 过滤掉 NaN 值以便绘图
                    valid_indices = ~np.isnan(layer_slice_data)
                    y_coords_valid = layer_slice_data[valid_indices]
                    x_coords_valid = x_coords[valid_indices]

                    if len(y_coords_valid) > 0: # 只有在有有效点时才绘制和记录范围
                        # 绘制散点图，减小 s 的值使点更细
                        plt.scatter(x_coords_valid, y_coords_valid,
                                    c=[colors(layer_index)], # 使用预定义的颜色
                                    s=1,  # <--- 减小点的大小，例如 1 或 2
                                    label=f'Layer {layer_index}')
                        all_y_coords_min.append(np.min(y_coords_valid))
                        all_y_coords_max.append(np.max(y_coords_valid))
                    else:
                        print(f"  - Layer {layer_index}: 没有有效的 Y 坐标点。")


                # 调整 Y 轴范围以更好地显示所有层
                if all_y_coords_min and all_y_coords_max: # 确保列表不为空
                    min_overall_y = min(all_y_coords_min)
                    max_overall_y = max(all_y_coords_max)
                    # 设置 Y 轴范围，留出一些边距
                    plt.ylim(max_overall_y + 30, min_overall_y - 30) # 增加边距
                else:
                    # 如果没有有效的点，使用图像高度作为默认范围
                    plt.ylim(image_slice.shape[0], 0)

                # 添加图例
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) # 将图例放到图像外部

                plt.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局，为图例腾出空间
                plt.show()

                print(f"\n绘制了切片 {slice_index} 的图像和所有 {num_layers} 条边界线的 Y 坐标。")
                print("观察不同颜色的点是否与图像中的解剖结构边界对齐。")

            else:
                 print("错误：数据维度不正确，或切片索引超出范围。")
                 print(f"Images shape: {images.shape}, Layers shape: {layers.shape}")

        else:
            print(f"错误：在 MAT 文件中找不到 key '{layer_key}' 或 '{image_key}'")

    except Exception as e:
        print(f"加载或处理 MAT 文件时出错: {e}")