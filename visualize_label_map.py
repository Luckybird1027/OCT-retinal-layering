import argparse
import os

import cv2
import numpy as np

# --- 项目中复用的定义和函数 ---
# 以下定义和函数逻辑源自您项目中的 utils/visualization.py 和 model/dataset.py 文件。
# 为了使此脚本能够独立运行，相关逻辑在此处进行了复现。

# 源自 model/dataset.py 中的灰度值到类别标签的映射
GRAY_TO_LABEL = {
    0: 0,  # 背景
    26: 1,  # RNFL (视网膜神经纤维层)
    51: 2,  # GCL+IPL (神经节细胞层 + 内丛状层)
    77: 3,  # INL (内核层)
    102: 4,  # OPL (外丛状层)
    128: 5,  # ONL (外核层)
    153: 6,  # IS/OS (内段/外段，光感受器层)
    179: 7,  # RPE (视网膜色素上皮)
    204: 8,  # Choroid (脉络膜)
    230: 9,  # Vitreous (玻璃体)
    255: 10  # Sclera (巩膜)
}
# 根据上述映射，总共有11个类别 (0到10)
NUM_CLASSES = 11


def create_lookup_table(gray_to_label_map):
    """
    根据灰度值到类别标签的映射，创建一个查找表。
    此逻辑与 model/dataset.py 中 OCTDataset类的初始化部分一致。
    """
    lookup_table = np.zeros(256, dtype=np.uint8)
    for gray_value, label_index in gray_to_label_map.items():
        lookup_table[gray_value] = label_index
    return lookup_table


# 源自 utils/visualization.py 中的颜色图生成逻辑
def generate_color_map_rgb(num_classes=NUM_CLASSES):
    """
    生成用于可视化的颜色映射表 (RGB格式)。
    此函数与 utils/visualization.py 中的 generate_color_map 功能一致。
    """
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    # 背景类 - 黑色
    color_map[0] = [0, 0, 0]
    # 视网膜不同层次的颜色 (与项目中定义一致)
    color_map[1] = [255, 0, 0]  # 红色 - RNFL
    color_map[2] = [0, 255, 0]  # 绿色 - GCL+IPL
    color_map[3] = [0, 0, 255]  # 蓝色 - INL
    color_map[4] = [255, 255, 0]  # 黄色 - OPL
    color_map[5] = [0, 255, 255]  # 青色 - ONL
    color_map[6] = [255, 0, 255]  # 紫色 - IS/OS
    color_map[7] = [128, 0, 0]  # 深红色 - RPE
    color_map[8] = [0, 128, 0]  # 深绿色 - Choroid
    color_map[9] = [0, 0, 128]  # 深蓝色 - Vitreous
    color_map[10] = [128, 128, 0]  # 橄榄色 - Sclera
    return color_map


# --- 主要功能函数 ---
def convert_gray_label_to_color(gray_label_path, output_color_path):
    """
    将输入的灰度标签图像转换为彩色可视化图像并保存。

    参数:
        gray_label_path (str): 输入的灰度标签图像文件路径。
                               图像应包含项目中定义的特定灰度值（如0, 26, 51等）。
        output_color_path (str): 输出的彩色标签图像文件路径。
    """
    # 1. 初始化查找表和颜色映射表
    lookup_table = create_lookup_table(GRAY_TO_LABEL)
    color_map_rgb = generate_color_map_rgb(NUM_CLASSES)

    # 2. 读取灰度标签图像
    gray_mask_image = cv2.imread(gray_label_path, cv2.IMREAD_GRAYSCALE)
    if gray_mask_image is None:
        print(f"错误：无法读取图像 '{gray_label_path}'。请检查文件路径和文件是否有效。")
        return False

    # 3. 将灰度值映射到类别索引
    #    假设 gray_mask_image 中的像素值是原始的掩码灰度值 (0, 26, 51, ...)
    label_indices_map = lookup_table[gray_mask_image]

    # 4. 创建彩色标签图
    height, width = gray_mask_image.shape
    color_label_image_rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for class_idx in range(NUM_CLASSES):
        # 找到所有属于当前类别的像素区域
        pixels_of_current_class = (label_indices_map == class_idx)
        # 为这些像素分配相应的颜色
        color_label_image_rgb[pixels_of_current_class] = color_map_rgb[class_idx]

    # 5. 保存彩色图像
    #    OpenCV 在写入图像时默认使用 BGR 颜色顺序。
    #    由于我们的 color_map_rgb 是RGB格式，因此需要转换。
    color_label_image_bgr = cv2.cvtColor(color_label_image_rgb, cv2.COLOR_RGB2BGR)

    try:
        # 确保输出目录存在，如果不存在则创建它
        output_dir = os.path.dirname(output_color_path)
        if output_dir and not os.path.exists(output_dir):  # 检查 output_dir 是否为空字符串
            os.makedirs(output_dir)
            print(f"已创建目录: {output_dir}")

        cv2.imwrite(output_color_path, color_label_image_bgr)
        print(f"彩色标签图像已成功保存到: {output_color_path}")
        return True
    except Exception as e:
        print(f"错误：保存图像到 '{output_color_path}' 失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="将项目特定的灰度标签图像转换为易于理解的彩色可视化图像。",
        formatter_class=argparse.RawTextHelpFormatter  # 用于在帮助信息中更好地显示换行符
    )
    parser.add_argument(
        "--input_gray_label_path",
        type=str,
        default="data/SJTU/train/mask/7_L_07.bmp",
        help="输入的灰度标签图像文件路径。\n"
             "例如: data/SJTU/train/mask/1_R_07.bmp\n"
             "此图像应为单通道灰度图，其中像素值对应于项目定义的特定灰度级\n"
             "(如 0 代表背景, 26 代表 RNFL, 51 代表 GCL+IPL, 等等)。",
    )
    parser.add_argument(
        "--output_color_label_path",
        type=str,
        default="chart/output_visualizations/7_L_07_color_label.png",
        help="输出的彩色标签图像文件路径。\n"
             "例如: chart/output_visualizations/1_R_07_color_label.png\n"
             "如果指定的输出目录不存在，脚本将尝试创建它。",

    )
    parser.add_argument(
        "--show",
        action="store_true",  # 当此参数存在时，其值为True
        help="可选参数。如果提供，脚本将在处理完成后使用OpenCV显示原始灰度图和生成的彩色图。"
    )

    args = parser.parse_args()

    success = convert_gray_label_to_color(args.input_gray_label_path, args.output_color_label_path)

    if success and args.show:
        try:
            # 重新读取已保存的图像进行显示
            gray_img_to_show = cv2.imread(args.input_gray_label_path, cv2.IMREAD_GRAYSCALE)
            # 读取已保存的彩色图（cv2.imread 读取的是BGR格式，可以直接显示）
            color_img_to_show = cv2.imread(args.output_color_label_path)

            if gray_img_to_show is not None and color_img_to_show is not None:
                cv2.imshow("Original Grayscale Label", gray_img_to_show)
                cv2.imshow("Generated Color Label", color_img_to_show)
                print("\n按任意键关闭图像显示窗口...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                if gray_img_to_show is None:
                    print(f"警告: 无法重新读取灰度图 '{args.input_gray_label_path}' 以进行显示。")
                if color_img_to_show is None:
                    print(f"警告: 无法重新读取已生成的彩色图 '{args.output_color_label_path}' 以进行显示。")
        except Exception as e:
            print(f"显示图像时发生错误: {e}")


if __name__ == "__main__":
    main()
