import os
from PIL import Image
import numpy as np
from numpy.random import shuffle

from mat_data_transformer import load_and_slice_mat_data

split_ratio = [0.1, 0.1, 0.8]  # test, val, train 比例
# 使用 os.path.join 构造路径，更健壮
source_data_dir = os.path.join(os.getcwd(), "data", "2015_BOE_Chiu", "mat")
# 建议将处理后的数据保存到新目录，避免覆盖或混淆原始数据
output_base_dir = os.path.join(os.getcwd(), "data", "2015_BOE_Chiu_Processed")

# 调用更新后的函数，获取图像和标签数据
print("Loading and slicing MAT data...")
# 可以指定 layer_key='manualLayers2' 如果需要的话
sliced_images_np, sliced_labels_np = load_and_slice_mat_data(source_data_dir, layer_key='manualLayers1')

# 检查是否成功加载数据
if sliced_images_np.size == 0:
    print("No data loaded. Exiting.")
    exit()
if sliced_labels_np.size == 0:
    print("Warning: Labels could not be generated. Only images will be processed.")
elif sliced_images_np.shape[0] != sliced_labels_np.shape[0]:
    print(
        f"Error: Mismatch between number of images ({sliced_images_np.shape[0]}) and labels ({sliced_labels_np.shape[0]}). Exiting.")
    exit()

# 创建输出目录结构 (test/img, test/mask, val/img, val/mask, train/img, train/mask)
sets = ["test", "val", "train"]
print(f"\nCreating output directories in: {output_base_dir}")
for s in sets:
    os.makedirs(os.path.join(output_base_dir, s, "img"), exist_ok=True)
    # 只有在标签存在时才创建 mask 目录
    if sliced_labels_np.size > 0:
        os.makedirs(os.path.join(output_base_dir, s, "mask"), exist_ok=True)

# 打乱索引以保持图像和标签对应
num_samples = len(sliced_images_np)
indices = np.arange(num_samples)
shuffle(indices)
print(f"\nShuffled {num_samples} samples.")

# 计算分割点
test_split_idx = int(num_samples * split_ratio[0])
val_split_idx = test_split_idx + int(num_samples * split_ratio[1])

print(
    f"Splitting into: Test={test_split_idx}, Val={val_split_idx - test_split_idx}, Train={num_samples - val_split_idx}")

# 划分数据集并保存
print("Splitting and saving data...")
for i, original_idx in enumerate(indices):
    image_np = sliced_images_np[original_idx]
    # 只有在标签存在时才获取标签
    label_np = sliced_labels_np[original_idx] if sliced_labels_np.size > 0 else None

    # 将 numpy 数组转换为 PIL Image 对象
    # 图像已经是 uint8
    img_pil = Image.fromarray(image_np)
    # 标签图也已经是 uint8 灰度图
    mask_pil = Image.fromarray(label_np) if label_np is not None else None

    # 根据索引确定属于哪个集 (test, val, train) 并构造文件名和路径
    if i < test_split_idx:
        set_name = "test"
        # 使用 i 作为文件名索引，确保从 0 开始且连续
        file_idx = i
    elif i < val_split_idx:
        set_name = "val"
        file_idx = i - test_split_idx
    else:
        set_name = "train"
        file_idx = i - val_split_idx

    # 使用 PNG 格式保存图像和标签，避免有损压缩
    img_filename = f"{set_name}_{file_idx}.png"
    mask_filename = f"{set_name}_{file_idx}.png"  # 标签也用 png

    img_path = os.path.join(output_base_dir, set_name, "img", img_filename)
    mask_path = os.path.join(output_base_dir, set_name, "mask", mask_filename) if mask_pil is not None else None

    # 保存图像
    try:
        img_pil.save(img_path)
    except Exception as e:
        print(f"Error saving image {img_path}: {e}")

    # 如果标签存在，则保存标签
    if mask_path and mask_pil:
        try:
            mask_pil.save(mask_path)
        except Exception as e:
            print(f"Error saving mask {mask_path}: {e}")

    # 打印进度 (可选)
    if (i + 1) % 100 == 0 or (i + 1) == num_samples:
        print(f"  Saved {i + 1}/{num_samples} samples...")

print("\nData splitting and saving complete.")
print(f"Output saved to: {output_base_dir}")
