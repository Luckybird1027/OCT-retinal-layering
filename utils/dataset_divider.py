import os

from PIL import Image
from numpy.random import shuffle

from utils.mat_data_loader import load_and_slice_mat_images

split_ratio = [0.1, 0.1, 0.8]
current_dir = os.getcwd() + "/../data/2015_BOE_Chiu/"
sliced_images_np = load_and_slice_mat_images(current_dir)

# 创建test、val和train文件夹，用于存储数据
if not os.path.exists(current_dir[:-1] + "\\test"):
    os.makedirs(current_dir[:-1] + "\\test")
if not os.path.exists(current_dir[:-1] + "\\val"):
    os.makedirs(current_dir[:-1] + "\\val")
if not os.path.exists(current_dir[:-1] + "\\train"):
    os.makedirs(current_dir[:-1] + "\\train")

# 将数据划分为测试集、验证集和训练集，比例分别为10%、10%和80%，并打乱数据，并将数据保存为jpg格式，分别放到test、val和train文件夹下
shuffle(sliced_images_np)
for i, image_np in enumerate(sliced_images_np):
    if i < int(len(sliced_images_np) * split_ratio[0]):
        Image.fromarray(image_np).save(current_dir[:-1] + "\\test\\test_{}.jpg".format(i))
    elif i < int(len(sliced_images_np) * (split_ratio[0] + split_ratio[1])):
        Image.fromarray(image_np).save(
            current_dir[:-1] + "\\val\\val_{}.jpg".format(i - int(len(sliced_images_np) * split_ratio[0])))
    else:
        Image.fromarray(image_np).save(current_dir[:-1] + "\\train\\train_{}.jpg".format(
            i - int(len(sliced_images_np) * (split_ratio[0] + split_ratio[1]))))
