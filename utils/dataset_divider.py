import os

from utils.mat_data_loader import load_and_slice_mat_images

current_dir = os.getcwd() + "/../data/2015_BOE_Chiu/"
sliced_images_np = load_and_slice_mat_images(current_dir)

# 将数据划分为测试集、验证集和训练集，比例分别为10%、10%和80%，并打乱数据，并将数据保存为jpg格式，分别放到test、val和train文件夹下
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
train, test, val = train_test_split(sliced_images_np, test_size=0.1, random_state=42)
