import os
import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt

current_dir = os.getcwd() + "/../data/2015_BOE_Chiu/"
file_name_array = os.listdir(current_dir)[0:1]

all_images = []
for file_name in file_name_array:
    if "mat" in file_name:
        print("loading {}".format(file_name))
        mat_data = sio.loadmat(os.path.join(current_dir, file_name))
        all_images.append(mat_data["images"])

print("all images shape: {}".format(all_images[0].shape))

first_image = all_images[0]

# 生成3D网格点，注意顺序调整为 y, x, z
y, x, z = np.mgrid[0:first_image.shape[0], 0:first_image.shape[1], 0:first_image.shape[2]]

# 提取非零值（避免显示全零区域）
mask = first_image > 0  # 根据数据调整阈值

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[mask], y[mask], z[mask], c=first_image[mask], cmap='gray', s=1)
plt.show()