import os

import numpy as np
import scipy.io as sio

def load_and_slice_mat_images(directory):
    """
    读取指定路径的所有mat文件，并将其切分成单张图像。
    
    参数:
    directory (str): 存放mat文件的路径
    
    返回:
    np.array: 切分后的图像数组

    使用示例:
    ```python
    current_dir = os.getcwd() + "/../data/2015_BOE_Chiu/"
    sliced_images_np = load_and_slice_mat_images(current_dir)
    ```
    """
    file_name_array = os.listdir(directory)
    all_images = []
    
    for file_name in file_name_array:
        if file_name.endswith(".mat"):
            print("loading {}".format(file_name))
            mat_data = sio.loadmat(os.path.join(directory, file_name))
            all_images.append(mat_data["images"])
    
    sliced_images = []
    for image in all_images:
        for i in range(image.shape[2]):
            sliced_images.append(image[:, :, i])
    
    sliced_images_np = np.array(sliced_images)
    print("sliced images np shape: {}".format(sliced_images_np.shape))
    return sliced_images_np