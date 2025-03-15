import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from augmentation import apply_random_augmentation
from preprocess import normalize_image, denoise_image, enhance_contrast

class OCTDataset(Dataset):
    """OCT图像数据集"""
    
    def __init__(self, image_dir, label_dir=None, transform=None, preprocess=True, augment=False):
        """
        初始化
        
        参数:
            image_dir: 图像目录
            label_dir: 标签目录（可选）
            transform: 额外的转换
            preprocess: 是否进行预处理
            augment: 是否进行数据增强
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.preprocess = preprocess
        self.augment = augment
        
        # 获取所有图像文件
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 读取图像
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 读取标签（如果有）
        label = None
        if self.label_dir is not None:
            label_path = os.path.join(self.label_dir, img_name)
            if os.path.exists(label_path):
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        # 预处理
        if self.preprocess:
            image = normalize_image(image)
            image = denoise_image(image)
            image = enhance_contrast(image)
        
        # 数据增强
        if self.augment:
            if label is not None:
                image, label = apply_random_augmentation(image, label)
            else:
                image = apply_random_augmentation(image)
        
        # 转换为PyTorch张量
        image = torch.from_numpy(image).float().unsqueeze(0)  # 添加通道维度
        
        if label is not None:
            label = torch.from_numpy(label).long()
            return image, label
        else:
            return image

# 创建数据加载器
def create_dataloader(image_dir, label_dir=None, batch_size=4, shuffle=True, preprocess=True, augment=False):
    """
    创建数据加载器
    
    参数:
        image_dir: 图像目录
        label_dir: 标签目录（可选）
        batch_size: 批次大小
        shuffle: 是否打乱数据
        preprocess: 是否进行预处理
        augment: 是否进行数据增强
    
    返回:
        PyTorch数据加载器
    """
    dataset = OCTDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        preprocess=preprocess,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    
    return dataloader
