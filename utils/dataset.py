import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.advanced_preprocess import advanced_denoising, advanced_enhancement

class OCTDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=(984, 760), transform=None, preprocess=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.transform = transform
        self.preprocess = preprocess
        
        self.img_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) 
                                if f.endswith('.jpg') or f.endswith('.png')])
        self.mask_files = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir) 
                                 if f.endswith('.jpg') or f.endswith('.png')])
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # 读取图像和标签
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]
        
        # 使用OpenCV读取图像（灰度模式）
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 确保图像尺寸一致
        img = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # 标准化图像
        img = img / 255.0
        
        # 应用预处理
        if self.preprocess:
            img = advanced_denoising(img)
            img = advanced_enhancement(img)
        
        # 应用数据增强
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # 将图像转换为PyTorch张量
        img = torch.from_numpy(img).float().unsqueeze(0)  # [1, H, W]
        mask = torch.from_numpy(mask).long()  # [H, W]
        
        return img, mask

# 创建数据加载器
def create_dataloader(images_dir, masks_dir, img_size=(984, 760), batch_size=4, shuffle=True, preprocess=True):
    """
    创建数据加载器
    
    参数:
        images_dir: 图像目录
        masks_dir: 标签目录
        img_size: 图像尺寸
        batch_size: 批次大小
        shuffle: 是否打乱数据
        preprocess: 是否进行预处理
    
    返回:
        PyTorch数据加载器
    """
    dataset = OCTDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        img_size=img_size,
        preprocess=preprocess
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    
    return dataloader
