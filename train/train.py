import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.unet import UNet
from utils.dataset import OCTDataset

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 计算Dice系数
def dice_coefficient(pred, target, epsilon=1e-6):
    # 将预测转换为one-hot表示
    pred = torch.argmax(pred, dim=1)
    
    # 计算每个类别的Dice系数
    dice_scores = []
    for class_idx in range(10):  # 10个类别
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() + epsilon
        dice = (2 * intersection + epsilon) / union
        dice_scores.append(dice)
    
    # 计算平均Dice系数
    return torch.tensor(dice_scores).mean()

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, save_dir='checkpoints'):
    # 创建保存检查点的目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 记录训练过程
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []
    
    best_val_dice = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算指标
            train_loss += loss.item()
            dice = dice_coefficient(outputs, masks)
            train_dice += dice.item()
            
            train_pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})
        
        # 计算平均损失和Dice系数
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # 计算指标
                val_loss += loss.item()
                dice = dice_coefficient(outputs, masks)
                val_dice += dice.item()
                
                val_pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})
        
        # 计算平均损失和Dice系数
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved best model with Dice score: {best_val_dice:.4f}')
        
        # 打印当前epoch的结果
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}')
        
        # 每个epoch保存一次模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
        }, os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    # 绘制训练过程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epoch')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='Train Dice')
    plt.plot(val_dices, label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.title('Dice vs. Epoch')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.show()
    
    return model

def main():
    # 设置随机种子
    set_seed()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 设置数据路径
    train_images_dir = 'data/RetinalOCT_Dataset/processed/train/images'
    train_masks_dir = 'data/RetinalOCT_Dataset/processed/train/masks'
    val_images_dir = 'data/RetinalOCT_Dataset/processed/val/images'
    val_masks_dir = 'data/RetinalOCT_Dataset/processed/val/masks'
    
    # 数据增强
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
    ])
    
    # 创建数据集和数据加载器
    train_dataset = OCTDataset(train_images_dir, train_masks_dir, transform=train_transform)
    val_dataset = OCTDataset(val_images_dir, val_masks_dir, transform=None)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # 创建模型
    model = UNet(in_channels=1, out_channels=10).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=30,
        save_dir='checkpoints'
    )
    
    print('Training completed!')

if __name__ == '__main__':
    main()
