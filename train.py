import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

from model.unet import UNet
from model.dataset import OCTDataset
from utils.augmentation import advanced_augmentation

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 计算Dice系数
def dice_coefficient(predict, target, epsilon=1e-6):
    # 将预测转换为one-hot表示
    predict = torch.argmax(predict, dim=1)

    # 计算每个类别的Dice系数
    dice_scores = []
    for class_idx in range(11):  # 11个类别
        predict_class = (predict == class_idx).float()
        target_class = (target == class_idx).float()

        intersection = (predict_class * target_class).sum()
        union = predict_class.sum() + target_class.sum() + epsilon
        dice = (2 * intersection + epsilon) / union
        dice_scores.append(dice)

    # 计算平均Dice系数
    return torch.tensor(dice_scores).mean()


# 自定义数据增强转换
class CustomTransform:
    def __init__(self, apply_prob=0.5):
        self.apply_prob = apply_prob

    def __call__(self, image, mask=None):
        if random.random() < self.apply_prob:
            image, mask = advanced_augmentation(image, mask)
        return {'image': image, 'mask': mask}


class CompoundLoss(nn.Module):
    """
    复合损失函数:
    L_total = α * L_Dice + β * L_Focal + γ * L_Edge
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, num_classes=11, edge_weight=0.5, focal_gamma=2.0, focal_alpha=0.25):
        super(CompoundLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
    def forward(self, inputs, targets):
        # 计算Dice损失
        dice_loss = self.dice_loss(inputs, targets)
        
        # 计算Focal损失
        focal_loss = self.focal_loss(inputs, targets)
        
        # 计算边界敏感损失
        edge_loss = self.edge_loss(inputs, targets)
        
        # 组合损失
        total_loss = self.alpha * dice_loss + self.beta * focal_loss + self.gamma * edge_loss
        
        return total_loss, {"dice_loss": dice_loss.item(), 
                           "focal_loss": focal_loss.item(), 
                           "edge_loss": edge_loss.item()}
    
    def dice_loss(self, inputs, targets, epsilon=1e-6):
        """
        计算多类别Dice损失
        """
        # 转换预测为概率分布
        probs = F.softmax(inputs, dim=1)
        
        # 计算每个类别的Dice损失
        dice_score = 0.0
        for class_idx in range(self.num_classes):
            # 获取当前类别的概率
            prob_class = probs[:, class_idx]
            
            # 获取当前类别的目标（one-hot）
            target_class = (targets == class_idx).float()
            
            # 计算交集
            intersection = (prob_class * target_class).sum(dim=(1, 2))
            
            # 计算并集
            cardinality = prob_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2)) + epsilon
            
            # 计算Dice系数
            dice = (2. * intersection + epsilon) / cardinality
            
            # 累加每个类别的Dice系数
            dice_score += dice.mean()
            
        # 计算平均Dice系数并转换为损失
        avg_dice = dice_score / self.num_classes
        return 1.0 - avg_dice
    
    def focal_loss(self, inputs, targets):
        """
        计算多类别Focal损失
        """
        # 将目标转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # 计算softmax概率
        probs = F.softmax(inputs, dim=1)
        
        # 应用focal loss公式: -alpha * (1-pt)^gamma * log(pt)
        pt = torch.sum(targets_one_hot * probs, dim=1)  # 预测正确类别的概率
        
        # 避免log(0)
        pt = torch.clamp(pt, min=1e-7, max=1.0)
        
        # 计算focal loss
        focal_weight = (1 - pt) ** self.focal_gamma
        focal = -self.focal_alpha * focal_weight * torch.log(pt)
        
        # 返回平均focal loss
        return focal.mean()
    
    def edge_loss(self, inputs, targets):
        """
        计算边界敏感损失: 
        L_Edge = sum_{c=1}^9 (1/N_c) * sum_{p∈E_c} ||y_hat_p - y_p|| * exp(-d_p)
        """
        batch_size = targets.size(0)
        edge_loss = 0.0
        
        # 获取预测类别
        preds = torch.argmax(inputs, dim=1)
        
        for b in range(batch_size):
            target = targets[b]  # (H, W)
            pred = preds[b]      # (H, W)
            
            edge_loss_sample = 0.0
            
            # 逐个类别计算边界损失
            for c in range(1, self.num_classes):  # 从类别1开始（忽略背景类别0）
                # 生成当前类别的二值掩码
                target_mask = (target == c).float()
                
                # 如果当前类别在目标中不存在，跳过
                if target_mask.sum() == 0:
                    continue
                
                # 计算边界 - 使用形态学操作
                kernel = torch.ones(3, 3, device=target.device)
                dilated = F.conv2d(target_mask.unsqueeze(0).unsqueeze(0), 
                                  kernel.unsqueeze(0).unsqueeze(0), 
                                  padding=1) > 0
                eroded = F.conv2d(target_mask.unsqueeze(0).unsqueeze(0), 
                                 kernel.unsqueeze(0).unsqueeze(0), 
                                 padding=1) == 9
                
                # 边界是膨胀和腐蚀的差
                boundary = (dilated - eroded).squeeze().float()
                
                # 计算距离变换
                target_mask_np = target_mask.cpu().numpy()
                dist_transform = distance_transform_edt(1 - target_mask_np)
                dist_transform = torch.from_numpy(dist_transform).to(target.device)
                
                # 计算边界区域
                edge_region = boundary > 0
                
                # 如果边界区域为空，跳过
                num_edge_pixels = edge_region.sum()
                if num_edge_pixels == 0:
                    continue
                
                # 计算边界区域的预测误差
                pred_mask = (pred == c).float()
                edge_diff = torch.abs(pred_mask - target_mask)[edge_region]
                
                # 应用距离权重
                distance_weight = torch.exp(-dist_transform[edge_region])
                weighted_diff = edge_diff * distance_weight
                
                # 计算当前类别的边界损失
                class_edge_loss = weighted_diff.sum() / num_edge_pixels
                edge_loss_sample += class_edge_loss
            
            # 累加批次中的样本损失
            edge_loss += edge_loss_sample / (self.num_classes - 1)  # 平均到类别数
            
        # 计算批次的平均损失
        return edge_loss / batch_size


def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, save_dir='checkpoints'):
    # 创建保存检查点的目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))

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
        train_dice_loss = 0.0
        train_focal_loss = 0.0
        train_edge_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs = model(images)
            loss, loss_components = criterion(outputs, masks)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算指标
            train_loss += loss.item()
            dice = dice_coefficient(outputs, masks)
            train_dice += dice.item()
            
            # 记录各个损失分量
            train_dice_loss += loss_components["dice_loss"]
            train_focal_loss += loss_components["focal_loss"]
            train_edge_loss += loss_components["edge_loss"]

            train_pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})

        # 计算平均损失和Dice系数
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_dice_loss /= len(train_loader)
        train_focal_loss /= len(train_loader)
        train_edge_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)

        # 评估阶段
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_dice_loss = 0.0
        val_focal_loss = 0.0
        val_edge_loss = 0.0

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)

                # 前向传播
                outputs = model(images)
                loss, loss_components = criterion(outputs, masks)

                # 计算指标
                val_loss += loss.item()
                dice = dice_coefficient(outputs, masks)
                val_dice += dice.item()
                
                # 记录各个损失分量
                val_dice_loss += loss_components["dice_loss"]
                val_focal_loss += loss_components["focal_loss"]
                val_edge_loss += loss_components["edge_loss"]

                val_pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})

        # 计算平均损失和Dice系数
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_dice_loss /= len(val_loader)
        val_focal_loss /= len(val_loader)
        val_edge_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        
        # 记录各个损失分量
        writer.add_scalar('Loss_Components/train_dice', train_dice_loss, epoch)
        writer.add_scalar('Loss_Components/train_focal', train_focal_loss, epoch)
        writer.add_scalar('Loss_Components/train_edge', train_edge_loss, epoch)
        writer.add_scalar('Loss_Components/val_dice', val_dice_loss, epoch)
        writer.add_scalar('Loss_Components/val_focal', val_focal_loss, epoch)
        writer.add_scalar('Loss_Components/val_edge', val_edge_loss, epoch)

        # 每个 Epoch 记录一些图像到 TensorBoard
        # 获取一个批次的图像和预测结果
        images, masks = next(iter(val_loader))
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

        # 添加原始图像、真实标签和预测结果到 TensorBoard
        img_grid = torchvision.utils.make_grid(images[:4], normalize=True)
        writer.add_image('Images', img_grid, epoch)

        # 将类别索引转换回灰度值
        reverse_lookup = {
            0: 0,
            1: 26,
            2: 51,
            3: 77,
            4: 102,
            5: 128,
            6: 153,
            7: 179,
            8: 204,
            9: 230,
            10: 255
        }
        mask_gray = torch.zeros_like(masks, dtype=torch.uint8).to(device)
        pred_gray = torch.zeros_like(predictions, dtype=torch.uint8).to(device)

        for c in range(11):
            mask_gray[masks == c] = reverse_lookup[c]
            pred_gray[predictions == c] = reverse_lookup[c]

        # 添加灰度标签和预测图到 TensorBoard
        torchvision.utils.make_grid(mask_gray.unsqueeze(1))
        torchvision.utils.make_grid(pred_gray.unsqueeze(1))

        # 将原始图像与灰度标签和预测图并排显示
        combined_grid = torchvision.utils.make_grid(
            torch.cat([images[:4], mask_gray.unsqueeze(1).float() / 255.0, pred_gray.unsqueeze(1).float() / 255.0],
                      dim=0), nrow=4
        )
        writer.add_image('Combined/Image_Mask_Prediction_Gray', combined_grid, epoch)

        # 保存最佳模型
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'保存最佳模型，Dice分数: {best_val_dice:.4f}')

        # 打印当前epoch的结果
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}, 训练Dice: {train_dice:.4f}')
        print(f'  验证损失: {val_loss:.4f}, 验证Dice: {val_dice:.4f}')

        # 每个epoch保存一次模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
        }, os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

    # 关闭TensorBoard的SummaryWriter
    writer.close()

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))

    # 绘制训练过程
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='评估损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('损失 vs. Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='训练Dice')
    plt.plot(val_dices, label='评估Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice系数')
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
    print(f'使用设备: {device}')

    # 设置数据路径
    train_images_dir = 'data/SJTU/train/img'
    train_masks_dir = 'data/SJTU/train/mask'
    val_images_dir = 'data/SJTU/val/img'
    val_masks_dir = 'data/SJTU/val/mask'

    # 定义自定义数据增强
    train_transform = CustomTransform(apply_prob=0.5)

    # 创建数据集和数据加载器
    train_dataset = OCTDataset(train_images_dir, train_masks_dir, transform=train_transform)
    val_dataset = OCTDataset(val_images_dir, val_masks_dir, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 创建模型
    model = UNet(in_channels=1, out_channels=11).to(device)

    # 定义复合损失函数和优化器
    criterion = CompoundLoss(alpha=1.0, beta=0.5, gamma=0.7, num_classes=11).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=20,
        save_dir='train/checkpoints'
    )

    print('训练完成!')


if __name__ == '__main__':
    main()
