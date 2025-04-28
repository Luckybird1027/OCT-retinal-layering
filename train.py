import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.dataset import OCTDataset
from model.unet import UNet
from utils.augmentation import advanced_augmentation
from utils.metrics import calculate_all_metrics, dice_coefficient as metrics_dice  # 重命名以区分
from utils.visualization import create_visualization_grid

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

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, num_classes=11, edge_weight=0.5, focal_gamma=2.0,
                 focal_alpha=0.25):
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

        return total_loss, {"dice_loss": dice_loss,
                            "focal_loss": focal_loss,
                            "edge_loss": edge_loss}

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
            pred = preds[b]  # (H, W)

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
                boundary = torch.logical_xor(dilated, eroded).squeeze().float()

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


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30,
                save_dir='checkpoints', early_stopping_patience=10, num_classes=11):
    # 创建保存检查点的目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))

    # 记录训练过程 - 扩展指标列表
    history = {
        'epoch': [],
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'val_iou': [], 'val_mabld': [], 'val_hausdorff_95': [], 'val_thickness_correlation': []
        # Add more metrics here if implemented and desired
        # 'val_breakpoint_density': [], 'val_intersection_count': []
    }

    best_val_dice = 0.0
    epochs_no_improve = 0
    best_epoch = 0

    # 获取一个固定的验证批次用于可视化 (确保每次 epoch 可视化相同的样本)
    fixed_val_batch = None
    try:
        fixed_val_batch = next(iter(val_loader))
    except StopIteration:
        print("警告：验证数据加载器为空，无法获取用于可视化的固定批次。")
        fixed_val_batch = None  # 或者处理这种情况

    for epoch in range(num_epochs):
        history['epoch'].append(epoch + 1)
        # --- 训练阶段 ---
        model.train()
        train_loss_epoch = 0.0
        train_dice_epoch = 0.0
        train_dice_loss_epoch = 0.0
        train_focal_loss_epoch = 0.0
        train_edge_loss_epoch = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_components = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_indices = torch.argmax(outputs, dim=1)
                dice_train_batch = metrics_dice(pred_indices.cpu().numpy(), masks.cpu().numpy(),
                                                num_classes=num_classes, ignore_index=0)

            train_loss_epoch += loss.item()
            train_dice_epoch += dice_train_batch
            train_dice_loss_epoch += loss_components["dice_loss"].item()
            train_focal_loss_epoch += loss_components["focal_loss"].item()
            train_edge_loss_epoch += loss_components["edge_loss"].item()

            train_pbar.set_postfix({'loss': loss.item(),
                                    'dice_loss': loss_components["dice_loss"].item(),
                                    'focal_loss': loss_components["focal_loss"].item(),
                                    'edge_loss': loss_components["edge_loss"].item(),
                                    'dice': dice_train_batch})  # 显示批次 dice

        # 计算平均训练损失和Dice系数
        avg_train_loss = train_loss_epoch / len(train_loader)
        avg_train_dice = train_dice_epoch / len(train_loader)
        avg_train_dice_loss = train_dice_loss_epoch / len(train_loader)
        avg_train_focal_loss = train_focal_loss_epoch / len(train_loader)
        avg_train_edge_loss = train_edge_loss_epoch / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['train_dice'].append(avg_train_dice)

        # --- 评估阶段 ---
        model.eval()
        val_loss_epoch = 0.0
        all_val_metrics = {
            'dice': [], 'iou': [], 'mabld': [], 'hausdorff_95': [], 'thickness_correlation': []
            # 'breakpoint_density': [], 'intersection_count': []
        }
        val_dice_loss_epoch = 0.0
        val_focal_loss_epoch = 0.0
        val_edge_loss_epoch = 0.0

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]')
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device)
                masks = masks.to(device)  # masks shape (B, H, W)

                outputs = model(images)  # outputs shape (B, C, H, W)
                loss, loss_components = criterion(outputs, masks)
                pred_indices = torch.argmax(outputs, dim=1)  # pred_indices shape (B, H, W)

                val_loss_epoch += loss.item()
                val_dice_loss_epoch += loss_components["dice_loss"].item()
                val_focal_loss_epoch += loss_components["focal_loss"].item()
                val_edge_loss_epoch += loss_components["edge_loss"].item()

                # --- 计算所有指标 ---
                # calculate_all_metrics 需要 numpy 输入 (B, H, W) or (H, W)
                pred_np = pred_indices.cpu().numpy()
                mask_np = masks.cpu().numpy()

                # 因为 calculate_all_metrics 可以处理批次，直接传入
                batch_metrics = calculate_all_metrics(pred_np, mask_np, num_classes=num_classes, ignore_index=0)

                # 收集批次指标 (加权)
                for key, value in batch_metrics.items():
                    if key in all_val_metrics and not np.isnan(value):
                        # 乘以实际有效的样本数（如果批次不完整）
                        all_val_metrics[key].append(value * images.size(0))

                val_pbar.set_postfix({'loss': loss.item(), 'dice': batch_metrics.get('dice', 0.0)})

        # 计算整个验证集的平均指标
        num_val_samples = len(val_loader.dataset)
        avg_val_metrics = {}
        for key, values in all_val_metrics.items():
            total_value = sum(v for v in values if not np.isnan(v))
            # 使用实际样本数进行平均
            avg_val_metrics[key] = total_value / num_val_samples if num_val_samples > 0 else 0.0

        avg_val_loss = val_loss_epoch / len(val_loader)
        avg_val_dice_loss = val_dice_loss_epoch / len(val_loader)
        avg_val_focal_loss = val_focal_loss_epoch / len(val_loader)
        avg_val_edge_loss = val_edge_loss_epoch / len(val_loader)

        # 存储平均验证指标
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_metrics.get('dice', 0.0))
        history['val_iou'].append(avg_val_metrics.get('iou', 0.0))
        history['val_mabld'].append(avg_val_metrics.get('mabld', np.nan))
        history['val_hausdorff_95'].append(avg_val_metrics.get('hausdorff_95', np.nan))
        history['val_thickness_correlation'].append(avg_val_metrics.get('thickness_correlation', np.nan))
        # history['val_breakpoint_density'].append(avg_val_metrics.get('breakpoint_density', 0.0))
        # history['val_intersection_count'].append(avg_val_metrics.get('intersection_count', 0.0))

        # --- TensorBoard 记录 ---
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Dice/train', avg_train_dice, epoch)
        writer.add_scalar('Dice/val', history['val_dice'][-1], epoch)
        writer.add_scalar('IoU/val', history['val_iou'][-1], epoch)
        # 只有当值不是 NaN 时才记录边界/厚度指标
        if not np.isnan(history['val_mabld'][-1]):
            writer.add_scalar('Boundary/MABLD_val', history['val_mabld'][-1], epoch)
        if not np.isnan(history['val_hausdorff_95'][-1]):
            writer.add_scalar('Boundary/Hausdorff95_val', history['val_hausdorff_95'][-1], epoch)
        if not np.isnan(history['val_thickness_correlation'][-1]):
            writer.add_scalar('Thickness/Correlation_val', history['val_thickness_correlation'][-1], epoch)

        # 记录学习率和损失分量
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss_Components/train_dice', avg_train_dice_loss, epoch)
        writer.add_scalar('Loss_Components/train_focal', avg_train_focal_loss, epoch)
        writer.add_scalar('Loss_Components/train_edge', avg_train_edge_loss, epoch)
        writer.add_scalar('Loss_Components/val_dice', avg_val_dice_loss, epoch)
        writer.add_scalar('Loss_Components/val_focal', avg_val_focal_loss, epoch)
        writer.add_scalar('Loss_Components/val_edge', avg_val_edge_loss, epoch)

        # --- TensorBoard 图像记录 (使用新的可视化网格) ---
        if fixed_val_batch is not None:
            images_vis, masks_vis = fixed_val_batch
            images_vis = images_vis.to(device)
            # masks_vis 保持在 CPU 上，因为 create_visualization_grid 需要 numpy
            # masks_vis = masks_vis.to(device) # 不需要移动到 device

            with torch.no_grad():
                outputs_vis = model(images_vis)
                predictions_vis = torch.argmax(outputs_vis, dim=1)

            # 创建可视化网格 (确保输入是正确的类型和设备)
            # image_tensor: (B, 1, H, W) on CPU, range [0, 1]
            # mask_tensor: (B, H, W) on CPU, indices
            # pred_tensor: (B, H, W) on CPU, indices
            visualization_grid = create_visualization_grid(
                image_tensor=images_vis.cpu(),  # 确保在 CPU 上
                mask_tensor=masks_vis.cpu(),  # 确保在 CPU 上
                pred_tensor=predictions_vis.cpu(),  # 确保在 CPU 上
                num_classes=num_classes,
                max_images=4  # 最多显示 4 张图的对比
            )
            writer.add_image('Validation_Samples/Visualization', visualization_grid, epoch)

        # --- 打印当前 epoch 结果 (包含新指标) ---
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'  训练 - Loss: {avg_train_loss:.4f}, Dice: {avg_train_dice:.4f}')
        print(
            f'         DiceLoss: {avg_train_dice_loss:.4f}, FocalLoss: {avg_train_focal_loss:.4f}, EdgeLoss: {avg_train_edge_loss:.4f}')
        # 打印时格式化 NaN
        mabld_str = f"{history['val_mabld'][-1]:.4f}" if not np.isnan(history['val_mabld'][-1]) else "N/A"
        hd95_str = f"{history['val_hausdorff_95'][-1]:.4f}" if not np.isnan(history['val_hausdorff_95'][-1]) else "N/A"
        thick_corr_str = f"{history['val_thickness_correlation'][-1]:.4f}" if not np.isnan(
            history['val_thickness_correlation'][-1]) else "N/A"
        print(
            f'  验证 - Loss: {avg_val_loss:.4f}, Dice: {history["val_dice"][-1]:.4f}, IoU: {history["val_iou"][-1]:.4f}')
        print(f'         MABLD: {mabld_str}, HD95: {hd95_str}, ThickCorr: {thick_corr_str}')
        print(
            f'         DiceLoss: {avg_val_dice_loss:.4f}, FocalLoss: {avg_val_focal_loss:.4f}, EdgeLoss: {avg_val_edge_loss:.4f}')

        # --- 更新学习率调度器 (基于验证 Dice) ---
        current_val_dice = history['val_dice'][-1]
        scheduler.step(current_val_dice)

        # --- 早停逻辑和保存最佳模型 (基于验证 Dice) ---
        if current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'---> 保存最佳模型 (Epoch {best_epoch}), 验证 Dice 分数: {best_val_dice:.4f}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f'---> 验证 Dice 未提升，连续 {epochs_no_improve} 个 epochs.')
            if epochs_no_improve >= early_stopping_patience:
                print(
                    f'---> 早停触发！连续 {early_stopping_patience} 个 epochs 验证 Dice 未提升。最佳模型在 Epoch {best_epoch}，Dice 为 {best_val_dice:.4f}')
                break

        # 每个epoch保存一次模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_dice': avg_train_dice,
            'val_dice': history['val_dice'][-1],
        }, os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

    # 关闭TensorBoard的SummaryWriter
    writer.close()

    # 保存最终模型 (可能是早停前的模型，也可能是训练完所有 epoch 的模型)
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f"最终模型已保存。最佳模型保存在 'best_model.pth' (Epoch {best_epoch}, Val Dice: {best_val_dice:.4f})")

    # --- 绘制训练过程 (包含新指标) ---
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False)  # 保存历史记录

    # 绘制 Loss 和 Dice
    plt.figure(figsize=(18, 12), dpi=150)  # 增加画布大小

    plt.subplot(2, 3, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='训练损失')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('损失 vs. Epoch')
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.plot(history_df['epoch'], history_df['train_dice'], label='训练 Dice')
    plt.plot(history_df['epoch'], history_df['val_dice'], label='验证 Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice 系数')
    plt.legend()
    plt.title('Dice vs. Epoch')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.plot(history_df['epoch'], history_df['val_iou'], label='验证 IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('验证 IoU vs. Epoch')
    plt.grid(True)

    plt.subplot(2, 3, 4)
    plt.plot(history_df['epoch'], history_df['val_mabld'], label='验证 MABLD')
    plt.xlabel('Epoch')
    plt.ylabel('MABLD (pixels)')
    plt.legend()
    plt.title('验证 MABLD vs. Epoch')
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(history_df['epoch'], history_df['val_hausdorff_95'], label='验证 HD95')
    plt.xlabel('Epoch')
    plt.ylabel('Hausdorff 95% (pixels)')
    plt.legend()
    plt.title('验证 HD95 vs. Epoch')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(history_df['epoch'], history_df['val_thickness_correlation'], label='验证层厚度相关性')
    plt.xlabel('Epoch')
    plt.ylabel('Pearson Correlation')
    plt.legend()
    plt.title('验证层厚度相关性 vs. Epoch')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history_metrics.png'))
    print(f"训练历史图表已保存到 {os.path.join(save_dir, 'training_history_metrics.png')}")
    # plt.show() # 取消注释以显示图表
    plt.close()  # 关闭图形，防止在循环或脚本中累积

    return model


def main():
    """
    主函数，用于训练UNet模型

    命令行参数:
    --train_images: 训练图像目录
    --train_masks: 训练掩码目录
    --val_images: 验证图像目录
    --val_masks: 验证掩码目录
    --batch_size: 批次大小
    --num_workers: 数据加载器线程数
    --in_channels: 输入通道数
    --out_channels: 输出通道数
    --epochs: 训练轮数
    --lr: 学习率
    --aug_prob: 数据增强应用概率
    --save_dir: 模型保存目录
    --seed: 随机种子
    --alpha: Dice损失权重
    --beta: Focal损失权重
    --gamma: 边界损失权重
    --num_classes: 输出通道数/类别数

    使用示例:
    python train.py --batch_size 1 --num_workers 4 --epochs 20 --lr 0.001 --num_classes 11
    """

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='OCT图像分割训练')

    # 数据相关参数
    parser.add_argument('--train_images', type=str, default='data/SJTU/train/img', help='训练图像目录')
    parser.add_argument('--train_masks', type=str, default='data/SJTU/train/mask', help='训练掩码目录')
    parser.add_argument('--val_images', type=str, default='data/SJTU/eval/img', help='验证图像目录')
    parser.add_argument('--val_masks', type=str, default='data/SJTU/eval/mask', help='验证掩码目录')
    parser.add_argument('--batch_size', type=int, default=1, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器线程数')

    # 模型相关参数
    parser.add_argument('--in_channels', type=int, default=1, help='输入通道数')
    parser.add_argument('--out_channels', type=int, default=11, help='输出通道数')

    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.0003, help='学习率')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='数据增强应用概率')
    parser.add_argument('--save_dir', type=str, default='train/checkpoints', help='模型保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 损失函数相关参数
    parser.add_argument('--alpha', type=float, default=0.2, help='Dice损失权重')
    parser.add_argument('--beta', type=float, default=0.8, help='Focal损失权重')
    parser.add_argument('--gamma', type=float, default=0.05, help='边界损失权重')

    # 新增参数
    parser.add_argument('--num_classes', type=int, default=11, help='输出通道数/类别数')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 设置数据路径
    train_images_dir = args.train_images
    train_masks_dir = args.train_masks
    val_images_dir = args.val_images
    val_masks_dir = args.val_masks

    # 定义自定义数据增强
    train_transform = CustomTransform(apply_prob=args.aug_prob)

    # 创建数据集和数据加载器
    train_dataset = OCTDataset(train_images_dir, train_masks_dir, transform=train_transform)
    val_dataset = OCTDataset(val_images_dir, val_masks_dir, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 创建模型
    model = UNet(in_channels=args.in_channels, out_channels=args.num_classes).to(device)

    # 定义复合损失函数和优化器
    criterion = CompoundLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, num_classes=args.num_classes).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 添加学习率调度器 (例如: 当验证 Dice 停止提升时降低学习率)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # 打印训练配置
    print(f"\n{'=' * 20} 训练配置 {'=' * 20}")
    print(f"批次大小: {args.batch_size}")
    print(f"初始学习率: {args.lr}")
    print(f"权重衰减 (L2正则化): {optimizer.param_groups[0]['weight_decay']}")
    print(f"训练轮数: {args.epochs}")
    print(f"损失函数权重: Dice({args.alpha}) + Focal({args.beta}) + Edge({args.gamma})")
    print(f"数据增强概率: {args.aug_prob}")
    print(f"使用学习率调度器: ReduceLROnPlateau (patience={scheduler.patience})")
    print(f"{'=' * 50}\n")

    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        early_stopping_patience=15,
        num_classes=args.num_classes
    )

    print('训练完成!')


if __name__ == '__main__':
    main()
