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
                 focal_alpha=0.25, ignore_index=0):
        super(CompoundLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.edge_weight = edge_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # 计算Dice损失
        dice_loss = self.dice_loss(inputs, targets)

        # 计算Focal损失
        focal_loss = self.focal_loss(inputs, targets)

        # 计算边界敏感损失
        edge_loss = self.edge_loss(inputs, targets)

        # 组合损失
        total_loss = self.alpha * dice_loss + self.beta * focal_loss + self.gamma * edge_loss

        # 返回总损失和各分量损失
        return total_loss, {"dice_loss": dice_loss,
                            "focal_loss": focal_loss,
                            "edge_loss": edge_loss}

    def dice_loss(self, inputs, targets, epsilon=1e-6):
        """
        计算多类别Dice损失 (忽略背景类)
        """
        # 转换预测为概率分布
        probs = F.softmax(inputs, dim=1)

        # 检查 targets 是否为空，避免在空 batch 上计算
        if targets.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        dice_score = 0.0
        num_valid_classes = 0 # 计数器，用于统计参与计算的有效类别数

        for class_idx in range(self.num_classes):
            # 跳过要忽略的类别 (背景类)
            if class_idx == self.ignore_index:
                continue

            # 获取当前类别的概率
            prob_class = probs[:, class_idx]

            # 获取当前类别的目标（one-hot）
            target_class = (targets == class_idx).float()

            # 重要：仅对在 target 中实际存在的类别计算 Dice，避免影响平均值
            # 检查当前类别在整个批次中是否存在
            # 如果一个类别在整个批次的目标掩码中都不存在，我们跳过它
            # 否则，即使某些样本没有这个类别，我们仍然计算并平均
            if target_class.sum() == 0:
                 # print(f"警告: 类别 {class_idx} 在当前批次的目标中不存在，跳过 Dice Loss 计算。")
                 continue # 跳过这个类别

            num_valid_classes += 1 # 这是一个有效的，需要计算的类别

            # 计算交集
            intersection = (prob_class * target_class).sum(dim=(1, 2))

            # 计算并集
            cardinality = prob_class.sum(dim=(1, 2)) + target_class.sum(dim=(1, 2)) # 注意：这里的分母没有 epsilon

            # 计算Dice系数 - 对批次中的每个样本计算
            dice = (2. * intersection + epsilon) / (cardinality + epsilon) # 在最终计算时加 epsilon

            # 累加每个类别的平均Dice系数 (在批次维度上求平均)
            dice_score += dice.mean() # 对批次内样本求平均

        # 如果没有任何有效类别（例如，mask 全是背景或只包含 ignore_index），则损失为0
        if num_valid_classes == 0:
             # print(f"警告: 在当前批次中未找到任何有效的非忽略类别，Dice Loss 设为 0。")
             return torch.tensor(0.0, device=inputs.device, requires_grad=True)


        # 计算平均 Dice 系数（仅基于有效类别）并转换为损失
        avg_dice = dice_score / num_valid_classes
        return 1.0 - avg_dice

    def focal_loss(self, inputs, targets):
        """
        计算多类别Focal损失
        """
        # 检查 targets 是否为空
        if targets.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        # 将目标转换为one-hot编码
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # 计算softmax概率
        probs = F.softmax(inputs, dim=1)

        # 应用focal loss公式: -alpha * (1-pt)^gamma * log(pt)
        # 注意：Focal Loss 本身不直接处理 ignore_index，它通过 one-hot 编码和 pt 的计算隐式处理。
        # 如果某个像素的 target 是 ignore_index，它在 one-hot 中不会有对应的 1，
        # pt 的计算会涉及所有非 ignore 类的概率。
        # 但标准的 Focal Loss 通常不区分背景前景，如果需要，可以修改 pt 的计算或加权。
        # 目前保持原样，因为它惩罚的是所有类别的错误分类。
        # 如果需要，可以修改 pt 的计算或加权。
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
        L_Edge = sum_{c=1}^N_classes (1/N_c) * sum_{p∈E_c} ||y_hat_p - y_p|| * exp(-d_p)
        """
        batch_size = targets.size(0)
        # 检查 targets 是否为空
        if batch_size == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)

        edge_loss = 0.0
        preds = torch.argmax(inputs, dim=1) # 获取预测类别 (B, H, W)

        for b in range(batch_size):
            target_b = targets[b] # (H, W)
            pred_b = preds[b] # (H, W)

            edge_loss_sample = 0.0
            num_valid_classes = 0 # 统计存在于目标中的有效类别数

            # 循环时也跳过 ignore_index
            for c in range(self.num_classes): # 遍历所有类别
                if c == self.ignore_index: # 跳过背景/忽略类别
                    continue

                target_mask_c = (target_b == c).float()

                # 如果当前类别在目标中不存在，跳过
                if target_mask_c.sum() == 0:
                    continue

                num_valid_classes += 1 # 这是一个有效类别

                # --- 计算精确边界 ---
                # 使用 F.conv2d 实现膨胀和腐蚀来找边界
                target_mask_float = target_mask_c.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
                kernel = torch.ones(1, 1, 3, 3, device=target_b.device)
                # 膨胀
                dilated = F.conv2d(target_mask_float, kernel, padding=1) > 0
                # 腐蚀
                eroded = F.conv2d(target_mask_float, kernel, padding=1) == kernel.numel()
                # 边界
                boundary = torch.logical_xor(dilated, eroded).squeeze().float() # (H, W)

                # --- 计算距离变换 ---
                dist_transform = distance_transform_edt(1 - target_mask_c.cpu().numpy())
                dist_transform = torch.from_numpy(dist_transform).to(target_b.device)

                # 获取边界区域的像素索引
                edge_pixels = boundary > 0

                num_edge_pixels = edge_pixels.sum()
                if num_edge_pixels == 0:
                    continue # 如果没有边界像素，跳过该类别

                # --- 计算边界损失 ---
                pred_mask_c = (pred_b == c).float() # (H, W)
                pixel_error = torch.abs(pred_mask_c[edge_pixels] - target_mask_c[edge_pixels])
                distance_weight = torch.exp(-dist_transform[edge_pixels])
                weighted_error = pixel_error * distance_weight
                class_edge_loss = weighted_error.sum() / num_edge_pixels
                edge_loss_sample += class_edge_loss

            if num_valid_classes > 0:
                edge_loss += edge_loss_sample / num_valid_classes # 对有效类别数取平均
            # else:
            #     print(f"警告: 样本 {b} 中没有找到任何有效的非忽略类别")


        # 返回批次的平均边界损失
        return edge_loss / batch_size


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30,
                save_dir='checkpoints', early_stopping_patience=10, num_classes=11, accumulation_steps=1): # 添加 accumulation_steps 参数
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

    print(f"启用梯度累积，步数: {accumulation_steps}") # 提示用户梯度累积已启用

    for epoch in range(num_epochs):
        history['epoch'].append(epoch + 1)
        # --- 训练阶段 ---
        model.train()
        train_loss_epoch = 0.0 # 累加原始损失值
        train_dice_epoch = 0.0
        train_dice_loss_epoch = 0.0
        train_focal_loss_epoch = 0.0
        train_edge_loss_epoch = 0.0

        optimizer.zero_grad() # 梯度清零放在 epoch 开始之前，适配梯度累积

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
        for i, (images, masks) in enumerate(train_pbar):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            raw_loss, loss_components = criterion(outputs, masks) # 获取原始损失

            # 标准化损失以便进行梯度累积
            loss = raw_loss / accumulation_steps
            loss.backward() # 计算并累积梯度

            # 每 accumulation_steps 步执行一次优化器更新
            if (i + 1) % accumulation_steps == 0:
                optimizer.step() # 更新模型权重
                optimizer.zero_grad() # 清空梯度，为下一个累积周期准备

            with torch.no_grad():
                pred_indices = torch.argmax(outputs, dim=1)
                # 计算批次 Dice (基于当前 mini-batch 的预测和标签)
                dice_train_batch = metrics_dice(pred_indices.cpu().numpy(), masks.cpu().numpy(),
                                                num_classes=num_classes, ignore_index=0)

            # 累加原始损失和各分量损失用于日志记录
            train_loss_epoch += raw_loss.item()
            train_dice_epoch += dice_train_batch # 累加每个 mini-batch 的 Dice
            # 损失分量也累加原始值
            train_dice_loss_epoch += loss_components["dice_loss"].item()
            train_focal_loss_epoch += loss_components["focal_loss"].item()
            train_edge_loss_epoch += loss_components["edge_loss"].item()

            # 更新进度条，显示原始 mini-batch 损失
            train_pbar.set_postfix({'loss': raw_loss.item(),
                                    'dice_loss': loss_components["dice_loss"].item(),
                                    'focal_loss': loss_components["focal_loss"].item(),
                                    'edge_loss': loss_components["edge_loss"].item(),
                                    'dice': dice_train_batch})

        # --- 在 epoch 结束时，处理可能未更新的梯度（如果总迭代次数不是 accumulation_steps 的倍数）---
        # if (len(train_loader) % accumulation_steps) != 0:
        #     print("执行 epoch 结束前的最后一次梯度更新")
        #     optimizer.step()
        #     optimizer.zero_grad()
        # 注意：此步骤通常可以省略，因为下一个epoch开始时会清零梯度。除非特别关心最后几个样本的梯度。

        # 计算平均训练损失和Dice系数 (除以迭代次数 len(train_loader))
        num_train_batches = len(train_loader)
        avg_train_loss = train_loss_epoch / num_train_batches
        avg_train_dice = train_dice_epoch / num_train_batches
        avg_train_dice_loss = train_dice_loss_epoch / num_train_batches
        avg_train_focal_loss = train_focal_loss_epoch / num_train_batches
        avg_train_edge_loss = train_edge_loss_epoch / num_train_batches
        history['train_loss'].append(avg_train_loss)
        history['train_dice'].append(avg_train_dice)

        # --- 评估阶段 (评估逻辑不变) ---
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
                loss, loss_components = criterion(outputs, masks) # 验证时不需要梯度累积，直接用原始损失
                pred_indices = torch.argmax(outputs, dim=1)  # pred_indices shape (B, H, W)

                val_loss_epoch += loss.item()
                val_dice_loss_epoch += loss_components["dice_loss"].item()
                val_focal_loss_epoch += loss_components["focal_loss"].item()
                val_edge_loss_epoch += loss_components["edge_loss"].item()

                # --- 计算所有指标 ---
                pred_np = pred_indices.cpu().numpy()
                mask_np = masks.cpu().numpy()
                batch_metrics = calculate_all_metrics(pred_np, mask_np, num_classes=num_classes, ignore_index=0)

                # 收集批次指标 (加权)
                current_batch_size = images.size(0) # 处理最后一个不完整批次
                for key, value in batch_metrics.items():
                    if key in all_val_metrics and not np.isnan(value):
                        all_val_metrics[key].append(value * current_batch_size) # 用实际样本数加权

                val_pbar.set_postfix({'loss': loss.item(), 'dice': batch_metrics.get('dice', 0.0)})

        # 计算整个验证集的平均指标
        num_val_samples = len(val_loader.dataset)
        avg_val_metrics = {}
        if num_val_samples > 0: # 避免除以零
            for key, values in all_val_metrics.items():
                total_value = sum(v for v in values if not np.isnan(v))
                avg_val_metrics[key] = total_value / num_val_samples
        else:
             for key in all_val_metrics.keys():
                 avg_val_metrics[key] = 0.0 # 或 np.nan

        # 计算平均验证损失
        num_val_batches = len(val_loader)
        avg_val_loss = val_loss_epoch / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_dice_loss = val_dice_loss_epoch / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_focal_loss = val_focal_loss_epoch / num_val_batches if num_val_batches > 0 else 0.0
        avg_val_edge_loss = val_edge_loss_epoch / num_val_batches if num_val_batches > 0 else 0.0


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

            with torch.no_grad():
                outputs_vis = model(images_vis)
                predictions_vis = torch.argmax(outputs_vis, dim=1)

            visualization_grid = create_visualization_grid(
                image_tensor=images_vis.cpu(),
                mask_tensor=masks_vis.cpu(),
                pred_tensor=predictions_vis.cpu(),
                num_classes=num_classes,
                max_images=min(4, images_vis.size(0)) # 确保 max_images 不超过实际批次大小
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
        scheduler.step(current_val_dice) # ReduceLROnPlateau 需要传入监控的指标

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

        # 每个epoch保存一次模型状态，包括优化器和调度器状态，方便中断后恢复
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), # 保存调度器状态
            'history': history, # 保存训练历史记录
            'best_val_dice': best_val_dice,
            'epochs_no_improve': epochs_no_improve,
            'best_epoch': best_epoch
        }, os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')) # 使用 checkpoint 而不是 model


    # 关闭TensorBoard的SummaryWriter
    writer.close()

    # 保存最终模型 (如果早停，这是触发早停前的模型；否则是最后一个 epoch 的模型)
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    print(f"最终模型已保存。最佳模型保存在 'best_model.pth' (Epoch {best_epoch}, Val Dice: {best_val_dice:.4f})")

    # --- 绘制训练过程 (包含新指标) ---
    history_df = pd.DataFrame(history)
    # 检查 history_df 是否为空
    if not history_df.empty:
        history_df.to_csv(os.path.join(save_dir, 'training_history.csv'), index=False) # 保存历史记录

        # 绘制 Loss 和 Dice 等指标
        plt.figure(figsize=(18, 12), dpi=150)

        plt.subplot(2, 3, 1)
        plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
        plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs. Epoch')
        plt.grid(True)

        plt.subplot(2, 3, 2)
        plt.plot(history_df['epoch'], history_df['train_dice'], label='Train Dice')
        plt.plot(history_df['epoch'], history_df['val_dice'], label='Val Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice')
        plt.ylim(0, 1) # 设置 Dice 范围为 0 到 1
        plt.legend()
        plt.title('Dice vs. Epoch')
        plt.grid(True)

        plt.subplot(2, 3, 3)
        plt.plot(history_df['epoch'], history_df['val_iou'], label='Val IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.ylim(0, 1) # 设置 IoU 范围为 0 到 1
        plt.legend()
        plt.title('Val IoU vs. Epoch')
        plt.grid(True)

        # 绘制可能包含 NaN 的指标时进行检查
        if 'val_mabld' in history_df.columns and not history_df['val_mabld'].isnull().all():
            plt.subplot(2, 3, 4)
            plt.plot(history_df['epoch'], history_df['val_mabld'], label='Val MABLD')
            plt.xlabel('Epoch')
            plt.ylabel('MABLD (pixels)')
            plt.legend()
            plt.title('Val MABLD vs. Epoch')
            plt.grid(True)
        else:
             print("警告: 未绘制 MABLD 图表，因为数据为空或全为 NaN。")


        if 'val_hausdorff_95' in history_df.columns and not history_df['val_hausdorff_95'].isnull().all():
            plt.subplot(2, 3, 5)
            plt.plot(history_df['epoch'], history_df['val_hausdorff_95'], label='Val HD95')
            plt.xlabel('Epoch')
            plt.ylabel('Hausdorff 95% (pixels)')
            plt.legend()
            plt.title('Val HD95 vs. Epoch')
            plt.grid(True)
        else:
             print("警告: 未绘制 HD95 图表，因为数据为空或全为 NaN。")


        if 'val_thickness_correlation' in history_df.columns and not history_df['val_thickness_correlation'].isnull().all():
            plt.subplot(2, 3, 6)
            plt.plot(history_df['epoch'], history_df['val_thickness_correlation'], label='Val Thickness Correlation')
            plt.xlabel('Epoch')
            plt.ylabel('Pearson Correlation')
            plt.legend()
            plt.title('Val Thickness Correlation vs. Epoch')
            plt.grid(True)
        else:
             print("警告: 未绘制层厚度相关性图表，因为数据为空或全为 NaN。")


        plt.tight_layout()
        save_path = os.path.join(save_dir, 'training_history_metrics.png')
        try:
            plt.savefig(save_path)
            print(f"训练历史图表已保存到 {save_path}")
        except Exception as e:
            print(f"保存图表失败: {e}")

        # plt.show() # 取消注释以显示图表
        plt.close()  # 关闭图形，防止在循环或脚本中累积
    else:
        print("警告: 训练历史为空，无法生成图表。")


    return model


def main():
    """
    主函数，用于训练UNet模型
    """

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='OCT图像分割训练')

    # 数据相关参数
    parser.add_argument('--train_images', type=str, default='data/SJTU/train/img', help='训练图像目录')
    parser.add_argument('--train_masks', type=str, default='data/SJTU/train/mask', help='训练掩码目录')
    parser.add_argument('--val_images', type=str, default='data/SJTU/eval/img', help='验证图像目录')
    parser.add_argument('--val_masks', type=str, default='data/SJTU/eval/mask', help='验证掩码目录')
    parser.add_argument('--img_height', type=int, default=512, help='调整后的图像高度')
    parser.add_argument('--img_width', type=int, default=704, help='调整后的图像宽度')
    parser.add_argument('--batch_size', type=int, default=2, help='每个设备上的批次大小 (用于梯度累积)')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器线程数')

    # 模型相关参数
    parser.add_argument('--in_channels', type=int, default=1, help='输入通道数')
    parser.add_argument('--growth_rate', type=int, default=64, help='DenseBottleneck 的增长率')

    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--aug_prob', type=float, default=0.5, help='数据增强应用概率')
    parser.add_argument('--save_dir', type=str, default='train/checkpoints_512x704_gr64', help='模型和日志保存目录 (建议区分)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--early_stopping_patience', type=int, default=20, help='早停耐心值 (epochs)')

    # 损失函数相关参数
    parser.add_argument('--alpha', type=float, default=0.2, help='Dice损失权重')
    parser.add_argument('--beta', type=float, default=0.8, help='Focal损失权重')
    parser.add_argument('--gamma', type=float, default=0.05, help='边界损失权重')

    # 新增/修改参数
    parser.add_argument('--num_classes', type=int, default=11, help='输出通道数/类别数')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='梯度累积步数 (模拟更大的批次)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='优化器权重衰减 (L2 正则化)')

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
    img_size = (args.img_height, args.img_width)

    # 定义自定义数据增强
    train_transform = CustomTransform(apply_prob=args.aug_prob)

    # 创建数据集和数据加载器
    train_dataset = OCTDataset(train_images_dir, train_masks_dir, img_size=img_size, transform=train_transform)
    val_dataset = OCTDataset(val_images_dir, val_masks_dir, img_size=img_size, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 创建模型
    model = UNet(in_channels=args.in_channels, out_channels=args.num_classes, growth_rate=args.growth_rate).to(device)

    # 定义复合损失函数和优化器
    criterion = CompoundLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, num_classes=args.num_classes).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # 计算有效批次大小
    effective_batch_size = args.batch_size * args.accumulation_steps

    # 打印训练配置
    print(f"\n{'=' * 20} 训练配置 {'=' * 20}")
    print(f"设备批次大小: {args.batch_size}")
    print(f"梯度累积步数: {args.accumulation_steps}")
    print(f"有效批次大小: {effective_batch_size}")
    print(f"输入图像尺寸: {img_size}")
    print(f"DenseBottleneck增长率: {args.growth_rate}")
    print(f"初始学习率: {args.lr}")
    print(f"权重衰减 (L2正则化): {args.weight_decay}")
    print(f"训练轮数: {args.epochs}")
    print(f"损失函数权重: Dice({args.alpha}) + Focal({args.beta}) + Edge({args.gamma})")
    print(f"数据增强概率: {args.aug_prob}")
    print(f"使用学习率调度器: ReduceLROnPlateau (patience={scheduler.patience})")
    print(f"早停耐心: {args.early_stopping_patience}")
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
        early_stopping_patience=args.early_stopping_patience,
        num_classes=args.num_classes,
        accumulation_steps=args.accumulation_steps
    )

    print('训练完成!')


if __name__ == '__main__':
    main()
