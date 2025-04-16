import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.unet import UNet
from model.dataset import OCTDataset
from utils.augmentation import advanced_augmentation


# 设置matplotlib支持中文
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False


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
                save_dir='checkpoints',
                local_rank=0, early_stopping_patience=10):
    # 创建保存检查点的目录
    if local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    # 创建TensorBoard的SummaryWriter
    if local_rank == 0:
        writer = SummaryWriter(os.path.join(save_dir, 'logs'))

    # 记录训练过程
    train_losses = []
    val_losses = []
    train_dices = []
    val_dices = []

    best_val_dice = 0.0
    epochs_no_improve = 0
    best_epoch = 0

    for epoch in range(num_epochs):
        # 设置分布式采样器的 epoch
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_dice_loss = 0.0
        train_focal_loss = 0.0
        train_edge_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', disable=local_rank != 0)
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
            train_dice_loss += loss_components["dice_loss"].item()
            train_focal_loss += loss_components["focal_loss"].item()
            train_edge_loss += loss_components["edge_loss"].item()

            train_pbar.set_postfix({'loss': loss.item(),
                                    'dice_loss': loss_components["dice_loss"].item(),
                                    'focal_loss': loss_components["focal_loss"].item(),
                                    'edge_loss': loss_components["edge_loss"].item(),
                                    'dice': dice.item()})

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

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]', disable=local_rank != 0)
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
                val_dice_loss += loss_components["dice_loss"].item()
                val_focal_loss += loss_components["focal_loss"].item()
                val_edge_loss += loss_components["edge_loss"].item()

                val_pbar.set_postfix({'loss': loss.item(),
                                      'dice_loss': loss_components["dice_loss"].item(),
                                      'focal_loss': loss_components["focal_loss"].item(),
                                      'edge_loss': loss_components["edge_loss"].item(),
                                      'dice': dice.item()})

        # 计算平均损失和Dice系数
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_dice_loss /= len(val_loader)
        val_focal_loss /= len(val_loader)
        val_edge_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        # 记录到TensorBoard
        if local_rank == 0:
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

        # 打印当前epoch的结果
        if local_rank == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'  学习率: {optimizer.param_groups[0]["lr"]:.6f}')
            print(
                f'  训练损失: {train_loss:.4f}, 训练Dice损失: {train_dice_loss:.4f}, 训练Focal损失: {train_focal_loss:.4f}, 训练边界损失: {train_edge_loss:.4f}, 训练Dice: {train_dice:.4f}')
            print(
                f'  验证损失: {val_loss:.4f}, 验证Dice损失: {val_dice_loss:.4f}, 验证Focal损失: {val_focal_loss:.4f}, 验证边界损失: {val_edge_loss:.4f}, 验证Dice: {val_dice:.4f}')

        # 更新学习率调度器 (基于验证 Dice)
        scheduler.step(val_dice)

        # 早停逻辑和保存最佳模型
        if local_rank == 0 and val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            if isinstance(model, DDP):
                torch.save(model.module.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            else:
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
        if local_rank == 0:
            if isinstance(model, DDP):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_dice': train_dice,
                    'val_dice': val_dice,
                }, os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))
            else:
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
    if local_rank == 0:
        writer.close()

    # 保存最终模型
    if local_rank == 0:
        # 保存最终模型 (可能是早停前的模型，也可能是训练完所有 epoch 的模型)
        if isinstance(model, DDP):
            torch.save(model.module.state_dict(), os.path.join(save_dir, 'final_model.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
        print(f"最终模型已保存。最佳模型保存在 'best_model.pth' (Epoch {best_epoch}, Val Dice: {best_val_dice:.4f})")

    # 绘制训练过程
    if local_rank == 0:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss vs. Epoch')

        plt.subplot(1, 2, 2)
        plt.plot(train_dices, label='Train_Dice')
        plt.plot(val_dices, label='Val_Dice')
        plt.xlabel('Epoch')
        plt.ylabel('Dice_coefficient')
        plt.legend()
        plt.title('Dice vs. Epoch')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_history.png'))
        plt.show()

    return model


def main(local_rank):
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

    使用示例:
    python -m torch.distributed.launch --nproc_per_node=2 train.py --batch_size 1 --num_workers 4 --epochs 20 --lr 0.001
    """

    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

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
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--aug_prob', type=float, default=0.75, help='数据增强应用概率')
    parser.add_argument('--save_dir', type=str, default='train/checkpoints', help='模型保存目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # 损失函数相关参数
    parser.add_argument('--alpha', type=float, default=1.0, help='Dice损失权重')
    parser.add_argument('--beta', type=float, default=0.5, help='Focal损失权重')
    parser.add_argument('--gamma', type=float, default=0.2, help='边界损失权重')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

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

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 创建模型
    model = UNet(in_channels=args.in_channels, out_channels=args.out_channels).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 定义复合损失函数和优化器
    criterion = CompoundLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma, num_classes=args.out_channels).to(
        device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # 添加学习率调度器 (例如: 当验证 Dice 停止提升时降低学习率)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    # 打印训练配置
    if local_rank == 0:
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
        early_stopping_patience=8
    )

    if local_rank == 0:
        print('训练完成!')

    # 销毁分布式环境
    dist.destroy_process_group()


if __name__ == "__main__":
    local_rank = int(os.environ["LOCAL_RANK"])
    main(local_rank)
