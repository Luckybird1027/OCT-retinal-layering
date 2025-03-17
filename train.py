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

from model.unet import UNet
from utils.dataset import OCTDataset
from utils.advanced_augmentation import advanced_augmentation

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
def dice_coefficient(pred, target, epsilon=1e-6):
    # 将预测转换为one-hot表示
    pred = torch.argmax(pred, dim=1)

    # 计算每个类别的Dice系数
    dice_scores = []
    for class_idx in range(11):  # 11个类别
        pred_class = (pred == class_idx).float()
        target_class = (target == class_idx).float()

        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() + epsilon
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


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=30, save_dir='checkpoints'):
    # 创建保存检查点的目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter(os.path.join(save_dir, 'logs'))

    # 记录训练过程
    train_losses = []
    test_losses = []
    train_dices = []
    test_dices = []

    best_test_dice = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_dice = 0.0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]')
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

        # 测试阶段
        model.eval()
        test_loss = 0.0
        test_dice = 0.0

        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Test]')
        with torch.no_grad():
            for images, masks in test_pbar:
                images = images.to(device)
                masks = masks.to(device)

                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, masks)

                # 计算指标
                test_loss += loss.item()
                dice = dice_coefficient(outputs, masks)
                test_dice += dice.item()

                test_pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})

        # 计算平均损失和Dice系数
        test_loss /= len(test_loader)
        test_dice /= len(test_loader)
        test_losses.append(test_loss)
        test_dices.append(test_dice)

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Dice/train', train_dice, epoch)
        writer.add_scalar('Dice/test', test_dice, epoch)

        # 每个 Epoch 记录一些图像到 TensorBoard
        # 获取一个批次的图像和预测结果
        images, masks = next(iter(test_loader))
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
        mask_gray_grid = torchvision.utils.make_grid(mask_gray.unsqueeze(1))
        pred_gray_grid = torchvision.utils.make_grid(pred_gray.unsqueeze(1))

        # 将原始图像与灰度标签和预测图并排显示
        combined_grid = torchvision.utils.make_grid(
            torch.cat([images[:4], mask_gray.unsqueeze(1).float() / 255.0, pred_gray.unsqueeze(1).float() / 255.0],
                      dim=0), nrow=4
        )
        writer.add_image('Combined/Image_Mask_Prediction_Gray', combined_grid, epoch)

        # 保存最佳模型
        if test_dice > best_test_dice:
            best_test_dice = test_dice
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'保存最佳模型，Dice分数: {best_test_dice:.4f}')

        # 打印当前epoch的结果
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  训练损失: {train_loss:.4f}, 训练Dice: {train_dice:.4f}')
        print(f'  验证损失: {test_loss:.4f}, 验证Dice: {test_dice:.4f}')

        # 每个epoch保存一次模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_dice': train_dice,
            'test_dice': test_dice,
        }, os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth'))

    # 关闭TensorBoard的SummaryWriter
    writer.close()

    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))

    # 绘制训练过程
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title('损失 vs. Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label='训练Dice')
    plt.plot(test_dices, label='测试Dice')
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
    test_images_dir = 'data/SJTU/test/img'
    test_masks_dir = 'data/SJTU/test/mask'

    # 使用自定义数据增强
    train_transform = CustomTransform(apply_prob=0.5)

    # 创建数据集和数据加载器
    train_dataset = OCTDataset(train_images_dir, train_masks_dir, transform=train_transform)
    test_dataset = OCTDataset(test_images_dir, test_masks_dir, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 创建模型
    model = UNet(in_channels=1, out_channels=11).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=20,
        save_dir='train/checkpoints'
    )

    print('训练完成!')


if __name__ == '__main__':
    main()
