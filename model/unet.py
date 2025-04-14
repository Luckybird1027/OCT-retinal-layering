import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=11):
        super(UNet, self).__init__()

        # 编码器部分
        self.enc1 = self._make_encoder_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = self._make_encoder_block(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = self._make_encoder_block(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = self._make_encoder_block(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # DenseNet风格的Bottleneck
        self.growth_rate = 128
        self.num_dense_layers = 4
        self.bottleneck_dense = self._make_dense_block(512, self.growth_rate, self.num_dense_layers)
        # 计算最终密集块输出通道数: 原始通道数 + 每层新增通道数 * 层数
        dense_out_channels = 512 + self.growth_rate * self.num_dense_layers
        # 过渡层将输出通道数转换回1024以匹配原设计
        self.bottleneck_transition = self._make_transition(dense_out_channels, 1024)
        
        # 解码器部分 - U-Net 3+ 全尺度连接

        # 定义用于调整各尺度特征图通道数的卷积层 (统一到 64 通道)
        self.conv_enc1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv_enc2 = nn.Conv2d(128, 64, kernel_size=1)
        self.conv_enc3 = nn.Conv2d(256, 64, kernel_size=1)
        self.conv_enc4 = nn.Conv2d(512, 64, kernel_size=1)
        self.conv_bottleneck = nn.Conv2d(1024, 64, kernel_size=1)

        # 定义解码器各阶段的卷积层 (统一到 64 通道) - 注意通道数变化
        self.conv_dec1 = nn.Conv2d(512, 64, kernel_size=1) # dec1 输出 512 通道
        self.conv_dec2 = nn.Conv2d(256, 64, kernel_size=1) # dec2 输出 256 通道
        self.conv_dec3 = nn.Conv2d(128, 64, kernel_size=1) # dec3 输出 128 通道
        # dec4 输出 64 通道，无需转换

        # 定义融合卷积层：将5个来源（不同尺度）的特征融合 (每个来源64通道 -> 5*64=320)
        # 输出通道数与该解码器阶段最终输出通道数一致
        self.fusion_conv_dec1 = nn.Sequential(nn.Conv2d(320, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.fusion_conv_dec2 = nn.Sequential(nn.Conv2d(320, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.fusion_conv_dec3 = nn.Sequential(nn.Conv2d(320, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True))
        self.fusion_conv_dec4 = nn.Sequential(nn.Conv2d(320, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True))

        # 定义最终的解码器卷积块 (输入通道数变为融合后的通道数)
        # 注意：这里的 _make_decoder_block 输入通道数需要调整
        # dec1: 接收 fusion_conv_dec1 的输出 (512)
        self.dec1 = self._make_decoder_block(512, 512)
        # dec2: 接收 fusion_conv_dec2 的输出 (256)
        self.dec2 = self._make_decoder_block(256, 256)
        # dec3: 接收 fusion_conv_dec3 的输出 (128)
        self.dec3 = self._make_decoder_block(128, 128)
        # dec4: 接收 fusion_conv_dec4 的输出 (64)
        self.dec4 = self._make_decoder_block(64, 64)

        # 输出层 (输入通道数现在是 dec4 的输出通道数 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def _make_dense_block(self, in_channels, growth_rate, num_layers):
        """
        创建DenseNet风格的密集块，每一层都接收前面所有层的特征作为输入
        
        参数:
        - in_channels: 输入通道数
        - growth_rate: 每一层新增的通道数
        - num_layers: 密集块中的层数
        
        返回:
        - 密集块的层列表
        """
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, kernel_size=3, padding=1)
            ))
        return nn.ModuleList(layers)
    
    def _make_transition(self, in_channels, out_channels):
        """
        创建用于降低特征图通道数的过渡层
        
        参数:
        - in_channels: 输入通道数
        - out_channels: 输出通道数
        
        返回:
        - 过渡层序列
        """
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_subpixel_block(self, in_channels, out_channels):
        """
        创建子像素卷积上采样块
        """
        return nn.Sequential(
            # 将输入通道映射到 out_channels*4 (因为上采样因子是2)
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True),
            # 使用PixelShuffle进行上采样，尺寸增加2倍
            nn.PixelShuffle(2)
        )
    
    def forward(self, x):
        # 编码器路径
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # 瓶颈部分 - 使用密集连接
        bottleneck_input = self.pool4(enc4)
        
        # 实现密集连接的前向传播
        dense_x = bottleneck_input
        for layer in self.bottleneck_dense:
            new_features = layer(dense_x)
            dense_x = torch.cat([dense_x, new_features], dim=1)
        
        # 应用过渡层得到最终的bottleneck输出
        bottleneck = self.bottleneck_transition(dense_x)
        
        # 解码器路径 - U-Net 3+ 全尺度连接

        # 解码器阶段 1 (输出 scale 1/8, 目标 512 ch)
        # 准备来源特征 (统一到 64 ch)
        f_enc1_d1 = F.max_pool2d(self.conv_enc1(enc1), kernel_size=8, stride=8) # 下采样 x8
        f_enc2_d1 = F.max_pool2d(self.conv_enc2(enc2), kernel_size=4, stride=4) # 下采样 x4
        f_enc3_d1 = F.max_pool2d(self.conv_enc3(enc3), kernel_size=2, stride=2) # 下采样 x2
        f_enc4_d1 = self.conv_enc4(enc4)                                        # 同尺度
        f_bn_d1   = F.interpolate(self.conv_bottleneck(bottleneck), scale_factor=2, mode='bilinear', align_corners=False) # 上采样 x2
        # 拼接并融合
        f_cat_d1 = torch.cat([f_enc1_d1, f_enc2_d1, f_enc3_d1, f_enc4_d1, f_bn_d1], dim=1) # 5*64=320 ch
        f_fuse_d1 = self.fusion_conv_dec1(f_cat_d1) # 320 -> 512 ch
        # 通过解码器块
        dec1 = self.dec1(f_fuse_d1) # 输出 512 ch, scale 1/8

        # 解码器阶段 2 (输出 scale 1/4, 目标 256 ch)
        # 准备来源特征 (统一到 64 ch)
        f_enc1_d2 = F.max_pool2d(self.conv_enc1(enc1), kernel_size=4, stride=4) # 下采样 x4
        f_enc2_d2 = F.max_pool2d(self.conv_enc2(enc2), kernel_size=2, stride=2) # 下采样 x2
        f_enc3_d2 = self.conv_enc3(enc3)                                        # 同尺度
        f_enc4_d2 = F.interpolate(self.conv_enc4(enc4), scale_factor=2, mode='bilinear', align_corners=False) # 上采样 x2
        f_dec1_d2 = F.interpolate(self.conv_dec1(dec1), scale_factor=2, mode='bilinear', align_corners=False) # 上采样 x2 (来自上一解码层)
        # 拼接并融合
        f_cat_d2 = torch.cat([f_enc1_d2, f_enc2_d2, f_enc3_d2, f_enc4_d2, f_dec1_d2], dim=1) # 5*64=320 ch
        f_fuse_d2 = self.fusion_conv_dec2(f_cat_d2) # 320 -> 256 ch
        # 通过解码器块
        dec2 = self.dec2(f_fuse_d2) # 输出 256 ch, scale 1/4

        # 解码器阶段 3 (输出 scale 1/2, 目标 128 ch)
        # 准备来源特征 (统一到 64 ch)
        f_enc1_d3 = F.max_pool2d(self.conv_enc1(enc1), kernel_size=2, stride=2) # 下采样 x2
        f_enc2_d3 = self.conv_enc2(enc2)                                        # 同尺度
        f_enc3_d3 = F.interpolate(self.conv_enc3(enc3), scale_factor=2, mode='bilinear', align_corners=False) # 上采样 x2
        f_enc4_d3 = F.interpolate(self.conv_enc4(enc4), scale_factor=4, mode='bilinear', align_corners=False) # 上采样 x4
        f_dec2_d3 = F.interpolate(self.conv_dec2(dec2), scale_factor=2, mode='bilinear', align_corners=False) # 上采样 x2 (来自上一解码层)
        # 拼接并融合
        f_cat_d3 = torch.cat([f_enc1_d3, f_enc2_d3, f_enc3_d3, f_enc4_d3, f_dec2_d3], dim=1) # 5*64=320 ch
        f_fuse_d3 = self.fusion_conv_dec3(f_cat_d3) # 320 -> 128 ch
        # 通过解码器块
        dec3 = self.dec3(f_fuse_d3) # 输出 128 ch, scale 1/2

        # 解码器阶段 4 (输出 scale 1, 目标 64 ch)
        # 准备来源特征 (统一到 64 ch)
        f_enc1_d4 = self.conv_enc1(enc1)                                        # 同尺度
        f_enc2_d4 = F.interpolate(self.conv_enc2(enc2), scale_factor=2, mode='bilinear', align_corners=False) # 上采样 x2
        f_enc3_d4 = F.interpolate(self.conv_enc3(enc3), scale_factor=4, mode='bilinear', align_corners=False) # 上采样 x4
        f_enc4_d4 = F.interpolate(self.conv_enc4(enc4), scale_factor=8, mode='bilinear', align_corners=False) # 上采样 x8
        f_dec3_d4 = F.interpolate(self.conv_dec3(dec3), scale_factor=2, mode='bilinear', align_corners=False) # 上采样 x2 (来自上一解码层)
        # 拼接并融合
        f_cat_d4 = torch.cat([f_enc1_d4, f_enc2_d4, f_enc3_d4, f_enc4_d4, f_dec3_d4], dim=1) # 5*64=320 ch
        f_fuse_d4 = self.fusion_conv_dec4(f_cat_d4) # 320 -> 64 ch
        # 通过解码器块
        dec4 = self.dec4(f_fuse_d4) # 输出 64 ch, scale 1

        # 输出层 - 不需要在这里应用softmax，因为使用CrossEntropyLoss会自动应用
        outputs = self.out(dec4)
        return outputs
