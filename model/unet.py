import torch
import torch.nn as nn


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
        
        # 解码器部分 - 使用子像素卷积替代转置卷积
        self.upconv1 = self._make_subpixel_block(1024, 512)
        self.dec1 = self._make_decoder_block(1024, 512)
        
        self.upconv2 = self._make_subpixel_block(512, 256)
        self.dec2 = self._make_decoder_block(512, 256)
        
        self.upconv3 = self._make_subpixel_block(256, 128)
        self.dec3 = self._make_decoder_block(256, 128)
        
        self.upconv4 = self._make_subpixel_block(128, 64)
        self.dec4 = self._make_decoder_block(128, 64)
        
        # 输出层
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
        x = bottleneck_input
        for layer in self.bottleneck_dense:
            # 计算新的特征
            new_features = layer(x)
            # 连接到现有特征
            x = torch.cat([x, new_features], dim=1)
        
        # 应用过渡层得到最终的bottleneck输出
        bottleneck = self.bottleneck_transition(x)
        
        # 解码器路径
        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.upconv2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.dec2(dec2)
        
        dec3 = self.upconv3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        
        dec4 = self.upconv4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec4 = self.dec4(dec4)
        
        # 输出层 - 不需要在这里应用softmax，因为使用CrossEntropyLoss会自动应用
        outputs = self.out(dec4)
        return outputs
