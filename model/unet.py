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

        # 瓶颈部分
        self.bottleneck = self._make_encoder_block(512, 1024)

        # --- 新增: 用于下采样编码器特征以匹配解码器尺度的池化层 ---
        # For dec3: enc1 (64x64 -> 32x32)
        self.pool1_d3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # For dec2: enc1 (64x64 -> 16x16), enc2 (32x32 -> 16x16)
        self.pool1_d2 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool2_d2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # For dec1: enc1 (64x64 -> 8x8), enc2 (32x32 -> 8x8), enc3 (16x16 -> 8x8)
        self.pool1_d1 = nn.MaxPool2d(kernel_size=8, stride=8)
        self.pool2_d1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool3_d1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解码器部分 - 使用子像素卷积替代转置卷积
        self.upconv1 = self._make_subpixel_block(1024, 512)
        # --- 修改 dec1 输入通道: upconv1(512) + enc4(512) + pool1_d1(64) + pool2_d1(128) + pool3_d1(256) = 1512 ---
        self.dec1 = self._make_decoder_block(512 + 512 + 64 + 128 + 256, 512)

        self.upconv2 = self._make_subpixel_block(512, 256)
        # --- 修改 dec2 输入通道: upconv2(256) + enc3(256) + pool1_d2(64) + pool2_d2(128) = 704 ---
        self.dec2 = self._make_decoder_block(256 + 256 + 64 + 128, 256)

        self.upconv3 = self._make_subpixel_block(256, 128)
        # --- 修改 dec3 输入通道: upconv3(128) + enc2(128) + pool1_d3(64) = 320 ---
        self.dec3 = self._make_decoder_block(128 + 128 + 64, 128)

        self.upconv4 = self._make_subpixel_block(128, 64)
        # --- 修改 dec4 输入通道: upconv4(64) + enc1(64) = 128 (无需额外连接更浅层的) ---
        self.dec4 = self._make_decoder_block(128, 64)

        # 输出层
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

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
        enc1 = self.enc1(x)         # N, 64, H, W
        enc2 = self.enc2(self.pool1(enc1)) # N, 128, H/2, W/2
        enc3 = self.enc3(self.pool2(enc2)) # N, 256, H/4, W/4
        enc4 = self.enc4(self.pool3(enc3)) # N, 512, H/8, W/8

        # 瓶颈
        bottleneck = self.bottleneck(self.pool4(enc4)) # N, 1024, H/16, W/16

        # 解码器路径 (实现 UNet 3+ 风格的全尺度跳跃连接)

        # --- 解码器层 1 ---
        up1 = self.upconv1(bottleneck) # N, 512, H/8, W/8
        # 下采样浅层编码器特征以匹配 enc4 的尺寸 (H/8, W/8)
        enc1_d1 = self.pool1_d1(enc1)
        enc2_d1 = self.pool2_d1(enc2)
        enc3_d1 = self.pool3_d1(enc3)
        # 拼接
        cat1 = torch.cat((up1, enc4, enc3_d1, enc2_d1, enc1_d1), dim=1) # 512 + 512 + 256 + 128 + 64 = 1512
        dec1 = self.dec1(cat1) # N, 512, H/8, W/8

        # --- 解码器层 2 ---
        up2 = self.upconv2(dec1) # N, 256, H/4, W/4
        # 下采样浅层编码器特征以匹配 enc3 的尺寸 (H/4, W/4)
        enc1_d2 = self.pool1_d2(enc1)
        enc2_d2 = self.pool2_d2(enc2)
        # 拼接
        cat2 = torch.cat((up2, enc3, enc2_d2, enc1_d2), dim=1) # 256 + 256 + 128 + 64 = 704
        dec2 = self.dec2(cat2) # N, 256, H/4, W/4

        # --- 解码器层 3 ---
        up3 = self.upconv3(dec2) # N, 128, H/2, W/2
        # 下采样浅层编码器特征以匹配 enc2 的尺寸 (H/2, W/2)
        enc1_d3 = self.pool1_d3(enc1)
        # 拼接
        cat3 = torch.cat((up3, enc2, enc1_d3), dim=1) # 128 + 128 + 64 = 320
        dec3 = self.dec3(cat3) # N, 128, H/2, W/2

        # --- 解码器层 4 ---
        up4 = self.upconv4(dec3) # N, 64, H, W
        # 无需下采样，直接拼接
        cat4 = torch.cat((up4, enc1), dim=1) # 64 + 64 = 128
        dec4 = self.dec4(cat4) # N, 64, H, W

        # 输出层
        outputs = self.out(dec4)
        return outputs
