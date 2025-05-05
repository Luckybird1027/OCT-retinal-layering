import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DenseNet Bottleneck Components ---

class _DenseLayer(nn.Module):
    """Basic building block for DenseBottleneck: BN-ReLU-Conv1x1-BN-ReLU-Conv3x3"""
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super(_DenseLayer, self).__init__()
        inter_channels = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        # x is a list of tensors or a single tensor
        prev_features = x if isinstance(x, list) else [x]
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        return new_features

class DenseBottleneck(nn.Module):
    """DenseNet-style bottleneck block"""
    def __init__(self, in_channels, growth_rate, num_layers, bn_size=4):
        super(DenseBottleneck, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_in_channels = in_channels + i * growth_rate
            layer = _DenseLayer(layer_in_channels, growth_rate, bn_size)
            self.layers.append(layer)

    def forward(self, init_features):
        features = [init_features]
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


# --- Squeeze-and-Excitation Block ---

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# --- Criss-Cross Attention Block ---
class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim, reduction=8):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // reduction, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // reduction, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1)) # Learnable scale parameter, initialized to 0

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        # Horizontal context
        proj_query_h = proj_query.permute(0, 2, 3, 1).contiguous().view(m_batchsize * height, width, -1)
        proj_key_h = proj_key.permute(0, 2, 3, 1).contiguous().view(m_batchsize * height, -1, width)
        proj_value_h = proj_value.permute(0, 2, 3, 1).contiguous().view(m_batchsize * height, width, -1)
        energy_h = torch.bmm(proj_query_h, proj_key_h) # Shape: (B*H, W, W)
        # energy_h = (proj_query_h @ proj_key_h.transpose(1, 2)) # Equivalent
        attention_h = self.softmax(energy_h) # Softmax along the last dimension (Key positions)
        context_h = torch.bmm(attention_h, proj_value_h)
        context_h = context_h.view(m_batchsize, height, width, -1).permute(0, 3, 1, 2).contiguous()

        # Vertical context
        proj_query_v = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, height, -1)
        proj_key_v = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_v = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, height, -1)
        energy_v = torch.bmm(proj_query_v, proj_key_v) # Shape: (B*W, H, H)
        # energy_v = (proj_query_v @ proj_key_v.transpose(1, 2)) # Equivalent
        attention_v = self.softmax(energy_v) # Softmax along the last dimension (Key positions)
        context_v = torch.bmm(attention_v, proj_value_v)
        context_v = context_v.view(m_batchsize, width, height, -1).permute(0, 3, 2, 1).contiguous() # Note the permutation

        # Aggregate context and apply residual connection
        out = self.gamma * (context_h + context_v) + x # Additive aggregation
        return out


# --- UNet with UNet 3+ Skip Connections, Dense Bottleneck, SE and CCA ---

class UNet(nn.Module):
    """
    U-Net architecture with modifications:
    1. UNet 3+ style full-scale skip connections.
    2. DenseNet-style bottleneck.
    3. Sub-pixel convolution for upsampling.
    4. Squeeze-and-Excitation (SE) blocks in encoder and bottleneck.
    5. Criss-Cross Attention (CCA) after bottleneck.
    """
    def __init__(self, in_channels=1, out_channels=11, growth_rate=64, bn_size=4, num_dense_layers=4, cca_reduction=8):
        super(UNet, self).__init__()

        # --- Encoder Blocks ---
        self.enc1 = self._make_encoder_block(in_channels, 64)      # Output: N, 64, H, W
        self.se1 = SEBlock(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = self._make_encoder_block(64, 128)              # Output: N, 128, H/2, W/2
        self.se2 = SEBlock(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = self._make_encoder_block(128, 256)             # Output: N, 256, H/4, W/4
        self.se3 = SEBlock(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = self._make_encoder_block(256, 512)             # Output: N, 512, H/8, W/8
        self.se4 = SEBlock(512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Dense Bottleneck ---
        bottleneck_in_channels = 512
        self.bottleneck = DenseBottleneck(
            in_channels=bottleneck_in_channels,
            growth_rate=growth_rate,
            num_layers=num_dense_layers,
            bn_size=bn_size
        )
        # Output: N, bottleneck_in + num_layers * growth_rate, H/16, W/16
        bottleneck_out_channels = bottleneck_in_channels + num_dense_layers * growth_rate # 512 + 4*128 = 1024
        self.se_bottleneck = SEBlock(bottleneck_out_channels)

        # --- Criss-Cross Attention ---
        self.cca = CrissCrossAttention(bottleneck_out_channels, reduction=cca_reduction)

        # --- Pooling Layers for UNet 3+ Skip Connections ---
        # These pool encoder features to match decoder feature map sizes
        self.pool1_d1 = nn.MaxPool2d(kernel_size=8, stride=8) # enc1 -> dec1 size
        self.pool2_d1 = nn.MaxPool2d(kernel_size=4, stride=4) # enc2 -> dec1 size
        self.pool3_d1 = nn.MaxPool2d(kernel_size=2, stride=2) # enc3 -> dec1 size

        self.pool1_d2 = nn.MaxPool2d(kernel_size=4, stride=4) # enc1 -> dec2 size
        self.pool2_d2 = nn.MaxPool2d(kernel_size=2, stride=2) # enc2 -> dec2 size

        self.pool1_d3 = nn.MaxPool2d(kernel_size=2, stride=2) # enc1 -> dec3 size

        # --- Decoder Blocks with Sub-pixel Upsampling ---
        # Decoder 1 (Receives features from CCA and enc1, enc2, enc3, enc4)
        self.upconv1 = self._make_subpixel_block(bottleneck_out_channels, 512) # Output: N, 512, H/8, W/8
        dec1_in_channels = 512 + 512 + 256 + 128 + 64 # upconv1 + enc4 + pooled enc3 + pooled enc2 + pooled enc1
        self.dec1 = self._make_decoder_block(dec1_in_channels, 512)           # Output: N, 512, H/8, W/8

        # Decoder 2 (Receives features from dec1 and enc1, enc2, enc3)
        self.upconv2 = self._make_subpixel_block(512, 256)                    # Output: N, 256, H/4, W/4
        dec2_in_channels = 256 + 256 + 128 + 64 # upconv2 + enc3 + pooled enc2 + pooled enc1
        self.dec2 = self._make_decoder_block(dec2_in_channels, 256)           # Output: N, 256, H/4, W/4

        # Decoder 3 (Receives features from dec2 and enc1, enc2)
        self.upconv3 = self._make_subpixel_block(256, 128)                    # Output: N, 128, H/2, W/2
        dec3_in_channels = 128 + 128 + 64 # upconv3 + enc2 + pooled enc1
        self.dec3 = self._make_decoder_block(dec3_in_channels, 128)           # Output: N, 128, H/2, W/2

        # Decoder 4 (Receives features from dec3 and enc1)
        self.upconv4 = self._make_subpixel_block(128, 64)                     # Output: N, 64, H, W
        dec4_in_channels = 64 + 64 # upconv4 + enc1
        self.dec4 = self._make_decoder_block(dec4_in_channels, 64)            # Output: N, 64, H, W

        # --- Output Layer ---
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)                 # Output: N, num_classes, H, W

    # --- Helper Methods for Building Blocks ---
    def _make_encoder_block(self, in_channels, out_channels):
        """Creates a standard double convolution block for the encoder."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(self, in_channels, out_channels):
        """Creates a standard double convolution block for the decoder."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _make_subpixel_block(self, in_channels, out_channels):
        """Creates an upsampling block using sub-pixel convolution (PixelShuffle)."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(2) # Upsamples by a factor of 2
        )

    # --- Forward Pass ---
    def forward(self, x):
        # --- Encoder Path ---
        enc1 = self.enc1(x)
        enc1 = self.se1(enc1)
        enc2 = self.enc2(self.pool1(enc1))
        enc2 = self.se2(enc2)
        enc3 = self.enc3(self.pool2(enc2))
        enc3 = self.se3(enc3)
        enc4 = self.enc4(self.pool3(enc3))
        enc4 = self.se4(enc4)

        # --- Bottleneck ---
        bottleneck_out = self.bottleneck(self.pool4(enc4))
        bottleneck_se = self.se_bottleneck(bottleneck_out)

        # --- Apply Criss-Cross Attention ---
        cca_out = self.cca(bottleneck_se) # Apply CCA after SE block

        # --- Decoder Path with UNet 3+ Skip Connections ---
        # Decoder 1
        up1 = self.upconv1(cca_out)
        # Pool encoder features to match dec1 size (H/8, W/8)
        enc1_d1 = self.pool1_d1(enc1)
        enc2_d1 = self.pool2_d1(enc2)
        enc3_d1 = self.pool3_d1(enc3)
        # Concatenate features from cca_out(up1), enc4, and pooled enc1, enc2, enc3
        cat1 = torch.cat((up1, enc4, enc3_d1, enc2_d1, enc1_d1), dim=1)
        dec1 = self.dec1(cat1)

        # Decoder 2
        up2 = self.upconv2(dec1)
        # Pool encoder features to match dec2 size (H/4, W/4)
        enc1_d2 = self.pool1_d2(enc1)
        enc2_d2 = self.pool2_d2(enc2)
        # Concatenate features from dec1(up2), enc3, and pooled enc1, enc2
        cat2 = torch.cat((up2, enc3, enc2_d2, enc1_d2), dim=1)
        dec2 = self.dec2(cat2)

        # Decoder 3
        up3 = self.upconv3(dec2)
        # Pool encoder features to match dec3 size (H/2, W/2)
        enc1_d3 = self.pool1_d3(enc1)
        # Concatenate features from dec2(up3), enc2, and pooled enc1
        cat3 = torch.cat((up3, enc2, enc1_d3), dim=1)
        dec3 = self.dec3(cat3)

        # Decoder 4
        up4 = self.upconv4(dec3)
        # Concatenate features from dec3(up4) and enc1
        cat4 = torch.cat((up4, enc1), dim=1)
        dec4 = self.dec4(cat4)

        # --- Output Layer ---
        outputs = self.out(dec4)
        return outputs
