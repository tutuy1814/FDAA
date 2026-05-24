
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    HAS_WAVELETS = True
except ImportError:
    HAS_WAVELETS = False


# =============================================================================
# V1 组件 (保留兼容性)
# =============================================================================

class SRMConv2d(nn.Module):
    """
    Spatial Rich Model (SRM) 滤波器 (V1: 固定权重)
    参考: Luo et al., "Generalizing Face Forgery Detection with High-frequency Features", CVPR 2021
    """
    def __init__(self, in_channels=3, out_channels=30):
        super().__init__()
        filter1 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32) / 4.0

        filter2 = torch.tensor([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=torch.float32) / 12.0

        filter3 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32) / 2.0

        filters = torch.stack([filter1, filter2, filter3], dim=0)  # [3, 5, 5]
        filters = filters.unsqueeze(1).repeat(out_channels // 3, in_channels, 1, 1)

        self.register_buffer('weight', filters)
        self.out_channels = out_channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2)


class DCTLayer(nn.Module):
    """
    离散余弦变换层 (V1: 简化CNN近似)
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialAdapter(nn.Module):
    """空间域Adapter (V1)"""
    def __init__(self, dim, reduction=4, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(dim // reduction, dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = self.up(self.dropout(self.act(self.down(x))))
        return x + self.scale * residual


class FrequencyAdapter(nn.Module):
    """频率域Adapter (V1)"""
    def __init__(self, dim, img_size=224, patch_size=16, reduction=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        self.srm = SRMConv2d(3, 30)
        self.dct = DCTLayer(3, 64)

        self.freq_fusion = nn.Sequential(
            nn.Linear(94, dim // reduction),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // reduction, dim)
        )

        if HAS_WAVELETS:
            self.dwt = DWTForward(J=1, wave='haar')
            self.idwt = DWTInverse(wave='haar')
            self.use_wavelet = True
        else:
            self.use_wavelet = False

        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, image=None):
        B, N, D = x.shape
        if image is not None:
            srm_feat = self.srm(image)
            srm_feat = F.adaptive_avg_pool2d(srm_feat, (int(math.sqrt(N)), int(math.sqrt(N))))
            srm_feat = srm_feat.flatten(2).transpose(1, 2)

            dct_feat = self.dct(image)
            dct_feat = F.adaptive_avg_pool2d(dct_feat, (int(math.sqrt(N)), int(math.sqrt(N))))
            dct_feat = dct_feat.flatten(2).transpose(1, 2)

            freq_feat = torch.cat([srm_feat, dct_feat], dim=-1)
            freq_out = self.freq_fusion(freq_feat)

            return x + self.scale * self.norm(freq_out)
        else:
            return x


class CrossDomainInteraction(nn.Module):
    """跨域交互模块 (V1)"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.spatial_to_freq = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.freq_to_spatial = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, spatial_feat, freq_feat):
        s2f, _ = self.spatial_to_freq(freq_feat, spatial_feat, spatial_feat)
        s2f = self.norm1(freq_feat + self.dropout(s2f))

        f2s, _ = self.freq_to_spatial(spatial_feat, freq_feat, freq_feat)
        f2s = self.norm2(spatial_feat + self.dropout(f2s))

        concat = torch.cat([s2f, f2s], dim=-1)
        gate = self.gate(concat)
        fused = gate * s2f + (1 - gate) * f2s

        return fused


# V1 完整模块 (别名 FDAAv1)
class FDAA(nn.Module):
    """
    Frequency-aware Dual-domain Adaptive Adapter (V1)
    """
    def __init__(self, dim, img_size=224, patch_size=16, reduction=4, num_heads=8, dropout=0.1):
        super().__init__()

        self.spatial_adapter = SpatialAdapter(dim, reduction, dropout)
        self.freq_adapter = FrequencyAdapter(dim, img_size, patch_size, reduction, dropout)
        self.cross_interaction = CrossDomainInteraction(dim, num_heads, dropout)

        self.output_norm = nn.LayerNorm(dim)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, image=None):
        spatial_out = self.spatial_adapter(x)
        freq_out = self.freq_adapter(x, image)
        fused = self.cross_interaction(spatial_out, freq_out)
        output = x + self.output_scale * self.output_norm(fused - x)
        return output


FDAAv1 = FDAA  # 别名


# =============================================================================
# V2 组件 (改进版)
# =============================================================================

class LearnableSRMConv2d(nn.Module):
    """
    可学习的 SRM 滤波器 (V2)
    保留手工初始化值作为起点，但允许训练时微调
    """
    def __init__(self, in_channels=3, out_channels=30):
        super().__init__()
        self.out_channels = out_channels

        # 3个基础SRM滤波器
        filter1 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 2, -4, 2, 0],
            [0, -1, 2, -1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32) / 4.0

        filter2 = torch.tensor([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=torch.float32) / 12.0

        filter3 = torch.tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32) / 2.0

        # 扩展为多通道滤波器，用 nn.Parameter 使其可学习
        filters = torch.stack([filter1, filter2, filter3], dim=0)  # [3, 5, 5]
        filters = filters.unsqueeze(1).repeat(out_channels // 3, in_channels, 1, 1)  # [out, in, 5, 5]

        self.weight = nn.Parameter(filters)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=2)


class RealFFTLayer(nn.Module):
    """
    真实的 FFT 频率特征提取层 (V2)
    使用 torch.fft.fft2 提取幅度谱和相位谱
    输出: [B, 6, H, W] (3ch magnitude + 3ch phase)
    """
    def __init__(self, out_channels=3):
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            freq_feat: [B, 3, H, W] 对数幅度谱 (保留空间结构)
        """
        # FFT变换
        fft = torch.fft.fft2(x, norm='ortho')
        fft_shift = torch.fft.fftshift(fft)

        # 对数幅度谱 — 保留频率分布信息
        magnitude = torch.abs(fft_shift)
        log_magnitude = torch.log1p(magnitude)

        return log_magnitude  # [B, 3, H, W]


class FrequencyBranch(nn.Module):
    """
    处理原始图像（不修改ViT token），提取频率域特征向量。
    架构: SRM(30ch) + FFT(3ch) → concat(33ch) → CNN Encoder → Pool → Linear → [B, D]
    """
    def __init__(self, embed_dim=768, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # 可学习SRM高频噪声提取
        self.srm = LearnableSRMConv2d(3, 30)

        # 真实FFT频率特征
        self.fft = RealFFTLayer(out_channels=3)

        # CNN编码器: 33ch → embed_dim
        self.encoder = nn.Sequential(
            # Stage 1: 33 → 64, 224→112
            nn.Conv2d(33, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Stage 2: 64 → 128, 112→56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Stage 3: 128 → 256, 56→28
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # Stage 4: 256 → 512, 28→14
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),

            # Stage 5: 512 → 512, 14→7
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.GELU(),
        )

        # 全局池化 → 投影到 embed_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W] 原始图像
        Returns:
            freq_feat: [B, embed_dim] 频率特征向量
        """
        # SRM高频残差
        srm_feat = self.srm(image)        # [B, 30, H, W]

        # FFT幅度谱
        fft_feat = self.fft(image)         # [B, 3, H, W]

        # 拼接频率特征
        freq_input = torch.cat([srm_feat, fft_feat], dim=1)  # [B, 33, H, W]

        # CNN编码
        encoded = self.encoder(freq_input)  # [B, 512, 7, 7]

        # 池化 + 投影
        pooled = self.pool(encoded).flatten(1)  # [B, 512]
        freq_feat = self.proj(pooled)           # [B, embed_dim]

        return freq_feat


class FDAAv2(nn.Module):
    """
    Frequency-aware Dual-domain Adaptive Adapter V2/V3

    关键改进（V3 over V2）：
    1. 使用真实 FFT (torch.fft.fft2) 替代假 DCT
    2. SRM 滤波器可学习（保留手工初始化）
    3. 独立频率分支，不修改 ViT token
    4. 更浅的 CNN 编码器（3 stages vs 5），减少空间信息损失
    5. embed_dim 默认 1024，匹配 CLIP ViT-L/14

    用法：
        fdaa = FDAAv2(embed_dim=1024)
        freq_feat = fdaa(image)  # [B, D], 用于后续 late fusion
    """
    def __init__(self, embed_dim=1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # 可学习SRM高频噪声提取
        self.srm = LearnableSRMConv2d(3, 30)

        # 真实FFT频率特征
        self.fft = RealFFTLayer(out_channels=3)

        # CNN 编码器（4 stages），无感受野空隙
        self.encoder = nn.Sequential(
            # Stage 1: 33 → 64, 224→112
            nn.Conv2d(33, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Stage 2: 64 → 128, 112→56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Stage 3: 128 → 256, 56→28（拆分原 stride=4 为两个 stride=2）
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # Stage 4: 256 → 256, 28→14（保持完整感受野覆盖）
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # 全局池化 → 投影到 embed_dim
        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W] 原始图像（值域 [0,1]，调用方需先反归一化）
        Returns:
            freq_feat: [B, embed_dim] 频率特征向量
        """
        # SRM高频残差
        srm_feat = self.srm(image)        # [B, 30, H, W]

        # FFT幅度谱
        fft_feat = self.fft(image)         # [B, 3, H, W]

        # 拼接频率特征
        freq_input = torch.cat([srm_feat, fft_feat], dim=1)  # [B, 33, H, W]

        # CNN编码
        encoded = self.encoder(freq_input)  # [B, 256, 14, 14]

        # 池化 + 投影
        freq_feat = self.proj(encoded)      # [B, embed_dim]

        return freq_feat


# =============================================================================
# V3 组件 (空间频率特征 + 相位谱)
# =============================================================================

class RealFFTLayerV2(nn.Module):
    """
    FFT 频率特征提取层 V2
    同时输出幅度谱和相位谱（6 通道），提供更完整的频率信息。
    相位谱包含边缘/结构信息，对检测不同生成器的伪影模式至关重要。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            freq_feat: [B, 6, H, W] (3ch log-magnitude + 3ch normalized phase)
        """
        fft = torch.fft.fft2(x, norm='ortho')
        fft_shift = torch.fft.fftshift(fft)

        # 对数幅度谱
        magnitude = torch.abs(fft_shift)
        log_magnitude = torch.log1p(magnitude)

        # 相位谱（归一化到 [-1, 1]）
        phase = torch.angle(fft_shift) / math.pi

        return torch.cat([log_magnitude, phase], dim=1)  # [B, 6, H, W]


class FDAAv3(nn.Module):
    """
    Frequency-aware Dual-domain Adaptive Adapter V3

    关键改进（V3 over V2）：
    1. 去掉 GlobalAvgPool，保留空间结构 → 输出 freq_tokens [B, N, D]
    2. 加入相位谱（RealFFTLayerV2: 6ch），输入从 33ch → 36ch
    3. 同时返回 freq_tokens 和 freq_global，MGFP 可利用逐 patch 频率证据
    4. CNN 输出 [B, C, h, w] → Conv1x1 投影 → reshape 为 token 序列

    用法：
        fdaa = FDAAv3(embed_dim=1024, num_patches=256)
        freq_tokens, freq_global = fdaa(image)
        # freq_tokens: [B, 256, 1024], freq_global: [B, 1024]
    """
    def __init__(self, embed_dim=1024, num_patches=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches

        # 可学习 SRM 高频噪声提取
        self.srm = LearnableSRMConv2d(3, 30)

        # FFT V2: magnitude + phase = 6 channels
        self.fft = RealFFTLayerV2()

        # CNN 编码器: 36ch → 256ch, 保留 14x14 空间结构
        # 输入 36 = 30(SRM) + 3(magnitude) + 3(phase)
        self.encoder = nn.Sequential(
            # Stage 1: 36 → 64, 224→112
            nn.Conv2d(36, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # Stage 2: 64 → 128, 112→56
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Stage 3: 128 → 256, 56→28
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # Stage 4: 256 → 256, 28→14
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        # 空间投影: [B, 256, 14, 14] → [B, 256, embed_dim] via Conv1x1
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(256, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
        )

        # Token 归一化
        self.token_norm = nn.LayerNorm(embed_dim)

        # Global 投影 (mean pool → linear)
        self.global_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.spatial_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.global_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W] 原始图像（值域 [0,1]）
        Returns:
            freq_tokens: [B, num_patches, embed_dim] 逐 patch 频率特征
            freq_global: [B, embed_dim] 全局频率特征
        """
        # SRM 高频残差
        srm_feat = self.srm(image)        # [B, 30, H, W]

        # FFT V2: magnitude + phase
        fft_feat = self.fft(image)         # [B, 6, H, W]

        # 拼接频率特征
        freq_input = torch.cat([srm_feat, fft_feat], dim=1)  # [B, 36, H, W]

        # CNN 编码（保留空间结构）
        encoded = self.encoder(freq_input)  # [B, 256, 14, 14]

        # 空间投影到 embed_dim
        spatial = self.spatial_proj(encoded)  # [B, embed_dim, 14, 14]

        # Reshape 为 token 序列
        B, D, h, w = spatial.shape
        n_tokens = h * w  # 14*14 = 196

        # 如果 num_patches != n_tokens，用 interpolate 对齐
        if n_tokens != self.num_patches:
            target_h = int(math.sqrt(self.num_patches))
            spatial = F.interpolate(spatial, size=(target_h, target_h),
                                    mode='bilinear', align_corners=False)

        freq_tokens = spatial.flatten(2).transpose(1, 2)  # [B, N, D]
        freq_tokens = self.token_norm(freq_tokens)

        # Global: mean pool
        freq_global = freq_tokens.mean(dim=1)  # [B, D]
        freq_global = self.global_proj(freq_global)

        return freq_tokens, freq_global


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing FDAA V1...")
    batch_size = 4
    num_patches = 196
    dim = 768

    x = torch.randn(batch_size, num_patches, dim)
    image = torch.randn(batch_size, 3, 224, 224)

    fdaa_v1 = FDAA(dim=dim, img_size=224, patch_size=16)
    output_v1 = fdaa_v1(x, image)
    print(f"V1 Input shape: {x.shape}")
    print(f"V1 Output shape: {output_v1.shape}")
    print(f"V1 parameters: {sum(p.numel() for p in fdaa_v1.parameters()):,}")

    print()
    print("=" * 60)
    print("Testing FDAA V2/V3 (improved, 3-stage encoder)...")

    dim_v2 = 1024  # CLIP ViT-L/14 内部维度
    fdaa_v2 = FDAAv2(embed_dim=dim_v2)
    freq_feat = fdaa_v2(image)
    print(f"V2 Input shape: {image.shape}")
    print(f"V2 Output shape: {freq_feat.shape}")
    print(f"V2 parameters: {sum(p.numel() for p in fdaa_v2.parameters()):,}")

    # Verify grad flow
    loss = freq_feat.sum()
    loss.backward()
    srm_grad = fdaa_v2.srm.weight.grad
    print(f"V2 SRM grad exists: {srm_grad is not None}")
    print(f"V2 SRM grad norm: {srm_grad.norm().item():.6f}" if srm_grad is not None else "")

    print()
    print("=" * 60)
    print("Testing FDAAv3 (spatial freq tokens + phase)...")

    fdaa_v3 = FDAAv3(embed_dim=1024, num_patches=256)
    freq_tokens, freq_global = fdaa_v3(image)
    print(f"V3 Input shape: {image.shape}")
    print(f"V3 freq_tokens shape: {freq_tokens.shape}")
    print(f"V3 freq_global shape: {freq_global.shape}")
    print(f"V3 parameters: {sum(p.numel() for p in fdaa_v3.parameters()):,}")

    loss_v3 = freq_tokens.sum() + freq_global.sum()
    loss_v3.backward()
    srm_grad_v3 = fdaa_v3.srm.weight.grad
    print(f"V3 SRM grad exists: {srm_grad_v3 is not None}")
    print(f"V3 SRM grad norm: {srm_grad_v3.norm().item():.6f}" if srm_grad_v3 is not None else "")
