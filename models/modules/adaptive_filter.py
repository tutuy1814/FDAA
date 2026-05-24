"""
Adaptive Filter Bank (AFB) - 自适应滤波器组
创新改进1：解决固定SRM滤波器无法适应不同生成器的问题

核心思想：
1. 保留固定SRM滤波器作为基础（捕获通用高频噪声）
2. 添加可学习滤波器组（自适应学习域特定模式）
3. 引入滤波器多样性约束（防止滤波器坍缩）
4. 动态滤波器选择机制（根据输入自适应选择）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRMFilterBank(nn.Module):
    """
    固定SRM滤波器组（基础）
    包含多种预定义的高频噪声检测滤波器
    """
    def __init__(self, in_channels=3):
        super().__init__()

        # 原始3个SRM滤波器
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

        # 添加更多边缘检测滤波器
        # Sobel X
        sobel_x = torch.tensor([
            [-1, 0, 1, 0, 0],
            [-2, 0, 2, 0, 0],
            [-1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32)

        # Sobel Y
        sobel_y = torch.tensor([
            [-1, -2, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 2, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=torch.float32)

        # Laplacian
        laplacian = torch.tensor([
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0]
        ], dtype=torch.float32) / 16.0

        # 组合所有滤波器
        filters = torch.stack([filter1, filter2, filter3, sobel_x, sobel_y, laplacian], dim=0)
        # 扩展为多输入通道 [6, 3, 5, 5]
        filters = filters.unsqueeze(1).repeat(1, in_channels, 1, 1)

        self.register_buffer('filters', filters)
        self.out_channels = 6

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            [B, 6, H, W] SRM特征
        """
        return F.conv2d(x, self.filters, padding=2)


class LearnableFilterBank(nn.Module):
    """
    可学习滤波器组
    自适应学习检测不同生成器特有的频率模式
    """
    def __init__(self, in_channels=3, num_filters=8, kernel_size=5):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        # 可学习滤波器参数
        self.filters = nn.Parameter(
            torch.randn(num_filters, in_channels, kernel_size, kernel_size) * 0.02
        )

        # 每个滤波器的可学习缩放因子
        self.scales = nn.Parameter(torch.ones(num_filters) * 0.1)

        # 初始化：部分滤波器初始化为类似高频检测的模式
        self._init_filters()

    def _init_filters(self):
        """初始化部分滤波器为有意义的模式"""
        with torch.no_grad():
            # 前4个滤波器初始化为不同方向的边缘检测
            directions = [
                [[0, 0, 0], [1, -2, 1], [0, 0, 0]],  # 水平
                [[0, 1, 0], [0, -2, 0], [0, 1, 0]],  # 垂直
                [[1, 0, 0], [0, -2, 0], [0, 0, 1]],  # 对角1
                [[0, 0, 1], [0, -2, 0], [1, 0, 0]],  # 对角2
            ]

            for i, d in enumerate(directions):
                if i < self.num_filters:
                    d_tensor = torch.tensor(d, dtype=torch.float32)
                    # 填充到kernel_size
                    pad = (self.kernel_size - 3) // 2
                    padded = F.pad(d_tensor, (pad, pad, pad, pad))
                    self.filters.data[i] = padded.unsqueeze(0).repeat(3, 1, 1)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            [B, num_filters, H, W] 可学习滤波器特征
        """
        padding = self.kernel_size // 2
        # 应用缩放
        scaled_filters = self.filters * self.scales.view(-1, 1, 1, 1)
        return F.conv2d(x, scaled_filters, padding=padding)

    def diversity_loss(self):
        """
        滤波器多样性损失
        防止所有滤波器学习相似的模式
        """
        # 展平滤波器
        filters_flat = self.filters.view(self.num_filters, -1)
        # L2归一化
        filters_norm = F.normalize(filters_flat, p=2, dim=1)
        # 计算余弦相似度矩阵
        similarity = torch.mm(filters_norm, filters_norm.T)
        # 创建掩码排除对角线
        mask = 1 - torch.eye(self.num_filters, device=self.filters.device)
        # 惩罚高相似度
        diversity_loss = (similarity.abs() * mask).sum() / (mask.sum() + 1e-8)
        return diversity_loss

    def sparsity_loss(self):
        """
        滤波器稀疏性损失
        鼓励滤波器具有稀疏结构（类似手工设计的滤波器）
        """
        return self.filters.abs().mean()


class FilterSelector(nn.Module):
    """
    动态滤波器选择网络
    根据输入图像的特性，自适应选择合适的滤波器组合
    """
    def __init__(self, in_channels=3, num_filters=14, hidden_dim=64):
        super().__init__()

        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(in_channels * 64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_filters),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            [B, num_filters] 滤波器权重
        """
        return self.selector(x)


class AdaptiveFilterBank(nn.Module):
    """
    自适应滤波器组 (AFB)

    完整模块，组合：
    1. 固定SRM滤波器（基础高频特征）
    2. 可学习滤波器（自适应特征）
    3. 动态选择机制（输入自适应）
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_learnable_filters: int = 8,
        kernel_size: int = 5,
        use_selector: bool = True,
        diversity_weight: float = 0.1,
        sparsity_weight: float = 0.01
    ):
        super().__init__()

        self.use_selector = use_selector
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight

        # 固定SRM滤波器
        self.srm_bank = SRMFilterBank(in_channels)

        # 可学习滤波器
        self.learnable_bank = LearnableFilterBank(
            in_channels, num_learnable_filters, kernel_size
        )

        # 总滤波器数量
        self.total_filters = self.srm_bank.out_channels + num_learnable_filters

        # 动态选择器
        if use_selector:
            self.selector = FilterSelector(in_channels, self.total_filters)

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(self.total_filters, self.total_filters, 1),
            nn.BatchNorm2d(self.total_filters),
            nn.ReLU(inplace=True)
        )

        # 输出通道数
        self.out_channels = self.total_filters

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            features: [B, out_channels, H, W] 滤波器特征
        """
        B = x.shape[0]

        # SRM特征
        srm_feat = self.srm_bank(x)  # [B, 6, H, W]

        # 可学习滤波器特征
        learnable_feat = self.learnable_bank(x)  # [B, 8, H, W]

        # 拼接
        combined = torch.cat([srm_feat, learnable_feat], dim=1)  # [B, 14, H, W]

        # 动态加权
        if self.use_selector:
            weights = self.selector(x)  # [B, 14]
            weights = weights.view(B, self.total_filters, 1, 1)
            combined = combined * weights

        # 特征融合
        output = self.fusion(combined)

        return output

    def get_regularization_loss(self):
        """
        获取正则化损失
        用于训练时添加到总损失中
        """
        losses = {}

        # 多样性损失
        diversity = self.learnable_bank.diversity_loss()
        losses['filter_diversity'] = self.diversity_weight * diversity

        # 稀疏性损失
        sparsity = self.learnable_bank.sparsity_loss()
        losses['filter_sparsity'] = self.sparsity_weight * sparsity

        losses['total_afb_reg'] = losses['filter_diversity'] + losses['filter_sparsity']

        return losses

    def visualize_filters(self, save_path=None):
        """可视化所有滤波器"""
        import matplotlib.pyplot as plt
        import numpy as np

        # 获取可学习滤波器
        filters = self.learnable_bank.filters.detach().cpu().numpy()
        num_filters = filters.shape[0]

        fig, axes = plt.subplots(2, (num_filters + 1) // 2, figsize=(12, 6))
        axes = axes.flatten()

        for i in range(num_filters):
            # 取第一个输入通道显示
            f = filters[i, 0]
            im = axes[i].imshow(f, cmap='coolwarm', vmin=-0.5, vmax=0.5)
            axes[i].set_title(f'Filter {i+1}')
            axes[i].axis('off')

        plt.colorbar(im, ax=axes, shrink=0.6)
        plt.suptitle('Learned Adaptive Filters')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class FrequencyAwareAFB(nn.Module):
    """
    频率感知自适应滤波器组
    在AFB基础上增加频率域分析
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_learnable_filters: int = 8,
        use_fft: bool = True
    ):
        super().__init__()

        self.afb = AdaptiveFilterBank(in_channels, num_learnable_filters)
        self.use_fft = use_fft

        if use_fft:
            # 频率域分支
            self.freq_conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, 32, 3, padding=1),  # 实部+虚部
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )

            # 融合
            total_channels = self.afb.out_channels + 16
        else:
            total_channels = self.afb.out_channels

        self.out_channels = total_channels

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, 1),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            [B, out_channels, H, W]
        """
        # 空间域滤波
        spatial_feat = self.afb(x)

        if self.use_fft:
            # 频率域特征
            x_fft = torch.fft.fft2(x, norm='ortho')
            x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))

            # 拼接实部和虚部
            freq_input = torch.cat([x_fft_shifted.real, x_fft_shifted.imag], dim=1)
            freq_feat = self.freq_conv(freq_input)

            # 融合
            combined = torch.cat([spatial_feat, freq_feat], dim=1)
        else:
            combined = spatial_feat

        output = self.output_proj(combined)
        return output

    def get_regularization_loss(self):
        return self.afb.get_regularization_loss()


# 测试代码
if __name__ == "__main__":
    print("Testing Adaptive Filter Bank (AFB)...")

    # 测试基础AFB
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    afb = AdaptiveFilterBank(
        in_channels=3,
        num_learnable_filters=8,
        use_selector=True
    )

    output = afb(x)
    print(f"AFB input shape: {x.shape}")
    print(f"AFB output shape: {output.shape}")
    print(f"AFB output channels: {afb.out_channels}")

    # 正则化损失
    reg_losses = afb.get_regularization_loss()
    print(f"Regularization losses: {reg_losses}")

    # 测试频率感知AFB
    freq_afb = FrequencyAwareAFB(
        in_channels=3,
        num_learnable_filters=8,
        use_fft=True
    )

    freq_output = freq_afb(x)
    print(f"\nFrequency-aware AFB output shape: {freq_output.shape}")
    print(f"Frequency-aware AFB output channels: {freq_afb.out_channels}")

    # 参数统计
    afb_params = sum(p.numel() for p in afb.parameters())
    freq_afb_params = sum(p.numel() for p in freq_afb.parameters())
    print(f"\nAFB parameters: {afb_params:,}")
    print(f"Frequency-aware AFB parameters: {freq_afb_params:,}")

    print("\nAll AFB tests passed!")
