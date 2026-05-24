"""
域适应模块 - 用于提升跨数据集泛化能力

包含:
1. GradientReversalLayer - 梯度反转层
2. DomainDiscriminator - 域判别器
3. DomainAdversarialModule - 域对抗训练模块
4. MMDLoss - 最大均值差异损失
5. CORALLoss - 相关对齐损失
6. FrequencyNormalization - 频率域归一化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math


class GradientReversalFunction(Function):
    """
    梯度反转函数
    前向传播时正常传递，反向传播时反转梯度
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """
    梯度反转层 (GRL)
    用于域对抗训练，使特征提取器学习域不变特征

    Reference:
    Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016
    """
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        """动态调整反转强度"""
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """
    域判别器
    判断特征来自哪个域（数据集/生成器）
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 1024,
        num_domains: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_domains)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, D] 特征向量
        Returns:
            domain_logits: [B, num_domains] 域预测logits
        """
        return self.discriminator(x)


class DomainAdversarialModule(nn.Module):
    """
    域对抗训练模块
    组合GRL和域判别器，实现域对抗学习
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 1024,
        num_domains: int = 2,
        dropout: float = 0.1,
        lambda_init: float = 0.0,
        lambda_max: float = 1.0,
        use_schedule: bool = True
    ):
        super().__init__()

        self.grl = GradientReversalLayer(lambda_init)
        self.discriminator = DomainDiscriminator(
            in_features=in_features,
            hidden_dim=hidden_dim,
            num_domains=num_domains,
            dropout=dropout
        )

        self.lambda_max = lambda_max
        self.use_schedule = use_schedule
        self.current_epoch = 0
        self.total_epochs = 1

    def set_epoch(self, epoch: int, total_epochs: int):
        """
        设置当前epoch，用于lambda调度
        使用逐渐增加的策略，让模型先学习判别性特征，再学习域不变特征
        """
        self.current_epoch = epoch
        self.total_epochs = total_epochs

        if self.use_schedule:
            # 使用sigmoid调度：从0逐渐增加到lambda_max
            p = epoch / total_epochs
            lambda_ = self.lambda_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1)
            self.grl.set_lambda(lambda_)

    def forward(self, features):
        """
        Args:
            features: [B, D] 特征向量
        Returns:
            domain_logits: [B, num_domains] 域预测logits
        """
        reversed_features = self.grl(features)
        domain_logits = self.discriminator(reversed_features)
        return domain_logits

    def get_current_lambda(self):
        return self.grl.lambda_


class MMDLoss(nn.Module):
    """
    最大均值差异损失 (Maximum Mean Discrepancy)
    用于对齐不同域的特征分布

    Reference:
    Long et al., "Learning Transferable Features with Deep Adaptation Networks", ICML 2015
    """
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """计算高斯核矩阵"""
        n_samples = source.size(0) + target.size(0)
        total = torch.cat([source, target], dim=0)

        # 计算L2距离
        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # 计算带宽
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        # 多核高斯
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        """
        Args:
            source: [B1, D] 源域特征
            target: [B2, D] 目标域特征
        Returns:
            mmd_loss: scalar
        """
        if source.size(0) == 0 or target.size(0) == 0:
            return torch.tensor(0.0, device=source.device)

        batch_size = min(source.size(0), target.size(0))
        source = source[:batch_size]
        target = target[:batch_size]

        kernels = self.gaussian_kernel(
            source, target,
            kernel_mul=self.kernel_mul,
            kernel_num=self.kernel_num
        )

        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]

        loss = torch.mean(XX + YY - XY - YX)
        return loss


class CORALLoss(nn.Module):
    """
    相关对齐损失 (CORrelation ALignment)
    对齐源域和目标域特征的二阶统计量（协方差矩阵）

    Reference:
    Sun et al., "Deep CORAL: Correlation Alignment for Deep Domain Adaptation", ECCV 2016
    """
    def __init__(self):
        super().__init__()

    def forward(self, source, target):
        """
        Args:
            source: [B1, D] 源域特征
            target: [B2, D] 目标域特征
        Returns:
            coral_loss: scalar
        """
        if source.size(0) <= 1 or target.size(0) <= 1:
            return torch.tensor(0.0, device=source.device)

        d = source.size(1)

        # 计算源域协方差矩阵
        source_mean = source.mean(0, keepdim=True)
        source_centered = source - source_mean
        source_cov = (source_centered.T @ source_centered) / (source.size(0) - 1)

        # 计算目标域协方差矩阵
        target_mean = target.mean(0, keepdim=True)
        target_centered = target - target_mean
        target_cov = (target_centered.T @ target_centered) / (target.size(0) - 1)

        # CORAL损失：协方差矩阵的Frobenius范数
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2 / (4 * d * d)

        return loss


class FrequencyNormalization(nn.Module):
    """
    频率域归一化
    减少生成器特定的频率模式，提升跨生成器泛化能力
    """
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # 可学习的缩放和偏移参数
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

        # 频带权重（可学习）
        self.band_weights = nn.Parameter(torch.ones(4))  # 低、中低、中高、高频

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] 频率特征图
        Returns:
            normalized: [B, C, H, W] 归一化后的特征
        """
        B, C, H, W = x.shape

        # 转换到频率域
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_mag = torch.abs(x_fft)
        x_phase = torch.angle(x_fft)

        # 创建频带掩码
        center_h, center_w = H // 2, W // 2
        y_coords = torch.arange(H, device=x.device).view(-1, 1).expand(H, W) - center_h
        x_coords = torch.arange(W, device=x.device).view(1, -1).expand(H, W) - center_w
        dist = torch.sqrt(y_coords.float() ** 2 + x_coords.float() ** 2)
        max_dist = math.sqrt(center_h ** 2 + center_w ** 2)

        # 四个频带
        band1 = (dist < max_dist * 0.25).float()  # 低频
        band2 = ((dist >= max_dist * 0.25) & (dist < max_dist * 0.5)).float()  # 中低频
        band3 = ((dist >= max_dist * 0.5) & (dist < max_dist * 0.75)).float()  # 中高频
        band4 = (dist >= max_dist * 0.75).float()  # 高频

        # 对每个频带进行归一化
        weights = F.softmax(self.band_weights, dim=0)

        # 频带加权
        band_mask = (
            weights[0] * band1 +
            weights[1] * band2 +
            weights[2] * band3 +
            weights[3] * band4
        )

        # 归一化幅度谱（按频带）
        x_mag_shifted = torch.fft.fftshift(x_mag, dim=(-2, -1))

        # 实例归一化
        mean = x_mag_shifted.mean(dim=(-2, -1), keepdim=True)
        std = x_mag_shifted.std(dim=(-2, -1), keepdim=True) + self.eps
        x_mag_norm = (x_mag_shifted - mean) / std

        # 应用频带权重
        x_mag_norm = x_mag_norm * band_mask.unsqueeze(0).unsqueeze(0)

        # 逆FFT移位
        x_mag_norm = torch.fft.ifftshift(x_mag_norm, dim=(-2, -1))

        # 重建复数频谱
        x_fft_norm = x_mag_norm * torch.exp(1j * x_phase)

        # 逆FFT回到空间域
        x_out = torch.fft.ifft2(x_fft_norm, norm='ortho').real

        # 可学习的缩放和偏移
        x_out = self.gamma * x_out + self.beta

        return x_out


class StyleNormalization(nn.Module):
    """
    风格归一化
    移除域/风格特定的信息，保留内容信息

    Reference:
    Inspired by AdaIN (Huang & Belongie, ICCV 2017)
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.num_features = num_features

        # 可学习的目标统计量（用于规范化所有输入到统一的"风格"）
        self.register_buffer('target_mean', torch.zeros(1, num_features))
        self.register_buffer('target_std', torch.ones(1, num_features))

        # 可学习的混合参数
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        """
        Args:
            x: [B, D] 特征向量
        Returns:
            normalized: [B, D] 风格归一化后的特征
        """
        # 计算实例统计量
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps

        # 归一化
        x_norm = (x - mean) / std

        # 部分风格归一化（保留部分原始信息）
        alpha = torch.sigmoid(self.alpha)
        x_out = alpha * x_norm + (1 - alpha) * x

        return x_out


class DomainAlignmentLoss(nn.Module):
    """
    综合域对齐损失
    组合多种域对齐方法
    """
    def __init__(
        self,
        use_mmd: bool = True,
        use_coral: bool = True,
        mmd_weight: float = 0.1,
        coral_weight: float = 0.1
    ):
        super().__init__()

        self.use_mmd = use_mmd
        self.use_coral = use_coral
        self.mmd_weight = mmd_weight
        self.coral_weight = coral_weight

        if use_mmd:
            self.mmd_loss = MMDLoss()
        if use_coral:
            self.coral_loss = CORALLoss()

    def forward(self, source_features, target_features):
        """
        Args:
            source_features: [B1, D] 源域特征
            target_features: [B2, D] 目标域特征
        Returns:
            loss_dict: dict containing alignment losses
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=source_features.device)

        if self.use_mmd:
            mmd = self.mmd_loss(source_features, target_features)
            loss_dict['mmd_loss'] = mmd
            total_loss = total_loss + self.mmd_weight * mmd

        if self.use_coral:
            coral = self.coral_loss(source_features, target_features)
            loss_dict['coral_loss'] = coral
            total_loss = total_loss + self.coral_weight * coral

        loss_dict['alignment_loss'] = total_loss

        return loss_dict


# 测试代码
if __name__ == "__main__":
    print("Testing domain adaptation modules...")

    batch_size = 16
    feature_dim = 1024
    num_domains = 2

    # 测试数据
    features = torch.randn(batch_size, feature_dim)
    domain_labels = torch.randint(0, num_domains, (batch_size,))

    # 测试GRL
    grl = GradientReversalLayer(lambda_=1.0)
    reversed_features = grl(features)
    print(f"GRL output shape: {reversed_features.shape}")

    # 测试域判别器
    discriminator = DomainDiscriminator(feature_dim, num_domains=num_domains)
    domain_logits = discriminator(features)
    print(f"Domain logits shape: {domain_logits.shape}")

    # 测试域对抗模块
    da_module = DomainAdversarialModule(feature_dim, num_domains=num_domains)
    da_module.set_epoch(5, 10)
    domain_pred = da_module(features)
    print(f"Domain prediction shape: {domain_pred.shape}")
    print(f"Current lambda: {da_module.get_current_lambda():.4f}")

    # 测试MMD损失
    source = torch.randn(batch_size // 2, feature_dim)
    target = torch.randn(batch_size // 2, feature_dim)
    mmd_loss = MMDLoss()
    mmd = mmd_loss(source, target)
    print(f"MMD loss: {mmd.item():.4f}")

    # 测试CORAL损失
    coral_loss = CORALLoss()
    coral = coral_loss(source, target)
    print(f"CORAL loss: {coral.item():.4f}")

    # 测试频率归一化
    freq_features = torch.randn(batch_size, 64, 14, 14)
    freq_norm = FrequencyNormalization(64)
    freq_out = freq_norm(freq_features)
    print(f"Frequency normalized shape: {freq_out.shape}")

    # 测试风格归一化
    style_norm = StyleNormalization(feature_dim)
    style_out = style_norm(features)
    print(f"Style normalized shape: {style_out.shape}")

    # 测试综合对齐损失
    align_loss = DomainAlignmentLoss()
    loss_dict = align_loss(source, target)
    print(f"Alignment losses: {loss_dict}")

    print("\nAll domain adaptation tests passed!")
