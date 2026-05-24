"""
Domain-Adversarial Frequency Learning (DAFL)
域对抗频率学习 - 改进版FDAA

核心改进：
1. 集成域对抗训练，学习域不变的频率特征
2. 使用自适应滤波器组(AFB)替代固定SRM
3. 渐进式域对抗调度，平衡判别性和泛化性
4. 多层级域对抗，在不同特征层级施加约束
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .adaptive_filter import AdaptiveFilterBank, FrequencyAwareAFB


class GradientReversalFunction(Function):
    """梯度反转函数"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """梯度反转层"""
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_


class DomainDiscriminator(nn.Module):
    """
    域判别器
    判断特征来自哪个生成器/数据集
    """
    def __init__(self, in_features, hidden_dim=512, num_domains=2, dropout=0.1):
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
        return self.discriminator(x)


class SpatialAdapterDA(nn.Module):
    """
    域感知空间Adapter
    在空间特征上施加域对抗约束
    """
    def __init__(self, dim, reduction=4, dropout=0.1):
        super().__init__()

        self.down = nn.Linear(dim, dim // reduction)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(dim // reduction, dim)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        # 初始化
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.xavier_uniform_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = self.up(self.dropout(self.act(self.down(x))))
        return x + self.scale * residual


class FrequencyAdapterDA(nn.Module):
    """
    域感知频率Adapter
    使用自适应滤波器组提取频率特征
    """
    def __init__(self, dim, img_size=224, patch_size=16, reduction=4, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        # 使用自适应滤波器组替代固定SRM
        self.afb = FrequencyAwareAFB(
            in_channels=3,
            num_learnable_filters=8,
            use_fft=True
        )

        # 频率特征投影
        self.freq_proj = nn.Sequential(
            nn.Linear(self.afb.out_channels, dim // reduction),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // reduction, dim)
        )

        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, image=None):
        """
        Args:
            x: [B, N, D] patch tokens
            image: [B, 3, H, W] 原始图像
        Returns:
            frequency-adapted features
        """
        B, N, D = x.shape

        if image is not None:
            # 使用自适应滤波器组提取频率特征
            freq_feat = self.afb(image)  # [B, C, H, W]

            # 池化到patch数量
            grid_size = int(math.sqrt(N))
            freq_feat = F.adaptive_avg_pool2d(freq_feat, (grid_size, grid_size))
            freq_feat = freq_feat.flatten(2).transpose(1, 2)  # [B, N, C]

            # 投影到目标维度
            freq_out = self.freq_proj(freq_feat)  # [B, N, D]

            return x + self.scale * self.norm(freq_out)
        else:
            return x

    def get_regularization_loss(self):
        """获取AFB正则化损失"""
        return self.afb.get_regularization_loss()


class CrossDomainInteractionDA(nn.Module):
    """
    跨域交互模块（增强版）
    增加域不变性约束
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        self.spatial_to_freq = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.freq_to_spatial = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        # 域不变投影（用于对齐不同域的特征分布）
        self.domain_invariant_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, spatial_feat, freq_feat):
        # 空间到频率的注意力
        s2f, _ = self.spatial_to_freq(freq_feat, spatial_feat, spatial_feat)
        s2f = self.norm1(freq_feat + self.dropout(s2f))

        # 频率到空间的注意力
        f2s, _ = self.freq_to_spatial(spatial_feat, freq_feat, freq_feat)
        f2s = self.norm2(spatial_feat + self.dropout(f2s))

        # 门控融合
        concat = torch.cat([s2f, f2s], dim=-1)
        gate = self.gate(concat)
        fused = gate * s2f + (1 - gate) * f2s

        # 域不变投影
        domain_inv = self.domain_invariant_proj(fused)

        return domain_inv


class FDAA_DA(nn.Module):
    """
    Domain-Adversarial Frequency-aware Dual-domain Adaptive Adapter (FDAA-DA)

    完整的域对抗频率学习模块，包含：
    1. 空间域Adapter
    2. 频率域Adapter（使用AFB）
    3. 跨域交互模块
    4. 域对抗分支（GRL + 域判别器）
    """
    def __init__(
        self,
        dim: int,
        img_size: int = 224,
        patch_size: int = 16,
        reduction: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_domains: int = 2,
        use_domain_adversarial: bool = True,
        lambda_init: float = 0.0,
        lambda_max: float = 1.0
    ):
        super().__init__()

        self.dim = dim
        self.use_domain_adversarial = use_domain_adversarial

        # 空间域Adapter
        self.spatial_adapter = SpatialAdapterDA(dim, reduction, dropout)

        # 频率域Adapter（使用AFB）
        self.freq_adapter = FrequencyAdapterDA(dim, img_size, patch_size, reduction, dropout)

        # 跨域交互
        self.cross_interaction = CrossDomainInteractionDA(dim, num_heads, dropout)

        # 输出层
        self.output_norm = nn.LayerNorm(dim)
        self.output_scale = nn.Parameter(torch.ones(1) * 0.1)

        # 域对抗分支
        if use_domain_adversarial:
            self.grl = GradientReversalLayer(lambda_init)
            self.domain_discriminator = DomainDiscriminator(
                in_features=dim,
                hidden_dim=512,
                num_domains=num_domains,
                dropout=dropout
            )
            self.lambda_max = lambda_max
            self.current_lambda = lambda_init

    def set_domain_lambda(self, epoch: int, total_epochs: int):
        """
        设置域对抗强度（渐进式调度）
        """
        if self.use_domain_adversarial:
            p = epoch / max(total_epochs, 1)
            # Sigmoid调度
            lambda_ = self.lambda_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1)
            self.grl.set_lambda(lambda_)
            self.current_lambda = lambda_

    def forward(self, x, image=None, return_domain_logits=False):
        """
        Args:
            x: [B, N, D] patch tokens
            image: [B, 3, H, W] 原始图像
            return_domain_logits: 是否返回域预测（训练时使用）
        Returns:
            output: [B, N, D] 适配后的特征
            domain_logits: [B, num_domains] 域预测（可选）
        """
        # 空间域适配
        spatial_out = self.spatial_adapter(x)

        # 频率域适配
        freq_out = self.freq_adapter(x, image)

        # 跨域交互
        fused = self.cross_interaction(spatial_out, freq_out)

        # 残差连接
        output = x + self.output_scale * self.output_norm(fused - x)

        # 域对抗分支
        if self.use_domain_adversarial and return_domain_logits:
            # 对全局特征进行域判别
            global_feat = output.mean(dim=1)  # [B, D]
            reversed_feat = self.grl(global_feat)
            domain_logits = self.domain_discriminator(reversed_feat)
            return output, domain_logits

        return output

    def get_regularization_loss(self):
        """获取所有正则化损失"""
        losses = self.freq_adapter.get_regularization_loss()
        return losses

    def get_current_lambda(self):
        if self.use_domain_adversarial:
            return self.current_lambda
        return 0.0


class MultiLevelDAFL(nn.Module):
    """
    多层级域对抗频率学习

    在多个层级施加域对抗约束，提供更强的泛化能力
    """
    def __init__(
        self,
        dim: int,
        img_size: int = 224,
        patch_size: int = 16,
        num_levels: int = 3,
        num_domains: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_levels = num_levels

        # 多个FDAA-DA模块
        self.fdaa_levels = nn.ModuleList([
            FDAA_DA(
                dim=dim,
                img_size=img_size,
                patch_size=patch_size,
                num_domains=num_domains,
                use_domain_adversarial=True,
                dropout=dropout
            ) for _ in range(num_levels)
        ])

        # 层级权重（可学习）
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

    def set_domain_lambda(self, epoch: int, total_epochs: int):
        """设置所有层级的域对抗强度"""
        for fdaa in self.fdaa_levels:
            fdaa.set_domain_lambda(epoch, total_epochs)

    def forward(self, x, image=None, return_domain_logits=False):
        """
        Args:
            x: [B, N, D] patch tokens
            image: [B, 3, H, W] 原始图像
            return_domain_logits: 是否返回域预测
        Returns:
            output: [B, N, D]
            domain_logits_list: list of [B, num_domains]（可选）
        """
        outputs = []
        domain_logits_list = []

        current_x = x
        for i, fdaa in enumerate(self.fdaa_levels):
            if return_domain_logits:
                level_out, domain_logits = fdaa(
                    current_x, image, return_domain_logits=True
                )
                domain_logits_list.append(domain_logits)
            else:
                level_out = fdaa(current_x, image, return_domain_logits=False)

            outputs.append(level_out)
            current_x = level_out

        # 加权融合多层级输出
        weights = F.softmax(self.level_weights, dim=0)
        final_output = sum(w * out for w, out in zip(weights, outputs))

        if return_domain_logits:
            return final_output, domain_logits_list

        return final_output

    def get_regularization_loss(self):
        """获取所有层级的正则化损失"""
        total_losses = {}
        for i, fdaa in enumerate(self.fdaa_levels):
            level_losses = fdaa.get_regularization_loss()
            for k, v in level_losses.items():
                total_losses[f'level{i}_{k}'] = v

        # 汇总
        total_losses['total_multi_level_reg'] = sum(
            v for k, v in total_losses.items() if 'total' not in k
        )

        return total_losses


# 测试代码
if __name__ == "__main__":
    print("Testing FDAA-DA (Domain-Adversarial Frequency Learning)...")

    batch_size = 4
    num_patches = 196
    dim = 1024
    num_domains = 3

    # 模拟输入
    x = torch.randn(batch_size, num_patches, dim)
    image = torch.randn(batch_size, 3, 224, 224)
    domain_labels = torch.randint(0, num_domains, (batch_size,))

    # 测试FDAA-DA
    fdaa_da = FDAA_DA(
        dim=dim,
        img_size=224,
        patch_size=16,
        num_domains=num_domains,
        use_domain_adversarial=True
    )

    # 设置训练epoch
    fdaa_da.set_domain_lambda(epoch=5, total_epochs=10)
    print(f"Current lambda: {fdaa_da.get_current_lambda():.4f}")

    # 前向传播（带域预测）
    output, domain_logits = fdaa_da(x, image, return_domain_logits=True)
    print(f"\nFDAA-DA input shape: {x.shape}")
    print(f"FDAA-DA output shape: {output.shape}")
    print(f"Domain logits shape: {domain_logits.shape}")

    # 正则化损失
    reg_losses = fdaa_da.get_regularization_loss()
    print(f"Regularization losses: {reg_losses}")

    # 测试多层级DAFL
    print("\n" + "="*50)
    print("Testing Multi-Level DAFL...")

    ml_dafl = MultiLevelDAFL(
        dim=dim,
        img_size=224,
        patch_size=16,
        num_levels=3,
        num_domains=num_domains
    )

    ml_dafl.set_domain_lambda(epoch=5, total_epochs=10)
    ml_output, ml_domain_logits = ml_dafl(x, image, return_domain_logits=True)

    print(f"Multi-Level output shape: {ml_output.shape}")
    print(f"Number of domain logits levels: {len(ml_domain_logits)}")

    # 参数统计
    fdaa_da_params = sum(p.numel() for p in fdaa_da.parameters())
    ml_dafl_params = sum(p.numel() for p in ml_dafl.parameters())
    print(f"\nFDAA-DA parameters: {fdaa_da_params:,}")
    print(f"Multi-Level DAFL parameters: {ml_dafl_params:,}")

    print("\nAll FDAA-DA tests passed!")
