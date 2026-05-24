"""
Generator-Aware Feature Alignment Module
生成器感知特征对齐模块 - 核心创新点

通过学习生成器特定的特征重加权，实现跨域泛化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


class GradientReversalFunction(torch.autograd.Function):
    """梯度反转层"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        return grads.neg() * ctx.lambda_, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class GeneratorAwareAlignment(nn.Module):
    """
    生成器感知特征对齐模块

    核心思想：
    1. 为每个生成器类型学习特定的特征重加权
    2. 通过辅助任务学习生成器不变的伪造特征
    3. 使用域对抗训练促进特征的跨域泛化
    """
    def __init__(
        self,
        dim: int = 1024,
        num_generators: int = 8,  # 支持多种生成器类型
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_generators = num_generators

        # 生成器特定的特征调制参数
        self.generator_scale = nn.Parameter(torch.ones(num_generators, dim))
        self.generator_shift = nn.Parameter(torch.zeros(num_generators, dim))

        # 生成器分类器（辅助任务）
        self.generator_classifier = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_generators)
        )

        # 域对抗分支 - 梯度反转促进域不变性
        self.domain_adversarial = nn.Sequential(
            GradientReversal(lambda_=1.0),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_generators)
        )

        # 通用伪造特征提取器
        self.universal_forgery_extractor = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

        # 特征融合门控
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.ones_(self.generator_scale)
        nn.init.zeros_(self.generator_shift)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: torch.Tensor,
        generator_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, D] 输入特征
            generator_labels: [B] 生成器标签 (训练时使用)
        Returns:
            dict containing:
                - aligned_features: [B, D] 对齐后的特征
                - generator_logits: [B, num_generators] 生成器分类logits
                - domain_logits: [B, num_generators] 域对抗logits
        """
        B, D = features.shape
        outputs = {}

        # 1. 提取通用伪造特征
        universal_features = self.universal_forgery_extractor(features)

        # 2. 生成器分类（辅助任务）
        generator_logits = self.generator_classifier(features.detach())
        outputs['generator_logits'] = generator_logits

        # 3. 域对抗分支
        domain_logits = self.domain_adversarial(features)
        outputs['domain_logits'] = domain_logits

        # 4. 特征调制
        if generator_labels is not None:
            # 训练时使用真实标签
            scale = self.generator_scale[generator_labels]  # [B, D]
            shift = self.generator_shift[generator_labels]  # [B, D]
        else:
            # 推理时使用预测的生成器类型
            with torch.no_grad():
                pred_gen = generator_logits.argmax(dim=-1)
            scale = self.generator_scale[pred_gen]
            shift = self.generator_shift[pred_gen]

        modulated_features = features * scale + shift

        # 5. 门控融合：通用特征 + 调制特征
        gate = self.fusion_gate(torch.cat([universal_features, modulated_features], dim=-1))
        aligned_features = gate * universal_features + (1 - gate) * modulated_features

        outputs['aligned_features'] = aligned_features
        outputs['universal_features'] = universal_features

        return outputs


class MultiHeadPrototypeContrastive(nn.Module):
    """
    多头原型对比学习模块

    改进点：
    1. 8个原型代替4个，增加覆盖范围
    2. 4头设计，捕获不同侧面的伪造特征
    3. 显式对比损失，推开真实样本，拉近伪造样本
    """
    def __init__(
        self,
        dim: int = 1024,
        num_prototypes: int = 8,
        num_heads: int = 4,
        temperature: float = 0.07,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_prototypes = num_prototypes
        self.num_heads = num_heads
        self.prototypes_per_head = num_prototypes // num_heads
        self.temperature = temperature

        # 多头原型 [num_heads, prototypes_per_head, dim]
        self.prototypes = nn.Parameter(
            torch.randn(num_heads, self.prototypes_per_head, dim) * 0.02
        )

        # 头部注意力权重
        self.head_attention = nn.Sequential(
            nn.Linear(dim, num_heads),
            nn.Softmax(dim=-1)
        )

        # 原型投影
        self.prototype_proj = nn.Linear(dim, dim)
        self.feature_proj = nn.Linear(dim, dim)

        # 输出融合
        self.output_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self._init_prototypes()

    def _init_prototypes(self):
        nn.init.xavier_uniform_(self.prototypes)

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, D] 输入特征
            labels: [B] 真/假标签 (0=real, 1=fake)
        Returns:
            dict containing:
                - enhanced_features: [B, D] 增强后的特征
                - prototype_similarity: [B, num_prototypes] 与原型的相似度
                - contrastive_loss: scalar 对比损失 (如果提供labels)
        """
        B, D = features.shape
        outputs = {}

        # 投影特征
        proj_features = self.feature_proj(features)  # [B, D]
        proj_features = F.normalize(proj_features, dim=-1)

        # 投影原型
        proj_prototypes = self.prototype_proj(
            self.prototypes.view(-1, D)
        ).view(self.num_heads, self.prototypes_per_head, D)  # [H, P, D]
        proj_prototypes = F.normalize(proj_prototypes, dim=-1)

        # 计算头部权重
        head_weights = self.head_attention(features)  # [B, num_heads]

        # 计算每个头的原型相似度
        all_similarities = []
        weighted_prototypes = []

        for h in range(self.num_heads):
            # 当前头的原型 [P, D]
            head_protos = proj_prototypes[h]

            # 计算相似度 [B, P]
            sim = torch.matmul(proj_features, head_protos.t()) / self.temperature
            all_similarities.append(sim)

            # 加权原型特征
            attn = F.softmax(sim, dim=-1)  # [B, P]
            weighted_proto = torch.matmul(attn, head_protos)  # [B, D]
            weighted_prototypes.append(weighted_proto * head_weights[:, h:h+1])

        # 合并所有头
        prototype_similarity = torch.cat(all_similarities, dim=-1)  # [B, num_prototypes]
        outputs['prototype_similarity'] = prototype_similarity

        # 融合加权原型特征
        aggregated_proto = sum(weighted_prototypes)  # [B, D]

        # 输出融合
        enhanced_features = self.output_fusion(
            torch.cat([features, aggregated_proto], dim=-1)
        )
        outputs['enhanced_features'] = enhanced_features

        # 计算对比损失（如果有标签）
        if labels is not None:
            contrastive_loss = self._compute_contrastive_loss(
                proj_features, proj_prototypes, labels
            )
            outputs['contrastive_loss'] = contrastive_loss

        return outputs

    def _compute_contrastive_loss(
        self,
        features: torch.Tensor,
        prototypes: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对比损失
        - Fake样本应该靠近伪造原型
        - Real样本应该远离伪造原型
        """
        B = features.shape[0]

        # 展平原型 [num_prototypes, D]
        flat_prototypes = prototypes.view(-1, self.dim)

        # 计算与所有原型的距离 [B, num_prototypes]
        distances = torch.cdist(features, flat_prototypes)

        # 最近原型距离
        min_distances = distances.min(dim=-1)[0]  # [B]

        # 对比损失
        # Fake (label=1): 最小化距离
        # Real (label=0): 最大化距离 (使用margin)
        margin = 1.0
        fake_mask = labels.float()
        real_mask = 1 - fake_mask

        fake_loss = (min_distances * fake_mask).sum() / (fake_mask.sum() + 1e-6)
        real_loss = (F.relu(margin - min_distances) * real_mask).sum() / (real_mask.sum() + 1e-6)

        return fake_loss + real_loss


class AdaptiveFrequencyFilterBank(nn.Module):
    """
    自适应频率滤波器组 (AFB)

    替代固定的SRM滤波器，学习适应不同生成器的频率特征
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_filters: int = 32,
        filter_size: int = 5,
        num_groups: int = 4
    ):
        super().__init__()
        self.num_filters = num_filters
        self.num_groups = num_groups
        self.filters_per_group = num_filters // num_groups

        # 可学习的滤波器组
        self.learnable_filters = nn.Parameter(
            torch.randn(num_filters, in_channels, filter_size, filter_size) * 0.01
        )

        # 频率中心参数 (用于初始化不同频率响应)
        self.freq_centers = nn.Parameter(
            torch.linspace(0.1, 0.9, num_groups).unsqueeze(1).expand(-1, self.filters_per_group).reshape(-1)
        )

        # 滤波器选择网络 (根据输入自适应选择滤波器权重)
        self.filter_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_filters),
            nn.Softmax(dim=-1)
        )

        # 输出融合
        self.output_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        self._init_filters()

    def _init_filters(self):
        """初始化滤波器为不同频率响应"""
        with torch.no_grad():
            for g in range(self.num_groups):
                # 每组滤波器对应不同频率范围
                freq = (g + 1) / self.num_groups
                for f in range(self.filters_per_group):
                    idx = g * self.filters_per_group + f
                    # 创建带通滤波器的近似
                    self._init_bandpass_filter(self.learnable_filters[idx], freq)

    def _init_bandpass_filter(self, filter_tensor, freq):
        """初始化为近似带通滤波器"""
        size = filter_tensor.shape[-1]
        center = size // 2
        for i in range(size):
            for j in range(size):
                dist = ((i - center) ** 2 + (j - center) ** 2) ** 0.5
                # 高斯带通响应
                response = torch.exp(-((dist / size - freq) ** 2) / 0.1)
                filter_tensor[:, i, j] = response * torch.randn_like(filter_tensor[:, i, j]) * 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] 输入图像
        Returns:
            [B, num_filters, H, W] 频率特征
        """
        B, C, H, W = x.shape

        # 计算自适应滤波器权重
        filter_weights = self.filter_selector(x)  # [B, num_filters]

        # 应用所有滤波器
        all_responses = F.conv2d(x, self.learnable_filters, padding=2)  # [B, num_filters, H, W]

        # 加权融合
        weighted_responses = all_responses * filter_weights.unsqueeze(-1).unsqueeze(-1)

        # 输出
        output = self.output_conv(weighted_responses)

        return output


class CrossGeneratorConsistency(nn.Module):
    """
    跨生成器一致性模块

    确保不同生成器的伪造特征映射到相似的表示空间
    """
    def __init__(self, dim: int = 1024, num_generators: int = 8):
        super().__init__()
        self.dim = dim
        self.num_generators = num_generators

        # 共享的规范化层
        self.shared_norm = nn.LayerNorm(dim)

        # 生成器特定的映射
        self.generator_maps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ) for _ in range(num_generators)
        ])

        # 一致性投影头
        self.consistency_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4)
        )

    def forward(
        self,
        features: torch.Tensor,
        generator_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, D]
            generator_labels: [B]
        Returns:
            dict with mapped features and consistency embeddings
        """
        B = features.shape[0]
        outputs = {}

        # 规范化
        norm_features = self.shared_norm(features)

        # 应用生成器特定映射
        mapped_features = torch.zeros_like(features)
        for g in range(self.num_generators):
            mask = (generator_labels == g)
            if mask.any():
                mapped_features[mask] = self.generator_maps[g](norm_features[mask])

        outputs['mapped_features'] = mapped_features

        # 一致性嵌入 (用于一致性损失)
        consistency_embed = self.consistency_head(mapped_features)
        outputs['consistency_embed'] = F.normalize(consistency_embed, dim=-1)

        return outputs

    def compute_consistency_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        generator_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算跨生成器一致性损失

        同类样本（都是fake或都是real）应该有相似的一致性嵌入
        """
        B = embeddings.shape[0]

        # 计算相似度矩阵
        sim_matrix = torch.matmul(embeddings, embeddings.t())  # [B, B]

        # 创建正样本mask：同类标签
        label_match = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # 排除自身
        eye_mask = 1 - torch.eye(B, device=embeddings.device)
        positive_mask = label_match * eye_mask

        # InfoNCE风格损失
        temperature = 0.1
        exp_sim = torch.exp(sim_matrix / temperature)

        # 正样本相似度
        pos_sim = (exp_sim * positive_mask).sum(dim=-1) / (positive_mask.sum(dim=-1) + 1e-6)

        # 所有样本相似度
        all_sim = (exp_sim * eye_mask).sum(dim=-1)

        # 损失
        loss = -torch.log(pos_sim / (all_sim + 1e-6) + 1e-6).mean()

        return loss
