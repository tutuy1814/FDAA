"""
消融实验专用检测器
支持各模块的独立开关控制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .modules.fdaa import FDAA, SpatialAdapter, FrequencyAdapter, CrossDomainInteraction
from .modules.mgfp import MGFP, PatchForgeryAttention, MultiGranularityAggregation


class AIGCDetectorAblation(nn.Module):
    """
    消融实验检测器

    支持以下配置:
    - use_fdaa_spatial: 是否使用空间域Adapter
    - use_fdaa_freq: 是否使用频率域Adapter
    - use_fdaa_cross: 是否使用跨域交互
    - use_mgfp: 是否使用多粒度伪造感知
    - pretrained: 是否使用预训练ViT-B/16 backbone (冻结)
    """
    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dim: int = 768,
        num_prototypes: int = 4,
        dropout: float = 0.1,
        pretrained: bool = False,
        # 消融开关
        use_fdaa_spatial: bool = True,
        use_fdaa_freq: bool = True,
        use_fdaa_cross: bool = True,
        use_mgfp: bool = True,
    ):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2
        self.pretrained = pretrained

        # 消融配置
        self.use_fdaa_spatial = use_fdaa_spatial
        self.use_fdaa_freq = use_fdaa_freq
        self.use_fdaa_cross = use_fdaa_cross
        self.use_mgfp = use_mgfp

        if pretrained:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.patch_embed = vit.conv_proj
            self.cls_token = vit.class_token
            self.backbone_encoder = vit.encoder
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.cls_token.requires_grad = False
            for param in self.backbone_encoder.parameters():
                param.requires_grad = False
        else:
            self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
                dropout=dropout, activation='gelu', batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # FDAA组件 (可选)
        if use_fdaa_spatial:
            self.spatial_adapter = SpatialAdapter(embed_dim, reduction=4, dropout=dropout)

        if use_fdaa_freq:
            self.freq_adapter = FrequencyAdapter(embed_dim, img_size, self.patch_size, reduction=4, dropout=dropout)

        if use_fdaa_cross and use_fdaa_spatial and use_fdaa_freq:
            self.cross_interaction = CrossDomainInteraction(embed_dim, num_heads=8, dropout=dropout)
            self.fdaa_norm = nn.LayerNorm(embed_dim)
            self.fdaa_scale = nn.Parameter(torch.ones(1) * 0.1)

        # MGFP模块 (可选)
        if use_mgfp:
            self.mgfp = MGFP(
                dim=embed_dim,
                num_patches=self.num_patches,
                num_prototypes=num_prototypes,
                use_hierarchical=False,
                dropout=dropout
            )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        if not pretrained:
            self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, image, return_attention=False):
        B = image.shape[0]

        # Patch embedding
        x = self.patch_embed(image)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Transformer encoder
        if self.pretrained:
            x = self.backbone_encoder(x)
        else:
            x = x + self.pos_embed
            x = self.encoder(x)

        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        # FDAA处理
        if self.use_fdaa_spatial and self.use_fdaa_freq and self.use_fdaa_cross:
            # 完整FDAA
            spatial_out = self.spatial_adapter(patch_tokens)
            freq_out = self.freq_adapter(patch_tokens, image)
            fused = self.cross_interaction(spatial_out, freq_out)
            patch_tokens = patch_tokens + self.fdaa_scale * self.fdaa_norm(fused - patch_tokens)
        elif self.use_fdaa_spatial and self.use_fdaa_freq:
            # 无跨域交互
            spatial_out = self.spatial_adapter(patch_tokens)
            freq_out = self.freq_adapter(patch_tokens, image)
            patch_tokens = (spatial_out + freq_out) / 2
        elif self.use_fdaa_spatial:
            # 仅空间域
            patch_tokens = self.spatial_adapter(patch_tokens)
        elif self.use_fdaa_freq:
            # 仅频率域
            patch_tokens = self.freq_adapter(patch_tokens, image)

        # MGFP处理
        if self.use_mgfp:
            output, aux_outputs = self.mgfp(cls_token, patch_tokens, return_attention=True)
        else:
            # 不使用MGFP，直接用CLS token
            output = cls_token
            aux_outputs = {'forgery_map': torch.zeros(B, self.num_patches, device=image.device)}

        # 分类
        logits = self.classifier(output)

        outputs = {'logits': logits, 'cls': logits}

        if return_attention:
            outputs['forgery_map'] = aux_outputs.get('forgery_map', None)

        return outputs

    def get_config_name(self):
        """获取配置名称"""
        parts = []
        if self.use_fdaa_spatial:
            parts.append("S")
        if self.use_fdaa_freq:
            parts.append("F")
        if self.use_fdaa_cross:
            parts.append("C")
        if self.use_mgfp:
            parts.append("M")

        if not parts:
            return "Baseline"
        return "FDAA(" + "+".join(parts[:3]) + ")" + ("+MGFP" if self.use_mgfp else "")


# 消融实验配置
ABLATION_CONFIGS = {
    'baseline': {
        'use_fdaa_spatial': False,
        'use_fdaa_freq': False,
        'use_fdaa_cross': False,
        'use_mgfp': False,
        'name': 'Baseline (ViT only)'
    },
    'fdaa_spatial': {
        'use_fdaa_spatial': True,
        'use_fdaa_freq': False,
        'use_fdaa_cross': False,
        'use_mgfp': False,
        'name': 'FDAA (Spatial only)'
    },
    'fdaa_freq': {
        'use_fdaa_spatial': False,
        'use_fdaa_freq': True,
        'use_fdaa_cross': False,
        'use_mgfp': False,
        'name': 'FDAA (Frequency only)'
    },
    'fdaa_dual': {
        'use_fdaa_spatial': True,
        'use_fdaa_freq': True,
        'use_fdaa_cross': False,
        'use_mgfp': False,
        'name': 'FDAA (Dual-domain, no cross)'
    },
    'fdaa_full': {
        'use_fdaa_spatial': True,
        'use_fdaa_freq': True,
        'use_fdaa_cross': True,
        'use_mgfp': False,
        'name': 'FDAA (Full)'
    },
    'mgfp_only': {
        'use_fdaa_spatial': False,
        'use_fdaa_freq': False,
        'use_fdaa_cross': False,
        'use_mgfp': True,
        'name': 'MGFP only'
    },
    'full': {
        'use_fdaa_spatial': True,
        'use_fdaa_freq': True,
        'use_fdaa_cross': True,
        'use_mgfp': True,
        'name': 'Full Model (FDAA + MGFP)'
    }
}


def create_ablation_model(config_name: str, **kwargs) -> AIGCDetectorAblation:
    """根据配置名创建消融模型"""
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(ABLATION_CONFIGS.keys())}")

    config = ABLATION_CONFIGS[config_name].copy()
    config.pop('name')  # 移除name字段
    config.update(kwargs)  # 允许覆盖其他参数

    return AIGCDetectorAblation(**config)


if __name__ == "__main__":
    print("测试消融模型配置...")

    for config_name, config in ABLATION_CONFIGS.items():
        model = create_ablation_model(config_name)
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n{config['name']}:")
        print(f"  参数量: {params:,}")
        print(f"  可训练: {trainable:,}")

        # 测试前向传播
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        print(f"  输出shape: {out['logits'].shape}")
