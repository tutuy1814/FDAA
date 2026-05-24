
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .modules.fdaa import FDAA
from .modules.mgfp import MGFP
from .modules.domain_adaptation import (
    DomainAdversarialModule,
    MMDLoss,
    CORALLoss,
    FrequencyNormalization,
    StyleNormalization,
    DomainAlignmentLoss
)


class AIGCDetectorGeneralized(nn.Module):
    """
    泛化增强版AI生成图像检测器

    在原有架构基础上增加:
    1. 域对抗模块 - 学习域不变特征
    2. 频率归一化 - 减少生成器特定频率模式
    3. 风格归一化 - 移除域特定风格信息
    4. 特征对齐损失 - 对齐不同域的特征分布

    结构:
    Input Image → CLIP ViT Backbone → FDAA (+FreqNorm) → MGFP (+StyleNorm) →
    → Domain Adversarial → Classifier
    """
    def __init__(
        self,
        backbone_name: str = "ViT-L/14",
        num_classes: int = 2,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 1024,
        num_adapter_layers: int = 3,
        num_prototypes: int = 4,
        use_hierarchical: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        # 新增泛化参数
        use_domain_adversarial: bool = True,
        num_domains: int = 2,
        use_freq_norm: bool = True,
        use_style_norm: bool = True,
        use_feature_alignment: bool = True,
        domain_adv_weight: float = 0.1,
        alignment_weight: float = 0.1
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # 泛化配置
        self.use_domain_adversarial = use_domain_adversarial
        self.use_freq_norm = use_freq_norm
        self.use_style_norm = use_style_norm
        self.use_feature_alignment = use_feature_alignment
        self.domain_adv_weight = domain_adv_weight
        self.alignment_weight = alignment_weight

        # 加载CLIP backbone
        self.backbone = self._load_backbone(backbone_name, freeze_backbone)

        # FDAA模块
        self.fdaa_modules = nn.ModuleList([
            FDAA(
                dim=embed_dim,
                img_size=img_size,
                patch_size=patch_size,
                reduction=4,
                num_heads=8,
                dropout=dropout
            ) for _ in range(num_adapter_layers)
        ])

        # 频率归一化（应用于FDAA输出）
        if use_freq_norm:
            self.freq_norm = FrequencyNormalization(embed_dim)

        # MGFP模块
        self.mgfp = MGFP(
            dim=embed_dim,
            num_patches=self.num_patches,
            num_prototypes=num_prototypes,
            use_hierarchical=use_hierarchical,
            dropout=dropout
        )

        # 风格归一化（应用于MGFP输出）
        if use_style_norm:
            self.style_norm = StyleNormalization(embed_dim)

        # 域对抗模块
        if use_domain_adversarial:
            self.domain_adversarial = DomainAdversarialModule(
                in_features=embed_dim,
                hidden_dim=embed_dim // 2,
                num_domains=num_domains,
                dropout=dropout,
                lambda_init=0.0,
                lambda_max=1.0,
                use_schedule=True
            )

        # 特征对齐损失
        if use_feature_alignment:
            self.alignment_loss = DomainAlignmentLoss(
                use_mmd=True,
                use_coral=True,
                mmd_weight=alignment_weight / 2,
                coral_weight=alignment_weight / 2
            )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # 辅助分类头
        if use_hierarchical:
            self.aux_classifier = nn.Linear(1, num_classes)

        self.use_hierarchical = use_hierarchical

        # 域损失函数
        self.domain_criterion = nn.CrossEntropyLoss()

    def _load_backbone(self, backbone_name: str, freeze: bool = True):
        """加载CLIP backbone"""
        try:
            import clip
            model, _ = clip.load(backbone_name, device='cpu')
            backbone = model.visual

            if freeze:
                for param in backbone.parameters():
                    param.requires_grad = False

            return backbone
        except ImportError:
            print("Warning: CLIP not installed. Using dummy backbone.")
            return self._create_dummy_backbone()

    def _create_dummy_backbone(self):
        """创建测试用的dummy backbone"""
        class DummyViT(nn.Module):
            def __init__(self, embed_dim, num_patches):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_patches = num_patches
                self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=14, stride=14)
                self.class_embedding = nn.Parameter(torch.randn(embed_dim))
                self.positional_embedding = nn.Parameter(torch.randn(num_patches + 1, embed_dim))
                self.ln_pre = nn.LayerNorm(embed_dim)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
                    num_layers=12
                )
                self.ln_post = nn.LayerNorm(embed_dim)

            def forward(self, x, return_all_tokens=False):
                x = self.conv1(x)
                x = x.flatten(2).transpose(1, 2)
                cls_token = self.class_embedding.unsqueeze(0).unsqueeze(0).expand(x.shape[0], -1, -1)
                x = torch.cat([cls_token, x], dim=1)
                x = x + self.positional_embedding
                x = self.ln_pre(x)
                x = self.transformer(x)
                x = self.ln_post(x)

                if return_all_tokens:
                    return x
                return x[:, 0]

        return DummyViT(self.embed_dim, self.num_patches)

    def set_epoch(self, epoch: int, total_epochs: int):
        """设置当前epoch，用于域对抗lambda调度"""
        if self.use_domain_adversarial:
            self.domain_adversarial.set_epoch(epoch, total_epochs)

    def forward_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """特征提取"""
        # 获取backbone特征
        try:
            all_tokens = self.backbone(image, return_all_tokens=True)
            cls_token = all_tokens[:, 0]
            patch_tokens = all_tokens[:, 1:]
        except TypeError:
            features = self.backbone(image)
            if features.dim() == 2:
                cls_token = features
                B = image.shape[0]
                patch_tokens = torch.randn(B, self.num_patches, self.embed_dim, device=image.device)
            else:
                cls_token = features[:, 0]
                patch_tokens = features[:, 1:]

        # 应用FDAA模块
        for fdaa in self.fdaa_modules:
            patch_tokens = fdaa(patch_tokens, image)

        # 频率归一化（在特征空间应用）
        if self.use_freq_norm:
            B, N, D = patch_tokens.shape
            H = W = int(N ** 0.5)
            # 重塑为2D特征图
            patch_features_2d = patch_tokens.transpose(1, 2).view(B, D, H, W)
            patch_features_2d = self.freq_norm(patch_features_2d)
            patch_tokens = patch_features_2d.view(B, D, -1).transpose(1, 2)

        return cls_token, patch_tokens

    def forward(
        self,
        image: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_attention: bool = False,
        compute_domain_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            image: [B, 3, H, W] 输入图像
            domain_labels: [B] 域标签 (0, 1, ..., num_domains-1)
            return_features: 是否返回中间特征
            return_attention: 是否返回注意力图
            compute_domain_loss: 是否计算域对抗损失

        Returns:
            outputs: dict
        """
        outputs = {}

        # 1. 特征提取 + FDAA + 频率归一化
        cls_token, patch_tokens = self.forward_features(image)

        # 2. MGFP模块
        mgfp_output, aux_outputs = self.mgfp(
            cls_token, patch_tokens, return_attention=True
        )

        # 3. 风格归一化
        if self.use_style_norm:
            mgfp_output = self.style_norm(mgfp_output)

        # 保存特征用于对齐
        outputs['features'] = mgfp_output

        # 4. 域对抗
        if self.use_domain_adversarial and compute_domain_loss:
            domain_logits = self.domain_adversarial(mgfp_output)
            outputs['domain_logits'] = domain_logits

            if domain_labels is not None:
                domain_loss = self.domain_criterion(domain_logits, domain_labels)
                outputs['domain_loss'] = domain_loss * self.domain_adv_weight

        # 5. 主分类
        logits = self.classifier(mgfp_output)
        outputs['logits'] = logits
        outputs['cls'] = logits

        # 6. 辅助分类
        if self.use_hierarchical and 'hierarchical_score' in aux_outputs:
            aux_logits = self.aux_classifier(aux_outputs['hierarchical_score'])
            outputs['aux_logits'] = aux_logits

        # 可选输出
        if return_features:
            outputs['features'] = mgfp_output

        if return_attention:
            outputs['forgery_map'] = aux_outputs['forgery_map']
            outputs['granularity_weights'] = aux_outputs['granularity_weights']
            if 'scale_maps' in aux_outputs:
                outputs['scale_maps'] = aux_outputs['scale_maps']

        return outputs

    def compute_alignment_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算特征对齐损失

        Args:
            source_features: [B1, D] 源域特征
            target_features: [B2, D] 目标域特征

        Returns:
            loss_dict: 对齐损失字典
        """
        if self.use_feature_alignment:
            return self.alignment_loss(source_features, target_features)
        return {'alignment_loss': torch.tensor(0.0, device=source_features.device)}

    def get_trainable_params(self):
        """获取可训练参数"""
        params = []

        # FDAA参数
        for fdaa in self.fdaa_modules:
            params.extend(fdaa.parameters())

        # MGFP参数
        params.extend(self.mgfp.parameters())

        # 分类器参数
        params.extend(self.classifier.parameters())

        if self.use_hierarchical:
            params.extend(self.aux_classifier.parameters())

        # 泛化模块参数
        if self.use_freq_norm:
            params.extend(self.freq_norm.parameters())

        if self.use_style_norm:
            params.extend(self.style_norm.parameters())

        if self.use_domain_adversarial:
            params.extend(self.domain_adversarial.parameters())

        return params

    def count_parameters(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {frozen:,}")

        return {'total': total, 'trainable': trainable, 'frozen': frozen}


class AIGCDetectorGeneralizedLite(nn.Module):
    """
    轻量版泛化增强检测器 (不依赖CLIP)
    """
    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dim: int = 768,
        num_prototypes: int = 4,
        dropout: float = 0.1,
        # 泛化参数
        use_domain_adversarial: bool = True,
        num_domains: int = 2,
        use_freq_norm: bool = True,
        use_style_norm: bool = True,
        domain_adv_weight: float = 0.1
    ):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2

        # 泛化配置
        self.use_domain_adversarial = use_domain_adversarial
        self.use_freq_norm = use_freq_norm
        self.use_style_norm = use_style_norm
        self.domain_adv_weight = domain_adv_weight

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        # FDAA模块
        self.fdaa = FDAA(
            dim=embed_dim,
            img_size=img_size,
            patch_size=self.patch_size,
            reduction=4,
            num_heads=8,
            dropout=dropout
        )

        # 频率归一化
        if use_freq_norm:
            self.freq_norm = FrequencyNormalization(embed_dim)

        # MGFP模块
        self.mgfp = MGFP(
            dim=embed_dim,
            num_patches=self.num_patches,
            num_prototypes=num_prototypes,
            use_hierarchical=False,
            dropout=dropout
        )

        # 风格归一化
        if use_style_norm:
            self.style_norm = StyleNormalization(embed_dim)

        # 域对抗模块
        if use_domain_adversarial:
            self.domain_adversarial = DomainAdversarialModule(
                in_features=embed_dim,
                hidden_dim=embed_dim // 2,
                num_domains=num_domains,
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

        # 域损失函数
        self.domain_criterion = nn.CrossEntropyLoss()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def set_epoch(self, epoch: int, total_epochs: int):
        """设置当前epoch"""
        if self.use_domain_adversarial:
            self.domain_adversarial.set_epoch(epoch, total_epochs)

    def forward(
        self,
        image: torch.Tensor,
        domain_labels: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        compute_domain_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        B = image.shape[0]
        outputs = {}

        # Patch embedding
        x = self.patch_embed(image)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)

        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        # FDAA
        patch_tokens = self.fdaa(patch_tokens, image)

        # 频率归一化
        if self.use_freq_norm:
            N, D = patch_tokens.shape[1], patch_tokens.shape[2]
            H = W = int(N ** 0.5)
            patch_features_2d = patch_tokens.transpose(1, 2).view(B, D, H, W)
            patch_features_2d = self.freq_norm(patch_features_2d)
            patch_tokens = patch_features_2d.view(B, D, -1).transpose(1, 2)

        # MGFP
        output, aux_outputs = self.mgfp(cls_token, patch_tokens, return_attention=True)

        # 风格归一化
        if self.use_style_norm:
            output = self.style_norm(output)

        outputs['features'] = output

        # 域对抗
        if self.use_domain_adversarial and compute_domain_loss:
            domain_logits = self.domain_adversarial(output)
            outputs['domain_logits'] = domain_logits

            if domain_labels is not None:
                domain_loss = self.domain_criterion(domain_logits, domain_labels)
                outputs['domain_loss'] = domain_loss * self.domain_adv_weight

        # Classification
        logits = self.classifier(output)
        outputs['logits'] = logits
        outputs['cls'] = logits

        if return_attention:
            outputs['forgery_map'] = aux_outputs['forgery_map']

        return outputs


def create_generalized_model(
    model_type: str = 'full',
    num_classes: int = 2,
    img_size: int = 224,
    num_domains: int = 2,
    **kwargs
) -> nn.Module:
    """
    创建泛化增强模型

    Args:
        model_type: 'full' (使用CLIP) 或 'lite' (不使用CLIP)
        num_classes: 类别数
        img_size: 图像大小
        num_domains: 域数量
        **kwargs: 其他参数

    Returns:
        model: 泛化增强检测器
    """
    if model_type == 'lite':
        return AIGCDetectorGeneralizedLite(
            num_classes=num_classes,
            img_size=img_size,
            num_domains=num_domains,
            **kwargs
        )
    else:
        return AIGCDetectorGeneralized(
            num_classes=num_classes,
            img_size=img_size,
            num_domains=num_domains,
            **kwargs
        )


# 测试代码
if __name__ == "__main__":
    print("Testing AIGCDetectorGeneralizedLite...")

    # 创建模型
    model = AIGCDetectorGeneralizedLite(
        num_classes=2,
        img_size=224,
        embed_dim=768,
        num_prototypes=4,
        use_domain_adversarial=True,
        num_domains=2,
        use_freq_norm=True,
        use_style_norm=True
    )

    # 设置epoch（用于domain adversarial调度）
    model.set_epoch(5, 10)

    # 模拟输入
    batch_size = 8
    image = torch.randn(batch_size, 3, 224, 224)
    domain_labels = torch.randint(0, 2, (batch_size,))

    # 前向传播
    outputs = model(image, domain_labels=domain_labels, return_attention=True)

    print(f"\nInput shape: {image.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Features shape: {outputs['features'].shape}")
    print(f"Domain logits shape: {outputs['domain_logits'].shape}")
    print(f"Domain loss: {outputs['domain_loss'].item():.4f}")
    print(f"Forgery map shape: {outputs['forgery_map'].shape}")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nTest passed!")
