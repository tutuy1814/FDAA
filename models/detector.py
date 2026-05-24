
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .modules.fdaa import FDAA, FDAAv2, FDAAv3
from .modules.mgfp import MGFP, MGFPv2, MGFPv3


def _extract_clip_tokens(visual, image):
    """
    正确提取 CLIP Visual Transformer 的 CLS token 和 patch tokens。

    手动执行 CLIP Visual 的 forward，避免调用不支持的参数。
    CLIP ViT-L/14: width=1024, output_dim=768, patch_size=14, num_patches=256 (16x16)

    Args:
        visual: CLIP visual model (model.visual)
        image: [B, 3, 224, 224]
    Returns:
        cls_token: [B, width] (1024 for ViT-L/14)
        patch_tokens: [B, N, width] (1024 for ViT-L/14)
    """
    # Patch embedding
    x = visual.conv1(image)                            # [B, width, grid, grid]
    x = x.reshape(x.shape[0], x.shape[1], -1)          # [B, width, grid**2]
    x = x.permute(0, 2, 1)                              # [B, grid**2, width]

    # Prepend CLS token
    cls_embed = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_embed, x], dim=1)                # [B, grid**2+1, width]
    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    # Transformer (CLIP uses LND format)
    x = x.permute(1, 0, 2)                              # NLD -> LND
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)                              # LND -> NLD

    # Split CLS and patch tokens BEFORE ln_post
    cls_token = x[:, 0, :]                               # [B, width]
    patch_tokens = x[:, 1:, :]                           # [B, N, width]

    # ln_post 仅用于 CLS token（其统计量在 CLIP 预训练时是为 CLS 训练的）
    cls_token = visual.ln_post(cls_token)
    # patch tokens 不经过 ln_post，由调用方独立归一化

    return cls_token, patch_tokens


# =============================================================================
# V1 检测器 (保留兼容性)
# =============================================================================

class AIGCDetector(nn.Module):
    """
    AI生成图像检测器 (V1)

    结构:
    1. CLIP ViT Backbone (冻结)
    2. FDAA模块 (插入到ViT最后几层)
    3. MGFP模块 (处理输出特征)
    4. 分类头
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
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self._is_clip = False

        self.backbone = self._load_backbone(backbone_name, freeze_backbone)

        self.fdaa_modules = nn.ModuleList([
            FDAA(
                dim=embed_dim, img_size=img_size, patch_size=patch_size,
                reduction=4, num_heads=8, dropout=dropout
            ) for _ in range(num_adapter_layers)
        ])

        self.mgfp = MGFP(
            dim=embed_dim, num_patches=self.num_patches,
            num_prototypes=num_prototypes, use_hierarchical=use_hierarchical,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        if use_hierarchical:
            self.aux_classifier = nn.Linear(embed_dim // 4, num_classes)

        self.use_hierarchical = use_hierarchical

    def _load_backbone(self, backbone_name: str, freeze: bool = True):
        try:
            import clip
            model, _ = clip.load(backbone_name, device='cpu')
            backbone = model.visual

            if freeze:
                for param in backbone.parameters():
                    param.requires_grad = False

            self._is_clip = True
            return backbone
        except ImportError:
            print("Warning: CLIP not installed. Using dummy backbone for testing.")
            return self._create_dummy_backbone()

    def _create_dummy_backbone(self):
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

    def forward_features(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._is_clip:
            # 使用正确的 CLIP token 提取
            cls_token, patch_tokens = _extract_clip_tokens(self.backbone, image)
        else:
            # Dummy/fallback backbone
            all_tokens = self.backbone(image, return_all_tokens=True)
            cls_token = all_tokens[:, 0]
            patch_tokens = all_tokens[:, 1:]

        for fdaa in self.fdaa_modules:
            patch_tokens = fdaa(patch_tokens, image)

        return cls_token, patch_tokens

    def forward(
        self,
        image: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        outputs = {}

        cls_token, patch_tokens = self.forward_features(image)

        mgfp_output, aux_outputs = self.mgfp(
            cls_token, patch_tokens, return_attention=True
        )

        logits = self.classifier(mgfp_output)
        outputs['logits'] = logits
        outputs['cls'] = logits

        if self.use_hierarchical and 'hierarchical_score' in aux_outputs:
            aux_logits = self.aux_classifier(aux_outputs['hierarchical_score'])
            outputs['aux_logits'] = aux_logits

        if return_features:
            outputs['features'] = mgfp_output

        if return_attention:
            outputs['forgery_map'] = aux_outputs['forgery_map']
            outputs['granularity_weights'] = aux_outputs['granularity_weights']
            if 'scale_maps' in aux_outputs:
                outputs['scale_maps'] = aux_outputs['scale_maps']

        return outputs

    def get_trainable_params(self):
        params = []
        for fdaa in self.fdaa_modules:
            params.extend(fdaa.parameters())
        params.extend(self.mgfp.parameters())
        params.extend(self.classifier.parameters())
        if self.use_hierarchical:
            params.extend(self.aux_classifier.parameters())
        return params

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {frozen:,}")
        return {'total': total, 'trainable': trainable, 'frozen': frozen}


# =============================================================================
# V2 检测器 (Late Fusion 架构) - 主模型
# =============================================================================

class AIGCDetectorV2(nn.Module):
    """
    AI生成图像检测器 V2 (Late Fusion 架构)

    关键改进：
    1. 冻结的 CLIP ViT-L/14 backbone — 正确提取 CLS + 全部 patch tokens
    2. FDAAv2 独立频率分支 — 处理原始图像，不修改 ViT token
    3. MGFPv2 Late Fusion — 融合 CLS + AttentionPool(Patches) + freq_feat
    4. embed_dim=1024 匹配 CLIP ViT-L/14 内部维度

    架构:
        Input Image
          ├── CLIP ViT-L/14 (frozen) → CLS [B,1024] + Patches [B,256,1024]
          ├── FDAAv2 (独立频率分支) → freq_feat [B,1024]
          └── MGFPv2 (Late Fusion: CLS + AttentionPool + freq_feat)
              └── Classifier → [B, 2]
    """
    def __init__(
        self,
        backbone_name: str = "ViT-L/14",
        num_classes: int = 2,
        img_size: int = 224,
        embed_dim: int = 1024,
        use_hierarchical: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self._is_clip = False

        # 加载 backbone
        self.backbone, self.patch_size = self._load_backbone(
            backbone_name, freeze_backbone
        )
        self.num_patches = (img_size // self.patch_size) ** 2

        # FDAAv2: 独立频率分支
        self.fdaa = FDAAv2(embed_dim=embed_dim, dropout=dropout)

        # MGFPv2: Late Fusion
        self.mgfp = MGFPv2(
            dim=embed_dim,
            num_patches=self.num_patches,
            use_hierarchical=use_hierarchical,
            num_heads=8,
            dropout=dropout
        )

        # Patch tokens 独立归一化（替代错误的 ln_post 应用）
        self.patch_norm = nn.LayerNorm(embed_dim)

        # CLIP 归一化参数（用于反归一化送入 FDAA）
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer('clip_mean', clip_mean)
        self.register_buffer('clip_std', clip_std)

        # 分类器头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # 辅助分类头
        self.use_hierarchical = use_hierarchical
        if use_hierarchical:
            self.aux_classifier = nn.Linear(embed_dim // 4, num_classes)

    def _load_backbone(self, backbone_name, freeze):
        """加载 backbone，优先 CLIP，fallback 到 torchvision ViT"""
        patch_size = 14  # 默认

        # 尝试加载 CLIP
        try:
            import clip
            model, _ = clip.load(backbone_name, device='cpu')
            backbone = model.visual

            if freeze:
                for param in backbone.parameters():
                    param.requires_grad = False

            # 推断 patch_size
            if hasattr(backbone, 'conv1'):
                patch_size = backbone.conv1.kernel_size[0] if hasattr(backbone.conv1, 'kernel_size') else 14

            self._is_clip = True
            print(f"[AIGCDetectorV2] Loaded CLIP {backbone_name} backbone (frozen={freeze}, width={self.embed_dim})")
            return backbone, patch_size

        except (ImportError, RuntimeError) as e:
            print(f"[AIGCDetectorV2] CLIP not available ({e}), using torchvision ViT-B/16")

        # Fallback: torchvision ViT-B/16
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        patch_size = 16

        # 包装为返回所有 tokens 的模块
        class ViTWrapper(nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.conv_proj = vit_model.conv_proj
                self.class_token = vit_model.class_token
                self.encoder = vit_model.encoder
                self.embed_dim = 768  # ViT-B/16 dim

            def forward_tokens(self, x):
                B = x.shape[0]
                x = self.conv_proj(x)
                x = x.flatten(2).transpose(1, 2)
                cls_tokens = self.class_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                x = self.encoder(x)
                return x[:, 0], x[:, 1:]

        backbone = ViTWrapper(vit)

        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False

        self._is_clip = False
        # ViT-B/16 output is 768, need projection to embed_dim if different
        if self.embed_dim != 768:
            self._vit_proj = nn.Linear(768, self.embed_dim)
        else:
            self._vit_proj = None

        print(f"[AIGCDetectorV2] Loaded torchvision ViT-B/16 backbone (frozen={freeze})")
        return backbone, patch_size

    def forward_backbone(self, image):
        """正确提取 backbone 的 CLS token 和 patch tokens"""
        if self._is_clip:
            # 使用正确的 CLIP token 提取 — 不再复制 CLS token！
            # CLS 已由 ln_post 归一化，patch tokens 需要独立归一化
            cls_token, patch_tokens = _extract_clip_tokens(self.backbone, image)
            patch_tokens = self.patch_norm(patch_tokens)
        else:
            # torchvision ViT fallback
            cls_token, patch_tokens = self.backbone.forward_tokens(image)
            if hasattr(self, '_vit_proj') and self._vit_proj is not None:
                cls_token = self._vit_proj(cls_token)
                patch_tokens = self._vit_proj(patch_tokens)

        return cls_token, patch_tokens

    def forward(
        self,
        image: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            image: [B, 3, H, W] 输入图像
        Returns:
            outputs: dict with 'logits', optionally 'features', 'forgery_map', etc.
        """
        outputs = {}

        # 1. Backbone 特征提取（正确提取所有 patch tokens）
        cls_token, patch_tokens = self.forward_backbone(image)

        # 2. FDAAv2: 独立频率分支（反归一化到 [0,1] 后处理）
        image_raw = image * self.clip_std + self.clip_mean  # 反归一化到 [0,1]
        image_raw = image_raw.clamp(0, 1)
        freq_feat = self.fdaa(image_raw)

        # 3. MGFPv2: Late Fusion
        mgfp_output, aux_outputs = self.mgfp(
            cls_token, patch_tokens, freq_feat, return_attention=True
        )

        # 4. 分类
        logits = self.classifier(mgfp_output)
        outputs['logits'] = logits
        outputs['cls'] = logits

        # 5. 辅助分类
        if self.use_hierarchical and 'hierarchical_score' in aux_outputs:
            aux_logits = self.aux_classifier(aux_outputs['hierarchical_score'])
            outputs['aux_logits'] = aux_logits

        # 可选输出
        if return_features:
            outputs['features'] = mgfp_output

        if return_attention:
            outputs['forgery_map'] = aux_outputs.get('forgery_map', aux_outputs.get('attention_map'))
            if 'scale_maps' in aux_outputs:
                outputs['scale_maps'] = aux_outputs['scale_maps']

        return outputs

    def get_trainable_params(self):
        """获取可训练参数（不包括冻结的backbone）"""
        params = []
        params.extend(self.patch_norm.parameters())
        params.extend(self.fdaa.parameters())
        params.extend(self.mgfp.parameters())
        params.extend(self.classifier.parameters())
        if self.use_hierarchical:
            params.extend(self.aux_classifier.parameters())
        return params

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {frozen:,}")
        return {'total': total, 'trainable': trainable, 'frozen': frozen}


# =============================================================================
# V3 检测器 (空间频率特征 + 频率引导注意力)
# =============================================================================

class AIGCDetectorV3(nn.Module):
    """
    AI生成图像检测器 V3

    关键改进（V3 over V2）：
    1. FDAAv3: 输出 freq_tokens [B,N,D] + freq_global [B,D]（保留空间信息）
    2. MGFPv3: FreqGuidedAttentionPooling（频率引导空间聚合）+ GatedFusion
    3. 相位谱: RealFFTLayerV2 同时输出幅度+相位（36ch 输入）
    4. 消融时 baseline+MGFP 无 freq_tokens → 退化为 fallback 查询
       Full 有频率引导 → FDAA 产生明确独立贡献

    架构:
        Input Image
          ├── CLIP ViT-L/14 (frozen) → CLS [B,1024] + Patches [B,256,1024]
          ├── FDAAv3 → freq_tokens [B,256,1024] + freq_global [B,1024]
          └── MGFPv3 (FreqGuidedPool + GatedFusion)
              └── Classifier → [B, 2]
    """
    def __init__(
        self,
        backbone_name: str = "ViT-L/14",
        num_classes: int = 2,
        img_size: int = 224,
        embed_dim: int = 1024,
        use_hierarchical: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = True
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.embed_dim = embed_dim
        self._is_clip = False

        # 加载 backbone
        self.backbone, self.patch_size = self._load_backbone(
            backbone_name, freeze_backbone
        )
        self.num_patches = (img_size // self.patch_size) ** 2

        # FDAAv3: 空间频率特征 + 相位谱
        self.fdaa = FDAAv3(
            embed_dim=embed_dim,
            num_patches=self.num_patches,
            dropout=dropout,
        )

        # MGFPv3: 频率引导注意力池化 + 门控融合
        self.mgfp = MGFPv3(
            dim=embed_dim,
            num_patches=self.num_patches,
            use_hierarchical=use_hierarchical,
            num_heads=8,
            dropout=dropout
        )

        # Patch tokens 独立归一化
        self.patch_norm = nn.LayerNorm(embed_dim)

        # CLIP 归一化参数（用于反归一化送入 FDAA）
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        self.register_buffer('clip_mean', clip_mean)
        self.register_buffer('clip_std', clip_std)

        # 分类器头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )

        # 辅助分类头
        self.use_hierarchical = use_hierarchical
        if use_hierarchical:
            self.aux_classifier = nn.Linear(embed_dim // 4, num_classes)

    def _load_backbone(self, backbone_name, freeze):
        """加载 backbone，优先 CLIP，fallback 到 torchvision ViT"""
        patch_size = 14

        try:
            import clip
            model, _ = clip.load(backbone_name, device='cpu')
            backbone = model.visual

            if freeze:
                for param in backbone.parameters():
                    param.requires_grad = False

            if hasattr(backbone, 'conv1'):
                patch_size = backbone.conv1.kernel_size[0] if hasattr(backbone.conv1, 'kernel_size') else 14

            self._is_clip = True
            print(f"[AIGCDetectorV3] Loaded CLIP {backbone_name} backbone (frozen={freeze}, width={self.embed_dim})")
            return backbone, patch_size

        except (ImportError, RuntimeError) as e:
            print(f"[AIGCDetectorV3] CLIP not available ({e}), using torchvision ViT-B/16")

        from torchvision.models import vit_b_16, ViT_B_16_Weights
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        patch_size = 16

        class ViTWrapper(nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.conv_proj = vit_model.conv_proj
                self.class_token = vit_model.class_token
                self.encoder = vit_model.encoder
                self.embed_dim = 768

            def forward_tokens(self, x):
                B = x.shape[0]
                x = self.conv_proj(x)
                x = x.flatten(2).transpose(1, 2)
                cls_tokens = self.class_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                x = self.encoder(x)
                return x[:, 0], x[:, 1:]

        backbone = ViTWrapper(vit)
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        self._is_clip = False
        if self.embed_dim != 768:
            self._vit_proj = nn.Linear(768, self.embed_dim)
        else:
            self._vit_proj = None

        print(f"[AIGCDetectorV3] Loaded torchvision ViT-B/16 backbone (frozen={freeze})")
        return backbone, patch_size

    def forward_backbone(self, image):
        """正确提取 backbone 的 CLS token 和 patch tokens"""
        if self._is_clip:
            cls_token, patch_tokens = _extract_clip_tokens(self.backbone, image)
            patch_tokens = self.patch_norm(patch_tokens)
        else:
            cls_token, patch_tokens = self.backbone.forward_tokens(image)
            if hasattr(self, '_vit_proj') and self._vit_proj is not None:
                cls_token = self._vit_proj(cls_token)
                patch_tokens = self._vit_proj(patch_tokens)
        return cls_token, patch_tokens

    def forward(
        self,
        image: torch.Tensor,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        outputs = {}

        # 1. Backbone 特征提取
        cls_token, patch_tokens = self.forward_backbone(image)

        # 2. FDAAv3: 空间频率特征（反归一化到 [0,1]）
        image_raw = image * self.clip_std + self.clip_mean
        image_raw = image_raw.clamp(0, 1)
        freq_tokens, freq_global = self.fdaa(image_raw)

        # 3. MGFPv3: 频率引导注意力 + 门控融合
        mgfp_output, aux_outputs = self.mgfp(
            cls_token, patch_tokens,
            freq_tokens=freq_tokens, freq_global=freq_global,
            return_attention=True
        )

        # 4. 分类
        logits = self.classifier(mgfp_output)
        outputs['logits'] = logits
        outputs['cls'] = logits

        # 5. 辅助分类
        if self.use_hierarchical and 'hierarchical_score' in aux_outputs:
            aux_logits = self.aux_classifier(aux_outputs['hierarchical_score'])
            outputs['aux_logits'] = aux_logits

        # 可选输出
        if return_features:
            outputs['features'] = mgfp_output

        if return_attention:
            outputs['forgery_map'] = aux_outputs.get('forgery_map', aux_outputs.get('attention_map'))
            if 'scale_maps' in aux_outputs:
                outputs['scale_maps'] = aux_outputs['scale_maps']

        return outputs

    def get_trainable_params(self):
        """获取可训练参数（不包括冻结的backbone）"""
        params = []
        params.extend(self.patch_norm.parameters())
        params.extend(self.fdaa.parameters())
        params.extend(self.mgfp.parameters())
        params.extend(self.classifier.parameters())
        if self.use_hierarchical:
            params.extend(self.aux_classifier.parameters())
        return params

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters: {frozen:,}")
        return {'total': total, 'trainable': trainable, 'frozen': frozen}


# =============================================================================
# Lite 检测器 (支持V1/V2切换)
# =============================================================================

class AIGCDetectorLite(nn.Module):
    """
    轻量版检测器 (不依赖CLIP，使用 torchvision ViT-B/16)

    Args:
        pretrained: 使用预训练ViT-B/16作为backbone
        version: 'v1' 或 'v2'，选择使用 FDAA/MGFP 的版本
    """
    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 224,
        embed_dim: int = 768,  # ViT-B/16 dim, 不同于 CLIP ViT-L/14 的 1024
        num_prototypes: int = 4,
        dropout: float = 0.1,
        pretrained: bool = False,
        version: str = 'v2'
    ):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.patch_size = 16
        self.num_patches = (img_size // self.patch_size) ** 2
        self.pretrained = pretrained
        self.version = version

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

        if version == 'v2':
            # V2: 独立频率分支 + Late Fusion
            self.fdaa = FDAAv2(embed_dim=embed_dim, dropout=dropout)
            self.mgfp = MGFPv2(
                dim=embed_dim, num_patches=self.num_patches,
                use_hierarchical=False, dropout=dropout
            )
        else:
            # V1: 原始架构
            self.fdaa = FDAA(
                dim=embed_dim, img_size=img_size, patch_size=self.patch_size,
                reduction=4, num_heads=8, dropout=dropout
            )
            self.mgfp = MGFP(
                dim=embed_dim, num_patches=self.num_patches,
                num_prototypes=num_prototypes, use_hierarchical=False,
                dropout=dropout
            )

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

    def forward(self, image, return_attention=False, return_features=False, **kwargs):
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

        if self.version == 'v2':
            # V2: 独立频率分支
            freq_feat = self.fdaa(image)
            output, aux_outputs = self.mgfp(
                cls_token, patch_tokens, freq_feat, return_attention=True
            )
        else:
            # V1: FDAA修改patch tokens
            patch_tokens = self.fdaa(patch_tokens, image)
            output, aux_outputs = self.mgfp(
                cls_token, patch_tokens, return_attention=True
            )

        logits = self.classifier(output)

        outputs = {'logits': logits, 'cls': logits}

        if return_features:
            outputs['features'] = output

        if return_attention:
            outputs['forgery_map'] = aux_outputs.get('forgery_map', aux_outputs.get('attention_map'))

        return outputs


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing AIGCDetectorLite V1...")
    model_v1 = AIGCDetectorLite(
        num_classes=2, img_size=224, embed_dim=768,
        num_prototypes=4, version='v1'
    )
    image = torch.randn(4, 3, 224, 224)
    outputs_v1 = model_v1(image, return_attention=True)
    print(f"V1 Logits shape: {outputs_v1['logits'].shape}")
    print(f"V1 Total params: {sum(p.numel() for p in model_v1.parameters()):,}")
    print(f"V1 Trainable params: {sum(p.numel() for p in model_v1.parameters() if p.requires_grad):,}")

    print()
    print("=" * 60)
    print("Testing AIGCDetectorLite V2...")
    model_v2 = AIGCDetectorLite(
        num_classes=2, img_size=224, embed_dim=768,
        version='v2'
    )
    outputs_v2 = model_v2(image, return_attention=True)
    print(f"V2 Logits shape: {outputs_v2['logits'].shape}")
    print(f"V2 Total params: {sum(p.numel() for p in model_v2.parameters()):,}")
    print(f"V2 Trainable params: {sum(p.numel() for p in model_v2.parameters() if p.requires_grad):,}")

    print()
    print("=" * 60)
    print("Testing AIGCDetectorV2 (full, with CLIP/ViT backbone)...")
    model_full = AIGCDetectorV2(
        backbone_name="ViT-L/14",
        num_classes=2, img_size=224, embed_dim=1024,
        use_hierarchical=True, dropout=0.1
    )
    outputs_full = model_full(image, return_features=True, return_attention=True)
    print(f"Full V2 Logits shape: {outputs_full['logits'].shape}")
    print(f"Full V2 Features shape: {outputs_full['features'].shape}")

    # 验证 patch tokens 不是重复的 CLS token
    cls, patches = model_full.forward_backbone(image)
    print(f"\nCLS token shape: {cls.shape}")
    print(f"Patch tokens shape: {patches.shape}")
    if patches.shape[1] > 1:
        is_unique = not torch.allclose(patches[:, 0], patches[:, 1], atol=1e-4)
        print(f"Patch tokens are unique (not repeated CLS): {is_unique}")

    model_full.count_parameters()
