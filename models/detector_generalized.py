

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .modules.fdaa_da import FDAA_DA, MultiLevelDAFL
from .modules.mgfp_pcr import MGFP_PCR
from .losses.generalization_losses import UnifiedGeneralizationLoss


class GeneralizedAIGCDetector(nn.Module):
    """
    泛化增强型AI生成图像检测器

    核心改进：
    - 域对抗频率学习 (DAFL)
    - 自适应滤波器组 (AFB)
    - 原型对比正则化 (PCR)
    """
    def __init__(
        self,
        backbone_name: str = "ViT-L/14",
        num_classes: int = 2,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 1024,
        # FDAA-DA配置
        num_adapter_layers: int = 3,
        use_multi_level_da: bool = True,
        # MGFP-PCR配置
        num_prototypes: int = 4,
        use_hierarchical: bool = True,
        use_pcr: bool = True,
        # 域配置
        num_domains: int = 2,
        use_domain_adversarial: bool = True,
        # 其他
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
        self.num_domains = num_domains
        self.use_domain_adversarial = use_domain_adversarial
        self.use_pcr = use_pcr

        # 加载backbone
        self.backbone = self._load_backbone(backbone_name, freeze_backbone)

        # FDAA-DA模块
        if use_multi_level_da:
            self.fdaa = MultiLevelDAFL(
                dim=embed_dim,
                img_size=img_size,
                patch_size=patch_size,
                num_levels=num_adapter_layers,
                num_domains=num_domains,
                dropout=dropout
            )
        else:
            self.fdaa_modules = nn.ModuleList([
                FDAA_DA(
                    dim=embed_dim,
                    img_size=img_size,
                    patch_size=patch_size,
                    num_domains=num_domains,
                    use_domain_adversarial=use_domain_adversarial,
                    dropout=dropout
                ) for _ in range(num_adapter_layers)
            ])

        self.use_multi_level_da = use_multi_level_da

        # MGFP-PCR模块
        self.mgfp = MGFP_PCR(
            dim=embed_dim,
            num_patches=self.num_patches,
            num_prototypes=num_prototypes,
            use_hierarchical=use_hierarchical,
            use_pcr=use_pcr,
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

        # 辅助分类头
        if use_hierarchical:
            self.aux_classifier = nn.Linear(1, num_classes)

        self.use_hierarchical = use_hierarchical

        # 当前epoch（用于域对抗调度）
        self.current_epoch = 0
        self.total_epochs = 1

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
        """创建测试用backbone"""

        class DummyViT(nn.Module):
            def __init__(self, embed_dim, num_patches, patch_size):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_patches = num_patches
                self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
                self.class_embedding = nn.Parameter(torch.randn(embed_dim))
                self.positional_embedding = nn.Parameter(torch.randn(num_patches + 1, embed_dim))
                self.ln_pre = nn.LayerNorm(embed_dim)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, batch_first=True),
                    num_layers=6
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

        return DummyViT(self.embed_dim, self.num_patches, self.patch_size)

    def set_epoch(self, epoch: int, total_epochs: int):
        """
        设置当前epoch，用于域对抗调度
        """
        self.current_epoch = epoch
        self.total_epochs = total_epochs

        if self.use_multi_level_da:
            self.fdaa.set_domain_lambda(epoch, total_epochs)
        else:
            for fdaa in self.fdaa_modules:
                fdaa.set_domain_lambda(epoch, total_epochs)

    def forward_features(
        self,
        image: torch.Tensor,
        return_domain_logits: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        特征提取

        Args:
            image: [B, 3, H, W]
            return_domain_logits: 是否返回域预测

        Returns:
            cls_token: [B, D]
            patch_tokens: [B, N, D]
            domain_logits: [B, num_domains] 或 list (可选)
        """
        # Backbone特征
        try:
            all_tokens = self.backbone(image, return_all_tokens=True)
            cls_token = all_tokens[:, 0]
            patch_tokens = all_tokens[:, 1:]
        except TypeError:
            features = self.backbone(image)
            if features.dim() == 2:
                cls_token = features
                B = image.shape[0]
                patch_tokens = torch.zeros(B, self.num_patches, self.embed_dim, device=image.device)
            else:
                cls_token = features[:, 0]
                patch_tokens = features[:, 1:]

        # FDAA-DA处理
        domain_logits = None

        if self.use_multi_level_da:
            if return_domain_logits and self.use_domain_adversarial:
                patch_tokens, domain_logits = self.fdaa(
                    patch_tokens, image, return_domain_logits=True
                )
            else:
                patch_tokens = self.fdaa(patch_tokens, image, return_domain_logits=False)
        else:
            domain_logits_list = []
            for fdaa in self.fdaa_modules:
                if return_domain_logits and self.use_domain_adversarial:
                    patch_tokens, dl = fdaa(patch_tokens, image, return_domain_logits=True)
                    domain_logits_list.append(dl)
                else:
                    patch_tokens = fdaa(patch_tokens, image, return_domain_logits=False)

            if domain_logits_list:
                domain_logits = domain_logits_list

        return cls_token, patch_tokens, domain_logits

    def forward(
        self,
        image: torch.Tensor,
        labels: torch.Tensor = None,
        domain_labels: torch.Tensor = None,
        return_features: bool = False,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            image: [B, 3, H, W] 输入图像
            labels: [B] 真假标签（训练时用于PCR）
            domain_labels: [B] 域标签（训练时用于域对抗）
            return_features: 是否返回特征
            return_attention: 是否返回注意力图

        Returns:
            outputs: dict
        """
        outputs = {}

        # 是否需要域预测
        need_domain = (
            self.training and
            self.use_domain_adversarial and
            domain_labels is not None
        )

        # 1. 特征提取 + FDAA-DA
        cls_token, patch_tokens, domain_logits = self.forward_features(
            image, return_domain_logits=need_domain
        )

        if domain_logits is not None:
            outputs['domain_logits'] = domain_logits

        # 2. MGFP-PCR
        mgfp_output, aux_outputs = self.mgfp(
            cls_token, patch_tokens,
            labels=labels if self.training else None,
            domain_labels=domain_labels if self.training else None,
            return_attention=True
        )

        # 3. 主分类
        logits = self.classifier(mgfp_output)
        outputs['logits'] = logits
        outputs['cls'] = logits  # 兼容性

        # 4. 辅助分类
        if self.use_hierarchical and 'hierarchical_score' in aux_outputs:
            aux_logits = self.aux_classifier(aux_outputs['hierarchical_score'])
            outputs['aux_logits'] = aux_logits

        # 5. 收集训练所需的信息
        if self.training:
            # 原型信息
            if 'proto_info' in aux_outputs:
                outputs['prototypes'] = aux_outputs['proto_info']['prototypes']

            # PCR损失
            if 'pcr_losses' in aux_outputs:
                outputs['pcr_losses'] = aux_outputs['pcr_losses']

            # 对比特征
            if 'contrast_feat' in aux_outputs:
                outputs['contrast_features'] = aux_outputs['contrast_feat']

            # 获取FDAA正则化损失
            if self.use_multi_level_da:
                outputs['afb_reg_losses'] = self.fdaa.get_regularization_loss()
            else:
                afb_losses = {}
                for i, fdaa in enumerate(self.fdaa_modules):
                    level_losses = fdaa.get_regularization_loss()
                    for k, v in level_losses.items():
                        afb_losses[f'level{i}_{k}'] = v
                outputs['afb_reg_losses'] = afb_losses

        # 可选输出
        if return_features:
            outputs['features'] = mgfp_output

        if return_attention:
            outputs['forgery_map'] = aux_outputs['forgery_map']
            outputs['granularity_weights'] = aux_outputs['granularity_weights']
            if 'scale_maps' in aux_outputs:
                outputs['scale_maps'] = aux_outputs['scale_maps']

        return outputs

    def get_trainable_params(self):
        """获取可训练参数"""
        params = []

        # FDAA参数
        if self.use_multi_level_da:
            params.extend(self.fdaa.parameters())
        else:
            for fdaa in self.fdaa_modules:
                params.extend(fdaa.parameters())

        # MGFP参数
        params.extend(self.mgfp.parameters())

        # 分类头参数
        params.extend(self.classifier.parameters())

        if self.use_hierarchical:
            params.extend(self.aux_classifier.parameters())

        return params

    def count_parameters(self):
        """统计参数量"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen
        }


class GeneralizedDetectorTrainer:
    """
    泛化检测器训练器

    封装训练逻辑，包括损失计算和优化
    """
    def __init__(
        self,
        model: GeneralizedAIGCDetector,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        device: str = 'cuda',
        # 损失配置
        loss_config: dict = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # 默认损失配置
        default_config = {
            'use_focal': True,
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'use_aux': True,
            'aux_weight': 0.5,
            'use_domain_adversarial': model.use_domain_adversarial,
            'domain_weight': 0.1,
            'use_prototype_alignment': model.use_pcr,
            'prototype_alignment_weight': 0.1,
            'use_prototype_diversity': model.use_pcr,
            'prototype_diversity_weight': 0.05,
            'use_contrastive': True,
            'contrastive_weight': 0.1,
            'use_consistency': True,
            'consistency_weight': 0.1,
            'use_cross_domain': model.use_domain_adversarial,
            'cross_domain_weight': 0.1
        }

        if loss_config:
            default_config.update(loss_config)

        self.criterion = UnifiedGeneralizationLoss(**default_config)

    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor = None
    ) -> Dict[str, float]:
        """
        单步训练

        Args:
            images: [B, 3, H, W]
            labels: [B]
            domain_labels: [B] (可选)

        Returns:
            loss_dict: 各损失值
        """
        self.model.train()

        images = images.to(self.device)
        labels = labels.to(self.device)
        if domain_labels is not None:
            domain_labels = domain_labels.to(self.device)

        # 前向传播
        outputs = self.model(
            images,
            labels=labels,
            domain_labels=domain_labels,
            return_features=True
        )

        # 计算损失
        loss_dict = self.criterion(
            logits=outputs['logits'],
            labels=labels,
            aux_logits=outputs.get('aux_logits'),
            domain_logits=outputs.get('domain_logits'),
            domain_labels=domain_labels,
            features=outputs.get('features'),
            prototypes=outputs.get('prototypes'),
            contrast_features=outputs.get('contrast_features'),
            afb_reg_losses=outputs.get('afb_reg_losses'),
            pcr_losses=outputs.get('pcr_losses')
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        self.optimizer.step()

        # 转换为Python float
        return {k: v.item() for k, v in loss_dict.items()}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        评估模型

        Returns:
            metrics: 评估指标
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        for batch in dataloader:
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels = batch[0], batch[1]

            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            probs = F.softmax(outputs['logits'], dim=-1)

            all_preds.append(outputs['logits'].argmax(dim=-1))
            all_labels.append(labels)
            all_probs.append(probs[:, 1])

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        probs = torch.cat(all_probs)

        # 计算指标
        accuracy = (preds == labels).float().mean().item()

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy())
        except:
            auc = 0.0

        return {
            'accuracy': accuracy,
            'auc': auc
        }


# 测试代码
if __name__ == "__main__":
    print("Testing Generalized AIGC Detector...")

    batch_size = 4
    img_size = 224
    num_domains = 3

    # 创建模型
    model = GeneralizedAIGCDetector(
        backbone_name="ViT-L/14",
        num_classes=2,
        img_size=img_size,
        patch_size=14,
        embed_dim=1024,
        num_adapter_layers=3,
        use_multi_level_da=True,
        num_prototypes=4,
        use_hierarchical=True,
        use_pcr=True,
        num_domains=num_domains,
        use_domain_adversarial=True
    )

    # 设置epoch
    model.set_epoch(epoch=5, total_epochs=10)

    # 模拟输入
    images = torch.randn(batch_size, 3, img_size, img_size)
    labels = torch.randint(0, 2, (batch_size,))
    domain_labels = torch.randint(0, num_domains, (batch_size,))

    # 训练模式前向
    model.train()
    outputs = model(
        images,
        labels=labels,
        domain_labels=domain_labels,
        return_features=True,
        return_attention=True
    )

    print("\n[Training Mode] Output keys:", list(outputs.keys()))
    print(f"  logits shape: {outputs['logits'].shape}")
    if 'domain_logits' in outputs:
        print(f"  domain_logits: {type(outputs['domain_logits'])}")
    if 'forgery_map' in outputs:
        print(f"  forgery_map shape: {outputs['forgery_map'].shape}")

    # 评估模式前向
    model.eval()
    with torch.no_grad():
        outputs_eval = model(images, return_attention=True)

    print("\n[Eval Mode] Output keys:", list(outputs_eval.keys()))

    # 参数统计
    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  Total: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")

    # 测试训练步骤
    print("\n" + "="*50)
    print("Testing training step...")

    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=1e-4)
    trainer = GeneralizedDetectorTrainer(
        model=model,
        optimizer=optimizer,
        device='cpu'
    )

    loss_dict = trainer.train_step(images, labels, domain_labels)
    print("\nLoss values:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print("\nAll tests passed!")
