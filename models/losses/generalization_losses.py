"""
Generalization-Focused Loss Functions
面向泛化的损失函数集合

包含：
1. 域对抗损失 (Domain Adversarial Loss)
2. 原型对比损失 (Prototype Contrastive Loss)
3. 滤波器正则化损失 (Filter Regularization Loss)
4. 一致性损失 (Consistency Loss)
5. 综合泛化损失 (Unified Generalization Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAdversarialLoss(nn.Module):
    """
    域对抗损失

    通过最小化域判别器的准确率，迫使特征编码器学习域不变特征
    """
    def __init__(self, num_domains: int = 2):
        super().__init__()
        self.num_domains = num_domains
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, domain_logits, domain_labels):
        """
        Args:
            domain_logits: [B, num_domains] 或 list of [B, num_domains]
            domain_labels: [B] 域标签

        Returns:
            loss: 域对抗损失
        """
        if isinstance(domain_logits, list):
            # 多层级域对抗
            total_loss = 0
            for logits in domain_logits:
                total_loss = total_loss + self.criterion(logits, domain_labels)
            return total_loss / len(domain_logits)
        else:
            return self.criterion(domain_logits, domain_labels)


class FocalLoss(nn.Module):
    """
    Focal Loss

    处理类别不平衡问题
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        Args:
            logits: [B, C] 预测logits
            targets: [B] 目标标签

        Returns:
            loss: focal loss
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # 计算focal weight
        focal_weight = (1 - pt) ** self.gamma

        # 计算alpha weight
        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss
        return loss.mean()


class ConsistencyLoss(nn.Module):
    """
    一致性损失

    确保主分类器和辅助分类器预测一致
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, main_logits, aux_logits):
        """
        Args:
            main_logits: [B, C] 主分类器logits
            aux_logits: [B, C] 辅助分类器logits

        Returns:
            loss: KL散度损失
        """
        main_prob = F.softmax(main_logits / self.temperature, dim=-1)
        aux_log_prob = F.log_softmax(aux_logits / self.temperature, dim=-1)

        loss = F.kl_div(aux_log_prob, main_prob, reduction='batchmean')
        return loss * (self.temperature ** 2)


class PrototypeAlignmentLoss(nn.Module):
    """
    原型对齐损失

    确保fake样本与伪造原型对齐
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, prototypes, labels):
        """
        Args:
            features: [B, D] 样本特征
            prototypes: [K, D] 伪造原型
            labels: [B] 标签 (0=real, 1=fake)

        Returns:
            loss: 原型对齐损失
        """
        # L2归一化
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(prototypes, dim=-1)

        # fake样本掩码
        fake_mask = (labels == 1)

        if fake_mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        fake_features = features[fake_mask]

        # 计算与原型的相似度
        similarity = torch.mm(fake_features, prototypes.T) / self.temperature

        # 鼓励与至少一个原型高度相似
        loss = -similarity.logsumexp(dim=-1).mean()

        return loss


class PrototypeDiversityLoss(nn.Module):
    """
    原型多样性损失

    防止原型坍缩到相似模式
    """
    def __init__(self):
        super().__init__()

    def forward(self, prototypes):
        """
        Args:
            prototypes: [K, D] 伪造原型

        Returns:
            loss: 多样性损失
        """
        # L2归一化
        prototypes = F.normalize(prototypes, dim=-1)

        # 计算相似度矩阵
        similarity = torch.mm(prototypes, prototypes.T)

        # 排除对角线
        K = prototypes.shape[0]
        mask = 1 - torch.eye(K, device=prototypes.device)

        # 惩罚高相似度
        loss = (similarity.abs() * mask).sum() / (mask.sum() + 1e-8)

        return loss


class FilterRegularizationLoss(nn.Module):
    """
    滤波器正则化损失

    包含多样性和稀疏性约束
    """
    def __init__(
        self,
        diversity_weight: float = 0.1,
        sparsity_weight: float = 0.01
    ):
        super().__init__()
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight

    def forward(self, filters):
        """
        Args:
            filters: [K, C, H, W] 可学习滤波器

        Returns:
            loss_dict: 包含各损失项
        """
        loss_dict = {}

        # 多样性损失
        K = filters.shape[0]
        filters_flat = filters.view(K, -1)
        filters_norm = F.normalize(filters_flat, dim=-1)
        similarity = torch.mm(filters_norm, filters_norm.T)
        mask = 1 - torch.eye(K, device=filters.device)
        diversity_loss = (similarity.abs() * mask).sum() / (mask.sum() + 1e-8)
        loss_dict['filter_diversity'] = self.diversity_weight * diversity_loss

        # 稀疏性损失
        sparsity_loss = filters.abs().mean()
        loss_dict['filter_sparsity'] = self.sparsity_weight * sparsity_loss

        loss_dict['total_filter_reg'] = (
            loss_dict['filter_diversity'] + loss_dict['filter_sparsity']
        )

        return loss_dict


class CrossDomainAlignmentLoss(nn.Module):
    """
    跨域对齐损失

    确保不同域的同类样本在特征空间中对齐
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels, domain_labels):
        """
        Args:
            features: [B, D] 样本特征
            labels: [B] 类别标签
            domain_labels: [B] 域标签

        Returns:
            loss: 跨域对齐损失
        """
        device = features.device

        # L2归一化
        features = F.normalize(features, dim=-1)

        unique_domains = domain_labels.unique()
        if len(unique_domains) < 2:
            return torch.tensor(0.0, device=device)

        # 对每个类别计算跨域对齐损失
        total_loss = 0
        count = 0

        for label in labels.unique():
            label_mask = (labels == label)
            label_features = features[label_mask]
            label_domains = domain_labels[label_mask]

            domain_means = []
            for domain in unique_domains:
                domain_mask = (label_domains == domain)
                if domain_mask.sum() > 0:
                    domain_mean = label_features[domain_mask].mean(dim=0)
                    domain_means.append(domain_mean)

            if len(domain_means) > 1:
                # 计算不同域之间的距离
                domain_means = torch.stack(domain_means)
                # 所有域均值的中心
                center = domain_means.mean(dim=0, keepdim=True)
                # 最小化到中心的距离
                distances = (domain_means - center).pow(2).sum(dim=-1)
                total_loss = total_loss + distances.mean()
                count += 1

        if count > 0:
            return total_loss / count

        return torch.tensor(0.0, device=device)


class UnifiedGeneralizationLoss(nn.Module):
    """
    综合泛化损失

    整合所有损失项用于训练
    """
    def __init__(
        self,
        # 分类损失
        use_focal: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        # 辅助损失
        use_aux: bool = True,
        aux_weight: float = 0.5,
        # 域对抗损失
        use_domain_adversarial: bool = True,
        domain_weight: float = 0.1,
        # 原型损失
        use_prototype_alignment: bool = True,
        prototype_alignment_weight: float = 0.1,
        use_prototype_diversity: bool = True,
        prototype_diversity_weight: float = 0.05,
        # 对比损失
        use_contrastive: bool = True,
        contrastive_weight: float = 0.1,
        # 一致性损失
        use_consistency: bool = True,
        consistency_weight: float = 0.1,
        # 跨域对齐
        use_cross_domain: bool = True,
        cross_domain_weight: float = 0.1
    ):
        super().__init__()

        # 配置
        self.use_focal = use_focal
        self.use_aux = use_aux
        self.use_domain_adversarial = use_domain_adversarial
        self.use_prototype_alignment = use_prototype_alignment
        self.use_prototype_diversity = use_prototype_diversity
        self.use_contrastive = use_contrastive
        self.use_consistency = use_consistency
        self.use_cross_domain = use_cross_domain

        # 权重
        self.aux_weight = aux_weight
        self.domain_weight = domain_weight
        self.prototype_alignment_weight = prototype_alignment_weight
        self.prototype_diversity_weight = prototype_diversity_weight
        self.contrastive_weight = contrastive_weight
        self.consistency_weight = consistency_weight
        self.cross_domain_weight = cross_domain_weight

        # 损失函数
        if use_focal:
            self.cls_loss = FocalLoss(focal_alpha, focal_gamma)
        else:
            self.cls_loss = nn.CrossEntropyLoss()

        if use_domain_adversarial:
            self.domain_loss = DomainAdversarialLoss()

        if use_prototype_alignment:
            self.proto_align_loss = PrototypeAlignmentLoss()

        if use_prototype_diversity:
            self.proto_div_loss = PrototypeDiversityLoss()

        if use_contrastive:
            from .mgfp_pcr import SupervisedContrastiveLoss
            self.contrastive_loss = SupervisedContrastiveLoss()

        if use_consistency:
            self.consistency_loss = ConsistencyLoss()

        if use_cross_domain:
            self.cross_domain_loss = CrossDomainAlignmentLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aux_logits: torch.Tensor = None,
        domain_logits: torch.Tensor = None,
        domain_labels: torch.Tensor = None,
        features: torch.Tensor = None,
        prototypes: torch.Tensor = None,
        contrast_features: torch.Tensor = None,
        afb_reg_losses: dict = None,
        pcr_losses: dict = None
    ):
        """
        计算综合损失

        Args:
            logits: [B, C] 主分类logits
            labels: [B] 目标标签
            aux_logits: [B, C] 辅助分类logits（可选）
            domain_logits: [B, D] 域预测logits（可选）
            domain_labels: [B] 域标签（可选）
            features: [B, F] 特征向量（可选）
            prototypes: [K, F] 伪造原型（可选）
            contrast_features: [B, F] 对比学习特征（可选）
            afb_reg_losses: AFB正则化损失（可选）
            pcr_losses: PCR损失（可选）

        Returns:
            loss_dict: 包含所有损失项的字典
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=logits.device)

        # 1. 主分类损失
        cls_loss = self.cls_loss(logits, labels)
        loss_dict['cls_loss'] = cls_loss
        total_loss = total_loss + cls_loss

        # 2. 辅助分类损失
        if self.use_aux and aux_logits is not None:
            aux_loss = self.cls_loss(aux_logits, labels)
            loss_dict['aux_loss'] = aux_loss
            total_loss = total_loss + self.aux_weight * aux_loss

        # 3. 域对抗损失
        if self.use_domain_adversarial and domain_logits is not None and domain_labels is not None:
            domain_loss = self.domain_loss(domain_logits, domain_labels)
            loss_dict['domain_loss'] = domain_loss
            total_loss = total_loss + self.domain_weight * domain_loss

        # 4. 原型对齐损失
        if self.use_prototype_alignment and features is not None and prototypes is not None:
            proto_align = self.proto_align_loss(features, prototypes, labels)
            loss_dict['prototype_alignment'] = proto_align
            total_loss = total_loss + self.prototype_alignment_weight * proto_align

        # 5. 原型多样性损失
        if self.use_prototype_diversity and prototypes is not None:
            proto_div = self.proto_div_loss(prototypes)
            loss_dict['prototype_diversity'] = proto_div
            total_loss = total_loss + self.prototype_diversity_weight * proto_div

        # 6. 对比损失
        if self.use_contrastive and contrast_features is not None:
            contrastive = self.contrastive_loss(contrast_features, labels)
            loss_dict['contrastive_loss'] = contrastive
            total_loss = total_loss + self.contrastive_weight * contrastive

        # 7. 一致性损失
        if self.use_consistency and aux_logits is not None:
            consistency = self.consistency_loss(logits, aux_logits)
            loss_dict['consistency_loss'] = consistency
            total_loss = total_loss + self.consistency_weight * consistency

        # 8. 跨域对齐损失
        if self.use_cross_domain and features is not None and domain_labels is not None:
            cross_domain = self.cross_domain_loss(features, labels, domain_labels)
            loss_dict['cross_domain_loss'] = cross_domain
            total_loss = total_loss + self.cross_domain_weight * cross_domain

        # 9. AFB正则化损失
        if afb_reg_losses is not None:
            for k, v in afb_reg_losses.items():
                loss_dict[f'afb_{k}'] = v
                if 'total' in k:
                    total_loss = total_loss + v

        # 10. PCR损失
        if pcr_losses is not None:
            for k, v in pcr_losses.items():
                loss_dict[f'pcr_{k}'] = v
            pcr_total = sum(v for v in pcr_losses.values())
            total_loss = total_loss + 0.1 * pcr_total

        loss_dict['total_loss'] = total_loss

        return loss_dict


# 导入SupervisedContrastiveLoss（避免循环导入）
try:
    from ..modules.mgfp_pcr import SupervisedContrastiveLoss
except ImportError:
    pass


# 测试代码
if __name__ == "__main__":
    print("Testing Generalization Losses...")

    batch_size = 16
    num_classes = 2
    feature_dim = 1024
    num_prototypes = 4
    num_domains = 3

    # 模拟数据
    logits = torch.randn(batch_size, num_classes)
    aux_logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    domain_logits = torch.randn(batch_size, num_domains)
    domain_labels = torch.randint(0, num_domains, (batch_size,))
    features = torch.randn(batch_size, feature_dim)
    prototypes = torch.randn(num_prototypes, feature_dim)
    contrast_features = F.normalize(torch.randn(batch_size, 128), dim=-1)

    # 测试综合损失
    criterion = UnifiedGeneralizationLoss(
        use_focal=True,
        use_aux=True,
        use_domain_adversarial=True,
        use_prototype_alignment=True,
        use_prototype_diversity=True,
        use_contrastive=False,  # 避免导入问题
        use_consistency=True,
        use_cross_domain=True
    )

    loss_dict = criterion(
        logits=logits,
        labels=labels,
        aux_logits=aux_logits,
        domain_logits=domain_logits,
        domain_labels=domain_labels,
        features=features,
        prototypes=prototypes
    )

    print("\nLoss values:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    print("\nAll loss tests passed!")
