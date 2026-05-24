"""
损失函数模块

包含:
1. Focal Loss - 处理类别不平衡
2. Contrastive Loss - 对比学习
3. AIGCDetectionLoss - 综合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss
    用于处理正负样本不平衡问题

    Reference:
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, C] 预测logits
            targets: [B] 真实标签
        Returns:
            loss: scalar
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                   label_smoothing=self.label_smoothing)
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())
            focal_weight = alpha_t * focal_weight

        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss
    用于学习更具判别性的特征表示

    Reference:
    Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020
    """
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] 特征向量
            labels: [B] 标签
        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]

        # 归一化特征
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # 创建mask：同类样本为正样本对
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 移除对角线（自身）
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        # 计算log softmax
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        # 计算正样本对的平均log概率
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)

        # 损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class SourceAwareContrastiveLoss(nn.Module):
    """
    源感知对比损失

    在 SupCon 基础上，跨源同标签样本对给予更高权重：
    - BigGAN-fake 靠近 ADM-fake = 高奖励（鼓励源不变性）
    - 同源同标签 = 标准权重

    这迫使模型学习源不变的判别特征，提升跨域泛化能力。
    """
    def __init__(self, temperature=0.07, base_temperature=0.07, cross_source_weight=2.0):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.cross_source_weight = cross_source_weight

    def forward(self, features, labels, sources=None):
        """
        Args:
            features: [B, D] 特征向量
            labels: [B] 标签
            sources: [B] 源 ID (int)，可选。为 None 时退化为标准 SupCon
        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.view(-1, 1)
        label_mask = torch.eq(labels, labels.T).float().to(device)

        # 移除对角线
        logits_mask = torch.ones_like(label_mask) - torch.eye(batch_size, device=device)
        pos_mask = label_mask * logits_mask

        # 源感知权重: 跨源同标签对权重更高
        if sources is not None:
            sources = sources.view(-1, 1)
            same_source = torch.eq(sources, sources.T).float().to(device)
            cross_source = 1.0 - same_source
            # 跨源同标签 → 更高权重
            weight_matrix = torch.ones_like(pos_mask)
            cross_source_pos = pos_mask * cross_source
            weight_matrix = weight_matrix + cross_source_pos * (self.cross_source_weight - 1.0)
            weighted_pos_mask = pos_mask * weight_matrix
        else:
            weighted_pos_mask = pos_mask

        # 计算 log softmax
        exp_logits = torch.exp(similarity_matrix) * logits_mask
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        # 加权正样本对的平均 log 概率
        mean_log_prob_pos = (weighted_pos_mask * log_prob).sum(dim=1) / (weighted_pos_mask.sum(dim=1) + 1e-6)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


class ConsistencyLoss(nn.Module):
    """
    一致性损失
    确保多粒度特征的预测一致性
    """
    def __init__(self):
        super().__init__()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, logits1, logits2):
        """
        Args:
            logits1: [B, C] 主分类logits
            logits2: [B, C] 辅助分类logits
        Returns:
            loss: scalar
        """
        p1 = F.log_softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        return self.kl_loss(p1, p2)


class ForgeryLocalizationLoss(nn.Module):
    """
    伪造定位损失
    用于监督伪造注意力图的学习
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, forgery_map, labels):
        """
        Args:
            forgery_map: [B, N] 预测的伪造注意力图
            labels: [B] 真实标签 (0: real, 1: fake)
        Returns:
            loss: scalar
        """
        # 对于真实图像，伪造注意力应该低
        # 对于伪造图像，伪造注意力应该高
        target = labels.float().unsqueeze(1).expand_as(forgery_map)
        return self.bce(forgery_map, target)


class AIGCDetectionLoss(nn.Module):
    """
    AIGC检测综合损失函数

    组合多种损失:
    1. 主分类损失 (Focal Loss)
    2. 辅助分类损失 (可选)
    3. 对比学习损失 (可选)
    4. 伪造定位损失 (可选)
    """
    def __init__(
        self,
        use_focal: bool = True,
        use_contrastive: bool = True,
        use_aux: bool = True,
        use_localization: bool = False,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2.0,
        contrastive_weight: float = 0.1,
        aux_weight: float = 0.5,
        localization_weight: float = 0.1,
        label_smoothing: float = 0.0
    ):
        super().__init__()

        self.use_focal = use_focal
        self.use_contrastive = use_contrastive
        self.use_aux = use_aux
        self.use_localization = use_localization

        if use_focal:
            self.cls_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                                       label_smoothing=label_smoothing)
        else:
            self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        if use_contrastive:
            self.contrastive_loss = SourceAwareContrastiveLoss()
            self.contrastive_weight = contrastive_weight

        if use_aux:
            self.aux_loss = nn.CrossEntropyLoss()
            self.aux_weight = aux_weight

        if use_localization:
            self.localization_loss = ForgeryLocalizationLoss()
            self.localization_weight = localization_weight

    def forward(self, outputs, labels, features=None, sources=None):
        """
        Args:
            outputs: dict containing model outputs
            labels: [B] 真实标签
            features: [B, D] 特征向量 (用于对比学习)
            sources: [B] 源 ID (用于源感知对比学习，可选)
        Returns:
            loss_dict: dict containing all losses
        """
        loss_dict = {}
        total_loss = 0.0

        # 1. 主分类损失
        cls_loss = self.cls_loss(outputs['logits'], labels)
        loss_dict['cls_loss'] = cls_loss
        total_loss += cls_loss

        # 2. 辅助分类损失
        if self.use_aux and 'aux_logits' in outputs:
            aux_loss = self.aux_loss(outputs['aux_logits'], labels)
            loss_dict['aux_loss'] = aux_loss
            total_loss += self.aux_weight * aux_loss

        # 3. 对比学习损失（源感知）
        if self.use_contrastive and features is not None:
            contrastive_loss = self.contrastive_loss(features, labels, sources=sources)
            loss_dict['contrastive_loss'] = contrastive_loss
            total_loss += self.contrastive_weight * contrastive_loss

        # 4. 伪造定位损失
        if self.use_localization and 'forgery_map' in outputs:
            localization_loss = self.localization_loss(outputs['forgery_map'], labels)
            loss_dict['localization_loss'] = localization_loss
            total_loss += self.localization_weight * localization_loss

        loss_dict['total_loss'] = total_loss

        return loss_dict


# 测试代码
if __name__ == "__main__":
    print("Testing losses...")

    batch_size = 8
    num_classes = 2
    embed_dim = 768
    num_patches = 196

    # 模拟数据
    logits = torch.randn(batch_size, num_classes)
    aux_logits = torch.randn(batch_size, num_classes)
    features = torch.randn(batch_size, embed_dim)
    forgery_map = torch.sigmoid(torch.randn(batch_size, num_patches))
    labels = torch.randint(0, 2, (batch_size,))

    outputs = {
        'logits': logits,
        'aux_logits': aux_logits,
        'forgery_map': forgery_map
    }

    # 测试综合损失
    criterion = AIGCDetectionLoss(
        use_focal=True,
        use_contrastive=True,
        use_aux=True,
        use_localization=True
    )

    loss_dict = criterion(outputs, labels, features)

    print("\nLoss values:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value.item():.4f}")
