
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeContrastiveRegularization(nn.Module):
    """
    原型对比正则化模块

    核心思想：
    1. 同类样本（fake）应该靠近对应原型
    2. 不同域的同类样本也应该能匹配到相同原型
    3. 原型之间应该保持多样性
    """
    def __init__(
        self,
        dim: int,
        proj_dim: int = 256,
        temperature: float = 0.07
    ):
        super().__init__()

        self.temperature = temperature
        self.proj_dim = proj_dim

        # 原型投影头
        self.prototype_projector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, proj_dim)
        )

        # 特征投影头
        self.feature_projector = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, proj_dim)
        )

    def forward(
        self,
        prototypes: torch.Tensor,
        features: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor = None
    ):
        """
        计算原型对比损失

        Args:
            prototypes: [K, D] 伪造原型
            features: [B, D] 样本特征
            labels: [B] 真假标签 (0=real, 1=fake)
            domain_labels: [B] 域标签（可选）

        Returns:
            loss_dict: 包含各种对比损失
        """
        loss_dict = {}
        device = features.device

        # 投影
        proto_proj = self.prototype_projector(prototypes)  # [K, proj_dim]
        feat_proj = self.feature_projector(features)        # [B, proj_dim]

        # L2归一化
        proto_proj = F.normalize(proto_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, dim=-1)

        # 1. 原型-样本对齐损失（fake样本应靠近原型）
        fake_mask = (labels == 1)
        if fake_mask.sum() > 0:
            fake_features = feat_proj[fake_mask]

            # 计算fake样本与所有原型的相似度
            similarity = torch.mm(fake_features, proto_proj.T) / self.temperature
            # [num_fake, K]

            # 软对齐：鼓励fake样本与至少一个原型高度相似
            alignment_loss = -similarity.logsumexp(dim=-1).mean()
            loss_dict['prototype_alignment'] = alignment_loss

        # 2. 原型多样性损失（原型之间应该不同）
        proto_similarity = torch.mm(proto_proj, proto_proj.T)
        K = prototypes.shape[0]
        mask = 1 - torch.eye(K, device=device)
        diversity_loss = (proto_similarity.abs() * mask).sum() / (mask.sum() + 1e-8)
        loss_dict['prototype_diversity'] = diversity_loss

        # 3. 跨域对齐损失（不同域的fake应映射到相同原型空间）
        if domain_labels is not None and fake_mask.sum() > 1:
            fake_features = feat_proj[fake_mask]
            fake_domains = domain_labels[fake_mask]

            unique_domains = fake_domains.unique()
            if len(unique_domains) > 1:
                # 计算不同域fake样本的原型分配
                domain_losses = []
                for domain in unique_domains:
                    domain_mask = (fake_domains == domain)
                    if domain_mask.sum() > 0:
                        domain_feat = fake_features[domain_mask]
                        # 计算该域样本与原型的相似度分布
                        sim = torch.mm(domain_feat, proto_proj.T) / self.temperature
                        soft_assign = F.softmax(sim, dim=-1)
                        # 该域的平均分配
                        domain_mean = soft_assign.mean(dim=0)
                        domain_losses.append(domain_mean)

                if len(domain_losses) > 1:
                    # 所有域的原型分配应该一致（KL散度）
                    domain_losses = torch.stack(domain_losses)  # [num_domains, K]
                    mean_dist = domain_losses.mean(dim=0, keepdim=True)
                    kl_loss = F.kl_div(
                        domain_losses.log(),
                        mean_dist.expand_as(domain_losses),
                        reduction='batchmean'
                    )
                    loss_dict['cross_domain_alignment'] = kl_loss

        # 4. Real-Fake对比损失
        real_mask = (labels == 0)
        if fake_mask.sum() > 0 and real_mask.sum() > 0:
            fake_feat = feat_proj[fake_mask].mean(dim=0, keepdim=True)
            real_feat = feat_proj[real_mask].mean(dim=0, keepdim=True)

            # Real和Fake特征应该远离
            contrast = F.cosine_similarity(fake_feat, real_feat)
            loss_dict['real_fake_contrast'] = contrast.abs().mean()

        return loss_dict


class DomainInvariantPrototype(nn.Module):
    """
    域不变原型模块

    使用动量更新和域混合策略，学习域不变的伪造原型
    """
    def __init__(
        self,
        dim: int,
        num_prototypes: int = 4,
        momentum: float = 0.99
    ):
        super().__init__()

        self.dim = dim
        self.num_prototypes = num_prototypes
        self.momentum = momentum

        # 可学习原型
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim))
        nn.init.xavier_uniform_(self.prototypes)

        # 动量原型（用于稳定训练）
        self.register_buffer('momentum_prototypes', torch.zeros(num_prototypes, dim))

        # 原型使用统计
        self.register_buffer('prototype_usage', torch.zeros(num_prototypes))

    @torch.no_grad()
    def update_momentum_prototypes(self):
        """动量更新原型"""
        self.momentum_prototypes = (
            self.momentum * self.momentum_prototypes +
            (1 - self.momentum) * self.prototypes.data
        )

    def forward(self, features, update_stats=True):
        """
        计算特征与原型的相似度

        Args:
            features: [B, D] 输入特征
            update_stats: 是否更新使用统计

        Returns:
            similarity: [B, K] 相似度分数
            assignments: [B] 硬分配
        """
        # 使用动量原型进行推理（更稳定）
        if self.training:
            protos = self.prototypes
        else:
            protos = self.momentum_prototypes

        # L2归一化
        protos_norm = F.normalize(protos, dim=-1)
        features_norm = F.normalize(features, dim=-1)

        # 计算相似度
        similarity = torch.mm(features_norm, protos_norm.T)  # [B, K]

        # 硬分配
        assignments = similarity.argmax(dim=-1)

        # 更新使用统计
        if update_stats and self.training:
            with torch.no_grad():
                for k in range(self.num_prototypes):
                    self.prototype_usage[k] += (assignments == k).sum().float()

        return similarity, assignments

    def get_usage_stats(self):
        """获取原型使用统计"""
        total = self.prototype_usage.sum()
        if total > 0:
            return self.prototype_usage / total
        return self.prototype_usage


class PatchForgeryAttentionPCR(nn.Module):
    """
    带原型对比正则化的Patch级伪造注意力模块
    """
    def __init__(
        self,
        dim: int,
        num_patches: int = 196,
        num_prototypes: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.dim = dim
        self.num_patches = num_patches
        self.num_prototypes = num_prototypes

        # Query, Key, Value投影
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        # 域不变原型
        self.domain_inv_prototypes = DomainInvariantPrototype(
            dim=dim,
            num_prototypes=num_prototypes,
            momentum=0.99
        )

        # 原型注意力
        self.prototype_attn = nn.MultiheadAttention(
            dim, num_heads=8, dropout=dropout, batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(dim)
        self.scale = dim ** -0.5

    def forward(self, patch_tokens, return_attention=False):
        """
        Args:
            patch_tokens: [B, N, D] patch特征
            return_attention: 是否返回注意力图

        Returns:
            weighted_feat: [B, D]
            forgery_map: [B, N]
            proto_info: dict 原型相关信息
        """
        B, N, D = patch_tokens.shape

        proto_info = {}

        # 获取原型
        prototypes = self.domain_inv_prototypes.prototypes  # [K, D]

        # 投影
        q = self.query(prototypes)  # [K, D]
        k = self.key(patch_tokens)   # [B, N, D]
        v = self.value(patch_tokens) # [B, N, D]

        # 扩展原型到batch
        q = q.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        # 计算注意力
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, K, N]

        # 伪造注意力图（所有原型的平均）
        forgery_map = attn_weights.mean(dim=1)  # [B, N]

        # 加权特征
        weighted_feat = torch.matmul(forgery_map.unsqueeze(1), v).squeeze(1)
        weighted_feat = self.output_proj(weighted_feat)

        # 原型信息
        proto_info['prototypes'] = prototypes
        proto_info['attn_weights'] = attn_weights
        proto_info['usage_stats'] = self.domain_inv_prototypes.get_usage_stats()

        if return_attention:
            return weighted_feat, forgery_map, proto_info

        return weighted_feat, proto_info


class MultiGranularityAggregationPCR(nn.Module):
    """
    带对比约束的多粒度特征聚合模块
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()

        # 各粒度投影
        self.global_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.local_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        self.forgery_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )

        # 自适应权重
        self.weight_net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
            nn.Softmax(dim=-1)
        )

        # 对比投影头
        self.contrast_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 2, 128)
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, cls_token, patch_tokens, forgery_feat):
        """
        Args:
            cls_token: [B, D]
            patch_tokens: [B, N, D]
            forgery_feat: [B, D]

        Returns:
            fused: [B, D]
            weights: [B, 3]
            contrast_feat: [B, 128] 用于对比学习的特征
        """
        # 投影
        global_f = self.global_proj(cls_token)
        local_f = self.local_proj(patch_tokens.mean(dim=1))
        forgery_f = self.forgery_proj(forgery_feat)

        # 计算权重
        concat = torch.cat([global_f, local_f, forgery_f], dim=-1)
        weights = self.weight_net(concat)

        # 加权融合
        fused = (
            weights[:, 0:1] * global_f +
            weights[:, 1:2] * local_f +
            weights[:, 2:3] * forgery_f
        )

        fused = self.fusion(fused)

        # 对比特征（用于监督对比学习）
        contrast_feat = self.contrast_proj(fused)
        contrast_feat = F.normalize(contrast_feat, dim=-1)

        return fused, weights, contrast_feat


class MGFP_PCR(nn.Module):
    """
    Multi-Granularity Forgery Perception with Prototype Contrastive Regularization

    完整模块，包含：
    1. 带PCR的Patch级伪造注意力
    2. 多粒度特征聚合
    3. 原型对比正则化
    4. 层级伪造感知（可选）
    """
    def __init__(
        self,
        dim: int,
        num_patches: int = 196,
        num_prototypes: int = 4,
        use_hierarchical: bool = True,
        use_pcr: bool = True,
        dropout: float = 0.1,
        temperature: float = 0.07
    ):
        super().__init__()

        self.use_hierarchical = use_hierarchical
        self.use_pcr = use_pcr

        # Patch伪造注意力（带PCR）
        self.patch_forgery_attention = PatchForgeryAttentionPCR(
            dim=dim,
            num_patches=num_patches,
            num_prototypes=num_prototypes,
            dropout=dropout
        )

        # 多粒度聚合
        self.multi_granularity_aggregation = MultiGranularityAggregationPCR(
            dim=dim,
            dropout=dropout
        )

        # 原型对比正则化
        if use_pcr:
            self.pcr = PrototypeContrastiveRegularization(
                dim=dim,
                proj_dim=256,
                temperature=temperature
            )

        # 层级伪造感知
        if use_hierarchical:
            self.pool_sizes = [1, 2, 4]
            self.scale_detectors = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim // 2, 1),
                    nn.Sigmoid()
                ) for _ in self.pool_sizes
            ])
            self.scale_fusion = nn.Linear(len(self.pool_sizes), 1)

        self.output_norm = nn.LayerNorm(dim)

    def forward(
        self,
        cls_token: torch.Tensor,
        patch_tokens: torch.Tensor,
        labels: torch.Tensor = None,
        domain_labels: torch.Tensor = None,
        return_attention: bool = False
    ):
        """
        Args:
            cls_token: [B, D]
            patch_tokens: [B, N, D]
            labels: [B] 真假标签（训练时用于PCR）
            domain_labels: [B] 域标签（训练时用于跨域对齐）
            return_attention: 是否返回注意力图

        Returns:
            output: [B, D]
            aux_outputs: dict
        """
        aux_outputs = {}

        # 1. Patch级伪造注意力
        forgery_feat, forgery_map, proto_info = self.patch_forgery_attention(
            patch_tokens, return_attention=True
        )
        aux_outputs['forgery_map'] = forgery_map
        aux_outputs['proto_info'] = proto_info

        # 2. 多粒度聚合
        fused_feat, granularity_weights, contrast_feat = self.multi_granularity_aggregation(
            cls_token, patch_tokens, forgery_feat
        )
        aux_outputs['granularity_weights'] = granularity_weights
        aux_outputs['contrast_feat'] = contrast_feat

        # 3. 原型对比正则化（训练时）
        if self.use_pcr and labels is not None:
            pcr_losses = self.pcr(
                prototypes=proto_info['prototypes'],
                features=fused_feat,
                labels=labels,
                domain_labels=domain_labels
            )
            aux_outputs['pcr_losses'] = pcr_losses

        # 4. 层级伪造感知
        if self.use_hierarchical:
            hierarchical_score, scale_maps = self._hierarchical_perception(patch_tokens)
            aux_outputs['hierarchical_score'] = hierarchical_score
            aux_outputs['scale_maps'] = scale_maps

        # 输出
        output = self.output_norm(fused_feat)

        if return_attention:
            return output, aux_outputs

        return output

    def _hierarchical_perception(self, patch_tokens):
        """层级伪造感知"""
        B, N, D = patch_tokens.shape
        H = W = int(math.sqrt(N))

        feat_2d = patch_tokens.transpose(1, 2).reshape(B, D, H, W)

        scale_scores = []
        scale_maps = []

        for i, pool_size in enumerate(self.pool_sizes):
            pooled = F.adaptive_avg_pool2d(feat_2d, pool_size)
            pooled = pooled.flatten(2).transpose(1, 2)

            scores = self.scale_detectors[i](pooled)
            scale_score = scores.mean(dim=1)
            scale_scores.append(scale_score)

            scale_map = scores.reshape(B, pool_size, pool_size)
            scale_map = F.interpolate(
                scale_map.unsqueeze(1), size=(H, W),
                mode='bilinear', align_corners=False
            )
            scale_maps.append(scale_map.squeeze(1))

        multi_scale_scores = torch.cat(scale_scores, dim=-1)
        final_score = self.scale_fusion(multi_scale_scores)

        return final_score, scale_maps

    def get_pcr_loss(self, aux_outputs):
        """从aux_outputs中获取PCR损失"""
        if 'pcr_losses' in aux_outputs:
            losses = aux_outputs['pcr_losses']
            total = 0
            for k, v in losses.items():
                total = total + v
            return total
        return torch.tensor(0.0)


class SupervisedContrastiveLoss(nn.Module):
    """
    监督对比损失

    用于使同类样本特征接近，不同类样本特征远离
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] L2归一化的特征
            labels: [B] 类别标签

        Returns:
            loss: scalar
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # 计算相似度矩阵
        similarity = torch.mm(features, features.T) / self.temperature

        # 创建标签掩码
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 排除自身
        self_mask = torch.eye(batch_size, device=device)
        mask = mask - self_mask

        # 正样本数量
        positives_count = mask.sum(dim=1)

        # 计算损失
        exp_sim = torch.exp(similarity) * (1 - self_mask)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # 只对有正样本的计算损失
        valid_mask = positives_count > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        mean_log_prob = (mask * log_prob).sum(dim=1) / (positives_count + 1e-8)
        loss = -mean_log_prob[valid_mask].mean()

        return loss


# 测试代码
if __name__ == "__main__":
    print("Testing MGFP-PCR (Prototype Contrastive Regularization)...")

    batch_size = 8
    num_patches = 196
    dim = 1024
    num_domains = 3

    # 模拟输入
    cls_token = torch.randn(batch_size, dim)
    patch_tokens = torch.randn(batch_size, num_patches, dim)
    labels = torch.randint(0, 2, (batch_size,))
    domain_labels = torch.randint(0, num_domains, (batch_size,))

    # 创建MGFP-PCR
    mgfp_pcr = MGFP_PCR(
        dim=dim,
        num_patches=num_patches,
        num_prototypes=4,
        use_hierarchical=True,
        use_pcr=True
    )

    # 前向传播
    output, aux_outputs = mgfp_pcr(
        cls_token, patch_tokens,
        labels=labels,
        domain_labels=domain_labels,
        return_attention=True
    )

    print(f"\nInput shapes:")
    print(f"  cls_token: {cls_token.shape}")
    print(f"  patch_tokens: {patch_tokens.shape}")

    print(f"\nOutput shapes:")
    print(f"  output: {output.shape}")
    print(f"  forgery_map: {aux_outputs['forgery_map'].shape}")
    print(f"  granularity_weights: {aux_outputs['granularity_weights'].shape}")
    print(f"  contrast_feat: {aux_outputs['contrast_feat'].shape}")

    if 'pcr_losses' in aux_outputs:
        print(f"\nPCR Losses:")
        for k, v in aux_outputs['pcr_losses'].items():
            print(f"  {k}: {v.item():.4f}")

    # PCR总损失
    pcr_loss = mgfp_pcr.get_pcr_loss(aux_outputs)
    print(f"\nTotal PCR loss: {pcr_loss.item():.4f}")

    # 原型使用统计
    print(f"\nPrototype usage stats: {aux_outputs['proto_info']['usage_stats']}")

    # 参数统计
    total_params = sum(p.numel() for p in mgfp_pcr.parameters())
    print(f"\nMGFP-PCR parameters: {total_params:,}")

    # 测试监督对比损失
    print("\n" + "="*50)
    print("Testing Supervised Contrastive Loss...")

    scl = SupervisedContrastiveLoss(temperature=0.07)
    contrast_loss = scl(aux_outputs['contrast_feat'], labels)
    print(f"Supervised contrastive loss: {contrast_loss.item():.4f}")

    print("\nAll MGFP-PCR tests passed!")
