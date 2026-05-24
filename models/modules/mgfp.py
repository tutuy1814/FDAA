
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# V1 组件 (保留兼容性)
# =============================================================================

class PatchForgeryAttention(nn.Module):
    """Patch级伪造注意力模块 (V1)"""
    def __init__(self, dim, num_patches=196, num_prototypes=4, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_patches = num_patches
        self.num_prototypes = num_prototypes

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.forgery_prototypes = nn.Parameter(torch.randn(num_prototypes, dim))
        nn.init.xavier_uniform_(self.forgery_prototypes)

        self.prototype_attn = nn.MultiheadAttention(dim, num_heads=8, dropout=dropout, batch_first=True)

        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.norm = nn.LayerNorm(dim)
        self.scale = dim ** -0.5

    def forward(self, patch_tokens, return_attention=False):
        B, N, D = patch_tokens.shape

        q = self.query(self.forgery_prototypes)
        k = self.key(patch_tokens)
        v = self.value(patch_tokens)

        q = q.unsqueeze(0).expand(B, -1, -1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        forgery_map = attn_weights.mean(dim=1)

        weighted_feat = torch.matmul(forgery_map.unsqueeze(1), v).squeeze(1)
        weighted_feat = self.output_proj(weighted_feat)

        if return_attention:
            return weighted_feat, forgery_map
        return weighted_feat


class MultiGranularityAggregation(nn.Module):
    """多粒度特征聚合模块 (V1)"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()

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

        self.weight_net = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
            nn.Softmax(dim=-1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, cls_token, patch_tokens, forgery_feat):
        global_f = self.global_proj(cls_token)
        local_f = self.local_proj(patch_tokens.mean(dim=1))
        forgery_f = self.forgery_proj(forgery_feat)

        concat = torch.cat([global_f, local_f, forgery_f], dim=-1)
        weights = self.weight_net(concat)

        fused = (weights[:, 0:1] * global_f +
                 weights[:, 1:2] * local_f +
                 weights[:, 2:3] * forgery_f)

        fused = self.fusion(fused)

        return fused, weights


class HierarchicalForgeryPerception(nn.Module):
    """层级伪造感知模块 (论文贡献点，V1/V2共用)"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        self.dim = dim
        self.out_dim = dim // 4
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

        # 输出 dim//4 维向量而非标量，提供更丰富的辅助监督信号
        self.scale_fusion = nn.Sequential(
            nn.Linear(len(self.pool_sizes), dim // 4),
            nn.GELU(),
        )

    def forward(self, patch_tokens):
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
            scale_map = F.interpolate(scale_map.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False)
            scale_maps.append(scale_map.squeeze(1))

        multi_scale_scores = torch.cat(scale_scores, dim=-1)  # [B, 3]
        final_score = self.scale_fusion(multi_scale_scores)    # [B, dim//4]

        return final_score, scale_maps


class MGFP(nn.Module):
    """Multi-Granularity Forgery Perception Module (V1)"""
    def __init__(self, dim, num_patches=196, num_prototypes=4, use_hierarchical=True, dropout=0.1):
        super().__init__()

        self.patch_forgery_attention = PatchForgeryAttention(
            dim=dim, num_patches=num_patches,
            num_prototypes=num_prototypes, dropout=dropout
        )

        self.multi_granularity_aggregation = MultiGranularityAggregation(
            dim=dim, dropout=dropout
        )

        self.use_hierarchical = use_hierarchical
        if use_hierarchical:
            self.hierarchical_perception = HierarchicalForgeryPerception(
                dim=dim, dropout=dropout
            )

        self.output_norm = nn.LayerNorm(dim)

    def forward(self, cls_token, patch_tokens, return_attention=False):
        aux_outputs = {}

        forgery_feat, forgery_map = self.patch_forgery_attention(
            patch_tokens, return_attention=True
        )
        aux_outputs['forgery_map'] = forgery_map

        fused_feat, granularity_weights = self.multi_granularity_aggregation(
            cls_token, patch_tokens, forgery_feat
        )
        aux_outputs['granularity_weights'] = granularity_weights

        if self.use_hierarchical:
            hierarchical_score, scale_maps = self.hierarchical_perception(patch_tokens)
            aux_outputs['hierarchical_score'] = hierarchical_score
            aux_outputs['scale_maps'] = scale_maps

        output = self.output_norm(fused_feat)

        if return_attention:
            return output, aux_outputs
        return output


MGFPv1 = MGFP  # 别名


# =============================================================================
# V2 组件 (改进版)
# =============================================================================

class AttentionPooling(nn.Module):
    """
    可学习注意力池化 (V2)

    使用学习的查询向量对 patch tokens 进行注意力加权聚合。
    替代随机初始化原型，使用 trunc_normal_ 初始化的单一查询。
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim

        # 可学习查询向量
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        # 多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, patch_tokens):
        """
        Args:
            patch_tokens: [B, N, D]
        Returns:
            pooled: [B, D] 注意力聚合后的局部特征
            attn_weights: [B, 1, N] 注意力权重 (可用于可视化)
        """
        B = patch_tokens.shape[0]

        # 扩展查询到batch
        query = self.query.expand(B, -1, -1)  # [B, 1, D]

        # 注意力聚合: query attend to all patches
        pooled, attn_weights = self.attn(
            query, patch_tokens, patch_tokens
        )  # pooled: [B, 1, D], attn_weights: [B, 1, N]

        pooled = self.norm(pooled.squeeze(1))  # [B, D]

        return pooled, attn_weights.squeeze(1)  # [B, D], [B, N]


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块 (V2核心)

    融合三种特征:
    1. global_feat: CLS token (ViT 全局语义)
    2. local_feat: AttentionPooling 输出 (局部关注特征)
    3. freq_feat: FDAAv2 频率分支输出 (频率域特征)

    使用交叉注意力 + 门控融合，替代简单 MLP concat。
    频率特征作为 Query，语义特征作为 KV，实现跨模态交互。
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()

        # 各分支归一化
        self.norm_g = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)
        self.norm_f = nn.LayerNorm(dim)

        # 交叉注意力：频率特征 attend to 语义特征
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(dim)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, global_feat, local_feat, freq_feat):
        """
        Args:
            global_feat: [B, D] CLS token
            local_feat: [B, D] AttentionPooling 输出
            freq_feat: [B, D] 频率分支输出
        Returns:
            fused: [B, D] 融合后的特征
        """
        # 归一化
        g = self.norm_g(global_feat)
        l = self.norm_l(local_feat)
        f = self.norm_f(freq_feat)

        # 交叉注意力：freq_feat 作为 Query, [global, local] 作为 KV
        # 将三路特征堆叠为序列
        semantic_kv = torch.stack([g, l], dim=1)    # [B, 2, D]
        freq_q = f.unsqueeze(1)                      # [B, 1, D]

        cross_out, _ = self.cross_attn(freq_q, semantic_kv, semantic_kv)  # [B, 1, D]
        cross_out = self.cross_norm(f + cross_out.squeeze(1))  # [B, D] 残差连接

        # 门控融合
        concat = torch.cat([g, l, cross_out], dim=-1)  # [B, 3D]
        gate_weights = self.gate(concat)                # [B, D]

        # 门控加权 + 投影
        gated = gate_weights * cross_out + (1 - gate_weights) * (g + l) * 0.5
        fused = self.out_proj(concat) + gated           # 残差连接

        return fused


# 保留旧名称的别名
LateFusionAggregation = CrossAttentionFusion


class MGFPv2(nn.Module):
    """
    Multi-Granularity Forgery Perception Module V2

    关键改进：
    1. AttentionPooling 替代随机原型 (PatchForgeryAttention)
    2. CrossAttentionFusion 替代简单 MLP concat
    3. 保留 HierarchicalForgeryPerception (论文贡献点)
    4. embed_dim 默认 1024，匹配 CLIP ViT-L/14

    用法：
        mgfp_v2 = MGFPv2(dim=1024)
        output, aux = mgfp_v2(cls_token, patch_tokens, freq_feat, return_attention=True)
    """
    def __init__(self, dim=1024, num_patches=256, use_hierarchical=True, num_heads=8, dropout=0.1):
        super().__init__()

        # 注意力池化 (替代随机原型)
        self.attention_pooling = AttentionPooling(
            dim=dim, num_heads=num_heads, dropout=dropout
        )

        # 交叉注意力融合 (替代简单 MLP concat)
        self.late_fusion = CrossAttentionFusion(
            dim=dim, num_heads=num_heads, dropout=dropout
        )

        # 层级伪造感知 (论文贡献点，保留)
        self.use_hierarchical = use_hierarchical
        if use_hierarchical:
            self.hierarchical_perception = HierarchicalForgeryPerception(
                dim=dim, dropout=dropout
            )

        self.output_norm = nn.LayerNorm(dim)

    def forward(self, cls_token, patch_tokens, freq_feat, return_attention=False):
        """
        Args:
            cls_token: [B, D] CLS token (from frozen ViT)
            patch_tokens: [B, N, D] patch tokens (from frozen ViT)
            freq_feat: [B, D] 频率分支特征 (from FDAAv2)
            return_attention: 是否返回辅助输出
        Returns:
            output: [B, D] 最终融合特征
            aux_outputs: dict (optional)
        """
        aux_outputs = {}

        # 1. 注意力池化: 从patch tokens提取局部关注特征
        local_feat, attn_weights = self.attention_pooling(patch_tokens)
        aux_outputs['attention_map'] = attn_weights  # [B, N] 可用于可视化
        # 为兼容性，也提供 forgery_map 键
        aux_outputs['forgery_map'] = attn_weights

        # 2. Late Fusion: 融合全局 + 局部 + 频率
        fused_feat = self.late_fusion(cls_token, local_feat, freq_feat)

        # 3. 层级伪造感知 (可选)
        if self.use_hierarchical:
            hierarchical_score, scale_maps = self.hierarchical_perception(patch_tokens)
            aux_outputs['hierarchical_score'] = hierarchical_score
            aux_outputs['scale_maps'] = scale_maps

        # 输出
        output = self.output_norm(fused_feat)

        if return_attention:
            return output, aux_outputs
        return output


# =============================================================================
# V3 组件 (频率引导注意力池化 + 空间融合)
# =============================================================================

class FreqGuidedAttentionPooling(nn.Module):
    """
    频率引导的注意力池化 (V3核心)

    freq_tokens cross-attend to patch_tokens，计算逐 patch 异常分数，
    然后用异常分数加权聚合 patch_tokens。

    关键区别：
    - V2 的 AttentionPooling 使用可学习查询（与 FDAA 无关）
    - V3 使用 freq_tokens 作为查询，将频率证据注入空间聚合过程
    - 消融时无 freq_tokens → 退化为均匀注意力，Full 有频率引导 → 产生明确差异
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim

        # 频率 → 空间 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        # 异常分数投影: 将 cross-attention 输出映射为标量分数
        self.anomaly_score = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1),
        )

        self.norm = nn.LayerNorm(dim)

        # Fallback: 无 freq_tokens 时的可学习查询
        self.fallback_query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.fallback_query, std=0.02)

    def forward(self, patch_tokens, freq_tokens=None):
        """
        Args:
            patch_tokens: [B, N, D] CLIP patch tokens
            freq_tokens: [B, N, D] 频率 tokens (from FDAAv3)，可选
        Returns:
            pooled: [B, D] 加权聚合后的局部特征
            attn_weights: [B, N] 逐 patch 异常分数（可用于可视化）
        """
        B, N, D = patch_tokens.shape

        if freq_tokens is not None:
            # 频率引导: freq_tokens attend to patch_tokens
            cross_out, _ = self.cross_attn(
                freq_tokens, patch_tokens, patch_tokens
            )  # [B, N, D]

            # 计算逐 patch 异常分数
            scores = self.anomaly_score(cross_out).squeeze(-1)  # [B, N]
            attn_weights = torch.softmax(scores, dim=-1)  # [B, N]
        else:
            # Fallback: 无频率引导，使用可学习查询
            query = self.fallback_query.expand(B, -1, -1)
            fallback_out, raw_attn = self.cross_attn(
                query, patch_tokens, patch_tokens
            )  # [B, 1, D], [B, 1, N]
            attn_weights = raw_attn.squeeze(1)  # [B, N]

        # 加权聚合
        pooled = torch.einsum('bn,bnd->bd', attn_weights, patch_tokens)  # [B, D]
        pooled = self.norm(pooled)

        return pooled, attn_weights


class GatedFusion(nn.Module):
    """
    门控融合模块 (V3)

    替代 V2 的 CrossAttentionFusion（其 freq_feat 作为 Q 只 attend 2 个 KV token，退化操作）。
    使用简单高效的门控机制融合 global + local + freq_global。
    """
    def __init__(self, dim, dropout=0.1):
        super().__init__()

        # 各分支归一化
        self.norm_g = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)
        self.norm_f = nn.LayerNorm(dim)

        # 门控权重
        self.gate_net = nn.Sequential(
            nn.Linear(dim * 3, dim * 3),
            nn.GELU(),
            nn.Linear(dim * 3, 3),
            nn.Softmax(dim=-1),
        )

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, global_feat, local_feat, freq_global):
        """
        Args:
            global_feat: [B, D] CLS token
            local_feat: [B, D] FreqGuidedAttentionPooling 输出
            freq_global: [B, D] 全局频率特征 (from FDAAv3)
        Returns:
            fused: [B, D] 融合后的特征
        """
        g = self.norm_g(global_feat)
        l = self.norm_l(local_feat)
        f = self.norm_f(freq_global)

        # 计算门控权重
        concat = torch.cat([g, l, f], dim=-1)  # [B, 3D]
        weights = self.gate_net(concat)  # [B, 3]

        # 加权融合
        fused = (weights[:, 0:1] * g +
                 weights[:, 1:2] * l +
                 weights[:, 2:3] * f)

        # 投影
        fused = self.out_proj(fused) + fused  # 残差连接

        return fused


class MGFPv3(nn.Module):
    """
    Multi-Granularity Forgery Perception Module V3

    关键改进：
    1. FreqGuidedAttentionPooling 替代 AttentionPooling — 频率引导空间聚合
    2. GatedFusion 替代 CrossAttentionFusion — 去掉退化的 cross-attn
    3. 消融时 baseline+MGFP（无 freq_tokens）退化为 fallback 查询池化
       Full 有频率引导 → 产生明确差异（解决 V2 的 FDAA 贡献被掩盖问题）
    4. 保留 HierarchicalForgeryPerception (论文贡献点)

    用法：
        mgfp_v3 = MGFPv3(dim=1024)
        output, aux = mgfp_v3(cls, patches, freq_tokens, freq_global, return_attention=True)
    """
    def __init__(self, dim=1024, num_patches=256, use_hierarchical=True,
                 num_heads=8, dropout=0.1):
        super().__init__()

        # 频率引导注意力池化
        self.freq_guided_pooling = FreqGuidedAttentionPooling(
            dim=dim, num_heads=num_heads, dropout=dropout
        )

        # 门控融合
        self.gated_fusion = GatedFusion(
            dim=dim, dropout=dropout
        )

        # 层级伪造感知
        self.use_hierarchical = use_hierarchical
        if use_hierarchical:
            self.hierarchical_perception = HierarchicalForgeryPerception(
                dim=dim, dropout=dropout
            )

        self.output_norm = nn.LayerNorm(dim)

    def forward(self, cls_token, patch_tokens, freq_tokens=None,
                freq_global=None, return_attention=False):
        """
        Args:
            cls_token: [B, D] CLS token (from frozen ViT)
            patch_tokens: [B, N, D] patch tokens (from frozen ViT)
            freq_tokens: [B, N, D] 频率 tokens (from FDAAv3)，可选
            freq_global: [B, D] 全局频率特征 (from FDAAv3)，可选
            return_attention: 是否返回辅助输出
        Returns:
            output: [B, D] 最终融合特征
            aux_outputs: dict (optional)
        """
        aux_outputs = {}

        # 1. 频率引导注意力池化
        local_feat, attn_weights = self.freq_guided_pooling(
            patch_tokens, freq_tokens=freq_tokens
        )
        aux_outputs['attention_map'] = attn_weights  # [B, N]
        aux_outputs['forgery_map'] = attn_weights

        # 2. 门控融合: global + local + freq_global
        if freq_global is None:
            freq_global = torch.zeros_like(cls_token)
        fused_feat = self.gated_fusion(cls_token, local_feat, freq_global)

        # 3. 层级伪造感知
        if self.use_hierarchical:
            hierarchical_score, scale_maps = self.hierarchical_perception(patch_tokens)
            aux_outputs['hierarchical_score'] = hierarchical_score
            aux_outputs['scale_maps'] = scale_maps

        # 输出
        output = self.output_norm(fused_feat)

        if return_attention:
            return output, aux_outputs
        return output


# =============================================================================
# 可视化工具 (V1/V2/V3共用)
# =============================================================================

class ForgeryMapVisualizer:
    """伪造注意力图可视化工具"""

    @staticmethod
    def visualize_forgery_map(forgery_map, image=None, save_path=None):
        import matplotlib.pyplot as plt

        if forgery_map.dim() == 1:
            H = W = int(math.sqrt(forgery_map.shape[0]))
            forgery_map = forgery_map.reshape(H, W)

        forgery_map = forgery_map.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2 if image is not None else 1, figsize=(10, 5))

        if image is not None:
            img = image.permute(1, 2, 0).cpu().numpy()
            img = (img - img.min()) / (img.max() - img.min())
            axes[0].imshow(img)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            im = axes[1].imshow(forgery_map, cmap='jet', interpolation='bilinear')
            axes[1].set_title('Forgery Attention Map')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1])
        else:
            im = axes.imshow(forgery_map, cmap='jet', interpolation='bilinear')
            axes.set_title('Forgery Attention Map')
            axes.axis('off')
            plt.colorbar(im, ax=axes)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    batch_size = 4
    num_patches_v1 = 196  # 14x14 (patch_size=16)
    dim_v1 = 768

    cls_token_v1 = torch.randn(batch_size, dim_v1)
    patch_tokens_v1 = torch.randn(batch_size, num_patches_v1, dim_v1)

    print("=" * 60)
    print("Testing MGFP V1 (dim=768, patches=196)...")
    mgfp_v1 = MGFP(dim=dim_v1, num_patches=num_patches_v1, num_prototypes=4, use_hierarchical=True)
    output_v1, aux_v1 = mgfp_v1(cls_token_v1, patch_tokens_v1, return_attention=True)
    print(f"V1 Output shape: {output_v1.shape}")
    print(f"V1 Forgery map shape: {aux_v1['forgery_map'].shape}")
    print(f"V1 parameters: {sum(p.numel() for p in mgfp_v1.parameters()):,}")

    print()
    print("=" * 60)
    print("Testing MGFP V2 (dim=1024, patches=256, CrossAttentionFusion)...")
    dim_v2 = 1024  # CLIP ViT-L/14 内部维度
    num_patches_v2 = 256  # 16x16 (CLIP ViT-L/14 patch_size=14, 224/14=16)

    cls_token_v2 = torch.randn(batch_size, dim_v2)
    patch_tokens_v2 = torch.randn(batch_size, num_patches_v2, dim_v2)
    freq_feat = torch.randn(batch_size, dim_v2)  # 模拟 FDAAv2 输出

    mgfp_v2 = MGFPv2(dim=dim_v2, num_patches=num_patches_v2, use_hierarchical=True)
    output_v2, aux_v2 = mgfp_v2(cls_token_v2, patch_tokens_v2, freq_feat, return_attention=True)
    print(f"V2 Output shape: {output_v2.shape}")
    print(f"V2 Attention map shape: {aux_v2['attention_map'].shape}")
    print(f"V2 parameters: {sum(p.numel() for p in mgfp_v2.parameters()):,}")

    # Verify grad flow
    loss = output_v2.sum()
    loss.backward()
    query_grad = mgfp_v2.attention_pooling.query.grad
    print(f"V2 Query grad exists: {query_grad is not None}")

    print()
    print("=" * 60)
    print("Testing MGFP V3 (freq-guided pooling + gated fusion)...")

    mgfp_v3 = MGFPv3(dim=dim_v2, num_patches=num_patches_v2, use_hierarchical=True)
    freq_tokens_v3 = torch.randn(batch_size, num_patches_v2, dim_v2)
    freq_global_v3 = torch.randn(batch_size, dim_v2)

    output_v3, aux_v3 = mgfp_v3(
        cls_token_v2, patch_tokens_v2,
        freq_tokens=freq_tokens_v3, freq_global=freq_global_v3,
        return_attention=True
    )
    print(f"V3 Output shape: {output_v3.shape}")
    print(f"V3 Attention map shape: {aux_v3['attention_map'].shape}")
    print(f"V3 parameters: {sum(p.numel() for p in mgfp_v3.parameters()):,}")

    # Without freq_tokens (ablation mode)
    output_v3_nf, _ = mgfp_v3(
        cls_token_v2, patch_tokens_v2,
        freq_tokens=None, freq_global=None,
        return_attention=True
    )
    print(f"V3 Output (no freq) shape: {output_v3_nf.shape}")

    loss_v3 = output_v3.sum()
    loss_v3.backward()
    print(f"V3 Grad flow OK: {mgfp_v3.freq_guided_pooling.anomaly_score[0].weight.grad is not None}")
