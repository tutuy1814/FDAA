from .detector import AIGCDetector, AIGCDetectorV2, AIGCDetectorV3, AIGCDetectorLite
from .modules.fdaa import (
    FDAA, FDAAv1, FDAAv2, FDAAv3,
    SpatialAdapter, FrequencyAdapter, CrossDomainInteraction,
    RealFFTLayerV2, LearnableSRMConv2d,
)
from .modules.mgfp import (
    MGFP, MGFPv1, MGFPv2, MGFPv3,
    PatchForgeryAttention, MultiGranularityAggregation,
    FreqGuidedAttentionPooling, GatedFusion,
    HierarchicalForgeryPerception,
)
from .losses.losses import (
    FocalLoss, ContrastiveLoss, SourceAwareContrastiveLoss,
    AIGCDetectionLoss,
)

__all__ = [
    # Detectors
    'AIGCDetector', 'AIGCDetectorV2', 'AIGCDetectorV3', 'AIGCDetectorLite',
    # FDAA
    'FDAA', 'FDAAv1', 'FDAAv2', 'FDAAv3',
    'SpatialAdapter', 'FrequencyAdapter', 'CrossDomainInteraction',
    'RealFFTLayerV2', 'LearnableSRMConv2d',
    # MGFP
    'MGFP', 'MGFPv1', 'MGFPv2', 'MGFPv3',
    'PatchForgeryAttention', 'MultiGranularityAggregation',
    'FreqGuidedAttentionPooling', 'GatedFusion',
    'HierarchicalForgeryPerception',
    # Losses
    'FocalLoss', 'ContrastiveLoss', 'SourceAwareContrastiveLoss',
    'AIGCDetectionLoss',
]
