from defrcn.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN, build_model
from defrcn.modeling.roi_heads import (
    ROI_BOX_HEAD_REGISTRY, ROI_HEADS_REGISTRY, ROIHeads, StandardROIHeads, build_box_head,
    build_roi_heads)
