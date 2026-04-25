
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicSemanticMarginLoss(nn.Module):
    """
    Dynamic Semantic-aware Margin Loss.

    This implementation accepts both positional and keyword arguments for:
        forward(logits, labels, roi_feats, bg_class_ind=None, external_prototypes=None, **kwargs)

    Returns:
        loss (scalar), modified_logits (R, K+1)
    """
    def __init__(self, gamma: float = 0.15, topk: int = 2, margin_scale: float = 0.25,
                 eps: float = 1e-8, reduction: str = "mean"):
        super().__init__()
        self.gamma = float(gamma)
        self.topk = int(topk)
        self.margin_scale = float(margin_scale)
        self.eps = float(eps)
        assert reduction in ("mean", "sum"), "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, roi_feats: torch.Tensor,
                bg_class_ind: int = None, external_prototypes: torch.Tensor = None, **kwargs):
        """
        Compute the dynamic margin CE loss.

        Args:
          logits: (R, K+1) raw logits
          labels: (R,) labels in [0, K] where K is background
          roi_feats: (R, C)
          bg_class_ind: optional background index
          external_prototypes: optional (K+1, C) prototypes (e.g., EMA). If provided, use them.
        Returns:
          loss (scalar), modified_logits (R, K+1)
        """
        device = logits.device
        if bg_class_ind is None:
            num_classes = logits.shape[1]
            bg_class_ind = num_classes - 1
        R, num_classes = logits.shape

        # Basic checks & normalize roi_feats shape
        if roi_feats is None:
            raise RuntimeError("DynamicSemanticMarginLoss requires roi_feats (R, C).")
        if roi_feats.dim() != 2:
            roi_feats = roi_feats.view(roi_feats.size(0), -1)
        if roi_feats.size(0) != R:
            raise RuntimeError(f"roi_feats first dim ({roi_feats.size(0)}) != logits first dim ({R}).")

        C = roi_feats.size(1)

        # ---------------- Step 1: Prototypes (use external_prototypes if provided) ----------------
        if external_prototypes is not None:
            # validate external_prototypes
            if external_prototypes.shape[0] != num_classes or external_prototypes.shape[1] != C:
                raise RuntimeError(f"external_prototypes shape {external_prototypes.shape} incompatible with expected ({num_classes},{C})")
            prototypes = external_prototypes.to(device=device, dtype=roi_feats.dtype)
            # build a counts mask from prototype norm (to mark invalid prototypes)
            counts = (prototypes.norm(p=2, dim=1) > self.eps).float()
        else:
            prototypes = torch.zeros((num_classes, C), device=device, dtype=roi_feats.dtype)
            counts = torch.zeros((num_classes,), device=device, dtype=roi_feats.dtype)

            fg_mask = (labels >= 0) & (labels < bg_class_ind)
            if fg_mask.any():
                fg_idx = fg_mask.nonzero(as_tuple=False).squeeze(1)
                fg_labels = labels[fg_idx]
                fg_feats = roi_feats[fg_idx]
                prototypes = prototypes.index_add(0, fg_labels, fg_feats)
                counts = counts.index_add(0, fg_labels, torch.ones_like(fg_labels, dtype=roi_feats.dtype))

            nonzero = counts > 0
            if nonzero.any():
                prototypes[nonzero] = prototypes[nonzero] / counts[nonzero].unsqueeze(1)
            # prototypes with no samples remain zero

        # ---------------- Step 2: cosine similarities and candidate mask ----------------
        proto_norm = F.normalize(prototypes, p=2, dim=1, eps=self.eps)   # (K+1, C)
        roi_norm = F.normalize(roi_feats, p=2, dim=1, eps=self.eps)       # (R, C)

        proto_l2 = prototypes.norm(p=2, dim=1)
        invalid_proto_mask = (proto_l2 <= self.eps)
        sim_matrix = roi_norm @ proto_norm.t()                            # (R, K+1)
        if invalid_proto_mask.any():
            sim_matrix[:, invalid_proto_mask] = -1e6

        idx = torch.arange(num_classes, device=device).unsqueeze(0)       # (1, K+1)
        gt_mask = (idx == labels.unsqueeze(1))                            # (R, K+1)
        bg_mask = (idx == bg_class_ind).expand(R, -1)                     # (R, K+1)

        candidate_mask = (sim_matrix - self.gamma) > 0.0
        candidate_mask = candidate_mask & (~gt_mask) & (~bg_mask)

        # ---------------- Step 3 & 4: select top-k and compute margins ----------------
        margins = torch.zeros_like(logits, device=device)                  # (R, K+1)
        if self.topk > 0:
            sims_for_topk = sim_matrix.clone()
            sims_for_topk[~candidate_mask] = -1e6
            # ensure k <= num_classes
            k = min(self.topk, sims_for_topk.shape[1])
            topk_vals, topk_idx = torch.topk(sims_for_topk, k=k, dim=1, largest=True, sorted=True)
            valid = topk_vals > (-1e5)
            selected_margins = self.margin_scale * (topk_vals - self.gamma)
            selected_margins = selected_margins * valid.float()
            margins = margins.scatter(1, topk_idx, selected_margins)
        else:
            margins = margins + (self.margin_scale * (sim_matrix - self.gamma) * candidate_mask.float())

        # ---------------- Step 4 applied: add margins to logits ----------------
        modified_logits = logits + margins

        # ---------------- Step 5: CE on modified logits ----------------
        loss = F.cross_entropy(modified_logits, labels, reduction=self.reduction)
        return loss
