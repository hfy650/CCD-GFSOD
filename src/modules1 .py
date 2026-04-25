# refine features 

import torch
from torch import nn
from torch.nn import functional as F
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils import comm, events
from einops import reduce
from torch import distributed as dist


class Refine(nn.Module):
    def __init__(self, input_shapes, num_classes, momentum=0.5, warmup_iters=0, eps=1e-12):
        """
        Centroid-based Feature Calibration Module (CFCM).
        Args:
            input_shapes: dict, e.g. {'res4': ShapeSpec}
            num_classes:  number of semantic centroids K (int)
            momentum:     EMA momentum coefficient β
            warmup_iters: skip centroid update before this iteration
            eps:          numerical stability epsilon
        """
        super().__init__()
        self.momentum = momentum
        self.num_classes = num_classes
        self.warmup_iters = warmup_iters
        self.eps = eps

        # 1×1 conv: projects calibrated features (keeps channel dim)
        # Initialized as identity mapping per channel
        self.fcs = nn.ModuleDict()
        for lvl, shape in input_shapes.items():
            fc = nn.Conv2d(shape.channels, shape.channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
            eye = torch.eye(shape.channels).reshape(shape.channels, shape.channels, 1, 1)
            with torch.no_grad():
                fc.weight.copy_(eye)
            nn.init.zeros_(fc.bias)
            self.fcs[lvl] = fc

        # ROI Align pooler — single level, scale = 1/stride
        self.pooler = ROIPooler(
            output_size=(1, 1),
            scales=[1.0 / s.stride for s in input_shapes.values()],
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        # Global centroids G = {g_1, ..., g_K}, shape (K, C)
        # Corresponds to paper Section 3.3, initialised to eps to avoid zero norm
        dim = list(input_shapes.values())[0].channels
        self.register_buffer("centroids", torch.zeros(num_classes, dim) + eps)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------
    @property
    def iterations(self):
        if not hasattr(self, "storage"):
            self.storage = events.get_event_storage()
        return self.storage.iter

    # ------------------------------------------------------------------
    # Phase 4 of Algorithm 1: Global Centroid Update (EMA)
    # Implements Eq.(5):  g_k ← (1-β)·g_k + β·l_k
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update_centroids(self, features: dict, proposals):
        """
        Update global centroids G via EMA using GT-box RoI features.

        Args:
            features:  dict {'res4': Tensor(B, C, H, W)}
            proposals: list[Instances], each with attribute `gt_boxes`
        """
        if self.momentum == 0 or self.iterations < self.warmup_iters:
            return

        # ---- ROI Align → RoI features f_i  (Phase 1 of Alg.1) --------
        features_list = [features[lvl] for lvl in sorted(features.keys())]
        gt_boxes = [x.gt_boxes for x in proposals]

        pooled = self.pooler(features_list, gt_boxes)   # (N, C, 1, 1)
        pooled = pooled.flatten(start_dim=1)            # (N, C)

        if pooled.numel() == 0:
            return

        # ---- Assign each RoI feature to its nearest centroid -----------
        # Eq.(1): k = argmax_{k} cosine_sim(f_i, g_k)
        # NOTE: No dropout here — paper does not mention it.
        f_norm = F.normalize(pooled, dim=1)             # (N, C)
        g_norm = F.normalize(self.centroids, dim=1)     # (K, C)
        sims   = f_norm.matmul(g_norm.T)                # (N, K)

        # One-hot assignment mask, shape (N, K)
        mask = torch.zeros_like(sims).scatter(
            1, sims.argmax(dim=1, keepdim=True), 1.0
        )

        # ---- Local centroid l_k (Phase 2 of Alg.1) --------------------
        # Eq.(2): l_k = (1/N_k) Σ_{i∈Ω_k} f_i
        sum_x  = mask.T.matmul(pooled)          # (K, C)
        count  = mask.sum(dim=0).unsqueeze(1)   # (K, 1)

        # ---- Cross-process aggregation (distributed training) ----------
        world_size = comm.get_world_size()
        if world_size > 1:
            sum_x_gather   = [torch.empty_like(sum_x)  for _ in range(world_size)]
            count_gather   = [torch.empty_like(count)   for _ in range(world_size)]
            dist.all_gather(sum_x_gather,  sum_x)
            dist.all_gather(count_gather,  count)
            sum_x = torch.stack(sum_x_gather, dim=0).sum(dim=0)
            count = torch.stack(count_gather, dim=0).sum(dim=0)

        # l_k for each partition
        centroids_local = sum_x / count.clamp_min(1)   # (K, C)

        # ---- EMA update (Phase 4 of Alg.1) ----------------------------
        # Eq.(5): g_k ← (1-β)·g_k + β·l_k
        # Only partitions with at least one sample are updated.
        beta   = (count > 0).float() * self.momentum   # (K, 1)
        beta   = beta.expand_as(centroids_local)        # (K, C)
        updated = (1.0 - beta) * self.centroids + beta * centroids_local
        self.centroids.copy_(updated)

    # ------------------------------------------------------------------
    # Phase 3 of Algorithm 1: Pixel-level Calibration
    # Implements Eq.(3)(4):
    #   w_i = exp(-||l_k - x_i||² / c)
    #   x'_i = x_i + w_i · (l_k - x_i)
    # ------------------------------------------------------------------
    def forward(self, features: dict):
        """
        Calibrate spatial features using (local) centroid-based offset.

        During warmup the module acts as a plain relu+conv identity.
        After warmup:
          1. Assign each spatial location to its nearest global centroid (Eq.1).
          2. Compute batch-wise local centroid per region (Eq.2).
          3. Calibrate pixel features with adaptive weight (Eq.3,4).
          4. Pass through 1×1 conv + ReLU.

        Returns:
            dict with same keys as `features`, values are calibrated maps F*.
        """
        outputs = {}
        for lvl, x in features.items():
            B, C, H, W = x.shape

            # Warmup: skip calibration, only apply conv
            if self.training and self.iterations < self.warmup_iters:
                outputs[lvl] = F.relu(self.fcs[lvl](x))
                continue

            # ----------------------------------------------------------
            # Step 1 – Assign spatial positions to nearest global centroid
            # Eq.(1): k = argmax cosine_sim(x_i, g_k)
            # ----------------------------------------------------------
            # sim: (B, K, H, W)
            sim = torch.einsum(
                "bchw, nc -> bnhw",
                F.normalize(x, dim=1),
                F.normalize(self.centroids, dim=1),
            )
            # Hard assignment mask: (B, K, H, W)
            assignment = torch.zeros_like(sim).scatter(
                1, sim.argmax(dim=1, keepdim=True), 1.0
            )

            # ----------------------------------------------------------
            # Step 2 – Local centroid l_k per spatial region
            # Eq.(2): l_k = (1/N_k) Σ_{i∈Ω_k} x_i
            # sum_x : (B, K, C),  count: (B, K, 1)
            # ----------------------------------------------------------
            sum_x           = torch.einsum("bnhw, bchw -> bnc", assignment, x)
            count           = reduce(assignment, "b n h w -> b n ()", "sum")   # (B,K,1)
            centroids_local = sum_x / count.clamp_min(1)                       # (B,K,C)

            # ----------------------------------------------------------
            # Step 3 – Pixel-level calibration
            # Reconstruct l_k at every spatial position: (B, C, H, W)
            # delta  = l_k - x_i
            # w_i    = exp(-||delta||² / c)       ← Eq.(4)
            # x'_i   = x_i + w_i * delta          ← Eq.(3)
            # ----------------------------------------------------------
            # Broadcast local centroid back to spatial map
            lk_map = torch.einsum("bnhw, bnc -> bchw", assignment, centroids_local)

            delta  = lk_map - x                                      # (B, C, H, W)

            # Eq.(4): w_i = exp(-||l_k - x_i||² / c)
            # ||delta||²/c  == delta².mean(dim=1, keepdim=True)  [numerically identical]
            w      = torch.exp(-delta.pow(2).mean(dim=1, keepdim=True))  # (B,1,H,W)

            # Eq.(3): calibrated feature
            x_cal  = x + w * delta                                   # (B, C, H, W)

            # 1×1 conv + ReLU (feature projection)
            outputs[lvl] = F.relu(self.fcs[lvl](x_cal))

        return outputs
