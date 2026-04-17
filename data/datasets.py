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
        Args:
            input_shapes: dict, {level: ShapeSpec}, e.g. {"p2": ShapeSpec(...), ...}
            num_classes: 类别数
        """
        super().__init__()
        self.momentum = momentum
        self.num_classes = num_classes
        self.warmup_iters = warmup_iters
        self.eps = eps

        # 为每层建一个 1x1 conv
        self.fcs = nn.ModuleDict()
        for lvl, shape in input_shapes.items():
            fc = nn.Conv2d(shape.channels, shape.channels, 1, 1, 0, bias=True)
            fc.weight.data = torch.eye(shape.channels).reshape(shape.channels, -1, 1, 1)
            nn.init.zeros_(fc.bias.data)
            self.fcs[lvl] = fc

        # ROI Pooler (支持多层)
        self.pooler = ROIPooler(
            output_size=(1, 1),
            scales=[1.0 / s.stride for s in input_shapes.values()],
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        # 全局质心
        dim = list(input_shapes.values())[0].channels
        self.register_buffer("centroids", torch.zeros(num_classes, dim) + eps)

    @property
    def iterations(self):
        if not hasattr(self, "storage"):
            self.storage = events.get_event_storage()
        return self.storage.iter

    @torch.no_grad()
    def update_centroids(self, features: dict, proposals):
        """
        features: 原始 FPN 特征图 dict (e.g. {"p2": Tensor, "p3": Tensor, ...})
        proposals: list[Instances]，含 gt_boxes（每个 Instances 对应一张图片）
        """
        if self.momentum == 0 or self.iterations < self.warmup_iters:
            return

        # --- 1) 把 features dict 转成 按层顺序的 list ---
        # 安全排序：常见 keys 是 "p2","p3"...，这里尝试提取数字确保正确顺序
        keys = list(features.keys())
        try:
            # 如果 keys 形如 "p2","p3"... 按数字排序
            keys_sorted = sorted(keys, key=lambda k: int(''.join(filter(str.isdigit, str(k)))))
        except Exception:
            # 回退到插入顺序（py3.7+ dict 保持插入顺序）
            keys_sorted = keys

        features_list = [features[k] for k in keys_sorted]

        # --- 2) 准备 GT boxes 列表供 pooler 使用 ---
        # pooler 接受 list[Boxes] 或 list[Tensor Nx4]
        gt_boxes = [inst.gt_boxes for inst in proposals]

        # 可能需要处理某些图片没有 gt 的情况（使 pooler 接受空 list）
        # pooler 一般能处理空 Boxes，但如果不行，需要跳过这些图片或构造空 Boxes。

        # ROIAlign -> pooled: (total_boxes, C, H, W)
        pooled = self.pooler(features_list, gt_boxes)  # (N_total_boxes, C, 1, 1) 常见
        pooled = torch.flatten(pooled, start_dim=1)  # (N_total_boxes, C)

        if pooled.numel() == 0:
            # 没有 gt box，直接返回
            return

        # --- 3) 相似度匹配（归一化） ---
        pooled_dropped = F.dropout(pooled, p=0.5, training=self.training)
        pooled_norm = F.normalize(pooled_dropped, dim=1)  # (N, C)
        centroids_norm = F.normalize(self.centroids, dim=1)  # (K, C)

        simm = pooled_norm.matmul(centroids_norm.T)  # (N, K)
        # 得到每个 pooled 对应的最佳质心索引
        topk_idx = simm.argmax(dim=1, keepdim=True)  # (N, 1), dtype=torch.long

        mask = torch.zeros_like(simm, device=simm.device).scatter(1, topk_idx, 1.0)  # (N, K) float

        # --- 4) 累积 sum 和 count（本卡） ---
        sum_x = mask.T.matmul(pooled)  # (K, C)  每类的向量和（本卡）
        count = mask.sum(dim=0).unsqueeze(1)  # (K, 1)  每类计数（本卡）

        # --- 5) 跨卡聚合（如果分布式） ---
        world_size = comm.get_world_size()
        if world_size > 1:
            # 准备接收张量
            sum_x_gt = [torch.empty_like(sum_x) for _ in range(world_size)]
            count_gt = [torch.empty_like(count) for _ in range(world_size)]
            dist.all_gather(sum_x_gt, sum_x)
            dist.all_gather(count_gt, count)
            sum_x_gt = torch.stack(sum_x_gt, dim=0).sum(dim=0)  # (K, C)
            count_gt = torch.stack(count_gt, dim=0).sum(dim=0)  # (K, 1)
        else:
            sum_x_gt = sum_x
            count_gt = count

        # --- 6) 计算新的类中心（使用跨卡 count_gt） ---
        denom = count_gt.clamp_min(1.0)  # 防止除0
        new_centroids = sum_x_gt / denom  # (K, C)

        # --- 7) 只更新那些全局 count>0 的类，按 momentum 混合更新 ---
        alpha = (count_gt > 0).float() * self.momentum  # (K,1)
        # 变换形状便于广播 (K,1) -> (K,C)
        alpha = alpha.to(new_centroids.device)
        alpha = alpha.expand_as(new_centroids)

        # 原地更新 centroids（使用 copy_ 确保引用不变）
        updated = (1.0 - alpha) * self.centroids + alpha * new_centroids
        self.centroids.copy_(updated)

    # def update_centroids(self, features: dict, proposals):
    #     """
    #     features: 原始 FPN 特征图 dict
    #     proposals: list[Instances]，含 gt_boxes
    #     """
    #     if self.momentum == 0 or self.iterations < self.warmup_iters:
    #         return
    #
    #     # 取每层特征图
    #     features_list = [features[lvl] for lvl in sorted(features.keys())]
    #     pooled = self.pooler(features_list, [x.gt_boxes for x in proposals])
    #     pooled = torch.flatten(pooled, start_dim=1)  # (N, C)
    #
    #     # 归一化匹配
    #     simm = F.normalize(F.dropout(pooled, p=0.5), dim=1).matmul(
    #         F.normalize(self.centroids, dim=1).T
    #     )
    #     mask = torch.zeros_like(simm).scatter(1, simm.argmax(dim=1, keepdim=True), 1.0)
    #
    #     # 累积更新
    #     sum_x = mask.T.matmul(pooled)
    #     count = mask.sum(dim=0).unsqueeze(1)
    #
    #     world_size = comm.get_world_size()
    #     if world_size > 1:
    #         sum_x_gt = [torch.empty_like(sum_x) for _ in range(world_size)]
    #         count_gt = [torch.empty_like(count) for _ in range(world_size)]
    #         dist.all_gather(sum_x_gt, sum_x)
    #         dist.all_gather(count_gt, count)
    #         sum_x_gt = torch.stack(sum_x_gt, dim=0).sum(dim=0)
    #         count_gt = torch.stack(count_gt, dim=0).sum(dim=0)
    #     else:
    #         sum_x_gt = sum_x
    #         count_gt = count
    #
    #     centroids = sum_x_gt / count.clamp_min(1)
    #     alpha = (count_gt > 0).float() * self.momentum
    #     self.centroids.set_((1 - alpha) * self.centroids + alpha * centroids)

    def forward(self, features: dict):
        """对每层特征单独校准，返回一个新的 dict"""
        outputs = {}
        for lvl, x in features.items():
            if self.training and self.iterations < self.warmup_iters:
                outputs[lvl] = F.relu(self.fcs[lvl](x))
                continue

            sim = torch.einsum(
                "bchw,nc->bnhw",
                F.normalize(x, dim=1),
                F.normalize(self.centroids, dim=1),
            )
            mask = torch.zeros_like(sim).scatter(1, sim.argmax(dim=1, keepdim=True), 1.0)

            sum_x = torch.einsum("bnhw,bchw->bnc", mask, x)
            count = reduce(mask, "b n h w -> b n ()", "sum")
            centroids = sum_x / count.clamp_min(1)
            delta = torch.einsum("bnhw,bnc->bchw", mask, centroids) - x
            alpha = torch.exp(-delta.square().mean(dim=1, keepdim=True))
            outputs[lvl] = F.relu(self.fcs[lvl](x + alpha * delta))

        return outputs

