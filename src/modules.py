# refine
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
        Adapted for single-level backbone feature (e.g. features['res4']).
        Args:
            input_shapes: dict, expected to contain single key 'res4': ShapeSpec
            num_classes: number of classes (int)
        """
        super().__init__()
        self.momentum = momentum
        self.num_classes = num_classes
        self.warmup_iters = warmup_iters
        self.eps = eps

        # 只为 res4 建一个 1x1 conv（如果 input_shapes 里只有一个 level，这里会处理）
        # 用 ModuleDict 保持接口一致
        self.fcs = nn.ModuleDict()
        for lvl, shape in input_shapes.items():
            fc = nn.Conv2d(shape.channels, shape.channels, kernel_size=1, stride=1, padding=0, bias=True)
            # 初始化为 identity-like：当 in==out 时设为单位矩阵每个通道自带通道映射
            if shape.channels == shape.channels:
                # conv weight shape: (out_channels, in_channels, 1, 1)
                eye = torch.eye(shape.channels).reshape(shape.channels, shape.channels, 1, 1)
                with torch.no_grad():
                    fc.weight.copy_(eye)
            else:
                nn.init.kaiming_normal_(fc.weight)
            nn.init.zeros_(fc.bias)
            self.fcs[lvl] = fc

        # ROI Pooler：因为只有一层特征，scales 是单元素列表
        # 注意： scales 的值应为 1.0/stride（ShapeSpec 中的 stride）
        self.pooler = ROIPooler(
            output_size=(1, 1),
            scales=[1.0 / s.stride for s in input_shapes.values()],
            sampling_ratio=0,
            pooler_type="ROIAlignV2",
        )

        # 全局质心：shape (num_classes, dim)
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
        单层版本的多样本质心更新。
        features: dict, e.g. {'res4': Tensor(B, C, H, W)}
        proposals: list[Instances]（每个 Instances 含 gt_boxes）
        """
        if self.momentum == 0 or self.iterations < self.warmup_iters:
            return

        # 只取单层特征（确保顺序与 pooler scales 对应）
        features_list = [features[lvl] for lvl in sorted(features.keys())]

        # gt_boxes 列表（每张图的 gt_boxes）
        gt_boxes = [x.gt_boxes for x in proposals]

        # ROIAlign -> (sum_all_gt, C, 1, 1)
        pooled = self.pooler(features_list, gt_boxes)  # (N, C, 1, 1)
        pooled = pooled.flatten(start_dim=1)  # (N, C)

        if pooled.numel() == 0:
            return  # 没有 GT 时直接返回

        # 归一化后与当前全局质心计算相似度，选择最相似的类作为归属
        sims = F.normalize(F.dropout(pooled, p=0.5, training=self.training), dim=1).matmul(
            F.normalize(self.centroids, dim=1).T
        )  # (N, num_classes)

        # 为每个 pooled 向量分配一个类别：one-hot mask (N, num_classes)
        mask = torch.zeros_like(sims).scatter(1, sims.argmax(dim=1, keepdim=True), 1.0)

        # 累加每个类的特征和计数
        # sum_x: (num_classes, C)
        sum_x = mask.T.matmul(pooled)
        # count: (num_classes, 1)
        count = mask.sum(dim=0).unsqueeze(1)

        # 跨进程聚合（如果使用分布式训练）
        world_size = comm.get_world_size()
        if world_size > 1:
            # 收集并相加所有进程的结果
            sum_x_gather = [torch.empty_like(sum_x) for _ in range(world_size)]
            count_gather = [torch.empty_like(count) for _ in range(world_size)]
            dist.all_gather(sum_x_gather, sum_x)
            dist.all_gather(count_gather, count)
            sum_x_gt = torch.stack(sum_x_gather, dim=0).sum(dim=0)
            count_gt = torch.stack(count_gather, dim=0).sum(dim=0)
        else:
            sum_x_gt = sum_x
            count_gt = count

        # 计算新的类中心（保护除以 0）
        centroids_new = sum_x_gt / count_gt.clamp_min(1)

        # 只有 count>0 的类才参与动量更新
        alpha = (count_gt > 0).float() * self.momentum  # (num_classes, 1)
        # 广播 alpha 以匹配 centroids shape (num_classes, C)
        alpha = alpha.expand_as(centroids_new)

        # 指数移动平均地更新全局质心（in-place）
        updated = (1 - alpha) * self.centroids + alpha * centroids_new
        self.centroids.copy_(updated)

    def forward(self, features: dict):
        """
        对单层特征（res4）做空间分区校准并返回同样的 dict 结构。
        """
        outputs = {}
        for lvl, x in features.items():  # 这里只有一个 lvl: 'res4'
            if self.training and self.iterations < self.warmup_iters:
                outputs[lvl] = F.relu(self.fcs[lvl](x))
                continue

            # sim: (B, num_classes, H, W)
            sim = torch.einsum(
                "bchw,nc->bnhw",
                F.normalize(x, dim=1),
                F.normalize(self.centroids, dim=1),
            )

            # 为每个空间位置分配最相似的全局质心
            mask = torch.zeros_like(sim).scatter(1, sim.argmax(dim=1, keepdim=True), 1.0)  # (B, N, H, W)

            # 聚合分区内的特征总和 -> (B, N, C)
            sum_x = torch.einsum("bnhw,bchw->bnc", mask, x)

            # 统计每个分区的元素个数 -> (B, N, 1)
            count = reduce(mask, "b n h w -> b n ()", "sum")

            # 临时分区质心 (B, N, C)
            centroids_local = sum_x / count.clamp_min(1)

            # 重构分区质心到空间位置并与原始特征比较
            delta = torch.einsum("bnhw,bnc->bchw", mask, centroids_local) - x  # (B, C, H, W)

            # alpha: 相似度自适应系数 (B, 1, H, W)
            alpha = torch.exp(-delta.square().mean(dim=1, keepdim=True))

            # 校准特征并通过 1x1 conv
            outputs[lvl] = F.relu(self.fcs[lvl](x + alpha * delta))

        return outputs
