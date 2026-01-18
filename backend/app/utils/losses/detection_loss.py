"""
检测损失函数
YOLO风格的检测损失实现
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    检测损失函数（YOLO风格）

    包含三部分损失：
    1. 分类损失（BCEWithLogitsLoss）
    2. 目标性损失（BCEWithLogitsLoss）
    3. 边界框损失（CIoU Loss）
    """

    def __init__(
        self,
        num_classes: int = 80,
        anchors: Optional[torch.Tensor] = None,
        iou_threshold: float = 0.5,
        cls_gain: float = 0.5,
        obj_gain: float = 1.0,
        bbox_gain: float = 0.05
    ):
        """
        初始化检测损失

        Args:
            num_classes: 类别数
            anchors: 锚点尺寸 [num_anchors, 2]
            iou_threshold: IoU阈值用于正样本匹配
            cls_gain: 分类损失权重
            obj_gain: 目标性损失权重
            bbox_gain: 边界框损失权重
        """
        super().__init__()

        self.num_classes = num_classes
        self.iou_threshold = iou_threshold

        self.cls_gain = cls_gain
        self.obj_gain = obj_gain
        self.bbox_gain = bbox_gain

        # 损失函数
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')

        # 默认锚点（YOLOv5的3个尺度，每个尺度3个锚点）
        if anchors is None:
            self.register_buffer(
                'anchors',
                torch.tensor([
                    [[10, 13], [16, 30], [33, 23]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[116, 90], [156, 198], [373, 326]]
                ], dtype=torch.float32)
            )
        else:
            self.register_buffer('anchors', anchors)

    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        计算检测损失

        Args:
            predictions: 预测输出列表 [P3_out, P4_out, P5_out]
                每个元素形状: [batch, 3, grid_h, grid_w, 5+num_classes]
            targets: 目标标注列表，每个元素包含:
                - 'boxes': [batch, max_objs, 4] (x1, y1, x2, y2) 归一化
                - 'labels': [batch, max_objs]
                - 'image_ids': 图像ID列表

        Returns:
            总损失值
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]

        # 初始化损失
        cls_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        bbox_loss = torch.tensor(0.0, device=device)

        # 处理每个预测层
        for i, pred in enumerate(predictions):
            batch_idx, target_cls, target_box, grid_mask = self._build_targets(
                pred, targets, i, device
            )

            if len(batch_idx) == 0:
                continue

            # 获取正样本对应的预测
            pred_pos = pred[batch_idx, :, grid_mask[0], grid_mask[1]]

            # 分解预测输出
            pred_box = pred_pos[:, :4]
            pred_obj = pred_pos[:, 4]
            pred_cls = pred_pos[:, 5:]

            # 目标性损失
            obj_targets = torch.zeros_like(pred_obj)
            obj_targets[batch_idx.unique()] = 1.0

            # 所有网格的目标性损失
            all_obj = pred[:, :, :, :, 4]
            obj_loss += self.bce_obj(all_obj, obj_targets).mean()

            # 只对正样本计算分类和边界框损失
            num_pos = len(batch_idx.unique())
            if num_pos > 0:
                # 分类损失
                cls_targets = F.one_hot(target_cls.long(), self.num_classes).float()
                cls_loss += self.bce_cls(pred_cls, cls_targets).mean()

                # 边界框损失（CIoU）
                bbox_loss += self._ciou_loss(pred_box, target_box).mean()

        # 加权求和
        total_loss = (
            self.cls_gain * cls_loss +
            self.obj_gain * obj_loss +
            self.bbox_gain * bbox_loss
        )

        return total_loss

    def _build_targets(
        self,
        prediction: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        layer_idx: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建训练目标（匹配锚点）

        Args:
            prediction: 当前层的预测输出 [batch, 3, grid_h, grid_w, 5+num_classes]
            targets: 目标标注
            layer_idx: 当前层索引（0, 1, 2）
            device: 设备

        Returns:
            (batch_idx, target_cls, target_box, grid_mask)
        """
        batch_size, num_anchors, grid_h, grid_w, _ = prediction.shape

        # 收集正样本
        batch_indices = []
        target_classes = []
        target_boxes = []
        grid_indices = []

        # 获取当前层的锚点
        layer_anchors = self.anchors[layer_idx]  # [3, 2]

        # 遍历批次中的每个样本
        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            boxes = target.get('boxes', torch.zeros((0, 4), device=device))
            labels = target.get('labels', torch.zeros((0,), dtype=torch.long, device=device))

            if len(boxes) == 0:
                continue

            # 转换边界框格式为 (center_x, center_y, width, height)
            boxes_cxcywh = self._xyxy_to_cxcywh(boxes)

            # 归一化到网格坐标
            boxes_cxcywh[:, [0, 2]] *= grid_w
            boxes_cxcywh[:, [1, 3]] *= grid_h

            # 为每个目标匹配最佳锚点
            for box, label in zip(boxes_cxcywh, labels):
                # 计算与锚点的IoU
                best_iou, best_anchor = self._match_anchor(box, layer_anchors)

                if best_iou > self.iou_threshold:
                    # 计算网格位置
                    gx = box[0].long().clamp(0, grid_w - 1)
                    gy = box[1].long().clamp(0, grid_h - 1)

                    batch_indices.append(batch_idx)
                    target_classes.append(label)
                    target_boxes.append(box[:4])
                    grid_indices.append((gy, gx))

        if len(batch_indices) == 0:
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0, 4), device=device),
                torch.zeros((2, 0), dtype=torch.long, device=device)
            )

        batch_idx = torch.tensor(batch_indices, dtype=torch.long, device=device)
        target_cls = torch.stack(target_classes)
        target_box = torch.stack(target_boxes)
        grid_y = torch.tensor([g[0] for g in grid_indices], dtype=torch.long, device=device)
        grid_x = torch.tensor([g[1] for g in grid_indices], dtype=torch.long, device=device)
        grid_mask = torch.stack([grid_y, grid_x])

        return batch_idx, target_cls, target_box, grid_mask

    def _match_anchor(
        self,
        box: torch.Tensor,
        anchors: torch.Tensor
    ) -> Tuple[float, int]:
        """
        将边界框匹配到最佳锚点

        Args:
            box: 边界框 (center_x, center_y, width, height)
            anchors: 锚点 [num_anchors, 2]

        Returns:
            (最佳IoU, 最佳锚点索引)
        """
        # 计算框与每个锚点的IoU
        ious = []
        for anchor in anchors:
            iou = self._compute_box_iou_1d(box[2:4], anchor)
            ious.append(iou)

        ious = torch.tensor(ious)
        best_iou, best_anchor = ious.max(0)

        return best_iou.item(), best_anchor.item()

    @staticmethod
    def _compute_box_iou_1d(box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        计算两个边界框的IoU（简化版，只考虑宽高）

        Args:
            box1: [width, height]
            box2: [width, height]

        Returns:
            IoU值
        """
        w1, h1 = box1
        w2, h2 = box2

        # 交集
        inter_w = min(w1, w2)
        inter_h = min(h1, h2)
        inter = inter_w * inter_h

        # 并集
        union = w1 * h1 + w2 * h2 - inter

        return inter / (union + 1e-6)

    @staticmethod
    def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
        """
        转换边界框格式: (x1, y1, x2, y2) -> (center_x, center_y, width, height)

        Args:
            boxes: 输入边界框 [N, 4]

        Returns:
            转换后的边界框 [N, 4]
        """
        x1, y1, x2, y2 = boxes.unbind(-1)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        return torch.stack([cx, cy, w, h], dim=-1)

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """
        转换边界框格式: (center_x, center_y, width, height) -> (x1, y1, x2, y2)

        Args:
            boxes: 输入边界框 [N, 4]

        Returns:
            转换后的边界框 [N, 4]
        """
        cx, cy, w, h = boxes.unbind(-1)

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _ciou_loss(
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        计算CIoU损失

        Args:
            pred_boxes: 预测边界框 (center_x, center_y, width, height) [N, 4]
            target_boxes: 目标边界框 (center_x, center_y, width, height) [N, 4]

        Returns:
            CIoU损失 [N]
        """
        # 转换为(x1, y1, x2, y2)格式
        pred_xyxy = DetectionLoss._cxcywh_to_xyxy(pred_boxes)
        target_xyxy = DetectionLoss._cxcywh_to_xyxy(target_boxes)

        # 计算IoU
        iou = DetectionLoss._compute_iou(pred_xyxy, target_xyxy)

        # 计算中心点距离
        pred_cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
        pred_cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
        target_cx = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
        target_cy = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2

        center_distance = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2

        # 计算最小外接矩形对角线
        x1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
        y1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
        x2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
        y2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])

        diagonal_distance = (x2 - x1) ** 2 + (y2 - y1) ** 2

        # CIoU = IoU - (center_distance / diagonal_distance)
        ciou = iou - (center_distance / (diagonal_distance + 1e-7))

        return 1 - ciou

    @staticmethod
    def _compute_iou(
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算两组边界框的IoU

        Args:
            boxes1: [N, 4] (x1, y1, x2, y2)
            boxes2: [N, 4] (x1, y1, x2, y2)

        Returns:
            IoU值 [N]
        """
        # 计算交集区域
        x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3], boxes2[:, 3])

        inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # 计算并集区域
        boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / (union_area + 1e-7)


def create_detection_loss(
    num_classes: int = 80,
    anchors: Optional[torch.Tensor] = None,
    **kwargs
) -> DetectionLoss:
    """
    创建检测损失函数

    Args:
        num_classes: 类别数
        anchors: 锚点尺寸
        **kwargs: 其他配置参数

    Returns:
        检测损失函数实例
    """
    return DetectionLoss(num_classes=num_classes, anchors=anchors, **kwargs)
