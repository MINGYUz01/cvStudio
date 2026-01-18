"""
检测指标计算
提供mAP、IoU等检测任务指标
"""

from typing import Dict, List, Optional, Tuple
import torch
import numpy as np


class DetectionMetrics:
    """
    检测指标计算器

    支持的指标：
    - mAP: 平均精度均值
    - mAP@50: IoU阈值为0.5的mAP
    - mAP@75: IoU阈值为0.75的mAP
    - Precision: 精确率
    - Recall: 召回率
    - IoU: 交并比
    """

    def __init__(self, num_classes: int = 80, iou_threshold: float = 0.5):
        """
        初始化检测指标计算器

        Args:
            num_classes: 类别数
            iou_threshold: IoU阈值
        """
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        """重置累积统计"""
        self.all_detections = []
        self.all_targets = []

    def update(
        self,
        predictions: List[Dict],
        targets: List[Dict]
    ):
        """
        更新累积统计

        Args:
            predictions: 预测结果列表，每个元素包含:
                - 'boxes': [N, 4] 边界框 (x1, y1, x2, y2)
                - 'scores': [N] 置信度分数
                - 'labels': [N] 类别ID
            targets: 真实标注列表，每个元素包含:
                - 'boxes': [M, 4] 边界框
                - 'labels': [M] 类别ID
        """
        self.all_detections.extend(predictions)
        self.all_targets.extend(targets)

    def compute(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        计算检测指标

        Args:
            predictions: 预测结果列表
            targets: 真实标注列表
            iou_threshold: IoU阈值

        Returns:
            指标字典
        """
        # 计算mAP
        mAP = self.compute_mAP(predictions, targets, iou_threshold)

        # 计算精确率和召回率
        precision, recall = self.compute_precision_recall(
            predictions, targets, iou_threshold
        )

        return {
            'mAP': mAP,
            'precision': precision,
            'recall': recall
        }

    def compute_mAP(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        iou_threshold: float = 0.5
    ) -> float:
        """
        计算mAP（平均精度均值）

        Args:
            predictions: 预测结果列表
            targets: 真实标注列表
            iou_threshold: IoU阈值

        Returns:
            mAP值
        """
        # 为每个类别计算AP
        aps = []

        for class_id in range(self.num_classes):
            # 收集该类别的预测和真实标注
            class_preds = []
            class_targets = []

            for pred, target in zip(predictions, targets):
                # 筛选该类别的预测
                pred_labels = pred.get('labels', torch.zeros((0,), dtype=torch.long))
                if len(pred_labels) > 0:
                    mask = pred_labels == class_id
                    if mask.any():
                        class_preds.append({
                            'boxes': pred['boxes'][mask].cpu(),
                            'scores': pred['scores'][mask].cpu()
                        })

                # 筛选该类别的真实标注
                target_labels = target.get('labels', torch.zeros((0,), dtype=torch.long))
                if len(target_labels) > 0:
                    mask = target_labels == class_id
                    if mask.any():
                        class_targets.append({
                            'boxes': target['boxes'][mask].cpu()
                        })

            if len(class_preds) == 0 or len(class_targets) == 0:
                continue

            # 计算该类别的AP
            ap = self._compute_ap_per_class(class_preds, class_targets, iou_threshold)
            aps.append(ap)

        # mAP = 所有类别AP的平均值
        return float(np.mean(aps)) if aps else 0.0

    def _compute_ap_per_class(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        iou_threshold: float
    ) -> float:
        """
        计算单个类别的AP

        Args:
            predictions: 该类别的预测列表
            targets: 该类别的真实标注列表
            iou_threshold: IoU阈值

        Returns:
            AP值
        """
        # 收集所有预测
        all_preds = []
        for i, pred in enumerate(predictions):
            boxes = pred['boxes'].numpy()
            scores = pred['scores'].numpy()
            for box, score in zip(boxes, scores):
                all_preds.append({
                    'image_id': i,
                    'box': box,
                    'score': score
                })

        # 按分数排序
        all_preds.sort(key=lambda x: x['score'], reverse=True)

        # 收集所有真实标注
        all_targets = []
        num_gts = 0
        for i, target in enumerate(targets):
            boxes = target['boxes'].numpy()
            for box in boxes:
                all_targets.append({
                    'image_id': i,
                    'box': box,
                    'matched': False
                })
                num_gts += 1

        if num_gts == 0:
            return 0.0

        # 计算TP和FP
        tp = np.zeros(len(all_preds))
        fp = np.zeros(len(all_preds))

        for i, pred in enumerate(all_preds):
            # 找到对应图像的真实标注
            image_targets = [t for t in all_targets if t['image_id'] == pred['image_id']]

            if not image_targets:
                fp[i] = 1
                continue

            # 计算与所有真实标注的IoU
            ious = []
            for target in image_targets:
                iou = self._compute_iou(pred['box'], target['box'])
                ious.append(iou)

            # 找到最大IoU
            max_iou = max(ious) if ious else 0
            max_idx = ious.index(max_iou) if ious else -1

            if max_iou >= iou_threshold:
                if not image_targets[max_idx]['matched']:
                    tp[i] = 1
                    image_targets[max_idx]['matched'] = True
                else:
                    fp[i] = 1  # 重复检测
            else:
                fp[i] = 1

        # 计算累积精确率和召回率
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / num_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

        # 计算AP（使用11点插值）
        ap = self._compute_ap_11_point(precisions, recalls)

        return ap

    def _compute_ap_11_point(
        self,
        precisions: np.ndarray,
        recalls: np.ndarray
    ) -> float:
        """
        使用11点插值计算AP

        Args:
            precisions: 精确率数组
            recalls: 召回率数组

        Returns:
            AP值
        """
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if mask.any():
                p = precisions[mask].max()
            else:
                p = 0.0
            ap += p / 11

        return ap

    def compute_precision_recall(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        iou_threshold: float = 0.5
    ) -> Tuple[float, float]:
        """
        计算精确率和召回率

        Args:
            predictions: 预测结果列表
            targets: 真实标注列表
            iou_threshold: IoU阈值

        Returns:
            (精确率, 召回率)
        """
        total_tp = 0
        total_fp = 0
        total_targets = 0

        for pred, target in zip(predictions, targets):
            pred_boxes = pred.get('boxes', torch.zeros((0, 4)))
            pred_scores = pred.get('scores', torch.zeros((0,)))
            pred_labels = pred.get('labels', torch.zeros((0,), dtype=torch.long))

            target_boxes = target.get('boxes', torch.zeros((0, 4)))
            target_labels = target.get('labels', torch.zeros((0,), dtype=torch.long))

            total_targets += len(target_boxes)

            if len(pred_boxes) == 0:
                continue

            # 对每个类别进行匹配
            for class_id in range(self.num_classes):
                # 获取该类别的预测和真实标注
                pred_mask = pred_labels == class_id
                target_mask = target_labels == class_id

                class_pred_boxes = pred_boxes[pred_mask]
                class_target_boxes = target_boxes[target_mask]

                if len(class_pred_boxes) == 0 or len(class_target_boxes) == 0:
                    if len(class_pred_boxes) > 0:
                        total_fp += len(class_pred_boxes)
                    continue

                # 计算IoU矩阵
                iou_matrix = self._compute_iou_matrix(
                    class_pred_boxes.numpy(),
                    class_target_boxes.numpy()
                )

                # 匹配（贪心算法）
                matched_pred = set()
                matched_target = set()

                for pred_idx in range(len(class_pred_boxes)):
                    for target_idx in range(len(class_target_boxes)):
                        if pred_idx in matched_pred or target_idx in matched_target:
                            continue
                        if iou_matrix[pred_idx, target_idx] >= iou_threshold:
                            total_tp += 1
                            matched_pred.add(pred_idx)
                            matched_target.add(target_idx)
                            break

                # 未匹配的预测为FP
                total_fp += len(class_pred_boxes) - len(matched_pred)

        # 计算精确率和召回率
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / total_targets if total_targets > 0 else 0.0

        return float(precision), float(recall)

    def _compute_iou_matrix(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """
        计算两组边界框的IoU矩阵

        Args:
            boxes1: [N, 4] 边界框数组
            boxes2: [M, 4] 边界框数组

        Returns:
            IoU矩阵 [N, M]
        """
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))

        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self._compute_iou(box1, box2)

        return iou_matrix

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """
        计算两个边界框的IoU

        Args:
            box1: [4] 边界框 (x1, y1, x2, y2)
            box2: [4] 边界框 (x1, y1, x2, y2)

        Returns:
            IoU值
        """
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        # 计算并集区域
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-7)

    def get_metrics(self) -> Dict[str, float]:
        """
        获取累积的指标

        Returns:
            指标字典
        """
        if not self.all_detections or not self.all_targets:
            return {
                'mAP': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }

        return self.compute(self.all_detections, self.all_targets, self.iou_threshold)


def compute_iou(
    box1: torch.Tensor,
    box2: torch.Tensor
) -> torch.Tensor:
    """
    计算两组边界框的IoU

    Args:
        box1: [N, 4] 边界框张量 (x1, y1, x2, y2)
        box2: [M, 4] 边界框张量 (x1, y1, x2, y2)

    Returns:
        IoU矩阵 [N, M]
    """
    # 扩展维度以便广播
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M]

    # 计算交集
    x1 = torch.max(box1[:, 0:1], box2[:, 0:1].t())  # [N, M]
    y1 = torch.max(box1[:, 1:2], box2[:, 1:2].t())  # [N, M]
    x2 = torch.min(box1[:, 2:3], box2[:, 2:3].t())  # [N, M]
    y2 = torch.min(box1[:, 3:4], box2[:, 3:4].t())  # [N, M]

    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)  # [N, M]

    # 计算IoU
    iou = inter / (area1[:, None] + area2[None, :] - inter + 1e-7)

    return iou
