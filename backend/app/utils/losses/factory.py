"""
损失函数工厂
根据任务类型创建对应的损失函数
"""

from typing import Dict, Optional
import torch
import torch.nn as nn


class LossFactory:
    """
    损失函数工厂类
    负责根据配置创建对应的损失函数
    """

    @staticmethod
    def create(
        task_type: str,
        config: Optional[Dict] = None
    ) -> nn.Module:
        """
        创建损失函数

        Args:
            task_type: 任务类型 ("classification" | "detection")
            config: 配置字典

        Returns:
            损失函数实例

        Raises:
            ValueError: 当任务类型不支持时
        """
        config = config or {}

        if task_type == "classification":
            return LossFactory._create_classification_loss(config)
        elif task_type == "detection":
            return LossFactory._create_detection_loss(config)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

    @staticmethod
    def _create_classification_loss(config: Dict) -> nn.Module:
        """
        创建分类损失函数

        Args:
            config: 配置字典，包含:
                - label_smoothing: 标签平滑系数
                - class_weights: 类别权重

        Returns:
            分类损失函数实例
        """
        from app.utils.losses.classification_loss import ClassificationLoss

        label_smoothing = config.get("label_smoothing", 0.0)
        class_weights = config.get("class_weights", None)

        return ClassificationLoss(
            label_smoothing=label_smoothing,
            class_weights=class_weights
        )

    @staticmethod
    def _create_detection_loss(config: Dict) -> nn.Module:
        """
        创建检测损失函数

        Args:
            config: 配置字典，包含:
                - num_classes: 类别数
                - anchors: 锚点尺寸
                - iou_threshold: IoU阈值
                - cls_gain: 分类损失权重
                - obj_gain: 目标性损失权重
                - bbox_gain: 边界框损失权重

        Returns:
            检测损失函数实例
        """
        from app.utils.losses.detection_loss import DetectionLoss

        num_classes = config.get("num_classes", 80)
        anchors = config.get("anchors", None)
        iou_threshold = config.get("iou_threshold", 0.5)
        cls_gain = config.get("cls_gain", 0.5)
        obj_gain = config.get("obj_gain", 1.0)
        bbox_gain = config.get("bbox_gain", 0.05)

        return DetectionLoss(
            num_classes=num_classes,
            anchors=anchors,
            iou_threshold=iou_threshold,
            cls_gain=cls_gain,
            obj_gain=obj_gain,
            bbox_gain=bbox_gain
        )


def create_loss(
    task_type: str,
    **kwargs
) -> nn.Module:
    """
    便捷函数：创建损失函数

    Args:
        task_type: 任务类型
        **kwargs: 配置参数

    Returns:
        损失函数实例
    """
    return LossFactory.create(task_type, kwargs)
