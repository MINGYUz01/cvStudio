"""
损失函数模块
提供分类和检测任务的损失函数
"""

from app.utils.losses.factory import LossFactory
from app.utils.losses.classification_loss import ClassificationLoss
from app.utils.losses.detection_loss import DetectionLoss

__all__ = [
    "LossFactory",
    "ClassificationLoss",
    "DetectionLoss",
]
