"""
模型构建模块
支持图像分类和目标检测模型的创建
"""

from app.utils.models.factory import ModelFactory
from app.utils.models.classification import ClassificationModel
from app.utils.models.detection import DetectionModel

__all__ = [
    "ModelFactory",
    "ClassificationModel",
    "DetectionModel",
]
