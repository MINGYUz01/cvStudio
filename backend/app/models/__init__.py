"""
数据模型模块
"""

from .user import User
from .dataset import Dataset
from .model import Model  # 保留旧模型，待废弃
from .model_architecture import ModelArchitecture
from .generated_code import GeneratedCode
from .training import TrainingRun, Checkpoint
from .inference import InferenceJob
from .augmentation import AugmentationStrategy

__all__ = [
    "User",
    "Dataset",
    "Model",  # 待废弃
    "ModelArchitecture",
    "GeneratedCode",
    "TrainingRun",
    "Checkpoint",
    "InferenceJob",
    "AugmentationStrategy"
]