"""
图像分类数据集
支持文件夹分类格式的数据加载
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from app.utils.augmentation.pipeline import AugmentationPipeline


class ClassificationDataset(Dataset):
    """
    图像分类数据集

    支持的数据集格式：
    - 文件夹分类格式：root_dir/class1/*.jpg, root_dir/class2/*.jpg, ...
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        image_size: int = 224,
        augmentation: Optional[Dict] = None,
        mode: str = "train",
        train_val_split: float = 0.8
    ):
        """
        初始化分类数据集

        Args:
            root_dir: 数据集根目录（类别文件夹）
            image_size: 目标图像尺寸
            augmentation: 数据增强配置
            mode: 'train' 或 'val'
            train_val_split: 训练集划分比例（用于从完整数据集中划分）
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.mode = mode
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        # 扫描类别文件夹，收集样本
        self._scan_directories()

        # 如果需要划分训练/验证集
        if train_val_split < 1.0:
            self._split_data(train_val_split)

        # 设置数据增强管道
        self.transform = self._build_transform(augmentation, mode)

    def _scan_directories(self):
        """扫描类别目录，收集图像路径"""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {self.root_dir}")

        # 查找所有类别文件夹
        class_dirs = [d for d in self.root_dir.iterdir() if d.is_dir()]
        class_dirs.sort()

        if not class_dirs:
            # 可能是平铺结构，直接收集所有图像
            self._scan_flat_directory()
            return

        # 建立类别映射
        self.classes = [d.name for d in class_dirs]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 收集每个类别的图像
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

        for class_dir in class_dirs:
            class_name = class_dir.name
            class_idx = self.class_to_idx[class_name]

            # 收集图像文件
            images = []
            for ext in supported_extensions:
                images.extend(class_dir.glob(f"*{ext}"))
                images.extend(class_dir.glob(f"*{ext.upper()}"))

            for img_path in images:
                self.samples.append({
                    'path': str(img_path),
                    'label': class_idx,
                    'class': class_name
                })

        if len(self.samples) == 0:
            raise ValueError(f"未找到任何图像文件: {self.root_dir}")

    def _scan_flat_directory(self):
        """扫描平铺目录结构（所有图像在一个文件夹）"""
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = []

        for ext in supported_extensions:
            images.extend(self.root_dir.glob(f"*{ext}"))
            images.extend(self.root_dir.glob(f"*{ext.upper()}"))

        # 假设只有一个类别
        self.classes = ['default']
        self.class_to_idx = {'default': 0}

        for img_path in images:
            self.samples.append({
                'path': str(img_path),
                'label': 0,
                'class': 'default'
            })

    def _split_data(self, train_val_split: float):
        """划分训练集和验证集"""
        if self.mode == "train":
            # 使用固定的随机种子
            indices = list(range(len(self.samples)))
            random.seed(42)
            random.shuffle(indices)

            split_idx = int(len(indices) * train_val_split)
            train_indices = set(indices[:split_idx])

            self.samples = [s for i, s in enumerate(self.samples) if i in train_indices]
        else:
            # 验证集使用剩余部分
            indices = list(range(len(self.samples)))
            random.seed(42)
            random.shuffle(indices)

            split_idx = int(len(indices) * train_val_split)
            val_indices = set(indices[split_idx:])

            self.samples = [s for i, s in enumerate(self.samples) if i in val_indices]

    def _build_transform(self, augmentation: Optional[Dict], mode: str):
        """构建数据变换管道"""
        return AugmentationPipeline(
            image_size=self.image_size,
            augmentation=augmentation if mode == "train" else None,
            mode=mode,
            task_type="classification"
        )

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            包含以下字段的字典:
                - image: 图像张量 [C, H, W]
                - label: 类别标签 (标量)
                - image_id: 图像标识符
        """
        sample = self.samples[idx]

        # 加载图像
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            # 如果加载失败，返回一个黑色图像
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'image_id': Path(sample['path']).stem
        }

    @classmethod
    def from_paths(
        cls,
        image_paths: List[str],
        labels: Optional[List[int]] = None,
        image_size: int = 224,
        augmentation: Optional[Dict] = None,
        mode: str = "train"
    ) -> 'ClassificationDataset':
        """
        从图像路径列表创建数据集

        Args:
            image_paths: 图像路径列表
            labels: 对应的标签列表（如果为None，假设为单类别）
            image_size: 目标图像尺寸
            augmentation: 数据增强配置
            mode: 'train' 或 'val'

        Returns:
            ClassificationDataset实例
        """
        dataset = cls.__new__(cls)
        dataset.root_dir = Path(image_paths[0]).parent
        dataset.image_size = image_size
        dataset.mode = mode
        dataset.classes = ['default'] if labels is None else [f"class_{i}" for i in set(labels)]
        dataset.class_to_idx = {c: i for i, c in enumerate(dataset.classes)}

        # 创建样本列表
        if labels is None:
            labels = [0] * len(image_paths)

        dataset.samples = [
            {
                'path': path,
                'label': label,
                'class': dataset.classes[label]
            }
            for path, label in zip(image_paths, labels)
        ]

        # 构建变换
        dataset.transform = dataset._build_transform(augmentation, mode)

        return dataset


def get_class_weights(dataset: ClassificationDataset) -> torch.Tensor:
    """
    计算类别权重（用于处理类别不平衡）

    Args:
        dataset: 分类数据集

    Returns:
        每个类别的权重张量
    """
    from collections import Counter

    labels = [s['label'] for s in dataset.samples]
    label_counts = Counter(labels)

    num_classes = len(dataset.classes)
    total_samples = len(labels)

    weights = torch.zeros(num_classes)
    for i in range(num_classes):
        count = label_counts.get(i, 0)
        if count > 0:
            weights[i] = total_samples / (num_classes * count)
        else:
            weights[i] = 1.0

    return weights
