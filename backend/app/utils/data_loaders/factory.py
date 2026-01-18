"""
数据加载器工厂
根据任务类型和数据格式创建对应的DataLoader
"""

import random
from pathlib import Path
from typing import Dict, Tuple, Optional
from torch.utils.data import DataLoader, Subset
import torch

from app.utils.data_loaders.classification_dataset import ClassificationDataset
from app.utils.data_loaders.detection_dataset import DetectionDataset


class DataLoaderFactory:
    """
    数据加载器工厂类
    负责根据配置创建训练和验证数据加载器
    """

    # 支持的数据格式
    SUPPORTED_FORMATS = {
        "classification": ["classification"],
        "detection": ["yolo", "coco", "voc"]
    }

    @staticmethod
    def create(config: Dict) -> Tuple[DataLoader, DataLoader]:
        """
        创建训练和验证数据加载器

        Args:
            config: 配置字典，包含以下字段:
                - task_type: 任务类型 ("classification" | "detection")
                - dataset_format: 数据集格式 ("classification" | "yolo" | "coco" | "voc")
                - dataset_path: 数据集路径
                - batch_size: 批次大小
                - image_size: 图像尺寸
                - num_classes: 类别数量
                - augmentation: 数据增强配置
                - train_val_split: 训练验证集划分比例 (默认0.8)
                - num_workers: 数据加载工作进程数
                - pin_memory: 是否使用pin_memory

        Returns:
            (train_loader, val_loader) 训练和验证数据加载器

        Raises:
            ValueError: 当任务类型或数据格式不支持时
        """
        task_type = config.get("task_type", "classification")
        dataset_format = config.get("dataset_format", "classification")
        dataset_path = config.get("dataset_path")
        batch_size = config.get("batch_size", 32)
        image_size = config.get("image_size", 224)
        num_classes = config.get("num_classes", 10)
        augmentation = config.get("augmentation", {})
        train_val_split = config.get("train_val_split", 0.8)
        num_workers = config.get("num_workers", 4)
        pin_memory = config.get("pin_memory", True)

        if not dataset_path:
            raise ValueError("dataset_path 不能为空")

        # 验证任务类型和格式的兼容性
        if task_type not in DataLoaderFactory.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的任务类型: {task_type}")
        if dataset_format not in DataLoaderFactory.SUPPORTED_FORMATS[task_type]:
            raise ValueError(
                f"任务类型 {task_type} 不支持格式 {dataset_format}。"
                f"支持的格式: {DataLoaderFactory.SUPPORTED_FORMATS[task_type]}"
            )

        # 设置随机种子以保证可重复性
        random.seed(42)
        torch.manual_seed(42)

        # 创建完整数据集
        if task_type == "classification":
            # 创建分类数据集
            full_dataset = ClassificationDataset(
                root_dir=dataset_path,
                image_size=image_size,
                augmentation=augmentation
            )
        else:  # detection
            # 创建检测数据集
            full_dataset = DetectionDataset(
                root_dir=dataset_path,
                format=dataset_format,
                image_size=image_size,
                num_classes=num_classes,
                augmentation=augmentation
            )

        # 划分训练集和验证集
        total_size = len(full_dataset)
        train_size = int(total_size * train_val_split)
        val_size = total_size - train_size

        # 使用固定的随机种子划分数据集
        indices = list(range(total_size))
        random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)

        # 获取DataLoader配置
        loader_config = DataLoaderFactory._get_loader_config(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            device=config.get("device", "cpu")
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_config
        )

        val_loader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_config
        )

        return train_loader, val_loader

    @staticmethod
    def _get_loader_config(
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        device: str
    ) -> Dict:
        """
        获取DataLoader配置

        Args:
            batch_size: 批次大小
            num_workers: 工作进程数
            pin_memory: 是否使用pin_memory
            device: 设备类型

        Returns:
            DataLoader配置字典
        """
        # Windows环境下num_workers设为0避免多进程问题
        import platform
        if platform.system() == "Windows":
            num_workers = 0

        # 只在CUDA设备上使用pin_memory
        use_pin_memory = pin_memory and device.startswith("cuda")

        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": use_pin_memory,
            "drop_last": True,
        }

    @staticmethod
    def create_from_paths(
        task_type: str,
        train_paths: list,
        val_paths: Optional[list] = None,
        labels: Optional[list] = None,
        batch_size: int = 32,
        image_size: int = 224,
        num_workers: int = 4,
        **kwargs
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        从图像路径列表创建数据加载器

        Args:
            task_type: 任务类型
            train_paths: 训练图像路径列表
            val_paths: 验证图像路径列表（可选）
            labels: 对应的标签列表（可选）
            batch_size: 批次大小
            image_size: 图像尺寸
            num_workers: 工作进程数
            **kwargs: 其他参数

        Returns:
            (train_loader, val_loader) 验证集可能为None
        """
        if task_type == "classification":
            train_dataset = ClassificationDataset.from_paths(
                image_paths=train_paths,
                labels=labels,
                image_size=image_size,
                **kwargs
            )

            if val_paths:
                val_dataset = ClassificationDataset.from_paths(
                    image_paths=val_paths,
                    labels=labels,
                    image_size=image_size,
                    **kwargs
                )
            else:
                val_dataset = None
        else:
            raise ValueError("从路径创建检测数据集暂不支持")

        # 获取DataLoader配置
        loader_config = DataLoaderFactory._get_loader_config(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=kwargs.get("pin_memory", True),
            device=kwargs.get("device", "cpu")
        )

        train_loader = DataLoader(train_dataset, shuffle=True, **loader_config)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_config) if val_dataset else None

        return train_loader, val_loader
