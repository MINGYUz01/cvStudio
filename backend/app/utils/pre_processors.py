"""
预处理器系统
支持不同任务类型的预处理逻辑（分类/检测/分割）
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
from PIL import Image
import numpy as np
import torch
import cv2
from loguru import logger


class PreProcessor(ABC):
    """
    预处理器基类

    所有任务类型的预处理器都应继承此类并实现process方法
    """

    @abstractmethod
    def process(
        self,
        image: Image.Image,
        device: str = "cpu"
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Dict[str, Any]]:
        """
        预处理图像

        Args:
            image: PIL图像对象
            device: 目标设备

        Returns:
            (预处理后的图像, 图像信息字典)
        """
        pass

    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """确保图像是RGB格式"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def _get_original_size(self, image: Image.Image) -> Tuple[int, int]:
        """获取原始图像尺寸"""
        return image.size  # (width, height)


class ClassificationPreProcessor(PreProcessor):
    """
    图像分类预处理器

    使用标准的ImageNet预处理：
    - Resize保持宽高比，然后中心裁剪
    - 或者直接Resize到指定尺寸
    - 归一化（ImageNet均值和标准差）
    """

    # ImageNet默认值
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]
    DEFAULT_SIZE = (224, 224)

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化分类预处理器

        Args:
            config: 配置字典，可包含：
                - input_size: 输入尺寸 (height, width)
                - resize_mode: resize模式 ('direct', 'center_crop')
                - mean: 归一化均值
                - std: 归一化标准差
        """
        self.config = config or {}
        self.input_size = tuple(self.config.get('input_size', self.DEFAULT_SIZE))
        self.resize_mode = self.config.get('resize_mode', 'direct')
        self.mean = self.config.get('normalize', {}).get('mean', self.DEFAULT_MEAN)
        self.std = self.config.get('normalize', {}).get('std', self.DEFAULT_STD)

    def process(
        self,
        image: Image.Image,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        预处理图像用于分类

        Args:
            image: PIL图像
            device: 目标设备

        Returns:
            (预处理后的tensor, 图像信息)
        """
        original_size = self._get_original_size(image)
        image = self._ensure_rgb(image)

        # Resize
        if self.resize_mode == 'center_crop':
            # 先resize到指定尺寸的1.14倍，然后中心裁剪
            target_size = tuple(int(s * 256 / 224) for s in self.input_size)
            image = image.resize(target_size, Image.BILINEAR)

            # 中心裁剪
            left = (image.width - self.input_size[1]) // 2
            top = (image.height - self.input_size[0]) // 2
            right = left + self.input_size[1]
            bottom = top + self.input_size[0]
            image = image.crop((left, top, right, bottom))
        else:
            # 直接resize
            image = image.resize(self.input_size, Image.BILINEAR)

        # 转换为tensor
        tensor = torch.from_numpy(np.array(image)).float() / 255.0

        # HWC -> CHW
        tensor = tensor.permute(2, 0, 1)

        # 归一化
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # 添加batch维度
        tensor = tensor.unsqueeze(0)

        # 移动到设备
        if 'cuda' in device or 'mps' in device:
            tensor = tensor.to(device)

        image_info = {
            'original_size': original_size,
            'input_size': self.input_size,
            'resize_mode': self.resize_mode,
            'mean': self.mean,
            'std': self.std,
        }

        logger.bind(component="classification_preprocessor").debug(
            f"分类预处理完成，原始尺寸={original_size}，输入尺寸={self.input_size}"
        )

        return tensor, image_info


class DetectionPreProcessor(PreProcessor):
    """
    目标检测预处理器

    使用Letterbox方式（保持宽高比，填充灰色边框）：
    - 计算缩放比例
    - 保持宽高比resize
    - 添加padding
    """

    DEFAULT_SIZE = (640, 640)
    DEFAULT_COLOR = (114, 114, 114)  # YOLO默认灰色

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化检测预处理器

        Args:
            config: 配置字典，可包含：
                - input_size: 输入尺寸 (height, width)
                - letterbox: 是否使用letterbox (默认True)
                - pad_color: 填充颜色 (R, G, B)
        """
        self.config = config or {}
        self.input_size = tuple(self.config.get('input_size', self.DEFAULT_SIZE))
        self.letterbox = self.config.get('letterbox', True)
        self.pad_color = self.config.get('pad_color', self.DEFAULT_COLOR)

    def process(
        self,
        image: Image.Image,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        预处理图像用于检测

        Args:
            image: PIL图像
            device: 目标设备

        Returns:
            (预处理后的tensor, 图像信息)
        """
        original_size = self._get_original_size(image)
        image = self._ensure_rgb(image)

        if self.letterbox:
            tensor, scale_info = self._letterbox_resize(image)
        else:
            tensor = self._direct_resize(image)
            scale_info = {
                'scale': (
                    self.input_size[1] / original_size[0],
                    self.input_size[0] / original_size[1]
                ),
                'pad': (0, 0)
            }

        # 移动到设备
        if 'cuda' in device or 'mps' in device:
            tensor = tensor.to(device)

        image_info = {
            'original_size': original_size,
            'input_size': self.input_size,
            'scale': scale_info['scale'],
            'pad': scale_info['pad'],
        }

        logger.bind(component="detection_preprocessor").debug(
            f"检测预处理完成，原始尺寸={original_size}，缩放={scale_info['scale']}"
        )

        return tensor, image_info

    def _letterbox_resize(
        self,
        image: Image.Image
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Letterbox方式resize"""
        img_w, img_h = image.size
        target_w, target_h = self.input_size

        # 计算缩放比例
        scale = min(target_w / img_w, target_h / img_h)

        # 计算resize后的尺寸
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize
        resized = image.resize((new_w, new_h), Image.BILINEAR)

        # 创建目标尺寸的画布
        canvas = Image.new('RGB', (target_w, target_h), self.pad_color)

        # 计算padding位置
        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2

        # 粘贴图像
        canvas.paste(resized, (pad_left, pad_top))

        # 转换为tensor
        tensor = torch.from_numpy(np.array(canvas)).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        scale_info = {
            'scale': (scale, scale),
            'pad': (pad_left, pad_top)
        }

        return tensor, scale_info

    def _direct_resize(self, image: Image.Image) -> torch.Tensor:
        """直接resize"""
        resized = image.resize(self.input_size, Image.BILINEAR)

        # 转换为tensor
        tensor = torch.from_numpy(np.array(resized)).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        return tensor


class SegmentationPreProcessor(PreProcessor):
    """
    语义分割预处理器

    与分类类似，但可能需要保持更高的空间分辨率：
    - 保持宽高比resize
    - 或直接resize（常用）
    """

    DEFAULT_SIZE = (512, 512)
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化分割预处理器

        Args:
            config: 配置字典，可包含：
                - input_size: 输入尺寸 (height, width)
                - keep_ratio: 是否保持宽高比 (默认False)
                - mean: 归一化均值
                - std: 归一化标准差
        """
        self.config = config or {}
        self.input_size = tuple(self.config.get('input_size', self.DEFAULT_SIZE))
        self.keep_ratio = self.config.get('keep_ratio', False)
        self.mean = self.config.get('normalize', {}).get('mean', self.DEFAULT_MEAN)
        self.std = self.config.get('normalize', {}).get('std', self.DEFAULT_STD)

    def process(
        self,
        image: Image.Image,
        device: str = "cpu"
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        预处理图像用于分割

        Args:
            image: PIL图像
            device: 目标设备

        Returns:
            (预处理后的tensor, 图像信息)
        """
        original_size = self._get_original_size(image)
        image = self._ensure_rgb(image)

        if self.keep_ratio:
            # 类似letterbox，但返回更多信息
            tensor, scale_info = self._resize_with_ratio(image)
        else:
            # 直接resize
            image = image.resize(self.input_size, Image.BILINEAR)
            scale_info = {
                'scale': (
                    self.input_size[1] / original_size[0],
                    self.input_size[0] / original_size[1]
                ),
                'pad': (0, 0)
            }

            # 转换为tensor
            tensor = torch.from_numpy(np.array(image)).float() / 255.0

        # HWC -> CHW
        tensor = tensor.permute(2, 0, 1)

        # 归一化
        mean = torch.tensor(self.mean).view(3, 1, 1)
        std = torch.tensor(self.std).view(3, 1, 1)
        tensor = (tensor - mean) / std

        # 添加batch维度
        tensor = tensor.unsqueeze(0)

        # 移动到设备
        if 'cuda' in device or 'mps' in device:
            tensor = tensor.to(device)

        image_info = {
            'original_size': original_size,
            'input_size': self.input_size,
            'scale': scale_info['scale'],
            'pad': scale_info['pad'],
        }

        logger.bind(component="segmentation_preprocessor").debug(
            f"分割预处理完成，原始尺寸={original_size}，输入尺寸={self.input_size}"
        )

        return tensor, image_info

    def _resize_with_ratio(self, image: Image.Image) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """保持宽高比resize"""
        img_w, img_h = image.size
        target_w, target_h = self.input_size

        scale = min(target_w / img_w, target_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized = image.resize((new_w, new_h), Image.BILINEAR)

        # 创建画布
        canvas = Image.new('RGB', (target_w, target_h), (0, 0, 0))

        pad_left = (target_w - new_w) // 2
        pad_top = (target_h - new_h) // 2
        canvas.paste(resized, (pad_left, pad_top))

        # 转换为tensor
        tensor = torch.from_numpy(np.array(canvas)).float() / 255.0

        scale_info = {
            'scale': (scale, scale),
            'pad': (pad_left, pad_top)
        }

        return tensor, scale_info


class PreProcessorFactory:
    """
    预处理器工厂

    根据任务类型返回对应的预处理器
    """

    @staticmethod
    def create(task_type: str, config: Optional[Dict[str, Any]] = None) -> PreProcessor:
        """
        创建对应任务类型的预处理器

        Args:
            task_type: 任务类型 (detection/classification/segmentation)
            config: 配置字典

        Returns:
            对应的预处理器实例

        Raises:
            ValueError: 不支持的任务类型
        """
        if task_type == 'detection':
            return DetectionPreProcessor(config)
        elif task_type == 'classification':
            return ClassificationPreProcessor(config)
        elif task_type == 'segmentation':
            return SegmentationPreProcessor(config)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

    @staticmethod
    def get_default_config(task_type: str) -> Dict[str, Any]:
        """
        获取各任务类型的默认配置

        Args:
            task_type: 任务类型

        Returns:
            默认配置字典
        """
        configs = {
            'classification': {
                'input_size': ClassificationPreProcessor.DEFAULT_SIZE,
                'resize_mode': 'direct',
                'normalize': {
                    'mean': ClassificationPreProcessor.DEFAULT_MEAN,
                    'std': ClassificationPreProcessor.DEFAULT_STD,
                }
            },
            'detection': {
                'input_size': DetectionPreProcessor.DEFAULT_SIZE,
                'letterbox': True,
            },
            'segmentation': {
                'input_size': SegmentationPreProcessor.DEFAULT_SIZE,
                'keep_ratio': False,
                'normalize': {
                    'mean': SegmentationPreProcessor.DEFAULT_MEAN,
                    'std': SegmentationPreProcessor.DEFAULT_STD,
                }
            },
        }
        return configs.get(task_type, {})


# 导出
__all__ = [
    'PreProcessor',
    'ClassificationPreProcessor',
    'DetectionPreProcessor',
    'SegmentationPreProcessor',
    'PreProcessorFactory',
]
