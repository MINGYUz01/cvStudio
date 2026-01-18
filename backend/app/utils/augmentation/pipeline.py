"""
数据增强管道
整合torchvision和自定义增强功能
"""

from typing import Dict, Optional, Tuple, Union, Callable
from PIL import Image
import torch
import numpy as np


class AugmentationPipeline:
    """
    数据增强管道

    支持分类和检测任务的数据增强
    检测任务会同步变换边界框标注
    """

    def __init__(
        self,
        image_size: int = 224,
        augmentation: Optional[Dict] = None,
        mode: str = "train",
        task_type: str = "classification"
    ):
        """
        初始化数据增强管道

        Args:
            image_size: 目标图像尺寸
            augmentation: 数据增强配置
            mode: 'train' 或 'val'
            task_type: 'classification' 或 'detection'
        """
        self.image_size = image_size
        self.augmentation = augmentation or {}
        self.mode = mode
        self.task_type = task_type

        # 构建变换管道
        self.transform = self._build_pipeline()

    def _build_pipeline(self) -> Callable:
        """
        构建变换管道

        Returns:
            变换函数
        """
        import torchvision.transforms as T

        transforms_list = []

        if self.mode == "train":
            # 训练模式的增强

            # 1. 随机调整大小和裁剪
            if self.augmentation.get("random_resized_crop", False):
                transforms_list.append(T.RandomResizedCrop(
                    self.image_size,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33)
                ))
            else:
                transforms_list.append(T.Resize((self.image_size, self.image_size)))

            # 2. 随机水平翻转
            if self.augmentation.get("flip_horizontal", False):
                transforms_list.append(T.RandomHorizontalFlip())

            # 3. 随机垂直翻转
            if self.augmentation.get("flip_vertical", False):
                transforms_list.append(T.RandomVerticalFlip())

            # 4. 随机旋转
            rotation_angle = self.augmentation.get("rotation_angle", 0)
            if rotation_angle > 0:
                transforms_list.append(T.RandomRotation(rotation_angle))

            # 5. 颜色抖动
            if self.augmentation.get("color_jitter", False):
                brightness = self.augmentation.get("brightness_factor", 0.2)
                contrast = self.augmentation.get("contrast_factor", 0.2)
                saturation = self.augmentation.get("saturation_factor", 0.2)
                hue = self.augmentation.get("hue_shift", 0.1)

                transforms_list.append(T.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue
                ))

            # 6. 高斯模糊（使用自定义变换）
            if self.augmentation.get("gaussian_blur", 0) > 0:
                transforms_list.append(GaussianBlur(
                    kernel_size=int(self.augmentation["gaussian_blur"] * 3) | 1,
                    sigma=self.augmentation["gaussian_blur"]
                ))

            # 7. 随机擦除
            if self.augmentation.get("random_erase", False):
                transforms_list.append(T.RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value=0
                ))

        else:
            # 验证模式：只调整大小
            transforms_list.append(T.Resize((self.image_size, self.image_size)))

        # 转换为Tensor
        transforms_list.append(T.ToTensor())

        # 标准化
        if self.augmentation.get("normalize", True):
            transforms_list.append(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ))

        return T.Compose(transforms_list)

    def __call__(
        self,
        image: Image.Image,
        boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        应用数据增强

        Args:
            image: PIL图像
            boxes: 边界框 [N, 4] (x1, y1, x2, y2) 归一化坐标（检测任务）
            labels: 标签 [N]（检测任务）

        Returns:
            分类任务: 增强后的图像张量
            检测任务: (图像张量, 边界框, 标签)
        """
        # 分类任务：直接应用变换
        if self.task_type == "classification" or boxes is None:
            return self.transform(image)

        # 检测任务：需要同步变换标注
        return self._apply_detection_augmentation(image, boxes, labels)

    def _apply_detection_augmentation(
        self,
        image: Image.Image,
        boxes: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        应用检测任务的数据增强（同步变换标注）

        Args:
            image: PIL图像
            boxes: 边界框 [N, 4] (x1, y1, x2, y2) 归一化坐标
            labels: 标签 [N]

        Returns:
            (增强后的图像, 变换后的边界框, 标签)
        """
        # 将PIL图像转换为numpy数组
        image_np = np.array(image)
        height, width = image_np.shape[:2]

        # 将归一化的边界框转换为像素坐标
        if len(boxes) > 0:
            boxes_px = boxes.clone()
            boxes_px[:, [0, 2]] *= width
            boxes_px[:, [1, 3]] *= height
        else:
            boxes_px = boxes

        # 应用增强
        aug_config = self.augmentation if self.mode == "train" else {}

        # 1. 水平翻转
        if aug_config.get("flip_horizontal", False):
            if np.random.rand() > 0.5:
                image_np = self._flip_horizontal(image_np)
                if len(boxes_px) > 0:
                    boxes_px = self._flip_boxes_horizontal(boxes_px, width)

        # 2. 垂直翻转
        if aug_config.get("flip_vertical", False):
            if np.random.rand() > 0.5:
                image_np = self._flip_vertical(image_np)
                if len(boxes_px) > 0:
                    boxes_px = self._flip_boxes_vertical(boxes_px, height)

        # 3. 颜色抖动
        if aug_config.get("color_jitter", False):
            image_np = self._apply_color_jitter(image_np, aug_config)

        # 4. 高斯模糊
        if aug_config.get("gaussian_blur", 0) > 0:
            if np.random.rand() > 0.5:
                image_np = self._apply_gaussian_blur(
                    image_np,
                    aug_config["gaussian_blur"]
                )

        # 调整图像大小
        image_resized = Image.fromarray(image_np).resize(
            (self.image_size, self.image_size),
            Image.BILINEAR
        )

        # 转换为Tensor并标准化
        import torchvision.transforms as T
        to_tensor = T.ToTensor()
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        image_tensor = normalize(to_tensor(image_resized))

        # 将边界框转换回归一化坐标
        if len(boxes_px) > 0:
            boxes_px[:, [0, 2]] /= self.image_size
            boxes_px[:, [1, 3]] /= self.image_size
            # 裁剪到[0,1]范围
            boxes_px = torch.clamp(boxes_px, 0, 1)

        return image_tensor, boxes_px, labels

    @staticmethod
    def _flip_horizontal(image: np.ndarray) -> np.ndarray:
        """水平翻转图像"""
        return np.fliplr(image).copy()

    @staticmethod
    def _flip_vertical(image: np.ndarray) -> np.ndarray:
        """垂直翻转图像"""
        return np.flipud(image).copy()

    @staticmethod
    def _flip_boxes_horizontal(boxes: torch.Tensor, width: float) -> torch.Tensor:
        """水平翻转边界框"""
        flipped = boxes.clone()
        flipped[:, [0, 2]] = width - boxes[:, [2, 0]]
        return flipped

    @staticmethod
    def _flip_boxes_vertical(boxes: torch.Tensor, height: float) -> torch.Tensor:
        """垂直翻转边界框"""
        flipped = boxes.clone()
        flipped[:, [1, 3]] = height - boxes[:, [3, 1]]
        return flipped

    @staticmethod
    def _apply_color_jitter(image: np.ndarray, config: Dict) -> np.ndarray:
        """应用颜色抖动"""
        from PIL import Image, ImageEnhance

        pil_image = Image.fromarray(image)

        # 亮度
        brightness_factor = config.get("brightness_factor", 0.2)
        if brightness_factor > 0:
            factor = np.random.uniform(max(0, 1 - brightness_factor), 1 + brightness_factor)
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(factor)

        # 对比度
        contrast_factor = config.get("contrast_factor", 0.2)
        if contrast_factor > 0:
            factor = np.random.uniform(max(0, 1 - contrast_factor), 1 + contrast_factor)
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(factor)

        # 饱和度
        saturation_factor = config.get("saturation_factor", 0.2)
        if saturation_factor > 0:
            factor = np.random.uniform(max(0, 1 - saturation_factor), 1 + saturation_factor)
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(factor)

        return np.array(pil_image)

    @staticmethod
    def _apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
        """应用高斯模糊"""
        import cv2
        kernel_size = int(sigma * 3) | 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


class GaussianBlur:
    """
    高斯模糊变换（torchvision兼容）
    """

    def __init__(self, kernel_size: int = 5, sigma: float = 1.0):
        """
        Args:
            kernel_size: 高斯核大小（奇数）
            sigma: 标准差
        """
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        应用高斯模糊

        Args:
            img: PIL图像

        Returns:
            模糊后的PIL图像
        """
        import cv2

        img_array = np.array(img)

        blurred = cv2.GaussianBlur(
            img_array,
            (self.kernel_size, self.kernel_size),
            self.sigma
        )

        return Image.fromarray(blurred)


class Compose:
    """
    自定义Compose类，支持需要同步变换标注的增强操作
    """

    def __init__(self, transforms: list):
        """
        Args:
            transforms: 变换列表
        """
        self.transforms = transforms

    def __call__(
        self,
        image: Image.Image,
        boxes: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple:
        """
        应用变换序列

        Args:
            image: PIL图像
            boxes: 边界框（可选）
            labels: 标签（可选）

        Returns:
            变换后的图像和标注
        """
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)

        return image, boxes, labels
