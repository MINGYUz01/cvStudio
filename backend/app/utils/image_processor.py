"""
图像处理工具模块
提供图像加载、处理、增强等通用功能
"""

import os
import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
import cv2
from loguru import logger


class ImageProcessor:
    """图像处理器"""

    def __init__(self):
        """初始化图像处理器"""
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        加载图像

        Args:
            image_path: 图像路径

        Returns:
            图像数组或None
        """
        try:
            # 使用PIL加载图像，然后转换为numpy数组
            image = Image.open(image_path)

            # 转换RGB模式
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return np.array(image)
        except Exception as e:
            logger.error(f"加载图像失败 {image_path}: {str(e)}")
            return None

    def get_image_info(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        获取图像信息

        Args:
            image_path: 图像路径

        Returns:
            图像信息字典或None
        """
        try:
            path_obj = Path(image_path)
            if not path_obj.exists():
                return None

            # 获取文件信息
            stat = path_obj.stat()
            size_bytes = stat.st_size

            # 使用PIL获取图像信息
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format.lower() if img.format else 'unknown'
                channels = len(img.getbands()) if img.getbands() else 3

                # 处理RGBA等格式
                if img.mode in ('RGBA', 'LA'):
                    channels = 4
                elif img.mode == 'L':
                    channels = 1

            return {
                'path': str(path_obj.absolute()),
                'filename': path_obj.name,
                'width': width,
                'height': height,
                'channels': channels,
                'format': format_name,
                'size_bytes': size_bytes,
                'aspect_ratio': width / height if height > 0 else 1.0
            }
        except Exception as e:
            logger.error(f"获取图像信息失败 {image_path}: {str(e)}")
            return None

    def get_thumbnail(self, image_path: str, size: Tuple[int, int] = (256, 256),
                     quality: int = 85) -> Optional[bytes]:
        """
        生成缩略图

        Args:
            image_path: 图像路径
            size: 缩略图尺寸
            quality: JPEG质量（1-100）

        Returns:
            缩略图二进制数据或None
        """
        try:
            with Image.open(image_path) as img:
                # 转换RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 生成缩略图，保持宽高比
                img.thumbnail(size, Image.Resampling.LANCZOS)

                # 保存为字节流
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=quality, optimize=True)
                return buffer.getvalue()
        except Exception as e:
            logger.error(f"生成缩略图失败 {image_path}: {str(e)}")
            return None

    def get_preview(self, image_path: str, max_size: Tuple[int, int] = (1024, 1024)) -> Optional[bytes]:
        """
        生成预览图

        Args:
            image_path: 图像路径
            max_size: 最大尺寸

        Returns:
            预览图二进制数据或None
        """
        try:
            with Image.open(image_path) as img:
                # 转换RGB模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 计算缩放比例
                width, height = img.size
                max_width, max_height = max_size

                scale = min(1.0, max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)

                if scale < 1.0:
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # 保存为字节流
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=90, optimize=True)
                return buffer.getvalue()
        except Exception as e:
            logger.error(f"生成预览图失败 {image_path}: {str(e)}")
            return None

    def image_to_base64(self, image_data: bytes) -> str:
        """
        将图像二进制数据转换为base64字符串

        Args:
            image_data: 图像二进制数据

        Returns:
            base64字符串
        """
        return base64.b64encode(image_data).decode('utf-8')

    def base64_to_image(self, base64_str: str) -> Optional[np.ndarray]:
        """
        将base64字符串转换为图像数组

        Args:
            base64_str: base64字符串

        Returns:
            图像数组或None
        """
        try:
            # 解码base64
            image_data = base64.b64decode(base64_str)

            # 转换为PIL图像
            img = Image.open(io.BytesIO(image_data))

            # 转换为numpy数组
            if img.mode != 'RGB':
                img = img.convert('RGB')

            return np.array(img)
        except Exception as e:
            logger.error(f"base64转图像失败: {str(e)}")
            return None

    def flip_image(self, image: np.ndarray, horizontal: bool = True,
                   vertical: bool = False) -> np.ndarray:
        """
        翻转图像

        Args:
            image: 输入图像
            horizontal: 水平翻转
            vertical: 垂直翻转

        Returns:
            翻转后的图像
        """
        try:
            result = image.copy()

            if horizontal:
                result = cv2.flip(result, 1)  # 水平翻转
            if vertical:
                result = cv2.flip(result, 0)  # 垂直翻转

            return result
        except Exception as e:
            logger.error(f"图像翻转失败: {str(e)}")
            return image

    def rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        旋转图像

        Args:
            image: 输入图像
            angle: 旋转角度（度）

        Returns:
            旋转后的图像
        """
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)

            # 计算旋转矩阵
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # 计算新的边界框
            cos = np.abs(rotation_matrix[0, 0])
            sin = np.abs(rotation_matrix[0, 1])
            new_width = int((height * sin) + (width * cos))
            new_height = int((height * cos) + (width * sin))

            # 调整旋转矩阵
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]

            # 旋转图像
            rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

            return rotated
        except Exception as e:
            logger.error(f"图像旋转失败: {str(e)}")
            return image

    def crop_image(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        裁剪图像

        Args:
            image: 输入图像
            x: 起始x坐标
            y: 起始y坐标
            width: 裁剪宽度
            height: 裁剪高度

        Returns:
            裁剪后的图像
        """
        try:
            img_height, img_width = image.shape[:2]

            # 确保裁剪区域在图像范围内
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            width = min(width, img_width - x)
            height = min(height, img_height - y)

            if width <= 0 or height <= 0:
                return image

            return image[y:y+height, x:x+width]
        except Exception as e:
            logger.error(f"图像裁剪失败: {str(e)}")
            return image

    def scale_image(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        缩放图像

        Args:
            image: 输入图像
            scale_factor: 缩放因子

        Returns:
            缩放后的图像
        """
        try:
            if scale_factor <= 0:
                return image

            height, width = image.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        except Exception as e:
            logger.error(f"图像缩放失败: {str(e)}")
            return image

    def adjust_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整图像亮度

        Args:
            image: 输入图像
            factor: 亮度因子（1.0为原始亮度）

        Returns:
            调整后的图像
        """
        try:
            # 确保因子在合理范围内
            factor = max(0.1, min(3.0, factor))

            # 转换为PIL图像进行处理
            pil_image = Image.fromarray(image)
            enhancer = ImageEnhance.Brightness(pil_image)
            enhanced = enhancer.enhance(factor)

            return np.array(enhanced)
        except Exception as e:
            logger.error(f"亮度调整失败: {str(e)}")
            return image

    def adjust_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整图像对比度

        Args:
            image: 输入图像
            factor: 对比度因子（1.0为原始对比度）

        Returns:
            调整后的图像
        """
        try:
            # 确保因子在合理范围内
            factor = max(0.1, min(3.0, factor))

            # 转换为PIL图像进行处理
            pil_image = Image.fromarray(image)
            enhancer = ImageEnhance.Contrast(pil_image)
            enhanced = enhancer.enhance(factor)

            return np.array(enhanced)
        except Exception as e:
            logger.error(f"对比度调整失败: {str(e)}")
            return image

    def adjust_saturation(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整图像饱和度

        Args:
            image: 输入图像
            factor: 饱和度因子（1.0为原始饱和度）

        Returns:
            调整后的图像
        """
        try:
            # 确保因子在合理范围内
            factor = max(0.0, min(2.0, factor))

            # 转换为PIL图像进行处理
            pil_image = Image.fromarray(image)
            enhancer = ImageEnhance.Color(pil_image)
            enhanced = enhancer.enhance(factor)

            return np.array(enhanced)
        except Exception as e:
            logger.error(f"饱和度调整失败: {str(e)}")
            return image

    def adjust_hue(self, image: np.ndarray, shift: float) -> np.ndarray:
        """
        调整图像色调

        Args:
            image: 输入图像
            shift: 色调偏移（度）

        Returns:
            调整后的图像
        """
        try:
            # 转换到HSV色彩空间
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(hsv)

            # 调整色调（环绕处理）
            h = h.astype(np.int16)  # 转换为支持负数
            h = (h + int(shift)) % 180  # OpenCV的H范围是0-179
            h = h.astype(np.uint8)

            # 重新组合并转换回RGB
            hsv_shifted = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2RGB)
        except Exception as e:
            logger.error(f"色调调整失败: {str(e)}")
            return image

    def add_gaussian_blur(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        添加高斯模糊

        Args:
            image: 输入图像
            sigma: 高斯核标准差

        Returns:
            模糊后的图像
        """
        try:
            if sigma <= 0:
                return image

            # 计算核大小（必须是奇数）
            kernel_size = int(sigma * 3) | 1  # 确保为奇数

            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        except Exception as e:
            logger.error(f"高斯模糊失败: {str(e)}")
            return image

    def add_noise(self, image: np.ndarray, std: float) -> np.ndarray:
        """
        添加高斯噪声

        Args:
            image: 输入图像
            std: 噪声标准差

        Returns:
            添加噪声后的图像
        """
        try:
            if std <= 0:
                return image

            # 生成高斯噪声
            noise = np.random.normal(0, std, image.shape)

            # 添加噪声并确保像素值在有效范围内
            noisy_image = image + noise
            noisy_image = np.clip(noisy_image, 0, 255)

            return noisy_image.astype(np.uint8)
        except Exception as e:
            logger.error(f"添加噪声失败: {str(e)}")
            return image

    def apply_augmentations(self, image: np.ndarray,
                          config: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """
        应用数据增强

        Args:
            image: 输入图像
            config: 增强配置

        Returns:
            (增强后的图像, 应用的操作列表)
        """
        try:
            result = image.copy()
            applied_operations = []

            # 翻转
            if config.get('flip_horizontal', False):
                result = self.flip_image(result, horizontal=True)
                applied_operations.append('水平翻转')

            if config.get('flip_vertical', False):
                result = self.flip_image(result, vertical=True)
                applied_operations.append('垂直翻转')

            # 旋转
            rotation_angle = config.get('rotation_angle', 0.0)
            if abs(rotation_angle) > 0.1:
                result = self.rotate_image(result, rotation_angle)
                applied_operations.append(f'旋转{rotation_angle:.1f}度')

            # 裁剪
            crop_params = config.get('crop_params')
            if crop_params and all(k in crop_params for k in ['x', 'y', 'width', 'height']):
                result = self.crop_image(
                    result,
                    crop_params['x'],
                    crop_params['y'],
                    crop_params['width'],
                    crop_params['height']
                )
                applied_operations.append(f'裁剪({crop_params["x"]},{crop_params["y"]},{crop_params["width"]},{crop_params["height"]})')

            # 缩放
            scale_factor = config.get('scale_factor', 1.0)
            if abs(scale_factor - 1.0) > 0.01:
                result = self.scale_image(result, scale_factor)
                applied_operations.append(f'缩放{scale_factor:.2f}倍')

            # 亮度
            brightness_factor = config.get('brightness_factor', 1.0)
            if abs(brightness_factor - 1.0) > 0.01:
                result = self.adjust_brightness(result, brightness_factor)
                applied_operations.append(f'亮度{brightness_factor:.2f}')

            # 对比度
            contrast_factor = config.get('contrast_factor', 1.0)
            if abs(contrast_factor - 1.0) > 0.01:
                result = self.adjust_contrast(result, contrast_factor)
                applied_operations.append(f'对比度{contrast_factor:.2f}')

            # 饱和度
            saturation_factor = config.get('saturation_factor', 1.0)
            if abs(saturation_factor - 1.0) > 0.01:
                result = self.adjust_saturation(result, saturation_factor)
                applied_operations.append(f'饱和度{saturation_factor:.2f}')

            # 色调
            hue_shift = config.get('hue_shift', 0.0)
            if abs(hue_shift) > 0.1:
                result = self.adjust_hue(result, hue_shift)
                applied_operations.append(f'色调偏移{hue_shift:.1f}度')

            # 高斯模糊
            gaussian_blur = config.get('gaussian_blur', 0.0)
            if gaussian_blur > 0.1:
                result = self.add_gaussian_blur(result, gaussian_blur)
                applied_operations.append(f'高斯模糊σ={gaussian_blur:.1f}')

            # 噪声
            noise_std = config.get('noise_std', 0.0)
            if noise_std > 0.1:
                result = self.add_noise(result, noise_std)
                applied_operations.append(f'高斯噪声σ={noise_std:.1f}')

            return result, applied_operations
        except Exception as e:
            logger.error(f"应用数据增强失败: {str(e)}")
            return image, []

    def save_image(self, image: np.ndarray, output_path: str,
                  quality: int = 95, format: str = 'JPEG') -> bool:
        """
        保存图像

        Args:
            image: 图像数组
            output_path: 输出路径
            quality: JPEG质量
            format: 图像格式

        Returns:
            是否保存成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # 转换为PIL图像
            pil_image = Image.fromarray(image)

            # 保存图像
            pil_image.save(output_path, format=format, quality=quality, optimize=True)
            return True
        except Exception as e:
            logger.error(f"保存图像失败 {output_path}: {str(e)}")
            return False

    def image_array_to_bytes(self, image: np.ndarray, format: str = 'JPEG',
                           quality: int = 90) -> Optional[bytes]:
        """
        将图像数组转换为字节流

        Args:
            image: 图像数组
            format: 图像格式
            quality: 图像质量

        Returns:
            图像字节数据或None
        """
        try:
            # 转换为PIL图像
            pil_image = Image.fromarray(image)

            # 保存到字节流
            buffer = io.BytesIO()
            if format.upper() == 'PNG':
                pil_image.save(buffer, format='PNG')
            else:
                pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)

            return buffer.getvalue()
        except Exception as e:
            logger.error(f"图像转字节失败: {str(e)}")
            return None


# 创建全局实例
image_processor = ImageProcessor()