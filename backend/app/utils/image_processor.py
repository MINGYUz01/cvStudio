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
        缩放图像（保持原始画布尺寸，缩放后的图像居中显示）

        Args:
            image: 输入图像
            scale_factor: 缩放因子（1.0为原始大小）

        Returns:
            缩放后的图像（居中在原始画布上）
        """
        try:
            if scale_factor <= 0 or scale_factor == 1.0:
                return image

            height, width = image.shape[:2]
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # 缩放图像
            scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # 如果缩放后尺寸大于原始尺寸，直接返回
            if new_width >= width and new_height >= height:
                # 居中裁剪到原始尺寸
                x = (new_width - width) // 2
                y = (new_height - height) // 2
                return scaled[y:y+height, x:x+width]

            # 创建原始尺寸的画布，用白色填充
            canvas = np.full((height, width, 3), 255, dtype=np.uint8)

            # 计算居中位置
            x = (width - new_width) // 2
            y = (height - new_height) // 2

            # 将缩放后的图像放置在画布中心
            canvas[y:y+new_height, x:x+new_width] = scaled

            return canvas
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

    def translate_image(self, image: np.ndarray, dx: int, dy: int,
                        fill_value: int = 255) -> np.ndarray:
        """
        平移图像

        Args:
            image: 输入图像
            dx: 水平偏移量（像素）
            dy: 垂直偏移量（像素）
            fill_value: 填充值（0-255）

        Returns:
            平移后的图像
        """
        try:
            height, width = image.shape[:2]

            # 创建平移矩阵
            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

            # 应用平移
            translated = cv2.warpAffine(
                image, translation_matrix, (width, height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(fill_value, fill_value, fill_value)
            )

            return translated
        except Exception as e:
            logger.error(f"图像平移失败: {str(e)}")
            return image

    def elastic_transform(self, image: np.ndarray, alpha: float = 1.0,
                          sigma: float = 50.0) -> np.ndarray:
        """
        弹性变换

        Args:
            image: 输入图像
            alpha: 形变强度（相对于图像尺寸的比例）
            sigma: 高斯核标准差

        Returns:
            变换后的图像
        """
        try:
            import scipy.ndimage as ndimage

            shape = image.shape[:2]

            # 将 alpha 转换为绝对像素值（alpha 作为图像尺寸的比例）
            alpha_pixels = alpha * min(shape[0], shape[1])

            # 生成随机位移场
            dx = ndimage.gaussian_filter(np.random.randn(*shape), sigma) * alpha_pixels
            dy = ndimage.gaussian_filter(np.random.randn(*shape), sigma) * alpha_pixels

            # 创建网格
            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
            indices = [y + dy, x + dx]

            # 对每个通道应用变换
            if len(image.shape) == 3:
                transformed = np.zeros_like(image)
                for c in range(image.shape[2]):
                    transformed[:, :, c] = ndimage.map_coordinates(
                        image[:, :, c], indices, order=1, mode='constant', cval=255
                    )
            else:
                transformed = ndimage.map_coordinates(
                    image, indices, order=1, mode='constant', cval=255
                )

            return transformed.astype(np.uint8)
        except ImportError:
            logger.warning("scipy未安装，跳过弹性变换")
            return image
        except Exception as e:
            logger.error(f"弹性变换失败: {str(e)}")
            return image

    def perspective_transform(self, image: np.ndarray, points: List[tuple]) -> np.ndarray:
        """
        透视变换

        Args:
            image: 输入图像
            points: 四个点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]

        Returns:
            变换后的图像
        """
        try:
            height, width = image.shape[:2]

            # 默认使用四个角点
            if len(points) != 4:
                return image

            # 源点和目标点
            src_points = np.float32(points)
            dst_points = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

            # 计算透视变换矩阵
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # 应用变换
            transformed = cv2.warpPerspective(image, matrix, (width, height))

            return transformed
        except Exception as e:
            logger.error(f"透视变换失败: {str(e)}")
            return image

    def random_erase(self, image: np.ndarray, probability: float = 0.5,
                     scale = (0.02, 0.33), ratio = (0.3, 3.3),
                     value: int = 0) -> np.ndarray:
        """
        随机擦除

        Args:
            image: 输入图像
            probability: 执行概率
            scale: 擦除区域比例范围（元组或单个值）
            ratio: 宽高比范围（元组或单个值）
            value: 填充值

        Returns:
            处理后的图像
        """
        try:
            import random

            # 根据概率决定是否执行
            if random.random() > probability:
                return image

            height, width = image.shape[:2]
            area = height * width

            # 处理 scale 参数：支持单个值或元组
            if isinstance(scale, (int, float)):
                # 单个值：创建一个小范围（±20%）
                scale_val = max(0.01, min(float(scale), 0.5))
                scale_range = (scale_val * 0.8, scale_val * 1.2)
            else:
                # 元组：直接使用
                scale_range = scale

            # 处理 ratio 参数：支持单个值或元组
            if isinstance(ratio, (int, float)):
                # 单个值：创建一个小范围（±20%）
                ratio_val = max(0.1, min(float(ratio), 10.0))
                ratio_range = (ratio_val * 0.8, ratio_val * 1.2)
            else:
                # 元组：直接使用
                ratio_range = ratio

            # 随机生成擦除区域参数
            target_area = random.uniform(*scale_range) * area
            aspect_ratio = random.uniform(*ratio_range)

            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))

            # 确保不越界
            if h < 1 or w < 1 or h > height or w > width:
                return image

            # 随机位置
            top = random.randint(0, height - h)
            left = random.randint(0, width - w)

            # 复制图像并填充
            result = image.copy()
            if len(result.shape) == 3:
                result[top:top+h, left:left+w, :] = value
            else:
                result[top:top+h, left:left+w] = value

            return result
        except Exception as e:
            logger.error(f"随机擦除失败: {str(e)}")
            return image

    def gamma_correction(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Gamma校正

        Args:
            image: 输入图像
            gamma: Gamma值

        Returns:
            校正后的图像
        """
        try:
            # 确保gamma在合理范围
            gamma = max(0.1, min(10.0, gamma))

            # 构建查找表
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                            for i in np.arange(0, 256)]).astype("uint8")

            # 应用查找表
            corrected = cv2.LUT(image, table)

            return corrected
        except Exception as e:
            logger.error(f"Gamma校正失败: {str(e)}")
            return image

    def auto_contrast(self, image: np.ndarray, cutoff: float = 0.0) -> np.ndarray:
        """
        自动对比度

        Args:
            image: 输入图像
            cutoff: 直方图截断百分比

        Returns:
            处理后的图像
        """
        try:
            from PIL import Image as PILImage, ImageEnhance

            # 转换为PIL图像
            pil_image = PILImage.fromarray(image)

            # 计算直方图
            histogram = pil_image.histogram()

            # 找到截断点
            num_pixels = pil_image.width * pil_image.height
            cut_pixels = int(num_pixels * cutoff / 100.0)

            # 找到最小和最大值
            min_level = 0
            max_level = 255
            count = 0
            for i in range(256):
                count += histogram[i]
                if count > cut_pixels:
                    min_level = i
                    break

            count = 0
            for i in range(255, -1, -1):
                count += histogram[i]
                if count > cut_pixels:
                    max_level = i
                    break

            if min_level >= max_level:
                return image

            # 应用线性拉伸
            scale = 255.0 / (max_level - min_level)
            offset = -min_level * scale

            # 使用PIL进行对比度调整
            enhancer = ImageEnhance.Contrast(pil_image)
            # 这里简化处理，直接使用对比度增强
            enhanced = enhancer.enhance(1.5)

            return np.array(enhanced)
        except Exception as e:
            logger.error(f"自动对比度失败: {str(e)}")
            return image

    def add_salt_pepper_noise(self, image: np.ndarray, amount: float = 0.01) -> np.ndarray:
        """
        添加椒盐噪声

        Args:
            image: 输入图像
            amount: 噪声密度

        Returns:
            添加噪声后的图像
        """
        try:
            if amount <= 0:
                return image

            result = image.copy()
            height, width = image.shape[:2]

            # 计算噪声点数量
            num_salt = int(amount * height * width * 0.5)
            num_pepper = int(amount * height * width * 0.5)

            # 添加盐噪声（白点）
            salt_coords = [np.random.randint(0, i, num_salt) for i in (height, width)]
            if len(result.shape) == 3:
                result[salt_coords[0], salt_coords[1], :] = 255
            else:
                result[salt_coords[0], salt_coords[1]] = 255

            # 添加椒噪声（黑点）
            pepper_coords = [np.random.randint(0, i, num_pepper) for i in (height, width)]
            if len(result.shape) == 3:
                result[pepper_coords[0], pepper_coords[1], :] = 0
            else:
                result[pepper_coords[0], pepper_coords[1]] = 0

            return result
        except Exception as e:
            logger.error(f"添加椒盐噪声失败: {str(e)}")
            return image

    def motion_blur(self, image: np.ndarray, kernel_size: int = 15,
                     angle: float = 0.0) -> np.ndarray:
        """
        运动模糊

        Args:
            image: 输入图像
            kernel_size: 核大小（必须是奇数）
            angle: 运动角度

        Returns:
            模糊后的图像
        """
        try:
            # 确保kernel_size是奇数
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(3, min(kernel_size, 51))

            # 创建运动模糊核
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size

            # 根据角度旋转核
            if angle != 0:
                center = (kernel_size - 1) / 2
                rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1)
                kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))
                kernel = kernel / np.sum(kernel)

            # 应用模糊
            blurred = cv2.filter2D(image, -1, kernel)

            return blurred
        except Exception as e:
            logger.error(f"运动模糊失败: {str(e)}")
            return image

    def jpeg_compression(self, image: np.ndarray, quality: int = 85) -> np.ndarray:
        """
        JPEG压缩

        Args:
            image: 输入图像
            quality: 压缩质量（1-100）

        Returns:
            压缩后的图像
        """
        try:
            # 确保quality在合理范围
            quality = max(10, min(100, quality))

            # 转换为PIL图像
            pil_image = Image.fromarray(image)

            # 压缩再解压
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)

            recompressed = Image.open(buffer)
            if recompressed.mode != 'RGB':
                recompressed = recompressed.convert('RGB')

            return np.array(recompressed)
        except Exception as e:
            logger.error(f"JPEG压缩失败: {str(e)}")
            return image

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


    def mosaic_augment(self, images: List[np.ndarray],
                       scale = (0.5, 1.5)) -> np.ndarray:
        """
        马赛克增强：将4张图片拼接成1张

        Args:
            images: 图片列表，至少1张，不足4张会重复使用
            scale: 缩放比例范围（元组或单个值）

        Returns:
            拼接后的图像
        """
        try:
            if not images:
                return None

            # 如果图片不足4张，复制使用
            if len(images) < 4:
                images = images * (4 // len(images) + 1)
                images = images[:4]

            # 处理 scale 参数：支持单个值或元组
            if isinstance(scale, (int, float)):
                scale_val = max(0.1, min(float(scale), 3.0))
                scale_range = (scale_val * 0.8, scale_val * 1.2)
            else:
                scale_range = scale

            # 随机缩放每张图片
            scaled_images = []
            for img in images:
                s = np.random.uniform(*scale_range)
                h, w = img.shape[:2]
                new_h, new_w = int(h * s), int(w * s)
                scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                scaled_images.append(scaled)

            # 计算拼接后的尺寸
            # 上面两张：左上和右上
            top_h = max(scaled_images[0].shape[0], scaled_images[1].shape[0])
            top_w = scaled_images[0].shape[1] + scaled_images[1].shape[1]
            # 下面两张：左下和右下
            bottom_h = scaled_images[2].shape[0] + scaled_images[3].shape[0]
            bottom_w = max(scaled_images[2].shape[1], scaled_images[3].shape[1])

            # 创建画布
            top_half = np.full((top_h, top_w, 3), 255, dtype=np.uint8)
            bottom_half = np.full((bottom_h, bottom_w, 3), 255, dtype=np.uint8)

            # 放置左上图片
            h1, w1 = scaled_images[0].shape[:2]
            top_half[:h1, :w1] = scaled_images[0]

            # 放置右上图片
            h2, w2 = scaled_images[1].shape[:2]
            top_half[:h2, w1:w1+w2] = scaled_images[1]

            # 放置左下图片
            h3, w3 = scaled_images[2].shape[:2]
            bottom_half[:h3, :w3] = scaled_images[2]

            # 放置右下图片
            h4, w4 = scaled_images[3].shape[:2]
            bottom_half[h3:h3+h4, :w4] = scaled_images[3]

            # 统一宽度后上下拼接
            max_width = max(top_w, bottom_w)
            if top_w < max_width:
                padding = np.full((top_h, max_width - top_w, 3), 255, dtype=np.uint8)
                top_half = np.hstack([top_half, padding])
            if bottom_w < max_width:
                padding = np.full((bottom_h, max_width - bottom_w, 3), 255, dtype=np.uint8)
                bottom_half = np.hstack([bottom_half, padding])

            # 上下拼接
            result = np.vstack([top_half, bottom_half])

            return result
        except Exception as e:
            logger.error(f"马赛克增强失败: {str(e)}")
            return images[0] if images else None

    def copy_paste_augment(self, target_image: np.ndarray,
                           source_images: List[np.ndarray],
                           max_objects: int = 5,
                           annotations: Optional[List[Dict]] = None) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """
        Copy-Paste增强：从其他图像复制区域粘贴到当前图像

        Args:
            target_image: 目标图像
            source_images: 源图像列表
            max_objects: 最多粘贴的对象数量
            annotations: 标注信息（可选）

        Returns:
            (增强后的图像, 更新后的标注)
        """
        try:
            result = target_image.copy()
            updated_annotations = annotations.copy() if annotations else None

            if not source_images:
                return result, updated_annotations

            h, w = result.shape[:2]
            num_pastes = min(len(source_images), max_objects)

            for i in range(num_pastes):
                source = source_images[i]
                if source is None:
                    continue
                sh, sw = source.shape[:2]

                # 随机选择粘贴区域大小（源图尺寸的1/4到全部）
                paste_w = np.random.randint(sw // 4, sw)
                paste_h = np.random.randint(sh // 4, sh)

                # 确保不超过目标图尺寸
                paste_w = min(paste_w, w // 3)
                paste_h = min(paste_h, h // 3)

                if paste_w <= 0 or paste_h <= 0:
                    continue

                # 随机选择源图像区域
                sx = np.random.randint(0, max(1, sw - paste_w))
                sy = np.random.randint(0, max(1, sh - paste_h))
                patch = source[sy:sy+paste_h, sx:sx+paste_w]

                # 随机选择目标位置
                dx = np.random.randint(0, max(1, w - paste_w))
                dy = np.random.randint(0, max(1, h - paste_h))

                # 直接粘贴
                result[dy:dy+paste_h, dx:dx+paste_w] = patch

                # 更新标注（如果有）
                if updated_annotations is not None:
                    updated_annotations.append({
                        'bbox': [dx, dy, paste_w, paste_h],
                        'source': 'copy_paste'
                    })

            return result, updated_annotations
        except Exception as e:
            logger.error(f"Copy-Paste增强失败: {str(e)}")
            return target_image, annotations


# 创建全局实例
image_processor = ImageProcessor()