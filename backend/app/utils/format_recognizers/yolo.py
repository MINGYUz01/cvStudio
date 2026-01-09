"""
YOLO格式数据集识别器
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class YOLORecognizer:
    """YOLO格式数据集识别器"""

    def __init__(self):
        self.required_files = ['obj.names', 'obj.data', 'train.txt']
        self.required_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.required_label_extensions = {'.txt'}

    def recognize(self, dataset_path: str) -> Dict:
        """
        识别YOLO格式数据集

        Args:
            dataset_path: 数据集路径

        Returns:
            Dict: 识别结果
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return {"format": "unknown", "error": "数据集路径不存在"}

        result = {
            "format": "yolo",
            "confidence": 0,
            "details": {},
            "error": None
        }

        try:
            # 检查是否有obj.names文件（类别名称）
            classes_file = self._find_classes_file(dataset_path)
            classes = []
            if classes_file and classes_file.exists():
                classes = self._load_classes(classes_file)
                result["details"]["classes"] = classes

            # 检查是否有obj.data文件（数据集配置）
            data_file = self._find_data_file(dataset_path)
            data_config = {}
            if data_file and data_file.exists():
                data_config = self._load_data_config(data_file)
                result["details"]["data_config"] = data_config

            # 查找图像和标签文件
            images_info = self._find_images_and_labels(dataset_path)
            result["details"].update(images_info)

            # 计算置信度
            confidence = self._calculate_confidence(
                bool(classes_file),
                bool(data_file),
                images_info.get("num_images", 0),
                images_info.get("num_labels", 0)
            )
            result["confidence"] = confidence

            # 如果置信度太低，可能是其他格式
            if confidence < 0.3:
                result["format"] = "unknown"
                result["error"] = "未找到足够的YOLO格式特征文件"

        except Exception as e:
            result["error"] = f"识别YOLO格式时出错: {str(e)}"
            result["confidence"] = 0

        return result

    def _find_classes_file(self, dataset_path: Path) -> Optional[Path]:
        """查找类别名称文件"""
        possible_names = ['obj.names', 'classes.txt', 'names.txt']

        for name in possible_names:
            file_path = dataset_path / name
            if file_path.exists():
                return file_path

            # 递归查找
            for file_path in dataset_path.rglob(name):
                if file_path.is_file():
                    return file_path

        return None

    def _find_data_file(self, dataset_path: Path) -> Optional[Path]:
        """查找数据配置文件"""
        possible_names = ['obj.data', 'dataset.data', 'yolo.data']

        for name in possible_names:
            file_path = dataset_path / name
            if file_path.exists():
                return file_path

            # 递归查找
            for file_path in dataset_path.rglob(name):
                if file_path.is_file():
                    return file_path

        return None

    def _load_classes(self, classes_file: Path) -> List[str]:
        """加载类别名称"""
        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
            return classes
        except Exception:
            return []

    def _load_data_config(self, data_file: Path) -> Dict:
        """加载数据配置"""
        config = {}
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if line and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # 移除引号
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    config[key] = value

        except Exception as e:
            print(f"加载数据配置文件失败: {e}")

        return config

    def _find_images_and_labels(self, dataset_path: Path) -> Dict:
        """查找图像和标签文件"""
        images = []
        labels = []

        # 递归查找
        for file_path in dataset_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in self.required_image_extensions:
                    images.append(file_path)
                elif ext in self.required_label_extensions:
                    # 检查是否为标签文件（不是obj.names等文件）
                    if file_path.name not in ['obj.names', 'classes.txt', 'names.txt']:
                        labels.append(file_path)

        # 统计信息
        num_images = len(images)
        num_labels = len(labels)

        # 分析标签内容
        label_stats = self._analyze_labels(labels) if labels else {}

        # 图像尺寸统计（抽样）
        image_stats = self._analyze_images(images[:100])  # 最多分析100张图

        return {
            "num_images": num_images,
            "num_labels": num_labels,
            "image_paths": [str(img) for img in images[:50]],  # 最多返回50个路径
            "label_paths": [str(lbl) for lbl in labels[:50]],  # 最多返回50个路径
            "label_stats": label_stats,
            "image_stats": image_stats
        }

    def _analyze_labels(self, label_files: List[Path]) -> Dict:
        """分析标签文件"""
        if not label_files:
            return {}

        class_counts = {}
        object_counts = []
        valid_labels = 0
        annotated_images = 0  # 有标注的图片数

        for label_file in label_files[:100]:  # 最多分析100个标签文件
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                file_objects = 0
                for line in lines:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:  # YOLO格式: class_id x_center y_center width height
                            class_id = int(parts[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
                            file_objects += 1
                            valid_labels += 1

                if file_objects > 0:
                    object_counts.append(file_objects)
                    annotated_images += 1  # 统计有标注的图片

            except Exception:
                continue

        return {
            "valid_labels": valid_labels,
            "annotated_images": annotated_images,  # 添加有标注的图片数
            "class_distribution": class_counts,
            "avg_objects_per_image": sum(object_counts) / len(object_counts) if object_counts else 0,
            "max_objects_per_image": max(object_counts) if object_counts else 0
        }

    def _analyze_images(self, image_files: List[Path]) -> Dict:
        """分析图像文件"""
        if not image_files:
            return {}

        try:
            from PIL import Image
        except ImportError:
            return {"error": "需要安装Pillow库来分析图像"}

        sizes = []
        formats = {}

        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    sizes.append((width, height))

                    img_format = img.format or "unknown"
                    formats[img_format] = formats.get(img_format, 0) + 1

            except Exception:
                continue

        if not sizes:
            return {}

        widths = [w for w, h in sizes]
        heights = [h for w, h in sizes]

        return {
            "width_range": [min(widths), max(widths)],
            "height_range": [min(heights), max(heights)],
            "avg_width": sum(widths) / len(widths),
            "avg_height": sum(heights) / len(heights),
            "format_distribution": formats,
            "total_analyzed": len(sizes)
        }

    def _calculate_confidence(self, has_classes: bool, has_data: bool,
                            num_images: int, num_labels: int) -> float:
        """计算YOLO格式置信度"""
        confidence = 0.0

        # 类别文件权重
        if has_classes:
            confidence += 0.3

        # 数据配置文件权重
        if has_data:
            confidence += 0.3

        # 图像和标签文件权重
        if num_images > 0:
            confidence += 0.2
            # 图像数量加分
            if num_images >= 10:
                confidence += 0.1

        if num_labels > 0:
            confidence += 0.2
            # 标签数量加分
            if num_labels >= 10:
                confidence += 0.1

        # 标签与图像比例加分
        if num_images > 0 and num_labels > 0:
            ratio = min(num_labels, num_images) / max(num_labels, num_images)
            if ratio > 0.8:  # 标签和图像数量接近
                confidence += 0.1

        return min(confidence, 1.0)