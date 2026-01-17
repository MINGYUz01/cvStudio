"""
文件夹分类格式数据集识别器
"""

import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class ClassificationRecognizer:
    """文件夹分类格式数据集识别器"""

    def __init__(self):
        self.required_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    def recognize(self, dataset_path: str) -> Dict:
        """
        识别文件夹分类格式数据集

        Args:
            dataset_path: 数据集路径

        Returns:
            Dict: 识别结果
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return {"format": "unknown", "error": "数据集路径不存在"}

        result = {
            "format": "classification",
            "confidence": 0,
            "details": {},
            "error": None
        }

        try:
            # 分析目录结构
            structure_info = self._analyze_structure(dataset_path)
            result["details"]["structure"] = structure_info

            # 检查是否为分类结构
            is_classification = self._is_classification_structure(structure_info)
            result["details"]["is_classification"] = is_classification

            if is_classification:
                # 提取类别信息
                classes_info = self._extract_classes(dataset_path, structure_info)
                result["details"]["classes"] = classes_info.get("class_names", [])
                result["details"]["class_directories"] = classes_info.get("class_directories", {})
                result["details"]["num_classes"] = classes_info.get("num_classes", 0)

                # 分析图像信息
                images_info = self._analyze_images(dataset_path, classes_info.get("class_directories", {}))
                result["details"].update(images_info)

                # 计算置信度
                confidence = self._calculate_confidence(
                    structure_info,
                    classes_info.get("num_classes", 0),
                    images_info.get("total_images", 0),
                    images_info.get("avg_images_per_class", 0)
                )
                result["confidence"] = confidence
            else:
                result["confidence"] = 0
                result["error"] = "不匹配文件夹分类格式结构"

        except Exception as e:
            result["error"] = f"识别文件夹分类格式时出错: {str(e)}"
            result["confidence"] = 0

        return result

    def _analyze_structure(self, dataset_path: Path) -> Dict:
        """分析目录结构"""
        structure = {
            "root_path": str(dataset_path),
            "total_items": 0,
            "directories": 0,
            "files": 0,
            "image_files": 0,
            "subdirectory_levels": 0,
            "has_image_directories": False,
            "direct_image_files": [],
            "directory_structure": {}
        }

        try:
            items = list(dataset_path.iterdir())
            structure["total_items"] = len(items)

            for item in items:
                if item.is_dir():
                    structure["directories"] += 1
                    dir_info = self._analyze_directory(item)
                    structure["directory_structure"][item.name] = dir_info

                    if dir_info.get("image_count", 0) > 0:
                        structure["has_image_directories"] = True

                elif item.is_file():
                    structure["files"] += 1
                    if item.suffix.lower() in self.required_image_extensions:
                        structure["image_files"] += 1
                        structure["direct_image_files"].append(item.name)

            # 计算子目录层级
            structure["subdirectory_levels"] = self._calculate_directory_levels(dataset_path)

        except Exception as e:
            structure["error"] = f"分析目录结构失败: {str(e)}"

        return structure

    def _analyze_directory(self, directory_path: Path, max_depth: int = 3) -> Dict:
        """分析单个目录"""
        dir_info = {
            "path": str(directory_path),
            "image_count": 0,
            "total_files": 0,
            "subdirectories": 0,
            "image_files": [],
            "has_subdirs": False,
            "image_extensions": {}
        }

        try:
            for item in directory_path.iterdir():
                if item.is_file() and item.suffix.lower() in self.required_image_extensions:
                    dir_info["image_count"] += 1
                    dir_info["total_files"] += 1
                    dir_info["image_files"].append(item.name)

                    ext = item.suffix.lower()
                    dir_info["image_extensions"][ext] = dir_info["image_extensions"].get(ext, 0) + 1

                elif item.is_file():
                    dir_info["total_files"] += 1

                elif item.is_dir():
                    dir_info["subdirectories"] += 1
                    dir_info["has_subdirs"] = True

        except Exception:
            pass

        return dir_info

    def _calculate_directory_levels(self, dataset_path: Path, current_level: int = 1) -> int:
        """计算目录层级"""
        try:
            has_subdirs = False
            for item in dataset_path.iterdir():
                if item.is_dir():
                    has_subdirs = True
                    if current_level < 3:  # 限制递归深度
                        return self._calculate_directory_levels(item, current_level + 1)

            return current_level if has_subdirs else current_level - 1

        except Exception:
            return current_level

    def _is_classification_structure(self, structure_info: Dict) -> bool:
        """判断是否为分类目录结构"""
        # 规则1: 有多个子目录，每个子目录包含图像文件
        if structure_info.get("has_image_directories", False):
            dir_structure = structure_info.get("directory_structure", {})
            image_dirs = [name for name, info in dir_structure.items()
                         if info.get("image_count", 0) > 0]

            if len(image_dirs) >= 2:  # 至少有2个包含图像的目录
                return True

        # 规则2: 根目录直接有图像文件，但目录数更多
        direct_images = structure_info.get("image_files", 0)
        dirs_with_images = len([info for info in structure_info.get("directory_structure", {}).values()
                               if info.get("image_count", 0) > 0])

        if dirs_with_images >= 2 and direct_images < dirs_with_images * 10:
            return True

        # 规则3: 没有直接的图像文件，但有多个目录
        if direct_images == 0 and structure_info.get("directories", 0) >= 2:
            # 检查是否至少有一半的目录包含图像
            total_dirs = structure_info.get("directories", 0)
            dirs_with_images = len([info for info in structure_info.get("directory_structure", {}).values()
                                   if info.get("image_count", 0) > 0])

            if dirs_with_images >= max(2, total_dirs // 2):
                return True

        return False

    def _extract_classes(self, dataset_path: Path, structure_info: Dict) -> Dict:
        """提取类别信息"""
        classes_info = {
            "num_classes": 0,
            "class_directories": {},
            "class_names": [],
            "total_class_images": 0,
            "min_images_per_class": float('inf'),
            "max_images_per_class": 0,
            "avg_images_per_class": 0,
            "class_balance_score": 0
        }

        dir_structure = structure_info.get("directory_structure", {})
        image_counts = []

        for dir_name, dir_info in dir_structure.items():
            image_count = dir_info.get("image_count", 0)
            if image_count > 0:  # 只统计包含图像的目录
                classes_info["class_directories"][dir_name] = {
                    "directory": dir_name,
                    "image_count": image_count,
                    "path": dir_info.get("path"),
                    "image_extensions": dir_info.get("image_extensions", {})
                }
                classes_info["class_names"].append(dir_name)
                classes_info["total_class_images"] += image_count
                image_counts.append(image_count)

        classes_info["num_classes"] = len(classes_info["class_names"])

        if image_counts:
            classes_info["min_images_per_class"] = min(image_counts)
            classes_info["max_images_per_class"] = max(image_counts)
            classes_info["avg_images_per_class"] = sum(image_counts) / len(image_counts)

            # 计算类别平衡分数（越接近1越平衡）
            if classes_info["avg_images_per_class"] > 0:
                variance = sum((count - classes_info["avg_images_per_class"]) ** 2
                             for count in image_counts) / len(image_counts)
                std_dev = variance ** 0.5
                classes_info["class_balance_score"] = max(0, 1 - (std_dev / classes_info["avg_images_per_class"]))
        else:
            classes_info["min_images_per_class"] = 0

        return classes_info

    def _analyze_images(self, dataset_path: Path, class_directories: Dict) -> Dict:
        """分析图像信息"""
        images_info = {
            "total_images": 0,
            "analyzed_images": 0,
            "size_distribution": {},
            "format_distribution": {},
            "avg_width": 0,
            "avg_height": 0,
            "width_range": [0, 0],
            "height_range": [0, 0],
            "sample_images": {},
            "image_paths": []  # 添加图像路径列表
        }

        try:
            from PIL import Image
        except ImportError:
            images_info["error"] = "需要安装Pillow库来分析图像"
            return images_info

        sizes = []
        formats = {}
        total_images = 0
        all_image_paths = []  # 收集所有图像路径

        # 从每个类别目录中抽样分析图像
        for class_name, class_info in list(class_directories.items())[:10]:  # 最多分析10个类别
            class_path = Path(class_info["path"])
            sample_count = 0

            for img_file in class_path.iterdir():
                if (img_file.is_file() and
                    img_file.suffix.lower() in self.required_image_extensions):

                    # 添加到所有图像路径列表
                    all_image_paths.append(str(img_file))

                    # 只分析前20张用于统计
                    if sample_count < 20:
                        try:
                            with Image.open(img_file) as img:
                                width, height = img.size
                                sizes.append((width, height))

                                img_format = img.format or "unknown"
                                formats[img_format] = formats.get(img_format, 0) + 1

                                sample_count += 1

                                # 保存样本图像信息
                                if class_name not in images_info["sample_images"]:
                                    images_info["sample_images"][class_name] = []

                                if len(images_info["sample_images"][class_name]) < 3:
                                    images_info["sample_images"][class_name].append({
                                        "filename": img_file.name,
                                        "width": width,
                                        "height": height,
                                        "format": img_format,
                                        "path": str(img_file)  # 添加路径信息
                                    })

                        except Exception:
                            continue

                    if total_images >= 200:  # 最多分析200张图像
                        break

            if total_images >= 200:
                break

        images_info["total_images"] = sum(info.get("image_count", 0) for info in class_directories.values())
        images_info["num_images"] = images_info["total_images"]  # 添加 num_images 别名以兼容
        images_info["analyzed_images"] = len(sizes)
        images_info["image_paths"] = all_image_paths  # 保存所有图像路径

        if sizes:
            widths = [w for w, h in sizes]
            heights = [h for w, h in sizes]

            images_info["avg_width"] = sum(widths) / len(widths)
            images_info["avg_height"] = sum(heights) / len(heights)
            images_info["width_range"] = [min(widths), max(widths)]
            images_info["height_range"] = [min(heights), max(heights)]
            images_info["format_distribution"] = formats

            # 尺寸分布统计
            size_ranges = {
                "small": 0,    # < 224x224
                "medium": 0,   # 224x224 - 512x512
                "large": 0,    # 512x512 - 1024x1024
                "xlarge": 0    # > 1024x1024
            }

            for w, h in sizes:
                pixels = w * h
                if pixels < 224 * 224:
                    size_ranges["small"] += 1
                elif pixels < 512 * 512:
                    size_ranges["medium"] += 1
                elif pixels < 1024 * 1024:
                    size_ranges["large"] += 1
                else:
                    size_ranges["xlarge"] += 1

            images_info["size_distribution"] = size_ranges

        return images_info

    def _calculate_confidence(self, structure_info: Dict, num_classes: int,
                            total_images: int, avg_images_per_class: float) -> float:
        """计算文件夹分类格式置信度"""
        confidence = 0.0

        # 类别数量权重
        if num_classes >= 2:
            confidence += 0.3
            if num_classes >= 3:
                confidence += 0.1
            if num_classes >= 5:
                confidence += 0.1

        # 图像数量权重
        if total_images > 0:
            confidence += 0.2
            if total_images >= 50:
                confidence += 0.1
            if total_images >= 200:
                confidence += 0.1

        # 平均每类图像数量权重
        if avg_images_per_class > 0:
            confidence += 0.1
            if avg_images_per_class >= 10:
                confidence += 0.1

        # 目录结构权重
        if structure_info.get("has_image_directories", False):
            confidence += 0.2

        # 类别平衡权重
        if num_classes > 0:
            balance_score = 1.0  # 简化处理，实际可以计算标准差
            confidence += balance_score * 0.1

        return min(confidence, 1.0)