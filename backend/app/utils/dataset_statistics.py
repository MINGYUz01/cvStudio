"""
数据集统计分析工具
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class DatasetStatistics:
    """数据集统计分析类"""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def analyze_dataset(self, dataset_format: str = "unknown") -> Dict[str, Any]:
        """
        分析数据集并返回统计信息

        Args:
            dataset_format: 数据集格式

        Returns:
            Dict: 完整的统计信息
        """
        stats = {
            "dataset_info": {
                "path": str(self.dataset_path),
                "format": dataset_format,
                "exists": self.dataset_path.exists()
            },
            "image_statistics": self._analyze_images(),
            "quality_metrics": {},
            "format_specific_stats": {}
        }

        if dataset_format == "yolo":
            stats["format_specific_stats"] = self._analyze_yolo_dataset()
        elif dataset_format == "coco":
            stats["format_specific_stats"] = self._analyze_coco_dataset()
        elif dataset_format == "voc":
            stats["format_specific_stats"] = self._analyze_voc_dataset()
        elif dataset_format == "classification":
            stats["format_specific_stats"] = self._analyze_classification_dataset()

        # 计算质量指标
        stats["quality_metrics"] = self._calculate_quality_metrics(stats)

        return stats

    def _analyze_images(self) -> Dict[str, Any]:
        """分析图像文件"""
        if not PIL_AVAILABLE:
            return {"error": "需要安装Pillow库来分析图像"}

        image_stats = {
            "total_images": 0,
            "total_size_mb": 0,
            "size_distribution": {},
            "aspect_ratio_distribution": {},
            "format_distribution": {},
            "resolution_stats": {},
            "valid_images": 0,
            "corrupted_images": 0
        }

        try:
            sizes = []
            aspect_ratios = []
            formats = []
            file_sizes = []

            # 查找所有图像文件
            all_images = []
            for ext in self.image_extensions:
                all_images.extend(self.dataset_path.rglob(f"*{ext}"))

            image_stats["total_images"] = len(all_images)

            for img_path in all_images:
                try:
                    # 文件大小
                    file_size = img_path.stat().st_size / (1024 * 1024)  # MB
                    file_sizes.append(file_size)

                    # 图像格式
                    with Image.open(img_path) as img:
                        width, height = img.size
                        sizes.append((width, height))

                        # 计算宽高比
                        aspect_ratio = width / height
                        aspect_ratios.append(aspect_ratio)

                        # 图像格式
                        img_format = img.format or "unknown"
                        formats.append(img_format)

                    image_stats["valid_images"] += 1

                except Exception:
                    image_stats["corrupted_images"] += 1
                    continue

            # 计算统计信息
            if sizes:
                widths = [w for w, h in sizes]
                heights = [h for w, h in sizes]
                areas = [w * h for w, h in sizes]

                image_stats["resolution_stats"] = {
                    "width": {
                        "min": min(widths),
                        "max": max(widths),
                        "mean": np.mean(widths),
                        "median": np.median(widths),
                        "std": np.std(widths)
                    },
                    "height": {
                        "min": min(heights),
                        "max": max(heights),
                        "mean": np.mean(heights),
                        "median": np.median(heights),
                        "std": np.std(heights)
                    },
                    "area": {
                        "min": min(areas),
                        "max": max(areas),
                        "mean": np.mean(areas),
                        "median": np.median(areas),
                        "std": np.std(areas)
                    }
                }

                # 尺寸分布
                size_ranges = {
                    "small (<224x224)": 0,
                    "medium (224x224-512x512)": 0,
                    "large (512x512-1024x1024)": 0,
                    "xlarge (>1024x1024)": 0
                }

                for w, h in sizes:
                    area = w * h
                    if area < 224 * 224:
                        size_ranges["small (<224x224)"] += 1
                    elif area < 512 * 512:
                        size_ranges["medium (224x224-512x512)"] += 1
                    elif area < 1024 * 1024:
                        size_ranges["large (512x512-1024x1024)"] += 1
                    else:
                        size_ranges["xlarge (>1024x1024)"] += 1

                image_stats["size_distribution"] = size_ranges

                # 宽高比分布
                ratio_ranges = {
                    "portrait (<0.9)": 0,
                    "square (0.9-1.1)": 0,
                    "landscape (>1.1)": 0
                }

                for ratio in aspect_ratios:
                    if ratio < 0.9:
                        ratio_ranges["portrait (<0.9)"] += 1
                    elif ratio <= 1.1:
                        ratio_ranges["square (0.9-1.1)"] += 1
                    else:
                        ratio_ranges["landscape (>1.1)"] += 1

                image_stats["aspect_ratio_distribution"] = ratio_ranges

            # 格式分布
            format_counts = Counter(formats)
            image_stats["format_distribution"] = dict(format_counts)

            # 文件大小统计
            if file_sizes:
                image_stats["total_size_mb"] = sum(file_sizes)
                image_stats["file_size_stats"] = {
                    "min_mb": min(file_sizes),
                    "max_mb": max(file_sizes),
                    "mean_mb": np.mean(file_sizes),
                    "median_mb": np.median(file_sizes)
                }

        except Exception as e:
            image_stats["error"] = str(e)

        return image_stats

    def _analyze_yolo_dataset(self) -> Dict[str, Any]:
        """分析YOLO格式数据集"""
        yolo_stats = {
            "annotation_files": 0,
            "class_distribution": {},
            "bbox_statistics": {},
            "objects_per_image": [],
            "avg_objects_per_image": 0,
            "classes_file": None,
            "data_config": {}
        }

        try:
            # 查找classes文件
            classes_files = []
            for name in ["obj.names", "classes.txt", "names.txt"]:
                for file_path in self.dataset_path.rglob(name):
                    if file_path.is_file():
                        classes_files.append(file_path)

            if classes_files:
                classes_file = classes_files[0]
                yolo_stats["classes_file"] = str(classes_file)
                classes = []
                try:
                    with open(classes_file, 'r', encoding='utf-8') as f:
                        classes = [line.strip() for line in f.readlines() if line.strip()]
                except Exception:
                    pass

            # 分析标注文件
            annotation_files = []
            objects_per_image = []
            all_bboxes = []
            class_counts = defaultdict(int)

            for txt_file in self.dataset_path.rglob("*.txt"):
                if txt_file.name in ["obj.names", "classes.txt", "names.txt"]:
                    continue

                annotation_files.append(txt_file)
                yolo_stats["annotation_files"] += 1

                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    image_objects = 0
                    for line in lines:
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                class_counts[class_id] += 1

                                # YOLO格式: class_id x_center y_center width height
                                x_center, y_center, width, height = map(float, parts[1:5])
                                all_bboxes.append({
                                    "width": width,
                                    "height": height,
                                    "area": width * height
                                })

                                image_objects += 1

                    if image_objects > 0:
                        objects_per_image.append(image_objects)

                except Exception:
                    continue

            # 统计信息
            yolo_stats["objects_per_image"] = objects_per_image
            if objects_per_image:
                yolo_stats["avg_objects_per_image"] = np.mean(objects_per_image)
                yolo_stats["max_objects_per_image"] = max(objects_per_image)
                yolo_stats["min_objects_per_image"] = min(objects_per_image)

            # 类别分布
            yolo_stats["class_distribution"] = dict(class_counts)

            # 边界框统计
            if all_bboxes:
                widths = [bbox["width"] for bbox in all_bboxes]
                heights = [bbox["height"] for bbox in all_bboxes]
                areas = [bbox["area"] for bbox in all_bboxes]

                yolo_stats["bbox_statistics"] = {
                    "width": {"min": min(widths), "max": max(widths), "mean": np.mean(widths)},
                    "height": {"min": min(heights), "max": max(heights), "mean": np.mean(heights)},
                    "area": {"min": min(areas), "max": max(areas), "mean": np.mean(areas)}
                }

        except Exception as e:
            yolo_stats["error"] = str(e)

        return yolo_stats

    def _analyze_coco_dataset(self) -> Dict[str, Any]:
        """分析COCO格式数据集"""
        coco_stats = {
            "annotation_files": [],
            "category_distribution": {},
            "bbox_statistics": {},
            "images_info": {},
            "annotations_info": {}
        }

        try:
            # 查找JSON标注文件
            json_files = list(self.dataset_path.rglob("*.json"))
            coco_stats["annotation_files"] = [str(f) for f in json_files]

            for json_file in json_files[:2]:  # 最多分析2个JSON文件
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 基本信息
                    if "images" in data:
                        coco_stats["images_info"][json_file.name] = {
                            "count": len(data["images"]),
                            "sample": data["images"][:3] if len(data["images"]) > 3 else data["images"]
                        }

                    if "annotations" in data:
                        coco_stats["annotations_info"][json_file.name] = {
                            "count": len(data["annotations"]),
                            "sample": data["annotations"][:3] if len(data["annotations"]) > 3 else data["annotations"]
                        }

                        # 分析边界框
                        bboxes = []
                        category_counts = defaultdict(int)

                        for annotation in data["annotations"]:
                            if "category_id" in annotation:
                                category_counts[annotation["category_id"]] += 1

                            if "bbox" in annotation:
                                bbox = annotation["bbox"]
                                if len(bbox) >= 4:
                                    x, y, w, h = bbox[:4]
                                    bboxes.append({
                                        "width": w,
                                        "height": h,
                                        "area": w * h
                                    })

                        coco_stats["category_distribution"] = dict(category_counts)

                        if bboxes:
                            widths = [bbox["width"] for bbox in bboxes]
                            heights = [bbox["height"] for bbox in bboxes]
                            areas = [bbox["area"] for bbox in bboxes]

                            coco_stats["bbox_statistics"] = {
                                "width": {"min": min(widths), "max": max(widths), "mean": np.mean(widths)},
                                "height": {"min": min(heights), "max": max(heights), "mean": np.mean(heights)},
                                "area": {"min": min(areas), "max": max(areas), "mean": np.mean(areas)}
                            }

                except Exception as e:
                    coco_stats["error"] = f"解析JSON文件失败: {str(e)}"
                    continue

        except Exception as e:
            coco_stats["error"] = str(e)

        return coco_stats

    def _analyze_voc_dataset(self) -> Dict[str, Any]:
        """分析VOC格式数据集"""
        voc_stats = {
            "xml_files": 0,
            "class_distribution": {},
            "bbox_statistics": {},
            "objects_per_image": [],
            "avg_objects_per_image": 0,
            "directory_structure": {}
        }

        try:
            # 检查目录结构
            for dir_name in ["Annotations", "JPEGImages", "ImageSets"]:
                dir_path = self.dataset_path / dir_name
                voc_stats["directory_structure"][dir_name] = {
                    "exists": dir_path.exists(),
                    "file_count": len(list(dir_path.iterdir())) if dir_path.exists() else 0
                }

            # 分析XML标注文件
            xml_files = list(self.dataset_path.rglob("*.xml"))
            voc_stats["xml_files"] = len(xml_files)

            objects_per_image = []
            class_counts = defaultdict(int)
            all_bboxes = []

            for xml_file in xml_files:
                try:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()

                    objects = root.findall("object")
                    image_objects = len(objects)
                    objects_per_image.append(image_objects)

                    for obj in objects:
                        name_elem = obj.find("name")
                        if name_elem is not None and name_elem.text:
                            class_counts[name_elem.text] += 1

                        bndbox = obj.find("bndbox")
                        if bndbox is not None:
                            xmin = float(bndbox.findtext("xmin", 0))
                            ymin = float(bndbox.findtext("ymin", 0))
                            xmax = float(bndbox.findtext("xmax", 0))
                            ymax = float(bndbox.findtext("ymax", 0))

                            width = xmax - xmin
                            height = ymax - ymin
                            area = width * height

                            all_bboxes.append({
                                "width": width,
                                "height": height,
                                "area": area
                            })

                except Exception:
                    continue

            # 统计信息
            voc_stats["objects_per_image"] = objects_per_image
            if objects_per_image:
                voc_stats["avg_objects_per_image"] = np.mean(objects_per_image)
                voc_stats["max_objects_per_image"] = max(objects_per_image)
                voc_stats["min_objects_per_image"] = min(objects_per_image)

            # 类别分布
            voc_stats["class_distribution"] = dict(class_counts)

            # 边界框统计
            if all_bboxes:
                widths = [bbox["width"] for bbox in all_bboxes]
                heights = [bbox["height"] for bbox in all_bboxes]
                areas = [bbox["area"] for bbox in all_bboxes]

                voc_stats["bbox_statistics"] = {
                    "width": {"min": min(widths), "max": max(widths), "mean": np.mean(widths)},
                    "height": {"min": min(heights), "max": max(heights), "mean": np.mean(heights)},
                    "area": {"min": min(areas), "max": max(areas), "mean": np.mean(areas)}
                }

        except Exception as e:
            voc_stats["error"] = str(e)

        return voc_stats

    def _analyze_classification_dataset(self) -> Dict[str, Any]:
        """分析分类数据集"""
        class_stats = {
            "class_directories": {},
            "num_classes": 0,
            "total_images": 0,
            "class_distribution": {},
            "imbalance_ratio": 0,
            "balance_score": 0
        }

        try:
            class_counts = {}
            total_images = 0

            for item in self.dataset_path.iterdir():
                if item.is_dir():
                    # 统计该类别的图像数量
                    image_count = 0
                    for ext in self.image_extensions:
                        images = list(item.glob(f"*{ext}"))
                        image_count += len(images)

                    if image_count > 0:
                        class_name = item.name
                        class_counts[class_name] = image_count
                        total_images += image_count

                        class_stats["class_directories"][class_name] = {
                            "path": str(item),
                            "image_count": image_count,
                            "sample_files": [img.name for img in (item.glob("*") if item.is_dir() else [])][:5]
                        }

            # 计算统计指标
            class_stats["num_classes"] = len(class_counts)
            class_stats["total_images"] = total_images
            class_stats["class_distribution"] = class_counts

            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                avg_count = total_images / len(class_counts)

                class_stats["imbalance_ratio"] = max_count / min_count if min_count > 0 else float('inf')

                # 平衡分数（1表示完全平衡）
                variance = sum((count - avg_count) ** 2 for count in class_counts.values()) / len(class_counts)
                std_dev = variance ** 0.5
                class_stats["balance_score"] = max(0, 1 - (std_dev / avg_count)) if avg_count > 0 else 0

        except Exception as e:
            class_stats["error"] = str(e)

        return class_stats

    def _calculate_quality_metrics(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """计算数据集质量指标"""
        quality_metrics = {
            "image_quality_score": 0,
            "annotation_quality_score": 0,
            "balance_quality_score": 0,
            "overall_quality_score": 0,
            "recommendations": []
        }

        try:
            image_stats = stats.get("image_statistics", {})
            format_stats = stats.get("format_specific_stats", {})

            # 图像质量评分 (0-100)
            image_score = 0
            if image_stats.get("total_images", 0) > 0:
                # 有效图像比例
                valid_ratio = image_stats.get("valid_images", 0) / image_stats.get("total_images", 1)
                image_score += valid_ratio * 40

                # 分辨率多样性（不要太单一也不要太分散）
                resolution_stats = image_stats.get("resolution_stats", {})
                if resolution_stats and "width" in resolution_stats:
                    width_cv = resolution_stats["width"]["std"] / resolution_stats["width"]["mean"] if resolution_stats["width"]["mean"] > 0 else 0
                    if width_cv < 0.5:  # 适中的分辨率变化
                        image_score += 20

                # 文件大小合理性
                if "file_size_stats" in image_stats:
                    file_stats = image_stats["file_size_stats"]
                    mean_size = file_stats["mean_mb"]
                    if 0.1 <= mean_size <= 5:  # 合理的文件大小范围
                        image_score += 20

                # 格式统一性
                format_dist = image_stats.get("format_distribution", {})
                if len(format_dist) <= 3:  # 格式不要太多样
                    image_score += 20

            quality_metrics["image_quality_score"] = min(100, image_score)

            # 标注质量评分
            annotation_score = 0
            if stats["dataset_info"]["format"] in ["yolo", "coco", "voc"]:
                if format_stats.get("annotation_files", 0) > 0:
                    annotation_score += 30

                # 类别分布合理性
                class_dist = format_stats.get("class_distribution", {})
                if class_dist:
                    annotation_score += 20

                # 每张图像的平均标注数
                avg_objects = format_stats.get("avg_objects_per_image", 0)
                if 1 <= avg_objects <= 10:  # 合理的标注密度
                    annotation_score += 25

                # 边界框质量
                bbox_stats = format_stats.get("bbox_statistics", {})
                if bbox_stats and "width" in bbox_stats:
                    annotation_score += 25

            elif stats["dataset_info"]["format"] == "classification":
                class_stats = format_stats
                if class_stats.get("num_classes", 0) >= 2:
                    annotation_score += 40

                balance_score = class_stats.get("balance_score", 0)
                annotation_score += balance_score * 60

            quality_metrics["annotation_quality_score"] = min(100, annotation_score)

            # 平衡质量评分
            balance_score = 0
            if stats["dataset_info"]["format"] == "classification":
                balance_score = format_stats.get("balance_score", 0) * 100
            else:
                class_dist = format_stats.get("class_distribution", {})
                if class_dist:
                    values = list(class_dist.values())
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        cv = std_val / mean_val if mean_val > 0 else 0
                        balance_score = max(0, 100 - cv * 50)

            quality_metrics["balance_quality_score"] = balance_score

            # 总体质量评分
            quality_metrics["overall_quality_score"] = (
                quality_metrics["image_quality_score"] * 0.3 +
                quality_metrics["annotation_quality_score"] * 0.5 +
                quality_metrics["balance_quality_score"] * 0.2
            )

            # 生成建议
            recommendations = []
            if quality_metrics["image_quality_score"] < 70:
                recommendations.append("建议检查图像质量和完整性")

            if quality_metrics["annotation_quality_score"] < 70:
                recommendations.append("建议检查标注文件的完整性和质量")

            if quality_metrics["balance_quality_score"] < 60:
                recommendations.append("数据集类别不平衡，建议进行数据平衡处理")

            if image_stats.get("total_images", 0) < 100:
                recommendations.append("数据集规模较小，建议收集更多数据")

            quality_metrics["recommendations"] = recommendations

        except Exception as e:
            quality_metrics["error"] = str(e)

        return quality_metrics