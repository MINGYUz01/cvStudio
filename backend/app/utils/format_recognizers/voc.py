"""
VOC格式数据集识别器
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class VOCRecognizer:
    """VOC格式数据集识别器"""

    def __init__(self):
        self.required_dirs = ['Annotations', 'JPEGImages', 'ImageSets']
        self.required_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.required_annotation_extensions = {'.xml'}

    def recognize(self, dataset_path: str) -> Dict:
        """
        识别VOC格式数据集

        Args:
            dataset_path: 数据集路径

        Returns:
            Dict: 识别结果
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return {"format": "unknown", "error": "数据集路径不存在"}

        result = {
            "format": "voc",
            "confidence": 0,
            "details": {},
            "error": None
        }

        try:
            # 检查VOC目录结构
            voc_structure = self._check_voc_structure(dataset_path)
            result["details"]["structure"] = voc_structure

            # 查找XML标注文件
            annotations_info = self._find_annotations(dataset_path)
            result["details"].update(annotations_info)

            # 查找图像文件
            images_info = self._find_images(dataset_path)
            result["details"].update(images_info)

            # 解析XML标注
            xml_stats = self._parse_xml_annotations(annotations_info.get("annotation_files", []))
            result["details"]["xml_stats"] = xml_stats

            # 检查ImageSets目录
            imagesets_info = self._check_imagesets(dataset_path)
            result["details"]["imagesets"] = imagesets_info

            # 计算置信度
            confidence = self._calculate_confidence(
                voc_structure,
                annotations_info.get("num_annotations", 0),
                images_info.get("num_images", 0),
                imagesets_info.get("has_imagesets", False)
            )
            result["confidence"] = confidence

            # 如果置信度太低，可能是其他格式
            if confidence < 0.3:
                result["format"] = "unknown"
                result["error"] = "未找到足够的VOC格式特征"

        except Exception as e:
            result["error"] = f"识别VOC格式时出错: {str(e)}"
            result["confidence"] = 0

        return result

    def _check_voc_structure(self, dataset_path: Path) -> Dict:
        """检查VOC目录结构"""
        structure = {
            "has_annotations": False,
            "has_jpegimages": False,
            "has_imagesets": False,
            "annotations_dir": None,
            "jpegimages_dir": None,
            "imagesets_dir": None,
            "structure_score": 0
        }

        # 检查主要目录
        for dir_name in ["Annotations", "JPEGImages", "ImageSets", "images", "labels"]:
            dir_path = dataset_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                if dir_name.lower() in ["annotations", "labels"]:
                    structure["has_annotations"] = True
                    structure["annotations_dir"] = str(dir_path)
                    structure["structure_score"] += 1
                elif dir_name.lower() in ["jpegimages", "images"]:
                    structure["has_jpegimages"] = True
                    structure["jpegimages_dir"] = str(dir_path)
                    structure["structure_score"] += 1
                elif dir_name.lower() == "imagesets":
                    structure["has_imagesets"] = True
                    structure["imagesets_dir"] = str(dir_path)
                    structure["structure_score"] += 1

        return structure

    def _find_annotations(self, dataset_path: Path) -> Dict:
        """查找XML标注文件"""
        annotation_files = []

        # 在Annotations目录中查找
        annotations_dir = dataset_path / "Annotations"
        if annotations_dir.exists():
            for xml_file in annotations_dir.glob("*.xml"):
                annotation_files.append(xml_file)

        # 在其他可能的目录中查找
        for dir_name in ["labels", "annotations"]:
            dir_path = dataset_path / dir_name
            if dir_path.exists() and dir_path != annotations_dir:
                for xml_file in dir_path.rglob("*.xml"):
                    annotation_files.append(xml_file)

        # 递归查找所有XML文件
        for xml_file in dataset_path.rglob("*.xml"):
            if xml_file not in annotation_files:
                annotation_files.append(xml_file)

        num_annotations = len(annotation_files)

        # 设置较大的限制，避免内存问题
        max_paths = 10000
        return {
            "num_annotations": num_annotations,
            "annotation_files": [str(f) for f in annotation_files[:max_paths]],
            "annotation_paths": [str(f) for f in annotation_files[:20]]  # 前20个用于解析
        }

    def _find_images(self, dataset_path: Path) -> Dict:
        """查找图像文件"""
        image_files = []

        # 在JPEGImages目录中查找
        jpegimages_dir = dataset_path / "JPEGImages"
        if jpegimages_dir.exists():
            for img_file in jpegimages_dir.rglob("*"):
                if img_file.is_file() and img_file.suffix.lower() in self.required_image_extensions:
                    image_files.append(img_file)

        # 在其他可能的目录中查找
        for dir_name in ["images", "ImageSets", "SegmentationObject"]:
            dir_path = dataset_path / dir_name
            if dir_path.exists() and dir_path != jpegimages_dir:
                for img_file in dir_path.rglob("*"):
                    if img_file.is_file() and img_file.suffix.lower() in self.required_image_extensions:
                        image_files.append(img_file)

        # 递归查找所有图像文件
        for img_file in dataset_path.rglob("*"):
            if (img_file.is_file() and
                img_file.suffix.lower() in self.required_image_extensions and
                img_file not in image_files):
                image_files.append(img_file)

        num_images = len(image_files)

        # 分析图像尺寸（抽样）
        image_stats = self._analyze_images(image_files[:100])  # 最多分析100张图

        # 设置较大的限制，避免内存问题
        max_paths = 10000
        return {
            "num_images": num_images,
            "image_files": [str(f) for f in image_files[:max_paths]],
            "image_paths": [str(f) for f in image_files[:max_paths]],  # 添加image_paths以保持一致性
            "image_stats": image_stats
        }

    def _parse_xml_annotations(self, annotation_files: List[str]) -> Dict:
        """解析XML标注文件"""
        if not annotation_files:
            return {"error": "没有找到XML标注文件"}

        stats = {
            "total_parsed": 0,
            "valid_xml": 0,
            "classes": {},
            "size_distribution": {},
            "object_counts": [],
            "sample_xml": None
        }

        for xml_path in annotation_files[:20]:  # 最多解析20个文件
            try:
                xml_file = Path(xml_path)
                if not xml_file.exists():
                    continue

                tree = ET.parse(xml_file)
                root = tree.getroot()
                stats["total_parsed"] += 1

                # 提取基本信息
                size_elem = root.find("size")
                if size_elem is not None:
                    width = int(size_elem.find("width").text) if size_elem.find("width") is not None else 0
                    height = int(size_elem.find("height").text) if size_elem.find("height") is not None else 0
                    size_key = f"{width}x{height}"
                    stats["size_distribution"][size_key] = stats["size_distribution"].get(size_key, 0) + 1

                # 提取对象信息
                objects = root.findall("object")
                object_count = len(objects)
                stats["object_counts"].append(object_count)

                for obj in objects:
                    name_elem = obj.find("name")
                    if name_elem is not None and name_elem.text:
                        class_name = name_elem.text
                        stats["classes"][class_name] = stats["classes"].get(class_name, 0) + 1

                stats["valid_xml"] += 1

                # 保存第一个XML作为示例
                if stats["sample_xml"] is None:
                    stats["sample_xml"] = {
                        "filename": root.findtext("filename", ""),
                        "size": {
                            "width": width if size_elem is not None else 0,
                            "height": height if size_elem is not None else 0,
                            "depth": int(size_elem.findtext("depth", "0")) if size_elem is not None else 0
                        },
                        "object_count": object_count
                    }

            except Exception as e:
                print(f"解析XML文件失败 {xml_path}: {e}")
                continue

        # 计算统计信息
        if stats["object_counts"]:
            stats["avg_objects_per_image"] = sum(stats["object_counts"]) / len(stats["object_counts"])
            stats["max_objects_per_image"] = max(stats["object_counts"])
            stats["min_objects_per_image"] = min(stats["object_counts"])
        else:
            stats["avg_objects_per_image"] = 0
            stats["max_objects_per_image"] = 0
            stats["min_objects_per_image"] = 0

        return stats

    def _check_imagesets(self, dataset_path: Path) -> Dict:
        """检查ImageSets目录"""
        imagesets_info = {
            "has_imagesets": False,
            "imagesets_dir": None,
            "splits": {},
            "total_split_files": 0
        }

        # 查找ImageSets目录
        imagesets_dir = dataset_path / "ImageSets"
        if not imagesets_dir.exists():
            return imagesets_info

        imagesets_info["has_imagesets"] = True
        imagesets_info["imagesets_dir"] = str(imagesets_dir)

        # 查找分割文件
        split_types = ["Main", "Layout", "Segmentation"]

        for split_type in split_types:
            split_dir = imagesets_dir / split_type
            if split_dir.exists():
                split_files = list(split_dir.glob("*.txt"))
                imagesets_info["splits"][split_type] = {
                    "dir": str(split_dir),
                    "files": [f.name for f in split_files],
                    "count": len(split_files)
                }
                imagesets_info["total_split_files"] += len(split_files)

        return imagesets_info

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

    def _calculate_confidence(self, voc_structure: Dict, num_annotations: int,
                            num_images: int, has_imagesets: bool) -> float:
        """计算VOC格式置信度"""
        confidence = 0.0

        # VOC格式的核心特征是XML标注文件，没有XML就不应该是VOC格式
        if num_annotations == 0:
            # 没有XML标注文件，置信度设为极低
            return 0.1  # 只给一点点基础分，表示可能是错误识别

        # XML标注文件权重（核心特征）
        confidence += 0.5  # 有XML文件就给很高权重
        # 标注数量加分
        if num_annotations >= 10:
            confidence += 0.1
        if num_annotations >= 50:
            confidence += 0.1

        # 目录结构权重
        structure_score = voc_structure.get("structure_score", 0)
        confidence += structure_score * 0.1  # 降低目录结构权重

        # 图像文件权重（只有在有XML的情况下才加分）
        if num_images > 0:
            confidence += 0.1
            # 图像数量加分
            if num_images >= 10:
                confidence += 0.05

        # ImageSets权重
        if has_imagesets:
            confidence += 0.1

        # 文件比例权重（XML和图像的比例）
        if num_annotations > 0 and num_images > 0:
            ratio = min(num_annotations, num_images) / max(num_annotations, num_images)
            if ratio > 0.5:  # 标注和图像数量比较接近
                confidence += 0.1

        return min(confidence, 1.0)