"""
COCO格式数据集识别器
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from app.utils.path_mapper import PathMapper


class COCORecognizer:
    """COCO格式数据集识别器"""

    def __init__(self):
        self.required_files = ['annotations.json', 'instances_train.json', 'instances_val.json']
        self.required_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    def recognize(self, dataset_path: str) -> Dict:
        """
        识别COCO格式数据集

        Args:
            dataset_path: 数据集路径

        Returns:
            Dict: 识别结果
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            return {"format": "unknown", "error": "数据集路径不存在"}

        result = {
            "format": "coco",
            "confidence": 0,
            "details": {},
            "error": None
        }

        try:
            # 查找COCO标注文件
            annotation_files = self._find_annotation_files(dataset_path)

            if annotation_files:
                # 解析第一个找到的标注文件
                main_annotation = annotation_files[0]
                coco_data = self._parse_coco_annotation(main_annotation)

                if coco_data:
                    result["details"]["coco_data"] = coco_data
                    result["details"]["annotation_file"] = str(main_annotation)

            # 查找图像文件
            images_info = self._find_images(dataset_path)
            result["details"].update(images_info)

            # 验证COCO格式结构
            coco_validation = self._validate_coco_structure(coco_data if annotation_files else {})
            result["details"]["validation"] = coco_validation

            # 计算置信度
            confidence = self._calculate_confidence(
                len(annotation_files),
                coco_validation.get("is_valid_coco", False),
                images_info.get("num_images", 0)
            )
            result["confidence"] = confidence

            # 如果置信度太低，可能是其他格式
            if confidence < 0.3:
                result["format"] = "unknown"
                result["error"] = "未找到足够的COCO格式特征"

            # 构建路径映射（COCO格式特殊处理）
            annotation_file = annotation_files[0] if annotation_files else None
            images_dir = None

            # 从image_directories中获取第一个
            if result["details"].get("image_directories"):
                images_dir = Path(result["details"]["image_directories"][0])

            path_mapping = {}
            if annotation_file:
                path_mapping = PathMapper.build_path_mapping(
                    dataset_path,
                    images_dir if images_dir else dataset_path,
                    annotation_file=annotation_file
                )
                path_mapping["label_format"] = "json"
                path_mapping["needs_conversion"] = True  # 标记需要转换为txt格式

                # 从COCO数据中提取对应关系
                if coco_data:
                    image_ids = {img["id"]: img.get("file_name", "") for img in coco_data.get("images", [])}
                    annotation_image_ids = {ann["image_id"] for ann in coco_data.get("annotations", [])}
                    missing_annotations = len(image_ids) - len(annotation_image_ids)

                    path_mapping["file_pairs"] = len(annotation_image_ids)
                    path_mapping["missing_labels"] = missing_annotations

            result["details"]["path_mapping"] = path_mapping

        except Exception as e:
            result["error"] = f"识别COCO格式时出错: {str(e)}"
            result["confidence"] = 0

        return result

    def _find_annotation_files(self, dataset_path: Path) -> List[Path]:
        """查找COCO标注文件"""
        annotation_files = []

        # 常见的COCO标注文件名
        possible_names = [
            'annotations.json',
            'instances_train.json',
            'instances_val.json',
            'instances_train2017.json',
            'instances_val2017.json',
            'person_keypoints_train2017.json',
            'person_keypoints_val2017.json',
            'captions_train2017.json',
            'captions_val2017.json'
        ]

        for name in possible_names:
            # 在根目录查找
            file_path = dataset_path / name
            if file_path.exists():
                annotation_files.append(file_path)

            # 递归查找
            for file_path in dataset_path.rglob(name):
                if file_path.is_file():
                    annotation_files.append(file_path)

        # 查找annotations目录
        annotations_dir = dataset_path / 'annotations'
        if annotations_dir.exists() and annotations_dir.is_dir():
            for json_file in annotations_dir.glob('*.json'):
                annotation_files.append(json_file)

        return list(set(annotation_files))  # 去重

    def _parse_coco_annotation(self, annotation_file: Path) -> Optional[Dict]:
        """解析COCO标注文件"""
        try:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检查基本的COCO结构
            required_keys = ['images', 'annotations', 'categories']
            if not all(key in data for key in required_keys):
                return None

            # 提取关键信息
            coco_info = {
                "info": data.get("info", {}),
                "licenses": data.get("licenses", []),
                "num_images": len(data.get("images", [])),
                "num_annotations": len(data.get("annotations", [])),
                "num_categories": len(data.get("categories", [])),
                "categories": self._extract_categories(data.get("categories", [])),
                "images_sample": data.get("images", [])[:10],  # 前10张图像信息
                "annotations_sample": data.get("annotations", [])[:10]  # 前10个标注
            }

            # 统计每个类别的标注数量
            category_counts = {}
            for annotation in data.get("annotations", []):
                category_id = annotation.get("category_id")
                if category_id is not None:
                    category_counts[category_id] = category_counts.get(category_id, 0) + 1

            coco_info["category_counts"] = category_counts

            return coco_info

        except Exception as e:
            print(f"解析COCO标注文件失败: {e}")
            return None

    def _extract_categories(self, categories: List[Dict]) -> Dict:
        """提取类别信息"""
        category_dict = {}
        for category in categories:
            if "id" in category and "name" in category:
                category_dict[category["id"]] = {
                    "id": category["id"],
                    "name": category["name"],
                    "supercategory": category.get("supercategory", "")
                }
        return category_dict

    def _find_images(self, dataset_path: Path) -> Dict:
        """查找图像文件"""
        images = []
        image_dirs = []

        # 常见的图像目录名
        possible_image_dirs = [
            'images',
            'train2017',
            'val2017',
            'train2014',
            'val2014',
            'test2017'
        ]

        # 查找可能的图像目录
        for dir_name in possible_image_dirs:
            dir_path = dataset_path / dir_name
            if dir_path.exists() and dir_path.is_dir():
                image_dirs.append(dir_path)

        # 递归查找所有图像文件
        for file_path in dataset_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.required_image_extensions:
                # 排除标注文件中的图像
                if not any(part in str(file_path).lower() for part in ['annotation', 'label']):
                    images.append(file_path)

        # 统计信息
        num_images = len(images)

        # 分析图像尺寸（抽样）
        image_stats = self._analyze_images(images[:100])  # 最多分析100张图

        # 设置较大的限制，避免内存问题
        max_paths = 10000
        return {
            "num_images": num_images,
            "image_paths": [str(img) for img in images[:max_paths]],
            "image_directories": [str(d) for d in image_dirs],
            "image_stats": image_stats
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

    def _validate_coco_structure(self, coco_data: Dict) -> Dict:
        """验证COCO格式结构"""
        if not coco_data:
            return {"is_valid_coco": False, "errors": ["空的COCO数据"]}

        errors = []
        warnings = []

        # 检查必要字段
        required_fields = ['num_images', 'num_annotations', 'num_categories']
        for field in required_fields:
            if field not in coco_data:
                errors.append(f"缺少必要字段: {field}")

        # 检查数据合理性
        if coco_data.get("num_images", 0) == 0:
            errors.append("没有图像数据")

        if coco_data.get("num_annotations", 0) == 0:
            warnings.append("没有标注数据")

        if coco_data.get("num_categories", 0) == 0:
            errors.append("没有类别数据")

        # 检查图像和标注的比例
        num_images = coco_data.get("num_images", 0)
        num_annotations = coco_data.get("num_annotations", 0)

        if num_images > 0 and num_annotations > 0:
            ratio = num_annotations / num_images
            if ratio < 0.1:
                warnings.append("标注数量相对于图像数量较少")
            elif ratio > 100:
                warnings.append("标注数量相对于图像数量较多")

        return {
            "is_valid_coco": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "summary": {
                "images": coco_data.get("num_images", 0),
                "annotations": coco_data.get("num_annotations", 0),
                "categories": coco_data.get("num_categories", 0)
            }
        }

    def _calculate_confidence(self, num_annotation_files: int, is_valid_coco: bool, num_images: int) -> float:
        """计算COCO格式置信度"""
        confidence = 0.0

        # 标注文件数量权重
        if num_annotation_files > 0:
            confidence += 0.4
            if num_annotation_files >= 2:
                confidence += 0.1

        # COCO结构验证权重
        if is_valid_coco:
            confidence += 0.4

        # 图像文件权重
        if num_images > 0:
            confidence += 0.2
            # 图像数量加分
            if num_images >= 10:
                confidence += 0.1

        return min(confidence, 1.0)