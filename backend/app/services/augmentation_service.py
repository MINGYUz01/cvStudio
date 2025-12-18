"""
数据增强服务模块
提供数据集图像的数据增强功能
"""

import os
import io
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import uuid
import tempfile
from datetime import datetime
from loguru import logger

from app.utils.image_processor import image_processor
from app.utils.format_recognizers import DatasetFormatRecognizer


class AugmentationService:
    """数据增强服务"""

    def __init__(self):
        """初始化数据增强服务"""
        self.format_recognizer = DatasetFormatRecognizer()
        self.temp_dir = Path(tempfile.gettempdir()) / "cvstudio_augmentation"
        self.temp_dir.mkdir(exist_ok=True)

    async def get_dataset_images(self, dataset_path: str, page: int = 1,
                                page_size: int = 20, sort_by: str = "filename",
                                sort_order: str = "asc") -> Dict[str, Any]:
        """
        获取数据集图像列表（支持分页）

        Args:
            dataset_path: 数据集路径
            page: 页码
            page_size: 每页大小
            sort_by: 排序字段
            sort_order: 排序顺序

        Returns:
            图像列表信息
        """
        try:
            # 识别数据集格式
            format_result = self.format_recognizer.recognize_format(dataset_path)
            best_format = format_result.get('best_format', {}).get('format', 'unknown')

            # 根据格式获取图像列表
            image_files = self._get_image_files_by_format(dataset_path, best_format)

            # 排序
            if sort_by == "filename":
                image_files.sort(key=lambda x: Path(x).name, reverse=(sort_order == "desc"))
            elif sort_by == "size":
                image_files.sort(key=lambda x: Path(x).stat().st_size, reverse=(sort_order == "desc"))
            elif sort_by == "date":
                image_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=(sort_order == "desc"))

            # 分页
            total = len(image_files)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_files = image_files[start_idx:end_idx]

            # 获取图像信息
            images = []
            for file_path in page_files:
                info = image_processor.get_image_info(file_path)
                if info:
                    # 尝试获取标注信息
                    annotations = self._get_image_annotations(file_path, dataset_path, best_format)
                    info['annotations'] = annotations
                    images.append(info)

            total_pages = (total + page_size - 1) // page_size

            return {
                'images': images,
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': total_pages
            }
        except Exception as e:
            logger.error(f"获取数据集图像列表失败: {str(e)}")
            return {
                'images': [],
                'total': 0,
                'page': page,
                'page_size': page_size,
                'total_pages': 0
            }

    def _get_image_files_by_format(self, dataset_path: str, format_type: str) -> List[str]:
        """
        根据数据集格式获取图像文件列表

        Args:
            dataset_path: 数据集路径
            format_type: 格式类型

        Returns:
            图像文件路径列表
        """
        image_files = []
        dataset_path = Path(dataset_path)

        try:
            if format_type == "classification":
                # 分类格式：从子目录获取图像
                for class_dir in dataset_path.iterdir():
                    if class_dir.is_dir():
                        for img_file in class_dir.iterdir():
                            if img_file.suffix.lower() in image_processor.supported_formats:
                                image_files.append(str(img_file))

            elif format_type in ["yolo", "coco", "voc"]:
                # 检测格式：从images目录或根目录获取图像
                images_dir = dataset_path / "images"
                if not images_dir.exists():
                    images_dir = dataset_path / "JPEGImages"
                if not images_dir.exists():
                    images_dir = dataset_path

                for img_file in images_dir.iterdir():
                    if img_file.suffix.lower() in image_processor.supported_formats:
                        image_files.append(str(img_file))

            else:
                # 未知格式：递归搜索所有图像文件
                for img_file in dataset_path.rglob("*"):
                    if img_file.suffix.lower() in image_processor.supported_formats:
                        image_files.append(str(img_file))

        except Exception as e:
            logger.error(f"获取图像文件列表失败: {str(e)}")

        return image_files

    def _get_image_annotations(self, image_path: str, dataset_path: str,
                              format_type: str) -> List[Dict[str, Any]]:
        """
        获取图像的标注信息

        Args:
            image_path: 图像路径
            dataset_path: 数据集路径
            format_type: 格式类型

        Returns:
            标注信息列表
        """
        try:
            if format_type == "yolo":
                return self._get_yolo_annotations(image_path, dataset_path)
            elif format_type == "coco":
                return self._get_coco_annotations(image_path, dataset_path)
            elif format_type == "voc":
                return self._get_voc_annotations(image_path, dataset_path)
            else:
                return []
        except Exception as e:
            logger.error(f"获取标注信息失败 {image_path}: {str(e)}")
            return []

    def _get_yolo_annotations(self, image_path: str, dataset_path: str) -> List[Dict[str, Any]]:
        """获取YOLO格式的标注"""
        annotations = []
        image_file = Path(image_path)
        dataset_root = Path(dataset_path)

        # 查找对应的标注文件
        label_file = image_file.with_suffix('.txt')
        if not label_file.exists():
            # 尝试在labels目录中查找
            labels_dir = dataset_root / "labels"
            if labels_dir.exists():
                label_file = labels_dir / f"{image_file.stem}.txt"

        if label_file.exists():
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])

                                # 获取类别名称
                                class_names_path = dataset_root / "obj.names"
                                class_name = f"class_{class_id}"
                                if class_names_path.exists():
                                    with open(class_names_path, 'r') as names_file:
                                        names = names_file.read().strip().split('\n')
                                        if class_id < len(names):
                                            class_name = names[class_id]

                                annotations.append({
                                    'class_id': class_id,
                                    'class_name': class_name,
                                    'bbox': [x_center, y_center, width, height],
                                    'type': 'bbox',
                                    'format': 'yolo_normalized'
                                })
            except Exception as e:
                logger.error(f"解析YOLO标注文件失败 {label_file}: {str(e)}")

        return annotations

    def _get_coco_annotations(self, image_path: str, dataset_path: str) -> List[Dict[str, Any]]:
        """获取COCO格式的标注"""
        annotations = []
        image_file = Path(image_path)
        dataset_root = Path(dataset_path)

        # 查找COCO标注文件
        coco_files = list(dataset_root.glob("*.json"))
        coco_files.extend(dataset_root.glob("annotations/*.json"))

        for coco_file in coco_files:
            try:
                with open(coco_file, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)

                # 查找图像ID
                image_id = None
                for img in coco_data.get('images', []):
                    if img.get('file_name') == image_file.name:
                        image_id = img.get('id')
                        break

                if image_id:
                    # 获取标注
                    for ann in coco_data.get('annotations', []):
                        if ann.get('image_id') == image_id:
                            category_id = ann.get('category_id')
                            bbox = ann.get('bbox', [])

                            # 获取类别名称
                            category_name = f"category_{category_id}"
                            for cat in coco_data.get('categories', []):
                                if cat.get('id') == category_id:
                                    category_name = cat.get('name')
                                    break

                            annotations.append({
                                'category_id': category_id,
                                'category_name': category_name,
                                'bbox': bbox,
                                'area': ann.get('area', 0),
                                'iscrowd': ann.get('iscrowd', 0),
                                'type': 'bbox',
                                'format': 'coco_absolute'
                            })
            except Exception as e:
                logger.error(f"解析COCO标注文件失败 {coco_file}: {str(e)}")

        return annotations

    def _get_voc_annotations(self, image_path: str, dataset_path: str) -> List[Dict[str, Any]]:
        """获取VOC格式的标注"""
        annotations = []
        image_file = Path(image_path)
        dataset_root = Path(dataset_path)

        # 查找对应的XML标注文件
        xml_file = image_file.with_suffix('.xml')
        if not xml_file.exists():
            # 尝试在Annotations目录中查找
            annotations_dir = dataset_root / "Annotations"
            if annotations_dir.exists():
                xml_file = annotations_dir / f"{image_file.stem}.xml"

        if xml_file.exists():
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(xml_file)
                root = tree.getroot()

                for obj in root.findall('object'):
                    name = obj.find('name')
                    if name is not None:
                        class_name = name.text

                        # 获取边界框
                        bbox = obj.find('bndbox')
                        if bbox is not None:
                            xmin = float(bbox.find('xmin').text) if bbox.find('xmin') is not None else 0
                            ymin = float(bbox.find('ymin').text) if bbox.find('ymin') is not None else 0
                            xmax = float(bbox.find('xmax').text) if bbox.find('xmax') is not None else 0
                            ymax = float(bbox.find('ymax').text) if bbox.find('ymax') is not None else 0

                            annotations.append({
                                'class_name': class_name,
                                'bbox': [xmin, ymin, xmax, ymax],
                                'type': 'bbox',
                                'format': 'voc_absolute'
                            })
            except Exception as e:
                logger.error(f"解析VOC标注文件失败 {xml_file}: {str(e)}")

        return annotations

    async def get_image_detail(self, dataset_path: str, image_path: str) -> Optional[Dict[str, Any]]:
        """
        获取单个图像的详细信息

        Args:
            dataset_path: 数据集路径
            image_path: 图像路径

        Returns:
            图像详细信息或None
        """
        try:
            # 识别数据集格式
            format_result = self.format_recognizer.recognize_format(dataset_path)
            best_format = format_result.get('best_format', {}).get('format', 'unknown')

            # 获取图像基本信息
            image_info = image_processor.get_image_info(image_path)
            if not image_info:
                return None

            # 获取标注信息
            annotations = self._get_image_annotations(image_path, dataset_path, best_format)

            # 生成缩略图和预览图URL（这里返回相对路径，实际服务中需要配合路由）
            thumbnail_url = f"/api/datasets/preview/thumbnail?path={image_path}"
            preview_url = f"/api/datasets/preview/image?path={image_path}"

            return {
                'image': image_info,
                'annotation_data': {
                    'format': best_format,
                    'annotations': annotations,
                    'total_annotations': len(annotations)
                },
                'thumbnail_url': thumbnail_url,
                'preview_url': preview_url
            }
        except Exception as e:
            logger.error(f"获取图像详情失败 {image_path}: {str(e)}")
            return None

    async def augment_image(self, image_path: str,
                           augmentation_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对单个图像应用多种数据增强

        Args:
            image_path: 图像路径
            augmentation_configs: 增强配置列表

        Returns:
            增强后的图像列表
        """
        try:
            # 加载原始图像
            original_image = image_processor.load_image(image_path)
            if original_image is None:
                return []

            # 生成原始图像的base64
            original_bytes = image_processor.image_array_to_bytes(original_image)
            original_base64 = image_processor.image_to_base64(original_bytes)

            augmented_images = []

            # 应用每种增强配置
            for i, config in enumerate(augmentation_configs):
                try:
                    # 应用增强
                    augmented_image, applied_operations = image_processor.apply_augmentations(
                        original_image, config
                    )

                    # 转换为base64
                    aug_bytes = image_processor.image_array_to_bytes(augmented_image)
                    aug_base64 = image_processor.image_to_base64(aug_bytes)

                    augmented_images.append({
                        'id': i + 1,
                        'original_path': image_path,
                        'augmented_data': aug_base64,
                        'augmentation_config': config,
                        'applied_operations': applied_operations,
                        'created_at': datetime.now().isoformat()
                    })
                except Exception as e:
                    logger.error(f"应用第{i+1}种增强失败: {str(e)}")
                    continue

            return augmented_images
        except Exception as e:
            logger.error(f"图像增强失败 {image_path}: {str(e)}")
            return []

    async def preview_augmentation(self, image_path: str,
                                 augmentation_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        预览数据增强效果

        Args:
            image_path: 图像路径
            augmentation_configs: 增强配置列表

        Returns:
            增强预览结果
        """
        try:
            # 生成原始图像
            original_image = image_processor.load_image(image_path)
            if original_image is None:
                raise ValueError(f"无法加载图像: {image_path}")

            original_bytes = image_processor.image_array_to_bytes(original_image)
            original_base64 = image_processor.image_to_base64(original_bytes)

            # 生成增强图像
            augmented_images = await self.augment_image(image_path, augmentation_configs)

            # 生成增强摘要
            augmentation_summary = {
                'total_configs': len(augmentation_configs),
                'successful_augmentations': len(augmented_images),
                'operations_used': list(set(op for img in augmented_images for op in img['applied_operations'])),
                'original_image_info': image_processor.get_image_info(image_path),
                'generated_at': datetime.now().isoformat()
            }

            return {
                'original_image': original_base64,
                'augmented_images': augmented_images,
                'augmentation_summary': augmentation_summary
            }
        except Exception as e:
            logger.error(f"增强预览失败 {image_path}: {str(e)}")
            raise

    async def get_dataset_statistics(self, dataset_path: str) -> Dict[str, Any]:
        """
        获取数据集详细统计信息

        Args:
            dataset_path: 数据集路径

        Returns:
            详细统计信息
        """
        try:
            # 识别数据集格式
            format_result = self.format_recognizer.recognize_format(dataset_path)
            best_format = format_result.get('best_format', {}).get('format', 'unknown')

            # 获取基本统计信息
            images_result = await self.get_dataset_images(dataset_path, page=1, page_size=1000)
            images = images_result.get('images', [])

            # 图像质量分析
            image_quality_analysis = self._analyze_image_quality(images)

            # 类别分布分析
            class_balance_analysis = self._analyze_class_distribution(images, best_format)

            # 尺寸分布分析
            size_distribution_analysis = self._analyze_size_distribution(images)

            # 标注质量分析（如果有标注）
            annotation_quality_analysis = None
            if any(img.get('annotations') for img in images):
                annotation_quality_analysis = self._analyze_annotation_quality(images)

            # 生成建议
            recommendations = self._generate_recommendations(
                image_quality_analysis, class_balance_analysis,
                size_distribution_analysis, annotation_quality_analysis
            )

            # 基本统计
            total_images = len(images)
            unique_classes = len(set(ann.get('class_name', 'unknown')
                                   for img in images
                                   for ann in img.get('annotations', [])))

            basic_stats = {
                'dataset_id': 0,  # 这里需要从数据库获取
                'num_images': total_images,
                'num_classes': unique_classes,
                'class_distribution': class_balance_analysis.get('distribution', {}),
                'image_size_distribution': size_distribution_analysis.get('distribution', {}),
                'format_details': {'format': best_format, 'confidence': format_result.get('best_format', {}).get('confidence', 0)},
                'quality_metrics': image_quality_analysis
            }

            return {
                'basic_stats': basic_stats,
                'image_quality_analysis': image_quality_analysis,
                'annotation_quality_analysis': annotation_quality_analysis,
                'class_balance_analysis': class_balance_analysis,
                'size_distribution_analysis': size_distribution_analysis,
                'recommendations': recommendations
            }
        except Exception as e:
            logger.error(f"获取数据集统计信息失败: {str(e)}")
            return {}

    def _analyze_image_quality(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析图像质量"""
        try:
            if not images:
                return {'average_score': 0, 'quality_distribution': {}}

            quality_scores = []
            format_counts = {}
            size_ranges = {'small': 0, 'medium': 0, 'large': 0}

            for img in images:
                # 简单的质量评分（基于分辨率）
                width, height = img.get('width', 0), img.get('height', 0)
                if width * height >= 1920 * 1080:  # Full HD及以上
                    score = 4.0
                    size_ranges['large'] += 1
                elif width * height >= 1280 * 720:  # HD及以上
                    score = 3.0
                    size_ranges['medium'] += 1
                elif width * height >= 640 * 480:  # VGA及以上
                    score = 2.0
                    size_ranges['small'] += 1
                else:
                    score = 1.0
                    size_ranges['small'] += 1

                quality_scores.append(score)

                # 统计格式
                fmt = img.get('format', 'unknown')
                format_counts[fmt] = format_counts.get(fmt, 0) + 1

            avg_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0

            return {
                'average_score': avg_score,
                'quality_distribution': {
                    'high': sum(1 for s in quality_scores if s >= 4),
                    'medium': sum(1 for s in quality_scores if 2 <= s < 4),
                    'low': sum(1 for s in quality_scores if s < 2)
                },
                'format_distribution': format_counts,
                'size_distribution': size_ranges,
                'total_analyzed': len(images)
            }
        except Exception as e:
            logger.error(f"图像质量分析失败: {str(e)}")
            return {'average_score': 0, 'quality_distribution': {}}

    def _analyze_class_distribution(self, images: List[Dict[str, Any]], format_type: str) -> Dict[str, Any]:
        """分析类别分布"""
        try:
            class_counts = {}
            total_annotations = 0

            for img in images:
                annotations = img.get('annotations', [])
                for ann in annotations:
                    class_name = ann.get('class_name', 'unknown')
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    total_annotations += 1

            # 计算百分比
            distribution = {}
            for class_name, count in class_counts.items():
                percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
                distribution[class_name] = {
                    'count': count,
                    'percentage': round(percentage, 2)
                }

            # 计算平衡性指标
            if class_counts:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                balance_ratio = min_count / max_count if max_count > 0 else 0
            else:
                balance_ratio = 0

            return {
                'distribution': distribution,
                'total_classes': len(class_counts),
                'total_annotations': total_annotations,
                'balance_ratio': round(balance_ratio, 3),
                'is_balanced': balance_ratio >= 0.5,  # 简单的平衡性判断
                'format_type': format_type
            }
        except Exception as e:
            logger.error(f"类别分布分析失败: {str(e)}")
            return {'distribution': {}, 'total_classes': 0, 'total_annotations': 0}

    def _analyze_size_distribution(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析尺寸分布"""
        try:
            sizes = []
            aspects = []

            for img in images:
                width, height = img.get('width', 0), img.get('height', 0)
                if width > 0 and height > 0:
                    sizes.append(width * height)
                    aspects.append(width / height)

            if not sizes:
                return {'distribution': {}, 'aspect_distribution': {}}

            # 尺寸分布
            sizes.sort()
            size_ranges = {
                'tiny (<0.1MP)': sum(1 for s in sizes if s < 100000),
                'small (0.1-0.5MP)': sum(1 for s in sizes if 100000 <= s < 500000),
                'medium (0.5-2MP)': sum(1 for s in sizes if 500000 <= s < 2000000),
                'large (2-8MP)': sum(1 for s in sizes if 2000000 <= s < 8000000),
                'huge (>8MP)': sum(1 for s in sizes if s >= 8000000)
            }

            # 宽高比分布
            aspect_ranges = {
                'portrait (<0.9)': sum(1 for a in aspects if a < 0.9),
                'square (0.9-1.1)': sum(1 for a in aspects if 0.9 <= a <= 1.1),
                'landscape (>1.1)': sum(1 for a in aspects if a > 1.1)
            }

            return {
                'distribution': size_ranges,
                'aspect_distribution': aspect_ranges,
                'min_resolution': min(sizes),
                'max_resolution': max(sizes),
                'avg_resolution': sum(sizes) / len(sizes),
                'min_aspect': min(aspects),
                'max_aspect': max(aspects),
                'avg_aspect': sum(aspects) / len(aspects),
                'total_analyzed': len(images)
            }
        except Exception as e:
            logger.error(f"尺寸分布分析失败: {str(e)}")
            return {'distribution': {}, 'aspect_distribution': {}}

    def _analyze_annotation_quality(self, images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析标注质量"""
        try:
            total_annotations = 0
            images_with_annotations = 0
            bbox_sizes = []

            for img in images:
                annotations = img.get('annotations', [])
                if annotations:
                    images_with_annotations += 1

                for ann in annotations:
                    total_annotations += 1
                    bbox = ann.get('bbox', [])
                    if len(bbox) >= 4:
                        if ann.get('format') == 'yolo_normalized':
                            # YOLO格式是相对坐标，转换为绝对尺寸
                            width, height = img.get('width', 1), img.get('height', 1)
                            w = bbox[2] * width
                            h = bbox[3] * height
                        else:
                            # 绝对坐标
                            if len(bbox) == 4 and ann.get('format') in ['coco_absolute', 'voc_absolute']:
                                if ann.get('format') == 'coco_absolute':
                                    w, h = bbox[2], bbox[3]  # x, y, width, height
                                else:  # voc_absolute
                                    w = bbox[2] - bbox[0]  # xmax - xmin
                                    h = bbox[3] - bbox[1]  # ymax - ymin
                            else:
                                continue

                        bbox_sizes.append(w * h)

            # 计算统计信息
            avg_annotations_per_image = total_annotations / len(images) if images else 0
            coverage_rate = images_with_annotations / len(images) if images else 0

            bbox_stats = {}
            if bbox_sizes:
                bbox_sizes.sort()
                bbox_stats = {
                    'min_size': min(bbox_sizes),
                    'max_size': max(bbox_sizes),
                    'avg_size': sum(bbox_sizes) / len(bbox_sizes),
                    'median_size': bbox_sizes[len(bbox_sizes) // 2]
                }

            return {
                'total_annotations': total_annotations,
                'images_with_annotations': images_with_annotations,
                'coverage_rate': round(coverage_rate, 3),
                'avg_annotations_per_image': round(avg_annotations_per_image, 2),
                'bbox_size_stats': bbox_stats
            }
        except Exception as e:
            logger.error(f"标注质量分析失败: {str(e)}")
            return {}

    def _generate_recommendations(self, image_quality: Dict[str, Any],
                                 class_balance: Dict[str, Any],
                                 size_dist: Dict[str, Any],
                                 annotation_quality: Optional[Dict[str, Any]]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        try:
            # 图像质量建议
            avg_quality = image_quality.get('average_score', 0)
            if avg_quality < 2.0:
                recommendations.append("建议提升图像分辨率，当前平均质量较低")
            elif avg_quality < 3.0:
                recommendations.append("考虑增强部分图像的分辨率以获得更好的训练效果")

            # 类别平衡建议
            balance_ratio = class_balance.get('balance_ratio', 1.0)
            if balance_ratio < 0.2:
                recommendations.append("数据集类别严重不平衡，建议进行数据增强或重采样")
            elif balance_ratio < 0.5:
                recommendations.append("数据集类别不平衡，建议对少数类进行数据增强")

            # 尺寸分布建议
            size_dist_data = size_dist.get('aspect_distribution', {})
            total_images = sum(size_dist_data.values())
            if total_images > 0:
                square_ratio = size_dist_data.get('square (0.9-1.1)', 0) / total_images
                if square_ratio < 0.3:
                    recommendations.append("建议增加更多正方形图像或使用随机裁剪")

            # 标注质量建议
            if annotation_quality:
                coverage_rate = annotation_quality.get('coverage_rate', 0)
                if coverage_rate < 0.8:
                    recommendations.append(f"只有{coverage_rate*100:.1f}%的图像有标注，建议完善标注覆盖率")

                avg_ann = annotation_quality.get('avg_annotations_per_image', 0)
                if avg_ann < 1.0:
                    recommendations.append("平均每张图像标注数量较少，建议检查标注完整性")

            if not recommendations:
                recommendations.append("数据集质量良好，可以用于模型训练")

        except Exception as e:
            logger.error(f"生成建议失败: {str(e)}")

        return recommendations

    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(exist_ok=True)
            logger.info("临时文件清理完成")
        except Exception as e:
            logger.error(f"清理临时文件失败: {str(e)}")


# 创建全局实例
augmentation_service = AugmentationService()