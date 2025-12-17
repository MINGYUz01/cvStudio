"""
数据集服务类
"""

import os
import shutil
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from fastapi import UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.models.dataset import Dataset
from app.utils.format_recognizers import DatasetFormatRecognizer
from app.core.config import settings


class DatasetService:
    """数据集服务类"""

    def __init__(self):
        self.format_recognizer = DatasetFormatRecognizer()
        self.storage_path = Path(settings.DATASET_STORAGE_PATH)
        self.thumbnail_path = Path(settings.THUMBNAIL_STORAGE_PATH)

        # 确保存储目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.thumbnail_path.mkdir(parents=True, exist_ok=True)

    async def create_dataset_from_upload(self,
                                       db: Session,
                                       name: str,
                                       description: str,
                                       files: List[UploadFile],
                                       user_id: int = None) -> Dataset:
        """
        从上传的文件创建数据集

        Args:
            db: 数据库会话
            name: 数据集名称
            description: 数据集描述
            files: 上传的文件列表
            user_id: 用户ID

        Returns:
            Dataset: 创建的数据集对象
        """
        # 检查数据集名称是否已存在
        existing_dataset = db.query(Dataset).filter(Dataset.name == name).first()
        if existing_dataset:
            raise HTTPException(status_code=400, detail=f"数据集名称 '{name}' 已存在")

        # 创建数据集存储目录
        dataset_dir = self.storage_path / name
        dataset_dir.mkdir(exist_ok=True)

        try:
            # 保存上传的文件
            saved_files = []
            for file in files:
                file_path = dataset_dir / file.filename
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                saved_files.append(str(file_path))

            # 识别数据集格式
            format_result = self.format_recognizer.recognize_format(str(dataset_dir))
            best_format = format_result["best_format"]

            # 提取数据集元信息
            metadata = self._extract_metadata(format_result)

            # 创建数据集记录
            dataset = Dataset(
                name=name,
                description=description,
                path=str(dataset_dir),
                format=best_format["format"],
                num_images=metadata.get("num_images", 0),
                num_classes=metadata.get("num_classes", 0),
                classes=metadata.get("classes", []),
                meta=metadata
            )

            db.add(dataset)
            db.commit()
            db.refresh(dataset)

            # 生成缩略图（异步执行）
            asyncio.create_task(self._generate_thumbnails_async(dataset.id, str(dataset_dir)))

            return dataset

        except Exception as e:
            # 如果失败，清理已创建的目录
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)
            raise HTTPException(status_code=500, detail=f"创建数据集失败: {str(e)}")

    def register_existing_dataset(self,
                                db: Session,
                                name: str,
                                description: str,
                                dataset_path: str,
                                user_id: int = None) -> Dataset:
        """
        注册现有数据集

        Args:
            db: 数据库会话
            name: 数据集名称
            description: 数据集描述
            dataset_path: 数据集路径
            user_id: 用户ID

        Returns:
            Dataset: 创建的数据集对象
        """
        # 检查路径是否存在
        path = Path(dataset_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"数据集路径不存在: {dataset_path}")

        # 检查数据集名称是否已存在
        existing_dataset = db.query(Dataset).filter(Dataset.name == name).first()
        if existing_dataset:
            raise HTTPException(status_code=400, detail=f"数据集名称 '{name}' 已存在")

        try:
            # 识别数据集格式
            format_result = self.format_recognizer.recognize_format(dataset_path)
            best_format = format_result["best_format"]

            if best_format["confidence"] < 0.3:
                raise HTTPException(status_code=400, detail=f"无法识别数据集格式，置信度太低: {best_format.get('error')}")

            # 提取数据集元信息
            metadata = self._extract_metadata(format_result)

            # 创建数据集记录
            dataset = Dataset(
                name=name,
                description=description,
                path=dataset_path,
                format=best_format["format"],
                num_images=metadata.get("num_images", 0),
                num_classes=metadata.get("num_classes", 0),
                classes=metadata.get("classes", []),
                meta=metadata
            )

            db.add(dataset)
            db.commit()
            db.refresh(dataset)

            # 生成缩略图（异步执行）
            asyncio.create_task(self._generate_thumbnails_async(dataset.id, dataset_path))

            return dataset

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"注册数据集失败: {str(e)}")

    def get_dataset(self, db: Session, dataset_id: int) -> Optional[Dataset]:
        """
        获取数据集

        Args:
            db: 数据库会话
            dataset_id: 数据集ID

        Returns:
            Optional[Dataset]: 数据集对象
        """
        return db.query(Dataset).filter(Dataset.id == dataset_id, Dataset.is_active == "active").first()

    def get_datasets(self, db: Session, skip: int = 0, limit: int = 100,
                    format_filter: str = None) -> List[Dataset]:
        """
        获取数据集列表

        Args:
            db: 数据库会话
            skip: 跳过记录数
            limit: 限制记录数
            format_filter: 格式过滤器

        Returns:
            List[Dataset]: 数据集列表
        """
        query = db.query(Dataset).filter(Dataset.is_active == "active")

        if format_filter:
            query = query.filter(Dataset.format == format_filter)

        return query.offset(skip).limit(limit).all()

    def update_dataset(self, db: Session, dataset_id: int,
                      name: str = None, description: str = None) -> Optional[Dataset]:
        """
        更新数据集信息

        Args:
            db: 数据库会话
            dataset_id: 数据集ID
            name: 新名称
            description: 新描述

        Returns:
            Optional[Dataset]: 更新后的数据集对象
        """
        dataset = self.get_dataset(db, dataset_id)
        if not dataset:
            return None

        if name:
            # 检查名称是否重复
            existing_dataset = db.query(Dataset).filter(
                Dataset.name == name, Dataset.id != dataset_id
            ).first()
            if existing_dataset:
                raise HTTPException(status_code=400, detail=f"数据集名称 '{name}' 已存在")
            dataset.name = name

        if description is not None:
            dataset.description = description

        db.commit()
        db.refresh(dataset)
        return dataset

    def delete_dataset(self, db: Session, dataset_id: int) -> bool:
        """
        删除数据集（软删除）

        Args:
            db: 数据库会话
            dataset_id: 数据集ID

        Returns:
            bool: 是否删除成功
        """
        dataset = self.get_dataset(db, dataset_id)
        if not dataset:
            return False

        dataset.is_active = "deleted"
        db.commit()

        # 可选：清理缩略图
        dataset_thumbnail_dir = self.thumbnail_path / str(dataset_id)
        if dataset_thumbnail_dir.exists():
            shutil.rmtree(dataset_thumbnail_dir)

        return True

    def rescan_dataset(self, db: Session, dataset_id: int) -> Optional[Dataset]:
        """
        重新扫描数据集，更新元信息

        Args:
            db: 数据库会话
            dataset_id: 数据集ID

        Returns:
            Optional[Dataset]: 更新后的数据集对象
        """
        dataset = self.get_dataset(db, dataset_id)
        if not dataset:
            return None

        try:
            # 重新识别数据集格式
            format_result = self.format_recognizer.recognize_format(dataset.path)
            best_format = format_result["best_format"]

            # 提取数据集元信息
            metadata = self._extract_metadata(format_result)

            # 更新数据集信息
            dataset.format = best_format["format"]
            dataset.num_images = metadata.get("num_images", 0)
            dataset.num_classes = metadata.get("num_classes", 0)
            dataset.classes = metadata.get("classes", [])
            dataset.meta = metadata

            db.commit()
            db.refresh(dataset)

            # 重新生成缩略图
            asyncio.create_task(self._generate_thumbnails_async(dataset.id, dataset.path))

            return dataset

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"重新扫描数据集失败: {str(e)}")

    def _extract_metadata(self, format_result: Dict) -> Dict:
        """
        从格式识别结果中提取元信息

        Args:
            format_result: 格式识别结果

        Returns:
            Dict: 提取的元信息
        """
        best_format = format_result["best_format"]
        details = best_format.get("details", {})

        metadata = {
            "format_confidence": best_format.get("confidence", 0),
            "recognition_error": best_format.get("error"),
            "all_recognition_results": format_result.get("all_results", {})
        }

        # 基础信息
        metadata["num_images"] = details.get("num_images", 0)
        metadata["num_classes"] = details.get("num_classes", 0)
        metadata["classes"] = details.get("classes", [])

        # 格式特定信息
        if best_format["format"] == "yolo":
            metadata.update({
                "data_config": details.get("data_config", {}),
                "label_stats": details.get("label_stats", {}),
                "image_stats": details.get("image_stats", {})
            })
        elif best_format["format"] == "coco":
            metadata.update({
                "coco_data": details.get("coco_data", {}),
                "validation": details.get("validation", {}),
                "image_stats": details.get("image_stats", {})
            })
        elif best_format["format"] == "voc":
            metadata.update({
                "xml_stats": details.get("xml_stats", {}),
                "structure": details.get("structure", {}),
                "imagesets": details.get("imagesets", {}),
                "image_stats": details.get("image_stats", {})
            })
        elif best_format["format"] == "classification":
            metadata.update({
                "class_directories": details.get("classes", {}),
                "structure": details.get("structure", {}),
                "size_distribution": details.get("size_distribution", {}),
                "format_distribution": details.get("format_distribution", {})
            })

        return metadata

    async def _generate_thumbnails_async(self, dataset_id: int, dataset_path: str):
        """
        异步生成缩略图

        Args:
            dataset_id: 数据集ID
            dataset_path: 数据集路径
        """
        try:
            await self._generate_thumbnails(dataset_id, dataset_path)
        except Exception as e:
            print(f"生成缩略图失败: {e}")

    async def get_dataset_preview(self, dataset_id: int, limit: int = 10) -> Dict:
        """
        获取数据集预览信息

        Args:
            dataset_id: 数据集ID
            limit: 预览图像数量限制

        Returns:
            Dict: 预览信息
        """
        from sqlalchemy.orm import Session
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            dataset = self.get_dataset(db, dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

            preview_data = {
                "dataset_id": dataset_id,
                "sample_images": [],
                "format_details": {},
                "statistics": {}
            }

            # 根据数据集格式获取样本图像
            sample_images = await self._get_sample_images(dataset, limit)
            preview_data["sample_images"] = sample_images

            # 添加格式特定详情
            preview_data["format_details"] = self._get_format_details(dataset)

            # 添加基础统计信息
            preview_data["statistics"] = {
                "format": dataset.format,
                "num_images": dataset.num_images,
                "num_classes": dataset.num_classes,
                "classes": dataset.classes[:10] if dataset.classes else []  # 只显示前10个类别
            }

            return preview_data

        finally:
            db.close()

    async def get_dataset_statistics(self, dataset_id: int) -> Dict:
        """
        获取数据集统计信息

        Args:
            dataset_id: 数据集ID

        Returns:
            Dict: 统计信息
        """
        from sqlalchemy.orm import Session
        from app.database import SessionLocal
        from app.utils.dataset_statistics import DatasetStatistics

        db = SessionLocal()
        try:
            dataset = self.get_dataset(db, dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

            # 使用专门的统计分析工具
            stats_analyzer = DatasetStatistics(dataset.path)
            detailed_stats = stats_analyzer.analyze_dataset(dataset.format)

            # 构建返回数据
            stats_data = {
                "dataset_id": dataset_id,
                "basic_info": {
                    "num_images": dataset.num_images,
                    "num_classes": dataset.num_classes,
                    "format": dataset.format,
                    "path": dataset.path
                },
                "detailed_statistics": detailed_stats,
                "summary": {
                    "total_images": detailed_stats.get("image_statistics", {}).get("total_images", 0),
                    "valid_images": detailed_stats.get("image_statistics", {}).get("valid_images", 0),
                    "corrupted_images": detailed_stats.get("image_statistics", {}).get("corrupted_images", 0),
                    "quality_score": detailed_stats.get("quality_metrics", {}).get("overall_quality_score", 0),
                    "recommendations": detailed_stats.get("quality_metrics", {}).get("recommendations", [])
                }
            }

            return stats_data

        finally:
            db.close()

    async def validate_dataset(self, dataset_id: int) -> Dict:
        """
        验证数据集

        Args:
            dataset_id: 数据集ID

        Returns:
            Dict: 验证结果
        """
        from sqlalchemy.orm import Session
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            dataset = self.get_dataset(db, dataset_id)
            if not dataset:
                raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": [],
                "suggestions": []
            }

            # 基础验证
            dataset_path = Path(dataset.path)
            if not dataset_path.exists():
                validation_result["is_valid"] = False
                validation_result["errors"].append("数据集路径不存在")
                return validation_result

            # 验证图像文件
            image_validation = await self._validate_image_files(dataset)
            validation_result["errors"].extend(image_validation["errors"])
            validation_result["warnings"].extend(image_validation["warnings"])

            # 验证标注文件（如果有）
            if dataset.format in ["yolo", "coco", "voc"]:
                annotation_validation = await self._validate_annotation_files(dataset)
                validation_result["errors"].extend(annotation_validation["errors"])
                validation_result["warnings"].extend(annotation_validation["warnings"])

            # 格式特定验证
            format_validation = await self._validate_dataset_format(dataset)
            validation_result["errors"].extend(format_validation["errors"])
            validation_result["warnings"].extend(format_validation["warnings"])
            validation_result["suggestions"].extend(format_validation["suggestions"])

            # 设置整体验证状态
            validation_result["is_valid"] = len(validation_result["errors"]) == 0

            return validation_result

        finally:
            db.close()

    async def _get_sample_images(self, dataset: Dataset, limit: int) -> List[Dict]:
        """获取样本图像信息"""
        sample_images = []
        dataset_path = Path(dataset.path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        try:
            # 根据数据集格式查找图像
            if dataset.format == "classification":
                # 从每个类别目录中获取样本
                class_dirs = []
                for item in dataset_path.iterdir():
                    if item.is_dir():
                        class_dirs.append(item)

                for class_dir in class_dirs[:min(len(class_dirs), limit)]:  # 每类一张图
                    for img_file in class_dir.iterdir():
                        if img_file.suffix.lower() in image_extensions:
                            sample_images.append({
                                "path": str(img_file),
                                "filename": img_file.name,
                                "class": class_dir.name,
                                "size": self._get_image_size(img_file)
                            })
                            break

            else:
                # 对于其他格式，查找所有图像文件
                all_images = []
                for ext in image_extensions:
                    all_images.extend(dataset_path.rglob(f"*{ext}"))

                for i, img_file in enumerate(all_images[:limit]):
                    sample_images.append({
                        "path": str(img_file),
                        "filename": img_file.name,
                        "class": "unknown",
                        "size": self._get_image_size(img_file)
                    })

        except Exception as e:
            print(f"获取样本图像失败: {e}")

        return sample_images

    def _get_image_size(self, img_path: Path) -> Dict:
        """获取图像尺寸信息"""
        try:
            from PIL import Image
            with Image.open(img_path) as img:
                return {"width": img.width, "height": img.height}
        except Exception:
            return {"width": 0, "height": 0}

    def _get_format_details(self, dataset: Dataset) -> Dict:
        """获取格式特定详情"""
        details = {
            "format": dataset.format,
            "confidence": 0,
            "specific_info": {}
        }

        if dataset.meta:
            details["confidence"] = dataset.meta.get("format_confidence", 0)
            details["specific_info"] = {
                k: v for k, v in dataset.meta.items()
                if k not in ["num_images", "num_classes", "classes"]
            }

        return details

    def _get_yolo_statistics(self, dataset: Dataset) -> Dict:
        """获取YOLO格式统计信息"""
        stats = {}
        if dataset.meta:
            stats["class_distribution"] = dataset.meta.get("label_stats", {}).get("class_distribution", {})
            stats["avg_objects_per_image"] = dataset.meta.get("label_stats", {}).get("avg_objects_per_image", 0)
            stats["image_size_distribution"] = dataset.meta.get("image_stats", {})
        return stats

    def _get_coco_statistics(self, dataset: Dataset) -> Dict:
        """获取COCO格式统计信息"""
        stats = {}
        if dataset.meta:
            coco_data = dataset.meta.get("coco_data", {})
            stats["class_distribution"] = coco_data.get("category_counts", {})
            stats["image_size_distribution"] = dataset.meta.get("image_stats", {})
            stats["coco_summary"] = {
                "num_images": coco_data.get("num_images", 0),
                "num_annotations": coco_data.get("num_annotations", 0),
                "num_categories": coco_data.get("num_categories", 0)
            }
        return stats

    def _get_voc_statistics(self, dataset: Dataset) -> Dict:
        """获取VOC格式统计信息"""
        stats = {}
        if dataset.meta:
            xml_stats = dataset.meta.get("xml_stats", {})
            stats["avg_objects_per_image"] = xml_stats.get("avg_objects_per_image", 0)
            stats["max_objects_per_image"] = xml_stats.get("max_objects_per_image", 0)
            stats["class_distribution"] = xml_stats.get("classes", {})
            stats["image_size_distribution"] = dataset.meta.get("image_stats", {})
        return stats

    def _get_classification_statistics(self, dataset: Dataset) -> Dict:
        """获取分类格式统计信息"""
        stats = {}
        if dataset.meta:
            class_directories = dataset.meta.get("class_directories", {})
            stats["class_distribution"] = {
                class_name: info.get("image_count", 0)
                for class_name, info in class_directories.items()
            }
            stats["size_distribution"] = dataset.meta.get("size_distribution", {})
            stats["format_distribution"] = dataset.meta.get("format_distribution", {})
        return stats

    async def _validate_image_files(self, dataset: Dataset) -> Dict:
        """验证图像文件"""
        validation = {"errors": [], "warnings": []}
        dataset_path = Path(dataset.path)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        try:
            # 检查是否有图像文件
            image_count = 0
            corrupted_files = []

            for ext in image_extensions:
                images = list(dataset_path.rglob(f"*{ext}"))
                image_count += len(images)

                # 抽样检查图像完整性
                for img_file in images[:10]:  # 最多检查10张
                    try:
                        from PIL import Image
                        with Image.open(img_file) as img:
                            img.verify()
                    except Exception:
                        corrupted_files.append(img_file.name)

            if image_count == 0:
                validation["errors"].append("数据集中没有找到图像文件")
            elif image_count < 10:
                validation["warnings"].append("数据集图像数量较少，可能影响训练效果")

            if corrupted_files:
                validation["errors"].append(f"发现 {len(corrupted_files)} 个损坏的图像文件")

        except Exception as e:
            validation["errors"].append(f"验证图像文件时出错: {str(e)}")

        return validation

    async def _validate_annotation_files(self, dataset: Dataset) -> Dict:
        """验证标注文件"""
        validation = {"errors": [], "warnings": []}
        dataset_path = Path(dataset.path)

        try:
            if dataset.format == "yolo":
                # 验证YOLO标注文件
                annotation_count = 0
                for txt_file in dataset_path.rglob("*.txt"):
                    if txt_file.name not in ["obj.names", "classes.txt", "names.txt"]:
                        annotation_count += 1

                if annotation_count == 0:
                    validation["errors"].append("没有找到YOLO标注文件(.txt)")

            elif dataset.format == "coco":
                # 验证COCO标注文件
                json_files = list(dataset_path.rglob("*.json"))
                if not json_files:
                    validation["errors"].append("没有找到COCO标注文件(.json)")

            elif dataset.format == "voc":
                # 验证VOC标注文件
                xml_count = 0
                for xml_file in dataset_path.rglob("*.xml"):
                    xml_count += 1

                if xml_count == 0:
                    validation["errors"].append("没有找到VOC标注文件(.xml)")

        except Exception as e:
            validation["errors"].append(f"验证标注文件时出错: {str(e)}")

        return validation

    async def _validate_dataset_format(self, dataset: Dataset) -> Dict:
        """验证数据集格式"""
        validation = {"errors": [], "warnings": [], "suggestions": []}

        # 检查格式置信度
        if dataset.meta:
            confidence = dataset.meta.get("format_confidence", 0)
            if confidence < 0.5:
                validation["warnings"].append("数据集格式识别置信度较低")
                validation["suggestions"].append("建议检查数据集结构或手动指定格式")

        # 检查类别和图像的比例
        if dataset.num_images > 0 and dataset.num_classes > 0:
            avg_images_per_class = dataset.num_images / dataset.num_classes
            if avg_images_per_class < 5:
                validation["warnings"].append("部分类别样本数量较少，可能影响训练效果")
                validation["suggestions"].append("建议为每个类别至少收集50-100个样本")

        return validation

    async def _generate_thumbnails(self, dataset_id: int, dataset_path: str):
        """
        生成数据集缩略图

        Args:
            dataset_id: 数据集ID
            dataset_path: 数据集路径
        """
        from PIL import Image
        import os

        try:
            # 创建缩略图存储目录
            thumbnail_dir = self.thumbnail_path / str(dataset_id)
            thumbnail_dir.mkdir(parents=True, exist_ok=True)

            dataset_path = Path(dataset_path)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            thumbnail_size = (256, 256)  # 缩略图尺寸

            # 查找图像文件
            all_images = []
            for ext in image_extensions:
                all_images.extend(dataset_path.rglob(f"*{ext}"))

            # 限制生成缩略图数量以避免性能问题
            max_thumbnails = 100
            processed_count = 0

            for img_path in all_images:
                if processed_count >= max_thumbnails:
                    break

                try:
                    # 生成缩略图文件路径
                    relative_path = img_path.relative_to(dataset_path)
                    thumbnail_path = thumbnail_dir / f"{relative_path.stem}_thumb{relative_path.suffix}"

                    # 确保缩略图目录存在
                    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)

                    # 生成缩略图
                    with Image.open(img_path) as img:
                        # 转换为RGB（处理RGBA等格式）
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # 创建缩略图，保持宽高比
                        img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

                        # 保存缩略图
                        img.save(thumbnail_path, 'JPEG', quality=85)

                    processed_count += 1

                except Exception as e:
                    print(f"生成缩略图失败 {img_path}: {e}")
                    continue

            print(f"数据集 {dataset_id} 缩略图生成完成，共处理 {processed_count} 张图像")

        except Exception as e:
            print(f"生成数据集缩略图时出错: {e}")

    async def get_thumbnail(self, dataset_id: int, image_path: str) -> Optional[str]:
        """
        获取图像的缩略图路径

        Args:
            dataset_id: 数据集ID
            image_path: 原始图像路径

        Returns:
            Optional[str]: 缩略图路径，如果不存在则返回None
        """
        try:
            img_path = Path(image_path)
            thumbnail_dir = self.thumbnail_path / str(dataset_id)

            # 生成缩略图文件名
            relative_path = img_path.relative_to(img_path.parents[-2])  # 获取相对于数据集目录的路径
            thumbnail_name = f"{img_path.stem}_thumb.jpg"

            # 在缩略图目录中查找
            for thumbnail_file in thumbnail_dir.rglob(f"{img_path.stem}_thumb*"):
                if thumbnail_file.exists():
                    return str(thumbnail_file)

            return None

        except Exception as e:
            print(f"获取缩略图失败: {e}")
            return None

    async def generate_single_thumbnail(self, dataset_id: int, image_path: str) -> Optional[str]:
        """
        为单个图像生成缩略图

        Args:
            dataset_id: 数据集ID
            image_path: 图像路径

        Returns:
            Optional[str]: 生成的缩略图路径
        """
        from PIL import Image

        try:
            img_path = Path(image_path)
            if not img_path.exists():
                return None

            # 创建缩略图存储目录
            thumbnail_dir = self.thumbnail_path / str(dataset_id)
            thumbnail_dir.mkdir(parents=True, exist_ok=True)

            # 生成缩略图文件路径
            thumbnail_path = thumbnail_dir / f"{img_path.stem}_thumb.jpg"

            # 生成缩略图
            with Image.open(img_path) as img:
                # 转换为RGB
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # 创建缩略图
                thumbnail_size = (256, 256)
                img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

                # 保存缩略图
                img.save(thumbnail_path, 'JPEG', quality=85)

            return str(thumbnail_path)

        except Exception as e:
            print(f"生成单个缩略图失败: {e}")
            return None

    def get_thumbnails_list(self, dataset_id: int) -> List[str]:
        """
        获取数据集的所有缩略图列表

        Args:
            dataset_id: 数据集ID

        Returns:
            List[str]: 缩略图路径列表
        """
        try:
            thumbnail_dir = self.thumbnail_path / str(dataset_id)
            if not thumbnail_dir.exists():
                return []

            thumbnails = []
            for thumbnail_file in thumbnail_dir.rglob("*_thumb.jpg"):
                if thumbnail_file.is_file():
                    thumbnails.append(str(thumbnail_file))

            return sorted(thumbnails)

        except Exception as e:
            print(f"获取缩略图列表失败: {e}")
            return []

    def clear_thumbnails(self, dataset_id: int) -> bool:
        """
        清理数据集的缩略图

        Args:
            dataset_id: 数据集ID

        Returns:
            bool: 是否清理成功
        """
        try:
            thumbnail_dir = self.thumbnail_path / str(dataset_id)
            if thumbnail_dir.exists():
                shutil.rmtree(thumbnail_dir)
            return True

        except Exception as e:
            print(f"清理缩略图失败: {e}")
            return False

    async def compare_datasets(self, db: Session, dataset_ids: List[int]) -> Dict:
        """
        比较多个数据集的统计信息

        Args:
            db: 数据库会话
            dataset_ids: 数据集ID列表

        Returns:
            Dict: 比较结果
        """
        try:
            if len(dataset_ids) < 2:
                raise HTTPException(status_code=400, detail="至少需要选择2个数据集进行比较")

            # 获取所有数据集信息
            datasets = []
            for dataset_id in dataset_ids:
                dataset = self.get_dataset(db, dataset_id)
                if not dataset:
                    raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")
                datasets.append(dataset)

            # 获取每个数据集的统计信息
            comparison_data = {
                "datasets": [],
                "comparison_metrics": {},
                "summary": {}
            }

            all_stats = []
            for dataset in datasets:
                stats = await self.get_dataset_statistics(dataset.id)
                comparison_data["datasets"].append({
                    "id": dataset.id,
                    "name": dataset.name,
                    "format": dataset.format,
                    "basic_info": stats["basic_info"],
                    "summary": stats["summary"]
                })
                all_stats.append(stats)

            # 比较指标
            comparison_metrics = {
                "image_count": {},
                "class_count": {},
                "quality_score": {},
                "format_distribution": {},
                "resolution_comparison": {}
            }

            # 图像数量比较
            image_counts = [(d["id"], d["basic_info"]["num_images"]) for d in comparison_data["datasets"]]
            comparison_metrics["image_count"] = {
                "values": dict(image_counts),
                "max": max(image_counts, key=lambda x: x[1])[0],
                "min": min(image_counts, key=lambda x: x[1])[0],
                "range": max(x[1] for x in image_counts) - min(x[1] for x in image_counts)
            }

            # 类别数量比较
            class_counts = [(d["id"], d["basic_info"]["num_classes"]) for d in comparison_data["datasets"]]
            comparison_metrics["class_count"] = {
                "values": dict(class_counts),
                "max": max(class_counts, key=lambda x: x[1])[0],
                "min": min(class_counts, key=lambda x: x[1])[0],
                "range": max(x[1] for x in class_counts) - min(x[1] for x in class_counts)
            }

            # 质量分数比较
            quality_scores = [(d["id"], d["summary"]["quality_score"]) for d in comparison_data["datasets"]]
            comparison_metrics["quality_score"] = {
                "values": dict(quality_scores),
                "best": max(quality_scores, key=lambda x: x[1])[0],
                "worst": min(quality_scores, key=lambda x: x[1])[0],
                "average": sum(x[1] for x in quality_scores) / len(quality_scores)
            }

            # 格式分布
            formats = [d["format"] for d in comparison_data["datasets"]]
            comparison_metrics["format_distribution"] = {
                "unique_formats": list(set(formats)),
                "format_counts": dict(Counter(formats))
            }

            comparison_data["comparison_metrics"] = comparison_metrics

            # 生成总结
            comparison_data["summary"] = {
                "total_datasets": len(datasets),
                "total_images": sum(d["basic_info"]["num_images"] for d in comparison_data["datasets"]),
                "unique_formats": len(set(d["format"] for d in datasets)),
                "recommendations": self._generate_comparison_recommendations(comparison_data)
            }

            return comparison_data

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"比较数据集失败: {str(e)}")

    def _generate_comparison_recommendations(self, comparison_data: Dict) -> List[str]:
        """生成数据集比较的建议"""
        recommendations = []

        try:
            # 基于质量分数的建议
            quality_scores = comparison_data["comparison_metrics"]["quality_score"]
            best_quality = quality_scores["best"]
            worst_quality = quality_scores["worst"]

            if quality_scores["average"] < 70:
                recommendations.append("整体数据集质量偏低，建议进行数据清洗和增强")

            if quality_scores["values"][best_quality] - quality_scores["values"][worst_quality] > 30:
                recommendations.append("数据集质量差异较大，建议以高质量数据集为基准进行改进")

            # 基于图像数量的建议
            image_count_range = comparison_data["comparison_metrics"]["image_count"]["range"]
            if image_count_range > 1000:
                recommendations.append("数据集规模差异较大，可能影响模型训练效果")

            # 基于格式一致性的建议
            unique_formats = comparison_data["comparison_metrics"]["format_distribution"]["unique_formats"]
            if len(unique_formats) > 1:
                recommendations.append(f"发现多种数据格式: {', '.join(unique_formats)}，建议统一格式以便于管理")

            # 基于类别数量的建议
            class_count_values = list(comparison_data["comparison_metrics"]["class_count"]["values"].values())
            if len(set(class_count_values)) > 1:
                recommendations.append("数据集类别数量不同，可能需要进行类别对齐")

        except Exception as e:
            print(f"生成比较建议失败: {e}")

        return recommendations