"""
数据集路径映射工具类
用于统一不同格式数据集的图像和标签路径
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional


class PathMapper:
    """路径映射工具类"""

    # 统一的目录名称
    UNIFIED_IMAGE_DIR = "images"
    UNIFIED_LABEL_DIR = "labels"

    @staticmethod
    def build_path_mapping(
        dataset_path: Path,
        image_dir: Path,
        label_dir: Optional[Path] = None,
        annotation_file: Optional[Path] = None
    ) -> Dict:
        """
        构建路径映射信息

        Args:
            dataset_path: 数据集根目录
            image_dir: 图像目录（绝对路径）
            label_dir: 标签目录（绝对路径，可选）
            annotation_file: 标注文件（绝对路径，可选，用于COCO）

        Returns:
            路径映射字典
        """
        mapping = {
            "unified_images": PathMapper.UNIFIED_IMAGE_DIR,
            "unified_labels": PathMapper.UNIFIED_LABEL_DIR,
        }

        # 计算相对路径
        if image_dir and image_dir.exists():
            try:
                mapping["images_dir"] = str(image_dir.relative_to(dataset_path))
            except ValueError:
                # image_dir不在dataset_path下，使用绝对路径
                mapping["images_dir"] = str(image_dir)
                mapping["images_dir_absolute"] = True

        # 标签目录（检测/分割格式）
        if label_dir and label_dir.exists():
            try:
                mapping["labels_dir"] = str(label_dir.relative_to(dataset_path))
            except ValueError:
                mapping["labels_dir"] = str(label_dir)
                mapping["labels_dir_absolute"] = True

        # 标注文件（COCO格式）
        if annotation_file and annotation_file.exists():
            try:
                mapping["annotation_file"] = str(annotation_file.relative_to(dataset_path))
            except ValueError:
                mapping["annotation_file"] = str(annotation_file)

        return mapping

    @staticmethod
    def verify_file_correspondence(
        image_dir: Path,
        label_dir: Optional[Path] = None,
        label_ext: str = ".txt"
    ) -> Dict:
        """
        验证图像和标签文件的对应关系

        Args:
            image_dir: 图像目录
            label_dir: 标签目录
            label_ext: 标签文件扩展名

        Returns:
            验证结果字典
        """
        if not label_dir or not label_dir.exists():
            return {
                "file_pairs": 0,
                "missing_labels": 0,
                "missing_images": 0,
                "valid": False,
                "error": "标签目录不存在"
            }

        # 获取图像文件（不含扩展名）
        image_stems = set()
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            for f in image_dir.glob(f"*{ext}"):
                image_stems.add(f.stem)

        # 获取标签文件（不含扩展名）
        label_stems = set()
        for f in label_dir.glob(f"*{label_ext}"):
            label_stems.add(f.stem)

        # 计算对应关系
        paired = image_stems & label_stems
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems

        return {
            "file_pairs": len(paired),
            "missing_labels": len(missing_labels),
            "missing_images": len(missing_images),
            "valid": len(paired) > 0,
            "sample_unpaired": list(missing_labels)[:10] if missing_labels else []
        }

    @staticmethod
    def get_training_paths(dataset_meta: Dict) -> Dict[str, str]:
        """
        获取训练时使用的路径

        根据数据集meta中的path_mapping，返回实际的图像和标签路径

        Args:
            dataset_meta: 数据集meta字典

        Returns:
            {"images": "实际图像路径", "labels": "实际标签路径"}
        """
        path_mapping = dataset_meta.get("path_mapping", {})
        dataset_path = dataset_meta.get("dataset_path", "")

        images_dir = path_mapping.get("images_dir", PathMapper.UNIFIED_IMAGE_DIR)
        labels_dir = path_mapping.get("labels_dir", PathMapper.UNIFIED_LABEL_DIR)

        return {
            "images": f"{dataset_path}/{images_dir}",
            "labels": f"{dataset_path}/{labels_dir}"
        }
