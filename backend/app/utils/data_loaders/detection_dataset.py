"""
目标检测数据集
支持YOLO、COCO、VOC格式的数据加载
"""

import random
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

from app.utils.augmentation.pipeline import AugmentationPipeline


class DetectionDataset(Dataset):
    """
    目标检测数据集

    支持的数据集格式：
    - YOLO格式: images/*.jpg, labels/*.txt
    - COCO格式: images/*.jpg, annotations.json
    - VOC格式: JPEGImages/*.jpg, Annotations/*.xml
    """

    # 类别名称映射（COCO 80类）
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    def __init__(
        self,
        root_dir: Union[str, Path],
        format: str = "yolo",
        image_size: int = 640,
        num_classes: int = 80,
        augmentation: Optional[Dict] = None,
        mode: str = "train",
        train_val_split: float = 0.8
    ):
        """
        初始化检测数据集

        Args:
            root_dir: 数据集根目录
            format: 数据格式 ('yolo', 'coco', 'voc')
            image_size: 目标图像尺寸
            num_classes: 类别数量
            augmentation: 数据增强配置
            mode: 'train' 或 'val'
            train_val_split: 训练集划分比例
        """
        self.root_dir = Path(root_dir)
        self.format = format.lower()
        self.image_size = image_size
        self.num_classes = num_classes
        self.mode = mode
        self.samples = []

        # 加载数据
        self._load_data()

        # 划分训练/验证集
        if train_val_split < 1.0:
            self._split_data(train_val_split)

        # 设置数据增强管道
        self.transform = AugmentationPipeline(
            image_size=self.image_size,
            augmentation=augmentation if mode == "train" else None,
            mode=mode,
            task_type="detection"
        )

    def _load_data(self):
        """根据格式加载数据"""
        if self.format == "yolo":
            self._load_yolo_format()
        elif self.format == "coco":
            self._load_coco_format()
        elif self.format == "voc":
            self._load_voc_format()
        else:
            raise ValueError(f"不支持的数据格式: {self.format}")

    def _load_yolo_format(self):
        """
        加载YOLO格式数据集
        目录结构:
        root_dir/
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
        """
        # 查找图像目录
        image_dirs = []
        for possible_name in ['images', 'img', 'JPEGImages']:
            img_dir = self.root_dir / possible_name
            if img_dir.exists():
                image_dirs.append(img_dir)

        # 查找标签目录
        label_dirs = []
        for possible_name in ['labels', 'annotations', 'txt']:
            lbl_dir = self.root_dir / possible_name
            if lbl_dir.exists():
                label_dirs.append(lbl_dir)

        if not image_dirs:
            # 尝试直接在根目录查找图像
            image_dirs = [self.root_dir]

        # 收集图像和对应的标签文件
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        samples = []

        for img_dir in image_dirs:
            # 递归查找所有图像
            for ext in supported_extensions:
                for img_path in img_dir.rglob(f"*{ext}"):
                    # 查找对应的标签文件
                    label_path = self._find_yolo_label(img_path, label_dirs)

                    if label_path or len(label_dirs) == 0:
                        samples.append({
                            'image_path': str(img_path),
                            'label_path': str(label_path) if label_path else None,
                            'image_id': img_path.stem
                        })

        self.samples = samples

    def _find_yolo_label(self, img_path: Path, label_dirs: List[Path]) -> Optional[Path]:
        """查找YOLO标签文件"""
        # 尝试在不同的标签目录中查找
        for label_dir in label_dirs:
            # 同名.txt文件
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                return label_path

            # 尝试保持相对路径
            rel_path = img_path.relative_to(img_path.parent.parent)
            label_path = label_dir / rel_path.with_suffix('.txt')
            if label_path.exists():
                return label_path

        return None

    def _load_coco_format(self):
        """
        加载COCO格式数据集
        需要annotations.json文件
        """
        # 查找标注文件
        annotation_files = list(self.root_dir.rglob("annotations.json"))
        if not annotation_files:
            annotation_files = list(self.root_dir.rglob("*.json"))

        if not annotation_files:
            raise ValueError(f"未找到COCO标注文件: {self.root_dir}")

        annotation_file = annotation_files[0]

        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # 建立图像ID到标注的映射
        image_id_to_annotations = {}
        for ann in coco_data.get('annotations', []):
            image_id = ann['image_id']
            if image_id not in image_id_to_annotations:
                image_id_to_annotations[image_id] = []
            image_id_to_annotations[image_id].append(ann)

        # 建立图像ID到文件名的映射
        image_id_to_filename = {}
        for img in coco_data.get('images', []):
            image_id_to_filename[img['id']] = img['file_name']

        # 查找图像目录
        image_dir = self.root_dir / 'images'
        if not image_dir.exists():
            image_dir = self.root_dir

        # 创建样本列表
        samples = []
        for image_id, filename in image_id_to_filename.items():
            image_path = image_dir / filename
            if image_path.exists():
                samples.append({
                    'image_path': str(image_path),
                    'annotations': image_id_to_annotations.get(image_id, []),
                    'image_id': Path(filename).stem
                })

        self.samples = samples
        self.coco_data = coco_data

    def _load_voc_format(self):
        """
        加载VOC格式数据集
        目录结构:
        root_dir/
        ├── JPEGImages/
        └── Annotations/
        """
        jpeg_dir = self.root_dir / 'JPEGImages'
        annotations_dir = self.root_dir / 'Annotations'

        if not jpeg_dir.exists():
            # 尝试其他可能的名称
            for possible_name in ['images', 'img', 'ImageSets']:
                jpeg_dir = self.root_dir / possible_name
                if jpeg_dir.exists():
                    break

        if not jpeg_dir.exists():
            jpeg_dir = self.root_dir

        if not annotations_dir.exists():
            annotations_dir = self.root_dir / 'annotations'
        if not annotations_dir.exists():
            annotations_dir = self.root_dir

        # 收集图像和对应的XML文件
        samples = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

        for ext in supported_extensions:
            for img_path in jpeg_dir.rglob(f"*{ext}"):
                xml_path = annotations_dir / f"{img_path.stem}.xml"
                samples.append({
                    'image_path': str(img_path),
                    'xml_path': str(xml_path) if xml_path.exists() else None,
                    'image_id': img_path.stem
                })

        self.samples = samples

    def _split_data(self, train_val_split: float):
        """划分训练集和验证集"""
        indices = list(range(len(self.samples)))
        random.seed(42)
        random.shuffle(indices)

        split_idx = int(len(indices) * train_val_split)

        if self.mode == "train":
            train_indices = set(indices[:split_idx])
            self.samples = [s for i, s in enumerate(self.samples) if i in train_indices]
        else:
            val_indices = set(indices[split_idx:])
            self.samples = [s for i, s in enumerate(self.samples) if i in val_indices]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            包含以下字段的字典:
                - image: 图像张量 [C, H, W]
                - boxes: 边界框 [N, 4] (x1, y1, x2, y2)，归一化到[0,1]
                - labels: 类别标签 [N]
                - image_id: 图像标识符
        """
        sample = self.samples[idx]

        # 加载图像
        try:
            image = Image.open(sample['image_path']).convert('RGB')
            img_width, img_height = image.size
        except Exception as e:
            # 如果加载失败，创建一个空白图像
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
            img_width, img_height = self.image_size, self.image_size

        # 加载标注
        boxes, labels = self._load_annotations(sample, img_width, img_height)

        # 应用数据增强
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        else:
            # 基本变换：调整大小并转换为Tensor
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(to_tensor(image))

        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': sample['image_id']
        }

    def _load_annotations(self, sample: Dict, img_width: int, img_height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载标注数据"""
        if self.format == "yolo":
            return self._load_yolo_annotations(sample, img_width, img_height)
        elif self.format == "coco":
            return self._load_coco_annotations(sample, img_width, img_height)
        elif self.format == "voc":
            return self._load_voc_annotations(sample, img_width, img_height)
        else:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

    def _load_yolo_annotations(self, sample: Dict, img_width: int, img_height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载YOLO格式标注"""
        label_path = sample.get('label_path')
        if not label_path or not Path(label_path).exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        boxes = []
        labels = []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                # YOLO格式: (center_x, center_y, width, height) 归一化坐标
                center_x = float(parts[1]) * img_width
                center_y = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                # 转换为 (x1, y1, x2, y2) 格式
                x1 = center_x - width / 2
                y1 = center_y - height / 2
                x2 = center_x + width / 2
                y2 = center_y + height / 2

                # 裁剪到图像范围内
                x1 = max(0, min(x1, img_width))
                y1 = max(0, min(y1, img_height))
                x2 = max(0, min(x2, img_width))
                y2 = max(0, min(y2, img_height))

                # 过滤无效框
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(class_id)

        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        # 归一化到[0,1]
        boxes[:, [0, 2]] /= img_width
        boxes[:, [1, 3]] /= img_height
        labels = torch.tensor(labels, dtype=torch.long)

        return boxes, labels

    def _load_coco_annotations(self, sample: Dict, img_width: int, img_height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载COCO格式标注"""
        annotations = sample.get('annotations', [])
        if not annotations:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        boxes = []
        labels = []

        for ann in annotations:
            if ann.get('iscrowd', 0) == 1:
                continue

            bbox = ann.get('bbox', [])
            if len(bbox) != 4:
                continue

            # COCO格式: (x, y, width, height)
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            # 裁剪到图像范围内
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann.get('category_id', 0) - 1)  # COCO类别ID从1开始

        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        # 归一化到[0,1]
        boxes[:, [0, 2]] /= img_width
        boxes[:, [1, 3]] /= img_height
        labels = torch.tensor(labels, dtype=torch.long)

        return boxes, labels

    def _load_voc_annotations(self, sample: Dict, img_width: int, img_height: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载VOC格式标注"""
        xml_path = sample.get('xml_path')
        if not xml_path or not Path(xml_path).exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        boxes = []
        labels = []

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 获取图像尺寸（XML中可能存储了原始尺寸）
            size = root.find('size')
            if size is not None:
                xml_width = int(size.find('width').text)
                xml_height = int(size.find('height').text)
            else:
                xml_width, xml_height = img_width, img_height

            # 用于缩放比例
            scale_x = img_width / xml_width if xml_width > 0 else 1
            scale_y = img_height / xml_height if xml_height > 0 else 1

            for obj in root.findall('object'):
                # 获取类别名称并转换为ID
                class_name = obj.find('name').text
                class_id = self._get_class_id_from_name(class_name)

                # 获取边界框
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text) * scale_x
                ymin = float(bndbox.find('ymin').text) * scale_y
                xmax = float(bndbox.find('xmax').text) * scale_x
                ymax = float(bndbox.find('ymax').text) * scale_y

                # 裁剪到图像范围内
                xmin = max(0, min(xmin, img_width))
                ymin = max(0, min(ymin, img_height))
                xmax = max(0, min(xmax, img_width))
                ymax = max(0, min(ymax, img_height))

                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)

        except Exception as e:
            pass

        if len(boxes) == 0:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        # 归一化到[0,1]
        boxes[:, [0, 2]] /= img_width
        boxes[:, [1, 3]] /= img_height
        labels = torch.tensor(labels, dtype=torch.long)

        return boxes, labels

    def _get_class_id_from_name(self, class_name: str) -> int:
        """将类别名称转换为ID"""
        # 尝试在COCO类别中查找
        if class_name in self.COCO_CLASSES:
            return self.COCO_CLASSES.index(class_name)

        # 使用哈希映射（保持一致性）
        return hash(class_name) % self.num_classes


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    检测数据集的collate函数
    处理不同样本中不同数量的边界框

    Args:
        batch: 批次数据

    Returns:
        合并后的批次数据
    """
    images = [item['image'] for item in batch]
    boxes = [item['boxes'] for item in batch]
    labels = [item['labels'] for item in batch]
    image_ids = [item['image_id'] for item in batch]

    # 堆叠图像
    images = torch.stack(images, dim=0)

    return {
        'images': images,
        'boxes': boxes,
        'labels': labels,
        'image_ids': image_ids
    }
