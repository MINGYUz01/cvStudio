"""
后处理器系统
支持不同任务类型的后处理逻辑（分类/检测/分割）
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import cv2
from loguru import logger


class PostProcessor(ABC):
    """
    后处理器基类

    所有任务类型的后处理器都应继承此类并实现process方法
    """

    @abstractmethod
    def process(
        self,
        output: Any,
        image_info: Dict[str, Any],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        处理模型输出，返回标准化的结果格式

        Args:
            output: 模型原始输出
            image_info: 图像信息（包含原始尺寸、预处理参数等）
            **kwargs: 任务特定的参数

        Returns:
            标准化的结果列表
        """
        pass

    def _normalize_output(self, output: Any) -> np.ndarray:
        """将输出转换为numpy数组"""
        if isinstance(output, torch.Tensor):
            return output.detach().cpu().numpy()
        elif isinstance(output, np.ndarray):
            return output
        elif isinstance(output, (list, tuple)):
            return np.array(output)
        else:
            raise ValueError(f"不支持的输出类型: {type(output)}")


class DetectionPostProcessor(PostProcessor):
    """
    目标检测后处理器

    支持多种检测格式：
    - YOLO格式: [batch, detections, 6] -> [x1, y1, x2, y2, conf, cls]
    - COCO格式: [batch, detections, 85]
    - XYXY格式: 直接的边界框坐标
    """

    def process(
        self,
        output: Any,
        image_info: Dict[str, Any],
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        format: str = "auto",
        class_names: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        处理检测模型输出

        Args:
            output: 检测模型输出
            image_info: 图像信息
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值（用于NMS）
            format: 输出格式 (auto/yolo/coco/xyxy)
            class_names: 类别名称列表

        Returns:
            检测结果列表
        """
        logger.bind(component="detection_postprocessor").debug("开始处理检测结果")

        # 转换为numpy
        output_array = self._normalize_output(output)

        # 移除batch维度
        if len(output_array.shape) == 3:
            output_array = output_array[0]

        original_size = image_info.get('original_size', (640, 640))
        input_size = image_info.get('input_size', (640, 640))

        # 自动检测格式并处理
        detections = self._extract_detections(output_array, format)

        if detections is None or len(detections) == 0:
            logger.bind(component="detection_postprocessor").debug("没有检测到目标")
            return []

        # 过滤低置信度
        if len(detections) > 0 and detections.shape[1] >= 5:
            conf_mask = detections[:, 4] >= confidence_threshold
            detections = detections[conf_mask]

        if len(detections) == 0:
            return []

        # 坐标缩放到原图尺寸
        detections = self._scale_coordinates(detections, input_size, original_size)

        # NMS
        detections = self._nms(detections, iou_threshold)

        # 转换为结果格式
        results = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            conf = float(det[4])
            cls_id = int(det[5]) if len(det) > 5 else 0

            label = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"

            results.append({
                'type': 'detection',
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'label': label,
                'class_id': cls_id,
                'confidence': float(conf)
            })

        logger.bind(component="detection_postprocessor").success(f"处理完成，检测到{len(results)}个目标")
        return results

    def _extract_detections(self, output: np.ndarray, format: str) -> Optional[np.ndarray]:
        """从输出中提取检测框"""
        shape = output.shape

        if format == "auto":
            # 自动检测格式
            if len(shape) == 2:
                # [detections, 6+]
                if shape[1] >= 6:
                    return output
            elif len(shape) == 3:
                # 选择第一个输出
                if shape[1] > 0 and shape[2] >= 6:
                    return output[0]
                # 或转置
                if shape[2] > 0 and shape[1] >= 6:
                    return output.transpose(0, 2, 1)[0]
        elif format == "yolo":
            # YOLO格式: [batch, detections, 6]
            if len(shape) == 3:
                return output[0]
            elif len(shape) == 2:
                return output
        elif format == "coco":
            # COCO格式: [batch, detections, 85]
            if len(shape) == 3:
                return output[0]
            elif len(shape) == 2:
                return output

        return output if len(output) > 0 else None

    def _scale_coordinates(
        self,
        detections: np.ndarray,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int]
    ) -> np.ndarray:
        """将坐标从输入尺寸缩放到原始图像尺寸"""
        scale_x = original_size[0] / input_size[0]
        scale_y = original_size[1] / input_size[1]

        detections[:, 0] *= scale_x  # x1
        detections[:, 1] *= scale_y  # y1
        detections[:, 2] *= scale_x  # x2
        detections[:, 3] *= scale_y  # y2

        return detections

    def _nms(
        self,
        detections: np.ndarray,
        iou_threshold: float
    ) -> np.ndarray:
        """
        非极大值抑制

        Args:
            detections: [N, 6+] 数组，每行为 [x1, y1, x2, y2, conf, cls, ...]
            iou_threshold: IOU阈值

        Returns:
            NMS后的检测结果
        """
        if len(detections) == 0:
            return detections

        # 按置信度排序
        confidences = detections[:, 4]
        indices = np.argsort(confidences)[::-1]

        keep = []
        while len(indices) > 0:
            # 保留最高置信度的框
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            # 计算IOU
            ious = self._calculate_iou(
                detections[current][:4],
                detections[indices[1:]][:, :4]
            )

            # 保留IOU小于阈值的框
            indices = indices[1:][ious <= iou_threshold]

        return detections[keep]

    def _calculate_iou(
        self,
        box: np.ndarray,
        boxes: np.ndarray
    ) -> np.ndarray:
        """计算一个框与多个框的IOU"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = box_area + boxes_area - intersection

        return intersection / (union + 1e-6)


class ClassificationPostProcessor(PostProcessor):
    """
    图像分类后处理器

    处理分类模型的输出：
    - 应用softmax（如果需要）
    - 返回Top-K结果
    - 类别名称映射
    """

    def process(
        self,
        output: Any,
        image_info: Dict[str, Any],
        top_k: int = 5,
        class_names: Optional[List[str]] = None,
        apply_softmax: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        处理分类模型输出

        Args:
            output: 分类模型输出
            image_info: 图像信息
            top_k: 返回前K个结果
            class_names: 类别名称列表
            apply_softmax: 是否应用softmax

        Returns:
            分类结果列表
        """
        logger.bind(component="classification_postprocessor").debug("开始处理分类结果")

        # 转换为numpy
        output_array = self._normalize_output(output)

        # 移除多余的维度
        output_array = output_array.flatten()

        # 应用softmax
        if apply_softmax:
            probabilities = self._softmax(output_array)
        else:
            # 假设已经是概率
            probabilities = output_array
            # 确保非负
            probabilities = np.maximum(probabilities, 0)
            # 归一化
            if probabilities.sum() > 0:
                probabilities = probabilities / probabilities.sum()

        # 获取top-k索引
        top_k = min(top_k, len(probabilities))
        top_indices = np.argsort(probabilities)[-top_k:][::-1]

        # 转换为结果格式
        results = []
        for idx in top_indices:
            prob = float(probabilities[idx])
            label = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"

            results.append({
                'type': 'classification',
                'label': label,
                'class_id': int(idx),
                'confidence': prob
            })

        logger.bind(component="classification_postprocessor").success(
            f"处理完成，Top-1: {results[0]['label']} ({results[0]['confidence']:.4f})"
        )
        return results

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """计算softmax"""
        exp_x = np.exp(x - np.max(x))  # 数值稳定性
        return exp_x / exp_x.sum()


class SegmentationPostProcessor(PostProcessor):
    """
    语义分割后处理器

    处理分割模型的输出：
    - Argmax获取类别
    - 生成掩码
    - 计算各类别占比
    """

    def process(
        self,
        output: Any,
        image_info: Dict[str, Any],
        threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
        return_mask: bool = True,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        处理分割模型输出

        Args:
            output: 分割模型输出 [batch, num_classes, height, width] 或 [batch, height, width, num_classes]
            image_info: 图像信息
            threshold: 置信度阈值
            class_names: 类别名称列表
            return_mask: 是否返回掩码数据

        Returns:
            分割结果列表，每个类别一个结果
        """
        logger.bind(component="segmentation_postprocessor").debug("开始处理分割结果")

        # 转换为numpy
        output_array = self._normalize_output(output)

        # 移除batch维度
        if len(output_array.shape) == 4:
            output_array = output_array[0]

        # 检测格式: CHW 还是 HWC
        is_chw = output_array.shape[0] < output_array.shape[-1]

        if is_chw:
            # [num_classes, height, width]
            num_classes, height, width = output_array.shape
            # Argmax获取类别
            class_map = np.argmax(output_array, axis=0)
        else:
            # [height, width, num_classes]
            height, width, num_classes = output_array.shape
            class_map = np.argmax(output_array, axis=2)

        # 计算各类别的占比
        unique, counts = np.unique(class_map, return_counts=True)
        total_pixels = height * width

        results = []
        for class_id, count in zip(unique, counts):
            area_percentage = (count / total_pixels) * 100

            # 获取该类别的平均置信度
            if is_chw:
                class_probs = output_array[class_id]
            else:
                class_probs = output_array[:, :, class_id]

            avg_confidence = float(class_probs[class_map == class_id].mean()) if count > 0 else 0.0

            label = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"

            results.append({
                'type': 'segmentation',
                'label': label,
                'class_id': int(class_id),
                'confidence': avg_confidence,
                'area_percentage': area_percentage,
                'pixel_count': int(count)
            })

        # 按占比排序
        results.sort(key=lambda x: x['area_percentage'], reverse=True)

        logger.bind(component="segmentation_postprocessor").success(
            f"处理完成，检测到{len(results)}个类别"
        )

        return results


class PostProcessorFactory:
    """
    后处理器工厂

    根据任务类型返回对应的后处理器
    """

    _processors = {
        'detection': DetectionPostProcessor(),
        'classification': ClassificationPostProcessor(),
        'segmentation': SegmentationPostProcessor(),
    }

    @classmethod
    def get_processor(cls, task_type: str) -> PostProcessor:
        """
        获取对应任务类型的后处理器

        Args:
            task_type: 任务类型 (detection/classification/segmentation)

        Returns:
            对应的后处理器实例

        Raises:
            ValueError: 不支持的任务类型
        """
        processor = cls._processors.get(task_type)
        if not processor:
            raise ValueError(f"不支持的任务类型: {task_type}")
        return processor

    @classmethod
    def register_processor(cls, task_type: str, processor: PostProcessor):
        """
        注册新的后处理器

        Args:
            task_type: 任务类型名称
            processor: 后处理器实例
        """
        cls._processors[task_type] = processor
        logger.bind(component="postprocessor_factory").info(f"注册后处理器: {task_type}")


# 导出
__all__ = [
    'PostProcessor',
    'DetectionPostProcessor',
    'ClassificationPostProcessor',
    'SegmentationPostProcessor',
    'PostProcessorFactory',
]
