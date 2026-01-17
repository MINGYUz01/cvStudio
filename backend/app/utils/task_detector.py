"""
任务类型检测器
根据模型输出形状自动推断任务类型（分类/检测）
"""

from typing import Optional, Tuple, Any, List
from abc import ABC, abstractmethod
import torch
import numpy as np
from loguru import logger


class TaskDetector(ABC):
    """任务类型检测器基类"""

    @classmethod
    @abstractmethod
    def detect(cls, model: Any, device: str = "cpu") -> str:
        """
        检测模型任务类型

        Args:
            model: 模型对象
            device: 运行设备

        Returns:
            任务类型：classification/detection
        """
        pass

    @classmethod
    def detect_from_output_shape(cls, shape: Tuple) -> str:
        """
        根据输出形状推断任务类型

        Args:
            shape: 模型输出形状

        Returns:
            任务类型：classification/detection/unknown
        """
        ndim = len(shape)
        logger.bind(component="task_detector").debug(f"分析输出形状: {shape}, ndim={ndim}")

        # 处理batch维度
        if ndim == 1:
            # 一维输出，可能是单样本分类
            return "classification"
        elif ndim == 2:
            # [batch, num_classes] 或 [num_classes]
            batch_size, last_dim = shape if len(shape) == 2 else (1, shape[0])
            if last_dim <= 10000:  # 合理的类别数量
                return "classification"
            elif last_dim > 10000:
                # 可能是检测的扁平化输出
                return "detection"
        elif ndim == 3:
            # [batch, detections, features] 或 [batch, num_classes, 1]
            batch, second, third = shape

            # YOLO/COCO格式检测: [batch, detections, 6+] 或 [batch, detections, 85]
            if third >= 4 and third <= 100:  # 坐标+置信度+类别
                return "detection"

            # SSD格式: [batch, num_classes+4, num_anchors]
            if second >= 10 and third >= 100:
                return "detection"

            # 可能的分类: [batch, num_classes, 1]
            if third == 1 and second <= 10000:
                return "classification"

        elif ndim == 4:
            # [batch, num_classes, height, width] 或 [batch, height, width, num_classes]
            # 分类模型可能在某些情况下返回4维输出
            batch, dim1, dim2, dim3 = shape

            # 可能的分类: [batch, num_classes, height, width] 其中height=width=1
            if (dim1 <= 10000 and dim2 == 1 and dim3 == 1) or (dim3 <= 10000 and dim1 == 1 and dim2 == 1):
                return "classification"

            # 如果是较大维度的4维输出，可能是检测模型
            return "detection"

        logger.bind(component="task_detector").warning(f"无法确定任务类型，输出形状: {shape}")
        return "unknown"

    @classmethod
    def detect_from_output_content(cls, output: Any) -> str:
        """
        根据输出内容推断任务类型（分析数值范围、分布）

        Args:
            output: 模型输出

        Returns:
            任务类型：classification/detection
        """
        # 转换为numpy进行分析
        if isinstance(output, torch.Tensor):
            output_np = output.detach().cpu().numpy()
        else:
            output_np = np.array(output)

        shape = output_np.shape

        # 分析数值特征
        value_min = float(np.min(output_np))
        value_max = float(np.max(output_np))
        value_mean = float(np.mean(output_np))
        value_std = float(np.std(output_np))

        logger.bind(component="task_detector").debug(
            f"输出数值特征: min={value_min:.4f}, max={value_max:.4f}, "
            f"mean={value_mean:.4f}, std={value_std:.4f}"
        )

        # 分类输出通常经过softmax，值在[0,1]之间，和接近1
        if len(shape) == 2 or (len(shape) == 3 and shape[-1] == 1):
            if value_min >= 0 and value_max <= 1:
                row_sums = output_np.sum(axis=-1) if len(shape) == 2 else output_np.squeeze().sum()
                if np.allclose(row_sums, 1.0, atol=0.1) if isinstance(row_sums, (int, float)) else False:
                    return "classification"

        # 先根据形状判断
        return cls.detect_from_output_shape(shape)


class PyTorchTaskDetector(TaskDetector):
    """PyTorch模型任务检测器"""

    @classmethod
    def detect(cls, model: Any, device: str = "cpu", input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> str:
        """
        通过dummy推理检测PyTorch模型任务类型

        Args:
            model: PyTorch模型
            device: 运行设备
            input_size: 输入尺寸 (batch, channels, height, width)

        Returns:
            任务类型：classification/detection
        """
        logger.bind(component="task_detector").info("开始检测PyTorch模型任务类型")

        try:
            # 设置为评估模式
            if hasattr(model, 'eval'):
                model.eval()

            # 创建dummy输入
            dummy_input = torch.randn(*input_size).to(device)

            # 前向传播
            with torch.no_grad():
                output = model(dummy_input)

            # 处理输出
            if isinstance(output, (list, tuple)):
                output = output[0]

            # 获取输出形状
            if isinstance(output, torch.Tensor):
                output_shape = output.shape
            else:
                output_shape = output.shape if hasattr(output, 'shape') else ()

            logger.bind(component="task_detector").info(f"模型输出形状: {output_shape}")

            # 检测任务类型
            task_type = cls.detect_from_output_shape(output_shape)

            # 进一步验证
            if task_type == "unknown":
                task_type = cls.detect_from_output_content(output)

            logger.bind(component="task_detector").success(f"检测到任务类型: {task_type}")
            return task_type

        except Exception as e:
            logger.bind(component="task_detector").error(f"检测任务类型失败: {e}")
            return "unknown"


class ONNXTaskDetector(TaskDetector):
    """ONNX模型任务检测器"""

    @classmethod
    def detect(cls, model: Any, device: str = "cpu") -> str:
        """
        通过分析ONNX模型输出检测任务类型

        Args:
            model: ONNX InferenceSession
            device: 运行设备（未使用，ONNX有自己的提供者）

        Returns:
            任务类型：classification/detection
        """
        logger.bind(component="task_detector").info("开始检测ONNX模型任务类型")

        try:
            # 获取输出信息
            outputs = model.get_outputs()

            if not outputs:
                logger.warning("ONNX模型没有输出")
                return "unknown"

            # 分析第一个输出的形状
            output_shape = outputs[0].shape

            # 处理动态维度
            clean_shape = tuple(
                224 if dim == '?' else dim for dim in output_shape
            )

            logger.bind(component="task_detector").info(f"ONNX模型输出形状: {output_shape}")

            # 检测任务类型
            task_type = cls.detect_from_output_shape(clean_shape)

            logger.bind(component="task_detector").success(f"检测到任务类型: {task_type}")
            return task_type

        except Exception as e:
            logger.bind(component="task_detector").error(f"检测ONNX模型任务类型失败: {e}")
            return "unknown"


class TaskTypeDetector:
    """
    统一的任务类型检测器入口

    支持自动检测模型类型并调用对应的检测器
    """

    @staticmethod
    def detect(
        model: Any,
        model_type: str = "pytorch",
        device: str = "cpu",
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)
    ) -> str:
        """
        自动检测模型任务类型

        Args:
            model: 模型对象
            model_type: 模型类型 (pytorch/onnx)
            device: 运行设备
            input_size: 输入尺寸 (仅用于PyTorch)

        Returns:
            任务类型：classification/detection/unknown
        """
        if model_type == "pytorch":
            return PyTorchTaskDetector.detect(model, device, input_size)
        elif model_type == "onnx":
            return ONNXTaskDetector.detect(model, device)
        else:
            logger.warning(f"不支持的模型类型: {model_type}")
            return "unknown"

    @staticmethod
    def detect_from_shape(shape: Tuple) -> str:
        """
        根据输出形状快速推断任务类型

        Args:
            shape: 输出形状元组

        Returns:
            任务类型：classification/detection/unknown
        """
        return TaskDetector.detect_from_output_shape(shape)

    @staticmethod
    def get_preferred_input_size(task_type: str) -> Tuple[int, int]:
        """
        获取各任务类型推荐的输入尺寸

        Args:
            task_type: 任务类型

        Returns:
            (height, width) 元组
        """
        sizes = {
            "classification": (224, 224),
            "detection": (640, 640),
        }
        return sizes.get(task_type, (224, 224))

    @staticmethod
    def get_default_params(task_type: str) -> dict:
        """
        获取各任务类型的默认推理参数

        Args:
            task_type: 任务类型

        Returns:
            默认参数字典
        """
        params = {
            "classification": {
                "top_k": 5,
                "apply_softmax": True,
            },
            "detection": {
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "max_detections": 300,
            },
        }
        return params.get(task_type, {})


# 任务类型常量
TASK_TYPE_CLASSIFICATION = "classification"
TASK_TYPE_DETECTION = "detection"
TASK_TYPE_UNKNOWN = "unknown"

TASK_TYPES = [
    TASK_TYPE_CLASSIFICATION,
    TASK_TYPE_DETECTION,
]

TASK_TYPE_NAMES = {
    TASK_TYPE_CLASSIFICATION: "图像分类",
    TASK_TYPE_DETECTION: "目标检测",
    TASK_TYPE_UNKNOWN: "未知类型",
}
