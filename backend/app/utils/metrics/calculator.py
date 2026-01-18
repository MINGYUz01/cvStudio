"""
指标计算器
统一的指标计算接口
"""

from typing import Dict, List, Any, Union
import torch


class MetricsCalculator:
    """
    指标计算器

    根据任务类型计算对应的评估指标
    """

    def __init__(self, task_type: str = "classification"):
        """
        初始化指标计算器

        Args:
            task_type: 任务类型 ("classification" | "detection")
        """
        self.task_type = task_type

        if task_type == "classification":
            from app.utils.metrics.classification_metrics import ClassificationMetrics
            self.metrics_calculator = ClassificationMetrics()
        elif task_type == "detection":
            from app.utils.metrics.detection_metrics import DetectionMetrics
            self.metrics_calculator = DetectionMetrics()
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

    def compute(
        self,
        predictions: Union[torch.Tensor, List],
        targets: Union[torch.Tensor, List],
        **kwargs
    ) -> Dict[str, float]:
        """
        计算指标

        Args:
            predictions: 模型预测
            targets: 真实标签
            **kwargs: 额外参数

        Returns:
            指标字典
        """
        return self.metrics_calculator.compute(predictions, targets, **kwargs)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        更新累积指标（用于批次处理）

        Args:
            predictions: 模型预测
            targets: 真实标签
        """
        if hasattr(self.metrics_calculator, 'update'):
            self.metrics_calculator.update(predictions, targets)

    def reset(self):
        """重置累积指标"""
        if hasattr(self.metrics_calculator, 'reset'):
            self.metrics_calculator.reset()

    def get_metrics(self) -> Dict[str, float]:
        """
        获取当前累积的指标

        Returns:
            指标字典
        """
        if hasattr(self.metrics_calculator, 'get_metrics'):
            return self.metrics_calculator.get_metrics()
        return {}


class AverageMeter:
    """
    平均值计算器
    用于追踪训练过程中的指标
    """

    def __init__(self):
        """初始化平均值计算器"""
        self.reset()

    def reset(self):
        """重置所有值"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        更新平均值

        Args:
            val: 新值
            n: 样本数
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class MetricsTracker:
    """
    指标追踪器
    用于追踪多个指标
    """

    def __init__(self, metrics: List[str]):
        """
        初始化指标追踪器

        Args:
            metrics: 要追踪的指标名称列表
        """
        self.meters = {name: AverageMeter() for name in metrics}

    def update(self, metrics: Dict[str, float], n: int = 1):
        """
        更新指标

        Args:
            metrics: 指标字典
            n: 样本数
        """
        for name, value in metrics.items():
            if name in self.meters:
                self.meters[name].update(value, n)

    def get_metrics(self) -> Dict[str, float]:
        """
        获取所有指标的平均值

        Returns:
            指标字典
        """
        return {name: meter.avg for name, meter in self.meters.items()}

    def reset(self):
        """重置所有指标"""
        for meter in self.meters.values():
            meter.reset()

    def __str__(self) -> str:
        """返回指标的字符串表示"""
        metrics_str = ", ".join([
            f"{name}: {meter.avg:.4f}"
            for name, meter in self.meters.items()
        ])
        return f"Metrics({metrics_str})"
