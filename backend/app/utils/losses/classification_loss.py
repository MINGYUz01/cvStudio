"""
分类损失函数
支持交叉熵损失、标签平滑等功能
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """
    分类损失函数

    支持的功能：
    - 交叉熵损失
    - 标签平滑
    - 类别权重
    - Focal Loss（可选）
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        focal_loss: bool = False,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        初始化分类损失

        Args:
            label_smoothing: 标签平滑系数 (0-1)
            class_weights: 类别权重张量 [num_classes]
            focal_loss: 是否使用Focal Loss
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
        """
        super().__init__()

        self.label_smoothing = label_smoothing
        self.class_weights = class_weights
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # 创建基础损失函数
        self.cross_entropy = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing if label_smoothing > 0 else 0.0,
            reduction='none'
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算分类损失

        Args:
            predictions: 模型预测 [batch, num_classes]
            targets: 真实标签 [batch]

        Returns:
            损失值（标量）
        """
        if self.focal_loss:
            return self._focal_loss(predictions, targets)
        else:
            loss = self.cross_entropy(predictions, targets)
            return loss.mean()

    def _focal_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算Focal Loss

        Args:
            predictions: 模型预测 [batch, num_classes]
            targets: 真实标签 [batch]

        Returns:
            Focal Loss值
        """
        # 获取概率
        probs = F.softmax(predictions, dim=1)

        # 创建one-hot编码
        num_classes = predictions.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes).float()

        # 计算交叉熵
        ce = F.cross_entropy(predictions, targets, reduction='none')

        # 获取正确类别的概率
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # 计算Focal Loss
        focal_weight = (1 - pt) ** self.focal_gamma
        focal_loss = self.focal_alpha * focal_weight * ce

        # 应用类别权重
        if self.class_weights is not None:
            weights = self.class_weights[targets]
            focal_loss = focal_loss * weights

        return focal_loss.mean()

    def get_class_weights(
        self,
        targets: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        根据标签分布计算类别权重

        Args:
            targets: 标签张量 [N]
            num_classes: 类别数

        Returns:
            类别权重张量 [num_classes]
        """
        from collections import Counter

        # 统计每个类别的样本数
        counts = Counter(targets.cpu().tolist())
        total = len(targets)

        # 计算权重: total / (num_classes * count)
        weights = torch.zeros(num_classes)
        for i in range(num_classes):
            count = counts.get(i, 1)
            weights[i] = total / (num_classes * count)

        return weights.to(targets.device)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    标签平滑交叉熵损失

    实现方式：
    - 将硬标签 (1, 0, 0) 转换为软标签 (1-eps, eps/(K-1), eps/(K-1))
    - 然后计算交叉熵
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        weight: Optional[torch.Tensor] = None
    ):
        """
        初始化标签平滑损失

        Args:
            smoothing: 平滑系数 (0-1)
            weight: 类别权重
        """
        super().__init__()

        self.smoothing = smoothing
        self.weight = weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算标签平滑交叉熵

        Args:
            predictions: 模型预测 [batch, num_classes]
            targets: 真实标签 [batch]

        Returns:
            损失值
        """
        num_classes = predictions.shape[-1]
        log_probs = F.log_softmax(predictions, dim=-1)

        # 创建软标签
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        # 计算损失
        loss = -torch.sum(smooth_targets * log_probs, dim=-1)

        # 应用类别权重
        if self.weight is not None:
            weight = self.weight[targets]
            loss = loss * weight

        return loss.mean()


class ArcFaceLoss(nn.Module):
    """
    ArcFace损失函数
    用于人脸识别等需要判别性特征的任务
    """

    def __init__(
        self,
        num_classes: int,
        embedding_size: int,
        scale: float = 64.0,
        margin: float = 0.5
    ):
        """
        初始化ArcFace损失

        Args:
            num_classes: 类别数
            embedding_size: 特征维度
            scale: 缩放因子
            margin: 角度边际
        """
        super().__init__()

        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.margin = margin

        # 权重参数
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_size)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算ArcFace损失

        Args:
            embeddings: 特征嵌入 [batch, embedding_size]
            targets: 真实标签 [batch]

        Returns:
            ArcFace损失值
        """
        # 归一化特征和权重
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)

        # 计算logits
        logits = F.linear(embeddings, weights)
        logits = logits * self.scale

        # ArcFace角度调整
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))

        # 为正确类别添加边际
        one_hot = F.one_hot(targets, self.num_classes)
        margin_logits = torch.cos(theta + self.margin * one_hot)

        # 合并logits
        logits = logits * (1 - one_hot) + margin_logits * one_hot
        logits = logits * self.scale

        # 计算交叉熵
        loss = F.cross_entropy(logits, targets)

        return loss


def create_classification_loss(
    label_smoothing: float = 0.0,
    class_weights: Optional[torch.Tensor] = None,
    focal_loss: bool = False
) -> ClassificationLoss:
    """
    创建分类损失函数

    Args:
        label_smoothing: 标签平滑系数
        class_weights: 类别权重
        focal_loss: 是否使用Focal Loss

    Returns:
        分类损失函数实例
    """
    return ClassificationLoss(
        label_smoothing=label_smoothing,
        class_weights=class_weights,
        focal_loss=focal_loss
    )
