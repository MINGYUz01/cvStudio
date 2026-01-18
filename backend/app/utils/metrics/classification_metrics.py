"""
分类指标计算
提供准确率、精确率、召回率、F1分数等指标
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


class ClassificationMetrics:
    """
    分类指标计算器

    支持的指标：
    - Accuracy: 准确率
    - Top-K Accuracy: Top-K准确率
    - Precision: 精确率
    - Recall: 召回率
    - F1 Score: F1分数
    - Confusion Matrix: 混淆矩阵
    """

    def __init__(self, num_classes: Optional[int] = None):
        """
        初始化分类指标计算器

        Args:
            num_classes: 类别数（用于计算混淆矩阵）
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """重置累积统计"""
        self.correct = 0
        self.total = 0
        self.all_predictions = []
        self.all_targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        更新累积统计

        Args:
            predictions: 模型预测 [batch, num_classes] 或 [batch]
            targets: 真实标签 [batch]
        """
        # 如果预测是logits，获取预测类别
        if predictions.dim() > 1:
            pred = predictions.argmax(dim=1)
        else:
            pred = predictions

        # 更新统计
        self.correct += (pred == targets).sum().item()
        self.total += targets.size(0)

        # 保存用于计算其他指标
        self.all_predictions.append(pred.cpu())
        self.all_targets.append(targets.cpu())

    def compute(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        topk: Tuple[int, ...] = (1, 5)
    ) -> Dict[str, float]:
        """
        计算分类指标

        Args:
            predictions: 模型预测 [batch, num_classes]
            targets: 真实标签 [batch]
            topk: 计算Top-K准确率的K值列表

        Returns:
            指标字典
        """
        with torch.no_grad():
            # Top-1准确率
            if predictions.dim() > 1:
                pred = predictions.argmax(dim=1)
            else:
                pred = predictions

            correct = pred.eq(targets).sum().item()
            accuracy = correct / targets.size(0)

            result = {'accuracy': accuracy}

            # Top-K准确率
            if predictions.dim() > 1:
                maxk = max(topk)
                _, pred_topk = predictions.topk(maxk, dim=1, largest=True, sorted=True)
                pred_topk = pred_topk.t()
                correct_topk = pred_topk.eq(targets.view(1, -1).expand_as(pred_topk))

                for k in topk:
                    if k <= predictions.size(1):
                        correct_k = correct_topk[:k].reshape(-1).float().sum(0, keepdim=True).item()
                        result[f'top{k}_accuracy'] = correct_k / targets.size(0)

            # 计算精确率、召回率、F1
            prf = self._compute_precision_recall_f1(pred, targets)
            result.update(prf)

        return result

    def _compute_precision_recall_f1(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        num_classes: Optional[int] = None
    ) -> Dict[str, float]:
        """
        计算精确率、召回率、F1分数

        Args:
            predictions: 预测类别 [batch]
            targets: 真实类别 [batch]
            num_classes: 类别数

        Returns:
            包含precision, recall, f1的字典
        """
        try:
            from sklearn.metrics import precision_recall_fscore_support
        except ImportError:
            # 如果sklearn不可用，使用简单实现
            return self._compute_prf_simple(predictions, targets)

        pred = predictions.cpu().numpy()
        target = targets.cpu().numpy()

        n_classes = num_classes or self.num_classes or (max(target.max(), pred.max()) + 1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            target, pred, average='weighted', zero_division=0, labels=list(range(n_classes))
        )

        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }

    def _compute_prf_simple(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        简单的精确率、召回率、F1计算（不依赖sklearn）

        Args:
            predictions: 预测类别 [batch]
            targets: 真实类别 [batch]

        Returns:
            包含precision, recall, f1的字典
        """
        pred = predictions.cpu().numpy()
        target = targets.cpu().numpy()

        # 计算混淆矩阵
        n_classes = max(target.max(), pred.max()) + 1
        conf_matrix = [[0] * n_classes for _ in range(n_classes)]

        for p, t in zip(pred, target):
            conf_matrix[int(t)][int(p)] += 1

        # 计算每个类别的指标
        precisions = []
        recalls = []
        f1s = []

        for i in range(n_classes):
            # True Positive, False Positive, False Negative
            tp = conf_matrix[i][i]
            fp = sum(conf_matrix[j][i] for j in range(n_classes) if j != i)
            fn = sum(conf_matrix[i][j] for j in range(n_classes) if j != i)

            # 精确率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            precisions.append(precision)

            # 召回率
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            recalls.append(recall)

            # F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1s.append(f1)

        # 加权平均（按样本数加权）
        class_counts = [sum(conf_matrix[i]) for i in range(n_classes)]
        total = sum(class_counts)

        if total > 0:
            precision = sum(p * c / total for p, c in zip(precisions, class_counts))
            recall = sum(r * c / total for r, c in zip(recalls, class_counts))
            f1 = sum(f * c / total for f, c in zip(f1s, class_counts))
        else:
            precision = recall = f1 = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        获取累积的指标

        Returns:
            指标字典
        """
        if self.total == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }

        result = {'accuracy': self.correct / self.total}

        # 计算其他指标
        if self.all_predictions and self.all_targets:
            all_pred = torch.cat(self.all_predictions)
            all_target = torch.cat(self.all_targets)

            prf = self._compute_precision_recall_f1(all_pred, all_target)
            result.update(prf)

        return result

    def confusion_matrix(self) -> Optional[List[List[int]]]:
        """
        计算混淆矩阵

        Returns:
            混淆矩阵 [num_classes, num_classes]
        """
        if not self.all_predictions or not self.all_targets:
            return None

        all_pred = torch.cat(self.all_predictions).numpy()
        all_target = torch.cat(self.all_targets).numpy()

        n_classes = self.num_classes or max(all_target.max(), all_pred.max()) + 1

        conf_matrix = [[0] * n_classes for _ in range(n_classes)]

        for p, t in zip(all_pred, all_target):
            conf_matrix[int(t)][int(p)] += 1

        return conf_matrix


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    topk: int = 1
) -> float:
    """
    计算准确率

    Args:
        predictions: 模型预测 [batch, num_classes]
        targets: 真实标签 [batch]
        topk: K值

    Returns:
        准确率
    """
    with torch.no_grad():
        if topk == 1:
            pred = predictions.argmax(dim=1)
            correct = pred.eq(targets).sum().item()
        else:
            maxk = topk
            _, pred = predictions.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(targets.view(1, -1).expand_as(pred))
            correct = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()

        return correct / targets.size(0)
