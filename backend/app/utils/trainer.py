"""
训练执行器
负责执行实际的训练循环、日志收集、checkpoint保存等
"""

import asyncio
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from app.utils.training_logger import training_logger, TrainingStatus
from app.api.websocket import manager


class TrainingSignals:
    """训练控制信号"""
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"


class Trainer:
    """训练执行器"""

    def __init__(
        self,
        experiment_id: str,
        config: Dict[str, Any],
        celery_task: Optional[Any] = None
    ):
        """
        初始化训练执行器

        Args:
            experiment_id: 实验ID
            config: 训练配置字典
            celery_task: Celery任务实例（可选）
        """
        self.experiment_id = experiment_id
        self.config = config
        self.celery_task = celery_task

        # 训练状态
        self.current_epoch = 0
        self.total_epochs = config.get("epochs", 100)
        self.best_metric = 0.0
        self.signal = None  # 控制信号

        # 进程管理
        self.pid = None

        # 日志器
        self.logger = logger.bind(experiment_id=experiment_id)

        # 设备配置
        self.device = self._setup_device()

    def _setup_device(self) -> torch.device:
        """
        设置训练设备

        Returns:
            torch.device对象
        """
        device_str = self.config.get("device", "cpu")
        if device_str == "cuda" or device_str.startswith("cuda:"):
            if torch.cuda.is_available():
                device = torch.device(device_str)
                self.logger.info(f"使用GPU: {device_str}")
            else:
                self.logger.warning(f"CUDA不可用，使用CPU")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
            self.logger.info("使用CPU")

        return device

    def train(self) -> Dict[str, Any]:
        """
        执行训练

        Returns:
            训练结果字典

        Raises:
            Exception: 当训练失败时
        """
        try:
            self.logger.info(f"开始训练: {self.experiment_id}")

            # 记录开始时间
            start_time = datetime.utcnow()

            # 添加开始日志
            training_logger.add_log(
                self.experiment_id,
                "INFO",
                f"开始训练: {self.config.get('experiment_name', 'unnamed')}",
                "trainer"
            )

            # 执行训练循环
            for epoch in range(self.current_epoch, self.total_epochs):
                # 检查控制信号
                if self._check_signal():
                    break

                self.current_epoch = epoch

                # 执行一个epoch
                metrics = self._train_epoch(epoch)

                # 收集指标
                training_logger.add_metrics(
                    self.experiment_id,
                    epoch,
                    metrics
                )

                # 创建异步事件循环以广播
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # 广播指标更新
                metrics_entry = {
                    "epoch": epoch,
                    "timestamp": datetime.utcnow().isoformat(),
                    **metrics
                }
                loop.run_until_complete(
                    training_logger.broadcast_metrics(
                        self.experiment_id,
                        metrics_entry,
                        manager
                    )
                )

                # 保存checkpoint
                save_period = self.config.get("save_period", 10)
                if (epoch + 1) % save_period == 0 or epoch == self.total_epochs - 1:
                    self._save_checkpoint(epoch, metrics)

                # 更新进度
                progress = (epoch + 1) / self.total_epochs * 100
                self._update_progress(progress)

            # 训练完成
            end_time = datetime.utcnow()
            duration = str(end_time - start_time)

            result = {
                "status": "completed",
                "final_epoch": self.current_epoch + 1,
                "best_metric": self.best_metric,
                "duration": duration
            }

            # 更新状态为完成
            training_logger.update_status(
                self.experiment_id,
                TrainingStatus.COMPLETED
            )

            # 广播状态变化
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(
                training_logger.broadcast_status(self.experiment_id, manager)
            )

            # 添加完成日志
            training_logger.add_log(
                self.experiment_id,
                "INFO",
                f"训练完成! 最终epoch: {self.current_epoch + 1}, 最佳指标: {self.best_metric:.4f}",
                "trainer"
            )

            self.logger.info(f"训练完成: {self.experiment_id}")
            return result

        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            training_logger.update_status(
                self.experiment_id,
                TrainingStatus.FAILED
            )

            # 添加错误日志
            training_logger.add_log(
                self.experiment_id,
                "ERROR",
                f"训练失败: {str(e)}",
                "trainer"
            )

            raise

    def _train_epoch(self, epoch: int) -> Dict[str, Any]:
        """
        训练一个epoch

        Args:
            epoch: 当前epoch编号

        Returns:
            指标字典
        """
        # 添加日志
        training_logger.add_log(
            self.experiment_id,
            "INFO",
            f"开始 Epoch {epoch + 1}/{self.total_epochs}",
            "trainer"
        )

        # 这里是实际的训练逻辑
        # 在实际实现中，这里应该：
        # 1. 加载数据
        # 2. 前向传播
        # 3. 计算损失
        # 4. 反向传播
        # 5. 更新参数

        # 为了演示，这里返回模拟数据
        # 实际使用时需要替换为真实的训练代码
        import random

        # 模拟训练指标（随epoch改善）
        base_loss = 0.5 - (epoch * 0.005)  # loss逐渐降低
        base_acc = 0.5 + (epoch * 0.004)  # accuracy逐渐提升

        metrics = {
            "train_loss": max(0.01, base_loss + random.uniform(-0.01, 0.01)),
            "train_acc": min(0.99, base_acc + random.uniform(-0.02, 0.02)),
            "val_loss": max(0.02, base_loss + 0.02 + random.uniform(-0.01, 0.01)),
            "val_acc": min(0.97, base_acc - 0.02 + random.uniform(-0.02, 0.02)),
        }

        # 添加日志
        training_logger.add_log(
            self.experiment_id,
            "INFO",
            f"Epoch {epoch + 1}/{self.total_epochs} - "
            f"Loss: {metrics['train_loss']:.4f}, "
            f"Acc: {metrics['train_acc']:.4f}, "
            f"Val Loss: {metrics['val_loss']:.4f}, "
            f"Val Acc: {metrics['val_acc']:.4f}",
            "trainer"
        )

        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        """
        保存checkpoint

        Args:
            epoch: 当前epoch
            metrics: 指标字典
        """
        self.logger.info(f"保存checkpoint: epoch {epoch}")

        # 判断是否为最佳模型
        task_type = self.config.get("task_type", "detection")

        # 根据任务类型选择主要指标
        if task_type == "detection":
            current_metric = metrics.get("val_acc", 0.0)
        elif task_type == "classification":
            current_metric = metrics.get("val_acc", 0.0)
        else:  # segmentation
            current_metric = metrics.get("val_acc", 0.0)

        is_best = current_metric > self.best_metric

        if is_best:
            self.best_metric = current_metric
            self.logger.info(f"新最佳模型! {task_type.upper()}: {current_metric:.4f}")

        # 保存逻辑
        # 注意：这里应该调用CheckpointManager
        # 为了保持模块独立，这里只记录日志
        checkpoint_path = Path(self.config.get("checkpoint_dir", "data/checkpoints"))
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_path / f"epoch_{epoch}.pt"

        # 模拟保存checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": {},  # 实际使用时这里应该有模型状态
            "optimizer_state_dict": {},  # 实际使用时这里应该有优化器状态
            "metrics": metrics,
            "is_best": is_best,
            "timestamp": datetime.utcnow().isoformat()
        }, checkpoint_file)

        # 添加日志
        training_logger.add_log(
            self.experiment_id,
            "INFO",
            f"Checkpoint已保存: {checkpoint_file} (最佳: {is_best})",
            "trainer"
        )

    def _check_signal(self) -> bool:
        """
        检查控制信号

        Returns:
            True表示应该中断训练
        """
        if self.signal == TrainingSignals.PAUSE:
            self.logger.info("训练已暂停")

            # 更新状态
            training_logger.update_status(
                self.experiment_id,
                TrainingStatus.PAUSED
            )

            # 广播状态变化
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(
                training_logger.broadcast_status(self.experiment_id, manager)
            )

            # 等待恢复信号
            # 注意：在实际实现中，这里需要实现等待逻辑
            # 简化版本：直接返回True中断训练
            return True

        elif self.signal == TrainingSignals.STOP:
            self.logger.info("训练已停止")

            # 更新状态
            training_logger.update_status(
                self.experiment_id,
                TrainingStatus.STOPPED
            )

            # 广播状态变化
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(
                training_logger.broadcast_status(self.experiment_id, manager)
            )

            return True

        return False

    def _update_progress(self, progress: float):
        """
        更新训练进度

        Args:
            progress: 进度百分比（0-100）
        """
        # 更新数据库（实际使用时需要实现）
        # 这里只记录日志
        self.logger.debug(f"训练进度: {progress:.1f}%")

        # 广播状态更新（包含进度信息）
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(
            training_logger.broadcast_status(self.experiment_id, manager)
        )

    def pause(self):
        """暂停训练"""
        self.signal = TrainingSignals.PAUSE
        self.logger.info("收到暂停信号")

    def resume(self):
        """恢复训练"""
        self.signal = TrainingSignals.RESUME
        training_logger.update_status(
            self.experiment_id,
            TrainingStatus.RUNNING
        )
        self.logger.info("收到恢复信号")

    def stop(self):
        """停止训练"""
        self.signal = TrainingSignals.STOP
        self.logger.info("收到停止信号")

    def load_checkpoint(self, checkpoint_path: str):
        """
        加载checkpoint（断点续训）

        Args:
            checkpoint_path: checkpoint文件路径
        """
        try:
            checkpoint = torch.load(checkpoint_path)
            self.current_epoch = checkpoint.get("epoch", 0) + 1
            self.best_metric = checkpoint.get("best_metric", 0.0)

            self.logger.info(
                f"从checkpoint恢复: epoch {self.current_epoch}, "
                f"最佳指标: {self.best_metric:.4f}"
            )

            training_logger.add_log(
                self.experiment_id,
                "INFO",
                f"从checkpoint恢复: {checkpoint_path}",
                "trainer"
            )

        except Exception as e:
            self.logger.error(f"加载checkpoint失败: {e}")
            raise
