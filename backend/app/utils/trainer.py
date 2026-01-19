"""
训练执行器
负责执行实际的训练循环、日志收集、checkpoint保存等
"""

import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from loguru import logger

from app.utils.training_logger import training_logger, TrainingStatus
from app.api.websocket import manager
from app.database import SessionLocal
from app.models.training import TrainingRun

# 导入训练组件
from app.utils.data_loaders.factory import DataLoaderFactory
from app.utils.models.factory import ModelFactory
from app.utils.losses.factory import LossFactory
from app.utils.metrics.calculator import MetricsCalculator


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

        # 训练组件（延迟初始化）
        self._initialized = False
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        self.train_loader = None
        self.val_loader = None
        self.metrics_calculator: Optional[MetricsCalculator] = None

        # 获取任务类型
        self.task_type = config.get("task_type", "classification")

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

    def _setup_training(self):
        """
        设置训练所需的所有组件
        包括数据加载器、模型、优化器、损失函数、学习率调度器、指标计算器
        """
        self.logger.info("开始初始化训练组件...")

        # 1. 创建数据加载器
        try:
            self.train_loader, self.val_loader = DataLoaderFactory.create({
                'task_type': self.task_type,
                'dataset_format': self.config.get('dataset_format', 'classification'),
                'dataset_path': self.config.get('dataset_path'),
                'batch_size': self.config.get('batch_size', 32),
                'image_size': self.config.get('image_size', 224),
                'num_classes': self.config.get('num_classes', 10),
                'augmentation': self.config.get('augmentation', {}),
                'train_val_split': self.config.get('train_val_split', 0.8),
                'num_workers': self.config.get('num_workers', 4),
                'pin_memory': self.config.get('pin_memory', True),
                'device': str(self.device)
            })
            self.logger.info(f"数据加载器创建成功: 训练集{len(self.train_loader)}批次, 验证集{len(self.val_loader)}批次")
        except Exception as e:
            self.logger.error(f"数据加载器创建失败: {e}")
            raise

        # 2. 创建模型
        try:
            # 优先使用 model_arch_info（包含 code_path 等完整信息）
            model_arch_info = self.config.get('model_arch_info')

            model_config = {
                'task_type': self.task_type,
                'num_classes': self.config.get('num_classes', 10),
                'input_channels': self.config.get('input_channels', 3),
                'pretrained': self.config.get('pretrained', False),
                'dropout': self.config.get('dropout', 0.0)
            }

            # 如果有完整的模型架构信息且有 code_path，使用动态加载
            if model_arch_info and model_arch_info.get('code_path'):
                model_config['architecture'] = model_arch_info  # 传递完整的架构信息（包含 code_path）
                self.logger.info(f"使用自定义模型: {model_arch_info.get('code_path')}")
            else:
                # 降级处理：使用 model_architecture（用于预设模型）
                model_config['architecture'] = self.config.get('model_architecture', 'resnet18')
                self.logger.info(f"使用预设模型: {model_config['architecture']}")

            self.model = ModelFactory.create(model_config)
            self.model.to(self.device)
            self.logger.info(f"模型创建成功并已加载到设备: {self.device}")
        except Exception as e:
            self.logger.error(f"模型创建失败: {e}")
            raise

        # 3. 创建优化器
        try:
            self.optimizer = self._create_optimizer()
            self.logger.info(f"优化器创建成功: {self.config.get('optimizer', 'adam')}")
        except Exception as e:
            self.logger.error(f"优化器创建失败: {e}")
            raise

        # 4. 创建损失函数
        try:
            self.criterion = LossFactory.create(self.task_type, self.config)
            self.criterion.to(self.device)
            self.logger.info("损失函数创建成功")
        except Exception as e:
            self.logger.error(f"损失函数创建失败: {e}")
            raise

        # 5. 创建学习率调度器
        try:
            scheduler_type = self.config.get('scheduler', 'none')
            if scheduler_type and scheduler_type.lower() != 'none':
                self.scheduler = self._create_scheduler()
                self.logger.info(f"学习率调度器创建成功: {scheduler_type}")
        except Exception as e:
            self.logger.warning(f"学习率调度器创建失败: {e}，将不使用调度器")

        # 6. 创建指标计算器
        self.metrics_calculator = MetricsCalculator(self.task_type)
        self.logger.info("指标计算器创建成功")

        # 标记初始化完成
        self._initialized = True
        self.logger.info("训练组件初始化完成")

    def _create_optimizer(self) -> optim.Optimizer:
        """
        创建优化器

        Returns:
            PyTorch优化器
        """
        optimizer_type = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0.0)
        momentum = self.config.get('momentum', 0.9)

        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay if weight_decay > 0 else 0.0005
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")

    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """
        创建学习率调度器

        Returns:
            PyTorch学习率调度器
        """
        scheduler_config = self.config.get('scheduler', 'step')

        # 兼容两种格式：字符串或字典
        if isinstance(scheduler_config, dict):
            scheduler_type = scheduler_config.get('type', 'step').lower()
        else:
            scheduler_type = str(scheduler_config).lower()

        # 处理 None 值
        if not scheduler_type or scheduler_type == 'none':
            raise ValueError("调度器类型为空")

        if scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_epochs
            )
        elif scheduler_type == 'reduce_on_plateau' or scheduler_type == 'reduceonplateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=10
            )
        else:
            raise ValueError(f"不支持的调度器: {scheduler_type}")

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

            # 更新数据库状态为running
            try:
                training_run_id = int(self.experiment_id.split('_')[1])
                db = SessionLocal()
                try:
                    training_run = db.query(TrainingRun).filter(
                        TrainingRun.id == training_run_id
                    ).first()
                    if training_run:
                        training_run.status = "running"
                        training_run.start_time = datetime.utcnow()
                        db.commit()
                        self.logger.info(f"数据库状态已更新为running: id={training_run_id}")
                finally:
                    db.close()
            except Exception as e:
                self.logger.error(f"更新数据库运行状态失败: {e}")

            # 初始化训练组件（在第一个epoch时）
            if not self._initialized:
                self._setup_training()

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

                # 收集指标（包含最佳指标）
                training_logger.add_metrics(
                    self.experiment_id,
                    epoch,
                    metrics,
                    best_metric=self.best_metric
                )

                # 创建异步事件循环以广播
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                # 广播指标更新（包含最佳指标）
                metrics_entry = {
                    "epoch": epoch,
                    "timestamp": datetime.utcnow().isoformat(),
                    "best_metric": self.best_metric,  # 添加最佳指标
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
                "final_epoch": self.current_epoch + 1,  # 显示用
                "best_metric": self.best_metric,
                "duration": duration
            }

            # 更新数据库中的训练状态
            try:
                training_run_id = int(self.experiment_id.split('_')[1])
                db = SessionLocal()
                try:
                    training_run = db.query(TrainingRun).filter(
                        TrainingRun.id == training_run_id
                    ).first()
                    if training_run:
                        training_run.status = "completed"
                        training_run.end_time = end_time
                        training_run.progress = 100.0
                        # 保持 0-based，让前端负责转换为 1-based 显示
                        training_run.current_epoch = self.current_epoch
                        training_run.best_metric = self.best_metric
                        db.commit()
                        self.logger.info(f"数据库状态已更新为completed: id={training_run_id}, best_metric={self.best_metric:.4f}")
                finally:
                    db.close()
            except Exception as e:
                self.logger.error(f"更新数据库完成状态失败: {e}")

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
                training_logger.broadcast_status(self.experiment_id, manager, best_metric=self.best_metric)
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

            # 更新数据库中的训练状态
            try:
                training_run_id = int(self.experiment_id.split('_')[1])
                db = SessionLocal()
                try:
                    training_run = db.query(TrainingRun).filter(
                        TrainingRun.id == training_run_id
                    ).first()
                    if training_run:
                        training_run.status = "failed"
                        training_run.end_time = datetime.utcnow()
                        training_run.error_message = str(e)
                        db.commit()
                finally:
                    db.close()
            except Exception as db_error:
                self.logger.error(f"更新数据库失败状态失败: {db_error}")

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

        # 设置模型为训练模式
        self.model.train()

        # 训练统计
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 训练循环
        for batch_idx, batch in enumerate(self.train_loader):
            # 根据任务类型解包批次数据
            if self.task_type == "classification":
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
            elif self.task_type == "detection":
                images = batch['images'].to(self.device)
                labels = batch['labels']
                # 检测任务的处理比较复杂，暂时使用简化版本
                continue
            else:
                self.logger.warning(f"不支持的任务类型: {self.task_type}")
                continue

            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # 计算损失
            if self.task_type == "classification":
                loss = self.criterion(outputs, labels)

                # 统计准确率
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 更新参数
            self.optimizer.step()

            # 累积损失
            train_loss += loss.item()

        # 计算平均训练指标
        avg_train_loss = train_loss / len(self.train_loader)
        train_accuracy = train_correct / train_total if train_total > 0 else 0.0

        # 验证
        val_metrics = self._validate_epoch(epoch)

        # 组合指标
        metrics = {
            "train_loss": avg_train_loss,
            "train_acc": train_accuracy,
            **val_metrics
        }

        # 更新最佳指标（每个epoch都检查）
        task_type = self.config.get("task_type", "classification")
        if task_type == "detection":
            current_metric = metrics.get("val_acc", 0.0)
        else:
            current_metric = metrics.get("val_acc", 0.0)

        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.logger.info(f"新最佳指标! Val Acc: {current_metric:.4f}")

        # 更新学习率调度器
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics.get("val_loss", avg_train_loss))
            else:
                self.scheduler.step()

        # 添加日志
        if self.task_type == "classification":
            training_logger.add_log(
                self.experiment_id,
                "INFO",
                f"Epoch {epoch + 1}/{self.total_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {metrics.get('val_loss', 0):.4f}, "
                f"Val Acc: {metrics.get('val_acc', 0):.4f}",
                "trainer"
            )

        return metrics

    def _validate_epoch(self, epoch: int) -> Dict[str, Any]:
        """
        验证一个epoch

        Args:
            epoch: 当前epoch编号

        Returns:
            验证指标字典
        """
        # 设置模型为评估模式
        self.model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                # 根据任务类型解包批次数据
                if self.task_type == "classification":
                    images = batch['image'].to(self.device)
                    labels = batch['label'].to(self.device)
                elif self.task_type == "detection":
                    continue  # 检测任务暂不实现验证
                else:
                    continue

                # 前向传播
                outputs = self.model(images)

                # 计算损失
                if self.task_type == "classification":
                    loss = self.criterion(outputs, labels)

                    # 统计准确率
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

                val_loss += loss.item()

        # 计算平均验证指标
        avg_val_loss = val_loss / len(self.val_loader)
        val_accuracy = val_correct / val_total if val_total > 0 else 0.0

        return {
            "val_loss": avg_val_loss,
            "val_acc": val_accuracy
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        """
        保存checkpoint

        Args:
            epoch: 当前epoch
            metrics: 指标字典
        """
        self.logger.info(f"保存checkpoint: epoch {epoch}")

        # 判断是否为最佳模型
        task_type = self.task_type

        # 根据任务类型选择主要指标
        if task_type == "detection":
            current_metric = metrics.get("val_acc", 0.0)  # 检测任务暂时使用准确率
        else:
            current_metric = metrics.get("val_acc", 0.0)

        is_best = current_metric > self.best_metric

        if is_best:
            self.best_metric = current_metric
            self.logger.info(f"新最佳模型! Val Acc: {current_metric:.4f}")

        # 保存逻辑
        checkpoint_path = Path(self.config.get("checkpoint_dir", "data/checkpoints"))
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        checkpoint_file = checkpoint_path / f"epoch_{epoch}.pt"
        best_file = checkpoint_path / "best_model.pt"

        # 准备保存的内容
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict() if self.model is not None else {},
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else {},
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "metrics": metrics,
            "best_metric": self.best_metric,
            "config": self.config,
            "timestamp": datetime.utcnow().isoformat()
        }

        # 保存当前epoch的checkpoint
        torch.save(checkpoint, checkpoint_file)

        # 如果是最佳模型，额外保存
        if is_best:
            torch.save(checkpoint, best_file)

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
                training_logger.broadcast_status(self.experiment_id, manager, best_metric=self.best_metric)
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
                training_logger.broadcast_status(self.experiment_id, manager, best_metric=self.best_metric)
            )

            return True

        return False

    def _update_progress(self, progress: float):
        """
        更新训练进度

        Args:
            progress: 进度百分比（0-100）
        """
        # 从experiment_id提取training_run_id (格式: exp_1)
        try:
            training_run_id = int(self.experiment_id.split('_')[1])
            db = SessionLocal()
            try:
                training_run = db.query(TrainingRun).filter(
                    TrainingRun.id == training_run_id
                ).first()
                if training_run:
                    training_run.progress = progress
                    training_run.current_epoch = self.current_epoch
                    training_run.best_metric = self.best_metric
                    db.commit()
                    self.logger.info(f"数据库已更新: progress={progress:.1f}%, epoch={self.current_epoch}, best_metric={self.best_metric:.4f}")
                else:
                    self.logger.warning(f"未找到training_run: id={training_run_id}")
            finally:
                db.close()
        except Exception as e:
            self.logger.error(f"更新数据库进度失败: {e}")

        # 广播状态更新（包含进度信息）
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(
            training_logger.broadcast_status(self.experiment_id, manager, best_metric=self.best_metric)
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
