"""
训练服务
提供训练任务的CRUD操作和控制逻辑
"""

from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from loguru import logger
from pathlib import Path
from datetime import datetime
import sys

from app.models.training import TrainingRun
from app.models.augmentation import AugmentationStrategy
from app.utils.config_parser import TrainingConfigParser
from app.utils.checkpoint_manager import CheckpointManager
from app.utils.training_logger import training_logger, TrainingStatus
from app.utils.experiment_manager import ExperimentManager


def debug_log(msg: str, level: str = "INFO"):
    """输出调试日志到控制台和文件"""
    log_msg = f"[TRAINING_SERVICE] {msg}"
    print(f"[DEBUG] {log_msg}", flush=True)  # 强制输出到控制台
    if level == "INFO":
        logger.info(log_msg)
    elif level == "ERROR":
        logger.error(log_msg)
    elif level == "WARNING":
        logger.warning(log_msg)


def _clean_model_name(name: str) -> str:
    """
    清理模型名称中的版本后缀

    例如: "ResNet50 v1.0" -> "ResNet50"
         "YOLOv8 v2.1.0" -> "YOLOv8"

    Args:
        name: 原始模型名称

    Returns:
        清理后的模型名称
    """
    import re
    # 匹配版本模式: v + 数字格式 (如 v1.0, v2.1.0)
    pattern = r'\s+v\d+(?:\.\d+)*\.?\s*$'
    cleaned = re.sub(pattern, '', name)
    return cleaned.strip()


class TrainingService:
    """训练服务"""

    def __init__(self):
        """初始化训练服务"""
        self.config_parser = TrainingConfigParser()
        self.logger = logger.bind(component="training_service")
        debug_log("训练服务已初始化")

    def create_training_run(
        self,
        db: Session,
        name: str,
        description: str,
        model_id: int,
        dataset_id: int,
        config: Dict[str, Any],
        user_id: int,
        pretrained_weight_id: Optional[int] = None
    ) -> TrainingRun:
        """
        创建训练任务

        Args:
            db: 数据库会话
            name: 训练任务名称
            description: 描述
            model_id: 模型ID
            dataset_id: 数据集ID
            config: 训练配置
            user_id: 创建用户ID
            pretrained_weight_id: 预训练权重ID

        Returns:
            创建的训练任务对象

        Raises:
            ValueError: 当参数无效时
        """
        try:
            # 创建数据库记录
            training_run = TrainingRun(
                name=name,
                description=description,
                model_id=model_id,
                dataset_id=dataset_id,
                hyperparams=config,
                status="pending",
                device=config.get("device", "cpu"),
                total_epochs=config.get("epochs", 100),
                current_epoch=0,
                progress=0.0,
                created_by=user_id,
                pretrained_weight_id=pretrained_weight_id
            )

            db.add(training_run)
            db.commit()
            db.refresh(training_run)

            # 创建实验目录
            experiment_manager = ExperimentManager(training_run.id)
            experiment_manager.create_experiment_dir()
            experiment_dir_path = str(experiment_manager.experiment_dir)

            # 如果有预训练权重，复制到实验目录
            if pretrained_weight_id:
                self._copy_pretrained_weight(
                    db, pretrained_weight_id, experiment_dir_path
                )
                config["pretrained_weight_id"] = pretrained_weight_id

            # 更新数据库记录中的实验目录路径
            training_run.experiment_dir = experiment_dir_path
            db.commit()

            # 保存训练配置到文件
            experiment_manager.save_config(config)

            self.logger.info(f"实验目录已创建: {experiment_dir_path}")

            # 创建训练日志会话
            experiment_id = f"exp_{training_run.id}"
            training_logger.create_session(
                experiment_id,
                config=config
            )
            training_logger.update_status(
                experiment_id,
                TrainingStatus.QUEUED
            )

            self.logger.info(f"训练任务已创建: {training_run.id} - {name}")

            return training_run

        except Exception as e:
            self.logger.error(f"创建训练任务失败: {e}")
            db.rollback()
            raise ValueError(f"创建训练任务失败: {e}")

    def start_training(
        self,
        training_run_id: int,
        model_arch: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ) -> str:
        """
        启动训练任务

        Args:
            training_run_id: 训练任务ID
            model_arch: 模型架构信息
            dataset_info: 数据集信息

        Returns:
            Celery任务ID

        Raises:
            ValueError: 当训练任务不存在或启动失败时
        """
        debug_log(f"=== 开始启动训练任务: training_run_id={training_run_id} ===")
        debug_log(f"模型架构: {model_arch.get('class_name', 'Unknown')}")
        debug_log(f"数据集: {dataset_info.get('name', 'Unknown')}")

        try:
            from app.database import SessionLocal
            db = SessionLocal()
            try:
                training_run = db.query(TrainingRun).filter(
                    TrainingRun.id == training_run_id
                ).first()

                if not training_run:
                    debug_log(f"训练任务不存在: {training_run_id}", "ERROR")
                    raise ValueError(f"训练任务不存在: {training_run_id}")

                debug_log(f"找到训练任务: {training_run.name}, 当前状态: {training_run.status}")
                debug_log(f"超参数配置: {training_run.hyperparams}")

                # 解析配置
                debug_log("开始解析前端配置...")
                config = self.config_parser.parse_frontend_config(
                    training_run.hyperparams,
                    model_arch,
                    dataset_info
                )
                debug_log(f"配置解析完成: {list(config.keys())}")

                experiment_id = f"exp_{training_run.id}"
                debug_log(f"实验ID: {experiment_id}")

                # 提交Celery任务
                debug_log("准备提交Celery任务...")
                from app.tasks.training_tasks import start_training as celery_start_training

                debug_log("调用 celery_start_training.delay()...")
                task = celery_start_training.delay(experiment_id, config)
                debug_log(f"Celery任务已提交: task_id={task.id}", "INFO")

                # 更新数据库状态
                training_run.status = "queued"
                training_run.start_time = datetime.utcnow()
                training_run.celery_task_id = task.id
                db.commit()
                debug_log(f"数据库状态已更新: status=queued, celery_task_id={task.id}")

                # 更新日志状态
                training_logger.update_status(
                    experiment_id,
                    TrainingStatus.QUEUED
                )

                # 添加日志
                training_logger.add_log(
                    experiment_id,
                    "INFO",
                    f"训练任务已提交到队列 (任务ID: {task.id})",
                    "service"
                )

                self.logger.info(
                    f"训练任务已启动: {training_run_id}, Celery任务: {task.id}"
                )

                debug_log(f"=== 训练任务启动完成: task_id={task.id} ===", "INFO")
                return task.id

            except Exception as e:
                db.rollback()
                debug_log(f"启动训练失败 (内部异常): {e}", "ERROR")
                import traceback
                debug_log(f"错误堆栈:\n{traceback.format_exc()}", "ERROR")
                raise
            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"启动训练任务失败: {e}")
            debug_log(f"启动训练任务失败 (外部异常): {e}", "ERROR")
            raise ValueError(f"启动训练任务失败: {e}")

    def control_training(
        self,
        training_run_id: int,
        action: str
    ) -> Dict[str, Any]:
        """
        控制训练任务

        Args:
            training_run_id: 训练任务ID
            action: 控制动作 (pause|resume|stop)

        Returns:
            操作结果字典

        Raises:
            ValueError: 当训练任务不存在或操作无效时
        """
        try:
            from app.database import SessionLocal
            db = SessionLocal()
            try:
                training_run = db.query(TrainingRun).filter(
                    TrainingRun.id == training_run_id
                ).first()

                if not training_run:
                    raise ValueError(f"训练任务不存在: {training_run_id}")

                experiment_id = f"exp_{training_run_id}"

                # 验证操作
                valid_actions = ["pause", "resume", "stop"]
                if action not in valid_actions:
                    raise ValueError(f"无效的操作: {action}")

                # 先更新 training_logger 状态（让训练循环检测到并停止）
                if action == "pause":
                    training_logger.update_status(experiment_id, TrainingStatus.PAUSED)
                    training_run.status = "paused"
                elif action == "resume":
                    training_logger.update_status(experiment_id, TrainingStatus.RUNNING)
                    training_run.status = "running"
                elif action == "stop":
                    training_logger.update_status(experiment_id, TrainingStatus.STOPPED)
                    training_run.status = "stopped"
                    training_run.end_time = datetime.utcnow()
                    # 计算最终进度
                    if training_run.total_epochs > 0:
                        training_run.progress = (training_run.current_epoch / training_run.total_epochs) * 100

                db.commit()

                # 添加日志
                training_logger.add_log(
                    experiment_id,
                    "INFO",
                    f"训练控制操作: {action}",
                    "service"
                )

                # 对于停止操作，尝试撤销 Celery 任务（作为额外的保险）
                if action == "stop" and training_run.celery_task_id:
                    try:
                        from backend.celery_app import celery_app
                        celery_app.control.revoke(
                            training_run.celery_task_id,
                            terminate=True,
                            signal='SIGKILL'
                        )
                        debug_log(f"已撤销训练任务: {training_run.celery_task_id}", "INFO")
                    except Exception as e:
                        debug_log(f"撤销训练任务失败: {e}", "WARNING")
                        # 继续执行，因为训练循环会通过 training_logger 状态检测并停止

                self.logger.info(
                    f"训练控制: {training_run_id} - {action}"
                )

                return {
                    "success": True,
                    "action": action,
                    "task_id": training_run.celery_task_id or "",
                    "experiment_id": experiment_id
                }

            except Exception as e:
                db.rollback()
                raise
            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"控制训练任务失败: {e}")
            raise ValueError(f"控制训练任务失败: {e}")

    def get_training_runs(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[TrainingRun]:
        """
        获取训练任务列表

        Args:
            db: 数据库会话
            skip: 跳过记录数
            limit: 限制返回数量
            status: 状态过滤

        Returns:
            训练任务列表
        """
        try:
            query = db.query(TrainingRun)

            if status:
                query = query.filter(TrainingRun.status == status)

            training_runs = query.order_by(
                TrainingRun.created_at.desc()
            ).offset(skip).limit(limit).all()

            self.logger.debug(f"获取训练任务列表: {len(training_runs)} 条记录")

            return training_runs

        except Exception as e:
            self.logger.error(f"获取训练任务列表失败: {e}")
            return []

    def get_training_run(
        self,
        db: Session,
        training_run_id: int
    ) -> Optional[TrainingRun]:
        """
        获取单个训练任务

        Args:
            db: 数据库会话
            training_run_id: 训练任务ID

        Returns:
            训练任务对象，如果不存在返回None
        """
        try:
            training_run = db.query(TrainingRun).filter(
                TrainingRun.id == training_run_id
            ).first()

            return training_run

        except Exception as e:
            self.logger.error(f"获取训练任务失败: {e}")
            return None

    def get_training_config_detail(
        self,
        db: Session,
        training_run_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        获取训练任务的完整配置详情

        返回包含数据集、模型架构、预训练权重、数据增强等完整信息

        Args:
            db: 数据库会话
            training_run_id: 训练任务ID

        Returns:
            包含完整配置信息的字典，如果训练任务不存在返回None
        """
        try:
            training = db.query(TrainingRun).filter(
                TrainingRun.id == training_run_id
            ).first()

            if not training:
                return None

            result = {
                "id": training.id,
                "name": training.name,
                "description": training.description,
                "status": training.status,
                "created_at": training.created_at.isoformat() if training.created_at else None,
                "hyperparams": training.hyperparams or {},
                "dataset": None,
                "model_architecture": None,
                "pretrained_weight": None,
                "augmentation": None
            }

            # 获取数据集信息
            if training.dataset_id:
                from app.models.dataset import Dataset
                dataset = db.query(Dataset).filter(
                    Dataset.id == training.dataset_id
                ).first()
                if dataset:
                    result["dataset"] = {
                        "id": dataset.id,
                        "name": dataset.name,
                        "format": dataset.format,
                        "num_images": dataset.num_images,
                        "num_classes": dataset.num_classes,
                        "classes": dataset.classes,
                        "path": dataset.path
                    }

            # 获取模型架构信息
            if training.model_id:
                from app.models.generated_code import GeneratedCode
                model = db.query(GeneratedCode).filter(
                    GeneratedCode.id == training.model_id,
                    GeneratedCode.is_active == "active"
                ).first()

                if model:
                    meta = model.meta if isinstance(model.meta, dict) else {}
                    result["model_architecture"] = {
                        "id": model.id,
                        "name": _clean_model_name(model.name),
                        "description": None,  # GeneratedCode 没有description字段
                        "file_path": model.file_path,
                        "input_size": meta.get("input_size"),
                        "task_type": meta.get("task_type")
                    }
                else:
                    # 兼容 Model 表
                    from app.models.model import Model
                    old_model = db.query(Model).filter(
                        Model.id == training.model_id
                    ).first()
                    if old_model:
                        import json
                        graph_json = old_model.graph_json if isinstance(old_model.graph_json, dict) else \
                            json.loads(old_model.graph_json) if old_model.graph_json else {}
                        result["model_architecture"] = {
                            "id": old_model.id,
                            "name": _clean_model_name(old_model.name),
                            "description": old_model.description,
                            "file_path": old_model.code_path,
                            "input_size": graph_json.get("input_size"),
                            "task_type": graph_json.get("task_type")
                        }

            # 获取预训练权重信息
            if training.pretrained_weight_id:
                from app.models.weight_library import WeightLibrary
                pretrained = db.query(WeightLibrary).filter(
                    WeightLibrary.id == training.pretrained_weight_id
                ).first()
                if pretrained:
                    result["pretrained_weight"] = {
                        "id": pretrained.id,
                        "name": pretrained.name,
                        "display_name": pretrained.display_name,
                        "task_type": pretrained.task_type,
                        "version": pretrained.version,
                        "source_type": pretrained.source_type
                    }

            # 解析数据增强配置
            hyperparams = training.hyperparams or {}
            aug_strategy_id = hyperparams.get("augmentation_strategy_id")

            if aug_strategy_id:
                # 通过 augmentation_strategy_id 获取策略详情
                aug_strategy = db.query(AugmentationStrategy).filter(
                    AugmentationStrategy.id == aug_strategy_id
                ).first()
                if aug_strategy:
                    result["augmentation"] = {
                        "enabled": True,
                        "strategy": aug_strategy.name,
                        "strategy_id": aug_strategy.id,
                        "description": aug_strategy.description,
                        "config": aug_strategy.pipeline
                    }
                else:
                    result["augmentation"] = {"enabled": False}
            elif "augmentation" in hyperparams:
                # 兼容旧的 augmentation 对象格式
                aug_config = hyperparams["augmentation"]
                result["augmentation"] = {
                    "enabled": aug_config.get("enabled", False),
                    "strategy": aug_config.get("strategy"),
                    "config": aug_config
                }
            else:
                result["augmentation"] = {"enabled": False}

            return result

        except Exception as e:
            self.logger.error(f"获取训练配置详情失败: {e}")
            return None

    def update_training_run(
        self,
        db: Session,
        training_run_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Optional[TrainingRun]:
        """
        更新训练任务（重命名、修改描述等）

        Args:
            db: 数据库会话
            training_run_id: 训练任务ID
            name: 新名称（可选）
            description: 新描述（可选）

        Returns:
            更新后的训练任务对象，如果不存在返回None
        """
        try:
            training_run = db.query(TrainingRun).filter(
                TrainingRun.id == training_run_id
            ).first()

            if not training_run:
                return None

            # 更新字段
            if name:
                training_run.name = name
            if description is not None:
                training_run.description = description

            db.commit()
            db.refresh(training_run)

            self.logger.info(f"训练任务已更新: {training_run_id}")

            return training_run

        except Exception as e:
            self.logger.error(f"更新训练任务失败: {e}")
            db.rollback()
            return None

    def delete_training_run(
        self,
        db: Session,
        training_run_id: int
    ) -> bool:
        """
        删除训练任务

        Args:
            db: 数据库会话
            training_run_id: 训练任务ID

        Returns:
            是否删除成功
        """
        try:
            training_run = db.query(TrainingRun).filter(
                TrainingRun.id == training_run_id
            ).first()

            if not training_run:
                return False

            # 删除实验目录（包含所有训练文件）
            experiment_manager = ExperimentManager(training_run_id)
            if experiment_manager.delete_experiment_dir():
                self.logger.info(f"实验目录已删除: {experiment_manager.experiment_dir}")

            # 删除checkpoint文件（旧目录结构，兼容性）
            checkpoint_dir = f"data/checkpoints/exp_{training_run_id}"
            checkpoint_manager = CheckpointManager(checkpoint_dir)

            # 删除数据库中的checkpoint记录
            deleted_count = checkpoint_manager.delete_checkpoints_by_run(
                training_run_id,
                delete_files=True
            )

            self.logger.info(
                f"已删除 {deleted_count} 个checkpoint (run_id: {training_run_id})"
            )

            # 删除日志会话
            experiment_id = f"exp_{training_run_id}"
            training_logger.delete_session(experiment_id)

            # 删除数据库记录
            db.delete(training_run)
            db.commit()

            self.logger.info(f"训练任务已删除: {training_run_id}")

            return True

        except Exception as e:
            self.logger.error(f"删除训练任务失败: {e}")
            db.rollback()
            return False

    def save_to_weights(
        self,
        training_run_id: int,
        weights_dir: str = "data/weights"
    ) -> str:
        """
        保存最佳checkpoint到权重库

        Args:
            training_run_id: 训练任务ID
            weights_dir: 权重库目录

        Returns:
            保存的权重路径

        Raises:
            ValueError: 当没有找到最佳checkpoint时
        """
        try:
            checkpoint_dir = f"data/checkpoints/exp_{training_run_id}"
            checkpoint_manager = CheckpointManager(checkpoint_dir)

            # 获取最佳checkpoint
            best_checkpoint = checkpoint_manager.get_best_checkpoint(training_run_id)

            if not best_checkpoint:
                raise ValueError(f"未找到最佳checkpoint (run_id: {training_run_id})")

            # 获取训练任务信息
            from app.database import SessionLocal
            db = SessionLocal()
            try:
                training_run = db.query(TrainingRun).filter(
                    TrainingRun.id == training_run_id
                ).first()

                if not training_run:
                    raise ValueError(f"训练任务不存在: {training_run_id}")

                model_name = training_run.name

                # 复制到权重库
                dest_path = checkpoint_manager.copy_to_weights(
                    best_checkpoint,
                    weights_dir,
                    model_name
                )

                self.logger.info(
                    f"最佳模型已保存到权重库: {training_run_id} -> {dest_path}"
                )

                # 添加日志
                experiment_id = f"exp_{training_run_id}"
                training_logger.add_log(
                    experiment_id,
                    "INFO",
                    f"最佳模型已保存到权重库: {dest_path}",
                    "service"
                )

                return dest_path

            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"保存到权重库失败: {e}")
            raise ValueError(f"保存到权重库失败: {e}")

    def save_to_weights_library(
        self,
        training_run_id: int,
        weight_name: str,
        description: str,
        include_last: bool = True,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        保存训练模型到权重库（带版本管理）

        保存best模型和最后一个epoch模型到权重库。
        best模型作为主版本（1.0），last模型作为子版本（1.1）。

        Args:
            training_run_id: 训练任务ID
            weight_name: 权重名称
            description: 权重描述
            include_last: 是否保存最后一个epoch模型
            db: 数据库会话（可选）

        Returns:
            {
                "success": True,
                "best_weight": {...},  # best模型的权重库记录
                "last_weight": {...},  # last模型的权重库记录（如果保存）
            }

        Raises:
            ValueError: 当训练任务不存在或没有checkpoint时
        """
        close_db = False
        if db is None:
            from app.database import SessionLocal
            db = SessionLocal()
            close_db = True

        try:
            # 获取训练任务信息
            training_run = db.query(TrainingRun).filter(
                TrainingRun.id == training_run_id
            ).first()

            if not training_run:
                raise ValueError(f"训练任务不存在: {training_run_id}")

            # 确定任务类型
            task_type = training_run.hyperparams.get("task_type", "classification")
            if task_type not in ["classification", "detection"]:
                task_type = "classification"

            # 获取checkpoint目录 - 支持多个可能的路径
            experiment_dir_rel = training_run.experiment_dir or f"data/experiments/exp_{training_run_id}"
            experiment_dir = Path(experiment_dir_rel)

            # 如果相对路径不存在，尝试从backend目录查找
            if not experiment_dir.exists():
                backend_exp_dir = Path("backend/data/experiments") / f"exp_{training_run_id}"
                if backend_exp_dir.exists():
                    experiment_dir = backend_exp_dir

            checkpoints_dir = experiment_dir / "checkpoints"

            self.logger.info(f"查找checkpoint，实验目录: {experiment_dir}, checkpoint目录: {checkpoints_dir}")

            # 先尝试从数据库获取checkpoint记录
            checkpoint_manager = CheckpointManager(str(experiment_dir))
            checkpoints = checkpoint_manager.list_checkpoints(training_run_id, db)

            # 如果数据库中没有记录，尝试从文件系统读取
            if not checkpoints:
                import torch
                self.logger.info(f"数据库中没有checkpoint记录，尝试从文件系统读取")

                # 收集所有可能的checkpoint文件
                checkpoint_files = []

                # 1. 检查checkpoints子目录
                if checkpoints_dir.exists():
                    checkpoint_files.extend(checkpoints_dir.glob("*.pt"))
                    self.logger.info(f"从checkpoints子目录找到: {len(checkpoint_files)} 个文件")

                # 2. 检查实验根目录（兼容旧格式）
                checkpoint_files.extend(experiment_dir.glob("checkpoint*.pt"))
                checkpoint_files.extend(experiment_dir.glob("epoch*.pt"))
                checkpoint_files.extend(experiment_dir.glob("best_model.pt"))

                self.logger.info(f"总共找到 {len(checkpoint_files)} 个checkpoint文件")

                if checkpoint_files:
                    checkpoints = []
                    for cf in sorted(checkpoint_files):
                        try:
                            ckpt = torch.load(cf, map_location='cpu')
                            # 判断是否为best模型
                            is_best = False
                            if "best_model.pt" in str(cf):
                                is_best = True
                            elif ckpt.get("is_best"):
                                is_best = ckpt.get("is_best")

                            # 获取epoch信息
                            epoch = ckpt.get("epoch", 0)

                            # 尝试从文件名解析epoch
                            if "epoch_" in cf.name:
                                try:
                                    epoch = int(cf.name.split("epoch_")[1].split(".pt")[0])
                                except:
                                    pass
                            elif "checkpoint_epoch_" in cf.name:
                                try:
                                    epoch = int(cf.name.split("checkpoint_epoch_")[1].split(".pt")[0])
                                except:
                                    pass

                            # 获取指标值
                            metrics = ckpt.get("metrics", {})
                            metric_value = metrics.get("val_acc", metrics.get("val_accuracy", 0))

                            checkpoints.append({
                                "epoch": epoch,
                                "path": str(cf),
                                "metric_value": metric_value,
                                "metrics": metrics,
                                "is_best": is_best,
                                "file_size": cf.stat().st_size
                            })
                            self.logger.info(f"找到checkpoint: {cf.name}, epoch={epoch}, is_best={is_best}, metric={metric_value:.4f}")
                        except Exception as e:
                            self.logger.warning(f"无法加载checkpoint文件 {cf}: {e}")

            if not checkpoints:
                raise ValueError(f"未找到任何checkpoint (run_id: {training_run_id}, 目录: {experiment_dir})")

            # 获取best checkpoint
            best_cp = None
            last_cp = None

            def is_best_checkpoint(cp):
                """检查是否为最佳checkpoint（兼容布尔值和字符串）"""
                val = cp.get("is_best")
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    return val.lower() == "true"
                return False

            for cp in checkpoints:
                if is_best_checkpoint(cp) and not best_cp:
                    best_cp = cp
                if not last_cp or cp.get("epoch", 0) > last_cp.get("epoch", 0):
                    last_cp = cp

            if not best_cp:
                # 如果没有标记为best的，使用metric_value最高的
                best_cp = max(checkpoints, key=lambda x: x.get("metric_value", 0))

            if not last_cp:
                last_cp = checkpoints[0]

            # 导入权重库服务
            from app.services.weight_library_service import WeightLibraryService
            weight_service = WeightLibraryService()

            # 确定输入尺寸
            input_size = training_run.hyperparams.get("input_size", [224, 224])
            if isinstance(input_size, int):
                input_size = [input_size, input_size]

            # 检查是否使用了预训练权重
            pretrained_weight_id = training_run.pretrained_weight_id
            parent_weight_id = None
            base_version = "1.0"

            if pretrained_weight_id:
                # 获取预训练权重信息
                from app.models.weight_library import WeightLibrary as WeightModel
                pretrained = db.query(WeightModel).filter(WeightModel.id == pretrained_weight_id).first()
                if pretrained:
                    parent_weight_id = pretrained_weight_id
                    actual_parent = pretrained

                    # 如果预训练权重是 last 模型（有 parent_version_id 且来自同一训练），
                    # 应该使用同一次训练的 best 模型作为父节点
                    if (pretrained.parent_version_id and
                        pretrained.source_training_id and
                        pretrained.source_type == 'trained'):
                        # 检查父节点是否是同一次训练的 best 模型
                        parent_node = db.query(WeightModel).filter(
                            WeightModel.id == pretrained.parent_version_id
                        ).first()
                        if (parent_node and
                            parent_node.source_training_id == pretrained.source_training_id):
                            # 使用 best 模型作为父节点
                            parent_weight_id = parent_node.id
                            actual_parent = parent_node
                            self.logger.info(
                                f"检测到预训练权重是 last 模型，使用同训练的 best 模型作为父节点: "
                                f"{actual_parent.name} v{actual_parent.version}"
                            )

                    # 版本号在父权重版本基础上递增
                    try:
                        parent_version_parts = actual_parent.version.split(".")
                        if len(parent_version_parts) >= 2:
                            major = int(parent_version_parts[0])
                            # minor 位 +1 作为新的 best 版本号
                            minor = int(parent_version_parts[1]) + 1
                            base_version = f"{major}.{minor}"
                        else:
                            base_version = "1.1"
                    except:
                        base_version = "1.1"

                    self.logger.info(f"使用预训练权重 {actual_parent.name} v{actual_parent.version}，新版本将作为子节点")

            # 1. 保存best模型
            best_file_path = Path(best_cp["path"])
            if not best_file_path.exists():
                raise ValueError(f"Best checkpoint文件不存在: {best_cp['path']}")

            # 生成目标文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_filename = f"{weight_name}_best_{timestamp}.pt"
            last_filename = f"{weight_name}_last_{timestamp}.pt"

            # 确定目标目录
            target_dir = Path(weight_service.STORAGE_PATH) / task_type
            target_dir.mkdir(parents=True, exist_ok=True)

            # 复制best模型到权重库
            best_target_path = target_dir / best_filename
            import shutil
            shutil.copy2(best_cp["path"], str(best_target_path))

            # 创建权重库记录
            from app.models.weight_library import WeightLibrary

            # 确定best权重是否为根节点
            best_is_root = parent_weight_id is None

            best_weight = WeightLibrary(
                name=weight_name,
                description=f"{description} (最佳模型)" if description else f"{weight_name} - 最佳模型",
                task_type=task_type,
                version=base_version,
                parent_version_id=parent_weight_id,  # 关联到预训练权重
                file_path=str(best_target_path),
                file_name=best_filename,
                file_size=best_target_path.stat().st_size,
                framework="pytorch",
                input_size=input_size,
                uploaded_by=training_run.created_by,
                source_type="trained",
                source_training_id=training_run_id,
                is_root=best_is_root,
                architecture_id=training_run.model_id
            )
            db.add(best_weight)
            db.flush()  # 获取ID但不提交

            last_weight_record = None

            # 2. 如果需要，保存last模型
            if include_last:
                # 检查best和last是否是同一个文件
                if last_cp["path"] == best_cp["path"]:
                    # 同一个文件，使用best的文件路径
                    last_target_path = best_target_path
                    last_filename_for_db = best_filename
                else:
                    # 不同文件，复制last模型
                    last_target_path = target_dir / last_filename
                    shutil.copy2(last_cp["path"], str(last_target_path))
                    last_filename_for_db = last_filename

                # last版本的版本号
                try:
                    base_parts = base_version.split(".")
                    if len(base_parts) >= 2:
                        major = int(base_parts[0])
                        minor = int(base_parts[1])
                        last_version = f"{major}.{minor + 1}"
                    else:
                        last_version = "1.1"
                except:
                    last_version = "1.1"

                last_weight_record = WeightLibrary(
                    name=weight_name,
                    description=f"{description} (最后Epoch)" if description else f"{weight_name} - 最后Epoch",
                    task_type=task_type,
                    version=last_version,
                    parent_version_id=best_weight.id,  # last是best的子节点
                    file_path=str(last_target_path),
                    file_name=last_filename_for_db,
                    file_size=last_target_path.stat().st_size,
                    framework="pytorch",
                    input_size=input_size,
                    uploaded_by=training_run.created_by,
                    source_type="trained",
                    source_training_id=training_run_id,
                    is_root=False,  # last永远不是根节点
                    architecture_id=training_run.model_id
                )
                db.add(last_weight_record)

            # 提交数据库
            db.commit()
            db.refresh(best_weight)
            if last_weight_record:
                db.refresh(last_weight_record)

            self.logger.success(
                f"权重已保存到权重库: {weight_name} (best: v{best_weight.version}, last: {last_weight_record.version if last_weight_record else 'N/A'})"
            )

            # 添加日志
            experiment_id = f"exp_{training_run_id}"
            training_logger.add_log(
                experiment_id,
                "INFO",
                f"权重已保存到权重库: {weight_name}",
                "service"
            )

            return {
                "success": True,
                "best_weight": best_weight,
                "last_weight": last_weight_record
            }

        except Exception as e:
            self.logger.error(f"保存到权重库失败: {e}")
            db.rollback()
            raise ValueError(f"保存到权重库失败: {e}")
        finally:
            if close_db:
                db.close()

    def get_training_metrics(
        self,
        training_run_id: int,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取训练指标数据

        Args:
            training_run_id: 训练任务ID
            limit: 限制返回数量

        Returns:
            指标数据列表
        """
        try:
            experiment_id = f"exp_{training_run_id}"
            metrics = training_logger.get_metrics(experiment_id, limit=limit)

            return metrics

        except Exception as e:
            self.logger.error(f"获取训练指标失败: {e}")
            return []

    def get_training_logs(
        self,
        training_run_id: int,
        level: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取训练日志

        Args:
            training_run_id: 训练任务ID
            level: 日志级别过滤（可选）
            limit: 限制返回数量

        Returns:
            日志数据列表
        """
        try:
            experiment_id = f"exp_{training_run_id}"
            logs = training_logger.get_logs(experiment_id, level=level, limit=limit)

            return logs

        except Exception as e:
            self.logger.error(f"获取训练日志失败: {e}")
            return []

    def _copy_pretrained_weight(
        self,
        db: Session,
        weight_id: int,
        experiment_dir: str
    ) -> str:
        """
        复制预训练权重到实验目录

        Args:
            db: 数据库会话
            weight_id: 预训练权重ID
            experiment_dir: 实验目录路径

        Returns:
            复制后的权重文件路径

        Raises:
            ValueError: 当权重不存在时
        """
        try:
            from app.models.weight_library import WeightLibrary
            import shutil

            # 获取权重记录
            weight = db.query(WeightLibrary).filter(
                WeightLibrary.id == weight_id
            ).first()

            if not weight:
                raise ValueError(f"预训练权重不存在: {weight_id}")

            # 源文件路径
            source_path = Path(weight.file_path)
            if not source_path.exists():
                raise ValueError(f"预训练权重文件不存在: {source_path}")

            # 目标路径
            target_path = Path(experiment_dir) / f"pretrained_{source_path.name}"

            # 复制文件
            shutil.copy2(source_path, target_path)

            self.logger.info(
                f"预训练权重已复制: {source_path} -> {target_path}"
            )

            return str(target_path)

        except Exception as e:
            self.logger.error(f"复制预训练权重失败: {e}")
            raise ValueError(f"复制预训练权重失败: {e}")
