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
from app.utils.config_parser import TrainingConfigParser
from app.utils.checkpoint_manager import CheckpointManager
from app.utils.training_logger import training_logger, TrainingStatus


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
        user_id: int
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
                created_by=user_id
            )

            db.add(training_run)
            db.commit()
            db.refresh(training_run)

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

                # 停止操作：先撤销正在运行的训练任务
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
                        # 继续执行状态更新

                # 提交Celery控制任务
                from app.tasks.training_tasks import control_training

                task = control_training.delay(experiment_id, action)

                # 更新状态
                if action == "pause":
                    training_run.status = "paused"
                    training_logger.update_status(
                        experiment_id,
                        TrainingStatus.PAUSED
                    )
                elif action == "resume":
                    training_run.status = "running"
                    training_logger.update_status(
                        experiment_id,
                        TrainingStatus.RUNNING
                    )
                elif action == "stop":
                    training_run.status = "stopped"
                    training_run.end_time = datetime.utcnow()
                    # 计算最终进度
                    if training_run.total_epochs > 0:
                        training_run.progress = (training_run.current_epoch / training_run.total_epochs) * 100
                    training_logger.update_status(
                        experiment_id,
                        TrainingStatus.STOPPED
                    )

                db.commit()

                # 添加日志
                training_logger.add_log(
                    experiment_id,
                    "INFO",
                    f"训练控制操作: {action}",
                    "service"
                )

                self.logger.info(
                    f"训练控制: {training_run_id} - {action} (任务: {task.id})"
                )

                return {
                    "success": True,
                    "action": action,
                    "task_id": task.id,
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

            # 删除checkpoint文件
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
