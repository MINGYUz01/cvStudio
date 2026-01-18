"""
Celery训练任务
定义训练相关的异步任务
"""

import sys
from pathlib import Path
# 添加backend根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from celery import current_task
from loguru import logger
from typing import Dict, Any

from celery_app import celery_app
from app.utils.trainer import Trainer
from app.utils.training_logger import training_logger, TrainingStatus


def debug_log(msg: str, level: str = "INFO"):
    """输出调试日志到控制台和文件"""
    log_msg = f"[CELERY_TASK] {msg}"
    print(f"[DEBUG] {log_msg}", flush=True)  # 强制输出到控制台
    if level == "INFO":
        logger.info(log_msg)
    elif level == "ERROR":
        logger.error(log_msg)
    elif level == "WARNING":
        logger.warning(log_msg)


@celery_app.task(
    name='app.tasks.training_tasks.start_training',
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={'max_retries': 3, 'countdown': 60}
)
def start_training(self, experiment_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    启动训练任务的Celery任务

    Args:
        self: Celery任务实例
        experiment_id: 实验ID
        config: 训练配置字典

    Returns:
        训练结果字典

    Raises:
        Exception: 当训练失败时
    """
    debug_log(f"=== 收到训练任务: experiment_id={experiment_id} ===")
    debug_log(f"Celery Task ID: {self.request.id}")
    debug_log(f"配置键: {list(config.keys())}")

    try:
        logger.info(f"开始训练任务: {experiment_id}")
        debug_log("更新训练状态为 RUNNING...")

        # 更新状态为running
        training_logger.update_status(experiment_id, TrainingStatus.RUNNING)

        # 添加任务ID到配置
        config["celery_task_id"] = self.request.id
        debug_log(f"Celery任务ID已添加到配置: {self.request.id}")

        debug_log("创建 Trainer 实例...")

        # 创建训练器实例
        trainer = Trainer(
            experiment_id=experiment_id,
            config=config,
            celery_task=self
        )

        debug_log("Trainer 实例创建成功，开始训练...")

        # 开始训练
        result = trainer.train()

        logger.info(f"训练任务完成: {experiment_id}")
        debug_log(f"=== 训练任务完成: {experiment_id} ===", "INFO")

        return result

    except Exception as e:
        logger.error(f"训练任务执行失败 {experiment_id}: {e}")
        debug_log(f"训练任务执行失败: {e}", "ERROR")
        import traceback
        debug_log(f"错误堆栈:\n{traceback.format_exc()}", "ERROR")

        # 更新状态为失败
        training_logger.update_status(experiment_id, TrainingStatus.FAILED)

        # 添加错误日志
        training_logger.add_log(
            experiment_id,
            "ERROR",
            f"训练失败: {str(e)}",
            "celery_task"
        )

        # 重新抛出异常以触发Celery重试机制
        raise


@celery_app.task(
    name='app.tasks.training_tasks.control_training',
    bind=True
)
def control_training(self, experiment_id: str, action: str) -> Dict[str, Any]:
    """
    控制训练任务

    Args:
        self: Celery任务实例
        experiment_id: 实验ID
        action: 控制动作 (pause|resume|stop)

    Returns:
        控制结果字典

    Raises:
        ValueError: 当动作无效时
    """
    try:
        logger.info(f"控制训练任务: {experiment_id} - {action}")

        # 验证动作
        valid_actions = ["pause", "resume", "stop"]
        if action not in valid_actions:
            raise ValueError(f"无效的控制动作: {action}")

        # 获取训练会话
        session = training_logger.get_session(experiment_id)

        if not session:
            raise ValueError(f"训练会话不存在: {experiment_id}")

        # 根据动作执行相应操作
        if action == "pause":
            # 更新状态
            training_logger.update_status(experiment_id, TrainingStatus.PAUSED)

            # 添加日志
            training_logger.add_log(
                experiment_id,
                "INFO",
                f"训练已暂停 (Celery任务: {self.request.id})",
                "control"
            )

            result = {
                "success": True,
                "action": action,
                "message": "训练已暂停"
            }

        elif action == "resume":
            # 更新状态
            training_logger.update_status(experiment_id, TrainingStatus.RUNNING)

            # 添加日志
            training_logger.add_log(
                experiment_id,
                "INFO",
                f"训练已恢复 (Celery任务: {self.request.id})",
                "control"
            )

            result = {
                "success": True,
                "action": action,
                "message": "训练已恢复"
            }

        elif action == "stop":
            # 更新状态
            training_logger.update_status(experiment_id, TrainingStatus.STOPPED)

            # 添加日志
            training_logger.add_log(
                experiment_id,
                "INFO",
                f"训练已停止 (Celery任务: {self.request.id})",
                "control"
            )

            result = {
                "success": True,
                "action": action,
                "message": "训练已停止"
            }

        logger.info(f"控制任务完成: {experiment_id} - {action}")

        return result

    except ValueError as e:
        logger.error(f"控制训练任务失败: {e}")
        raise
    except Exception as e:
        logger.error(f"控制训练任务失败 {experiment_id}: {e}")
        raise


@celery_app.task(
    name='app.tasks.training_tasks.save_checkpoint',
    bind=True
)
def save_checkpoint_task(
    self,
    experiment_id: int,
    epoch: int,
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    保存checkpoint的Celery任务（可选）

    Args:
        self: Celery任务实例
        experiment_id: 实验ID
        epoch: 当前epoch
        metrics: 指标字典

    Returns:
        保存结果
    """
    try:
        logger.info(f"保存checkpoint: exp_{experiment_id} - epoch {epoch}")

        # 这里可以添加异步保存逻辑
        # 目前checkpoint在训练过程中同步保存

        result = {
            "success": True,
            "epoch": epoch,
            "message": "Checkpoint保存成功"
        }

        return result

    except Exception as e:
        logger.error(f"保存checkpoint失败: {e}")
        raise


@celery_app.task(name='app.tasks.training_tasks.health_check')
def health_check() -> Dict[str, Any]:
    """
    健康检查任务

    Returns:
        健康状态信息
    """
    try:
        from app.utils.training_logger import training_logger

        # 获取活动会话数
        active_sessions = len(training_logger.training_sessions)

        return {
            "status": "healthy",
            "active_sessions": active_sessions,
            "timestamp": training_logger.get_current_timestamp()
        }

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@celery_app.task(name='app.tasks.training_tasks.cleanup_old_sessions')
def cleanup_old_sessions(max_age_hours: int = 24) -> Dict[str, Any]:
    """
    清理旧的训练会话

    Args:
        max_age_hours: 最大保留时间（小时）

    Returns:
        清理结果
    """
    try:
        from datetime import datetime, timedelta
        from app.utils.training_logger import training_logger

        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)

        cleaned_count = 0

        for exp_id, session in list(training_logger.training_sessions.items()):
            created_at = session.get("created_at")
            if created_at and created_at < cutoff_time:
                # 删除旧会话
                training_logger.delete_session(exp_id)
                cleaned_count += 1
                logger.info(f"已清理旧会话: {exp_id}")

        return {
            "success": True,
            "cleaned_count": cleaned_count,
            "max_age_hours": max_age_hours
        }

    except Exception as e:
        logger.error(f"清理旧会话失败: {e}")
        return {
            "success": False,
            "error": str(e)
        }
