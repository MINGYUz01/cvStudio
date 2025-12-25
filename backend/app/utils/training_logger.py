"""
训练日志收集器
收集训练过程中的日志、指标并通过WebSocket实时推送
"""

import asyncio
from loguru import logger
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
import json


class TrainingStatus(str, Enum):
    """训练状态枚举"""
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingLogger:
    """训练日志收集器"""

    def __init__(self):
        # 存储所有训练任务的日志缓冲区
        # {experiment_id: {"logs": [], "metrics": [], "status": TrainingStatus}}
        self.training_sessions: Dict[str, dict] = {}

        # 日志级别过滤
        self.log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # 每个训练任务保留的最大日志条数
        self.max_logs = 1000

    def create_session(self, experiment_id: str, config: dict = None):
        """
        创建新的训练会话

        Args:
            experiment_id: 实验/训练任务ID
            config: 训练配置信息
        """
        if experiment_id in self.training_sessions:
            logger.warning(f"训练会话 {experiment_id} 已存在，将被覆盖")

        self.training_sessions[experiment_id] = {
            "logs": [],
            "metrics": [],
            "status": TrainingStatus.QUEUED,
            "config": config or {},
            "created_at": datetime.utcnow().isoformat() + "Z",
            "started_at": None,
            "ended_at": None,
            "current_epoch": 0,
            "total_epochs": 0
        }

        logger.info(f"创建训练会话: {experiment_id}")

    def update_status(self, experiment_id: str, status: TrainingStatus):
        """
        更新训练状态

        Args:
            experiment_id: 实验/训练任务ID
            status: 新的训练状态
        """
        if experiment_id not in self.training_sessions:
            logger.warning(f"训练会话 {experiment_id} 不存在")
            return

        old_status = self.training_sessions[experiment_id]["status"]
        self.training_sessions[experiment_id]["status"] = status

        # 设置时间戳
        if status == TrainingStatus.RUNNING and not self.training_sessions[experiment_id]["started_at"]:
            self.training_sessions[experiment_id]["started_at"] = datetime.utcnow().isoformat() + "Z"
        elif status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.STOPPED]:
            self.training_sessions[experiment_id]["ended_at"] = datetime.utcnow().isoformat() + "Z"

        logger.info(f"训练 {experiment_id} 状态更新: {old_status} -> {status}")

    def add_log(
        self,
        experiment_id: str,
        level: str,
        message: str,
        source: str = "trainer"
    ):
        """
        添加训练日志

        Args:
            experiment_id: 实验/训练任务ID
            level: 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL）
            message: 日志消息
            source: 日志来源
        """
        if experiment_id not in self.training_sessions:
            logger.warning(f"训练会话 {experiment_id} 不存在，日志将被忽略")
            return

        # 验证日志级别
        if level.upper() not in self.log_levels:
            level = "INFO"

        log_entry = {
            "level": level.upper(),
            "message": message,
            "source": source,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # 添加到日志缓冲区
        session = self.training_sessions[experiment_id]
        session["logs"].append(log_entry)

        # 限制日志数量
        if len(session["logs"]) > self.max_logs:
            session["logs"] = session["logs"][-self.max_logs:]

        return log_entry

    def add_metrics(
        self,
        experiment_id: str,
        epoch: int,
        metrics: dict
    ):
        """
        添加训练指标

        Args:
            experiment_id: 实验/训练任务ID
            epoch: 当前epoch数
            metrics: 指标字典（包含train_loss, train_acc, val_loss, val_acc等）
        """
        if experiment_id not in self.training_sessions:
            logger.warning(f"训练会话 {experiment_id} 不存在，指标将被忽略")
            return

        metrics_entry = {
            "epoch": epoch,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **metrics
        }

        # 添加到指标缓冲区
        session = self.training_sessions[experiment_id]
        session["metrics"].append(metrics_entry)

        # 更新当前epoch
        session["current_epoch"] = epoch

        logger.debug(f"训练 {experiment_id} Epoch {epoch} 指标: {metrics}")

        return metrics_entry

    def get_logs(
        self,
        experiment_id: str,
        level: Optional[str] = None,
        limit: int = 100
    ) -> List[dict]:
        """
        获取训练日志

        Args:
            experiment_id: 实验/训练任务ID
            level: 日志级别过滤（None表示不过滤）
            limit: 返回的日志条数限制

        Returns:
            日志列表
        """
        if experiment_id not in self.training_sessions:
            return []

        logs = self.training_sessions[experiment_id]["logs"]

        # 按级别过滤
        if level:
            logs = [log for log in logs if log["level"] == level.upper()]

        # 限制数量并返回最新的日志
        return logs[-limit:]

    def get_metrics(
        self,
        experiment_id: str,
        limit: int = 100
    ) -> List[dict]:
        """
        获取训练指标

        Args:
            experiment_id: 实验/训练任务ID
            limit: 返回的指标条数限制

        Returns:
            指标列表
        """
        if experiment_id not in self.training_sessions:
            return []

        metrics = self.training_sessions[experiment_id]["metrics"]
        return metrics[-limit:]

    def get_session_info(self, experiment_id: str) -> Optional[dict]:
        """
        获取训练会话信息

        Args:
            experiment_id: 实验/训练任务ID

        Returns:
            会话信息字典
        """
        if experiment_id not in self.training_sessions:
            return None

        session = self.training_sessions[experiment_id]

        return {
            "experiment_id": experiment_id,
            "status": session["status"],
            "config": session["config"],
            "created_at": session["created_at"],
            "started_at": session["started_at"],
            "ended_at": session["ended_at"],
            "current_epoch": session["current_epoch"],
            "total_epochs": session["total_epochs"],
            "log_count": len(session["logs"]),
            "metrics_count": len(session["metrics"])
        }

    def delete_session(self, experiment_id: str):
        """
        删除训练会话

        Args:
            experiment_id: 实验/训练任务ID
        """
        if experiment_id in self.training_sessions:
            del self.training_sessions[experiment_id]
            logger.info(f"删除训练会话: {experiment_id}")

    async def broadcast_log(
        self,
        experiment_id: str,
        log_entry: dict,
        manager
    ):
        """
        通过WebSocket广播日志

        Args:
            experiment_id: 实验/训练任务ID
            log_entry: 日志条目
            manager: WebSocket连接管理器
        """
        await manager.send_training_update(experiment_id, {
            "type": "log",
            "data": log_entry
        })

    async def broadcast_metrics(
        self,
        experiment_id: str,
        metrics_entry: dict,
        manager
    ):
        """
        通过WebSocket广播指标

        Args:
            experiment_id: 实验/训练任务ID
            metrics_entry: 指标条目
            manager: WebSocket连接管理器
        """
        await manager.send_training_update(experiment_id, {
            "type": "metrics_update",
            "data": metrics_entry
        })

    async def broadcast_status(
        self,
        experiment_id: str,
        manager
    ):
        """
        通过WebSocket广播状态变化

        Args:
            experiment_id: 实验/训练任务ID
            manager: WebSocket连接管理器
        """
        session = self.training_sessions[experiment_id]

        await manager.send_training_update(experiment_id, {
            "type": "status_change",
            "data": {
                "status": session["status"],
                "current_epoch": session["current_epoch"],
                "total_epochs": session["total_epochs"],
                "started_at": session["started_at"],
                "ended_at": session["ended_at"],
                "message": f"训练状态已更新为: {session['status']}"
            }
        })


# 全局训练日志收集器实例
training_logger = TrainingLogger()
