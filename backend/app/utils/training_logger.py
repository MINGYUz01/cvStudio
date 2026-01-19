"""
训练日志收集器
收集训练过程中的日志、指标并通过WebSocket实时推送
使用Redis进行跨进程数据共享
"""

import asyncio
from loguru import logger
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
import json

try:
    import redis
    from app.core.config import settings
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis不可用，训练指标将只在内存中存储（跨进程不可共享）")


class TrainingStatus(str, Enum):
    """训练状态枚举"""
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainingLogger:
    """训练日志收集器 - 支持Redis跨进程共享"""

    METRICS_KEY_PREFIX = "training:metrics:"
    STATUS_KEY_PREFIX = "training:status:"
    LOGS_KEY_PREFIX = "training:logs:"
    SESSION_KEY_PREFIX = "training:session:"

    def __init__(self):
        # 内存缓存（用于本地访问）
        self.training_sessions: Dict[str, dict] = {}

        # 日志级别过滤
        self.log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        # 每个训练任务保留的最大日志条数
        self.max_logs = 1000

        # Redis客户端（延迟初始化）
        self._redis_client = None

    @property
    def redis_client(self):
        """延迟初始化Redis客户端"""
        if self._redis_client is None and REDIS_AVAILABLE:
            try:
                self._redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True
                )
                # 测试连接
                self._redis_client.ping()
                logger.info("Redis连接成功，训练指标将跨进程共享")
            except Exception as e:
                logger.warning(f"Redis连接失败: {e}，训练指标将只在内存中存储")
                self._redis_client = False  # 标记为失败
        return self._redis_client if self._redis_client is not False else None

    def _get_metrics_key(self, experiment_id: str) -> str:
        return f"{self.METRICS_KEY_PREFIX}{experiment_id}"

    def _get_status_key(self, experiment_id: str) -> str:
        return f"{self.STATUS_KEY_PREFIX}{experiment_id}"

    def _get_logs_key(self, experiment_id: str) -> str:
        return f"{self.LOGS_KEY_PREFIX}{experiment_id}"

    def _get_session_key(self, experiment_id: str) -> str:
        return f"{self.SESSION_KEY_PREFIX}{experiment_id}"

    def create_session(self, experiment_id: str, config: dict = None):
        """
        创建新的训练会话

        Args:
            experiment_id: 实验/训练任务ID
            config: 训练配置信息
        """
        session_data = {
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

        # 内存存储
        if experiment_id in self.training_sessions:
            logger.warning(f"训练会话 {experiment_id} 已存在，将被覆盖")
        self.training_sessions[experiment_id] = session_data

        # Redis存储
        redis = self.redis_client
        if redis:
            try:
                redis.set(
                    self._get_session_key(experiment_id),
                    json.dumps(session_data),
                    ex=86400  # 24小时过期
                )
                # 初始化空的指标列表
                redis.delete(self._get_metrics_key(experiment_id))
                logger.info(f"创建训练会话（Redis）: {experiment_id}")
            except Exception as e:
                logger.error(f"Redis创建会话失败: {e}")

        logger.info(f"创建训练会话: {experiment_id}")

    def update_status(self, experiment_id: str, status: TrainingStatus):
        """
        更新训练状态

        Args:
            experiment_id: 实验/训练任务ID
            status: 新的训练状态
        """
        now = datetime.utcnow().isoformat() + "Z"

        # 更新内存
        if experiment_id in self.training_sessions:
            old_status = self.training_sessions[experiment_id]["status"]
            self.training_sessions[experiment_id]["status"] = status

            if status == TrainingStatus.RUNNING and not self.training_sessions[experiment_id]["started_at"]:
                self.training_sessions[experiment_id]["started_at"] = now
            elif status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.STOPPED]:
                self.training_sessions[experiment_id]["ended_at"] = now

            logger.info(f"训练 {experiment_id} 状态更新: {old_status} -> {status}")

        # 更新Redis状态键
        redis = self.redis_client
        if redis:
            try:
                redis.set(
                    self._get_status_key(experiment_id),
                    status,
                    ex=86400
                )

                # 同时更新Redis会话数据
                session_data = redis.get(self._get_session_key(experiment_id))
                if session_data:
                    session = json.loads(session_data)
                    session["status"] = status
                    if status == TrainingStatus.RUNNING and not session.get("started_at"):
                        session["started_at"] = now
                    elif status in [TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.STOPPED]:
                        session["ended_at"] = now
                    redis.set(
                        self._get_session_key(experiment_id),
                        json.dumps(session),
                        ex=86400
                    )
                    logger.info(f"Redis会话状态已更新: {experiment_id}, status={status}")
            except Exception as e:
                logger.error(f"Redis更新状态失败: {e}")

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
        # 验证日志级别
        if level.upper() not in self.log_levels:
            level = "INFO"

        log_entry = {
            "level": level.upper(),
            "message": message,
            "source": source,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # 内存存储
        if experiment_id in self.training_sessions:
            session = self.training_sessions[experiment_id]
            session["logs"].append(log_entry)
            if len(session["logs"]) > self.max_logs:
                session["logs"] = session["logs"][-self.max_logs:]

        return log_entry

    def add_metrics(
        self,
        experiment_id: str,
        epoch: int,
        metrics: dict,
        best_metric: float = None
    ):
        """
        添加训练指标

        Args:
            experiment_id: 实验/训练任务ID
            epoch: 当前epoch数
            metrics: 指标字典（包含train_loss, train_acc, val_loss, val_acc等）
            best_metric: 最佳指标值（可选）
        """
        metrics_entry = {
            "epoch": epoch,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **metrics
        }
        # 添加最佳指标（如果提供）
        if best_metric is not None:
            metrics_entry["best_metric"] = best_metric

        # 内存存储
        if experiment_id in self.training_sessions:
            session = self.training_sessions[experiment_id]
            session["metrics"].append(metrics_entry)
            session["current_epoch"] = epoch

        # Redis存储
        redis = self.redis_client
        if redis:
            try:
                # 存储指标到列表
                key = self._get_metrics_key(experiment_id)
                redis.rpush(key, json.dumps(metrics_entry))
                redis.ltrim(key, -1000, -1)
                redis.expire(key, 86400)

                # 更新会话数据中的current_epoch
                session_data = redis.get(self._get_session_key(experiment_id))
                if session_data:
                    session = json.loads(session_data)
                    session["current_epoch"] = epoch
                    redis.set(
                        self._get_session_key(experiment_id),
                        json.dumps(session),
                        ex=86400
                    )

                logger.debug(f"Redis存储指标: {experiment_id}, epoch={epoch}")
            except Exception as e:
                logger.error(f"Redis存储指标失败: {e}")

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
        # 优先从内存获取
        if experiment_id in self.training_sessions:
            logs = self.training_sessions[experiment_id]["logs"]
            if level:
                logs = [log for log in logs if log["level"] == level.upper()]
            return logs[-limit:]

        return []

    def get_metrics(
        self,
        experiment_id: str,
        limit: int = 100
    ) -> List[dict]:
        """
        获取训练指标 - 优先从Redis获取

        Args:
            experiment_id: 实验/训练任务ID
            limit: 返回的指标条数限制

        Returns:
            指标列表
        """
        # 先尝试从Redis获取
        redis = self.redis_client
        if redis:
            try:
                key = self._get_metrics_key(experiment_id)
                # 获取最新的N条指标
                metrics_data = redis.lrange(key, -limit, -1)
                if metrics_data:
                    metrics = [json.loads(data) for data in metrics_data]
                    logger.debug(f"从Redis获取指标: {experiment_id}, count={len(metrics)}")
                    return metrics
            except Exception as e:
                logger.error(f"从Redis获取指标失败: {e}")

        # 降级到内存
        if experiment_id in self.training_sessions:
            metrics = self.training_sessions[experiment_id]["metrics"]
            return metrics[-limit:]

        return []

    def get_session_info(self, experiment_id: str) -> Optional[dict]:
        """
        获取训练会话信息

        Args:
            experiment_id: 实验/训练任务ID

        Returns:
            会话信息字典
        """
        # 优先从Redis获取
        redis = self.redis_client
        if redis:
            try:
                session_data = redis.get(self._get_session_key(experiment_id))
                if session_data:
                    return json.loads(session_data)
            except Exception as e:
                logger.error(f"从Redis获取会话失败: {e}")

        # 降级到内存
        if experiment_id in self.training_sessions:
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

        return None

    def delete_session(self, experiment_id: str):
        """
        删除训练会话

        Args:
            experiment_id: 实验/训练任务ID
        """
        # 内存清理
        if experiment_id in self.training_sessions:
            del self.training_sessions[experiment_id]

        # Redis清理
        redis = self.redis_client
        if redis:
            try:
                redis.delete(self._get_session_key(experiment_id))
                redis.delete(self._get_metrics_key(experiment_id))
                redis.delete(self._get_status_key(experiment_id))
            except Exception as e:
                logger.error(f"Redis删除会话失败: {e}")

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
        logger.info(f"[WS_BROADCAST] 广播训练指标: experiment_id={experiment_id}, epoch={metrics_entry.get('epoch')}, metrics={metrics_entry}")
        await manager.send_training_update(experiment_id, {
            "type": "metrics_update",
            "data": metrics_entry
        })
        logger.info(f"[WS_BROADCAST] 指标广播完成: experiment_id={experiment_id}")

    async def broadcast_status(
        self,
        experiment_id: str,
        manager,
        best_metric: float = None
    ):
        """
        通过WebSocket广播状态变化

        Args:
            experiment_id: 实验/训练任务ID
            manager: WebSocket连接管理器
            best_metric: 最佳指标（可选，如果提供则包含在广播消息中）
        """
        # 尝试从Redis获取会话信息
        redis = self.redis_client

        # 默认值
        status = TrainingStatus.RUNNING
        current_epoch = 0
        total_epochs = 0
        started_at = None
        ended_at = None

        # 优先从Redis获取
        if redis:
            try:
                # 获取状态
                status_data = redis.get(self._get_status_key(experiment_id))
                if status_data:
                    status = status_data

                # 获取会话信息
                session_data = redis.get(self._get_session_key(experiment_id))
                if session_data:
                    session = json.loads(session_data)
                    current_epoch = session.get("current_epoch", 0)
                    total_epochs = session.get("total_epochs", 0)
                    started_at = session.get("started_at")
                    ended_at = session.get("ended_at")
                    # 如果会话中有best_metric且没有传入best_metric参数，使用会话中的值
                    if best_metric is None and "best_metric" in session:
                        best_metric = session.get("best_metric")
            except Exception as e:
                logger.error(f"从Redis获取状态信息失败: {e}")

        # 降级到内存
        if not current_epoch and experiment_id in self.training_sessions:
            session = self.training_sessions[experiment_id]
            current_epoch = session.get("current_epoch", 0)
            total_epochs = session.get("total_epochs", 0)
            started_at = session.get("started_at")
            ended_at = session.get("ended_at")
            # 如果内存中有best_metric且没有传入best_metric参数，使用内存中的值
            if best_metric is None and "best_metric" in session:
                best_metric = session.get("best_metric")

        # 构建消息数据
        message_data = {
            "status": status,
            "current_epoch": current_epoch,
            "total_epochs": total_epochs,
            "started_at": started_at,
            "ended_at": ended_at,
            "message": f"训练状态已更新为: {status}"
        }

        # 如果有best_metric，添加到消息中
        if best_metric is not None:
            message_data["best_metric"] = best_metric

        await manager.send_training_update(experiment_id, {
            "type": "status_change",
            "data": message_data
        })
        logger.info(f"[WS_BROADCAST] 广播状态变化: experiment_id={experiment_id}, status={status}, best_metric={best_metric}")


# 全局训练日志收集器实例
training_logger = TrainingLogger()
