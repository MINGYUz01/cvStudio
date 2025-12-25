"""
WebSocket连接管理器
管理所有WebSocket连接，提供消息广播和推送功能
"""

from typing import Dict, Set, List
from fastapi import WebSocket
from loguru import logger
import json


class ConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self):
        # 活跃连接存储 {client_id: WebSocket}
        self.active_connections: Dict[str, WebSocket] = {}

        # 系统状态订阅者
        self.system_subscribers: Set[str] = set()

        # 训练任务订阅者 {experiment_id: Set[client_id]}
        self.training_subscribers: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """
        接受新的WebSocket连接

        Args:
            websocket: WebSocket连接对象
            client_id: 客户端唯一标识
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket客户端已连接: {client_id}")

    def disconnect(self, client_id: str):
        """
        断开WebSocket连接并清理订阅

        Args:
            client_id: 客户端唯一标识
        """
        # 从活跃连接中移除
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        # 从系统订阅者中移除
        if client_id in self.system_subscribers:
            self.system_subscribers.remove(client_id)

        # 从训练订阅者中移除
        for exp_id in list(self.training_subscribers.keys()):
            if client_id in self.training_subscribers[exp_id]:
                self.training_subscribers[exp_id].discard(client_id)
                # 如果该训练任务没有订阅者了，删除该条目
                if not self.training_subscribers[exp_id]:
                    del self.training_subscribers[exp_id]

        logger.info(f"WebSocket客户端已断开: {client_id}")

    async def send_personal_message(self, message: dict, client_id: str):
        """
        向指定客户端发送消息

        Args:
            message: 消息内容（字典格式）
            client_id: 客户端唯一标识
        """
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"向客户端 {client_id} 发送消息失败: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: dict):
        """
        向所有连接的客户端广播消息

        Args:
            message: 消息内容（字典格式）
        """
        # 复制一份客户端列表，避免在迭代时修改
        disconnected_clients = []

        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"向客户端 {client_id} 广播消息失败: {e}")
                disconnected_clients.append(client_id)

        # 清理断开的连接
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    async def broadcast_to_subscribers(self, message: dict, subscribers: Set[str]):
        """
        向指定订阅者集合广播消息

        Args:
            message: 消息内容（字典格式）
            subscribers: 订阅者客户端ID集合
        """
        disconnected_clients = []

        for client_id in subscribers:
            if client_id in self.active_connections:
                try:
                    websocket = self.active_connections[client_id]
                    await websocket.send_json(message)
                except Exception as e:
                    logger.error(f"向订阅者 {client_id} 发送消息失败: {e}")
                    disconnected_clients.append(client_id)

        # 清理断开的连接
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    def subscribe_system(self, client_id: str):
        """
        订阅系统状态更新

        Args:
            client_id: 客户端唯一标识
        """
        self.system_subscribers.add(client_id)
        logger.debug(f"客户端 {client_id} 订阅系统状态")

    def unsubscribe_system(self, client_id: str):
        """
        取消订阅系统状态更新

        Args:
            client_id: 客户端唯一标识
        """
        self.system_subscribers.discard(client_id)
        logger.debug(f"客户端 {client_id} 取消订阅系统状态")

    def subscribe_training(self, client_id: str, experiment_id: str):
        """
        订阅特定训练任务的更新

        Args:
            client_id: 客户端唯一标识
            experiment_id: 实验/训练任务ID
        """
        if experiment_id not in self.training_subscribers:
            self.training_subscribers[experiment_id] = set()

        self.training_subscribers[experiment_id].add(client_id)
        logger.debug(f"客户端 {client_id} 订阅训练任务 {experiment_id}")

    def unsubscribe_training(self, client_id: str, experiment_id: str):
        """
        取消订阅特定训练任务

        Args:
            client_id: 客户端唯一标识
            experiment_id: 实验/训练任务ID
        """
        if experiment_id in self.training_subscribers:
            self.training_subscribers[experiment_id].discard(client_id)
            # 如果没有订阅者了，删除该条目
            if not self.training_subscribers[experiment_id]:
                del self.training_subscribers[experiment_id]

        logger.debug(f"客户端 {client_id} 取消订阅训练任务 {experiment_id}")

    async def send_system_update(self, message: dict):
        """
        向所有系统状态订阅者发送更新

        Args:
            message: 系统状态消息
        """
        await self.broadcast_to_subscribers(message, self.system_subscribers)

    async def send_training_update(self, experiment_id: str, message: dict):
        """
        向特定训练任务的订阅者发送更新

        Args:
            experiment_id: 实验/训练任务ID
            message: 训练更新消息
        """
        if experiment_id in self.training_subscribers:
            await self.broadcast_to_subscribers(
                message,
                self.training_subscribers[experiment_id]
            )

    def get_connection_count(self) -> int:
        """获取当前活跃连接数"""
        return len(self.active_connections)

    def get_subscriber_count(self) -> Dict[str, int]:
        """获取订阅者统计信息"""
        return {
            "system": len(self.system_subscribers),
            "training": sum(len(subs) for subs in self.training_subscribers.values()),
            "training_tasks": len(self.training_subscribers)
        }


# 全局连接管理器实例
manager = ConnectionManager()
