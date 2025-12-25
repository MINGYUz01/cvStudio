"""
WebSocket API路由
提供系统状态流和训练日志流的WebSocket端点
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from loguru import logger
import uuid
from app.api.websocket import manager

router = APIRouter()


@router.websocket("/ws/system")
async def websocket_system_stats(
    websocket: WebSocket,
    client_id: str = Query(None)
):
    """
    系统状态WebSocket端点

    连接URL: ws://localhost:8000/api/v1/ws/system?client_id=xxx

    消息格式:
    {
        "type": "system_stats",
        "data": {
            "gpu_util": 75.5,
            "gpu_temp": 65,
            "vram_used": 4.2,
            "vram_total": 8.0,
            "cpu_util": 45.2,
            "ram_used": 8.5,
            "ram_total": 16.0,
            "timestamp": "2025-12-25T10:30:00Z"
        }
    }
    """
    # 生成或使用提供的client_id
    if not client_id:
        client_id = str(uuid.uuid4())

    # 接受连接
    await manager.connect(websocket, client_id)

    # 订阅系统状态更新
    manager.subscribe_system(client_id)

    try:
        # 发送连接确认消息
        await websocket.send_json({
            "type": "connection_established",
            "data": {
                "client_id": client_id,
                "subscription": "system_stats",
                "message": "已成功订阅系统状态更新"
            }
        })

        logger.info(f"客户端 {client_id} 已连接到系统状态流")

        # 保持连接并接收客户端消息
        while True:
            # 接收客户端消息（用于保活和控制）
            data = await websocket.receive_text()

            # 处理客户端消息
            logger.debug(f"从客户端 {client_id} 收到消息: {data}")

    except WebSocketDisconnect:
        logger.info(f"客户端 {client_id} 主动断开系统状态流连接")
    except Exception as e:
        logger.error(f"系统状态流连接错误 (客户端: {client_id}): {e}")
    finally:
        # 清理连接和订阅
        manager.disconnect(client_id)


@router.websocket("/ws/training/{experiment_id}")
async def websocket_training_logs(
    websocket: WebSocket,
    experiment_id: str,
    client_id: str = Query(None)
):
    """
    训练日志WebSocket端点

    连接URL: ws://localhost:8000/api/v1/ws/training/{experiment_id}?client_id=xxx

    消息格式:
    1. 日志消息:
    {
        "type": "log",
        "data": {
            "level": "INFO",
            "message": "Epoch 10/100 - Loss: 0.054",
            "timestamp": "2025-12-25T10:30:00Z"
        }
    }

    2. 指标更新:
    {
        "type": "metrics_update",
        "data": {
            "epoch": 10,
            "train_loss": 0.054,
            "train_acc": 0.92,
            "val_loss": 0.062,
            "val_acc": 0.89,
            "timestamp": "2025-12-25T10:30:00Z"
        }
    }

    3. 状态变化:
    {
        "type": "status_change",
        "data": {
            "status": "completed",
            "start_time": "2025-12-25T09:00:00Z",
            "end_time": "2025-12-25T10:30:00Z",
            "message": "训练已完成"
        }
    }
    """
    # 生成或使用提供的client_id
    if not client_id:
        client_id = str(uuid.uuid4())

    # 接受连接
    await manager.connect(websocket, client_id)

    # 订阅训练任务更新
    manager.subscribe_training(client_id, experiment_id)

    try:
        # 发送连接确认消息
        await websocket.send_json({
            "type": "connection_established",
            "data": {
                "client_id": client_id,
                "subscription": f"training_logs:{experiment_id}",
                "experiment_id": experiment_id,
                "message": f"已成功订阅训练任务 {experiment_id} 的日志更新"
            }
        })

        logger.info(f"客户端 {client_id} 已连接到训练任务 {experiment_id} 的日志流")

        # 保持连接并接收客户端消息
        while True:
            # 接收客户端消息（用于保活和控制）
            data = await websocket.receive_text()

            # 处理客户端消息
            logger.debug(f"从客户端 {client_id} 收到消息: {data}")

    except WebSocketDisconnect:
        logger.info(f"客户端 {client_id} 主动断开训练日志流连接 (实验: {experiment_id})")
    except Exception as e:
        logger.error(f"训练日志流连接错误 (客户端: {client_id}, 实验: {experiment_id}): {e}")
    finally:
        # 清理连接和订阅
        manager.disconnect(client_id)


@router.get("/ws/stats")
async def get_websocket_stats():
    """
    获取WebSocket连接统计信息

    Returns:
        {
            "active_connections": 10,
            "subscribers": {
                "system": 5,
                "training": 8,
                "training_tasks": 3
            }
        }
    """
    return {
        "active_connections": manager.get_connection_count(),
        "subscribers": manager.get_subscriber_count()
    }
