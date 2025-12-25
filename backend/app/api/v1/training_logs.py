"""
训练日志管理API
提供训练日志的创建、查询和管理接口
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Dict
from loguru import logger

from app.utils.training_logger import training_logger, TrainingStatus
from app.api.websocket import manager

router = APIRouter()


class CreateSessionRequest(BaseModel):
    """创建训练会话请求"""
    experiment_id: str
    config: Optional[dict] = {}
    total_epochs: Optional[int] = 0


class UpdateStatusRequest(BaseModel):
    """更新训练状态请求"""
    status: TrainingStatus


class AddLogRequest(BaseModel):
    """添加日志请求"""
    level: str = "INFO"
    message: str
    source: str = "trainer"


class AddMetricsRequest(BaseModel):
    """添加指标请求"""
    epoch: int
    metrics: Dict


@router.post("/logs/session")
async def create_training_session(request: CreateSessionRequest):
    """
    创建新的训练会话

    Args:
        request: 创建会话请求

    Returns:
        创建的会话信息
    """
    try:
        training_logger.create_session(
            experiment_id=request.experiment_id,
            config=request.config
        )

        # 设置总epoch数
        if request.experiment_id in training_logger.training_sessions:
            training_logger.training_sessions[request.experiment_id]["total_epochs"] = request.total_epochs

        # 广播状态变化
        await training_logger.broadcast_status(request.experiment_id, manager)

        return {
            "success": True,
            "message": f"训练会话 {request.experiment_id} 已创建",
            "data": training_logger.get_session_info(request.experiment_id)
        }
    except Exception as e:
        logger.error(f"创建训练会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建训练会话失败: {str(e)}"
        )


@router.put("/logs/{experiment_id}/status")
async def update_training_status(experiment_id: str, request: UpdateStatusRequest):
    """
    更新训练状态

    Args:
        experiment_id: 实验/训练任务ID
        request: 状态更新请求

    Returns:
        更新结果
    """
    try:
        training_logger.update_status(experiment_id, request.status)

        # 广播状态变化
        await training_logger.broadcast_status(experiment_id, manager)

        return {
            "success": True,
            "message": f"训练状态已更新为 {request.status}",
            "data": training_logger.get_session_info(experiment_id)
        }
    except Exception as e:
        logger.error(f"更新训练状态失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新训练状态失败: {str(e)}"
        )


@router.post("/logs/{experiment_id}/log")
async def add_training_log(experiment_id: str, request: AddLogRequest):
    """
    添加训练日志

    Args:
        experiment_id: 实验/训练任务ID
        request: 日志请求

    Returns:
        添加的日志条目
    """
    try:
        log_entry = training_logger.add_log(
            experiment_id=experiment_id,
            level=request.level,
            message=request.message,
            source=request.source
        )

        # 广播日志
        if log_entry:
            await training_logger.broadcast_log(experiment_id, log_entry, manager)

        return {
            "success": True,
            "message": "日志已添加",
            "data": log_entry
        }
    except Exception as e:
        logger.error(f"添加训练日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加训练日志失败: {str(e)}"
        )


@router.post("/logs/{experiment_id}/metrics")
async def add_training_metrics(experiment_id: str, request: AddMetricsRequest):
    """
    添加训练指标

    Args:
        experiment_id: 实验/训练任务ID
        request: 指标请求

    Returns:
        添加的指标条目
    """
    try:
        metrics_entry = training_logger.add_metrics(
            experiment_id=experiment_id,
            epoch=request.epoch,
            metrics=request.metrics
        )

        # 广播指标
        if metrics_entry:
            await training_logger.broadcast_metrics(experiment_id, metrics_entry, manager)

        return {
            "success": True,
            "message": "指标已添加",
            "data": metrics_entry
        }
    except Exception as e:
        logger.error(f"添加训练指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"添加训练指标失败: {str(e)}"
        )


@router.get("/logs/{experiment_id}")
async def get_training_logs(
    experiment_id: str,
    level: Optional[str] = None,
    limit: int = 100
):
    """
    获取训练日志

    Args:
        experiment_id: 实验/训练任务ID
        level: 日志级别过滤（可选）
        limit: 返回的日志条数限制

    Returns:
        日志列表
    """
    try:
        logs = training_logger.get_logs(experiment_id, level=level, limit=limit)

        return {
            "success": True,
            "data": logs,
            "count": len(logs)
        }
    except Exception as e:
        logger.error(f"获取训练日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取训练日志失败: {str(e)}"
        )


@router.get("/logs/{experiment_id}/metrics")
async def get_training_metrics(
    experiment_id: str,
    limit: int = 100
):
    """
    获取训练指标

    Args:
        experiment_id: 实验/训练任务ID
        limit: 返回的指标条数限制

    Returns:
        指标列表
    """
    try:
        metrics = training_logger.get_metrics(experiment_id, limit=limit)

        return {
            "success": True,
            "data": metrics,
            "count": len(metrics)
        }
    except Exception as e:
        logger.error(f"获取训练指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取训练指标失败: {str(e)}"
        )


@router.get("/logs/{experiment_id}/info")
async def get_session_info(experiment_id: str):
    """
    获取训练会话信息

    Args:
        experiment_id: 实验/训练任务ID

    Returns:
        会话信息
    """
    try:
        info = training_logger.get_session_info(experiment_id)

        if not info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"训练会话 {experiment_id} 不存在"
            )

        return {
            "success": True,
            "data": info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取会话信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取会话信息失败: {str(e)}"
        )


@router.delete("/logs/{experiment_id}")
async def delete_training_session(experiment_id: str):
    """
    删除训练会话

    Args:
        experiment_id: 实验/训练任务ID

    Returns:
        删除结果
    """
    try:
        training_logger.delete_session(experiment_id)

        return {
            "success": True,
            "message": f"训练会话 {experiment_id} 已删除"
        }
    except Exception as e:
        logger.error(f"删除训练会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除训练会话失败: {str(e)}"
        )


@router.get("/logs/sessions")
async def list_sessions():
    """
    列出所有训练会话

    Returns:
        会话列表
    """
    try:
        sessions = []
        for exp_id in training_logger.training_sessions:
            sessions.append(training_logger.get_session_info(exp_id))

        return {
            "success": True,
            "data": sessions,
            "count": len(sessions)
        }
    except Exception as e:
        logger.error(f"列出会话失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"列出会话失败: {str(e)}"
        )
