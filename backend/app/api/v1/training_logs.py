"""
训练日志管理API
提供训练日志的创建、查询和管理接口
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from loguru import logger
import shutil
import tempfile
import os

from app.utils.training_logger import training_logger, TrainingStatus
from app.api.websocket import manager
from app.utils.experiment_manager import ExperimentManager

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


@router.post("/session")
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


@router.put("/{experiment_id}/status")
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


@router.post("/{experiment_id}/log")
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


@router.post("/{experiment_id}/metrics")
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


@router.get("/{experiment_id}")
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


@router.get("/{experiment_id}/metrics")
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


@router.get("/{experiment_id}/info")
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


@router.delete("/{experiment_id}")
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


@router.get("/sessions")
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


# ==================== 实验文件管理API ====================

@router.get("/exp/{experiment_id}/info")
async def get_experiment_info(experiment_id: str):
    """
    获取实验目录信息

    Args:
        experiment_id: 实验ID (格式: exp_1 或直接使用数字 1)

    Returns:
        实验信息
    """
    try:
        # 解析实验ID
        if experiment_id.startswith("exp_"):
            exp_id = int(experiment_id.split("_")[1])
        else:
            exp_id = int(experiment_id)

        experiment_manager = ExperimentManager(exp_id)
        info = experiment_manager.get_experiment_info()

        if not info["exists"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"实验目录不存在: {experiment_id}"
            )

        return {
            "success": True,
            "data": info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实验信息失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取实验信息失败: {str(e)}"
        )


@router.get("/exp/{experiment_id}/files")
async def list_experiment_files(experiment_id: str):
    """
    列出实验目录下的所有文件

    Args:
        experiment_id: 实验ID

    Returns:
        文件列表
    """
    try:
        # 解析实验ID
        if experiment_id.startswith("exp_"):
            exp_id = int(experiment_id.split("_")[1])
        else:
            exp_id = int(experiment_id)

        experiment_manager = ExperimentManager(exp_id)
        files = experiment_manager.get_files_list()

        return {
            "success": True,
            "data": files,
            "count": len(files)
        }
    except Exception as e:
        logger.error(f"列出实验文件失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"列出实验文件失败: {str(e)}"
        )


@router.get("/exp/{experiment_id}/config")
async def get_experiment_config(experiment_id: str):
    """
    获取实验配置

    Args:
        experiment_id: 实验ID

    Returns:
        配置数据
    """
    try:
        # 解析实验ID
        if experiment_id.startswith("exp_"):
            exp_id = int(experiment_id.split("_")[1])
        else:
            exp_id = int(experiment_id)

        experiment_manager = ExperimentManager(exp_id)
        config = experiment_manager.load_config()

        if config is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"配置文件不存在: {experiment_id}"
            )

        return {
            "success": True,
            "data": config
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实验配置失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取实验配置失败: {str(e)}"
        )


@router.get("/exp/{experiment_id}/metrics")
async def get_experiment_metrics(
    experiment_id: str,
    limit: int = -1
):
    """
    获取实验指标历史

    Args:
        experiment_id: 实验ID
        limit: 返回的条数限制，-1表示全部

    Returns:
        指标列表
    """
    try:
        # 解析实验ID
        if experiment_id.startswith("exp_"):
            exp_id = int(experiment_id.split("_")[1])
        else:
            exp_id = int(experiment_id)

        experiment_manager = ExperimentManager(exp_id)
        metrics = experiment_manager.load_metrics()

        if limit > 0:
            metrics = metrics[-limit:]

        return {
            "success": True,
            "data": metrics,
            "count": len(metrics)
        }
    except Exception as e:
        logger.error(f"获取实验指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取实验指标失败: {str(e)}"
        )


@router.get("/exp/{experiment_id}/log")
async def get_experiment_log(
    experiment_id: str,
    num_lines: int = 100
):
    """
    获取实验日志内容

    Args:
        experiment_id: 实验ID
        num_lines: 返回的行数，-1表示全部

    Returns:
        日志内容
    """
    try:
        # 解析实验ID
        if experiment_id.startswith("exp_"):
            exp_id = int(experiment_id.split("_")[1])
        else:
            exp_id = int(experiment_id)

        experiment_manager = ExperimentManager(exp_id)
        log_lines = experiment_manager.get_log_content(num_lines)

        return {
            "success": True,
            "data": "".join(log_lines),
            "lines": len(log_lines)
        }
    except Exception as e:
        logger.error(f"获取实验日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取实验日志失败: {str(e)}"
        )


@router.get("/exp/{experiment_id}/checkpoints")
async def list_experiment_checkpoints(experiment_id: str):
    """
    列出实验的checkpoint文件

    Args:
        experiment_id: 实验ID

    Returns:
        checkpoint列表
    """
    try:
        # 解析实验ID
        if experiment_id.startswith("exp_"):
            exp_id = int(experiment_id.split("_")[1])
        else:
            exp_id = int(experiment_id)

        experiment_manager = ExperimentManager(exp_id)
        checkpoints = experiment_manager.list_checkpoints()

        return {
            "success": True,
            "data": checkpoints,
            "count": len(checkpoints)
        }
    except Exception as e:
        logger.error(f"列出checkpoint失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"列出checkpoint失败: {str(e)}"
        )


@router.get("/exp/{experiment_id}/download")
async def download_experiment(experiment_id: str):
    """
    下载实验文件夹（压缩包）

    Args:
        experiment_id: 实验ID

    Returns:
        压缩文件
    """
    try:
        # 解析实验ID
        if experiment_id.startswith("exp_"):
            exp_id = int(experiment_id.split("_")[1])
        else:
            exp_id = int(experiment_id)

        experiment_manager = ExperimentManager(exp_id)

        if not experiment_manager.experiment_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"实验目录不存在: {experiment_id}"
            )

        # 创建临时zip文件
        temp_dir = tempfile.gettempdir()
        zip_filename = f"exp_{exp_id}.zip"
        zip_path = os.path.join(temp_dir, zip_filename)

        # 创建zip文件
        shutil.make_archive(
            base_name=os.path.join(temp_dir, f"exp_{exp_id}"),
            format="zip",
            root_dir=experiment_manager.experiment_dir.parent,
            base_dir=experiment_manager.experiment_dir.name
        )

        return FileResponse(
            path=zip_path,
            filename=zip_filename,
            media_type="application/zip"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载实验失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"下载实验失败: {str(e)}"
        )


@router.get("/exp/list")
async def list_all_experiments():
    """
    列出所有实验

    Returns:
        实验列表
    """
    try:
        experiments = ExperimentManager.list_all_experiments()

        return {
            "success": True,
            "data": experiments,
            "count": len(experiments)
        }
    except Exception as e:
        logger.error(f"列出实验失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"列出实验失败: {str(e)}"
        )
