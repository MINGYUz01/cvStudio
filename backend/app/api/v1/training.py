"""
训练相关API路由
提供训练任务的CRUD操作和控制接口
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional, List
from loguru import logger

from app.database import get_db
from app.services.training_service import TrainingService
from app.api.websocket import manager
from app.schemas.training import (
    TrainingRunCreate,
    TrainingRunResponse,
    TrainingRunUpdate,
    TrainingControlRequest,
    TrainingControlResponse,
    TrainingSaveRequest,
    TrainingSaveResponse,
    CheckpointInfo,
    MetricsEntry,
    LogEntry,
    ExperimentListItem
)

router = APIRouter()
training_service = TrainingService()


@router.get("/", response_model=List[ExperimentListItem])
async def list_training_runs(
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db)
):
    """
    获取训练任务列表

    Args:
        status: 状态过滤（可选）
        page: 页码（从1开始）
        page_size: 每页数量
        db: 数据库会话

    Returns:
        训练任务列表
    """
    try:
        skip = (page - 1) * page_size
        training_runs = training_service.get_training_runs(
            db=db,
            skip=skip,
            limit=page_size,
            status=status
        )

        # 转换为简化的响应格式
        result = [
            ExperimentListItem(
                id=run.id,
                name=run.name,
                status=run.status,
                task_type=run.hyperparams.get("task_type") if run.hyperparams else None,
                progress=run.progress,
                current_epoch=run.current_epoch,
                total_epochs=run.total_epochs,
                best_metric=run.best_metric,
                device=run.device,
                start_time=run.start_time,
                created_at=run.created_at
            )
            for run in training_runs
        ]

        return result

    except Exception as e:
        logger.error(f"获取训练列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/", response_model=TrainingRunResponse, status_code=status.HTTP_201_CREATED)
async def create_training_run(
    training_data: TrainingRunCreate,
    db: Session = Depends(get_db)
):
    """
    创建训练任务

    Args:
        training_data: 训练任务数据
        db: 数据库会话

    Returns:
        创建的训练任务
    """
    try:
        training_run = training_service.create_training_run(
            db=db,
            name=training_data.name,
            description=training_data.description or "",
            model_id=training_data.model_id,
            dataset_id=training_data.dataset_id,
            config=training_data.config,
            user_id=training_data.user_id
        )

        return training_run

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"创建训练任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{training_id}", response_model=TrainingRunResponse)
async def get_training_run(
    training_id: int,
    db: Session = Depends(get_db)
):
    """
    获取单个训练任务详情

    Args:
        training_id: 训练任务ID
        db: 数据库会话

    Returns:
        训练任务详情
    """
    training_run = training_service.get_training_run(db, training_id)

    if not training_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"训练任务不存在: {training_id}"
        )

    return training_run


@router.put("/{training_id}", response_model=TrainingRunResponse)
async def update_training_run(
    training_id: int,
    training_data: TrainingRunUpdate,
    db: Session = Depends(get_db)
):
    """
    更新训练任务（重命名、修改描述等）

    Args:
        training_id: 训练任务ID
        training_data: 更新数据
        db: 数据库会话

    Returns:
        更新后的训练任务
    """
    training_run = training_service.update_training_run(
        db=db,
        training_run_id=training_id,
        name=training_data.name,
        description=training_data.description
    )

    if not training_run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"训练任务不存在: {training_id}"
        )

    return training_run


@router.delete("/{training_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training_run(
    training_id: int,
    db: Session = Depends(get_db)
):
    """
    删除训练任务

    Args:
        training_id: 训练任务ID
        db: 数据库会话
    """
    success = training_service.delete_training_run(db, training_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"训练任务不存在: {training_id}"
        )

    return None


@router.post("/{training_id}/control", response_model=TrainingControlResponse)
async def control_training_run(
    training_id: int,
    request: TrainingControlRequest,
    db: Session = Depends(get_db)
):
    """
    控制训练任务（暂停、恢复、停止）

    Args:
        training_id: 训练任务ID
        request: 控制请求
        db: 数据库会话

    Returns:
        控制结果
    """
    try:
        result = training_service.control_training(
            training_run_id=training_id,
            action=request.action
        )

        # 广播状态变化
        experiment_id = f"exp_{training_id}"
        from app.utils.training_logger import training_logger
        import asyncio
        asyncio.create_task(training_logger.broadcast_status(experiment_id, manager))

        return TrainingControlResponse(
            success=result["success"],
            action=result["action"],
            task_id=result["task_id"],
            experiment_id=result["experiment_id"],
            message=f"训练已{result['action']}"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"控制训练任务失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{training_id}/metrics", response_model=List[MetricsEntry])
async def get_training_metrics(
    training_id: int,
    limit: int = 100
):
    """
    获取训练指标数据

    Args:
        training_id: 训练任务ID
        limit: 限制返回数量

    Returns:
        指标数据列表
    """
    try:
        metrics = training_service.get_training_metrics(
            training_run_id=training_id,
            limit=limit
        )

        return [
            MetricsEntry(
                epoch=m["epoch"],
                timestamp=m["timestamp"],
                train_loss=m.get("train_loss"),
                train_acc=m.get("train_acc"),
                val_loss=m.get("val_loss"),
                val_acc=m.get("val_acc"),
                extra_metrics={k: v for k, v in m.items()
                              if k not in ["epoch", "timestamp", "train_loss",
                                         "train_acc", "val_loss", "val_acc"]}
            )
            for m in metrics
        ]

    except Exception as e:
        logger.error(f"获取训练指标失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{training_id}/logs", response_model=List[LogEntry])
async def get_training_logs(
    training_id: int,
    level: Optional[str] = None,
    limit: int = 100
):
    """
    获取训练日志

    Args:
        training_id: 训练任务ID
        level: 日志级别过滤（可选）
        limit: 限制返回数量

    Returns:
        日志数据列表
    """
    try:
        logs = training_service.get_training_logs(
            training_run_id=training_id,
            level=level,
            limit=limit
        )

        return [
            LogEntry(
                level=log["level"],
                message=log["message"],
                source=log["source"],
                timestamp=log["timestamp"]
            )
            for log in logs
        ]

    except Exception as e:
        logger.error(f"获取训练日志失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/{training_id}/checkpoints", response_model=List[CheckpointInfo])
async def list_training_checkpoints(
    training_id: int,
    db: Session = Depends(get_db)
):
    """
    获取训练任务的所有checkpoint

    Args:
        training_id: 训练任务ID
        db: 数据库会话

    Returns:
        Checkpoint列表
    """
    try:
        from app.utils.checkpoint_manager import CheckpointManager

        checkpoint_dir = f"data/checkpoints/exp_{training_id}"
        checkpoint_manager = CheckpointManager(checkpoint_dir)

        checkpoints = checkpoint_manager.list_checkpoints(
            run_id=training_id,
            db=db
        )

        return [
            CheckpointInfo(**cp)
            for cp in checkpoints
        ]

    except Exception as e:
        logger.error(f"获取checkpoint列表失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/{training_id}/save", response_model=TrainingSaveResponse)
async def save_training_to_weights(
    training_id: int,
    request: TrainingSaveRequest,
    db: Session = Depends(get_db)
):
    """
    保存最佳checkpoint到权重库

    Args:
        training_id: 训练任务ID
        request: 保存请求
        db: 数据库会话

    Returns:
        保存结果
    """
    try:
        weights_path = training_service.save_to_weights(
            training_run_id=training_id,
            weights_dir=request.weights_dir
        )

        return TrainingSaveResponse(
            success=True,
            message="最佳模型已保存到权重库",
            path=weights_path
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"保存到权重库失败: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )