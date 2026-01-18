"""
训练相关API路由
提供训练任务的CRUD操作和控制接口
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional, List
from loguru import logger
import sys

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

# ============================================================================
# 调试日志辅助函数
# ============================================================================

def debug_log(msg: str, level: str = "INFO"):
    """输出调试日志到控制台和文件"""
    log_msg = f"[TRAINING_API] {msg}"
    print(f"[DEBUG] {log_msg}", flush=True)  # 强制输出到控制台
    if level == "INFO":
        logger.info(log_msg)
    elif level == "ERROR":
        logger.error(log_msg)
    elif level == "WARNING":
        logger.warning(log_msg)

debug_log("训练API模块已加载")


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
    debug_log(f"收到创建训练任务请求: name={training_data.name}, model_id={training_data.model_id}, dataset_id={training_data.dataset_id}")
    debug_log(f"训练配置: {training_data.config}", "INFO")

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

        debug_log(f"训练任务创建成功: id={training_run.id}, status={training_run.status}")
        return training_run

    except ValueError as e:
        debug_log(f"创建训练任务失败 (ValueError): {e}", "ERROR")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        debug_log(f"创建训练任务失败 (Exception): {e}", "ERROR")
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
    debug_log(f"获取训练任务详情: id={training_id}")
    training_run = training_service.get_training_run(db, training_id)

    if not training_run:
        debug_log(f"训练任务不存在: {training_id}", "ERROR")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"训练任务不存在: {training_id}"
        )

    return training_run


@router.post("/{training_id}/start")
async def start_training_run(
    training_id: int,
    db: Session = Depends(get_db)
):
    """
    启动训练任务

    Args:
        training_id: 训练任务ID
        db: 数据库会话

    Returns:
        启动结果
    """
    debug_log(f"收到启动训练请求: training_id={training_id}")

    try:
        # 获取训练任务
        training_run = training_service.get_training_run(db, training_id)
        if not training_run:
            debug_log(f"训练任务不存在: {training_id}", "ERROR")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"训练任务不存在: {training_id}"
            )

        debug_log(f"训练任务当前状态: {training_run.status}")

        # 检查状态
        if training_run.status in ["running", "queued"]:
            debug_log(f"训练任务已在运行或队列中: {training_run.status}", "WARNING")
            return {
                "success": True,
                "message": f"训练任务已在{training_run.status}状态",
                "task_id": training_run.celery_task_id
            }

        # 获取模型架构信息
        from app.services.model_service import ModelService
        model_service = ModelService()

        debug_log(f"获取模型架构信息: model_id={training_run.model_id}")
        model_code = model_service.get_model_code(db, training_run.model_id)
        if not model_code:
            debug_log(f"模型代码不存在: {training_run.model_id}", "ERROR")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"模型代码不存在: {training_run.model_id}"
            )

        model_arch = {
            "architecture": model_code.code_content,
            "class_name": model_code.code_content.get("class_name", "Model"),
            "input_size": model_code.code_content.get("input_size", 224)
        }

        # 获取数据集信息
        from app.services.dataset_service import DatasetService
        dataset_service = DatasetService()

        debug_log(f"获取数据集信息: dataset_id={training_run.dataset_id}")
        dataset = dataset_service.get_dataset(db, training_run.dataset_id)
        if not dataset:
            debug_log(f"数据集不存在: {training_id}", "ERROR")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"数据集不存在: {training_id}"
            )

        dataset_info = {
            "id": dataset.id,
            "name": dataset.name,
            "format": dataset.format,
            "root_path": dataset.root_path,
            "num_classes": dataset.num_classes,
            "num_images": dataset.num_images
        }

        debug_log(f"模型架构: {model_arch.get('class_name')}, 数据集: {dataset_info['name']}")

        # 调用启动训练方法
        debug_log("调用 training_service.start_training...")
        task_id = training_service.start_training(
            training_run_id=training_id,
            model_arch=model_arch,
            dataset_info=dataset_info
        )

        debug_log(f"训练任务已提交到Celery: task_id={task_id}", "INFO")

        # 更新数据库中的Celery任务ID
        training_run.celery_task_id = task_id
        db.commit()

        return {
            "success": True,
            "message": "训练任务已启动",
            "task_id": task_id
        }

    except HTTPException:
        raise
    except Exception as e:
        debug_log(f"启动训练任务失败: {e}", "ERROR")
        import traceback
        debug_log(f"错误堆栈: {traceback.format_exc()}", "ERROR")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"启动训练失败: {str(e)}"
        )


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