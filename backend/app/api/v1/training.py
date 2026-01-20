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
from app.utils.training_logger import training_logger
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


@router.get("/")
async def list_training_runs(
    status: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    获取训练任务列表

    Args:
        status: 状态过滤（可选）
        skip: 跳过记录数
        limit: 限制返回数量
        db: 数据库会话

    Returns:
        训练任务列表（完整TrainingRun格式）
    """
    try:
        training_runs = training_service.get_training_runs(
            db=db,
            skip=skip,
            limit=limit,
            status=status
        )

        debug_log(f"获取到 {len(training_runs)} 条训练任务")

        # 直接返回TrainingRun对象列表（会被序列化为JSON）
        result = []
        for run in training_runs:
            result.append({
                "id": run.id,
                "name": run.name,
                "description": run.description,
                "model_id": run.model_id,
                "dataset_id": run.dataset_id,
                "hyperparams": run.hyperparams,
                "status": run.status,
                "progress": run.progress,
                "current_epoch": run.current_epoch,
                "total_epochs": run.total_epochs,
                "best_metric": run.best_metric,
                "device": run.device,
                "log_file": run.log_file,
                "error_message": run.error_message,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "updated_at": run.updated_at.isoformat() if run.updated_at else None,
            })

        # 打印第一个任务的状态用于调试
        if result:
            first = result[0]
            debug_log(f"第一个任务: id={first['id']}, status={first['status']}, progress={first['progress']}, "
                      f"epoch={first['current_epoch']}/{first['total_epochs']}, best_metric={first['best_metric']}")

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


@router.get("/{training_id}")
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
        训练任务详情（完整TrainingRun格式）
    """
    debug_log(f"获取训练任务详情: id={training_id}")
    training_run = training_service.get_training_run(db, training_id)

    if not training_run:
        debug_log(f"训练任务不存在: {training_id}", "ERROR")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"训练任务不存在: {training_id}"
        )

    # 添加调试日志
    debug_log(f"训练任务状态: status={training_run.status}, progress={training_run.progress}, "
              f"current_epoch={training_run.current_epoch}, best_metric={training_run.best_metric}")

    # 返回完整格式
    return {
        "id": training_run.id,
        "name": training_run.name,
        "description": training_run.description,
        "model_id": training_run.model_id,
        "dataset_id": training_run.dataset_id,
        "hyperparams": training_run.hyperparams,
        "status": training_run.status,
        "progress": training_run.progress,
        "current_epoch": training_run.current_epoch,
        "total_epochs": training_run.total_epochs,
        "best_metric": training_run.best_metric,
        "device": training_run.device,
        "log_file": training_run.log_file,
        "error_message": training_run.error_message,
        "start_time": training_run.start_time.isoformat() if training_run.start_time else None,
        "end_time": training_run.end_time.isoformat() if training_run.end_time else None,
        "created_at": training_run.created_at.isoformat() if training_run.created_at else None,
        "updated_at": training_run.updated_at.isoformat() if training_run.updated_at else None,
    }


@router.get("/{training_id}/metrics")
async def get_training_metrics(
    training_id: int,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    获取训练任务的历史指标数据

    Args:
        training_id: 训练任务ID
        limit: 返回的指标条数限制
        db: 数据库会话

    Returns:
        指标数据列表
    """
    experiment_id = f"exp_{training_id}"
    metrics = training_logger.get_metrics(experiment_id, limit)

    return {
        "experiment_id": experiment_id,
        "training_id": training_id,
        "metrics": metrics,
        "count": len(metrics)
    }


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

        # 获取模型架构信息（支持 Model 表和 PresetModel 表）
        from app.models.model import Model, PresetModel
        import json

        debug_log(f"获取模型架构信息: model_id={training_run.model_id}")

        # 首先尝试从 Model 表获取
        model = db.query(Model).filter(Model.id == training_run.model_id).first()

        if model:
            # 从 graph_json 获取模型架构信息
            graph_json = model.graph_json if isinstance(model.graph_json, dict) else json.loads(model.graph_json) if model.graph_json else {}
            model_arch = {
                "architecture": graph_json,
                "class_name": graph_json.get("class_name", "Model"),
                "input_size": graph_json.get("input_size", 224),
                "code_path": model.code_path
            }
            debug_log(f"从 Model 表获取模型: class_name={model_arch['class_name']}")
        else:
            # 尝试从 PresetModel 表获取
            debug_log(f"Model 表中未找到 model_id={training_run.model_id}，尝试从 PresetModel 获取")
            preset_model = db.query(PresetModel).filter(PresetModel.id == training_run.model_id).first()

            if not preset_model:
                debug_log(f"PresetModel 中也未找到: model_id={training_run.model_id}", "ERROR")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"模型不存在: {training_run.model_id}"
                )

            # 从 PresetModel 的 architecture_data 获取模型信息
            arch_data = preset_model.architecture_data if isinstance(preset_model.architecture_data, dict) else {}
            extra_meta = preset_model.extra_metadata if isinstance(preset_model.extra_metadata, dict) else {}

            # 提取 input_size，优先从 extra_metadata.input_shape 获取
            input_size = 224
            if extra_meta.get("input_shape") and len(extra_meta["input_shape"]) >= 3:
                input_size = extra_meta["input_shape"][2]  # [C, H, W] 取 W
            elif arch_data.get("input_size"):
                input_size = arch_data["input_size"]
            elif extra_meta.get("input_size"):
                input_size = extra_meta["input_size"]

            # 生成 class_name
            class_name = arch_data.get("class_name") or extra_meta.get("class_name")
            if not class_name:
                # 从预设模型名称生成类名
                class_name = preset_model.name.replace("-", "").replace(" ", "_").replace(".", "")

            # PresetModel 到预训练模型的映射
            # 根据预设模型的 category 和 name 选择合适的预训练模型
            PRESET_MODEL_MAP = {
                # 分类任务
                "Image Classification": "resnet18",
                "LeNet-5": "lenet",  # 简单模型，会在 ModelFactory 中特殊处理
                "Simple CNN": "simple_cnn",
                "AlexNet": "alexnet",
                "VGG-16": "vgg16",
                "ResNet-18": "resnet18",
                # 检测任务
                "YOLO-like Detection": "yolov5n",
            }

            # 获取预设模型对应的预训练模型名称
            pretrained_model_name = PRESET_MODEL_MAP.get(
                preset_model.name,
                # 默认根据 category 选择
                "resnet18" if preset_model.category == "classification" else "yolov5n"
            )

            model_arch = {
                "architecture": pretrained_model_name,  # 使用字符串而不是字典
                "class_name": class_name,
                "input_size": input_size,
                "code_path": None,
                "preset_model_id": preset_model.id,
                "preset_model_name": preset_model.name,
                "category": preset_model.category
            }
            debug_log(f"从 PresetModel 表获取模型: name={preset_model.name}, 使用预训练模型={pretrained_model_name}")

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
            "path": dataset.path,
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

    如果任务正在运行或排队中，会先取消对应的Celery任务

    Args:
        training_id: 训练任务ID
        db: 数据库会话
    """
    debug_log(f"收到删除训练任务请求: training_id={training_id}")

    # 获取训练任务（用于检查状态和获取celery_task_id）
    training_run = training_service.get_training_run(db, training_id)

    if not training_run:
        debug_log(f"训练任务不存在: {training_id}", "ERROR")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"训练任务不存在: {training_id}"
        )

    # 如果任务正在运行或排队中，取消Celery任务
    if training_run.celery_task_id and training_run.status in ["running", "queued"]:
        try:
            from backend.celery_app import celery_app
            celery_app.control.revoke(training_run.celery_task_id, terminate=True, signal='SIGKILL')
            debug_log(f"已取消Celery任务: {training_run.celery_task_id}, 状态: {training_run.status}")
        except Exception as e:
            debug_log(f"取消Celery任务失败: {e}", "WARNING")
            # 继续删除流程

    # 删除训练任务
    success = training_service.delete_training_run(db, training_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"训练任务不存在: {training_id}"
        )

    debug_log(f"训练任务已删除: {training_id}")
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
        try:
            await training_logger.broadcast_status(experiment_id, manager)
        except Exception as e:
            debug_log(f"广播状态变化失败: {e}", "WARNING")

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