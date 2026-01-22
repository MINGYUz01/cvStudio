"""
训练相关的Pydantic Schema定义
用于API请求和响应的数据验证
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime

# 使用TYPE_CHECKING避免循环导入
if TYPE_CHECKING:
    from app.schemas.weight_library import (
        DatasetInfo,
        ModelArchitectureInfo,
        PretrainedWeightInfo,
        AugmentationConfigInfo
    )


class TrainingRunCreate(BaseModel):
    """创建训练任务请求"""

    name: str = Field(..., description="训练任务名称", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="训练任务描述")
    model_id: int = Field(..., description="模型架构ID")
    dataset_id: int = Field(..., description="数据集ID")
    config: Dict[str, Any] = Field(..., description="训练配置参数")
    user_id: int = Field(..., description="创建用户ID")
    pretrained_weight_id: Optional[int] = Field(None, description="预训练权重ID")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "YOLO训练实验",
                "description": "使用COCO数据集训练YOLO模型",
                "model_id": 1,
                "dataset_id": 1,
                "config": {
                    "task_type": "detection",
                    "epochs": 100,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "optimizer": "Adam",
                    "device": "cuda:0"
                },
                "user_id": 1
            }
        }


class TrainingRunUpdate(BaseModel):
    """更新训练任务请求"""

    name: Optional[str] = Field(None, description="新名称", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="新描述")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "更新后的名称",
                "description": "更新后的描述"
            }
        }


class TrainingRunResponse(BaseModel):
    """训练任务响应"""

    id: int
    name: str
    description: Optional[str]
    model_id: int
    dataset_id: int
    hyperparams: Dict[str, Any]
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    best_metric: Optional[float]
    device: str
    log_file: Optional[str]
    error_message: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "YOLO训练实验",
                "description": "使用COCO数据集训练YOLO模型",
                "model_id": 1,
                "dataset_id": 1,
                "hyperparams": {
                    "task_type": "detection",
                    "epochs": 100
                },
                "status": "running",
                "progress": 45.5,
                "current_epoch": 45,
                "total_epochs": 100,
                "best_metric": 0.8234,
                "device": "cuda:0",
                "log_file": "data/logs/training_exp_1.log",
                "error_message": None,
                "start_time": "2025-01-04T10:30:00Z",
                "end_time": None,
                "created_at": "2025-01-04T10:30:00Z",
                "updated_at": "2025-01-04T12:45:00Z"
            }
        }


class TrainingControlRequest(BaseModel):
    """训练控制请求"""

    action: str = Field(
        ...,
        description="控制动作",
        pattern="^(pause|resume|stop)$"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {"action": "pause"},
                {"action": "resume"},
                {"action": "stop"}
            ]
        }


class TrainingControlResponse(BaseModel):
    """训练控制响应"""

    success: bool
    action: str
    task_id: str
    experiment_id: str
    message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "action": "pause",
                "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "experiment_id": "exp_1",
                "message": "训练已暂停"
            }
        }


class TrainingSaveRequest(BaseModel):
    """保存到权重库请求"""

    weights_dir: str = Field(
        default="data/weights",
        description="权重库目录路径"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "weights_dir": "data/weights"
            }
        }


class TrainingSaveResponse(BaseModel):
    """保存到权重库响应"""

    success: bool
    message: str
    path: str

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "最佳模型已保存到权重库",
                "path": "data/weights/YOLO训练实验_best.pt"
            }
        }


class WeightLibraryListItem(BaseModel):
    """权重库列表项"""

    id: int
    name: str
    display_name: str
    description: Optional[str] = None
    task_type: str
    version: str
    file_name: str
    file_size_mb: Optional[float] = None
    framework: str
    is_auto_detected: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingSaveToWeightsRequest(BaseModel):
    """保存训练到权重库请求"""

    name: str = Field(..., description="权重名称", min_length=1, max_length=100)
    description: Optional[str] = Field(None, description="权重描述")
    include_last: bool = Field(True, description="是否保存最后一个epoch模型")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "ResNet50最佳模型",
                "description": "在CIFAR-10上训练的最佳模型",
                "include_last": True
            }
        }


class TrainingSaveToWeightsResponse(BaseModel):
    """保存训练到权重库响应"""

    success: bool
    message: str
    best_weight: Optional[WeightLibraryListItem] = None
    last_weight: Optional[WeightLibraryListItem] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "权重已保存到权重库",
                "best_weight": {
                    "id": 1,
                    "name": "ResNet50最佳模型",
                    "display_name": "ResNet50最佳模型 v1.0",
                    "description": "在CIFAR-10上训练的最佳模型",
                    "task_type": "classification",
                    "version": "1.0",
                    "file_name": "resnet50_best_20250120.pt",
                    "file_size_mb": 92.5,
                    "framework": "pytorch",
                    "is_auto_detected": False,
                    "created_at": "2025-01-20T12:00:00Z"
                },
                "last_weight": {
                    "id": 2,
                    "name": "ResNet50最佳模型",
                    "display_name": "ResNet50最佳模型 v1.1",
                    "description": "在CIFAR-10上训练的最后epoch模型",
                    "task_type": "classification",
                    "version": "1.1",
                    "file_name": "resnet50_last_20250120.pt",
                    "file_size_mb": 92.5,
                    "framework": "pytorch",
                    "is_auto_detected": False,
                    "created_at": "2025-01-20T12:00:00Z"
                }
            }
        }


class CheckpointInfo(BaseModel):
    """Checkpoint信息"""

    id: int
    epoch: int
    metric_value: Optional[float]
    metrics: Dict[str, Any]
    path: str
    is_best: bool
    file_size: Optional[int]
    created_at: Optional[datetime]

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "epoch": 50,
                "metric_value": 0.8234,
                "metrics": {
                    "train_loss": 0.0543,
                    "train_acc": 0.8234,
                    "val_loss": 0.0621,
                    "val_acc": 0.8123
                },
                "path": "data/checkpoints/exp_1/checkpoint_epoch_50.pt",
                "is_best": True,
                "file_size": 1048576,
                "created_at": "2025-01-04T12:45:00Z"
            }
        }


class MetricsEntry(BaseModel):
    """训练指标条目"""

    epoch: int
    timestamp: str
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    # 可扩展其他指标
    extra_metrics: Dict[str, float] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "epoch": 10,
                "timestamp": "2025-01-04T11:30:00Z",
                "train_loss": 0.0543,
                "train_acc": 0.8234,
                "val_loss": 0.0621,
                "val_acc": 0.8123,
                "extra_metrics": {}
            }
        }


class LogEntry(BaseModel):
    """日志条目"""

    level: str
    message: str
    source: str
    timestamp: str

    class Config:
        json_schema_extra = {
            "example": {
                "level": "INFO",
                "message": "Epoch 10/100: Loss: 0.0543, Acc: 0.8234",
                "source": "trainer",
                "timestamp": "2025-01-04T11:30:00Z"
            }
        }


class ExperimentListItem(BaseModel):
    """实验列表项（简化版）"""

    id: int
    name: str
    status: str
    task_type: Optional[str] = None
    progress: float
    current_epoch: int
    total_epochs: int
    best_metric: Optional[float]
    device: str
    start_time: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingStatusResponse(BaseModel):
    """训练状态响应"""

    status: str
    current_epoch: int
    total_epochs: int
    progress: float
    best_metric: Optional[float]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "status": "running",
                "current_epoch": 45,
                "total_epochs": 100,
                "progress": 45.0,
                "best_metric": 0.8234,
                "start_time": "2025-01-04T10:30:00Z",
                "end_time": None,
                "duration": "2:15:30"
            }
        }


# ==================== 训练配置详情相关模式 ====================

class TrainingConfigDetailResponse(BaseModel):
    """训练任务完整配置详情"""
    # 训练任务基本信息
    id: int
    name: str
    description: Optional[str] = None
    status: str
    created_at: datetime

    # 超参数配置
    hyperparams: Dict[str, Any]

    # 关联资源信息（使用Dict避免循环导入）
    dataset: Optional[Dict[str, Any]] = None
    model_architecture: Optional[Dict[str, Any]] = None
    pretrained_weight: Optional[Dict[str, Any]] = None
    augmentation: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
