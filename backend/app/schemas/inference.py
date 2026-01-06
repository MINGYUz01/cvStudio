"""
推理相关的Pydantic Schema定义
用于API请求和响应的数据验证
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime


class InferenceJobCreate(BaseModel):
    """创建推理任务请求"""

    name: str = Field(..., description="推理任务名称", min_length=1, max_length=100)
    model_id: int = Field(..., description="模型ID")
    input_path: str = Field(..., description="输入文件/文件夹路径")
    output_path: Optional[str] = Field(None, description="输出路径（可选）")
    inference_type: str = Field(
        default="single",
        description="推理类型：single(单图) / batch(批量)",
        pattern="^(single|batch)$"
    )
    confidence_threshold: float = Field(default=0.5, description="置信度阈值", ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, description="IOU阈值（用于NMS）", ge=0.0, le=1.0)
    batch_size: int = Field(default=1, description="批量推理时的batch size", ge=1, le=64)
    device: Optional[str] = Field(None, description="推理设备：auto/cuda/cpu")
    created_by: int = Field(..., description="创建用户ID")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "YOLO推理任务",
                "model_id": 1,
                "input_path": "data/test_images",
                "output_path": "data/outputs/inference_1",
                "inference_type": "batch",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "batch_size": 8,
                "device": "auto",
                "created_by": 1
            }
        }


class InferenceJobUpdate(BaseModel):
    """更新推理任务请求"""

    name: Optional[str] = Field(None, description="新名称", min_length=1, max_length=100)
    status: Optional[str] = Field(
        None,
        description="任务状态",
        pattern="^(pending|running|completed|failed|cancelled)$"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "更新后的名称",
                "status": "pending"
            }
        }


class InferenceJobResponse(BaseModel):
    """推理任务响应"""

    id: int
    name: str
    model_id: int
    input_path: str
    output_path: Optional[str]
    status: str
    inference_type: str
    confidence_threshold: float
    iou_threshold: float
    batch_size: int
    device: str
    total_images: int
    processed_images: int
    fps: Optional[float]
    error_message: Optional[str]
    results: Optional[Dict[str, Any]]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": 1,
                "name": "YOLO推理任务",
                "model_id": 1,
                "input_path": "data/test_images",
                "output_path": "data/outputs/inference_1",
                "status": "completed",
                "inference_type": "batch",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "batch_size": 8,
                "device": "cuda:0",
                "total_images": 100,
                "processed_images": 100,
                "fps": 65.2,
                "error_message": None,
                "results": {
                    "total_detections": 1250,
                    "avg_confidence": 0.78
                },
                "start_time": "2025-01-04T10:30:00Z",
                "end_time": "2025-01-04T10:31:32Z",
                "created_at": "2025-01-04T10:30:00Z",
                "updated_at": "2025-01-04T10:31:32Z"
            }
        }


class InferencePredictRequest(BaseModel):
    """单图推理请求"""

    model_id: int = Field(..., description="模型ID")
    image_path: str = Field(..., description="图像路径")
    confidence_threshold: float = Field(default=0.5, description="置信度阈值", ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, description="IOU阈值", ge=0.0, le=1.0)
    device: Optional[str] = Field(None, description="推理设备：auto/cuda/cpu")

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": 1,
                "image_path": "data/test_images/test_001.jpg",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "device": "auto"
            }
        }


class DetectionResult(BaseModel):
    """检测结果"""

    bbox: List[float] = Field(..., description="边界框 [x1, y1, x2, y2]")
    label: str = Field(..., description="类别标签")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "bbox": [100.5, 150.2, 300.8, 400.3],
                "label": "person",
                "confidence": 0.876
            }
        }


class ClassificationResult(BaseModel):
    """分类结果"""

    label: str = Field(..., description="类别标签")
    confidence: float = Field(..., description="置信度", ge=0.0, le=1.0)
    probabilities: Optional[Dict[str, float]] = Field(None, description="各类别概率")

    class Config:
        json_schema_extra = {
            "example": {
                "label": "cat",
                "confidence": 0.923,
                "probabilities": {
                    "cat": 0.923,
                    "dog": 0.055,
                    "bird": 0.022
                }
            }
        }


class InferenceMetrics(BaseModel):
    """推理性能指标"""

    inference_time: float = Field(..., description="推理耗时（毫秒）")
    fps: Optional[float] = Field(None, description="FPS（批量推理时）")
    preprocessing_time: Optional[float] = Field(None, description="预处理耗时（毫秒）")
    postprocessing_time: Optional[float] = Field(None, description="后处理耗时（毫秒）")
    device: str = Field(..., description="使用的设备")
    image_size: Optional[List[int]] = Field(None, description="图像尺寸 [width, height]")

    class Config:
        json_schema_extra = {
            "example": {
                "inference_time": 14.2,
                "fps": 70.4,
                "preprocessing_time": 2.1,
                "postprocessing_time": 1.8,
                "device": "cuda:0",
                "image_size": [640, 480]
            }
        }


class InferencePredictResponse(BaseModel):
    """推理结果响应"""

    results: List[Dict[str, Any]] = Field(..., description="推理结果列表")
    metrics: InferenceMetrics = Field(..., description="性能指标")
    image_path: str = Field(..., description="输入图像路径")
    model_id: int = Field(..., description="使用的模型ID")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "bbox": [100.5, 150.2, 300.8, 400.3],
                        "label": "person",
                        "confidence": 0.876
                    },
                    {
                        "bbox": [400.2, 200.1, 550.6, 450.9],
                        "label": "car",
                        "confidence": 0.654
                    }
                ],
                "metrics": {
                    "inference_time": 14.2,
                    "fps": 70.4,
                    "preprocessing_time": 2.1,
                    "postprocessing_time": 1.8,
                    "device": "cuda:0",
                    "image_size": [640, 480]
                },
                "image_path": "data/test_images/test_001.jpg",
                "model_id": 1
            }
        }


class InferenceBatchRequest(BaseModel):
    """批量推理请求"""

    model_id: int = Field(..., description="模型ID")
    image_paths: List[str] = Field(..., description="图像路径列表")
    output_dir: str = Field(..., description="输出目录")
    batch_size: int = Field(default=8, description="batch size", ge=1, le=64)
    confidence_threshold: float = Field(default=0.5, description="置信度阈值", ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, description="IOU阈值", ge=0.0, le=1.0)
    device: Optional[str] = Field(None, description="推理设备")
    save_annotated: bool = Field(default=True, description="是否保存标注图像")
    export_format: str = Field(
        default="json",
        description="结果导出格式：json/yolo/coco",
        pattern="^(json|yolo|coco)$"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_id": 1,
                "image_paths": [
                    "data/test_images/img1.jpg",
                    "data/test_images/img2.jpg"
                ],
                "output_dir": "data/outputs/batch_1",
                "batch_size": 8,
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "device": "auto",
                "save_annotated": True,
                "export_format": "json"
            }
        }


class InferenceControlRequest(BaseModel):
    """推理控制请求"""

    action: str = Field(
        ...,
        description="控制动作",
        pattern="^(cancel|pause|resume)$"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {"action": "cancel"},
                {"action": "pause"},
                {"action": "resume"}
            ]
        }


class InferenceControlResponse(BaseModel):
    """推理控制响应"""

    success: bool
    action: str
    job_id: int
    message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "action": "cancel",
                "job_id": 1,
                "message": "推理任务已取消"
            }
        }


class InferenceProgressUpdate(BaseModel):
    """推理进度更新（WebSocket推送）"""

    job_id: int
    processed: int
    total: int
    progress: float
    fps: Optional[float] = None
    eta_seconds: Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": 1,
                "processed": 50,
                "total": 100,
                "progress": 50.0,
                "fps": 65.2,
                "eta_seconds": 0.77
            }
        }


class BatchInferenceResponse(BaseModel):
    """批量推理响应"""

    job_id: int
    status: str
    total_images: int
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": 1,
                "status": "processing",
                "total_images": 100,
                "message": "批量推理任务已创建，正在处理中"
            }
        }
