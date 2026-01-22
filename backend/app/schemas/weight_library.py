"""
权重库相关的Pydantic模式

包括权重文件上传、查询、更新、删除等操作的模式定义。
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


# ==================== 权重库基础模式 ====================

class WeightLibraryBase(BaseModel):
    """权重库基础模式"""
    name: str = Field(..., min_length=1, max_length=100, description="权重名称")
    description: Optional[str] = Field(None, max_length=1000, description="权重描述")
    task_type: str = Field(..., description="任务类型：classification/detection")
    version: str = Field("1.0", max_length=20, description="版本号")


class WeightLibraryCreate(WeightLibraryBase):
    """创建权重库模式"""
    input_size: Optional[List[int]] = Field(None, description="输入尺寸 [height, width]")
    class_names: Optional[List[str]] = Field(None, description="类别名称列表")
    normalize_params: Optional[Dict[str, List[float]]] = Field(None, description="归一化参数 {'mean': [...], 'std': [...]}")


class WeightLibraryUpdate(BaseModel):
    """更新权重库模式"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="权重名称")
    description: Optional[str] = Field(None, max_length=1000, description="权重描述")
    version: Optional[str] = Field(None, max_length=20, description="版本号")
    input_size: Optional[List[int]] = Field(None, description="输入尺寸")
    class_names: Optional[List[str]] = Field(None, description="类别名称列表")
    normalize_params: Optional[Dict[str, List[float]]] = Field(None, description="归一化参数")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="其他元数据")


class WeightLibraryResponse(WeightLibraryBase):
    """权重库响应模式"""
    id: int
    file_path: str
    file_name: str
    file_size: Optional[int] = None
    file_size_mb: Optional[float] = None
    framework: str
    input_size: Optional[List[int]] = None
    class_names: Optional[List[str]] = None
    normalize_params: Optional[Dict[str, List[float]]] = None
    extra_metadata: Optional[Dict[str, Any]] = None
    is_auto_detected: bool = False
    is_active: str
    parent_version_id: Optional[int] = None
    source_type: str = "uploaded"
    source_training_id: Optional[int] = None
    is_root: bool = True
    architecture_id: Optional[int] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class WeightLibraryListItem(BaseModel):
    """权重库列表项模式（简化版）"""
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
    is_root: bool = True
    source_type: str = "uploaded"
    architecture_id: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class WeightLibraryList(BaseModel):
    """权重库列表响应模式"""
    weights: List[WeightLibraryListItem]
    total: int


# ==================== 权重上传相关模式 ====================

class WeightUploadRequest(BaseModel):
    """权重上传请求模式"""
    name: str = Field(..., min_length=1, max_length=100, description="权重名称")
    task_type: str = Field(..., description="任务类型：classification/detection/auto")
    description: Optional[str] = Field(None, max_length=1000, description="权重描述")
    input_size: Optional[List[int]] = Field(None, description="输入尺寸 [height, width]")
    class_names: Optional[List[str]] = Field(None, description="类别名称列表")
    normalize_params: Optional[Dict[str, List[float]]] = Field(None, description="归一化参数")


class WeightUploadResponse(BaseModel):
    """权重上传响应模式"""
    success: bool
    message: str
    weight_id: Optional[int] = None
    weight: Optional[WeightLibraryResponse] = None


# ==================== 权重检测相关模式 ====================

class TaskTypeDetectionRequest(BaseModel):
    """任务类型检测请求模式"""
    file_path: str = Field(..., description="权重文件路径")


class TaskTypeDetectionResponse(BaseModel):
    """任务类型检测响应模式"""
    task_type: str
    confidence: float = 0.0
    is_auto_detected: bool


# ==================== 权重版本管理相关模式 ====================

class WeightVersionCreate(BaseModel):
    """创建权重新版本模式"""
    parent_weight_id: int = Field(..., description="父版本ID")
    description: Optional[str] = Field(None, description="版本描述")


class WeightVersionHistory(BaseModel):
    """权重版本历史模式"""
    versions: List[WeightLibraryListItem]
    total: int


# ==================== 权重元数据相关模式 ====================

class WeightMetadata(BaseModel):
    """权重元数据模式"""
    total_params: Optional[int] = None
    trainable_params: Optional[int] = None
    size_mb: Optional[float] = None
    input_size: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    input_name: Optional[str] = None
    output_names: Optional[List[str]] = None


# ==================== 权重树形结构相关模式 ====================

class WeightTreeItem(BaseModel):
    """权重树节点模式"""
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
    source_type: str = "uploaded"
    source_training_id: Optional[int] = None
    is_root: bool = True
    architecture_id: Optional[int] = None
    parent_version_id: Optional[int] = None
    created_at: datetime
    children: List['WeightTreeItem'] = []

    class Config:
        from_attributes = True


class WeightTreeResponse(BaseModel):
    """权重树响应模式"""
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
    is_root: bool
    source_type: str
    source_training_id: Optional[int] = None
    architecture_id: Optional[int] = None
    parent_version_id: Optional[int] = None
    created_at: datetime
    children: List['WeightTreeItem'] = []

    class Config:
        from_attributes = True


# 更新forward引用
WeightTreeResponse.model_rebuild()
WeightTreeItem.model_rebuild()


class WeightRootList(BaseModel):
    """根节点权重列表响应模式"""
    weights: List[WeightLibraryListItem]
    total: int


# ==================== 训练配置详情相关模式 ====================

class DatasetInfo(BaseModel):
    """数据集信息"""
    id: int
    name: str
    format: str
    num_images: Optional[int] = None
    num_classes: Optional[int] = None
    classes: Optional[List[str]] = None
    path: Optional[str] = None


class ModelArchitectureInfo(BaseModel):
    """模型架构信息"""
    id: int
    name: str
    description: Optional[str] = None
    file_path: Optional[str] = None
    input_size: Optional[List[int]] = None
    task_type: Optional[str] = None


class PretrainedWeightInfo(BaseModel):
    """预训练权重信息"""
    id: int
    name: str
    display_name: str
    task_type: str
    version: str
    source_type: str  # uploaded/trained


class AugmentationConfigInfo(BaseModel):
    """数据增强配置信息"""
    enabled: bool = False
    strategy: Optional[str] = None
    strategy_id: Optional[int] = None
    description: Optional[str] = None
    config: Optional[Union[Dict[str, Any], List[Any]]] = None  # 支持字典和列表（pipeline）


class WeightTrainingConfigResponse(BaseModel):
    """权重训练配置响应模式（扩展版）"""
    weight_id: int
    weight_name: str
    training_config: Optional[Dict[str, Any]] = None
    source_training: Optional[Dict[str, Any]] = None

    # 完整的关联信息
    dataset: Optional[DatasetInfo] = None
    model_architecture: Optional[ModelArchitectureInfo] = None
    pretrained_weight: Optional[PretrainedWeightInfo] = None
    augmentation: Optional[AugmentationConfigInfo] = None


class WeightForTraining(BaseModel):
    """可用于训练的权重模式"""
    id: int
    name: str
    display_name: str
    description: Optional[str] = None
    task_type: str
    version: str
    file_path: str
    architecture_id: Optional[int] = None
    architecture_name: Optional[str] = None
    created_at: datetime
