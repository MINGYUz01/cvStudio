"""
数据集相关的Pydantic模式
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class DatasetBase(BaseModel):
    """数据集基础模式"""
    name: str = Field(..., min_length=1, max_length=100, description="数据集名称")
    description: Optional[str] = Field(None, max_length=1000, description="数据集描述")


class DatasetCreate(DatasetBase):
    """创建数据集模式"""
    pass


class DatasetUpload(DatasetCreate):
    """上传数据集模式"""
    pass


class DatasetRegister(BaseModel):
    """注册现有数据集模式"""
    name: str = Field(..., min_length=1, max_length=100, description="数据集名称")
    description: Optional[str] = Field(None, max_length=1000, description="数据集描述")
    dataset_path: str = Field(..., min_length=1, description="数据集路径")

    @validator('dataset_path')
    def validate_dataset_path(cls, v):
        """验证数据集路径"""
        from pathlib import Path
        if not Path(v).exists():
            raise ValueError('数据集路径不存在')
        return v


class DatasetUpdate(BaseModel):
    """更新数据集模式"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="数据集名称")
    description: Optional[str] = Field(None, max_length=1000, description="数据集描述")


class DatasetResponse(DatasetBase):
    """数据集响应模式"""
    id: int
    path: str
    format: str
    num_images: int
    num_classes: int
    classes: List[str] = []
    meta: Dict[str, Any] = {}
    is_active: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class DatasetList(BaseModel):
    """数据集列表响应模式"""
    datasets: List[DatasetResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DatasetPreview(BaseModel):
    """数据集预览模式"""
    dataset_id: int
    sample_images: List[Dict[str, Any]] = []
    format_details: Dict[str, Any] = {}
    statistics: Dict[str, Any] = {}


class DatasetStatistics(BaseModel):
    """数据集统计模式"""
    dataset_id: int
    num_images: int
    num_classes: int
    class_distribution: Dict[str, int] = {}
    image_size_distribution: Dict[str, Any] = {}
    format_details: Dict[str, Any] = {}
    quality_metrics: Dict[str, Any] = {}


class FormatRecognitionResult(BaseModel):
    """格式识别结果模式"""
    best_format: Dict[str, Any]
    all_results: Dict[str, Any]
    dataset_path: str


class ThumbnailInfo(BaseModel):
    """缩略图信息模式"""
    dataset_id: int
    image_path: str
    thumbnail_path: str
    size: tuple
    created_at: datetime


class DatasetValidation(BaseModel):
    """数据集验证模式"""
    is_valid: bool
    errors: List[str] = []
    warnings: List[str] = []
    suggestions: List[str] = []


# 数据集元信息相关模式
class ImageStats(BaseModel):
    """图像统计信息"""
    width_range: List[int] = []
    height_range: List[int] = []
    avg_width: float = 0
    avg_height: float = 0
    format_distribution: Dict[str, int] = {}
    total_analyzed: int = 0


class ClassStats(BaseModel):
    """类别统计信息"""
    class_name: str
    count: int
    percentage: float = 0
    sample_images: List[str] = []


class AnnotationStats(BaseModel):
    """标注统计信息"""
    total_annotations: int = 0
    avg_annotations_per_image: float = 0
    max_annotations_per_image: int = 0
    class_distribution: Dict[str, int] = {}
    bbox_size_distribution: Dict[str, Any] = {}


class QualityMetrics(BaseModel):
    """质量指标"""
    image_quality_score: float = 0
    annotation_quality_score: float = 0
    completeness_score: float = 0
    consistency_score: float = 0
    overall_score: float = 0


class DatasetMetadata(BaseModel):
    """数据集元信息"""
    format: str
    num_images: int
    num_classes: int
    classes: List[str] = []
    image_stats: ImageStats
    class_stats: List[ClassStats] = []
    annotation_stats: Optional[AnnotationStats] = None
    quality_metrics: QualityMetrics
    format_specific_info: Dict[str, Any] = {}