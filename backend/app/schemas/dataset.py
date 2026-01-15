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

    @validator('classes', pre=True)
    def normalize_classes(cls, v):
        """规范化classes字段，支持字典或列表格式"""
        if isinstance(v, dict):
            # 如果是字典，提取class_names列表
            return v.get('class_names', [])
        elif isinstance(v, list):
            return v
        return []

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


# 图像预览相关模式
class ImageInfo(BaseModel):
    """图像信息模式"""
    path: str
    filename: str
    width: int
    height: int
    channels: int
    format: str
    size_bytes: int
    annotations: List[Dict[str, Any]] = []


class ImageListResponse(BaseModel):
    """图像列表响应模式"""
    images: List[ImageInfo]
    total: int
    page: int
    page_size: int
    total_pages: int


class ImageDetail(BaseModel):
    """图像详情模式"""
    image: ImageInfo
    annotation_data: Dict[str, Any] = {}
    thumbnail_url: Optional[str] = None
    preview_url: Optional[str] = None


# 数据增强相关模式
class AugmentationConfig(BaseModel):
    """数据增强配置模式"""
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotation_angle: float = 0.0  # 旋转角度（度）
    brightness_factor: float = 1.0  # 亮度因子
    contrast_factor: float = 1.0  # 对比度因子
    saturation_factor: float = 1.0  # 饱和度因子
    crop_params: Optional[Dict[str, int]] = None  # 裁剪参数 {x, y, width, height}
    scale_factor: float = 1.0  # 缩放因子
    hue_shift: float = 0.0  # 色调偏移
    gaussian_blur: float = 0.0  # 高斯模糊
    noise_std: float = 0.0  # 噪声标准差


class AugmentedImage(BaseModel):
    """增强后的图像模式"""
    original_path: str
    augmented_data: str  # base64编码的图像数据
    augmentation_config: AugmentationConfig
    applied_operations: List[str] = []


class AugmentationPreview(BaseModel):
    """数据增强预览模式"""
    original_image: str  # base64编码
    augmented_images: List[AugmentedImage]
    augmentation_summary: Dict[str, Any]


# 数据统计分析相关模式
class DetailedDatasetStatistics(BaseModel):
    """详细数据集统计模式"""
    basic_stats: DatasetStatistics
    image_quality_analysis: Dict[str, Any]
    annotation_quality_analysis: Optional[Dict[str, Any]] = None
    class_balance_analysis: Dict[str, Any]
    size_distribution_analysis: Dict[str, Any]
    recommendations: List[str] = []


class ClassDistribution(BaseModel):
    """类别分布模式"""
    class_name: str
    count: int
    percentage: float
    average_confidence: float = 0.0
    sample_images: List[str] = []


class SizeDistribution(BaseModel):
    """尺寸分布模式"""
    size_range: str
    count: int
    percentage: float
    sample_resolutions: List[str] = []


# 分页浏览相关模式
class PaginationParams(BaseModel):
    """分页参数模式"""
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(20, ge=1, le=100, description="每页大小")
    sort_by: Optional[str] = Field("filename", description="排序字段")
    sort_order: str = Field("asc", pattern="^(asc|desc)$", description="排序顺序")


class FilterParams(BaseModel):
    """过滤参数模式"""
    format_filter: Optional[str] = None
    size_filter: Optional[str] = None
    class_filter: Optional[List[str]] = None
    annotation_filter: Optional[bool] = None


# ==================== 数据增强策略相关模式 ====================

class AugmentationParamDef(BaseModel):
    """增强算子参数定义"""
    name: str
    label_zh: str
    label_en: str
    type: str  # boolean, integer, float, range, select
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Dict[str, Any]]] = None
    description: str = ""


class AugmentationOperatorSchema(BaseModel):
    """增强算子模式"""
    id: str
    name_zh: str
    name_en: str
    category: str
    category_label_zh: str
    category_label_en: str
    description: str
    enabled: bool = True
    params: List[AugmentationParamDef] = []


class PipelineItem(BaseModel):
    """流水线项目"""
    instanceId: str
    operatorId: str
    enabled: bool = True
    params: Dict[str, Any] = {}


class AugmentationStrategyCreate(BaseModel):
    """创建增强策略模式"""
    name: str = Field(..., min_length=1, max_length=100, description="策略名称")
    description: Optional[str] = Field(None, max_length=1000, description="策略描述")
    pipeline: List[PipelineItem] = Field(default_factory=list, description="算子流水线")


class AugmentationStrategyUpdate(BaseModel):
    """更新增强策略模式"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="策略名称")
    description: Optional[str] = Field(None, max_length=1000, description="策略描述")
    pipeline: Optional[List[PipelineItem]] = Field(None, description="算子流水线")


class AugmentationStrategyResponse(BaseModel):
    """增强策略响应模式"""
    id: int
    user_id: int
    name: str
    description: Optional[str] = None
    pipeline: List[PipelineItem] = []
    is_default: bool = False
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AugmentationStrategyList(BaseModel):
    """增强策略列表响应模式"""
    strategies: List[AugmentationStrategyResponse]
    total: int