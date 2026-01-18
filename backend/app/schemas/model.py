"""
模型相关的Pydantic模式

包括模型架构(ModelArchitecture)和生成代码(GeneratedCode)的模式定义。
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


# ==================== 模型架构相关模式 ====================

class ModelArchitectureBase(BaseModel):
    """模型架构基础模式"""
    name: str = Field(..., min_length=1, max_length=100, description="架构名称")
    description: Optional[str] = Field(None, max_length=1000, description="架构描述")
    version: str = Field("v1.0", max_length=20, description="版本号")
    type: str = Field("Custom", max_length=50, description="模型类型")


class ModelArchitectureCreate(ModelArchitectureBase):
    """创建模型架构模式"""
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="节点列表")
    connections: List[Dict[str, Any]] = Field(default_factory=list, description="连接列表")


class ModelArchitectureUpdate(BaseModel):
    """更新模型架构模式"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="架构名称")
    description: Optional[str] = Field(None, max_length=1000, description="架构描述")
    version: Optional[str] = Field(None, max_length=20, description="版本号")
    type: Optional[str] = Field(None, max_length=50, description="模型类型")
    nodes: Optional[List[Dict[str, Any]]] = Field(None, description="节点列表")
    connections: Optional[List[Dict[str, Any]]] = Field(None, description="连接列表")


class ModelArchitectureResponse(ModelArchitectureBase):
    """模型架构响应模式"""
    id: int
    file_path: str
    file_name: str
    node_count: int
    connection_count: int
    meta: Optional[Dict[str, Any]] = None
    is_active: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ModelArchitectureListItem(BaseModel):
    """模型架构列表项模式（简化版）"""
    id: int
    name: str
    description: str
    version: str
    type: str
    node_count: int
    connection_count: int
    file_name: str
    created: str
    updated: str

    class Config:
        from_attributes = True


class ModelArchitectureList(BaseModel):
    """模型架构列表响应模式"""
    architectures: List[ModelArchitectureListItem]
    total: int


# ==================== 生成代码相关模式 ====================

class GeneratedCodeBase(BaseModel):
    """生成代码基础模式"""
    name: str = Field(..., min_length=1, max_length=100, description="代码名称")


class GeneratedCodeCreate(GeneratedCodeBase):
    """创建生成代码模式"""
    code: str = Field(..., description="生成的Python代码")
    template_tag: Optional[str] = Field(None, max_length=50, description="使用的模板标签")
    meta: Optional[Dict[str, Any]] = Field(None, description="元数据")


class GeneratedCodeUpdate(BaseModel):
    """更新生成代码模式"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="代码名称")
    code: Optional[str] = Field(None, description="Python代码")
    template_tag: Optional[str] = Field(None, max_length=50, description="模板标签")
    meta: Optional[Dict[str, Any]] = Field(None, description="元数据")


class GeneratedCodeResponse(GeneratedCodeBase):
    """生成代码响应模式"""
    id: int
    file_path: str
    file_name: str
    code_size: int
    template_tag: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    is_active: str
    created_at: datetime

    class Config:
        from_attributes = True


class GeneratedCodeListItem(BaseModel):
    """生成代码列表项模式（简化版）"""
    id: int
    name: str
    file_name: str
    code_size: int
    template_tag: Optional[str] = None
    created: str

    class Config:
        from_attributes = True


class GeneratedCodeList(BaseModel):
    """生成代码列表响应模式"""
    codes: List[GeneratedCodeListItem]
    total: int


# ==================== 代码生成请求/响应模式 ====================

class CodeGenerationRequest(BaseModel):
    """代码生成请求模式（兼容现有API）"""
    name: str = Field(..., description="模型名称")
    nodes: List[Dict[str, Any]] = Field(..., description="节点列表")
    connections: List[Dict[str, Any]] = Field(..., description="连接列表")
    template_tag: Optional[str] = Field(None, description="模板标签")
    input_shape: Optional[List[int]] = Field(None, description="输入形状")
    num_classes: Optional[int] = Field(None, description="输出类别数")


class CodeGenerationResponse(BaseModel):
    """代码生成响应模式"""
    success: bool
    code: Optional[str] = None
    code_id: Optional[int] = None
    message: str
    warnings: List[str] = []


# ==================== 预设模型相关模式 ====================

class PresetModelBase(BaseModel):
    """预设模型基础模式"""
    name: str = Field(..., min_length=1, max_length=100, description="预设模型名称")
    description: Optional[str] = Field(None, max_length=1000, description="预设模型描述")
    category: str = Field(..., max_length=50, description="分类：cnn, rnn, transformer, classification, detection")
    difficulty: str = Field(..., max_length=20, description="难度：beginner, intermediate, advanced")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    architecture_data: Dict[str, Any] = Field(..., description="架构数据，包含nodes和connections")


class PresetModelCreate(PresetModelBase):
    """创建预设模型模式（管理员使用）"""
    thumbnail: Optional[str] = Field(None, description="缩略图base64")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class PresetModelUpdate(BaseModel):
    """更新预设模型模式"""
    name: Optional[str] = Field(None, min_length=1, max_length=100, description="预设模型名称")
    description: Optional[str] = Field(None, max_length=1000, description="预设模型描述")
    category: Optional[str] = Field(None, max_length=50, description="分类")
    difficulty: Optional[str] = Field(None, max_length=20, description="难度")
    tags: Optional[List[str]] = Field(None, description="标签列表")
    thumbnail: Optional[str] = Field(None, description="缩略图base64")
    architecture_data: Optional[Dict[str, Any]] = Field(None, description="架构数据")
    extra_metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")
    is_active: Optional[bool] = Field(None, description="是否激活")


class PresetModelResponse(PresetModelBase):
    """预设模型响应模式"""
    id: int
    thumbnail: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None
    is_active: bool
    node_count: int
    connection_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PresetModelListItem(BaseModel):
    """预设模型列表项模式（简化版）"""
    id: int
    name: str
    description: str
    category: str
    difficulty: str
    tags: List[str]
    node_count: int
    connection_count: int

    class Config:
        from_attributes = True


class PresetModelList(BaseModel):
    """预设模型列表响应模式"""
    presets: List[PresetModelListItem]
    total: int


class CreateFromPresetRequest(BaseModel):
    """从预设模型创建架构请求模式"""
    name: str = Field(..., min_length=1, max_length=100, description="新架构名称")
    description: Optional[str] = Field(None, max_length=1000, description="新架构描述")
