"""
代码生成相关的Pydantic模式

定义API请求和响应的数据模型。

作者: CV Studio 开发团队
日期: 2025-12-25
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class CodeGenerationRequest(BaseModel):
    """代码生成请求"""
    nodes: List[dict] = Field(..., description="节点列表")
    connections: List[dict] = Field(..., description="连接列表")
    model_name: str = Field("GeneratedModel", description="模型类名")
    template_tag: Optional[str] = Field(None, description="模板标签")


class CodeValidationRequest(BaseModel):
    """代码验证请求"""
    code: str = Field(..., description="要验证的代码")
    model_name: str = Field(..., description="模型类名")


class ValidationResult(BaseModel):
    """验证结果"""
    valid: bool
    syntax_valid: bool = False
    executable: bool = False
    parameters_valid: bool = False
    forward_pass_success: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CodeGenerationResponse(BaseModel):
    """代码生成响应"""
    code: str = Field(..., description="生成的完整代码")
    model_name: str = Field(..., description="模型类名")
    validation: ValidationResult = Field(..., description="验证结果")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")


class CodeValidationResponse(BaseModel):
    """代码验证响应"""
    validation: ValidationResult
    test_results: Dict[str, Any] = Field(default_factory=dict)
