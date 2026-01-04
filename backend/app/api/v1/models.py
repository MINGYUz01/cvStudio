"""
模型相关API路由

提供模型图验证、分析、形状推断等功能。
支持将前端 ModelBuilder 构建的模型图转换为可执行的 PyTorch 代码。

作者: CV Studio 开发团队
日期: 2025-12-25
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

from app.utils.graph_traversal import analyze_graph_structure, Graph
from app.utils.shape_inference import infer_shapes_from_graph
from app.services.code_generator_service import CodeGeneratorService
from app.schemas.code_generation import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    CodeValidationRequest,
    CodeValidationResponse
)


router = APIRouter()


# ==============================
# 数据模型
# ==============================

class NodeModel(BaseModel):
    """节点模型"""
    id: str
    type: str
    label: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None
    inputs: Optional[List[str]] = []
    outputs: Optional[List[str]] = []
    data: Dict[str, Any] = {}


class ConnectionModel(BaseModel):
    """连接模型"""
    id: Optional[str] = None
    source: str
    target: str


class GraphModel(BaseModel):
    """图模型"""
    nodes: List[NodeModel]
    connections: List[ConnectionModel]


class ValidationResult(BaseModel):
    """验证结果"""
    valid: bool
    errors: List[str]
    warnings: List[str]


class AnalysisResult(BaseModel):
    """分析结果"""
    execution_order: Dict[str, List[str]]
    num_parameters: int
    depth: int


class ShapeInfo(BaseModel):
    """形状信息"""
    input: Optional[List[str]] = None
    output: List[str]
    input_str: Optional[str] = None
    output_str: str


class ShapeInferenceResult(BaseModel):
    """形状推断结果"""
    valid: bool
    shapes: Dict[str, ShapeInfo]
    errors: List[str]
    warnings: List[str]


class AnalyzeAndInferResult(BaseModel):
    """组合分析结果"""
    validation: ValidationResult
    analysis: AnalysisResult
    shapes: Dict[str, ShapeInfo]


# ==============================
# API 端点
# ==============================

@router.get("/")
async def models_root():
    """模型模块根路径"""
    return {
        "message": "模型模块",
        "status": "运行中",
        "endpoints": {
            "validate": "POST /api/v1/models/validate - 验证模型图",
            "analyze": "POST /api/v1/models/analyze - 分析模型结构",
            "infer-shapes": "POST /api/v1/models/infer-shapes - 推断张量形状",
            "analyze-and-infer": "POST /api/v1/models/analyze-and-infer - 组合分析",
            "generate": "POST /api/v1/models/generate - 生成PyTorch代码",
            "validate-code": "POST /api/v1/models/validate-code - 验证生成的代码",
            "templates": "GET /api/v1/models/templates - 获取可用模板列表"
        }
    }


@router.post("/validate", response_model=ValidationResult)
async def validate_model_graph(graph: GraphModel):
    """
    验证模型图的合法性

    检查项：
    - 循环依赖检测
    - 孤立节点检测
    - 连接完整性验证
    - 节点类型验证
    - 参数完整性检查
    """
    try:
        # 转换为字典格式
        graph_dict = graph.model_dump()

        # 分析图结构
        result = analyze_graph_structure(graph_dict)

        # 返回验证结果
        return ValidationResult(
            valid=result["validation"]["valid"],
            errors=result["validation"]["errors"],
            warnings=result["validation"]["warnings"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"验证失败: {str(e)}"
        )


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_model_graph(graph: GraphModel):
    """
    分析模型结构

    返回：
    - 节点执行顺序（前向/反向）
    - 层节点列表
    - 输入/输出节点
    - 网络深度
    """
    try:
        # 转换为字典格式
        graph_dict = graph.model_dump()

        # 分析图结构
        result = analyze_graph_structure(graph_dict)

        # 检查验证结果
        if not result["validation"]["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "图验证失败",
                    "errors": result["validation"]["errors"],
                    "warnings": result["validation"]["warnings"]
                }
            )

        # 返回分析结果
        execution_order = result["execution_order"]
        return AnalysisResult(
            execution_order={
                "forward": execution_order["forward_order"],
                "backward": execution_order["backward_order"],
                "layers": execution_order["layers"],
                "inputs": execution_order["input_nodes"],
                "outputs": execution_order["output_nodes"]
            },
            num_parameters=0,  # 待代码生成后计算
            depth=execution_order["depth"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"分析失败: {str(e)}"
        )


@router.post("/infer-shapes", response_model=ShapeInferenceResult)
async def infer_tensor_shapes(graph: GraphModel):
    """
    推断张量形状

    基于模型图结构和执行顺序，推断每个节点的输入输出张量形状。
    支持所有 PyTorch 原生算子的形状计算。
    """
    try:
        # 转换为字典格式
        graph_dict = graph.model_dump()

        # 分析图结构
        structure_result = analyze_graph_structure(graph_dict)

        # 检查验证结果
        if not structure_result["validation"]["valid"]:
            return ShapeInferenceResult(
                valid=False,
                shapes={},
                errors=structure_result["validation"]["errors"],
                warnings=structure_result["validation"]["warnings"]
            )

        # 推断形状
        shape_result = infer_shapes_from_graph(
            structure_result["graph"],
            structure_result["execution_order"]
        )

        # 转换为API响应格式
        shapes_dict = {}
        for node_id, shape_info in shape_result["frontend_shapes"].items():
            shapes_dict[node_id] = ShapeInfo(**shape_info)

        return ShapeInferenceResult(
            valid=shape_result["validation"]["valid"],
            shapes=shapes_dict,
            errors=shape_result["validation"]["errors"],
            warnings=shape_result["validation"]["warnings"]
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"形状推断失败: {str(e)}"
        )


@router.post("/analyze-and-infer", response_model=AnalyzeAndInferResult)
async def analyze_and_infer(graph: GraphModel):
    """
    组合接口：验证、分析和形状推断

    一次性完成图验证、结构分析和形状推断，减少前端请求次数。
    推荐使用此接口以获得最佳性能。
    """
    try:
        # 转换为字典格式
        graph_dict = graph.model_dump()

        # 分析图结构
        structure_result = analyze_graph_structure(graph_dict)

        # 准备验证结果
        validation_result = ValidationResult(
            valid=structure_result["validation"]["valid"],
            errors=structure_result["validation"]["errors"],
            warnings=structure_result["validation"]["warnings"]
        )

        # 如果验证失败，提前返回
        if not structure_result["validation"]["valid"]:
            # 创建空的分析和形状结果
            return AnalyzeAndInferResult(
                validation=validation_result,
                analysis=AnalysisResult(
                    execution_order={},
                    num_parameters=0,
                    depth=0
                ),
                shapes={}
            )

        # 推断形状
        shape_result = infer_shapes_from_graph(
            structure_result["graph"],
            structure_result["execution_order"]
        )

        # 准备分析结果
        execution_order = structure_result["execution_order"]
        analysis_result = AnalysisResult(
            execution_order={
                "forward": execution_order["forward_order"],
                "backward": execution_order["backward_order"],
                "layers": execution_order["layers"],
                "inputs": execution_order["input_nodes"],
                "outputs": execution_order["output_nodes"]
            },
            num_parameters=0,  # 待代码生成后计算
            depth=execution_order["depth"]
        )

        # 准备形状结果
        shapes_dict = {}
        for node_id, shape_info in shape_result["frontend_shapes"].items():
            shapes_dict[node_id] = ShapeInfo(**shape_info)

        return AnalyzeAndInferResult(
            validation=validation_result,
            analysis=analysis_result,
            shapes=shapes_dict
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"分析失败: {str(e)}"
        )

# ==============================
# 代码生成相关端点
# ==============================

# 初始化代码生成服务
code_generator_service = CodeGeneratorService()


@router.post("/generate", response_model=CodeGenerationResponse)
async def generate_pytorch_code(
    graph: GraphModel,
    model_name: str = Query("GeneratedModel", description="模型类名"),
    template_tag: Optional[str] = Query(None, description="模板标签（预留扩展）")
):
    """
    生成PyTorch模型代码

    将前端ModelBuilder构建的模型图转换为可执行的PyTorch代码。

    功能：
    - 图结构验证
    - 张量形状推断
    - 自动生成__init__和forward方法
    - 代码语法检查和验证
    - 前向传播测试

    支持的层类型：
    - Conv2d, Linear, BatchNorm2d, LayerNorm
    - ReLU, LeakyReLU, SiLU, Sigmoid, Softmax
    - MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
    - Flatten, Dropout, Upsample
    - Concat, Add, Identity
    """
    try:
        # 转换为字典格式
        graph_dict = graph.model_dump()

        # 调用服务层生成代码
        result = await code_generator_service.generate_code(
            graph_json=graph_dict,
            model_name=model_name,
            template_tag=template_tag
        )

        return CodeGenerationResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"代码生成失败: {str(e)}"
        )


@router.post("/validate-code", response_model=CodeValidationResponse)
async def validate_generated_code(request: CodeValidationRequest):
    """
    验证生成的PyTorch代码

    对已生成的代码进行全面的验证检查。

    验证项目：
    1. AST语法检查 - 检查代码语法是否正确
    2. 参数完整性验证 - 检查层参数是否完整
    3. 可执行性验证 - 测试代码能否成功导入
    4. 前向传播测试 - 使用随机输入测试模型
    """
    try:
        # 调用服务层验证代码
        result = await code_generator_service.validate_code(
            code=request.code,
            model_name=request.model_name
        )

        return CodeValidationResponse(
            validation=result,
            test_results=result.get("test_results", {})
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"代码验证失败: {str(e)}"
        )


@router.get("/templates")
async def list_templates():
    """
    获取可用的代码模板列表

    返回系统中所有可用的代码生成模板及其详细信息。
    """
    return {
        "templates": [
            {
                "tag": "base",
                "name": "基础模型模板",
                "description": "标准的PyTorch模型类模板，包含完整的文档字符串和元数据",
                "supported_layers": [
                    "Conv2d", "Linear", "BatchNorm2d", "LayerNorm",
                    "ReLU", "LeakyReLU", "SiLU", "Sigmoid", "Softmax",
                    "MaxPool2d", "AvgPool2d", "AdaptiveAvgPool2d",
                    "Flatten", "Dropout", "Upsample",
                    "Concat", "Add", "Identity"
                ],
                "features": [
                    "自动生成语义化层名",
                    "包含完整文档字符串",
                    "支持模型元数据导出",
                    "PEP8代码格式",
                    "支持14种PyTorch原生算子"
                ]
            }
        ],
        "default_template": "base",
        "total_templates": 1
    }
