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
from pathlib import Path
from datetime import datetime

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

# 模型文件存储目录
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


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

        # 调用服务层生成代码（仅预览，不自动保存）
        result = await code_generator_service.generate_code(
            graph_json=graph_dict,
            model_name=model_name,
            template_tag=template_tag
        )

        # 注意：生成的代码不会自动保存到服务器
        # 用户需要点击"保存到库"按钮才会通过 /save-code 端点保存

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


# ==============================
# 生成的模型文件管理端点
# ==============================

@router.get("/generated-files")
async def list_generated_files():
    """
    获取已生成的模型文件列表

    返回服务器上保存的所有生成的模型文件信息。
    """
    try:
        files = []
        if MODEL_DIR.exists():
            for filepath in MODEL_DIR.glob("*.py"):
                stat = filepath.stat()
                files.append({
                    "filename": filepath.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })

        return {
            "files": sorted(files, key=lambda x: x["created"], reverse=True),
            "total": len(files)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取文件列表失败: {str(e)}"
        )


@router.get("/generated-files/{filename}")
async def get_generated_file(filename: str):
    """
    获取生成的模型文件内容

    根据文件名返回模型文件的完整代码内容。
    """
    try:
        # 安全检查：防止路径遍历攻击
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(400, "无效的文件名")

        filepath = MODEL_DIR / filename

        if not filepath.exists():
            raise HTTPException(404, "文件不存在")

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "filename": filename,
            "content": content,
            "size": filepath.stat().st_size
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取文件失败: {str(e)}"
        )


@router.delete("/generated-files/{filename}")
async def delete_generated_file(filename: str):
    """
    删除生成的模型文件

    根据文件名删除服务器上的模型文件。
    """
    try:
        # 安全检查：防止路径遍历攻击
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(400, "无效的文件名")

        filepath = MODEL_DIR / filename

        if not filepath.exists():
            raise HTTPException(404, "文件不存在")

        filepath.unlink()

        return {"message": "文件已删除", "filename": filename}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除文件失败: {str(e)}"
        )


@router.post("/save-code")
async def save_code_to_library(request: dict):
    """
    直接保存代码到模型库

    允许用户将预览的代码直接保存到服务器，无需重新生成。
    """
    try:
        code = request.get("code")
        filename = request.get("filename")
        model_name = request.get("model_name", "Model")

        if not code:
            raise HTTPException(400, "代码内容不能为空")

        # 如果没有提供文件名，自动生成
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = model_name.replace(" ", "_").replace("-", "_")
            filename = f"{safe_name}_{timestamp}.py"

        # 安全检查文件名
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(400, "无效的文件名")

        filepath = MODEL_DIR / filename

        # 写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

        return {
            "message": "代码已保存",
            "filename": filename,
            "filepath": str(filepath)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"保存代码失败: {str(e)}"
        )


# ==============================
# 模型架构管理端点
# ==============================

# 模型架构存储目录
ARCHITECTURE_DIR = Path("data/architectures")
ARCHITECTURE_DIR.mkdir(parents=True, exist_ok=True)


class ArchitectureModel(BaseModel):
    """模型架构数据模型"""
    name: str = Field(..., description="架构名称")
    description: str = Field("", description="架构描述")
    version: str = Field("v1.0", description="版本号")
    type: str = Field("Custom", description="架构类型")
    nodes: List[Dict[str, Any]] = Field(default_factory=list, description="节点列表")
    connections: List[Dict[str, str]] = Field(default_factory=list, description="连接列表")
    thumbnail: Optional[str] = Field(None, description="缩略图（base64）")


@router.post("/architectures")
async def save_architecture(
    architecture: ArchitectureModel,
    overwrite: bool = Query(False, description="是否覆盖同名文件"),
    target_filename: Optional[str] = Query(None, description="指定保存的目标文件名（用于更新原文件，不管名称是否改变）")
):
    """
    保存模型架构到服务器

    将模型架构数据保存为 JSON 文件到 data/architectures/ 目录。
    - 如果指定了 target_filename，则保存到指定文件（更新原文件）
    - 如果 overwrite=true，则覆盖同名文件；否则自动添加时间戳。
    """
    try:
        import json

        # 确定目标文件名
        if target_filename:
            # 使用指定的文件名（用于更新原文件）
            filename = target_filename
            filepath = ARCHITECTURE_DIR / filename
        else:
            # 根据架构名称生成文件名
            safe_name = architecture.name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            filename = f"{safe_name}.json"
            filepath = ARCHITECTURE_DIR / filename

            # 如果文件已存在且不覆盖，添加时间戳
            if filepath.exists() and not overwrite:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{safe_name}_{timestamp}.json"
                filepath = ARCHITECTURE_DIR / filename

        # 检查文件是否已存在（用于判断是更新还是新建）
        if filepath.exists():
            # 覆盖模式：保留原有的创建时间
            with open(filepath, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                created = old_data.get("created", datetime.now().isoformat())

            data = {
                "name": architecture.name,
                "description": architecture.description,
                "version": architecture.version,
                "type": architecture.type,
                "created": created,
                "updated": datetime.now().isoformat(),
                "nodes": architecture.nodes,
                "connections": architecture.connections,
                "thumbnail": architecture.thumbnail
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            return {
                "message": "架构已更新",
                "filename": filename,
                "filepath": str(filepath),
                "updated": True
            }

        # 新建文件
        data = {
            "name": architecture.name,
            "description": architecture.description,
            "version": architecture.version,
            "type": architecture.type,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "nodes": architecture.nodes,
            "connections": architecture.connections,
            "thumbnail": architecture.thumbnail
        }

        # 写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return {
            "message": "架构已保存",
            "filename": filename,
            "filepath": str(filepath),
            "updated": False
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"保存架构失败: {str(e)}"
        )


@router.get("/architectures")
async def list_architectures():
    """
    获取已保存的模型架构列表

    返回服务器上保存的所有模型架构信息。
    """
    try:
        import json

        architectures = []
        if ARCHITECTURE_DIR.exists():
            for filepath in ARCHITECTURE_DIR.glob("*.json"):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    stat = filepath.stat()
                    architectures.append({
                        "filename": filepath.name,
                        "name": data.get("name", filepath.stem),
                        "description": data.get("description", ""),
                        "version": data.get("version", "v1.0"),
                        "type": data.get("type", "Custom"),
                        "node_count": len(data.get("nodes", [])),
                        "connection_count": len(data.get("connections", [])),
                        "created": data.get("created", datetime.fromtimestamp(stat.st_ctime).isoformat()),
                        "updated": data.get("updated", datetime.fromtimestamp(stat.st_mtime).isoformat()),
                        "size": stat.st_size
                    })
                except Exception as e:
                    # 跳过损坏的文件
                    continue

        return {
            "architectures": sorted(architectures, key=lambda x: x["updated"], reverse=True),
            "total": len(architectures)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取架构列表失败: {str(e)}"
        )


@router.get("/architectures/{filename}")
async def get_architecture(filename: str):
    """
    获取模型架构的详细内容

    根据文件名返回模型架构的完整数据。
    """
    try:
        import json

        # 安全检查：防止路径遍历攻击
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(400, "无效的文件名")

        filepath = ARCHITECTURE_DIR / filename

        if not filepath.exists():
            raise HTTPException(404, "架构文件不存在")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "filename": filename,
            **data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取架构失败: {str(e)}"
        )


@router.delete("/architectures/{filename}")
async def delete_architecture(filename: str):
    """
    删除模型架构

    根据文件名删除服务器上的模型架构文件。
    """
    try:
        # 安全检查：防止路径遍历攻击
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(400, "无效的文件名")

        filepath = ARCHITECTURE_DIR / filename

        if not filepath.exists():
            raise HTTPException(404, "架构文件不存在")

        filepath.unlink()

        return {"message": "架构已删除", "filename": filename}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除架构失败: {str(e)}"
        )
