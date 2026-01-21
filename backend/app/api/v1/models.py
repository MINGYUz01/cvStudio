"""
模型相关API路由

提供模型图验证、分析、形状推断等功能。
支持将前端 ModelBuilder 构建的模型图转换为可执行的 PyTorch 代码。

作者: CV Studio 开发团队
日期: 2025-12-25
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session


def debug_log(msg: str, level: str = "INFO"):
    """调试日志输出"""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} | {level}     | {msg}")

from app.utils.graph_traversal import analyze_graph_structure, Graph
from app.utils.shape_inference import infer_shapes_from_graph
from app.services.code_generator_service import CodeGeneratorService
from app.services.model_service import (
    ModelArchitectureService,
    GeneratedCodeService
)
from app.schemas.code_generation import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    CodeValidationRequest,
    CodeValidationResponse
)
from app.schemas.model import (
    ModelArchitectureCreate,
    ModelArchitectureResponse,
    ModelArchitectureList,
    GeneratedCodeResponse,
    GeneratedCodeList
)
from app.dependencies import get_db


router = APIRouter()

# 初始化服务
code_generator_service = CodeGeneratorService()
architecture_service = ModelArchitectureService()
code_service = GeneratedCodeService()


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
# 生成的模型文件管理端点（数据库版本）
# ==============================

@router.get("/generated-files", response_model=GeneratedCodeList)
async def list_generated_files(
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(100, ge=1, le=100, description="返回数量"),
    db: Session = Depends(get_db)
):
    """
    获取已生成的模型文件列表

    从数据库返回所有保存的模型代码文件信息。
    """
    try:
        codes = code_service.list_codes(db, skip=skip, limit=limit)
        total = code_service.count_codes(db)

        items = []
        for code in codes:
            items.append({
                "id": code.id,
                "name": code.name,
                "file_name": code.file_name,
                "code_size": code.code_size,
                "template_tag": code.template_tag,
                "created": code.created_at.isoformat() if code.created_at else ""
            })

        return GeneratedCodeList(
            codes=items,
            total=total
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取文件列表失败: {str(e)}"
        )


@router.get("/generated-files/{code_id}")
async def get_generated_file(
    code_id: int,
    db: Session = Depends(get_db)
):
    """
    获取生成的模型文件内容

    根据ID返回模型文件的完整代码内容。
    """
    try:
        code = code_service.get_code(db, code_id)
        if not code:
            raise HTTPException(404, "文件不存在")

        content = code_service.load_code_file(code)

        return {
            "id": code.id,
            "filename": code.file_name,
            "content": content,
            "size": code.code_size
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取文件失败: {str(e)}"
        )


@router.delete("/generated-files/{code_id}")
async def delete_generated_file(
    code_id: int,
    db: Session = Depends(get_db)
):
    """
    删除生成的模型文件

    根据ID删除模型文件（同时删除数据库记录和物理文件）。
    """
    try:
        success = code_service.delete_code(db, code_id, physical=True)
        if not success:
            raise HTTPException(404, "文件不存在")

        return {"message": "文件已删除", "id": code_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除文件失败: {str(e)}"
        )


@router.post("/save-code")
async def save_code_to_library(
    request: dict,
    db: Session = Depends(get_db)
):
    """
    保存代码到模型库，并创建可训练的Model记录

    将代码保存到数据库和文件系统，同时创建Model表记录用于训练。
    """
    try:
        code = request.get("code")
        model_name = request.get("model_name", "Model")
        template_tag = request.get("template_tag")
        meta = request.get("meta")

        if not code:
            raise HTTPException(400, "代码内容不能为空")

        # 保存到数据库和文件（GeneratedCode表）
        saved_code = code_service.create_code(
            db=db,
            name=model_name,
            code=code,
            template_tag=template_tag,
            meta=meta
        )

        # 同时创建Model表记录，用于训练
        from app.models.model import Model as ModelTable
        import json

        # 构建graph_json，包含nodes和connections
        graph_json = {
            "class_name": model_name,
            "model_name": model_name,
            "input_size": meta.get("input_size", 224) if meta else 224,
            "nodes": meta.get("nodes", []) if meta else [],
            "connections": meta.get("connections", []) if meta else [],
        }

        # 检查是否已存在同名模型，更新它而不是创建新的
        existing_model = db.query(ModelTable).filter(
            ModelTable.name == model_name,
            ModelTable.is_active == "active"
        ).first()

        if existing_model:
            # 更新现有模型
            existing_model.graph_json = graph_json
            existing_model.code_path = saved_code.file_path
            existing_model.template_tag = template_tag
            existing_model.updated_at = datetime.now()
            db.commit()
            db.refresh(existing_model)
            model_record = existing_model
            debug_log(f"更新已存在的Model记录: id={existing_model.id}, name={model_name}")
        else:
            # 创建新的Model记录
            model_record = ModelTable(
                name=model_name,
                description=f"自动生成的可训练模型: {model_name}",
                graph_json=graph_json,
                code_path=saved_code.file_path,
                template_tag=template_tag,
                version="1.0",
                is_active="active"
            )
            db.add(model_record)
            db.commit()
            db.refresh(model_record)
            debug_log(f"创建新的Model记录: id={model_record.id}, name={model_name}")

        return {
            "message": "代码已保存并创建可训练模型",
            "code_id": saved_code.id,
            "model_id": model_record.id,  # 返回Model表ID，用于创建训练任务
            "filename": saved_code.file_name,
            "filepath": saved_code.file_path
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"保存代码失败: {str(e)}"
        )


# ==============================
# 可训练模型管理（Model表）
# ==============================

@router.get("/trainable")
async def list_trainable_models(
    db: Session = Depends(get_db)
):
    """
    获取可训练模型列表

    返回Model表中的所有可训练模型，这些模型包含生成的代码文件。
    """
    try:
        from app.models.model import Model as ModelTable

        models = db.query(ModelTable).filter(
            ModelTable.is_active == "active"
        ).order_by(
            ModelTable.updated_at.desc()
        ).all()

        result = []
        for m in models:
            result.append({
                "id": m.id,
                "name": m.name,
                "description": m.description,
                "code_path": m.code_path,
                "version": m.version,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "updated_at": m.updated_at.isoformat() if m.updated_at else None,
                "has_code": bool(m.code_path),
                "class_name": m.graph_json.get("class_name", "Model") if m.graph_json else "Model"
            })

        debug_log(f"获取可训练模型列表: 共{len(result)}个")

        return {
            "models": result,
            "total": len(result)
        }

    except Exception as e:
        debug_log(f"获取可训练模型列表失败: {e}", "ERROR")
        raise HTTPException(
            status_code=500,
            detail=f"获取模型列表失败: {str(e)}"
        )


# ==============================
# 模型架构管理端点（数据库版本）
# ==============================

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
    overwrite: bool = Query(False, description="是否覆盖同名架构"),
    target_id: Optional[int] = Query(None, description="指定目标ID（用于更新）"),
    target_filename: Optional[str] = Query(None, description="指定目标文件名（用于更新，兼容旧版本）"),
    db: Session = Depends(get_db)
):
    """
    保存模型架构到数据库

    将模型架构数据保存到数据库和文件系统。
    - 如果指定了 target_id，则更新指定ID的架构
    - 如果指定了 target_filename，则更新指定文件名的架构（兼容旧版本）
    - 如果 overwrite=true，则覆盖同名架构；否则报错
    """
    try:
        # 构建创建数据
        create_data = ModelArchitectureCreate(
            name=architecture.name,
            description=architecture.description,
            type=architecture.type,
            nodes=architecture.nodes,
            connections=architecture.connections
        )

        # 确定更新目标：优先使用target_id，其次使用target_filename
        update_target = None
        if target_id:
            update_target = str(target_id)
        elif target_filename:
            update_target = target_filename

        # 通过服务层保存
        saved_arch = architecture_service.create_architecture(
            db=db,
            data=create_data,
            user_id=None,  # TODO: 从认证中获取用户ID
            overwrite=overwrite,
            target_filename=update_target
        )

        # 判断是更新还是新建
        is_update = update_target is not None

        return {
            "message": "架构已更新" if is_update else "架构已保存",
            "id": saved_arch.id,
            "filename": saved_arch.file_name,
            "filepath": saved_arch.file_path,
            "updated": is_update
        }

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"保存架构失败: {str(e)}"
        )


@router.get("/architectures", response_model=ModelArchitectureList)
async def list_architectures(
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(100, ge=1, le=100, description="返回数量"),
    db: Session = Depends(get_db)
):
    """
    获取已保存的模型架构列表

    从数据库返回所有模型架构信息。
    """
    try:
        architectures = architecture_service.list_architectures(db, skip=skip, limit=limit)
        total = architecture_service.count_architectures(db)

        items = []
        for arch in architectures:
            items.append({
                "id": arch.id,
                "name": arch.name,
                "description": arch.description or "",
                "type": arch.type,
                "node_count": arch.node_count,
                "connection_count": arch.connection_count,
                "file_name": arch.file_name,
                "created": arch.created_at.isoformat() if arch.created_at else "",
                "updated": arch.updated_at.isoformat() if arch.updated_at else ""
            })

        return ModelArchitectureList(
            architectures=items,
            total=total
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取架构列表失败: {str(e)}"
        )


@router.get("/architectures/{architecture_id}")
async def get_architecture(
    architecture_id: int,
    db: Session = Depends(get_db)
):
    """
    获取模型架构的详细内容

    根据ID返回模型架构的完整数据。
    """
    try:
        architecture = architecture_service.get_architecture(db, architecture_id)
        if not architecture:
            raise HTTPException(404, "架构不存在")

        # 从文件加载完整数据
        data = architecture_service.load_architecture_file(architecture)

        return data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"读取架构失败: {str(e)}"
        )


@router.delete("/architectures/{architecture_id}")
async def delete_architecture(
    architecture_id: int,
    db: Session = Depends(get_db)
):
    """
    删除模型架构

    根据ID删除模型架构（同时删除数据库记录和物理文件）。
    """
    try:
        success = architecture_service.delete_architecture(db, architecture_id, physical=True)
        if not success:
            raise HTTPException(404, "架构不存在")

        return {"message": "架构已删除", "id": architecture_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"删除架构失败: {str(e)}"
        )
