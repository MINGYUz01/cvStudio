"""
权重库API路由

提供权重文件的上传、查询、更新、删除等功能。
支持任务类型自动检测和版本管理。

作者: CV Studio 开发团队
日期: 2025-01-15
"""

from fastapi import APIRouter, HTTPException, Query, Depends, UploadFile, File, Form
from typing import List, Optional
from sqlalchemy.orm import Session

from app.services.weight_library_service import WeightLibraryService
from app.schemas.weight_library import (
    WeightLibraryCreate,
    WeightLibraryUpdate,
    WeightLibraryResponse,
    WeightLibraryList,
    WeightLibraryListItem,
    WeightUploadRequest,
    WeightUploadResponse,
    TaskTypeDetectionResponse,
    WeightVersionCreate,
    WeightVersionHistory,
    WeightRootList,
    WeightTreeItem,
    WeightTreeResponse,
    WeightTrainingConfigResponse,
    WeightForTraining,
)
from app.dependencies import get_db
from loguru import logger


router = APIRouter()

# 初始化服务
weight_service = WeightLibraryService()


# ==============================
# API 端点
# ==============================

@router.get("/")
async def weights_root():
    """权重模块根路径"""
    return {
        "message": "权重库模块",
        "status": "运行中",
        "endpoints": {
            "upload": "POST /api/v1/weights/upload - 上传权重文件",
            "list": "GET /api/v1/weights - 获取权重列表",
            "detail": "GET /api/v1/weights/{weight_id} - 获取权重详情",
            "delete": "DELETE /api/v1/weights/{weight_id} - 删除权重",
            "detect": "POST /api/v1/weights/detect - 检测任务类型",
            "versions": "GET /api/v1/weights/{weight_id}/versions - 获取版本历史",
            "create-version": "POST /api/v1/weights/{weight_id}/versions - 创建新版本"
        }
    }


@router.post("/upload", response_model=WeightUploadResponse)
async def upload_weight(
    file: UploadFile = File(..., description="权重文件"),
    name: str = Form(..., description="权重名称"),
    task_type: str = Form(..., description="任务类型：classification/detection/auto"),
    description: Optional[str] = Form(None, description="权重描述"),
    input_size: Optional[str] = Form(None, description="输入尺寸JSON字符串，如 [224, 224]"),
    class_names: Optional[str] = Form(None, description="类别名称JSON字符串，如 [\"cat\", \"dog\"]"),
    normalize_params: Optional[str] = Form(None, description="归一化参数JSON字符串"),
    db: Session = Depends(get_db)
):
    """
    上传权重文件到权重库

    支持的文件格式：.pt, .pth, .pkl, .onnx
    支持的任务类型：classification, detection
    自动检测：task_type设为"auto"时自动检测任务类型

    返回创建的权重记录信息。
    """
    try:
        import json

        # 解析JSON字符串参数
        parsed_input_size = None
        if input_size:
            try:
                parsed_input_size = json.loads(input_size)
            except json.JSONDecodeError:
                raise HTTPException(400, "input_size 格式错误，应为JSON数组")

        parsed_class_names = None
        if class_names:
            try:
                parsed_class_names = json.loads(class_names)
            except json.JSONDecodeError:
                raise HTTPException(400, "class_names 格式错误，应为JSON数组")

        parsed_normalize_params = None
        if normalize_params:
            try:
                parsed_normalize_params = json.loads(normalize_params)
            except json.JSONDecodeError:
                raise HTTPException(400, "normalize_params 格式错误，应为JSON对象")

        # 调用服务层上传
        weight = await weight_service.upload_weight(
            file=file,
            name=name,
            task_type=task_type,
            description=description,
            input_size=parsed_input_size,
            class_names=parsed_class_names,
            normalize_params=parsed_normalize_params,
            uploaded_by=None,  # TODO: 从认证中获取用户ID
            db=db
        )

        return WeightUploadResponse(
            success=True,
            message="权重文件上传成功",
            weight_id=weight.id,
            weight=WeightLibraryResponse.model_validate(weight)
        )

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"上传权重失败: {e}")
        raise HTTPException(500, f"上传权重失败: {str(e)}")


@router.get("", response_model=WeightLibraryList)
async def list_weights(
    task_type: Optional[str] = Query(None, description="过滤任务类型"),
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(100, ge=1, le=100, description="返回数量"),
    db: Session = Depends(get_db)
):
    """
    获取权重文件列表

    支持按任务类型过滤和分页。
    """
    try:
        weights = weight_service.get_weights(
            db=db,
            task_type=task_type,
            is_active=True,
            skip=skip,
            limit=limit
        )

        # 统计总数
        total = len(weight_service.get_weights(
            db=db,
            task_type=task_type,
            is_active=True,
            skip=0,
            limit=10000  # 获取全部用于计数
        ))

        items = []
        for weight in weights:
            items.append(WeightLibraryListItem(
                id=weight.id,
                name=weight.name,
                display_name=weight.display_name,
                description=weight.description,
                task_type=weight.task_type,
                version=weight.version,
                file_name=weight.file_name,
                file_size_mb=round(weight.file_size / 1024 / 1024, 2) if weight.file_size else None,
                framework=weight.framework,
                is_auto_detected=weight.is_auto_detected,
                created_at=weight.created_at
            ))

        return WeightLibraryList(
            weights=items,
            total=total
        )

    except Exception as e:
        logger.error(f"获取权重列表失败: {e}")
        raise HTTPException(500, f"获取权重列表失败: {str(e)}")


# ==============================
# 权重树形结构相关端点
# 注意：这些特定路径必须在 /{weight_id} 之前定义！
# ==============================

@router.get("/roots", response_model=WeightRootList)
async def get_root_weights(
    task_type: Optional[str] = Query(None, description="过滤任务类型"),
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(100, ge=1, le=100, description="返回数量"),
    db: Session = Depends(get_db)
):
    """
    获取根节点权重列表

    只返回is_root=True的权重（导入的权重和训练best权重）
    """
    try:
        weights = weight_service.get_root_weights(
            db=db,
            task_type=task_type,
            skip=skip,
            limit=limit
        )

        # 统计总数
        total = len(weight_service.get_root_weights(
            db=db,
            task_type=task_type,
            skip=0,
            limit=10000
        ))

        items = []
        for weight in weights:
            items.append(WeightLibraryListItem(
                id=weight.id,
                name=weight.name,
                display_name=weight.display_name,
                description=weight.description,
                task_type=weight.task_type,
                version=weight.version,
                file_name=weight.file_name,
                file_size_mb=round(weight.file_size / 1024 / 1024, 2) if weight.file_size else None,
                framework=weight.framework,
                is_auto_detected=weight.is_auto_detected,
                is_root=weight.is_root,
                source_type=weight.source_type,
                architecture_id=weight.architecture_id,
                created_at=weight.created_at
            ))

        return WeightRootList(
            weights=items,
            total=total
        )

    except Exception as e:
        logger.error(f"获取根节点权重列表失败: {e}")
        raise HTTPException(500, f"获取根节点权重列表失败: {str(e)}")


@router.get("/tree", response_model=List[WeightTreeResponse])
async def get_weight_tree(
    db: Session = Depends(get_db)
):
    """
    获取完整的权重树形结构

    返回所有根节点及其子节点组成的树
    """
    try:
        trees = weight_service.build_weight_tree(db)
        return trees

    except Exception as e:
        logger.error(f"获取权重树失败: {e}")
        raise HTTPException(500, f"获取权重树失败: {str(e)}")


@router.get("/for-training", response_model=List[WeightForTraining])
async def get_weights_for_training(
    architecture_id: Optional[int] = Query(None, description="模型架构ID"),
    task_type: Optional[str] = Query(None, description="任务类型"),
    db: Session = Depends(get_db)
):
    """
    获取可用于训练的权重列表

    根据模型架构ID和任务类型筛选可用的预训练权重
    """
    try:
        weights = weight_service.get_weights_for_training(
            db=db,
            architecture_id=architecture_id,
            task_type=task_type
        )

        items = []
        for weight in weights:
            # 获取架构名称
            arch_name = None
            if weight.architecture:
                arch_name = weight.architecture.name

            items.append(WeightForTraining(
                id=weight.id,
                name=weight.name,
                display_name=weight.display_name,
                description=weight.description,
                task_type=weight.task_type,
                version=weight.version,
                file_path=weight.file_path,
                architecture_id=weight.architecture_id,
                architecture_name=arch_name,
                created_at=weight.created_at
            ))

        return items

    except Exception as e:
        logger.error(f"获取可用于训练的权重失败: {e}")
        raise HTTPException(500, f"获取可用于训练的权重失败: {str(e)}")


@router.get("/by-task/{task_type}", response_model=WeightLibraryList)
async def list_weights_by_task(
    task_type: str,
    skip: int = Query(0, ge=0, description="跳过数量"),
    limit: int = Query(100, ge=1, le=100, description="返回数量"),
    db: Session = Depends(get_db)
):
    """
    按任务类型获取权重列表

    快捷获取指定任务类型的所有权重。
    """
    try:
        # 验证任务类型
        valid_types = ['classification', 'detection']
        if task_type not in valid_types:
            raise HTTPException(400, f"无效的任务类型，应为: {', '.join(valid_types)}")

        weights = weight_service.get_weights(
            db=db,
            task_type=task_type,
            is_active=True,
            skip=skip,
            limit=limit
        )

        items = []
        for weight in weights:
            items.append(WeightLibraryListItem(
                id=weight.id,
                name=weight.name,
                display_name=weight.display_name,
                description=weight.description,
                task_type=weight.task_type,
                version=weight.version,
                file_name=weight.file_name,
                file_size_mb=round(weight.file_size / 1024 / 1024, 2) if weight.file_size else None,
                framework=weight.framework,
                is_auto_detected=weight.is_auto_detected,
                created_at=weight.created_at
            ))

        return WeightLibraryList(
            weights=items,
            total=len(items)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"按任务类型获取权重失败: {e}")
        raise HTTPException(500, f"按任务类型获取权重失败: {str(e)}")


# ==============================
# 带参数的权重路由（必须放在最后）
# ==============================

@router.get("/{weight_id}", response_model=WeightLibraryResponse)
async def get_weight(
    weight_id: int,
    db: Session = Depends(get_db)
):
    """
    获取权重文件详情

    根据ID返回权重文件的完整信息。
    """
    try:
        weight = weight_service.get_weight(db, weight_id)
        if not weight:
            raise HTTPException(404, "权重不存在")

        return WeightLibraryResponse.model_validate(weight)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取权重详情失败: {e}")
        raise HTTPException(500, f"获取权重详情失败: {str(e)}")


@router.delete("/{weight_id}")
async def delete_weight(
    weight_id: int,
    db: Session = Depends(get_db)
):
    """
    删除权重文件

    根据ID删除权重文件（同时删除数据库记录和物理文件）。
    """
    try:
        success = weight_service.delete_weight(db, weight_id)
        if not success:
            raise HTTPException(404, "权重不存在")

        return {"message": "权重已删除", "id": weight_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除权重失败: {e}")
        raise HTTPException(500, f"删除权重失败: {str(e)}")


@router.post("/{weight_id}/versions", response_model=WeightLibraryResponse)
async def create_weight_version(
    weight_id: int,
    file: UploadFile = File(..., description="新权重文件"),
    description: Optional[str] = Form(None, description="版本描述"),
    db: Session = Depends(get_db)
):
    """
    创建权重的新版本

    为现有权重创建新版本，自动递增版本号。
    """
    try:
        # 保存上传的文件到临时位置
        import tempfile
        import shutil
        from pathlib import Path

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # 调用服务层创建新版本
        new_weight = weight_service.create_new_version(
            db=db,
            parent_weight_id=weight_id,
            file_path=temp_path,
            description=description
        )

        # 清理临时文件
        Path(temp_path).unlink(missing_ok=True)

        if not new_weight:
            raise HTTPException(404, "父权重不存在")

        return WeightLibraryResponse.model_validate(new_weight)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"创建权重版本失败: {e}")
        raise HTTPException(500, f"创建权重版本失败: {str(e)}")


@router.get("/{weight_id}/versions", response_model=WeightVersionHistory)
async def get_weight_versions(
    weight_id: int,
    db: Session = Depends(get_db)
):
    """
    获取权重版本历史

    返回指定权重的所有版本信息。
    """
    try:
        versions = weight_service.get_version_history(db, weight_id)

        items = []
        for v in versions:
            items.append(WeightLibraryListItem(
                id=v.id,
                name=v.name,
                display_name=v.display_name,
                description=v.description,
                task_type=v.task_type,
                version=v.version,
                file_name=v.file_name,
                file_size_mb=round(v.file_size / 1024 / 1024, 2) if v.file_size else None,
                framework=v.framework,
                is_auto_detected=v.is_auto_detected,
                created_at=v.created_at
            ))

        return WeightVersionHistory(
            versions=items,
            total=len(items)
        )

    except Exception as e:
        logger.error(f"获取权重版本历史失败: {e}")
        raise HTTPException(500, f"获取权重版本历史失败: {str(e)}")


@router.get("/{weight_id}/tree", response_model=WeightTreeResponse)
async def get_weight_subtree(
    weight_id: int,
    db: Session = Depends(get_db)
):
    """
    获取指定权重的子树

    返回指定权重作为根节点的完整子树
    """
    try:
        subtree = weight_service.get_weight_subtree(db, weight_id)
        if not subtree:
            raise HTTPException(404, "权重不存在")

        return subtree

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取权重子树失败: {e}")
        raise HTTPException(500, f"获取权重子树失败: {str(e)}")


@router.get("/{weight_id}/config", response_model=WeightTrainingConfigResponse)
async def get_weight_training_config(
    weight_id: int,
    db: Session = Depends(get_db)
):
    """
    获取权重的训练配置信息

    返回权重关联的训练配置（如果有）
    """
    try:
        config = weight_service.get_weight_training_config(db, weight_id)
        return config

    except Exception as e:
        logger.error(f"获取权重训练配置失败: {e}")
        raise HTTPException(500, f"获取权重训练配置失败: {str(e)}")
