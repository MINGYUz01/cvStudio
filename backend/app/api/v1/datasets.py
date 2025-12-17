"""
数据集相关API路由
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.dataset_service import DatasetService
from app.schemas.dataset import (
    DatasetCreate, DatasetRegister, DatasetUpdate, DatasetResponse,
    DatasetList, DatasetPreview, DatasetStatistics, FormatRecognitionResult,
    DatasetValidation
)
from app.utils.responses import APIResponse, PaginatedResponse
from app.dependencies import get_current_user
from app.models.user import User

router = APIRouter()
dataset_service = DatasetService()


@router.post("/upload", response_model=APIResponse[DatasetResponse])
async def upload_dataset(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    name: str = Form(...),
    description: str = Form(None),
    files: List[UploadFile] = File(...)
):
    """
    上传并创建数据集
    """
    try:
        dataset = await dataset_service.create_dataset_from_upload(
            db=db,
            name=name,
            description=description,
            files=files,
            user_id=current_user.id
        )
        return APIResponse(
            success=True,
            message=f"数据集 '{name}' 创建成功",
            data=dataset
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传数据集失败: {str(e)}")


@router.post("/register", response_model=APIResponse[DatasetResponse])
async def register_dataset(
    dataset_data: DatasetRegister,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    注册现有数据集
    """
    try:
        dataset = dataset_service.register_existing_dataset(
            db=db,
            name=dataset_data.name,
            description=dataset_data.description,
            dataset_path=dataset_data.dataset_path,
            user_id=current_user.id
        )
        return APIResponse(
            success=True,
            message=f"数据集 '{dataset_data.name}' 注册成功",
            data=dataset
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"注册数据集失败: {str(e)}")


@router.get("/", response_model=PaginatedResponse[DatasetResponse])
async def get_datasets(
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(20, ge=1, le=100, description="返回的记录数"),
    format_filter: Optional[str] = Query(None, description="格式过滤器"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取数据集列表
    """
    try:
        datasets = dataset_service.get_datasets(
            db=db,
            skip=skip,
            limit=limit,
            format_filter=format_filter
        )

        # 获取总数用于分页
        from app.models.dataset import Dataset
        query = db.query(Dataset).filter(Dataset.is_active == "active")
        if format_filter:
            query = query.filter(Dataset.format == format_filter)
        total = query.count()

        return PaginatedResponse(
            success=True,
            message="获取数据集列表成功",
            data=datasets,
            total=total,
            page=skip // limit + 1,
            page_size=limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据集列表失败: {str(e)}")


@router.get("/{dataset_id}", response_model=APIResponse[DatasetResponse])
async def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取单个数据集详情
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        return APIResponse(
            success=True,
            message="获取数据集详情成功",
            data=dataset
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据集详情失败: {str(e)}")


@router.put("/{dataset_id}", response_model=APIResponse[DatasetResponse])
async def update_dataset(
    dataset_id: int,
    dataset_data: DatasetUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    更新数据集信息
    """
    try:
        dataset = dataset_service.update_dataset(
            db=db,
            dataset_id=dataset_id,
            name=dataset_data.name,
            description=dataset_data.description
        )
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        return APIResponse(
            success=True,
            message="数据集信息更新成功",
            data=dataset
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新数据集失败: {str(e)}")


@router.delete("/{dataset_id}", response_model=APIResponse[bool])
async def delete_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    删除数据集（软删除）
    """
    try:
        success = dataset_service.delete_dataset(db, dataset_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        return APIResponse(
            success=True,
            message="数据集删除成功",
            data=True
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除数据集失败: {str(e)}")


@router.post("/{dataset_id}/rescan", response_model=APIResponse[DatasetResponse])
async def rescan_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    重新扫描数据集，更新元信息
    """
    try:
        dataset = dataset_service.rescan_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        return APIResponse(
            success=True,
            message="数据集重新扫描成功",
            data=dataset
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新扫描数据集失败: {str(e)}")


@router.post("/recognize-format", response_model=APIResponse[FormatRecognitionResult])
async def recognize_dataset_format(
    dataset_path: str = Form(...),
    current_user: User = Depends(get_current_user)
):
    """
    识别数据集格式
    """
    try:
        from app.utils.format_recognizers import DatasetFormatRecognizer
        recognizer = DatasetFormatRecognizer()

        result = recognizer.recognize_format(dataset_path)
        return APIResponse(
            success=True,
            message="数据集格式识别成功",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别数据集格式失败: {str(e)}")


@router.get("/{dataset_id}/preview", response_model=APIResponse[DatasetPreview])
async def preview_dataset(
    dataset_id: int,
    limit: int = Query(10, ge=1, le=50, description="预览图像数量"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取数据集预览
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 这个方法将在下一个TODO中实现
        preview_data = await dataset_service.get_dataset_preview(dataset_id, limit)

        return APIResponse(
            success=True,
            message="获取数据集预览成功",
            data=preview_data
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据集预览失败: {str(e)}")


@router.get("/{dataset_id}/statistics", response_model=APIResponse[DatasetStatistics])
async def get_dataset_statistics(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取数据集统计信息
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 这个方法将在下一个TODO中实现
        stats_data = await dataset_service.get_dataset_statistics(dataset_id)

        return APIResponse(
            success=True,
            message="获取数据集统计信息成功",
            data=stats_data
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取数据集统计信息失败: {str(e)}")


@router.post("/{dataset_id}/validate", response_model=APIResponse[DatasetValidation])
async def validate_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    验证数据集
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 这个方法将在下一个TODO中实现
        validation_result = await dataset_service.validate_dataset(dataset_id)

        return APIResponse(
            success=True,
            message="数据集验证完成",
            data=validation_result
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"验证数据集失败: {str(e)}")


@router.get("/formats/supported", response_model=APIResponse[List[str]])
async def get_supported_formats():
    """
    获取支持的数据集格式列表
    """
    try:
        from app.core.config import settings
        supported_formats = settings.SUPPORTED_DATASET_FORMATS

        return APIResponse(
            success=True,
            message="获取支持的数据集格式成功",
            data=supported_formats
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取支持的数据集格式失败: {str(e)}")


@router.get("/{dataset_id}/thumbnails", response_model=APIResponse[List[str]])
async def get_dataset_thumbnails(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取数据集缩略图列表
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        thumbnails = dataset_service.get_thumbnails_list(dataset_id)
        return APIResponse(
            success=True,
            message="获取缩略图列表成功",
            data=thumbnails
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取缩略图列表失败: {str(e)}")


@router.get("/{dataset_id}/thumbnail")
async def get_dataset_thumbnail(
    dataset_id: int,
    image_path: str = Query(..., description="原始图像路径"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取单个图像的缩略图
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        thumbnail_path = await dataset_service.get_thumbnail(dataset_id, image_path)
        if not thumbnail_path:
            raise HTTPException(status_code=404, detail="缩略图不存在")

        from fastapi.responses import FileResponse
        return FileResponse(thumbnail_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取缩略图失败: {str(e)}")


@router.post("/{dataset_id}/thumbnails/generate", response_model=APIResponse[bool])
async def generate_dataset_thumbnails(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    为数据集生成缩略图
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 异步生成缩略图
        asyncio.create_task(dataset_service._generate_thumbnails_async(dataset_id, dataset.path))

        return APIResponse(
            success=True,
            message="缩略图生成任务已启动",
            data=True
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动缩略图生成失败: {str(e)}")


@router.delete("/{dataset_id}/thumbnails", response_model=APIResponse[bool])
async def clear_dataset_thumbnails(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    清理数据集缩略图
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        success = dataset_service.clear_thumbnails(dataset_id)
        return APIResponse(
            success=True,
            message="缩略图清理完成",
            data=success
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理缩略图失败: {str(e)}")


@router.post("/compare", response_model=APIResponse[Dict])
async def compare_datasets(
    dataset_ids: List[int],
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    比较多个数据集的统计信息
    """
    try:
        comparison_result = await dataset_service.compare_datasets(db, dataset_ids)
        return APIResponse(
            success=True,
            message="数据集比较完成",
            data=comparison_result
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"比较数据集失败: {str(e)}")