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
    DatasetValidation, ImageListResponse, ImageDetail, AugmentationConfig,
    AugmentationPreview, DetailedDatasetStatistics, PaginationParams,
    FilterParams, AugmentedImage
)
from app.utils.responses import APIResponse, PaginatedResponse, paginated_response
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

        return paginated_response(
            message="获取数据集列表成功",
            data=datasets,
            page=skip // limit + 1,
            page_size=limit,
            total=total
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


# ========== 第4天新增的API端点 ==========

@router.get("/{dataset_id}/images", response_model=APIResponse[ImageListResponse])
async def get_dataset_images(
    dataset_id: int,
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页大小"),
    sort_by: str = Query("filename", description="排序字段"),
    sort_order: str = Query("asc", pattern="^(asc|desc)$", description="排序顺序"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取数据集图像列表（支持分页）
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 使用数据增强服务获取图像列表
        from app.services.augmentation_service import augmentation_service

        result = await augmentation_service.get_dataset_images(
            dataset_path=dataset.path,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order
        )

        return APIResponse(
            success=True,
            message="获取图像列表成功",
            data=ImageListResponse(**result)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图像列表失败: {str(e)}")


@router.get("/{dataset_id}/images/{image_path:path}", response_model=APIResponse[ImageDetail])
async def get_image_detail(
    dataset_id: int,
    image_path: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取单个图像的详细信息
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 验证图像路径是否在数据集路径内
        from pathlib import Path
        dataset_root = Path(dataset.path).resolve()
        image_full_path = (dataset_root / image_path).resolve()

        try:
            image_full_path.relative_to(dataset_root)
        except ValueError:
            raise HTTPException(status_code=400, detail="图像路径不在数据集范围内")

        # 使用数据增强服务获取图像详情
        from app.services.augmentation_service import augmentation_service

        result = await augmentation_service.get_image_detail(
            dataset_path=dataset.path,
            image_path=str(image_full_path)
        )

        if not result:
            raise HTTPException(status_code=404, detail="图像不存在或无法读取")

        return APIResponse(
            success=True,
            message="获取图像详情成功",
            data=ImageDetail(**result)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取图像详情失败: {str(e)}")


@router.get("/{dataset_id}/images/{image_path:path}/thumbnail")
async def get_image_thumbnail(
    dataset_id: int,
    image_path: str,
    size: int = Query(256, ge=64, le=512, description="缩略图尺寸"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取图像缩略图
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 验证图像路径
        from pathlib import Path
        dataset_root = Path(dataset.path).resolve()
        image_full_path = (dataset_root / image_path).resolve()

        try:
            image_full_path.relative_to(dataset_root)
        except ValueError:
            raise HTTPException(status_code=400, detail="图像路径不在数据集范围内")

        # 生成缩略图
        thumbnail_data = image_processor.get_thumbnail(str(image_full_path), (size, size))
        if not thumbnail_data:
            raise HTTPException(status_code=404, detail="无法生成缩略图")

        from fastapi.responses import Response
        return Response(content=thumbnail_data, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取缩略图失败: {str(e)}")


@router.get("/{dataset_id}/images/{image_path:path}/preview")
async def get_image_preview(
    dataset_id: int,
    image_path: str,
    max_size: int = Query(1024, ge=512, le=2048, description="预览图最大尺寸"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取图像预览
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 验证图像路径
        from pathlib import Path
        dataset_root = Path(dataset.path).resolve()
        image_full_path = (dataset_root / image_path).resolve()

        try:
            image_full_path.relative_to(dataset_root)
        except ValueError:
            raise HTTPException(status_code=400, detail="图像路径不在数据集范围内")

        # 生成预览图
        preview_data = image_processor.get_preview(str(image_full_path), (max_size, max_size))
        if not preview_data:
            raise HTTPException(status_code=404, detail="无法生成预览图")

        from fastapi.responses import Response
        return Response(content=preview_data, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取预览图失败: {str(e)}")


@router.post("/{dataset_id}/augment", response_model=APIResponse[AugmentationPreview])
async def augment_dataset_image(
    dataset_id: int,
    image_path: str = Query(..., description="图像相对路径"),
    augmentation_configs: List[AugmentationConfig] = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    对数据集中的图像进行数据增强预览
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 验证图像路径
        from pathlib import Path
        dataset_root = Path(dataset.path).resolve()
        image_full_path = (dataset_root / image_path).resolve()

        try:
            image_full_path.relative_to(dataset_root)
        except ValueError:
            raise HTTPException(status_code=400, detail="图像路径不在数据集范围内")

        # 检查图像是否存在
        if not image_full_path.exists():
            raise HTTPException(status_code=404, detail="图像文件不存在")

        # 转换配置为字典格式
        configs = [config.dict() for config in augmentation_configs]

        # 使用数据增强服务
        from app.services.augmentation_service import augmentation_service

        result = await augmentation_service.preview_augmentation(
            image_path=str(image_full_path),
            augmentation_configs=configs
        )

        return APIResponse(
            success=True,
            message="图像增强预览生成成功",
            data=AugmentationPreview(**result)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"图像增强预览失败: {str(e)}")


@router.get("/{dataset_id}/detailed-stats", response_model=APIResponse[DetailedDatasetStatistics])
async def get_detailed_dataset_statistics(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取数据集详细统计信息
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 使用数据增强服务获取详细统计
        from app.services.augmentation_service import augmentation_service

        result = await augmentation_service.get_dataset_statistics(dataset.path)

        return APIResponse(
            success=True,
            message="获取详细统计信息成功",
            data=DetailedDatasetStatistics(**result)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取详细统计信息失败: {str(e)}")


@router.post("/{dataset_id}/batch-augment", response_model=APIResponse[List[AugmentedImage]])
async def batch_augment_images(
    dataset_id: int,
    image_paths: List[str] = Query(..., description="图像路径列表"),
    augmentation_config: AugmentationConfig = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    批量数据增强（限制数量以避免性能问题）
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 限制批量处理数量
        if len(image_paths) > 10:
            raise HTTPException(status_code=400, detail="批量增强最多支持10张图像")

        # 验证所有图像路径
        from pathlib import Path
        dataset_root = Path(dataset.path).resolve()
        valid_paths = []

        for rel_path in image_paths:
            try:
                full_path = (dataset_root / rel_path).resolve()
                full_path.relative_to(dataset_root)
                if full_path.exists():
                    valid_paths.append(str(full_path))
            except (ValueError, FileNotFoundError):
                continue

        if not valid_paths:
            raise HTTPException(status_code=404, detail="没有找到有效的图像文件")

        # 批量处理
        from app.services.augmentation_service import augmentation_service
        config_dict = augmentation_config.dict()

        results = []
        for image_path in valid_paths:
            augmented_images = await augmentation_service.augment_image(
                image_path=image_path,
                augmentation_configs=[config_dict]
            )
            if augmented_images:
                results.extend(augmented_images)

        return APIResponse(
            success=True,
            message=f"批量增强完成，处理了{len(valid_paths)}张图像",
            data=[AugmentedImage(**img) for img in results]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量增强失败: {str(e)}")


@router.get("/{dataset_id}/filter-options", response_model=APIResponse[Dict])
async def get_dataset_filter_options(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取数据集过滤选项
    """
    try:
        dataset = dataset_service.get_dataset(db, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail=f"数据集 {dataset_id} 不存在")

        # 使用数据增强服务获取图像列表用于分析
        from app.services.augmentation_service import augmentation_service

        images_result = await augmentation_service.get_dataset_images(
            dataset_path=dataset.path,
            page=1,
            page_size=100  # 获取样本数据
        )

        images = images_result.get('images', [])

        # 分析过滤选项
        formats = list(set(img.get('format', 'unknown') for img in images))
        size_ranges = ['small', 'medium', 'large']

        # 获取类别列表
        classes = set()
        for img in images:
            for ann in img.get('annotations', []):
                class_name = ann.get('class_name')
                if class_name:
                    classes.add(class_name)

        filter_options = {
            'formats': formats,
            'size_ranges': size_ranges,
            'classes': sorted(list(classes)),
            'has_annotations': any(img.get('annotations') for img in images),
            'total_images_sample': len(images)
        }

        return APIResponse(
            success=True,
            message="获取过滤选项成功",
            data=filter_options
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取过滤选项失败: {str(e)}")