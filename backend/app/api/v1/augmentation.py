"""
数据增强相关API路由
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.database import get_db
from app.schemas.dataset import (
    AugmentationOperatorSchema,
    AugmentationStrategyCreate,
    AugmentationStrategyUpdate,
    AugmentationStrategyResponse,
    AugmentationStrategyList,
    AugmentationPreview,
    AugmentationConfig,
    AugmentedImage,
    PipelineItem
)
from app.utils.responses import APIResponse
from app.dependencies import get_current_user
from app.models.user import User
from app.models.augmentation import AugmentationStrategy
from app.utils.image_processor import image_processor
from app.core.augmentation_registry import (
    get_operators_dict,
    get_all_operators,
    operator_to_dict,
    get_operator_by_id
)
from loguru import logger
import random

router = APIRouter()


@router.get("/operators", response_model=APIResponse[dict])
async def get_augmentation_operators(
    current_user: User = Depends(get_current_user)
):
    """
    获取所有可用的数据增强算子列表

    返回按分类组织的算子列表，每个算子包含：
    - 算子ID、中英文名称
    - 功能描述
    - 参数定义（类型、范围、默认值、中文标签）
    """
    try:
        operators_dict = get_operators_dict()
        return APIResponse(
            success=True,
            message="获取增强算子列表成功",
            data=operators_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取增强算子列表失败: {str(e)}")


@router.get("/strategies", response_model=APIResponse[AugmentationStrategyList])
async def get_augmentation_strategies(
    skip: int = Query(0, ge=0, description="跳过的记录数"),
    limit: int = Query(20, ge=1, le=100, description="返回的记录数"),
    search: Optional[str] = Query(None, description="搜索关键词"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取用户的数据增强策略预设列表

    支持分页和搜索功能
    """
    try:
        query = db.query(AugmentationStrategy).filter(
            AugmentationStrategy.user_id == current_user.id
        )

        # 搜索功能
        if search:
            query = query.filter(
                or_(
                    AugmentationStrategy.name.contains(search),
                    AugmentationStrategy.description.contains(search)
                )
            )

        # 分页
        total = query.count()
        strategies = query.order_by(
            AugmentationStrategy.is_default.desc(),
            AugmentationStrategy.updated_at.desc()
        ).offset(skip).limit(limit).all()

        strategy_data = []
        for strategy in strategies:
            # 转换为字典格式
            strategy_dict = {
                "id": strategy.id,
                "user_id": strategy.user_id,
                "name": strategy.name,
                "description": strategy.description,
                "pipeline": strategy.pipeline if isinstance(strategy.pipeline, list) else [],
                "is_default": bool(strategy.is_default),
                "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
                "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None
            }
            strategy_data.append(AugmentationStrategyResponse(**strategy_dict))

        result = AugmentationStrategyList(
            strategies=strategy_data,
            total=total
        )

        return APIResponse(
            success=True,
            message="获取增强策略列表成功",
            data=result
        )
    except Exception as e:
        logger.error(f"获取增强策略列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取增强策略列表失败: {str(e)}")


@router.get("/strategies/{strategy_id}", response_model=APIResponse[AugmentationStrategyResponse])
async def get_augmentation_strategy(
    strategy_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    获取单个数据增强策略详情
    """
    try:
        strategy = db.query(AugmentationStrategy).filter(
            AugmentationStrategy.id == strategy_id,
            AugmentationStrategy.user_id == current_user.id
        ).first()

        if not strategy:
            raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")

        strategy_dict = {
            "id": strategy.id,
            "user_id": strategy.user_id,
            "name": strategy.name,
            "description": strategy.description,
            "pipeline": strategy.pipeline if isinstance(strategy.pipeline, list) else [],
            "is_default": bool(strategy.is_default),
            "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
            "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None
        }

        return APIResponse(
            success=True,
            message="获取增强策略详情成功",
            data=AugmentationStrategyResponse(**strategy_dict)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取增强策略详情失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取增强策略详情失败: {str(e)}")


@router.post("/strategies", response_model=APIResponse[AugmentationStrategyResponse])
async def create_augmentation_strategy(
    strategy_data: AugmentationStrategyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    创建新的数据增强策略预设
    """
    try:
        # 检查同名策略
        existing = db.query(AugmentationStrategy).filter(
            AugmentationStrategy.user_id == current_user.id,
            AugmentationStrategy.name == strategy_data.name
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="已存在同名策略")

        # 创建新策略
        new_strategy = AugmentationStrategy(
            user_id=current_user.id,
            name=strategy_data.name,
            description=strategy_data.description,
            pipeline=[p.model_dump() for p in strategy_data.pipeline]
        )

        db.add(new_strategy)
        db.commit()
        db.refresh(new_strategy)

        strategy_dict = {
            "id": new_strategy.id,
            "user_id": new_strategy.user_id,
            "name": new_strategy.name,
            "description": new_strategy.description,
            "pipeline": new_strategy.pipeline if isinstance(new_strategy.pipeline, list) else [],
            "is_default": bool(new_strategy.is_default),
            "created_at": new_strategy.created_at.isoformat() if new_strategy.created_at else None,
            "updated_at": new_strategy.updated_at.isoformat() if new_strategy.updated_at else None
        }

        return APIResponse(
            success=True,
            message="增强策略创建成功",
            data=AugmentationStrategyResponse(**strategy_dict)
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"创建增强策略失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建增强策略失败: {str(e)}")


@router.put("/strategies/{strategy_id}", response_model=APIResponse[AugmentationStrategyResponse])
async def update_augmentation_strategy(
    strategy_id: int,
    strategy_data: AugmentationStrategyUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    更新数据增强策略预设
    """
    try:
        strategy = db.query(AugmentationStrategy).filter(
            AugmentationStrategy.id == strategy_id,
            AugmentationStrategy.user_id == current_user.id
        ).first()

        if not strategy:
            raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")

        # 检查同名策略
        if strategy_data.name:
            existing = db.query(AugmentationStrategy).filter(
                AugmentationStrategy.user_id == current_user.id,
                AugmentationStrategy.name == strategy_data.name,
                AugmentationStrategy.id != strategy_id
            ).first()
            if existing:
                raise HTTPException(status_code=400, detail="已存在同名策略")

        # 更新字段
        if strategy_data.name is not None:
            strategy.name = strategy_data.name
        if strategy_data.description is not None:
            strategy.description = strategy_data.description
        if strategy_data.pipeline is not None:
            strategy.pipeline = [p.dict() for p in strategy_data.pipeline]

        db.commit()
        db.refresh(strategy)

        strategy_dict = {
            "id": strategy.id,
            "user_id": strategy.user_id,
            "name": strategy.name,
            "description": strategy.description,
            "pipeline": strategy.pipeline if isinstance(strategy.pipeline, list) else [],
            "is_default": bool(strategy.is_default),
            "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
            "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None
        }

        return APIResponse(
            success=True,
            message="增强策略更新成功",
            data=AugmentationStrategyResponse(**strategy_dict)
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"更新增强策略失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新增强策略失败: {str(e)}")


@router.delete("/strategies/{strategy_id}", response_model=APIResponse[bool])
async def delete_augmentation_strategy(
    strategy_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    删除数据增强策略预设
    """
    try:
        strategy = db.query(AugmentationStrategy).filter(
            AugmentationStrategy.id == strategy_id,
            AugmentationStrategy.user_id == current_user.id
        ).first()

        if not strategy:
            raise HTTPException(status_code=404, detail=f"策略 {strategy_id} 不存在")

        db.delete(strategy)
        db.commit()

        return APIResponse(
            success=True,
            message="增强策略删除成功",
            data=True
        )
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"删除增强策略失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除增强策略失败: {str(e)}")


@router.post("/preview", response_model=APIResponse[AugmentationPreview])
async def preview_augmentation(
    image_path: str = Query(..., description="图像路径"),
    dataset_id: Optional[int] = Query(None, description="数据集ID（用于mosaic等需要多图的算子）"),
    seed: Optional[int] = Query(None, description="随机种子"),
    pipeline: List[PipelineItem] = Body(..., description="增强流水线"),
    current_user: User = Depends(get_current_user)
):
    """
    预览数据增强效果

    应用指定的增强流水线到图像，返回增强后的图像预览

    支持多图算子（如mosaic、copy_paste）：
    - 传入dataset_id后，系统会从数据集中额外获取图片用于多图增强
    """
    try:
        from pathlib import Path
        import os
        import glob

        # 验证图像路径
        img_path = Path(image_path)
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="图像文件不存在")

        # 获取数据集中的其他图片（用于mosaic、copy_paste等需要多图的算子）
        other_images = []
        if dataset_id:
            # 获取图片所在目录
            img_dir = img_path.parent
            # 获取目录中的其他图片
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            try:
                all_images = []
                for ext in image_extensions:
                    all_images.extend(glob.glob(str(img_dir / f"*{ext}")))
                    all_images.extend(glob.glob(str(img_dir / f"*{ext.upper()}")))

                # 过滤掉当前图片，最多取10张
                other_image_paths = [p for p in all_images if p != str(img_path)][:10]

                # 加载其他图片
                for img_p in other_image_paths:
                    img = image_processor.load_image(img_p)
                    if img is not None:
                        other_images.append(img)
            except Exception as e:
                logger.warning(f"获取数据集其他图片失败: {str(e)}")

        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            import numpy as np
            np.random.seed(seed)

        # 加载原始图像
        original_image = image_processor.load_image(str(img_path))
        if original_image is None:
            raise HTTPException(status_code=400, detail="无法加载图像")

        # 生成原始图像的base64
        original_bytes = image_processor.image_array_to_bytes(original_image)
        original_base64 = image_processor.image_to_base64(original_bytes)

        augmented_images = []
        applied_operations_summary = []

        # 应用每个增强操作
        result_image = original_image.copy()
        for item in pipeline:
            # PipelineItem 是 Pydantic 模型，使用点号访问属性
            operator_id = item.operatorId
            enabled = item.enabled
            params = item.params

            if not enabled:
                continue

            operator = get_operator_by_id(operator_id)
            if not operator:
                continue

            # 根据算子ID应用相应的增强方法
            try:
                result_image = _apply_operator(result_image, operator_id, params, other_images)
                op_name_zh = operator.name_zh
                applied_operations_summary.append(op_name_zh)
            except Exception as e:
                logger.warning(f"应用算子 {operator_id} 失败: {str(e)}")

        # 生成增强图像的base64
        aug_bytes = image_processor.image_array_to_bytes(result_image)
        aug_base64 = image_processor.image_to_base64(aug_bytes)

        augmented_images.append(AugmentedImage(
            original_path=str(img_path),
            augmented_data=aug_base64,
            augmentation_config=AugmentationConfig(),
            applied_operations=applied_operations_summary
        ))

        summary = {
            "total_operators": len(pipeline),
            "applied_operators": len(applied_operations_summary),
            "operations_used": applied_operations_summary,
            "image_info": image_processor.get_image_info(str(img_path))
        }

        result = AugmentationPreview(
            original_image=original_base64,
            augmented_images=augmented_images,
            augmentation_summary=summary
        )

        return APIResponse(
            success=True,
            message="增强预览生成成功",
            data=result
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"增强预览失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"增强预览失败: {str(e)}")


def _apply_operator(image, operator_id: str, params: dict, other_images: list = None):
    """
    应用单个增强算子

    Args:
        image: 输入图像
        operator_id: 算子ID
        params: 算子参数
        other_images: 其他图像列表（用于多图算子如mosaic、copy_paste）

    Returns:
        增强后的图像
    """
    if other_images is None:
        other_images = []

    # 支持概率参数的算子列表（这些算子可能会根据概率跳过执行）
    PROBABILITY_OPERATORS = {
        "rotate", "scale", "translate", "elastic_transform",
        "brightness", "contrast", "saturation", "hue_shift",
        "gamma_correction", "auto_contrast",
        "gaussian_blur", "motion_blur", "gaussian_noise", "salt_pepper_noise"
    }

    # 检查是否执行（基于概率）
    if operator_id in PROBABILITY_OPERATORS:
        probability = params.get("probability", 1.0)
        import random as rnd
        if rnd.random() > probability:
            return image  # 不执行，返回原图

    # 几何变换类
    if operator_id == "horizontal_flip":
        return image_processor.flip_image(image, horizontal=True, vertical=False)
    elif operator_id == "vertical_flip":
        return image_processor.flip_image(image, horizontal=False, vertical=True)
    elif operator_id == "rotate":
        angle = params.get("angle", 0.0)
        return image_processor.rotate_image(image, angle)
    elif operator_id == "scale":
        factor = params.get("factor", 1.0)
        return image_processor.scale_image(image, factor)
    elif operator_id == "crop":
        return image_processor.crop_image(
            image,
            params.get("x", 0),
            params.get("y", 0),
            params.get("width", 100),
            params.get("height", 100)
        )
    elif operator_id == "translate":
        return image_processor.translate_image(
            image,
            params.get("dx", 0),
            params.get("dy", 0),
            params.get("fill_value", 255)
        )
    elif operator_id == "elastic_transform":
        return image_processor.elastic_transform(
            image,
            params.get("alpha", 1.0),
            params.get("sigma", 50.0)
        )

    # 颜色变换类
    elif operator_id == "brightness":
        factor = params.get("factor", 1.0)
        return image_processor.adjust_brightness(image, factor)
    elif operator_id == "contrast":
        factor = params.get("factor", 1.0)
        return image_processor.adjust_contrast(image, factor)
    elif operator_id == "saturation":
        factor = params.get("factor", 1.0)
        return image_processor.adjust_saturation(image, factor)
    elif operator_id == "hue_shift":
        shift = params.get("shift", 0.0)
        return image_processor.adjust_hue(image, shift)
    elif operator_id == "gamma_correction":
        gamma = params.get("gamma", 1.0)
        return image_processor.gamma_correction(image, gamma)
    elif operator_id == "auto_contrast":
        cutoff = params.get("cutoff", 0.0)
        return image_processor.auto_contrast(image, cutoff)

    # 模糊与噪声类
    elif operator_id == "gaussian_blur":
        sigma = params.get("sigma", 0.0)
        return image_processor.add_gaussian_blur(image, sigma)
    elif operator_id == "motion_blur":
        kernel_size = params.get("kernel_size", 15)
        angle = params.get("angle", 0.0)
        return image_processor.motion_blur(image, kernel_size, angle)
    elif operator_id == "gaussian_noise":
        std = params.get("std", 0.0)
        return image_processor.add_noise(image, std)
    elif operator_id == "salt_pepper_noise":
        amount = params.get("amount", 0.01)
        return image_processor.add_salt_pepper_noise(image, amount)

    # 其他类
    elif operator_id == "random_erase":
        probability = params.get("probability", 0.5)
        scale = params.get("scale", (0.02, 0.33))
        ratio = params.get("ratio", (0.3, 3.3))
        value = params.get("value", 0)
        return image_processor.random_erase(image, probability, scale, ratio, value)
    elif operator_id == "jpeg_compression":
        quality = params.get("quality", 85)
        return image_processor.jpeg_compression(image, quality)
    elif operator_id == "mosaic":
        # 马赛克增强：将4张图片拼接成1张
        probability = params.get("probability", 0.5)
        import random as rnd
        if rnd.random() > probability:
            return image  # 不执行，返回原图
        scale = params.get("scale", (0.5, 1.5))
        # 将当前图片和其他图片组合
        mosaic_images = [image] + other_images
        return image_processor.mosaic_augment(mosaic_images, scale)
    elif operator_id == "copy_paste":
        # Copy-Paste增强
        probability = params.get("probability", 0.5)
        import random as rnd
        if rnd.random() > probability or not other_images:
            return image  # 不执行或没有其他图片，返回原图
        max_objects = params.get("max_objects", 5)
        result, _ = image_processor.copy_paste_augment(image, other_images, max_objects)
        return result

    return image
