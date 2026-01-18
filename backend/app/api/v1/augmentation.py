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


def _sample_range_param(value, default=(0, 1)):
    """
    从范围参数中采样

    Args:
        value: 参数值（可能是范围列表或固定值）
        default: 默认范围

    Returns:
        采样后的值
    """
    import random as rnd
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return rnd.uniform(value[0], value[1])
    return value


def _sample_int_range_param(value, default=(0, 10)):
    """
    从整数范围参数中采样

    Args:
        value: 参数值（可能是范围列表或固定值）
        default: 默认范围

    Returns:
        采样后的整数值
    """
    import random as rnd
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return rnd.randint(int(value[0]), int(value[1]))
    return int(value)


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

    import random as rnd
    import numpy as np

    # 所有支持概率参数的算子
    PROBABILITY_OPERATORS = {
        # 几何变换
        "horizontal_flip", "vertical_flip", "rotate", "scale", "translate",
        "shift_scale_rotate", "random_resized_crop", "perspective", "shear",
        "elastic_transform",
        # 颜色变换
        "hsv_color", "brightness", "contrast", "saturation", "hue_shift",
        "rgb_shift", "to_gray", "gamma_correction", "auto_contrast",
        # 模糊与噪声
        "gaussian_blur", "motion_blur", "gaussian_noise", "salt_pepper_noise",
        # 其他
        "random_erase", "jpeg_compression", "mosaic", "copy_paste"
    }

    # 检查是否执行（基于概率）
    if operator_id in PROBABILITY_OPERATORS:
        probability = params.get("probability", 1.0)
        if rnd.random() > probability:
            return image  # 不执行，返回原图

    # ==================== 几何变换类 ====================
    if operator_id == "horizontal_flip":
        return image_processor.flip_image(image, horizontal=True, vertical=False)

    elif operator_id == "vertical_flip":
        return image_processor.flip_image(image, horizontal=False, vertical=True)

    elif operator_id == "rotate":
        # angle_limit: 角度范围，随机采样
        angle_limit = params.get("angle_limit", (-15, 15))
        angle = _sample_range_param(angle_limit, (-15, 15))
        return image_processor.rotate_image(image, angle)

    elif operator_id == "scale":
        # scale_limit: 缩放因子范围，随机采样
        scale_limit = params.get("scale_limit", (0.8, 1.2))
        scale = _sample_range_param(scale_limit, (0.8, 1.2))
        return image_processor.scale_image(image, scale)

    elif operator_id == "crop":
        # 固定裁剪（使用固定值）
        return image_processor.crop_image(
            image,
            params.get("x", 0),
            params.get("y", 0),
            params.get("width", 100),
            params.get("height", 100)
        )

    elif operator_id == "translate":
        # translate_limit: 平移比例范围，随机采样
        translate_limit = params.get("translate_limit", (-0.1, 0.1))
        ratio = _sample_range_param(translate_limit, (-0.1, 0.1))
        h, w = image.shape[:2]
        dx = int(ratio * w)
        dy = int(ratio * h)
        fill_value = params.get("fill_value", 0)
        return image_processor.translate_image(image, dx, dy, fill_value)

    elif operator_id == "shift_scale_rotate":
        # 综合几何变换
        shift_limit = params.get("shift_limit", (-0.1, 0.1))
        scale_limit = params.get("scale_limit", (-0.1, 0.1))
        rotate_limit = params.get("rotate_limit", (-15, 15))
        border_mode = params.get("border_mode", "reflect")

        h, w = image.shape[:2]
        shift_x = _sample_range_param(shift_limit, (-0.1, 0.1)) * w
        shift_y = _sample_range_param(shift_limit, (-0.1, 0.1)) * h
        scale = 1.0 + _sample_range_param(scale_limit, (-0.1, 0.1))
        angle = _sample_range_param(rotate_limit, (-15, 15))

        return _shift_scale_rotate(image, shift_x, shift_y, scale, angle, border_mode)

    elif operator_id == "random_resized_crop":
        # 随机裁剪缩放
        height = params.get("height", 224)
        width = params.get("width", 224)
        scale = params.get("scale", (0.8, 1.0))
        ratio = params.get("ratio", (0.75, 1.33))
        return _random_resized_crop(image, height, width, scale, ratio)

    elif operator_id == "perspective":
        # 透视变换
        scale = params.get("scale", (0.0, 0.05))
        intensity = _sample_range_param(scale, (0.0, 0.05))
        return _perspective_transform_random(image, intensity)

    elif operator_id == "shear":
        # 剪切变换
        shear_x = params.get("shear_x", (-10, 10))
        shear_y = params.get("shear_y", (-10, 10))
        angle_x = _sample_range_param(shear_x, (-10, 10))
        angle_y = _sample_range_param(shear_y, (-10, 10))
        return _shear_transform(image, angle_x, angle_y)

    elif operator_id == "elastic_transform":
        return image_processor.elastic_transform(
            image,
            params.get("alpha", 1.0),
            params.get("sigma", 50.0)
        )

    # ==================== 颜色变换类 ====================
    elif operator_id == "hsv_color":
        # HSV颜色空间调整
        h_limit = params.get("h_limit", (-20, 20))
        s_limit = params.get("s_limit", (-30, 30))
        v_limit = params.get("v_limit", (-20, 20))
        h_shift = _sample_range_param(h_limit, (-20, 20))
        s_shift = _sample_range_param(s_limit, (-30, 30)) / 100.0  # 转换为比例
        v_shift = _sample_range_param(v_limit, (-20, 20)) / 100.0
        return _hsv_color_adjust(image, h_shift, s_shift, v_shift)

    elif operator_id == "brightness":
        # 亮度调整
        limit = params.get("limit", (-0.2, 0.2))
        factor = 1.0 + _sample_range_param(limit, (-0.2, 0.2))
        return image_processor.adjust_brightness(image, factor)

    elif operator_id == "contrast":
        # 对比度调整
        limit = params.get("limit", (-0.2, 0.2))
        factor = 1.0 + _sample_range_param(limit, (-0.2, 0.2))
        return image_processor.adjust_contrast(image, factor)

    elif operator_id == "saturation":
        # 饱和度调整
        limit = params.get("limit", (-0.3, 0.3))
        factor = 1.0 + _sample_range_param(limit, (-0.3, 0.3))
        return image_processor.adjust_saturation(image, factor)

    elif operator_id == "hue_shift":
        # 色调偏移
        shift_limit = params.get("shift_limit", (-20, 20))
        shift = _sample_range_param(shift_limit, (-20, 20))
        return image_processor.adjust_hue(image, shift)

    elif operator_id == "rgb_shift":
        # RGB通道偏移
        r_limit = params.get("r_limit", (-20, 20))
        g_limit = params.get("g_limit", (-20, 20))
        b_limit = params.get("b_limit", (-20, 20))
        r_shift = _sample_range_param(r_limit, (-20, 20))
        g_shift = _sample_range_param(g_limit, (-20, 20))
        b_shift = _sample_range_param(b_limit, (-20, 20))
        return _rgb_shift_adjust(image, r_shift, g_shift, b_shift)

    elif operator_id == "to_gray":
        # 转灰度
        return _to_gray_convert(image)

    elif operator_id == "gamma_correction":
        # Gamma校正
        gamma_limit = params.get("gamma_limit", (80, 120))
        gamma = _sample_range_param(gamma_limit, (80, 120)) / 100.0
        return image_processor.gamma_correction(image, gamma)

    elif operator_id == "auto_contrast":
        cutoff = params.get("cutoff", (0, 20))
        cutoff_val = _sample_range_param(cutoff, (0, 20))
        return image_processor.auto_contrast(image, cutoff_val)

    # ==================== 模糊与噪声类 ====================
    elif operator_id == "gaussian_blur":
        blur_limit = params.get("blur_limit", (3, 7))
        kernel_size = _sample_int_range_param(blur_limit, (3, 7))
        # 确保是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = kernel_size / 3.0  # 根据核大小计算sigma
        return _gaussian_blur_with_kernel(image, kernel_size, sigma)

    elif operator_id == "motion_blur":
        blur_limit = params.get("blur_limit", (3, 7))
        angle_range = params.get("angle_range", (0, 360))
        kernel_size = _sample_int_range_param(blur_limit, (3, 7))
        if kernel_size % 2 == 0:
            kernel_size += 1
        angle = _sample_range_param(angle_range, (0, 360))
        return image_processor.motion_blur(image, kernel_size, angle)

    elif operator_id == "gaussian_noise":
        var_limit = params.get("var_limit", (10, 50))
        variance = _sample_range_param(var_limit, (10, 50))
        std = np.sqrt(variance)
        return image_processor.add_noise(image, std)

    elif operator_id == "salt_pepper_noise":
        amount = params.get("amount", (0.001, 0.05))
        amount_val = _sample_range_param(amount, (0.001, 0.05))
        return image_processor.add_salt_pepper_noise(image, amount_val)

    # ==================== 其他类 ====================
    elif operator_id == "random_erase":
        scale = params.get("scale", (0.02, 0.33))
        ratio = params.get("ratio", (0.3, 3.3))
        value = params.get("value", 0)
        return image_processor.random_erase(image, 1.0, scale, ratio, value)  # 概率已在前面检查

    elif operator_id == "jpeg_compression":
        quality_lower = params.get("quality_lower", 70)
        quality_upper = params.get("quality_upper", 100)
        quality = rnd.randint(quality_lower, quality_upper)
        return image_processor.jpeg_compression(image, quality)

    elif operator_id == "mosaic":
        # 马赛克增强
        scale = params.get("scale", (0.5, 1.5))
        # 将当前图片和其他图片组合
        mosaic_images = [image] + other_images
        return image_processor.mosaic_augment(mosaic_images, scale)

    elif operator_id == "copy_paste":
        # Copy-Paste增强
        if not other_images:
            return image
        max_objects = params.get("max_objects", 5)
        result, _ = image_processor.copy_paste_augment(image, other_images, max_objects)
        return result

    return image


# ==================== 新增算子实现函数 ====================

def _shift_scale_rotate(image, shift_x, shift_y, scale, angle, border_mode="reflect"):
    """综合几何变换：平移+缩放+旋转"""
    try:
        h, w = image.shape[:2]

        # 构建变换矩阵
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += shift_x
        M[1, 2] += shift_y

        # 确定边界模式
        border_modes = {
            "reflect": cv2.BORDER_REFLECT_101,
            "constant": cv2.BORDER_CONSTANT,
            "wrap": cv2.BORDER_WRAP
        }
        border_flag = border_modes.get(border_mode, cv2.BORDER_REFLECT_101)
        border_value = (0, 0, 0) if border_mode == "constant" else None

        # 计算新图像大小
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # 调整变换矩阵
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        # 应用变换
        result = cv2.warpAffine(image, M, (new_w, new_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=border_flag,
                              borderValue=border_value)
        return result
    except Exception as e:
        logger.warning(f"综合几何变换失败: {str(e)}")
        return image


def _random_resized_crop(image, height, width, scale_range, ratio_range):
    """随机裁剪缩放"""
    try:
        h, w = image.shape[:2]

        # 随机采样参数
        scale = np.random.uniform(*scale_range) if isinstance(scale_range, (list, tuple)) else scale_range
        ratio = np.random.uniform(*ratio_range) if isinstance(ratio_range, (list, tuple)) else ratio_range

        # 计算裁剪区域
        area = h * w
        target_area = scale * area

        aspect_ratio = ratio
        crop_h = int(round((target_area * aspect_ratio) ** 0.5))
        crop_w = int(round((target_area / aspect_ratio) ** 0.5))

        # 确保不越界
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        if crop_h < 1 or crop_w < 1:
            # 回退到简单缩放
            return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        # 随机位置
        top = np.random.randint(0, h - crop_h + 1) if crop_h < h else 0
        left = np.random.randint(0, w - crop_w + 1) if crop_w < w else 0

        # 裁剪
        cropped = image[top:top+crop_h, left:left+crop_w]

        # 缩放到目标尺寸
        result = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        return result
    except Exception as e:
        logger.warning(f"随机裁剪缩放失败: {str(e)}")
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _perspective_transform_random(image, intensity):
    """随机透视变换"""
    try:
        h, w = image.shape[:2]

        # 计算四个角点的偏移
        max_offset = min(h, w) * intensity

        # 原始角点
        src_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])

        # 随机偏移
        dst_points = np.float32([
            [np.random.uniform(-max_offset, max_offset), np.random.uniform(-max_offset, max_offset)],
            [w + np.random.uniform(-max_offset, max_offset), np.random.uniform(-max_offset, max_offset)],
            [w + np.random.uniform(-max_offset, max_offset), h + np.random.uniform(-max_offset, max_offset)],
            [np.random.uniform(-max_offset, max_offset), h + np.random.uniform(-max_offset, max_offset)]
        ])

        # 确保点在图像内
        dst_points = np.clip(dst_points, 0, [w, h])

        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # 应用变换
        result = cv2.warpPerspective(image, matrix, (w, h))
        return result
    except Exception as e:
        logger.warning(f"透视变换失败: {str(e)}")
        return image


def _shear_transform(image, angle_x, angle_y):
    """剪切变换"""
    try:
        h, w = image.shape[:2]

        # 转换角度为弧度
        tan_x = np.tan(np.radians(angle_x))
        tan_y = np.tan(np.radians(angle_y))

        # 构建剪切矩阵
        shear_matrix = np.float32([
            [1, tan_x, 0],
            [tan_y, 1, 0]
        ])

        # 计算新图像大小
        new_w = int(w + h * abs(tan_x))
        new_h = int(h + w * abs(tan_y))

        # 调整矩阵以保持图像居中
        shear_matrix[0, 2] = (new_w - w) / 2 if tan_x > 0 else 0
        shear_matrix[1, 2] = (new_h - h) / 2 if tan_y > 0 else 0

        # 应用变换
        result = cv2.warpAffine(image, shear_matrix, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0))
        return result
    except Exception as e:
        logger.warning(f"剪切变换失败: {str(e)}")
        return image


def _hsv_color_adjust(image, h_shift, s_shift, v_shift):
    """HSV颜色空间调整"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)

        # 调整色调（环绕处理）
        h = h.astype(np.int16)
        h = (h + int(h_shift / 2)) % 180  # OpenCV H范围0-179
        h = h.astype(np.uint8)

        # 调整饱和度
        s = s.astype(np.float32)
        s = np.clip(s * (1 + s_shift), 0, 255).astype(np.uint8)

        # 调明明度
        v = v.astype(np.float32)
        v = np.clip(v * (1 + v_shift), 0, 255).astype(np.uint8)

        # 合并并转换回RGB
        hsv_shifted = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_shifted, cv2.COLOR_HSV2RGB)
    except Exception as e:
        logger.warning(f"HSV颜色调整失败: {str(e)}")
        return image


def _rgb_shift_adjust(image, r_shift, g_shift, b_shift):
    """RGB通道偏移"""
    try:
        result = image.copy()
        result[:, :, 0] = np.clip(image[:, :, 0] + r_shift, 0, 255)
        result[:, :, 1] = np.clip(image[:, :, 1] + g_shift, 0, 255)
        result[:, :, 2] = np.clip(image[:, :, 2] + b_shift, 0, 255)
        return result
    except Exception as e:
        logger.warning(f"RGB通道偏移失败: {str(e)}")
        return image


def _to_gray_convert(image):
    """转灰度"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 转回三通道
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    except Exception as e:
        logger.warning(f"转灰度失败: {str(e)}")
        return image


def _gaussian_blur_with_kernel(image, kernel_size, sigma):
    """使用指定核大小的高斯模糊"""
    try:
        if kernel_size <= 1:
            return image
        # 确保是奇数
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(3, min(kernel_size, 51))
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    except Exception as e:
        logger.warning(f"高斯模糊失败: {str(e)}")
        return image
