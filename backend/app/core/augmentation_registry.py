"""
数据增强算子注册表
定义所有可用的数据增强算子及其元数据

参考YOLO和albumentations的最佳实践进行参数设计
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class ParamType(str, Enum):
    """参数类型枚举"""
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    RANGE = "range"           # 现有：范围类型，使用元组表示最小值和最大值
    SELECT = "select"
    FLOAT_RANGE = "float_range"  # 新增：浮点数范围，训练时随机采样
    INT_RANGE = "int_range"      # 新增：整数范围


@dataclass
class AugmentationParam:
    """增强算子参数定义"""
    name: str  # 参数名称（英文，用于代码）
    label_zh: str  # 参数中文名称
    label_en: str  # 参数英文名称
    param_type: ParamType  # 参数类型
    default: Any  # 默认值
    min_value: Optional[float] = None  # 最小值
    max_value: Optional[float] = None  # 最大值
    step: Optional[float] = None  # 步长
    options: Optional[List[Dict[str, Any]]] = None  # 选项列表（用于select类型）
    description: str = ""  # 参数描述


@dataclass
class AugmentationOperator:
    """增强算子定义"""
    id: str  # 算子唯一标识（英文，用于代码）
    name_zh: str  # 算子中文名称
    name_en: str  # 算子英文名称
    category: str  # 分类
    description: str  # 功能描述
    params: List[AugmentationParam] = field(default_factory=list)  # 参数列表
    enabled: bool = True  # 是否启用


# 算子分类（中文）
CATEGORY_LABELS_ZH = {
    "geometric": "几何变换",
    "color": "颜色变换",
    "blur_noise": "模糊与噪声",
    "other": "其他"
}

# 算子分类（英文）
CATEGORY_LABELS_EN = {
    "geometric": "Geometric",
    "color": "Color",
    "blur_noise": "Blur & Noise",
    "other": "Other"
}


# 所有支持的增强算子定义
AUGMENTATION_OPERATORS: List[AugmentationOperator] = [
    # ==================== 几何变换类 ====================
    AugmentationOperator(
        id="horizontal_flip",
        name_zh="水平翻转",
        name_en="Horizontal Flip",
        category="geometric",
        description="沿垂直轴镜像翻转图像，常用于数据增强以增加数据多样性",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            )
        ]
    ),

    AugmentationOperator(
        id="vertical_flip",
        name_zh="垂直翻转",
        name_en="Vertical Flip",
        category="geometric",
        description="沿水平轴镜像翻转图像，常用于数据增强以增加数据多样性",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率（分类任务可设为0.1，检测任务一般不使用）"
            )
        ]
    ),

    # 旋转算子 - 重构为角度范围
    AugmentationOperator(
        id="rotate",
        name_zh="随机旋转",
        name_en="Random Rotate",
        category="geometric",
        description="在指定角度范围内随机旋转图像（推荐：分类±30°，检测±10°）",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="angle_limit",
                label_zh="旋转角度范围",
                label_en="Angle Range",
                param_type=ParamType.FLOAT_RANGE,
                default=(-15.0, 15.0),
                min_value=-180.0,
                max_value=180.0,
                step=1.0,
                description="旋转角度范围（度），训练时在此范围内随机采样"
            )
        ]
    ),

    # 缩放算子 - 重构为缩放因子范围
    AugmentationOperator(
        id="scale",
        name_zh="随机缩放",
        name_en="Random Scale",
        category="geometric",
        description="在指定范围内随机缩放图像",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="scale_limit",
                label_zh="缩放因子范围",
                label_en="Scale Range",
                param_type=ParamType.FLOAT_RANGE,
                default=(0.8, 1.2),
                min_value=0.5,
                max_value=2.0,
                step=0.05,
                description="缩放因子范围，1.0为原始大小，训练时在此范围内随机采样"
            )
        ]
    ),

    # 平移算子 - 重构为相对比例
    AugmentationOperator(
        id="translate",
        name_zh="随机平移",
        name_en="Random Translate",
        category="geometric",
        description="在水平和垂直方向按比例随机平移图像",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="translate_limit",
                label_zh="平移比例范围",
                label_en="Translate Range",
                param_type=ParamType.FLOAT_RANGE,
                default=(-0.1, 0.1),
                min_value=-0.5,
                max_value=0.5,
                step=0.05,
                description="平移比例范围（相对于图像尺寸），正值为向右/下"
            ),
            AugmentationParam(
                name="fill_value",
                label_zh="填充值",
                label_en="Fill Value",
                param_type=ParamType.INTEGER,
                default=0,
                min_value=0,
                max_value=255,
                description="平移后空白区域的填充值（0-255），0为黑色"
            )
        ]
    ),

    # 综合几何变换算子（新增）
    AugmentationOperator(
        id="shift_scale_rotate",
        name_zh="综合几何变换",
        name_en="Shift Scale Rotate",
        category="geometric",
        description="同时进行平移、缩放和旋转的综合性几何变换",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="shift_limit",
                label_zh="平移范围",
                label_en="Shift Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-0.1, 0.1),
                min_value=-0.3,
                max_value=0.3,
                step=0.05,
                description="平移比例范围（相对于图像尺寸）"
            ),
            AugmentationParam(
                name="scale_limit",
                label_zh="缩放范围",
                label_en="Scale Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-0.1, 0.1),
                min_value=-0.3,
                max_value=0.3,
                step=0.05,
                description="缩放比例范围，例如-0.1到0.1表示0.9到1.1倍"
            ),
            AugmentationParam(
                name="rotate_limit",
                label_zh="旋转角度范围",
                label_en="Rotate Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-15.0, 15.0),
                min_value=-45.0,
                max_value=45.0,
                step=5.0,
                description="旋转角度范围（度）"
            ),
            AugmentationParam(
                name="border_mode",
                label_zh="边界模式",
                label_en="Border Mode",
                param_type=ParamType.SELECT,
                default="reflect",
                options=[
                    {"value": "reflect", "label": "反射填充"},
                    {"value": "constant", "label": "常数填充"},
                    {"value": "wrap", "label": "循环填充"}
                ],
                description="边界填充模式"
            )
        ]
    ),

    # 随机裁剪算子（新增）
    AugmentationOperator(
        id="random_resized_crop",
        name_zh="随机裁剪缩放",
        name_en="Random Resized Crop",
        category="geometric",
        description="随机裁剪图像区域并调整到指定大小，分类任务常用",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="height",
                label_zh="输出高度",
                label_en="Height",
                param_type=ParamType.INTEGER,
                default=224,
                min_value=32,
                max_value=1024,
                step=32,
                description="输出图像高度（像素）"
            ),
            AugmentationParam(
                name="width",
                label_zh="输出宽度",
                label_en="Width",
                param_type=ParamType.INTEGER,
                default=224,
                min_value=32,
                max_value=1024,
                step=32,
                description="输出图像宽度（像素）"
            ),
            AugmentationParam(
                name="scale",
                label_zh="裁剪区域比例",
                label_en="Scale",
                param_type=ParamType.FLOAT_RANGE,
                default=(0.8, 1.0),
                min_value=0.08,
                max_value=1.0,
                step=0.05,
                description="裁剪区域占原图的比例范围"
            ),
            AugmentationParam(
                name="ratio",
                label_zh="宽高比范围",
                label_en="Ratio",
                param_type=ParamType.FLOAT_RANGE,
                default=(0.75, 1.33),
                min_value=0.5,
                max_value=2.0,
                step=0.1,
                description="裁剪区域的宽高比范围"
            )
        ]
    ),

    # 透视变换算子（新增）
    AugmentationOperator(
        id="perspective",
        name_zh="透视变换",
        name_en="Perspective",
        category="geometric",
        description="模拟不同视角的透视变换效果",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="scale",
                label_zh="变换强度",
                label_en="Scale",
                param_type=ParamType.FLOAT_RANGE,
                default=(0.0, 0.05),
                min_value=0.0,
                max_value=0.2,
                step=0.01,
                description="透视变换强度范围，值越大效果越明显"
            )
        ]
    ),

    # 剪切变换算子（新增）
    AugmentationOperator(
        id="shear",
        name_zh="剪切变换",
        name_en="Shear",
        category="geometric",
        description="对图像应用剪切变换，模拟倾斜视角",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="shear_x",
                label_zh="X方向剪切角度",
                label_en="Shear X",
                param_type=ParamType.FLOAT_RANGE,
                default=(-10.0, 10.0),
                min_value=-45.0,
                max_value=45.0,
                step=5.0,
                description="水平方向剪切角度范围（度）"
            ),
            AugmentationParam(
                name="shear_y",
                label_zh="Y方向剪切角度",
                label_en="Shear Y",
                param_type=ParamType.FLOAT_RANGE,
                default=(-10.0, 10.0),
                min_value=-45.0,
                max_value=45.0,
                step=5.0,
                description="垂直方向剪切角度范围（度）"
            )
        ]
    ),

    # 保留原有的裁剪算子（固定裁剪）
    AugmentationOperator(
        id="crop",
        name_zh="固定裁剪",
        name_en="Crop",
        category="geometric",
        description="裁剪图像指定区域（固定位置和大小）",
        params=[
            AugmentationParam(
                name="x",
                label_zh="起始X坐标",
                label_en="X",
                param_type=ParamType.INTEGER,
                default=0,
                min_value=0,
                description="裁剪区域左上角X坐标"
            ),
            AugmentationParam(
                name="y",
                label_zh="起始Y坐标",
                label_en="Y",
                param_type=ParamType.INTEGER,
                default=0,
                min_value=0,
                description="裁剪区域左上角Y坐标"
            ),
            AugmentationParam(
                name="width",
                label_zh="裁剪宽度",
                label_en="Width",
                param_type=ParamType.INTEGER,
                default=100,
                min_value=1,
                description="裁剪区域宽度（像素）"
            ),
            AugmentationParam(
                name="height",
                label_zh="裁剪高度",
                label_en="Height",
                param_type=ParamType.INTEGER,
                default=100,
                min_value=1,
                description="裁剪区域高度（像素）"
            )
        ]
    ),

    AugmentationOperator(
        id="elastic_transform",
        name_zh="弹性变换",
        name_en="Elastic Transform",
        category="geometric",
        description="应用弹性形变效果，模拟自然界的形变",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="alpha",
                label_zh="形变强度",
                label_en="Alpha",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.1,
                max_value=10.0,
                step=0.5,
                description="形变强度，值越大形变越明显"
            ),
            AugmentationParam(
                name="sigma",
                label_zh="平滑度",
                label_en="Sigma",
                param_type=ParamType.FLOAT,
                default=50.0,
                min_value=10.0,
                max_value=100.0,
                step=5.0,
                description="高斯核的标准差，值越大形变越平滑"
            )
        ]
    ),

    # ==================== 颜色变换类 ====================
    # HSV颜色空间调整（新增）
    AugmentationOperator(
        id="hsv_color",
        name_zh="HSV颜色调整",
        name_en="HSV Color",
        category="color",
        description="联合调整HSV颜色空间的色调、饱和度和明度",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="h_limit",
                label_zh="色调偏移范围",
                label_en="Hue Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-20.0, 20.0),
                min_value=-180.0,
                max_value=180.0,
                step=5.0,
                description="色调偏移范围（度）"
            ),
            AugmentationParam(
                name="s_limit",
                label_zh="饱和度范围",
                label_en="Saturation Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-30.0, 30.0),
                min_value=-100.0,
                max_value=100.0,
                step=5.0,
                description="饱和度调整范围（百分比）"
            ),
            AugmentationParam(
                name="v_limit",
                label_zh="明度范围",
                label_en="Value Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-20.0, 20.0),
                min_value=-100.0,
                max_value=100.0,
                step=5.0,
                description="明度调整范围（百分比）"
            )
        ]
    ),

    # 亮度调整 - 重构为范围
    AugmentationOperator(
        id="brightness",
        name_zh="亮度调整",
        name_en="Brightness",
        category="color",
        description="随机调整图像明暗程度",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="limit",
                label_zh="亮度变化范围",
                label_en="Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-0.2, 0.2),
                min_value=-0.5,
                max_value=0.5,
                step=0.05,
                description="亮度相对变化范围，负值变暗，正值变亮"
            )
        ]
    ),

    # 对比度调整 - 重构为范围
    AugmentationOperator(
        id="contrast",
        name_zh="对比度调整",
        name_en="Contrast",
        category="color",
        description="随机调整图像对比度",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="limit",
                label_zh="对比度变化范围",
                label_en="Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-0.2, 0.2),
                min_value=-0.5,
                max_value=0.5,
                step=0.05,
                description="对比度相对变化范围，负值降低，正值增强"
            )
        ]
    ),

    # 饱和度调整 - 重构为范围
    AugmentationOperator(
        id="saturation",
        name_zh="饱和度调整",
        name_en="Saturation",
        category="color",
        description="随机调整颜色鲜艳程度",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="limit",
                label_zh="饱和度变化范围",
                label_en="Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-0.3, 0.3),
                min_value=-1.0,
                max_value=1.0,
                step=0.1,
                description="饱和度相对变化范围，负值变灰，正值更鲜艳"
            )
        ]
    ),

    # 色调偏移 - 重构为范围
    AugmentationOperator(
        id="hue_shift",
        name_zh="色调偏移",
        name_en="Hue Shift",
        category="color",
        description="随机调整颜色色调",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="shift_limit",
                label_zh="色调偏移范围",
                label_en="Shift Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-20.0, 20.0),
                min_value=-180.0,
                max_value=180.0,
                step=5.0,
                description="色调偏移范围（度）"
            )
        ]
    ),

    # RGB通道偏移（新增）
    AugmentationOperator(
        id="rgb_shift",
        name_zh="RGB通道偏移",
        name_en="RGB Shift",
        category="color",
        description="分别调整RGB三个通道的像素值",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="r_limit",
                label_zh="红色通道偏移",
                label_en="Red Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-20.0, 20.0),
                min_value=-100.0,
                max_value=100.0,
                step=5.0,
                description="红色通道的像素值偏移范围"
            ),
            AugmentationParam(
                name="g_limit",
                label_zh="绿色通道偏移",
                label_en="Green Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-20.0, 20.0),
                min_value=-100.0,
                max_value=100.0,
                step=5.0,
                description="绿色通道的像素值偏移范围"
            ),
            AugmentationParam(
                name="b_limit",
                label_zh="蓝色通道偏移",
                label_en="Blue Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(-20.0, 20.0),
                min_value=-100.0,
                max_value=100.0,
                step=5.0,
                description="蓝色通道的像素值偏移范围"
            )
        ]
    ),

    # 转灰度（新增）
    AugmentationOperator(
        id="to_gray",
        name_zh="转灰度",
        name_en="To Gray",
        category="color",
        description="将彩色图像转换为灰度图像",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.1,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                description="转换为灰度的概率（通常设置较低值）"
            )
        ]
    ),

    # Gamma校正 - 重构为范围
    AugmentationOperator(
        id="gamma_correction",
        name_zh="Gamma校正",
        name_en="Gamma Correction",
        category="color",
        description="非线性亮度调整，常用于显示器校正",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="gamma_limit",
                label_zh="Gamma值范围",
                label_en="Gamma Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(80.0, 120.0),
                min_value=50.0,
                max_value=200.0,
                step=5.0,
                description="Gamma值范围，100为无变化"
            )
        ]
    ),

    AugmentationOperator(
        id="auto_contrast",
        name_zh="自动对比度",
        name_en="Auto Contrast",
        category="color",
        description="自动进行直方图均衡化，增强图像对比度",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="cutoff",
                label_zh="截断百分比",
                label_en="Cutoff",
                param_type=ParamType.FLOAT_RANGE,
                default=(0.0, 20.0),
                min_value=0.0,
                max_value=50.0,
                step=1.0,
                description="直方图截断百分比范围，用于忽略极值像素"
            )
        ]
    ),

    # ==================== 模糊与噪声类 ====================
    # 高斯模糊 - 重构为核大小范围
    AugmentationOperator(
        id="gaussian_blur",
        name_zh="高斯模糊",
        name_en="Gaussian Blur",
        category="blur_noise",
        description="应用高斯模糊效果，常用于图像平滑处理",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="blur_limit",
                label_zh="模糊核大小范围",
                label_en="Blur Limit",
                param_type=ParamType.INT_RANGE,
                default=(3, 7),
                min_value=3,
                max_value=31,
                step=2,
                description="高斯核大小范围（奇数），值越大模糊越明显"
            )
        ]
    ),

    # 运动模糊 - 重构为角度范围
    AugmentationOperator(
        id="motion_blur",
        name_zh="运动模糊",
        name_en="Motion Blur",
        category="blur_noise",
        description="模拟运动产生的模糊效果",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="blur_limit",
                label_zh="模糊核大小范围",
                label_en="Blur Limit",
                param_type=ParamType.INT_RANGE,
                default=(3, 7),
                min_value=3,
                max_value=31,
                step=2,
                description="运动模糊核大小范围（奇数）"
            ),
            AugmentationParam(
                name="angle_range",
                label_zh="运动角度范围",
                label_en="Angle Range",
                param_type=ParamType.FLOAT_RANGE,
                default=(0.0, 360.0),
                min_value=0.0,
                max_value=360.0,
                step=15.0,
                description="运动方向角度范围（度）"
            )
        ]
    ),

    # 高斯噪声 - 重构为方差范围
    AugmentationOperator(
        id="gaussian_noise",
        name_zh="高斯噪声",
        name_en="Gaussian Noise",
        category="blur_noise",
        description="添加高斯分布的随机噪声",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="var_limit",
                label_zh="噪声方差范围",
                label_en="Variance Limit",
                param_type=ParamType.FLOAT_RANGE,
                default=(10.0, 50.0),
                min_value=0.0,
                max_value=100.0,
                step=5.0,
                description="噪声方差范围，值越大噪声越明显"
            )
        ]
    ),

    # 椒盐噪声 - 重构为密度范围
    AugmentationOperator(
        id="salt_pepper_noise",
        name_zh="椒盐噪声",
        name_en="Salt & Pepper Noise",
        category="blur_noise",
        description="添加随机黑白像素点，模拟传感器噪声",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.2,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="amount",
                label_zh="噪声密度范围",
                label_en="Amount Range",
                param_type=ParamType.FLOAT_RANGE,
                default=(0.001, 0.05),
                min_value=0.0,
                max_value=0.2,
                step=0.001,
                description="噪声密度范围（0-1），即被噪声替换的像素比例"
            )
        ]
    ),

    # ==================== 其他类 ====================
    AugmentationOperator(
        id="random_erase",
        name_zh="随机擦除",
        name_en="Random Erase",
        category="other",
        description="随机遮挡图像区域，常用于正则化防止过拟合",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行擦除操作的概率"
            ),
            AugmentationParam(
                name="scale",
                label_zh="擦除区域比例",
                label_en="Scale",
                param_type=ParamType.FLOAT_RANGE,
                default=(0.02, 0.33),
                min_value=0.01,
                max_value=0.5,
                step=0.01,
                description="擦除区域占图像的比例范围"
            ),
            AugmentationParam(
                name="ratio",
                label_zh="宽高比范围",
                label_en="Ratio",
                param_type=ParamType.FLOAT_RANGE,
                default=(0.3, 3.3),
                min_value=0.1,
                max_value=5.0,
                step=0.1,
                description="擦除区域的宽高比范围"
            ),
            AugmentationParam(
                name="value",
                label_zh="填充值",
                label_en="Value",
                param_type=ParamType.INTEGER,
                default=0,
                min_value=0,
                max_value=255,
                description="擦除区域的填充值（0-255），0为黑色，255为白色"
            )
        ]
    ),

    AugmentationOperator(
        id="jpeg_compression",
        name_zh="JPEG压缩",
        name_en="JPEG Compression",
        category="other",
        description="模拟JPEG有损压缩效果",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.3,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="quality_lower",
                label_zh="最低质量",
                label_en="Min Quality",
                param_type=ParamType.INTEGER,
                default=70,
                min_value=10,
                max_value=100,
                step=5,
                description="压缩质量下限（1-100）"
            ),
            AugmentationParam(
                name="quality_upper",
                label_zh="最高质量",
                label_en="Max Quality",
                param_type=ParamType.INTEGER,
                default=100,
                min_value=10,
                max_value=100,
                step=5,
                description="压缩质量上限（1-100），训练时在此范围内随机"
            )
        ]
    ),

    AugmentationOperator(
        id="mosaic",
        name_zh="马赛克增强",
        name_en="Mosaic",
        category="other",
        description="将4张图片拼接成1张，是YOLO系列常用的数据增强方法",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行马赛克增强的概率"
            ),
            AugmentationParam(
                name="scale",
                label_zh="缩放范围",
                label_en="Scale Range",
                param_type=ParamType.FLOAT_RANGE,
                default=(0.5, 1.5),
                min_value=0.3,
                max_value=2.0,
                step=0.1,
                description="每张子图的缩放比例范围"
            )
        ]
    ),

    AugmentationOperator(
        id="copy_paste",
        name_zh="Copy-Paste增强",
        name_en="Copy-Paste",
        category="other",
        description="从其他图像复制区域粘贴到当前图像，常用于实例分割/目标检测",
        params=[
            AugmentationParam(
                name="probability",
                label_zh="执行概率",
                label_en="Probability",
                param_type=ParamType.FLOAT,
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行copy-paste的概率"
            ),
            AugmentationParam(
                name="max_objects",
                label_zh="最大对象数",
                label_en="Max Objects",
                param_type=ParamType.INTEGER,
                default=5,
                min_value=1,
                max_value=20,
                step=1,
                description="最多粘贴的对象数量"
            )
        ]
    ),
]


def get_all_operators() -> List[AugmentationOperator]:
    """获取所有增强算子"""
    return AUGMENTATION_OPERATORS


def get_operator_by_id(operator_id: str) -> Optional[AugmentationOperator]:
    """根据ID获取算子"""
    for op in AUGMENTATION_OPERATORS:
        if op.id == operator_id:
            return op
    return None


def get_operators_by_category(category: str) -> List[AugmentationOperator]:
    """根据分类获取算子"""
    return [op for op in AUGMENTATION_OPERATORS if op.category == category]


def get_all_categories() -> List[str]:
    """获取所有分类"""
    return list(CATEGORY_LABELS_ZH.keys())


def operator_to_dict(operator: AugmentationOperator) -> Dict[str, Any]:
    """将算子转换为字典格式"""
    return {
        "id": operator.id,
        "name_zh": operator.name_zh,
        "name_en": operator.name_en,
        "category": operator.category,
        "category_label_zh": CATEGORY_LABELS_ZH.get(operator.category, operator.category),
        "category_label_en": CATEGORY_LABELS_EN.get(operator.category, operator.category),
        "description": operator.description,
        "enabled": operator.enabled,
        "params": [
            {
                "name": p.name,
                "label_zh": p.label_zh,
                "label_en": p.label_en,
                "type": p.param_type.value,
                "default": _serialize_default(p.default, p.param_type),
                "min_value": p.min_value,
                "max_value": p.max_value,
                "step": p.step,
                "options": p.options,
                "description": p.description
            }
            for p in operator.params
        ]
    }


def _serialize_default(value: Any, param_type: ParamType) -> Any:
    """序列化默认值，确保元组正确转换为列表"""
    if param_type in (ParamType.RANGE, ParamType.FLOAT_RANGE, ParamType.INT_RANGE):
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return [value[0], value[1]]
    return value


def get_operators_dict() -> Dict[str, Any]:
    """获取所有算子的字典格式"""
    categories = {}
    for op in AUGMENTATION_OPERATORS:
        cat = op.category
        if cat not in categories:
            categories[cat] = {
                "category": cat,
                "label_zh": CATEGORY_LABELS_ZH.get(cat, cat),
                "label_en": CATEGORY_LABELS_EN.get(cat, cat),
                "operators": []
            }
        categories[cat]["operators"].append(operator_to_dict(op))
    return categories
