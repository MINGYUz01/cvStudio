"""
数据增强算子注册表
定义所有可用的数据增强算子及其元数据
"""

from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum


class ParamType(str, Enum):
    """参数类型枚举"""
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    RANGE = "range"
    SELECT = "select"


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
        params=[]
    ),

    AugmentationOperator(
        id="vertical_flip",
        name_zh="垂直翻转",
        name_en="Vertical Flip",
        category="geometric",
        description="沿水平轴镜像翻转图像，常用于数据增强以增加数据多样性",
        params=[]
    ),

    AugmentationOperator(
        id="rotate",
        name_zh="旋转",
        name_en="Rotate",
        category="geometric",
        description="按指定角度旋转图像，支持任意角度旋转",
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
                name="angle",
                label_zh="旋转角度",
                label_en="Angle",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=-180.0,
                max_value=180.0,
                step=1.0,
                description="旋转角度（度），正值为逆时针旋转"
            )
        ]
    ),

    AugmentationOperator(
        id="scale",
        name_zh="缩放",
        name_en="Scale",
        category="geometric",
        description="按比例缩放图像尺寸",
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
                name="factor",
                label_zh="缩放因子",
                label_en="Scale Factor",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.5,
                max_value=1.5,
                step=0.05,
                description="缩放比例，1.0为原始大小，大于1放大，小于1缩小"
            )
        ]
    ),

    AugmentationOperator(
        id="crop",
        name_zh="裁剪",
        name_en="Crop",
        category="geometric",
        description="裁剪图像指定区域",
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
        id="translate",
        name_zh="平移",
        name_en="Translate",
        category="geometric",
        description="在水平和垂直方向平移图像",
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
                name="dx",
                label_zh="水平偏移",
                label_en="Delta X",
                param_type=ParamType.INTEGER,
                default=0,
                min_value=-200,
                max_value=200,
                step=10,
                description="水平方向偏移量（像素），正值为向右"
            ),
            AugmentationParam(
                name="dy",
                label_zh="垂直偏移",
                label_en="Delta Y",
                param_type=ParamType.INTEGER,
                default=0,
                min_value=-200,
                max_value=200,
                step=10,
                description="垂直方向偏移量（像素），正值为向下"
            ),
            AugmentationParam(
                name="fill_value",
                label_zh="填充值",
                label_en="Fill Value",
                param_type=ParamType.INTEGER,
                default=255,
                min_value=0,
                max_value=255,
                description="平移后空白区域的填充值（0-255）"
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
                default=0.5,
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
    AugmentationOperator(
        id="brightness",
        name_zh="亮度调整",
        name_en="Brightness",
        category="color",
        description="调整图像明暗程度",
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
                name="factor",
                label_zh="亮度因子",
                label_en="Factor",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.5,
                max_value=1.5,
                step=0.05,
                description="亮度调整因子，1.0为原始亮度，大于1变亮，小于1变暗"
            )
        ]
    ),

    AugmentationOperator(
        id="contrast",
        name_zh="对比度调整",
        name_en="Contrast",
        category="color",
        description="调整图像对比度",
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
                name="factor",
                label_zh="对比度因子",
                label_en="Factor",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.5,
                max_value=1.5,
                step=0.05,
                description="对比度调整因子，1.0为原始对比度，大于1增强，小于1减弱"
            )
        ]
    ),

    AugmentationOperator(
        id="saturation",
        name_zh="饱和度调整",
        name_en="Saturation",
        category="color",
        description="调整颜色鲜艳程度",
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
                name="factor",
                label_zh="饱和度因子",
                label_en="Factor",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.5,
                max_value=1.5,
                step=0.05,
                description="饱和度调整因子，1.0为原始饱和度，大于1更鲜艳，小于1更灰暗"
            )
        ]
    ),

    AugmentationOperator(
        id="hue_shift",
        name_zh="色调偏移",
        name_en="Hue Shift",
        category="color",
        description="调整颜色色调",
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
                name="shift",
                label_zh="色调偏移",
                label_en="Shift",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=-30.0,
                max_value=30.0,
                step=5.0,
                description="色调偏移量（度），正值为顺时针偏移"
            )
        ]
    ),

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
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="gamma",
                label_zh="Gamma值",
                label_en="Gamma",
                param_type=ParamType.FLOAT,
                default=1.0,
                min_value=0.5,
                max_value=1.5,
                step=0.05,
                description="Gamma值，1.0为无变化，大于1变暗，小于1变亮"
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
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="cutoff",
                label_zh="截断百分比",
                label_en="Cutoff",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=0.0,
                max_value=50.0,
                step=1.0,
                description="直方图截断百分比，用于忽略极值像素"
            )
        ]
    ),

    # ==================== 模糊与噪声类 ====================
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
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="sigma",
                label_zh="模糊强度",
                label_en="Sigma",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=0.0,
                max_value=5.0,
                step=0.1,
                description="高斯核标准差，值越大模糊效果越明显"
            )
        ]
    ),

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
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="kernel_size",
                label_zh="核大小",
                label_en="Kernel Size",
                param_type=ParamType.INTEGER,
                default=15,
                min_value=3,
                max_value=31,
                step=2,
                description="运动模糊核大小（必须是奇数）"
            ),
            AugmentationParam(
                name="angle",
                label_zh="运动角度",
                label_en="Angle",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=0.0,
                max_value=360.0,
                step=15.0,
                description="运动方向角度（度）"
            )
        ]
    ),

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
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="std",
                label_zh="噪声标准差",
                label_en="Std Dev",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=0.0,
                max_value=25.0,
                step=0.5,
                description="噪声标准差，值越大噪声越明显"
            )
        ]
    ),

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
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.1,
                description="执行此增强的概率"
            ),
            AugmentationParam(
                name="amount",
                label_zh="噪声密度",
                label_en="Amount",
                param_type=ParamType.FLOAT,
                default=0.0,
                min_value=0.0,
                max_value=0.1,
                step=0.001,
                description="噪声密度（0-1），即被噪声替换的像素比例"
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
                param_type=ParamType.RANGE,
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
                param_type=ParamType.RANGE,
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
                name="quality",
                label_zh="压缩质量",
                label_en="Quality",
                param_type=ParamType.INTEGER,
                default=85,
                min_value=10,
                max_value=100,
                step=5,
                description="JPEG压缩质量（1-100），值越小压缩越明显"
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
                param_type=ParamType.RANGE,
                default=(0.5, 1.5),
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
                "default": p.default,
                "min_value": p.min_value,
                "max_value": p.max_value,
                "step": p.step,
                "options": p.options,
                "description": p.description
            }
            for p in operator.params
        ]
    }


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
