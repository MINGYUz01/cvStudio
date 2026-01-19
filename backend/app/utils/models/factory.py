"""
模型工厂
根据配置创建PyTorch模型
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class ModelFactory:
    """
    模型工厂类
    负责根据配置创建对应的PyTorch模型
    """

    # 支持的预训练模型列表
    PRETRAINED_MODELS = {
        # ResNet系列
        "resnet18": "models.resnet18",
        "resnet34": "models.resnet34",
        "resnet50": "models.resnet50",
        "resnet101": "models.resnet101",
        "resnet152": "models.resnet152",
        # EfficientNet系列
        "efficientnet_b0": "models.efficientnet_b0",
        "efficientnet_b1": "models.efficientnet_b1",
        "efficientnet_b2": "models.efficientnet_b2",
        "efficientnet_b3": "models.efficientnet_b3",
        # Vision Transformer
        "vit_b_16": "models.vit_b_16",
        "vit_b_32": "models.vit_b_32",
        "vit_l_16": "models.vit_l_16",
        # ConvNeXt
        "convnext_tiny": "models.convnext_tiny",
        "convnext_small": "models.convnext_small",
        "convnext_base": "models.convnext_base",
        # MobileNet
        "mobilenet_v3_small": "models.mobilenet_v3_small",
        "mobilenet_v3_large": "models.mobilenet_v3_large",
        # ShuffleNet
        "shufflenet_v2_x1_0": "models.shufflenet_v2_x1_0",
        "shufflenet_v2_x2_0": "models.shufflenet_v2_x2_0",
        # DenseNet
        "densenet121": "models.densenet121",
        "densenet169": "models.densenet169",
        "densenet201": "models.densenet201",
    }

    @staticmethod
    def create(config: Dict) -> nn.Module:
        """
        根据配置创建模型

        Args:
            config: 配置字典，包含以下字段:
                - task_type: 任务类型 ("classification" | "detection")
                - architecture: 模型架构名称或配置字典
                - num_classes: 输出类别数
                - input_channels: 输入通道数（默认3）
                - pretrained: 是否使用预训练权重（默认False）
                - dropout: Dropout概率（默认0.0）

        Returns:
            PyTorch模型实例

        Raises:
            ValueError: 当任务类型或架构不支持时
        """
        task_type = config.get("task_type", "classification")

        if task_type == "classification":
            return ModelFactory._create_classification_model(config)
        elif task_type == "detection":
            return ModelFactory._create_detection_model(config)
        else:
            raise ValueError(f"不支持的任务类型: {task_type}")

    @staticmethod
    def _create_classification_model(config: Dict) -> nn.Module:
        """
        创建分类模型

        Args:
            config: 配置字典

        Returns:
            分类模型实例
        """
        from app.utils.models.classification import ClassificationModel

        architecture = config.get("architecture", "resnet18")
        num_classes = config.get("num_classes", 10)
        input_channels = config.get("input_channels", 3)
        pretrained = config.get("pretrained", False)
        dropout = config.get("dropout", 0.0)

        # 如果是用户自定义架构（来自代码生成器）
        if isinstance(architecture, dict):
            return ModelFactory._load_generated_model(architecture, num_classes)

        # 使用预设模型
        return ClassificationModel(
            architecture=architecture,
            num_classes=num_classes,
            input_channels=input_channels,
            pretrained=pretrained,
            dropout=dropout
        )

    @staticmethod
    def _create_detection_model(config: Dict) -> nn.Module:
        """
        创建检测模型

        Args:
            config: 配置字典

        Returns:
            检测模型实例
        """
        from app.utils.models.detection import DetectionModel

        architecture = config.get("architecture", "yolov5n")
        num_classes = config.get("num_classes", 80)
        input_channels = config.get("input_channels", 3)

        return DetectionModel(
            architecture=architecture,
            num_classes=num_classes,
            input_channels=input_channels
        )

    @staticmethod
    def _load_generated_model(architecture_config: Dict, num_classes: int) -> nn.Module:
        """
        动态加载代码生成器生成的模型

        Args:
            architecture_config: 架构配置（包含code_path字段）
            num_classes: 类别数

        Returns:
            生成的模型实例
        """
        # 如果有代码路径，动态导入
        code_path = architecture_config.get("code_path")
        if code_path:
            try:
                import importlib.util
                import sys
                from pathlib import Path

                # 检查文件是否存在
                path = Path(code_path)
                if not path.exists():
                    raise ValueError(f"模型代码文件不存在: {code_path}")

                # 使用唯一模块名避免冲突
                module_name = f"generated_model_{id(architecture_config)}"
                spec = importlib.util.spec_from_file_location(module_name, code_path)
                if spec is None or spec.loader is None:
                    raise ValueError(f"无法加载模块: {code_path}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

                # 获取模型类 - 兼容 "model_class_name" 和 "class_name" 两种字段名
                model_class_name = architecture_config.get("model_class_name") or architecture_config.get("class_name", "GeneratedModel")

                if not hasattr(module, model_class_name):
                    # 尝试查找模块中的第一个 nn.Module 类
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
                            model_class_name = attr_name
                            break
                    else:
                        raise ValueError(f"模块中未找到模型类: {model_class_name}")

                model_class = getattr(module, model_class_name)

                # 尝试不同的方式实例化模型
                try:
                    # 1. 首先尝试传递 num_classes 参数
                    return model_class(num_classes=num_classes)
                except TypeError:
                    try:
                        # 2. 如果不接受参数，尝试无参数实例化
                        return model_class()
                    except Exception as e:
                        raise ValueError(
                            f"无法实例化模型类 {model_class_name}: "
                            f"尝试了 with(num_classes={num_classes}) 和 without() 都失败。"
                            f"请确保生成的模型 __init__ 接受 num_classes 参数或不接受任何参数。"
                        )
            except Exception as e:
                raise ValueError(f"加载生成的模型失败: {e}")

        # 如果有代码字符串，动态执行
        code = architecture_config.get("code")
        if code:
            try:
                namespace = {}
                exec(code, namespace)

                # 查找模型类
                for name, obj in namespace.items():
                    if isinstance(obj, type) and issubclass(obj, nn.Module):
                        return obj(num_classes=num_classes)

                raise ValueError("代码中未找到有效的模型类")
            except Exception as e:
                raise ValueError(f"执行生成的代码失败: {e}")

        raise ValueError("无法加载生成的模型：缺少代码路径或代码字符串")

    @staticmethod
    def list_pretrained_models() -> list:
        """
        获取所有支持的预训练模型列表

        Returns:
            模型名称列表
        """
        return sorted(ModelFactory.PRETRAINED_MODELS.keys())

    @staticmethod
    def is_pretrained_supported(model_name: str) -> bool:
        """
        检查模型是否支持预训练权重

        Args:
            model_name: 模型名称

        Returns:
            是否支持预训练权重
        """
        return model_name in ModelFactory.PRETRAINED_MODELS

    @staticmethod
    def get_model_info(model_name: str) -> Optional[Dict]:
        """
        获取模型信息

        Args:
            model_name: 模型名称

        Returns:
            模型信息字典或None
        """
        if model_name not in ModelFactory.PRETRAINED_MODELS:
            return None

        # 根据模型类型返回不同信息
        if model_name.startswith("resnet"):
            return {
                "name": model_name,
                "family": "ResNet",
                "params": {
                    "resnet18": 11.7,
                    "resnet34": 21.8,
                    "resnet50": 25.6,
                    "resnet101": 44.5,
                    "resnet152": 60.2,
                }.get(model_name, 0),
                "input_size": 224,
            }
        elif model_name.startswith("efficientnet"):
            return {
                "name": model_name,
                "family": "EfficientNet",
                "params": {
                    "efficientnet_b0": 5.3,
                    "efficientnet_b1": 7.8,
                    "efficientnet_b2": 9.1,
                    "efficientnet_b3": 12.2,
                }.get(model_name, 0),
                "input_size": 224,
            }
        elif model_name.startswith("vit"):
            return {
                "name": model_name,
                "family": "Vision Transformer",
                "params": {
                    "vit_b_16": 86.6,
                    "vit_b_32": 88.2,
                    "vit_l_16": 307.5,
                }.get(model_name, 0),
                "input_size": 224,
            }
        elif model_name.startswith("mobilenet"):
            return {
                "name": model_name,
                "family": "MobileNet",
                "params": {
                    "mobilenet_v3_small": 2.5,
                    "mobilenet_v3_large": 5.5,
                }.get(model_name, 0),
                "input_size": 224,
            }

        return {
            "name": model_name,
            "family": "Unknown",
            "params": 0,
            "input_size": 224,
        }


def create_model(
    task_type: str,
    architecture: str = "resnet18",
    num_classes: int = 10,
    **kwargs
) -> nn.Module:
    """
    便捷函数：创建模型

    Args:
        task_type: 任务类型
        architecture: 模型架构
        num_classes: 类别数
        **kwargs: 其他参数

    Returns:
        模型实例
    """
    config = {
        "task_type": task_type,
        "architecture": architecture,
        "num_classes": num_classes,
        **kwargs
    }
    return ModelFactory.create(config)
