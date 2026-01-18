"""
图像分类模型
支持多种预训练模型架构
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


class ClassificationModel(nn.Module):
    """
    图像分类模型

    支持多种预训练模型架构，包括：
    - ResNet系列 (18, 34, 50, 101, 152)
    - EfficientNet系列 (b0, b1, b2, b3)
    - Vision Transformer (vit_b_16, vit_b_32, vit_l_16)
    - ConvNeXt (tiny, small, base)
    - MobileNet v3 (small, large)
    - ShuffleNet v2 (x1_0, x2_0)
    - DenseNet (121, 169, 201)
    """

    # 模型创建映射
    MODEL_CREATORS = {
        # ResNet
        "resnet18": lambda weights: models.resnet18(weights=weights),
        "resnet34": lambda weights: models.resnet34(weights=weights),
        "resnet50": lambda weights: models.resnet50(weights=weights),
        "resnet101": lambda weights: models.resnet101(weights=weights),
        "resnet152": lambda weights: models.resnet152(weights=weights),
        # EfficientNet
        "efficientnet_b0": lambda weights: models.efficientnet_b0(weights=weights),
        "efficientnet_b1": lambda weights: models.efficientnet_b1(weights=weights),
        "efficientnet_b2": lambda weights: models.efficientnet_b2(weights=weights),
        "efficientnet_b3": lambda weights: models.efficientnet_b3(weights=weights),
        # Vision Transformer
        "vit_b_16": lambda weights: models.vit_b_16(weights=weights),
        "vit_b_32": lambda weights: models.vit_b_32(weights=weights),
        "vit_l_16": lambda weights: models.vit_l_16(weights=weights),
        # ConvNeXt
        "convnext_tiny": lambda weights: models.convnext_tiny(weights=weights),
        "convnext_small": lambda weights: models.convnext_small(weights=weights),
        "convnext_base": lambda weights: models.convnext_base(weights=weights),
        # MobileNet
        "mobilenet_v3_small": lambda weights: models.mobilenet_v3_small(weights=weights),
        "mobilenet_v3_large": lambda weights: models.mobilenet_v3_large(weights=weights),
        # ShuffleNet
        "shufflenet_v2_x1_0": lambda weights: models.shufflenet_v2_x1_0(weights=weights),
        "shufflenet_v2_x2_0": lambda weights: models.shufflenet_v2_x2_0(weights=weights),
        # DenseNet
        "densenet121": lambda weights: models.densenet121(weights=weights),
        "densenet169": lambda weights: models.densenet169(weights=weights),
        "densenet201": lambda weights: models.densenet201(weights=weights),
    }

    def __init__(
        self,
        architecture: str = "resnet18",
        num_classes: int = 10,
        input_channels: int = 3,
        pretrained: bool = False,
        dropout: float = 0.0
    ):
        """
        初始化分类模型

        Args:
            architecture: 模型架构名称
            num_classes: 输出类别数
            input_channels: 输入通道数（默认3，RGB）
            pretrained: 是否使用预训练权重
            dropout: Dropout概率
        """
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.input_channels = input_channels

        # 加载基础模型
        if architecture not in self.MODEL_CREATORS:
            raise ValueError(
                f"不支持的模型架构: {architecture}。"
                f"支持的模型: {list(self.MODEL_CREATORS.keys())}"
            )

        # 设置预训练权重
        weights = "DEFAULT" if pretrained else None

        # 创建基础模型
        base_model = self.MODEL_CREATORS[architecture](weights)

        # 修改模型以适应输入通道数和类别数
        self.model = self._modify_model(
            base_model,
            architecture,
            num_classes,
            input_channels,
            dropout
        )

    def _modify_model(
        self,
        base_model: nn.Module,
        architecture: str,
        num_classes: int,
        input_channels: int,
        dropout: float
    ) -> nn.Module:
        """
        修改基础模型以适应自定义输入和输出

        Args:
            base_model: 基础模型
            architecture: 架构名称
            num_classes: 输出类别数
            input_channels: 输入通道数
            dropout: Dropout概率

        Returns:
            修改后的模型
        """
        # 处理不同的模型架构
        if architecture.startswith("resnet"):
            return self._modify_resnet(base_model, num_classes, input_channels, dropout)
        elif architecture.startswith("efficientnet"):
            return self._modify_efficientnet(base_model, num_classes, input_channels, dropout)
        elif architecture.startswith("vit"):
            return self._modify_vit(base_model, num_classes, input_channels)
        elif architecture.startswith("convnext"):
            return self._modify_convnext(base_model, num_classes, input_channels)
        elif architecture.startswith("mobilenet"):
            return self._modify_mobilenet(base_model, num_classes, input_channels, dropout)
        elif architecture.startswith("shufflenet"):
            return self._modify_shufflenet(base_model, num_classes, input_channels)
        elif architecture.startswith("densenet"):
            return self._modify_densenet(base_model, num_classes, input_channels)
        else:
            # 默认处理：尝试修改最后的分类层
            return self._modify_generic(base_model, num_classes, input_channels, dropout)

    def _modify_resnet(
        self,
        model: nn.Module,
        num_classes: int,
        input_channels: int,
        dropout: float
    ) -> nn.Module:
        """修改ResNet模型"""
        # 修改第一层卷积（如果输入通道不是3）
        if input_channels != 3:
            original_conv1 = model.conv1
            model.conv1 = nn.Conv2d(
                input_channels,
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias
            )
            # 复制权重（只复制前input_channels个通道）
            with torch.no_grad():
                model.conv1.weight[:, :3] = original_conv1.weight
                # 对于额外的通道，使用平均值
                if input_channels > 3:
                    for i in range(3, input_channels):
                        model.conv1.weight[:, i] = model.conv1.weight[:, 0]

        # 修改最后的全连接层
        in_features = model.fc.in_features
        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.fc = nn.Linear(in_features, num_classes)

        return model

    def _modify_efficientnet(
        self,
        model: nn.Module,
        num_classes: int,
        input_channels: int,
        dropout: float
    ) -> nn.Module:
        """修改EfficientNet模型"""
        # EfficientNet的输入通道修改比较复杂，这里只修改分类器
        in_features = model.classifier[1].in_features

        if dropout > 0:
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        else:
            model.classifier = nn.Linear(in_features, num_classes)

        return model

    def _modify_vit(
        self,
        model: nn.Module,
        num_classes: int,
        input_channels: int
    ) -> nn.Module:
        """修改Vision Transformer模型"""
        # ViT的输入通道修改需要修改patch embedding
        if input_channels != 3:
            model.conv_proj = nn.Conv2d(
                input_channels,
                model.conv_proj.out_channels,
                kernel_size=model.conv_proj.kernel_size,
                stride=model.conv_proj.stride,
                padding=model.conv_proj.padding
            )

        # 修改分类头
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

        return model

    def _modify_convnext(
        self,
        model: nn.Module,
        num_classes: int,
        input_channels: int
    ) -> nn.Module:
        """修改ConvNeXt模型"""
        # 修改分类器
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes
        )
        return model

    def _modify_mobilenet(
        self,
        model: nn.Module,
        num_classes: int,
        input_channels: int,
        dropout: float
    ) -> nn.Module:
        """修改MobileNet模型"""
        # 修改分类器
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            num_classes
        )
        return model

    def _modify_shufflenet(
        self,
        model: nn.Module,
        num_classes: int,
        input_channels: int
    ) -> nn.Module:
        """修改ShuffleNet模型"""
        # 修改分类器
        model.fc = nn.Linear(
            model.fc.in_features,
            num_classes
        )
        return model

    def _modify_densenet(
        self,
        model: nn.Module,
        num_classes: int,
        input_channels: int
    ) -> nn.Module:
        """修改DenseNet模型"""
        # 修改分类器
        model.classifier = nn.Linear(
            model.classifier.in_features,
            num_classes
        )
        return model

    def _modify_generic(
        self,
        model: nn.Module,
        num_classes: int,
        input_channels: int,
        dropout: float
    ) -> nn.Module:
        """通用模型修改方法"""
        # 尝试找到分类层
        classifier = None
        for name in ['fc', 'classifier', 'head']:
            if hasattr(model, name):
                classifier = getattr(model, name)
                break

        if classifier is not None:
            in_features = classifier.in_features if hasattr(classifier, 'in_features') else None
            if in_features:
                if dropout > 0:
                    new_classifier = nn.Sequential(
                        nn.Dropout(p=dropout),
                        nn.Linear(in_features, num_classes)
                    )
                else:
                    new_classifier = nn.Linear(in_features, num_classes)

                # 找到并替换分类层
                for name in ['fc', 'classifier', 'head']:
                    if hasattr(model, name):
                        setattr(model, name, new_classifier)
                        break

        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch, channels, height, width]

        Returns:
            输出张量 [batch, num_classes]
        """
        return self.model(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取特征表示（在分类层之前）

        Args:
            x: 输入张量

        Returns:
            特征张量
        """
        # 这需要根据具体模型实现
        # 简化版本：直接返回模型输出
        return self.forward(x)


class SimpleCNN(nn.Module):
    """
    简单的CNN分类模型
    用于快速测试和演示
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 3,
        dropout: float = 0.5
    ):
        """
        初始化简单CNN

        Args:
            num_classes: 类别数
            input_channels: 输入通道数
            dropout: Dropout概率
        """
        super().__init__()

        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),  # 假设输入是224x224
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch, channels, height, width]

        Returns:
            输出张量 [batch, num_classes]
        """
        # 特征提取
        x = self.features(x)

        # 展平
        x = torch.flatten(x, 1)

        # 分类
        x = self.classifier(x)

        return x


def create_simple_cnn(num_classes: int = 10, input_channels: int = 3) -> SimpleCNN:
    """
    创建简单CNN模型

    Args:
        num_classes: 类别数
        input_channels: 输入通道数

    Returns:
        SimpleCNN模型实例
    """
    return SimpleCNN(num_classes=num_classes, input_channels=input_channels)
