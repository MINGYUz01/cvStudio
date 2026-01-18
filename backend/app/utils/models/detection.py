"""
目标检测模型
YOLO风格的检测模型实现
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionModel(nn.Module):
    """
    目标检测模型（YOLO风格）

    包含三个部分：
    - Backbone: 特征提取网络
    - Neck: 特征融合网络
    - Head: 检测头
    """

    def __init__(
        self,
        architecture: str = "yolov5n",
        num_classes: int = 80,
        input_channels: int = 3
    ):
        """
        初始化检测模型

        Args:
            architecture: 检测架构名称
            num_classes: 检测类别数
            input_channels: 输入通道数
        """
        super().__init__()

        self.architecture = architecture
        self.num_classes = num_classes
        self.input_channels = input_channels

        # 解析架构配置
        self.depth_multiple = self._get_depth_multiple(architecture)
        self.width_multiple = self._get_width_multiple(architecture)

        # 构建网络
        self.backbone = self._build_backbone()
        self.neck = self._build_neck()
        self.head = self._build_head()

    def _get_depth_multiple(self, architecture: str) -> float:
        """获取深度倍数"""
        depth_map = {
            "yolov5n": 0.33,
            "yolov5s": 0.33,
            "yolov5m": 0.67,
            "yolov5l": 1.0,
            "yolov5x": 1.33,
        }
        return depth_map.get(architecture, 0.33)

    def _get_width_multiple(self, architecture: str) -> float:
        """获取宽度倍数"""
        width_map = {
            "yolov5n": 0.25,
            "yolov5s": 0.50,
            "yolov5m": 0.75,
            "yolov5l": 1.0,
            "yolov5x": 1.25,
        }
        return width_map.get(architecture, 0.25)

    def _make_divisible(self, v: int, divisor: int = 8) -> int:
        """确保通道数能被divisor整除"""
        return int(v * self.width_multiple + divisor / 2) // divisor * divisor

    def _build_backbone(self) -> nn.Module:
        """构建特征提取骨干网络"""
        return YOLOBackbone(
            input_channels=self.input_channels,
            width_multiple=self.width_multiple,
            depth_multiple=self.depth_multiple
        )

    def _build_neck(self) -> nn.Module:
        """构建特征融合颈部网络"""
        return YOLONeck(
            width_multiple=self.width_multiple,
            depth_multiple=self.depth_multiple
        )

    def _build_head(self) -> nn.Module:
        """构建检测头"""
        return YOLOHead(
            num_classes=self.num_classes,
            width_multiple=self.width_multiple
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量 [batch, 3, height, width]

        Returns:
            检测输出列表 [P3_out, P4_out, P5_out]
            每个输出形状: [batch, anchors, grid_h, grid_w, 5+num_classes]
        """
        # Backbone特征提取
        backbone_features = self.backbone(x)

        # Neck特征融合
        neck_features = self.neck(backbone_features)

        # Head检测
        detections = self.head(neck_features)

        return detections


class YOLOBackbone(nn.Module):
    """
    YOLO骨干网络
    CSPDarknet风格的特征提取网络
    """

    def __init__(
        self,
        input_channels: int = 3,
        width_multiple: float = 0.25,
        depth_multiple: float = 0.33
    ):
        """
        初始化骨干网络

        Args:
            input_channels: 输入通道数
            width_multiple: 宽度倍数
            depth_multiple: 深度倍数
        """
        super().__init__()

        # 计算通道数
        def make_divisible(v, divisor=8):
            return int(v * width_multiple + divisor / 2) // divisor * divisor

        c1 = make_divisible(64)
        c2 = make_divisible(128)
        c3 = make_divisible(256)
        c4 = make_divisible(512)

        # Focus层（下采样）
        self.focus = nn.Sequential(
            Conv(input_channels * 4, c1, 3, 1),
            Conv(c1, c2, 3, 2),  # 下采样
        )

        # CSP1_1
        self.csp1 = Conv(c2, c3, 3, 2)  # 下采样

        # CSP2_1
        self.csp2 = Conv(c3, c4, 3, 2)  # 下采样

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播

        Args:
            x: 输入张量

        Returns:
            多尺度特征列表
        """
        x = self.focus(x)
        x = self.csp1(x)
        x = self.csp2(x)

        # 返回多尺度特征（简化版本）
        return [x, x, x]


class YOLONeck(nn.Module):
    """
    YOLO颈部网络
    PANet风格的特征融合网络
    """

    def __init__(
        self,
        width_multiple: float = 0.25,
        depth_multiple: float = 0.33
    ):
        """
        初始化颈部网络

        Args:
            width_multiple: 宽度倍数
            depth_multiple: 深度倍数
        """
        super().__init__()

        def make_divisible(v, divisor=8):
            return int(v * width_multiple + divisor / 2) // divisor * divisor

        in_channels = [make_divisible(256), make_divisible(512), make_divisible(1024)]

        # 上采样路径
        self.up_conv1 = Conv(in_channels[2], in_channels[1], 1, 1)
        self.up_conv2 = Conv(in_channels[1], in_channels[0], 1, 1)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播

        Args:
            x: 骨干网络输出特征

        Returns:
            融合后的特征列表
        """
        # 简化版本：直接返回输入特征
        return x


class YOLOHead(nn.Module):
    """
    YOLO检测头
    输出边界框、目标性和类别预测
    """

    def __init__(
        self,
        num_classes: int = 80,
        width_multiple: float = 0.25
    ):
        """
        初始化检测头

        Args:
            num_classes: 类别数
            width_multiple: 宽度倍数
        """
        super().__init__()

        self.num_classes = num_classes

        def make_divisible(v, divisor=8):
            return int(v * width_multiple + divisor / 2) // divisor * divisor

        # 每个尺度的输出通道数
        # 输出格式: (x, y, w, h, objectness, class1, class2, ...)
        self.num_outputs = 5 + num_classes

        # 三个检测头（对应三个尺度）
        self.head1 = nn.Conv2d(make_divisible(256), 3 * self.num_outputs, 1)
        self.head2 = nn.Conv2d(make_divisible(512), 3 * self.num_outputs, 1)
        self.head3 = nn.Conv2d(make_divisible(1024), 3 * self.num_outputs, 1)

        # 锚点（3个尺度，每个尺度3个锚点）
        self.anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],     # P3/8
            [[30, 61], [62, 45], [59, 119]],    # P4/16
            [[116, 90], [156, 198], [373, 326]]  # P5/32
        ], dtype=torch.float32)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        前向传播

        Args:
            features: 颈部网络输出特征 [P3, P4, P5]

        Returns:
            检测输出列表 [P3_out, P4_out, P5_out]
        """
        # 应用检测头
        out1 = self.head1(features[0])  # P3
        out2 = self.head2(features[1])  # P4
        out3 = self.head3(features[2])  # P5

        # 重排输出: [batch, anchors, grid_h, grid_w, 5+num_classes]
        out1 = self._reshape_output(out1)
        out2 = self._reshape_output(out2)
        out3 = self._reshape_output(out3)

        return [out1, out2, out3]

    def _reshape_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        重排检测头输出

        Args:
            x: 原始输出 [batch, 3*(5+num_classes), grid_h, grid_w]

        Returns:
            重排后的输出 [batch, 3, grid_h, grid_w, 5+num_classes]
        """
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, 3, -1, height, width)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        return x


class Conv(nn.Module):
    """
    标准卷积块: Conv2d + BatchNorm2d + SiLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1
    ):
        """
        初始化卷积块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            groups: 分组数
        """
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """
    标准瓶颈块
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True
    ):
        """
        初始化瓶颈块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            shortcut: 是否使用残差连接
        """
        super().__init__()

        c_ = out_channels // 2  # 隐藏通道数

        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_, out_channels, 3, 1)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        out = self.cv2(self.cv1(x))
        return x + out if self.add else out


class CSP(nn.Module):
    """
    CSP (Cross Stage Partial) 块
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int = 1,
        shortcut: bool = True
    ):
        """
        初始化CSP块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            num_bottlenecks: 瓶颈块数量
            shortcut: 是否使用残差连接
        """
        super().__init__()

        c_ = out_channels // 2  # 隐藏通道数

        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.cv3 = Conv(2 * c_, out_channels, 1, 1)

        self.bottlenecks = nn.Sequential(*[
            Bottleneck(c_, c_, shortcut) for _ in range(num_bottlenecks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x1 = self.bottlenecks(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat([x1, x2], dim=1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5
    ):
        """
        初始化SPPF块

        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 最大池化核大小
        """
        super().__init__()

        c_ = in_channels // 2  # 隐藏通道数

        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


def create_yolo_model(
    variant: str = "yolov5n",
    num_classes: int = 80,
    pretrained: bool = False
) -> DetectionModel:
    """
    创建YOLO检测模型

    Args:
        variant: 模型变体 (n, s, m, l, x)
        num_classes: 类别数
        pretrained: 是否使用预训练权重

    Returns:
        DetectionModel实例
    """
    model = DetectionModel(
        architecture=f"yolov5{variant}",
        num_classes=num_classes
    )

    # 加载预训练权重（如果需要）
    if pretrained:
        # 这里可以实现从URL加载预训练权重
        pass

    return model
