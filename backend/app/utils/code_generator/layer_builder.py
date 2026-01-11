"""
层代码构建器

本模块负责为每种层类型生成PyTorch代码，包括：
1. __init__方法中的层定义
2. forward方法中的张量操作
3. 特殊节点处理（Concat、Add、Flatten等）

作者: CV Studio 开发团队
日期: 2025-12-25
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from app.utils.graph_traversal import Graph, Node
from app.utils.shape_inference import NodeShapeInfo, TensorShape


@dataclass
class LayerDefinition:
    """层定义数据类"""
    layer_type: str
    name: str
    code: str
    params: Dict[str, Any]
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    description: Optional[str] = None


@dataclass
class Operation:
    """forward操作数据类"""
    node_id: str
    node_type: str
    code: str
    comment: Optional[str] = None
    input_vars: List[str] = None
    output_var: str = None


class LayerBuilder:
    """
    层代码构建器

    职责：
    1. 为每种层类型生成__init__中的层定义代码
    2. 生成forward方法中的张量流动代码
    3. 处理特殊节点（Concat、Add、Flatten等）
    4. 生成语义化层命名（conv1, bn1, fc1等）
    """

    # 层类型到PyTorch类的映射
    LAYER_TYPE_MAP = {
        "Conv2d": "nn.Conv2d",
        "ConvTranspose2d": "nn.ConvTranspose2d",
        "Linear": "nn.Linear",
        "BatchNorm2d": "nn.BatchNorm2d",
        "LayerNorm": "nn.LayerNorm",
        "GroupNorm": "nn.GroupNorm",
        "InstanceNorm2d": "nn.InstanceNorm2d",
        "MaxPool2d": "nn.MaxPool2d",
        "AvgPool2d": "nn.AvgPool2d",
        "AdaptiveAvgPool2d": "nn.AdaptiveAvgPool2d",
        "ReLU": "nn.ReLU",
        "ReLU6": "nn.ReLU6",
        "LeakyReLU": "nn.LeakyReLU",
        "SiLU": "nn.SiLU",
        "Sigmoid": "nn.Sigmoid",
        "Softmax": "nn.Softmax",
        "Tanh": "nn.Tanh",
        "GELU": "nn.GELU",
        "Dropout": "nn.Dropout",
        "DropPath": "DropPath",  # 需要自定义实现
        "Flatten": "nn.Flatten",
        "Upsample": "nn.Upsample",
        "Identity": "nn.Identity",
        "MultiheadAttention": "nn.MultiheadAttention",
    }

    def __init__(self):
        """初始化层构建器"""
        self.layer_counter = {}  # 用于生成唯一层名
        self.layer_defs = []     # 层定义列表

    def build_init_method(
        self,
        graph: Graph,
        execution_order: dict,
        shape_map: Dict[str, NodeShapeInfo]
    ) -> Dict[str, Any]:
        """
        构建__init__方法代码

        Args:
            graph: 计算图对象
            execution_order: 执行顺序字典
            shape_map: 节点形状映射

        Returns:
            {
                "code": str,              # __init__方法代码
                "layer_defs": List[dict], # 层定义信息列表
                "layer_names": Dict[str, str]  # 节点ID到层名的映射
            }
        """
        layers = execution_order["layers"]
        init_lines = []
        self.layer_defs = []
        self.layer_counter = {}
        layer_names = {}

        for node_id in layers:
            node = graph.nodes[node_id]
            shape_info = shape_map.get(node_id)

            # 生成层名
            layer_name = self._generate_layer_name(node)
            layer_names[node_id] = layer_name

            # 构建层定义
            layer_def = self._build_layer_definition(node, shape_info)
            layer_def.name = layer_name

            # 生成代码行
            init_lines.append(
                f"self.{layer_name} = {layer_def.code}"
            )

            self.layer_defs.append(layer_def)

        # 组装__init__方法
        init_code = self._format_init_method(init_lines)

        return {
            "code": init_code,
            "layer_defs": self.layer_defs,
            "layer_names": layer_names
        }

    def build_forward_method(
        self,
        graph: Graph,
        execution_order: dict,
        shape_map: Dict[str, NodeShapeInfo],
        layer_names: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """
        构建forward方法代码

        Args:
            graph: 计算图对象
            execution_order: 执行顺序字典
            shape_map: 节点形状映射
            layer_names: 节点ID到层名的映射

        Returns:
            {
                "code": str,                    # forward方法代码
                "operations": List[dict],        # 操作列表
                "variables": Dict[str, str]     # 节点ID到变量名的映射
            }
        """
        forward_order = execution_order["forward_order"]
        forward_lines = []
        operations = []
        variables = {}

        for node_id in forward_order:
            node = graph.nodes[node_id]

            # 跳过Input节点（输入参数x就是Input的输出）
            if node.type == "Input":
                variables[node_id] = "x"
                continue

            # 获取前驱节点
            predecessors = graph.get_predecessors(node_id)

            if len(predecessors) == 0:
                # 异常情况：没有输入
                continue

            # 对于单输入节点，输出变量始终是 x
            # 对于多输入节点（Concat、Add），输出变量也是 x
            # variables 存储的是该节点的输出变量名
            variables[node_id] = "x"

            # 生成操作语句
            operation = self._build_operation(
                node, predecessors, layer_names, variables, shape_map
            )

            forward_lines.append(operation.code)
            operations.append(operation)

        # 组装forward方法
        forward_code = self._format_forward_method(forward_lines)

        return {
            "code": forward_code,
            "operations": operations,
            "variables": variables
        }

    def _generate_layer_name(self, node: Node) -> str:
        """
        生成语义化层名

        命名规则：
        - Conv2d: conv1, conv2, conv3...
        - Linear: fc1, fc2...
        - BatchNorm2d: bn1, bn2...
        - MaxPool2d/AvgPool2d: pool1, pool2...
        - ReLU: relu1, relu2...（通常内联）
        - Dropout: dropout1, dropout2...
        - Flatten: flatten
        """
        node_type = node.type
        counter_key = node_type.lower()

        if counter_key not in self.layer_counter:
            self.layer_counter[counter_key] = 0

        self.layer_counter[counter_key] += 1
        count = self.layer_counter[counter_key]

        # 命名映射
        name_map = {
            "conv2d": "conv",
            "convtranspose2d": "convt",
            "linear": "fc",
            "batchnorm2d": "bn",
            "layernorm": "ln",
            "groupnorm": "gn",
            "instancenorm2d": "in",
            "maxpool2d": "maxpool",
            "avgpool2d": "avgpool",
            "adaptiveavgpool2d": "adapool",
            "relu": "relu",
            "relu6": "relu6",
            "leakyrelu": "leaky_relu",
            "silu": "silu",
            "sigmoid": "sigmoid",
            "softmax": "softmax",
            "tanh": "tanh",
            "gelu": "gelu",
            "dropout": "dropout",
            "droppath": "droppath",
            "flatten": "flatten",
            "upsample": "upsample",
            "identity": "identity",
            "multiheadattention": "mha",
        }

        base_name = name_map.get(node_type.lower(), node_type.lower())

        # Flatten和Identity通常只有一个实例
        if node_type in ["Flatten", "Identity"]:
            return base_name

        return f"{base_name}{count}"

    def _build_layer_definition(
        self,
        node: Node,
        shape_info: NodeShapeInfo
    ) -> LayerDefinition:
        """
        构建单个层的定义代码

        Args:
            node: 节点对象
            shape_info: 形状信息

        Returns:
            LayerDefinition对象
        """
        node_type = node.type
        params = node.params

        # 根据层类型生成代码
        if node_type == "Conv2d":
            return self._build_conv2d(params, shape_info)
        elif node_type == "ConvTranspose2d":
            return self._build_conv_transpose2d(params, shape_info)
        elif node_type == "Linear":
            return self._build_linear(params, shape_info)
        elif node_type == "BatchNorm2d":
            return self._build_batchnorm2d(params, shape_info)
        elif node_type == "LayerNorm":
            return self._build_layer_norm(params, shape_info)
        elif node_type == "GroupNorm":
            return self._build_group_norm(params, shape_info)
        elif node_type == "InstanceNorm2d":
            return self._build_instance_norm2d(params, shape_info)
        elif node_type == "MaxPool2d":
            return self._build_maxpool2d(params)
        elif node_type == "AvgPool2d":
            return self._build_avgpool2d(params)
        elif node_type == "AdaptiveAvgPool2d":
            return self._build_adaptive_avgpool2d(params)
        elif node_type in ["ReLU", "LeakyReLU", "SiLU", "Sigmoid", "Softmax", "ReLU6", "Tanh", "GELU"]:
            return self._build_activation(node_type, params)
        elif node_type == "Dropout":
            return self._build_dropout(params)
        elif node_type == "DropPath":
            return self._build_drop_path(params)
        elif node_type == "Flatten":
            return self._build_flatten(params)
        elif node_type == "Upsample":
            return self._build_upsample(params)
        elif node_type == "Identity":
            return self._build_identity()
        elif node_type == "MultiheadAttention":
            return self._build_multihead_attention(params, shape_info)
        else:
            raise ValueError(f"不支持的层类型: {node_type}")

    def _build_conv2d(self, params: dict, shape_info: NodeShapeInfo) -> LayerDefinition:
        """
        构建Conv2d层定义

        示例输出：
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        """
        in_channels = params.get("in", params.get("in_channels", 3))
        out_channels = params.get("out", params.get("out_channels", 64))
        kernel_size = params.get("k", params.get("kernel_size", 3))
        stride = params.get("s", params.get("stride", 1))
        padding = params.get("p", params.get("padding", 0))

        # 如果有形状信息，优先使用形状推断的值
        if shape_info and shape_info.input_shape:
            in_channels = shape_info.input_shape.channels
        if shape_info and shape_info.output_shape:
            out_channels = shape_info.output_shape.channels

        # 构建参数列表
        args = [str(in_channels), str(out_channels)]
        kwargs = {"kernel_size": kernel_size}

        if stride != 1:
            kwargs["stride"] = stride
        if padding != 0:
            kwargs["padding"] = padding

        # 其他可选参数
        if "dilation" in params:
            kwargs["dilation"] = params["dilation"]
        if "groups" in params:
            kwargs["groups"] = params["groups"]
        if "bias" in params:
            kwargs["bias"] = params["bias"]

        # 格式化代码
        code = f"nn.Conv2d({', '.join(args)}"
        if kwargs:
            code += ", " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        code += ")"

        return LayerDefinition(
            layer_type="Conv2d",
            name="",  # 将由调用者设置
            code=code,
            params={"in_channels": in_channels, "out_channels": out_channels, **kwargs},
            in_channels=in_channels,
            out_channels=out_channels,
            description=f"Conv2d: {in_channels}->{out_channels}, k={kernel_size}"
        )

    def _build_linear(self, params: dict, shape_info: NodeShapeInfo) -> LayerDefinition:
        """
        构建Linear层定义

        示例输出：
        nn.Linear(512, 10)
        """
        in_features = params.get("in", params.get("in_features", params.get("in_f", 0)))
        out_features = params.get("out", params.get("out_features", params.get("out_f", 1000)))

        # 如果有形状信息，优先使用形状推断的值
        if shape_info and shape_info.input_shape:
            if shape_info.input_shape.features:
                in_features = shape_info.input_shape.features
            else:
                # 从4D张量推断
                in_features = (
                    shape_info.input_shape.channels *
                    shape_info.input_shape.height *
                    shape_info.input_shape.width
                )

        code = f"nn.Linear({in_features}, {out_features})"

        return LayerDefinition(
            layer_type="Linear",
            name="",
            code=code,
            params={"in_features": in_features, "out_features": out_features},
            in_channels=in_features,
            out_channels=out_features,
            description=f"Linear: {in_features}->{out_features}"
        )

    def _build_batchnorm2d(self, params: dict, shape_info: NodeShapeInfo) -> LayerDefinition:
        """
        构建BatchNorm2d层定义

        示例输出：
        nn.BatchNorm2d(64)
        """
        num_features = params.get("num_f", params.get("num_features", 64))

        # 如果有形状信息，优先使用形状推断的值
        if shape_info and shape_info.input_shape:
            num_features = shape_info.input_shape.channels

        code = f"nn.BatchNorm2d({num_features})"

        return LayerDefinition(
            layer_type="BatchNorm2d",
            name="",
            code=code,
            params={"num_features": num_features},
            in_channels=num_features,
            out_channels=num_features,
            description=f"BatchNorm2d: {num_features}"
        )

    def _build_layer_norm(self, params: dict, shape_info: NodeShapeInfo) -> LayerDefinition:
        """
        构建LayerNorm层定义

        示例输出：
        nn.LayerNorm(normalized_shape)
        """
        # 简化实现：通常从通道数推断
        normalized_size = params.get("normalized_size", params.get("num_features", 64))

        if shape_info and shape_info.input_shape:
            normalized_size = shape_info.input_shape.channels

        code = f"nn.LayerNorm({normalized_size})"

        return LayerDefinition(
            layer_type="LayerNorm",
            name="",
            code=code,
            params={"normalized_size": normalized_size},
            description=f"LayerNorm: {normalized_size}"
        )

    def _build_maxpool2d(self, params: dict) -> LayerDefinition:
        """
        构建MaxPool2d层定义

        示例输出：
        nn.MaxPool2d(kernel_size=2, stride=2)
        """
        kernel_size = params.get("k", params.get("kernel_size", 2))
        stride = params.get("s", params.get("stride", kernel_size))
        padding = params.get("p", params.get("padding", 0))

        code = f"nn.MaxPool2d(kernel_size={kernel_size}, stride={stride}"
        if padding != 0:
            code += f", padding={padding}"
        code += ")"

        return LayerDefinition(
            layer_type="MaxPool2d",
            name="",
            code=code,
            params={"kernel_size": kernel_size, "stride": stride, "padding": padding},
            description=f"MaxPool2d: k={kernel_size}"
        )

    def _build_avgpool2d(self, params: dict) -> LayerDefinition:
        """
        构建AvgPool2d层定义

        示例输出：
        nn.AvgPool2d(kernel_size=2, stride=2)
        """
        kernel_size = params.get("k", params.get("kernel_size", 2))
        stride = params.get("s", params.get("stride", kernel_size))
        padding = params.get("p", params.get("padding", 0))

        code = f"nn.AvgPool2d(kernel_size={kernel_size}, stride={stride}"
        if padding != 0:
            code += f", padding={padding}"
        code += ")"

        return LayerDefinition(
            layer_type="AvgPool2d",
            name="",
            code=code,
            params={"kernel_size": kernel_size, "stride": stride, "padding": padding},
            description=f"AvgPool2d: k={kernel_size}"
        )

    def _build_adaptive_avgpool2d(self, params: dict) -> LayerDefinition:
        """
        构建AdaptiveAvgPool2d层定义

        示例输出：
        nn.AdaptiveAvgPool2d((1, 1))
        """
        output_size = params.get("output_size", 1)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)

        code = f"nn.AdaptiveAvgPool2d({output_size})"

        return LayerDefinition(
            layer_type="AdaptiveAvgPool2d",
            name="",
            code=code,
            params={"output_size": output_size},
            description=f"AdaptiveAvgPool2d: {output_size}"
        )

    def _build_activation(self, node_type: str, params: dict) -> LayerDefinition:
        """
        构建激活函数层定义

        支持ReLU, ReLU6, LeakyReLU, SiLU, Sigmoid, Softmax, Tanh, GELU
        """
        layer_class = self.LAYER_TYPE_MAP[node_type]
        kwargs = {}

        if node_type == "LeakyReLU":
            negative_slope = params.get("negative_slope", 0.01)
            if negative_slope != 0.01:
                kwargs["negative_slope"] = negative_slope
            inplace = params.get("inplace", True)
            kwargs["inplace"] = inplace
        elif node_type == "Softmax":
            dim = params.get("dim", 1)
            kwargs["dim"] = dim
        elif node_type in ["ReLU6", "GELU", "Tanh"]:
            # ReLU6, GELU, Tanh的inplace参数处理
            inplace = params.get("inplace", False)
            if inplace and node_type != "Tanh":
                kwargs["inplace"] = inplace
        else:
            # ReLU, SiLU, Sigmoid等
            inplace = params.get("inplace", True)
            if inplace:
                kwargs["inplace"] = inplace

        # 格式化代码
        if kwargs:
            code = f"{layer_class}({', '.join(f'{k}={v}' for k, v in kwargs.items())})"
        else:
            code = f"{layer_class}()"

        return LayerDefinition(
            layer_type=node_type,
            name="",
            code=code,
            params=kwargs,
            description=f"{node_type}"
        )

    def _build_dropout(self, params: dict) -> LayerDefinition:
        """
        构建Dropout层定义

        示例输出：
        nn.Dropout(0.5)
        """
        p = params.get("p", params.get("probability", 0.5))
        inplace = params.get("inplace", False)

        code = f"nn.Dropout({p}"
        if inplace:
            code += ", inplace=True"
        code += ")"

        return LayerDefinition(
            layer_type="Dropout",
            name="",
            code=code,
            params={"p": p, "inplace": inplace},
            description=f"Dropout: {p}"
        )

    def _build_flatten(self, params: dict) -> LayerDefinition:
        """
        构建Flatten层定义

        示例输出：
        nn.Flatten()
        """
        start_dim = params.get("start_dim", 1)

        if start_dim == 1:
            code = "nn.Flatten()"
        else:
            code = f"nn.Flatten(start_dim={start_dim})"

        return LayerDefinition(
            layer_type="Flatten",
            name="",
            code=code,
            params={"start_dim": start_dim},
            description=f"Flatten: start_dim={start_dim}"
        )

    def _build_upsample(self, params: dict) -> LayerDefinition:
        """
        构建Upsample层定义

        示例输出：
        nn.Upsample(scale_factor=2, mode='nearest')
        """
        scale_factor = params.get("scale_factor", 2)
        mode = params.get("mode", "nearest")
        align_corners = params.get("align_corners", None)

        code = f"nn.Upsample(scale_factor={scale_factor}, mode='{mode}'"
        if align_corners is not None:
            code += f", align_corners={align_corners}"
        code += ")"

        return LayerDefinition(
            layer_type="Upsample",
            name="",
            code=code,
            params={"scale_factor": scale_factor, "mode": mode},
            description=f"Upsample: {scale_factor}x"
        )

    def _build_identity(self) -> LayerDefinition:
        """
        构建Identity层定义

        示例输出：
        nn.Identity()
        """
        return LayerDefinition(
            layer_type="Identity",
            name="",
            code="nn.Identity()",
            params={},
            description="Identity"
        )

    def _build_conv_transpose2d(self, params: dict, shape_info: NodeShapeInfo) -> LayerDefinition:
        """
        构建ConvTranspose2d层定义（转置卷积/反卷积）

        示例输出：
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        """
        in_channels = params.get("in", params.get("in_channels", 64))
        out_channels = params.get("out", params.get("out_channels", 32))
        kernel_size = params.get("k", params.get("kernel_size", 3))
        stride = params.get("s", params.get("stride", 1))
        padding = params.get("p", params.get("padding", 0))
        output_padding = params.get("output_padding", 0)

        # 如果有形状信息，优先使用形状推断的值
        if shape_info and shape_info.input_shape:
            in_channels = shape_info.input_shape.channels
        if shape_info and shape_info.output_shape:
            out_channels = shape_info.output_shape.channels

        # 构建参数列表
        args = [str(in_channels), str(out_channels)]
        kwargs = {"kernel_size": kernel_size}

        if stride != 1:
            kwargs["stride"] = stride
        if padding != 0:
            kwargs["padding"] = padding
        if output_padding != 0:
            kwargs["output_padding"] = output_padding

        # 其他可选参数
        if "dilation" in params:
            kwargs["dilation"] = params["dilation"]
        if "groups" in params:
            kwargs["groups"] = params["groups"]

        # 格式化代码
        code = f"nn.ConvTranspose2d({', '.join(args)}"
        if kwargs:
            code += ", " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        code += ")"

        return LayerDefinition(
            layer_type="ConvTranspose2d",
            name="",
            code=code,
            params={"in_channels": in_channels, "out_channels": out_channels, **kwargs},
            in_channels=in_channels,
            out_channels=out_channels,
            description=f"ConvTranspose2d: {in_channels}->{out_channels}, k={kernel_size}"
        )

    def _build_group_norm(self, params: dict, shape_info: NodeShapeInfo) -> LayerDefinition:
        """
        构建GroupNorm层定义

        示例输出：
        nn.GroupNorm(8, 64)
        """
        num_groups = params.get("num_groups", 8)
        num_channels = params.get("num_channels", params.get("num_f", 64))

        # 如果有形状信息，优先使用形状推断的值
        if shape_info and shape_info.input_shape:
            num_channels = shape_info.input_shape.channels

        code = f"nn.GroupNorm({num_groups}, {num_channels})"

        return LayerDefinition(
            layer_type="GroupNorm",
            name="",
            code=code,
            params={"num_groups": num_groups, "num_channels": num_channels},
            in_channels=num_channels,
            out_channels=num_channels,
            description=f"GroupNorm: {num_groups} groups, {num_channels} channels"
        )

    def _build_instance_norm2d(self, params: dict, shape_info: NodeShapeInfo) -> LayerDefinition:
        """
        构建InstanceNorm2d层定义

        示例输出：
        nn.InstanceNorm2d(64)
        """
        num_features = params.get("num_features", params.get("num_f", 64))

        # 如果有形状信息，优先使用形状推断的值
        if shape_info and shape_info.input_shape:
            num_features = shape_info.input_shape.channels

        code = f"nn.InstanceNorm2d({num_features})"

        return LayerDefinition(
            layer_type="InstanceNorm2d",
            name="",
            code=code,
            params={"num_features": num_features},
            in_channels=num_features,
            out_channels=num_features,
            description=f"InstanceNorm2d: {num_features}"
        )

    def _build_drop_path(self, params: dict) -> LayerDefinition:
        """
        构建DropPath层定义（随机路径深度/Stochastic Depth）

        注意：DropPath不是PyTorch原生层，需要自定义实现。
        这里生成使用timm库的DropPath或提供自定义实现的注释。

        示例输出：
        DropPath(0.1)
        """
        drop_prob = params.get("p", params.get("drop_prob", 0.1))

        code = f"DropPath({drop_prob})"

        return LayerDefinition(
            layer_type="DropPath",
            name="",
            code=code,
            params={"drop_prob": drop_prob},
            description=f"DropPath: {drop_prob}"
        )

    def _build_multihead_attention(self, params: dict, shape_info: NodeShapeInfo) -> LayerDefinition:
        """
        构建MultiheadAttention层定义

        示例输出：
        nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        """
        embed_dim = params.get("embed_dim", 256)
        num_heads = params.get("num_heads", 8)
        dropout = params.get("dropout", 0.0)
        bias = params.get("bias", True)
        add_bias_kv = params.get("add_bias_kv", False)
        kdim = params.get("kdim", None)
        vdim = params.get("vdim", None)

        # 如果有形状信息，优先使用形状推断的值
        if shape_info and shape_info.input_shape:
            if shape_info.input_shape.features:
                embed_dim = shape_info.input_shape.features
            elif shape_info.input_shape.channels:
                embed_dim = shape_info.input_shape.channels

        kwargs = {"embed_dim": embed_dim, "num_heads": num_heads, "batch_first": True}

        if dropout != 0.0:
            kwargs["dropout"] = dropout
        if not bias:
            kwargs["bias"] = bias
        if add_bias_kv:
            kwargs["add_bias_kv"] = add_bias_kv
        if kdim:
            kwargs["kdim"] = kdim
        if vdim:
            kwargs["vdim"] = vdim

        # 格式化代码
        code = "nn.MultiheadAttention("
        code += ", ".join(f"{k}={v}" for k, v in kwargs.items())
        code += ")"

        return LayerDefinition(
            layer_type="MultiheadAttention",
            name="",
            code=code,
            params=kwargs,
            description=f"MultiheadAttention: embed_dim={embed_dim}, heads={num_heads}"
        )

    def _build_operation(
        self,
        node: Node,
        predecessors: List[str],
        layer_names: Dict[str, str],
        variables: Dict[str, str],
        shape_map: Dict[str, NodeShapeInfo]
    ) -> Operation:
        """
        构建forward方法中的操作语句

        Args:
            node: 节点对象
            predecessors: 前驱节点ID列表
            layer_names: 节点ID到层名的映射
            variables: 节点ID到变量名的映射
            shape_map: 形状映射

        Returns:
            Operation对象
        """
        node_type = node.type
        params = node.params

        # 获取输入变量
        input_vars = [variables[p] for p in predecessors]

        # 多输入节点（Concat、Add）
        if len(predecessors) > 1:
            if node_type == "Concat":
                return self._build_concat_operation(node, input_vars, params)
            elif node_type == "Add":
                return self._build_add_operation(node, input_vars)

        # 单输入节点
        input_var = input_vars[0]

        # Identity层：直接传递
        if node_type == "Identity":
            return Operation(
                node_id=node.id,
                node_type=node_type,
                code=f"x = {input_var}",
                comment="Identity pass-through",
                input_vars=input_vars,
                output_var="x"
            )

        # Flatten层：调用self.flatten
        if node_type == "Flatten":
            shape_info = shape_map.get(node.id)
            comment = self._generate_flatten_comment(shape_info)
            return Operation(
                node_id=node.id,
                node_type=node_type,
                code=f"x = self.flatten({input_var})",
                comment=comment,
                input_vars=input_vars,
                output_var="x"
            )

        # 其他层：调用self.layer_name
        layer_name = layer_names.get(node.id)

        # MultiheadAttention特殊处理（需要调用forward方法）
        if node_type == "MultiheadAttention":
            embed_dim = params.get("embed_dim", node.params.get("embed_dim", 256))
            code = f"x, _ = self.{layer_name}({input_var}, {input_var}, {input_var})"
            return Operation(
                node_id=node.id,
                node_type=node_type,
                code=code,
                comment=f"{node_type} (self-attention)",
                input_vars=input_vars,
                output_var="x"
            )

        # 激活函数和Dropout可能是inplace的
        if node_type in ["ReLU", "LeakyReLU", "SiLU", "Sigmoid", "Dropout", "ReLU6", "GELU"]:
            inplace = params.get("inplace", False)
            if inplace:
                return Operation(
                    node_id=node.id,
                    node_type=node_type,
                    code=f"{input_var} = self.{layer_name}({input_var})",
                    comment=f"{node_type} (inplace)",
                    input_vars=input_vars,
                    output_var=input_var
                )

        return Operation(
            node_id=node.id,
            node_type=node_type,
            code=f"x = self.{layer_name}({input_var})",
            comment=f"{node_type}",
            input_vars=input_vars,
            output_var="x"
        )

    def _build_concat_operation(
        self,
        node: Node,
        input_vars: List[str],
        params: dict
    ) -> Operation:
        """
        构建Concat操作

        示例：
        x = torch.cat([x1, x2, x3], dim=1)
        """
        dim = params.get("dim", params.get("axis", 1))
        vars_str = ", ".join(input_vars)
        code = f"x = torch.cat([{vars_str}], dim={dim})"

        return Operation(
            node_id=node.id,
            node_type="Concat",
            code=code,
            comment=f"Concat on dim={dim}",
            input_vars=input_vars,
            output_var="x"
        )

    def _build_add_operation(
        self,
        node: Node,
        input_vars: List[str]
    ) -> Operation:
        """
        构建Add操作（残差连接）

        示例：
        x = x1 + x2 + x3
        """
        if len(input_vars) == 2:
            code = f"x = {input_vars[0]} + {input_vars[1]}"
            comment = "Residual add"
        else:
            expression = " + ".join(input_vars)
            code = f"x = {expression}"
            comment = f"Add {len(input_vars)} tensors"

        return Operation(
            node_id=node.id,
            node_type="Add",
            code=code,
            comment=comment,
            input_vars=input_vars,
            output_var="x"
        )

    def _generate_flatten_comment(self, shape_info: NodeShapeInfo) -> str:
        """
        生成Flatten操作的注释，说明形状变化

        Args:
            shape_info: 形状信息

        Returns:
            注释字符串
        """
        if shape_info and shape_info.input_shape and shape_info.output_shape:
            input_shape = shape_info.input_shape
            output_shape = shape_info.output_shape

            if input_shape.features:
                # 1D张量
                return f"Flatten: [{input_shape.features}] -> [{output_shape.features}]"
            else:
                # 4D张量
                return (
                    f"Flatten: [B, {input_shape.channels}, "
                    f"{input_shape.height}, {input_shape.width}] "
                    f"-> [B, {output_shape.features}]"
                )

        return "Flatten"

    def _format_init_method(self, init_lines: List[str]) -> str:
        """
        格式化__init__方法代码

        Args:
            init_lines: 层定义代码行列表

        Returns:
            格式化后的__init__方法代码（带4空格缩进，用于类内部）
        """
        lines = []
        # 添加4个空格缩进，使方法成为类的一部分
        lines.append("    def __init__(self):")
        lines.append("        \"\"\"")
        lines.append("        初始化模型")
        lines.append("")
        lines.append("        定义所有层结构")
        lines.append("        \"\"\"")
        # 使用现代Python 3语法，不需要硬编码类名
        lines.append("        super().__init__()")
        lines.append("")
        for line in init_lines:
            lines.append(f"        {line}")

        return "\n".join(lines)

    def _format_forward_method(self, forward_lines: List[str]) -> str:
        """
        格式化forward方法代码

        Args:
            forward_lines: 操作代码行列表

        Returns:
            格式化后的forward方法代码（带4空格缩进，用于类内部）
        """
        lines = []
        # 添加4个空格缩进，使方法成为类的一部分
        lines.append("    def forward(self, x):")
        lines.append("        \"\"\"")
        lines.append("        前向传播")
        lines.append("")
        lines.append("        Args:")
        lines.append("            x: 输入张量")
        lines.append("")
        lines.append("        Returns:")
        lines.append("            输出张量")
        lines.append("        \"\"\"")
        if forward_lines:
            lines.append("")
            for line in forward_lines:
                lines.append(f"        {line}")
            lines.append("")
        lines.append("        return x")

        return "\n".join(lines)
