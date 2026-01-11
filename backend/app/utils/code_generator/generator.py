"""
PyTorch代码生成引擎

本模块负责协调整个PyTorch代码生成流程：
1. 调用LayerBuilder构建层代码
2. 生成导入语句
3. 组装完整的模型类代码
4. 集成CodeValidator进行验证
5. 应用CodeOptimizer进行代码优化
6. 生成元数据

作者: CV Studio 开发团队
日期: 2025-12-25
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import keyword
import re

from app.utils.graph_traversal import Graph
from app.utils.shape_inference import NodeShapeInfo, TensorShape
from .layer_builder import LayerBuilder
from .validator import CodeValidator
from .optimizer import CodeOptimizer


class CodeGenerator:
    """
    PyTorch代码生成引擎

    职责：
    1. 协调整体代码生成流程
    2. 调用LayerBuilder构建层代码
    3. 组装完整的模型类代码
    4. 集成CodeValidator进行代码验证
    5. 应用CodeOptimizer进行代码优化
    6. 生成元数据
    """

    @staticmethod
    def sanitize_model_name(name: str, original_name: str = None) -> tuple:
        """
        规范化模型名称，使其符合Python类名规范

        规则：
        1. 只保留字母、数字和下划线
        2. 必须以字母或下划线开头
        3. 不能是Python关键字
        4. 使用驼峰命名法（PascalCase）

        Args:
            name: 原始模型名称
            original_name: 原始名称（用于显示），如果为None则使用name

        Returns:
            (sanitized_name, display_name) 元组
            - sanitized_name: 符合Python规范的类名
            - display_name: 用于显示的原始名称

        Examples:
            "Mini-ResNet" -> ("MiniResNet", "Mini-ResNet")
            "my_model" -> ("MyModel", "my_model")
            "123model" -> ("Model123", "123model")
            "class" -> ("ModelClass", "class")
        """
        if original_name is None:
            original_name = name

        # 如果名称已经是有效的Python类名，直接返回
        if name.isidentifier() and not keyword.iskeyword(name):
            # 检查首字母是否大写（PascalCase）
            if name[0].isupper() or len(name) == 0:
                return name, original_name

        # 处理步骤：
        # 1. 将特殊字符替换为空格（用于分割单词）
        # 2. 将每个单词首字母大写
        # 3. 合并单词

        # 替换常见的分隔符为空格
        normalized = re.sub(r'[-_\s+.]+', ' ', name)

        # 移除开头的数字
        if normalized and normalized[0].isdigit():
            # 找到第一个字母的位置
            match = re.search(r'[a-zA-Z]', normalized)
            if match:
                # 在字母前添加 Model 前缀
                word_start = match.start()
                prefix = normalized[word_start:]
                normalized = f"Model {prefix}"
            else:
                normalized = "Model"

        # 将每个单词首字母大写并合并
        words = [w for w in normalized.split() if w]
        if not words:
            sanitized = "Model"
        else:
            # 每个单词首字母大写，其余小写
            words = [w.capitalize() for w in words]
            sanitized = ''.join(words)

        # 确保不以数字开头
        while sanitized and sanitized[0].isdigit():
            sanitized = 'M' + sanitized

        # 检查是否是Python关键字
        if keyword.iskeyword(sanitized):
            sanitized = "Model" + sanitized.capitalize()

        # 确保首字母大写
        if sanitized and sanitized[0].islower():
            sanitized = sanitized.capitalize()

        # 最终检查：必须是有效的标识符
        if not sanitized.isidentifier():
            # 如果仍然无效，使用默认名称
            sanitized = "GeneratedModel"

        return sanitized, original_name

    def __init__(
        self,
        template_dir: str = None,
        enable_optimization: bool = True,
        optimization_level: str = "basic"
    ):
        """
        初始化代码生成器

        Args:
            template_dir: 模板目录路径（暂时不使用，预留）
            enable_optimization: 是否启用代码优化
            optimization_level: 优化级别 ("basic", "aggressive")
        """
        self.layer_builder = LayerBuilder()
        self.validator = CodeValidator()
        self.enable_optimization = enable_optimization
        self.optimization_level = optimization_level

        if enable_optimization:
            enable_sequential = optimization_level == "aggressive"
            self.optimizer = CodeOptimizer(
                enable_sequential=enable_sequential,
                enable_inline=True
            )
        else:
            self.optimizer = None

    def generate(
        self,
        graph: Graph,
        execution_order: dict,
        shape_map: Dict[str, NodeShapeInfo],
        model_name: str = "GeneratedModel"
    ) -> Dict[str, Any]:
        """
        生成完整的PyTorch模型代码

        Args:
            graph: 计算图对象
            execution_order: 执行顺序字典
            shape_map: 节点形状映射
            model_name: 模型类名

        Returns:
            {
                "code": str,                    # 生成的完整代码
                "model_class_name": str,        # 模型类名（规范化后的）
                "original_model_name": str,     # 原始模型名称
                "init_method": str,             # __init__方法代码
                "forward_method": str,          # forward方法代码
                "layer_count": int,             # 层数量
                "validation": {...},            # 验证结果
                "imports": List[str],           # 导入语句
                "metadata": {...}               # 元数据
            }
        """
        # 0. 规范化模型名称，使其符合Python类名规范
        sanitized_name, original_name = self.sanitize_model_name(model_name)

        # 1. 构建__init__方法
        init_result = self.layer_builder.build_init_method(
            graph, execution_order, shape_map
        )

        # 2. 构建forward方法
        forward_result = self.layer_builder.build_forward_method(
            graph, execution_order, shape_map,
            init_result["layer_names"]
        )

        # 3. 生成导入语句
        imports = self._generate_imports(graph, execution_order)

        # 4. 组装完整代码
        full_code = self._assemble_full_code(
            model_name=sanitized_name,  # 使用规范化的名称
            display_name=original_name,  # 传递原始名称用于显示
            imports=imports,
            init_method=init_result["code"],
            forward_method=forward_result["code"],
            docstring=self._generate_docstring(
                graph, execution_order, init_result["layer_defs"]
            ),
            layer_defs=init_result["layer_defs"],
            operations=forward_result["operations"]
        )

        # 5. 应用代码优化
        optimization_info = None
        if self.enable_optimization and self.optimizer:
            optimization_result = self.optimizer.optimize(
                code=full_code,
                graph=graph,
                layer_defs=init_result["layer_defs"],
                operations=forward_result["operations"]
            )
            full_code = optimization_result["code"]
            optimization_info = {
                "applied": optimization_result["optimizations"],
                "original_size": optimization_result["original_size"],
                "optimized_size": optimization_result["optimized_size"]
            }

        # 6. 验证生成的代码
        # 获取输入形状用于前向传播测试
        input_nodes = execution_order["input_nodes"]
        test_input_shape = None
        if input_nodes and input_nodes[0] in shape_map:
            input_shape_info = shape_map[input_nodes[0]].output_shape
            # 将 TensorShape 转换为 tuple 格式: (batch, channels, height, width)
            if input_shape_info.features is not None:
                # 1D 张量: (batch, features)
                test_input_shape = (1, input_shape_info.features)
            else:
                # 4D 张量: (batch, channels, height, width)
                test_input_shape = (
                    1,  # batch_size
                    input_shape_info.channels,
                    input_shape_info.height,
                    input_shape_info.width
                )

        validation = self.validator.validate_code(
            full_code,
            sanitized_name,  # 使用规范化的名称
            init_result["layer_defs"],
            forward_result["operations"],
            test_input_shape
        )

        # 7. 生成元数据
        metadata = self._generate_metadata(
            graph, execution_order, shape_map,
            init_result["layer_defs"], validation
        )
        if optimization_info:
            metadata["optimization"] = optimization_info

        return {
            "code": full_code,
            "model_class_name": sanitized_name,  # 规范化后的类名
            "original_model_name": original_name,  # 原始名称
            "init_method": init_result["code"],
            "forward_method": forward_result["code"],
            "layer_count": len(execution_order["layers"]),
            "validation": validation,
            "imports": imports,
            "metadata": metadata
        }

    def _generate_imports(
        self,
        graph: Graph,
        execution_order: dict
    ) -> List[str]:
        """
        生成导入语句

        Args:
            graph: 计算图对象
            execution_order: 执行顺序字典

        Returns:
            导入语句列表
        """
        imports = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
        ]

        # 检查是否需要特殊导入
        layers = execution_order["layers"]
        has_drop_path = False
        has_multihead_attention = False

        for node_id in layers:
            node = graph.nodes[node_id]

            if node.type == "DropPath":
                has_drop_path = True
            elif node.type == "MultiheadAttention":
                has_multihead_attention = True

        # 如果有DropPath，添加DropPath实现或timm导入
        if has_drop_path:
            imports.append("")
            imports.append("# DropPath实现（随机路径深度）")
            imports.append("class DropPath(nn.Module):")
            imports.append("    \"\"\"Stochastic Depth正则化")
            imports.append("")
            imports.append("    来自: https://github.com/rwightman/pytorch-image-models")
            imports.append("    \"\"\"")
            imports.append("    def __init__(self, drop_prob: float = 0.):")
            imports.append("        super().__init__()")
            imports.append("        self.drop_prob = drop_prob")
            imports.append("")
            imports.append("    def forward(self, x):")
            imports.append("        if self.drop_prob == 0. or not self.training:")
            imports.append("            return x")
            imports.append("        keep_prob = 1 - self.drop_prob")
            imports.append("        shape = (x.shape[0],) + (1,) * (x.ndim - 1)")
            imports.append("        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)")
            imports.append("        random_tensor.floor_()")
            imports.append("        output = x.div(keep_prob) * random_tensor")
            imports.append("        return output")

        return imports

    def _generate_docstring(
        self,
        graph: Graph,
        execution_order: dict,
        layer_defs: List[dict]
    ) -> str:
        """
        生成模型文档字符串

        Args:
            graph: 计算图对象
            execution_order: 执行顺序字典
            layer_defs: 层定义列表

        Returns:
            文档字符串
        """
        lines = []
        lines.append("模型结构:")

        for layer_def in layer_defs:
            desc = layer_def.description if hasattr(layer_def, 'description') else ""
            if desc:
                lines.append(f"  - {layer_def.name}: {desc}")
            else:
                lines.append(f"  - {layer_def.name}: {layer_def.layer_type}")

        return "\n".join(lines)

    def _assemble_full_code(
        self,
        model_name: str,
        imports: List[str],
        init_method: str,
        forward_method: str,
        docstring: str,
        layer_defs: List[dict],
        operations: List[dict],
        display_name: str = None
    ) -> str:
        """
        组装完整的模型类代码

        Args:
            model_name: 模型类名（规范化的）
            imports: 导入语句列表
            init_method: __init__方法代码
            forward_method: forward方法代码
            docstring: 文档字符串
            layer_defs: 层定义列表
            operations: 操作列表
            display_name: 用于显示的原始名称（可选）

        Returns:
            完整的代码字符串
        """
        lines = []

        # 使用原始名称用于显示（如果提供）
        name_for_display = display_name if display_name else model_name

        # 文件头注释
        lines.append('"""')
        lines.append(f"{name_for_display} - 自动生成的PyTorch模型")
        lines.append("")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append('"""')
        lines.append("")

        # 导入语句
        for imp in imports:
            lines.append(imp)
        lines.append("")
        lines.append("")

        # 类定义（使用规范化的名称）
        lines.append(f"class {model_name}(nn.Module):")
        lines.append('    """')
        lines.append(f'    {name_for_display} 模型')
        lines.append("")
        lines.append("    " + docstring.replace("\n", "\n    "))
        lines.append("")
        lines.append("    架构信息:")
        lines.append(f"    - 层数量: {len(layer_defs)}")
        lines.append(f"    - 参数数量: 待计算")
        lines.append('    """')
        lines.append("")

        # __init__方法
        lines.append(init_method)
        lines.append("")

        # forward方法
        lines.append(forward_method)
        lines.append("")

        # 模型元数据
        lines.append("# 模型元数据")
        lines.append("MODEL_INFO = {")
        lines.append(f'    "name": "{name_for_display}",')  # 使用原始名称
        lines.append(f'    "class_name": "{model_name}",')   # 添加类名
        lines.append(f'    "layer_count": {len(layer_defs)},')
        lines.append(f'    "num_parameters": 0,  # 待计算')
        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines.append(f'    "generation_time": "{generation_time}",')
        lines.append("}")

        return "\n".join(lines)

    def _generate_metadata(
        self,
        graph: Graph,
        execution_order: dict,
        shape_map: Dict[str, NodeShapeInfo],
        layer_defs: List[dict],
        validation: dict
    ) -> Dict[str, Any]:
        """
        生成模型元数据

        Args:
            graph: 计算图对象
            execution_order: 执行顺序字典
            shape_map: 节点形状映射
            layer_defs: 层定义列表
            validation: 验证结果

        Returns:
            元数据字典
        """
        # 获取输入形状
        input_nodes = execution_order["input_nodes"]
        input_shape = None
        if input_nodes and input_nodes[0] in shape_map:
            input_shape_info = shape_map[input_nodes[0]].output_shape
            input_shape = self._tensor_shape_to_list(input_shape_info)

        # 获取输出形状
        output_nodes = execution_order["output_nodes"]
        output_shape = None
        if output_nodes and output_nodes[0] in shape_map:
            output_shape_info = shape_map[output_nodes[0]].output_shape
            output_shape = self._tensor_shape_to_list(output_shape_info)

        # 估算参数量
        num_parameters = self._estimate_parameters(layer_defs)

        return {
            "layer_count": len(execution_order["layers"]),
            "depth": execution_order["depth"],
            "num_parameters": num_parameters,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "validation_passed": validation["valid"],
            "forward_pass_success": validation.get("forward_pass_success", False)
        }

    def _tensor_shape_to_list(self, shape: TensorShape) -> List[Any]:
        """
        将TensorShape转换为列表格式

        Args:
            shape: TensorShape对象

        Returns:
            形状列表，例如 ["B", 3, 224, 224]
        """
        if shape.features:
            # 1D张量
            return ["B", shape.features]
        else:
            # 2D/3D/4D张量
            return ["B", shape.channels, shape.height, shape.width]

    def _estimate_parameters(self, layer_defs: List[dict]) -> int:
        """
        估算模型参数量

        这是一个粗略估计，仅用于显示目的

        Args:
            layer_defs: 层定义列表

        Returns:
            参数总数估计值
        """
        total_params = 0

        for layer_def in layer_defs:
            # 支持LayerDefinition对象或字典
            if hasattr(layer_def, 'layer_type'):
                layer_type = layer_def.layer_type
                params = layer_def.params if hasattr(layer_def, 'params') else {}
            else:
                layer_type = layer_def["layer_type"]
                params = layer_def["params"]

            if layer_type == "Conv2d":
                # Conv2d: kernel_size * kernel_size * in_channels * out_channels + out_channels
                in_channels = params.get("in_channels", 0)
                out_channels = params.get("out_channels", 0)
                kernel_size = params.get("kernel_size", 3)
                conv_params = kernel_size * kernel_size * in_channels * out_channels + out_channels
                total_params += conv_params

            elif layer_type == "Linear":
                # Linear: in_features * out_features + out_features
                in_features = params.get("in_features", 0)
                out_features = params.get("out_features", 0)
                linear_params = in_features * out_features + out_features
                total_params += linear_params

            elif layer_type == "BatchNorm2d":
                # BatchNorm2d: 2 * num_features (scale + bias)
                num_features = params.get("num_features", 0)
                bn_params = 2 * num_features
                total_params += bn_params

            # 其他层的参数量可以忽略或粗略估计

        return total_params
