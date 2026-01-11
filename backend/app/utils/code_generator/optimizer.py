"""
代码优化器

本模块负责优化生成的PyTorch代码，包括：
1. 合并连续的nn.Sequential块
2. 内联简单的激活函数
3. 删除未使用的层
4. 添加类型注解
5. 优化变量命名

作者: CV Studio 开发团队
日期: 2025-12-25
"""

from typing import Dict, List, Any, Optional
import re


class CodeOptimizer:
    """
    代码优化器

    职责：
    1. 优化生成的PyTorch代码
    2. 提高代码可读性和执行效率
    3. 应用代码风格优化
    """

    def __init__(self, enable_sequential: bool = True, enable_inline: bool = True):
        """
        初始化代码优化器

        Args:
            enable_sequential: 是否启用nn.Sequential块合并
            enable_inline: 是否启用激活函数内联优化
        """
        self.enable_sequential = enable_sequential
        self.enable_inline = enable_inline

    def optimize(
        self,
        code: str,
        graph=None,
        layer_defs: List[dict] = None,
        operations: List[dict] = None
    ) -> Dict[str, Any]:
        """
        优化生成的代码

        Args:
            code: 原始代码
            graph: 计算图（可选）
            layer_defs: 层定义列表（可选）
            operations: 操作列表（可选）

        Returns:
            优化结果
            {
                "code": str,              # 优化后的代码
                "optimizations": List[str],  # 应用的优化列表
                "original_size": int,     # 原始代码行数
                "optimized_size": int     # 优化后代码行数
            }
        """
        optimizations = []
        original_code = code

        # 1. 合并nn.Sequential块
        if self.enable_sequential and layer_defs and operations:
            code = self._merge_sequential_blocks(code, layer_defs, operations)
            if code != original_code:
                optimizations.append("合并nn.Sequential块")

        # 2. 内联激活函数
        if self.enable_inline:
            new_code = self._inline_activations(code)
            if new_code != code:
                code = new_code
                optimizations.append("内联激活函数")

        # 3. 添加类型注解
        new_code = self._add_type_annotations(code)
        if new_code != code:
            code = new_code
            optimizations.append("添加类型注解")

        # 4. 优化变量命名
        code = self._optimize_variable_names(code)

        return {
            "code": code,
            "optimizations": optimizations,
            "original_size": len(original_code.split("\n")),
            "optimized_size": len(code.split("\n"))
        }

    def _merge_sequential_blocks(
        self,
        code: str,
        layer_defs: List[dict],
        operations: List[dict]
    ) -> str:
        """
        合并连续的层为nn.Sequential块

        识别模式：
        - Conv2d + BatchNorm2d + ReLU
        - Linear + BatchNorm1d + ReLU
        - Conv2d + ReLU
        """
        # 识别可合并的连续块
        sequential_blocks = self._identify_sequential_blocks(layer_defs, operations)

        if not sequential_blocks:
            return code

        # 构建替换映射
        for block in sequential_blocks:
            old_lines = block["lines"]
            new_line = f"self.{block['name']} = nn.Sequential({', '.join(block['layers'])})"

            # 替换代码中的层定义
            code = code.replace(old_lines, new_line)

        return code

    def _identify_sequential_blocks(
        self,
        layer_defs: List[dict],
        operations: List[dict]
    ) -> List[dict]:
        """
        识别可以合并为Sequential的连续块

        识别规则：
        1. 连续的Conv2d -> BatchNorm2d -> ReLU
        2. 连续的Linear -> BatchNorm1d -> ReLU
        3. 连续的Conv2d -> ReLU

        Returns:
            可合并的块列表
        """
        blocks = []
        i = 0

        while i < len(layer_defs) - 2:
            # 检查Conv+BN+ReLU模式
            if (layer_defs[i]["layer_type"] == "Conv2d" and
                layer_defs[i + 1]["layer_type"] == "BatchNorm2d" and
                layer_defs[i + 2]["layer_type"] in ["ReLU", "ReLU6"]):

                block_name = f"convbnrelu_{i // 3 + 1}"
                blocks.append({
                    "name": block_name,
                    "pattern": "conv_bn_relu",
                    "layers": [
                        layer_defs[i]["name"],
                        layer_defs[i + 1]["name"],
                        layer_defs[i + 2]["name"]
                    ],
                    "indices": [i, i + 1, i + 2]
                })
                i += 3
            # 检查Conv+ReLU模式
            elif (layer_defs[i]["layer_type"] == "Conv2d" and
                  layer_defs[i + 1]["layer_type"] in ["ReLU", "ReLU6"]):

                block_name = f"convrelu_{i // 2 + 1}"
                blocks.append({
                    "name": block_name,
                    "pattern": "conv_relu",
                    "layers": [
                        layer_defs[i]["name"],
                        layer_defs[i + 1]["name"]
                    ],
                    "indices": [i, i + 1]
                })
                i += 2
            else:
                i += 1

        return blocks

    def _inline_activations(self, code: str) -> str:
        """
        内联简单的激活函数

        将 self.relu(x) 替换为 F.relu(x)
        """
        # 匹配激活函数调用模式
        activation_map = {
            "ReLU": "F.relu",
            "ReLU6": "F.relu6",
            "Sigmoid": "torch.sigmoid",
            "Tanh": "torch.tanh",
            "GELU": "F.gelu",
        }

        new_code = code
        for pytorch_name, functional_name in activation_map.items():
            # 匹配 x = self.relu1(x) 模式
            pattern = rf'x = self\.\w+{pytorch_name.lower()}.*?\(x\)'
            replacement = f'x = {functional_name}(x)'
            new_code = re.sub(pattern, replacement, new_code)

        return new_code

    def _add_type_annotations(self, code: str) -> str:
        """
        添加类型注解到forward方法
        """
        # 添加Tensor类型导入 - 只在基础torch导入后添加
        if "from torch import Tensor" not in code:
            # 使用更精确的替换，避免影响 import torch.nn 或其他相关导入
            # 只替换独立的 "import torch" 行
            code = re.sub(
                r'^import torch$',
                'import torch\nfrom torch import Tensor',
                code,
                flags=re.MULTILINE
            )

        # 修改forward方法签名
        code = re.sub(
            r'def forward\(self, x\):',
            'def forward(self, x: Tensor) -> Tensor:',
            code
        )

        return code

    def _optimize_variable_names(self, code: str) -> str:
        """
        优化变量命名，提高可读性
        """
        # 将 x0, x1, x2 等替换为更有意义的名称
        # 这里可以进一步优化，根据上下文命名
        return code


class SequentialBlock:
    """Sequential块数据类"""

    def __init__(
        self,
        name: str,
        layers: List[str],
        pattern: str,
        start_idx: int,
        end_idx: int
    ):
        self.name = name
        self.layers = layers
        self.pattern = pattern
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __repr__(self):
        return f"SequentialBlock(name={self.name}, pattern={self.pattern}, layers={len(self.layers)})"


# ==============================
# 辅助函数
# ==============================

def create_optimization_report(
    original_code: str,
    optimized_code: str,
    optimizations: List[str]
) -> str:
    """
    创建优化报告

    Args:
        original_code: 原始代码
        optimized_code: 优化后代码
        optimizations: 应用的优化列表

    Returns:
        优化报告字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append("代码优化报告")
    lines.append("=" * 60)
    lines.append(f"原始代码行数: {len(original_code.split(chr(10)))}")
    lines.append(f"优化后代码行数: {len(optimized_code.split(chr(10)))}")
    lines.append(f"减少行数: {len(original_code.split(chr(10))) - len(optimized_code.split(chr(10)))}")
    lines.append("")
    lines.append("应用的优化:")
    for opt in optimizations:
        lines.append(f"  - {opt}")
    lines.append("=" * 60)

    return "\n".join(lines)
