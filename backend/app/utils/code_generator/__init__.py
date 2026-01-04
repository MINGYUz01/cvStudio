"""
PyTorch代码生成工具包

本包提供将前端模型图转换为可执行的PyTorch代码的功能。

核心模块：
- LayerBuilder: 构建层定义和forward方法
- CodeValidator: 验证生成的代码
- CodeGenerator: 协调整体代码生成流程
- TemplateRenderer: Jinja2模板渲染

作者: CV Studio 开发团队
日期: 2025-12-25
"""

from .generator import CodeGenerator
from .layer_builder import LayerBuilder
from .validator import CodeValidator

__all__ = [
    "CodeGenerator",
    "LayerBuilder",
    "CodeValidator",
]
