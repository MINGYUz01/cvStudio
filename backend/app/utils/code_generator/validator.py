"""
代码验证器

本模块负责验证生成的PyTorch代码的正确性：
1. AST语法检查
2. 参数完整性验证
3. 动态导入验证（可执行性测试）
4. 前向传播测试

作者: CV Studio 开发团队
日期: 2025-12-25
"""

import ast
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
import traceback

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class CodeValidator:
    """
    代码验证器

    职责：
    1. AST语法检查
    2. 动态导入验证（可执行性测试）
    3. 参数完整性验证
    4. 前向传播测试（使用随机输入）
    """

    def validate_code(
        self,
        code: str,
        model_class_name: str,
        layer_defs: List[dict],
        tensor_flow: List[dict],
        input_shape: tuple = None
    ) -> Dict[str, Any]:
        """
        综合验证生成的代码

        Args:
            code: 要验证的代码字符串
            model_class_name: 模型类名
            layer_defs: 层定义列表
            tensor_flow: 张量流动信息列表
            input_shape: 输入张量形状（用于前向传播测试）

        Returns:
            {
                "valid": bool,
                "syntax_valid": bool,
                "executable": bool,
                "parameters_valid": bool,
                "forward_pass_success": bool,
                "errors": List[str],
                "warnings": List[str],
                "test_results": {...}
            }
        """
        errors = []
        warnings = []

        # 1. AST语法检查
        syntax_valid, syntax_errors = self._check_syntax(code)
        if not syntax_valid:
            errors.extend(syntax_errors)
            return {
                "valid": False,
                "syntax_valid": False,
                "executable": False,
                "parameters_valid": False,
                "forward_pass_success": False,
                "errors": errors,
                "warnings": warnings,
                "test_results": {}
            }

        # 2. 参数完整性验证
        params_valid, params_errors = self._validate_parameters(layer_defs)
        if not params_valid:
            errors.extend(params_errors)

        # 3. 可执行性验证（动态导入）
        executable, exec_errors = self._check_executability(
            code, model_class_name
        )
        if not executable:
            errors.extend(exec_errors)

        # 4. 前向传播测试（如果可执行）
        forward_success = False
        test_results = {}

        if executable and TORCH_AVAILABLE:
            forward_success, test_results, forward_errors = \
                self._test_forward_pass(code, model_class_name, tensor_flow, input_shape)
            if not forward_success:
                errors.extend(forward_errors)

        valid = (
            syntax_valid and
            params_valid and
            executable and
            forward_success
        )

        # 生成警告
        if not params_valid:
            warnings.append("部分参数可能不完整，建议检查层定义")

        return {
            "valid": valid,
            "syntax_valid": syntax_valid,
            "executable": executable,
            "parameters_valid": params_valid,
            "forward_pass_success": forward_success,
            "errors": errors,
            "warnings": warnings,
            "test_results": test_results
        }

    def _check_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """
        使用AST进行语法检查

        Args:
            code: 要检查的代码字符串

        Returns:
            (valid, errors) - 是否有效和错误列表
        """
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            errors = [
                f"语法错误（行{e.lineno}）: {e.msg}",
            ]
            if e.text:
                errors.append(f"  {e.text.strip()}")
            return False, errors
        except Exception as e:
            return False, [f"解析错误: {str(e)}"]

    def _validate_parameters(
        self,
        layer_defs: List[dict]
    ) -> Tuple[bool, List[str]]:
        """
        验证层参数的完整性

        检查项：
        1. 必需参数是否存在
        2. 参数值是否合理（正数、合理范围）
        3. 通道数/特征数是否匹配

        Args:
            layer_defs: 层定义列表

        Returns:
            (valid, errors) - 是否有效和错误列表
        """
        errors = []

        for i, layer_def in enumerate(layer_defs):
            # 支持LayerDefinition对象或字典
            if hasattr(layer_def, 'layer_type'):
                layer_type = layer_def.layer_type
                params = layer_def.params if hasattr(layer_def, 'params') else {}
            else:
                layer_type = layer_def.get("layer_type", "")
                params = layer_def.get("params", {})

            # 检查必需参数
            required = {
                "Conv2d": ["in_channels", "out_channels"],
                "Linear": ["in_features", "out_features"],
                "BatchNorm2d": ["num_features"],
                "MaxPool2d": ["kernel_size"],
                "AvgPool2d": ["kernel_size"],
                "AdaptiveAvgPool2d": ["output_size"],
            }

            if layer_type in required:
                for param in required[layer_type]:
                    if param not in params or params[param] is None:
                        errors.append(
                            f"层{i+1}({layer_type})缺少必需参数: {param}"
                        )
                    elif isinstance(params[param], (int, float)) and params[param] <= 0:
                        errors.append(
                            f"层{i+1}({layer_type})参数{param}值无效: {params[param]}"
                        )

        return len(errors) == 0, errors

    def _check_executability(
        self,
        code: str,
        model_class_name: str
    ) -> Tuple[bool, List[str]]:
        """
        检查代码是否可以成功导入

        使用临时文件和importlib进行动态导入测试

        Args:
            code: 要检查的代码
            model_class_name: 模型类名

        Returns:
            (valid, errors) - 是否可执行和错误列表
        """
        errors = []

        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_path = f.name

            # 动态导入
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "temp_model",
                temp_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # 检查模型类是否存在
                if not hasattr(module, model_class_name):
                    errors.append(f"模型类 '{model_class_name}' 未找到")
                    return False, errors

                return True, []
            else:
                errors.append("无法创建模块规范")
                return False, errors

        except ImportError as e:
            errors.append(f"导入错误: {str(e)}")
            return False, errors
        except Exception as e:
            errors.append(f"执行错误: {str(e)}")
            return False, errors
        finally:
            # 清理临时文件
            try:
                Path(temp_path).unlink()
            except:
                pass

    def _test_forward_pass(
        self,
        code: str,
        model_class_name: str,
        tensor_flow: List[dict],
        input_shape: tuple = None
    ) -> Tuple[bool, dict, List[str]]:
        """
        测试模型的前向传播

        1. 实例化模型
        2. 创建随机输入张量
        3. 执行前向传播
        4. 检查输出形状

        Args:
            code: 模型代码
            model_class_name: 模型类名
            tensor_flow: 张量流动信息
            input_shape: 输入形状（可选）

        Returns:
            (success, test_results, errors) - 是否成功、测试结果和错误列表
        """
        errors = []
        test_results = {}

        try:
            # 导入模型类
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.py',
                delete=False,
                encoding='utf-8'
            ) as f:
                f.write(code)
                temp_path = f.name

            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "temp_model",
                temp_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            ModelClass = getattr(module, model_class_name)

            # 实例化模型
            model = ModelClass()
            model.eval()  # 设置为评估模式

            # 确定输入形状
            if input_shape is None:
                # 默认使用 (1, 3, 224, 224)
                input_shape = (1, 3, 224, 224)

            # 创建随机输入
            dummy_input = torch.randn(*input_shape)

            # 前向传播
            with torch.no_grad():
                output = model(dummy_input)

            # 记录测试结果
            test_results = {
                "input_shape": list(input_shape),
                "output_shape": list(output.shape),
                "output_mean": float(output.mean().item()),
                "output_std": float(output.std().item()),
                "num_parameters": sum(
                    p.numel() for p in model.parameters()
                ),
                "model_size_mb": sum(
                    p.numel() * p.element_size() for p in model.parameters()
                ) / (1024 * 1024)
            }

            return True, test_results, []

        except RuntimeError as e:
            errors.append(f"运行时错误: {str(e)}")
            if "shape" in str(e).lower() or "size" in str(e).lower():
                errors.append("  提示：可能是层之间的形状不匹配")
            return False, {}, errors
        except Exception as e:
            errors.append(f"测试错误: {str(e)}")
            errors.append(f"  详细信息: {traceback.format_exc()}")
            return False, {}, errors
        finally:
            try:
                Path(temp_path).unlink()
            except:
                pass

    def validate_code_only_syntax(
        self,
        code: str
    ) -> Dict[str, Any]:
        """
        仅进行语法验证的轻量级方法

        用于快速验证代码语法，不执行前向传播测试

        Args:
            code: 要验证的代码

        Returns:
            {
                "valid": bool,
                "syntax_valid": bool,
                "errors": List[str]
            }
        """
        syntax_valid, syntax_errors = self._check_syntax(code)

        return {
            "valid": syntax_valid,
            "syntax_valid": syntax_valid,
            "errors": syntax_errors
        }
