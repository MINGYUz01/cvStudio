"""
代码生成服务

本服务负责协调整个代码生成流程：
1. 整合图遍历、形状推断、代码生成
2. 错误处理和日志记录
3. 结果格式化

作者: CV Studio 开发团队
日期: 2025-12-25
"""

from typing import Dict, Any
from fastapi import HTTPException
from loguru import logger

from app.utils.graph_traversal import analyze_graph_structure
from app.utils.shape_inference import infer_shapes_from_graph
from app.utils.code_generator.generator import CodeGenerator
from app.utils.code_generator.validator import CodeValidator


class CodeGeneratorService:
    """
    代码生成服务

    职责：
    1. 协调整个代码生成流程
    2. 整合图遍历、形状推断、代码生成
    3. 错误处理和日志记录
    4. 结果格式化
    """

    def __init__(self):
        """初始化服务"""
        self.generator = CodeGenerator()
        self.validator = CodeValidator()

    async def generate_code(
        self,
        graph_json: dict,
        model_name: str = "GeneratedModel",
        template_tag: str = None
    ) -> Dict[str, Any]:
        """
        生成PyTorch代码的入口方法

        Args:
            graph_json: 前端传来的图JSON
            model_name: 模型类名
            template_tag: 模板标签（预留扩展）

        Returns:
            生成结果字典
        """
        logger.info(f"开始生成代码，模型名: {model_name}")

        try:
            # 1. 分析图结构
            logger.debug("步骤1: 分析图结构")
            structure_result = analyze_graph_structure(graph_json)

            if not structure_result["validation"]["valid"]:
                errors = structure_result["validation"]["errors"]
                warnings = structure_result["validation"]["warnings"]
                logger.error(f"图验证失败，错误: {errors}, 警告: {warnings}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "图结构验证失败",
                        "errors": errors,
                        "warnings": warnings
                    }
                )

            # 2. 推断形状
            logger.debug("步骤2: 推断张量形状")
            shape_result = infer_shapes_from_graph(
                structure_result["graph"],
                structure_result["execution_order"]
            )

            if not shape_result["validation"]["valid"]:
                errors = shape_result["validation"]["errors"]
                warnings = shape_result["validation"]["warnings"]
                logger.error(f"形状推断失败，错误: {errors}, 警告: {warnings}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "形状推断失败",
                        "errors": errors,
                        "warnings": warnings
                    }
                )

            # 3. 生成代码
            logger.debug("步骤3: 生成PyTorch代码")
            generation_result = self.generator.generate(
                graph=structure_result["graph"],
                execution_order=structure_result["execution_order"],
                shape_map=shape_result["shape_map"],
                model_name=model_name
            )

            logger.info(f"代码生成成功，层数: {generation_result['layer_count']}")

            # 4. 格式化返回结果
            return {
                "code": generation_result["code"],
                "model_name": generation_result["model_class_name"],
                "validation": generation_result["validation"],
                "metadata": {
                    "layer_count": generation_result["layer_count"],
                    "num_parameters": generation_result["metadata"]["num_parameters"],
                    "input_shape": generation_result["metadata"]["input_shape"],
                    "output_shape": generation_result["metadata"]["output_shape"],
                    "depth": generation_result["metadata"]["depth"],
                    "validation_passed": generation_result["validation"]["valid"]
                }
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"代码生成失败: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"代码生成过程中发生错误: {str(e)}"
            )

    async def validate_code(
        self,
        code: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        验证已生成的代码

        Args:
            code: 代码字符串
            model_name: 模型类名

        Returns:
            验证结果
        """
        logger.info(f"验证代码，模型名: {model_name}")

        try:
            # 使用CodeValidator的validator
            result = self.validator.validate_code(
                code=code,
                model_class_name=model_name,
                layer_defs=[],
                tensor_flow=[]
            )

            return result

        except Exception as e:
            logger.error(f"代码验证失败: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"代码验证过程中发生错误: {str(e)}"
            )
