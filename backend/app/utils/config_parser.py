"""
训练配置解析器
将前端发送的训练配置转换为标准化的后端训练配置
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger


class TrainingConfigParser:
    """训练配置解析器"""

    # 任务类型对应的必需参数
    TASK_TYPE_PARAMS = {
        "detection": {
            "required": ["epochs", "batch_size", "image_size", "optimizer"],
            "optional": [
                "learning_rate", "weight_decay", "momentum", "save_period",
                "conf_thres", "iou_thres", "max_det"
            ]
        },
        "classification": {
            "required": ["epochs", "batch_size", "optimizer"],
            "optional": [
                "learning_rate", "weight_decay", "scheduler", "label_smoothing",
                "dropout_rate"
            ]
        },
        "segmentation": {
            "required": ["epochs", "batch_size", "image_size", "optimizer"],
            "optional": [
                "learning_rate", "weight_decay", "loss_type", "dice_weight"
            ]
        }
    }

    def __init__(self):
        """初始化配置解析器"""
        self.logger = logger.bind(component="config_parser")

    def parse_frontend_config(
        self,
        config: Dict[str, Any],
        model_arch: Dict[str, Any],
        dataset_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        解析前端传来的训练配置

        Args:
            config: 前端训练配置
            model_arch: 模型架构信息
            dataset_info: 数据集信息

        Returns:
            标准化的训练配置字典

        Raises:
            ValueError: 当配置参数无效时
        """
        try:
            task_type = config.get("task_type", "detection")

            # 验证必需参数
            self._validate_config(config, task_type)

            # 构建完整配置
            parsed_config = {
                # 基础配置
                "task_type": task_type,
                "experiment_name": config.get("name", "unnamed"),
                "device": config.get("device", "cpu"),
                "description": config.get("description", ""),

                # 训练参数
                "epochs": config["epochs"],
                "batch_size": config["batch_size"],
                "learning_rate": config.get("learning_rate", 0.001),
                "optimizer": config["optimizer"],
                "weight_decay": config.get("weight_decay", 0.0005),
                "momentum": config.get("momentum", 0.9),  # SGD使用

                # 学习率调度器
                "scheduler": config.get("scheduler", "step"),
                "scheduler_params": config.get("scheduler_params", {}),

                # 数据配置
                "dataset_path": dataset_info.get("path", ""),
                "dataset_format": dataset_info.get("format", "unknown"),
                "num_classes": dataset_info.get("num_classes", 80),
                "image_size": config.get("image_size", 640),

                # 模型配置
                "model_architecture": model_arch,
                "input_channels": 3,

                # Checkpoint配置
                "save_period": config.get("save_period", 10),
                "checkpoint_dir": f"data/checkpoints/{config.get('name', 'unnamed')}",

                # 数据增强配置
                "augmentation": config.get("augmentation", {}),

                # 任务特定参数
                "task_specific_params": self._extract_task_specific_params(
                    config, task_type
                ),
            }

            self.logger.info(f"配置解析成功: {parsed_config['experiment_name']}")
            return parsed_config

        except Exception as e:
            self.logger.error(f"配置解析失败: {e}")
            raise ValueError(f"配置解析失败: {e}")

    def _validate_config(self, config: Dict[str, Any], task_type: str) -> None:
        """
        验证配置参数

        Args:
            config: 配置字典
            task_type: 任务类型

        Raises:
            ValueError: 当缺少必需参数时
        """
        params_def = self.TASK_TYPE_PARAMS.get(task_type, {})
        required = params_def.get("required", [])

        missing = [p for p in required if p not in config or config[p] is None]
        if missing:
            raise ValueError(
                f"任务类型'{task_type}'缺少必需参数: {', '.join(missing)}"
            )

        # 验证参数值的有效性
        if "epochs" in config and config["epochs"] <= 0:
            raise ValueError("epochs必须大于0")

        if "batch_size" in config and config["batch_size"] <= 0:
            raise ValueError("batch_size必须大于0")

        if "learning_rate" in config and config["learning_rate"] <= 0:
            raise ValueError("learning_rate必须大于0")

        self.logger.debug(f"配置验证通过: {task_type}")

    def _extract_task_specific_params(
        self,
        config: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """
        提取任务特定参数

        Args:
            config: 配置字典
            task_type: 任务类型

        Returns:
            任务特定参数字典
        """
        task_params = {}

        if task_type == "detection":
            # 检测任务特定参数
            task_params = {
                "conf_thres": config.get("conf_thres", 0.25),
                "iou_thres": config.get("iou_thres", 0.45),
                "max_det": config.get("max_det", 300),
            }

        elif task_type == "classification":
            # 分类任务特定参数
            task_params = {
                "label_smoothing": config.get("label_smoothing", 0.0),
                "dropout_rate": config.get("dropout_rate", 0.0),
            }

        elif task_type == "segmentation":
            # 分割任务特定参数
            task_params = {
                "loss_type": config.get("loss_type", "ce"),
                "dice_weight": config.get("dice_weight", 0.5),
            }

        return task_params

    def generate_training_script(
        self,
        config: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        生成训练脚本（可选功能）

        Args:
            config: 解析后的配置
            output_path: 输出脚本路径

        Returns:
            生成的脚本内容
        """
        script_content = f'''#!/usr/bin/env python3
"""
自动生成的训练脚本
实验名称: {config['experiment_name']}
任务类型: {config['task_type']}
"""

import sys
import torch
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """主训练函数"""
    print(f"开始训练: {config['experiment_name']}")
    print(f"任务类型: {config['task_type']}")
    print(f"设备: {config['device']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch Size: {config['batch_size']}")

    # TODO: 实现实际的训练逻辑
    # 这里应该调用训练器或者执行训练脚本

    return {{
        "status": "completed",
        "final_epoch": {config['epochs']},
        "best_metric": 0.0
    }}

if __name__ == "__main__":
    result = main()
    print(f"训练完成: {{result}}")
'''

        # 保存脚本
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(script_content, encoding='utf-8')

        self.logger.info(f"训练脚本已生成: {output_path}")
        return script_content

    def validate_hyperparams(
        self,
        hyperparams: Dict[str, Any],
        task_type: str
    ) -> Dict[str, Any]:
        """
        验证并修正超参数

        Args:
            hyperparams: 超参数字典
            task_type: 任务类型

        Returns:
            修正后的超参数字典
        """
        validated = hyperparams.copy()

        # 修正学习率范围
        if "learning_rate" in validated:
            lr = validated["learning_rate"]
            if lr < 1e-6:
                self.logger.warning(f"学习率过小({lr})，调整为1e-6")
                validated["learning_rate"] = 1e-6
            elif lr > 1.0:
                self.logger.warning(f"学习率过大({lr})，调整为1.0")
                validated["learning_rate"] = 1.0

        # 修正权重衰减范围
        if "weight_decay" in validated:
            wd = validated["weight_decay"]
            if wd < 0:
                self.logger.warning(f"权重衰减不能为负({wd})，调整为0")
                validated["weight_decay"] = 0.0
            elif wd > 0.1:
                self.logger.warning(f"权重衰减过大({wd})，调整为0.1")
                validated["weight_decay"] = 0.1

        # 修正batch size
        if "batch_size" in validated:
            bs = validated["batch_size"]
            if bs < 1:
                raise ValueError(f"batch_size必须大于0，当前值: {bs}")
            if bs > 256:
                self.logger.warning(f"batch_size过大({bs})，可能导致内存不足")

        return validated

    def get_optimizer_config(
        self,
        optimizer_name: str,
        learning_rate: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        获取优化器配置

        Args:
            optimizer_name: 优化器名称
            learning_rate: 学习率
            **kwargs: 其他参数

        Returns:
            优化器配置字典
        """
        config = {
            "type": optimizer_name.lower(),
            "lr": learning_rate,
        }

        if optimizer_name.lower() == "sgd":
            config.update({
                "momentum": kwargs.get("momentum", 0.9),
                "weight_decay": kwargs.get("weight_decay", 0.0005),
            })

        elif optimizer_name.lower() == "adam":
            config.update({
                "betas": kwargs.get("betas", (0.9, 0.999)),
                "weight_decay": kwargs.get("weight_decay", 0.0),
                "eps": kwargs.get("eps", 1e-8),
            })

        elif optimizer_name.lower() == "adamw":
            config.update({
                "betas": kwargs.get("betas", (0.9, 0.999)),
                "weight_decay": kwargs.get("weight_decay", 0.01),
                "eps": kwargs.get("eps", 1e-8),
            })

        return config

    def get_scheduler_config(
        self,
        scheduler_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        获取学习率调度器配置

        Args:
            scheduler_name: 调度器名称
            **kwargs: 其他参数

        Returns:
            调度器配置字典
        """
        config = {
            "type": scheduler_name.lower(),
        }

        if scheduler_name.lower() == "step":
            config.update({
                "step_size": kwargs.get("step_size", 30),
                "gamma": kwargs.get("gamma", 0.1),
            })

        elif scheduler_name.lower() == "cosine":
            config.update({
                "T_max": kwargs.get("T_max", 100),
                "eta_min": kwargs.get("eta_min", 0),
            })

        elif scheduler_name.lower() == "reduceuronplateau":
            config.update({
                "mode": kwargs.get("mode", "min"),
                "factor": kwargs.get("factor", 0.1),
                "patience": kwargs.get("patience", 10),
                "threshold": kwargs.get("threshold", 1e-4),
            })

        elif scheduler_name.lower() == "exponential":
            config.update({
                "gamma": kwargs.get("gamma", 0.95),
            })

        return config
