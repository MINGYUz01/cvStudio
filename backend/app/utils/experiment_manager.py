"""
实验目录管理器
负责管理每个训练任务的文件存储，包括配置、指标、日志和checkpoint
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger


class ExperimentManager:
    """
    实验目录管理器

    为每个训练任务创建独立的目录结构，用于存储：
    - config.json: 训练配置
    - train.log: 训练日志
    - metrics.json: 每个epoch的指标
    - checkpoints/: checkpoint文件
    - outputs/: 训练输出（图表等）
    """

    def __init__(self, experiment_id: int, base_dir: str = "data/experiments"):
        """
        初始化实验管理器

        Args:
            experiment_id: 实验/训练任务ID
            base_dir: 实验根目录
        """
        self.experiment_id = experiment_id
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / f"exp_{experiment_id}"

        # 子目录
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.outputs_dir = self.experiment_dir / "outputs"

        # 文件路径
        self.config_file = self.experiment_dir / "config.json"
        self.metrics_file = self.experiment_dir / "metrics.json"
        self.log_file = self.experiment_dir / "train.log"

    def create_experiment_dir(self) -> Path:
        """
        创建实验目录结构

        Returns:
            实验目录路径
        """
        try:
            # 创建所有必要的目录
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.outputs_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"实验目录已创建: {self.experiment_dir}")
            return self.experiment_dir
        except Exception as e:
            logger.error(f"创建实验目录失败: {e}")
            raise

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        保存训练配置到config.json

        Args:
            config: 训练配置字典
        """
        try:
            # 确保目录存在
            self.create_experiment_dir()

            # 添加元数据
            config_with_meta = {
                "experiment_id": self.experiment_id,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "config": config
            }

            # 写入配置文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_with_meta, f, indent=2, ensure_ascii=False)

            logger.debug(f"配置已保存: {self.config_file}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise

    def load_config(self) -> Optional[Dict[str, Any]]:
        """
        从config.json加载训练配置

        Returns:
            配置字典，如果文件不存在返回None
        """
        try:
            if not self.config_file.exists():
                return None

            with open(self.config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return data.get("config", data)
        except Exception as e:
            logger.error(f"加载配置失败: {e}")
            return None

    def save_metrics(self, metrics_entry: Dict[str, Any]) -> None:
        """
        保存或追加指标到metrics.json

        Args:
            metrics_entry: 单个指标条目，包含epoch、timestamp和各种指标值
        """
        try:
            # 确保目录存在
            self.create_experiment_dir()

            # 读取现有指标
            existing_metrics = []
            if self.metrics_file.exists():
                try:
                    with open(self.metrics_file, 'r', encoding='utf-8') as f:
                        existing_metrics = json.load(f)
                except json.JSONDecodeError:
                    existing_metrics = []

            # 检查是否已存在相同epoch的指标，存在则更新，不存在则追加
            epoch = metrics_entry.get("epoch")
            if epoch is not None:
                # 查找并更新或追加
                found = False
                for i, m in enumerate(existing_metrics):
                    if m.get("epoch") == epoch:
                        existing_metrics[i] = metrics_entry
                        found = True
                        break
                if not found:
                    existing_metrics.append(metrics_entry)

            # 写回文件
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(existing_metrics, f, indent=2, ensure_ascii=False)

            logger.debug(f"指标已保存: epoch={epoch}, total={len(existing_metrics)}")
        except Exception as e:
            logger.error(f"保存指标失败: {e}")
            raise

    def load_metrics(self) -> List[Dict[str, Any]]:
        """
        从metrics.json加载所有指标

        Returns:
            指标列表，如果文件不存在返回空列表
        """
        try:
            if not self.metrics_file.exists():
                return []

            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                metrics = json.load(f)

            return metrics if isinstance(metrics, list) else []
        except Exception as e:
            logger.error(f"加载指标失败: {e}")
            return []

    def append_log(self, log_entry: Dict[str, Any]) -> None:
        """
        追加日志到train.log

        Args:
            log_entry: 日志条目，包含level、message、source、timestamp
        """
        try:
            # 确保目录存在
            self.create_experiment_dir()

            # 格式化日志行
            timestamp = log_entry.get("timestamp", datetime.utcnow().isoformat())
            level = log_entry.get("level", "INFO")
            message = log_entry.get("message", "")
            source = log_entry.get("source", "trainer")

            log_line = f"[{timestamp}] [{level}] [{source}] {message}\n"

            # 追加到日志文件
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_line)

        except Exception as e:
            logger.error(f"写入日志失败: {e}")

    def get_log_content(self, num_lines: int = 100) -> List[str]:
        """
        获取日志文件内容

        Args:
            num_lines: 返回的行数，-1表示全部

        Returns:
            日志行列表
        """
        try:
            if not self.log_file.exists():
                return []

            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if num_lines == -1:
                return lines
            return lines[-num_lines:]
        except Exception as e:
            logger.error(f"读取日志失败: {e}")
            return []

    def get_checkpoint_path(self, epoch: Optional[int] = None, best: bool = False) -> Path:
        """
        获取checkpoint文件路径

        Args:
            epoch: epoch编号，None表示最新
            best: 是否为最佳模型

        Returns:
            checkpoint文件路径
        """
        if best:
            return self.checkpoints_dir / "best_model.pt"
        elif epoch is not None:
            return self.checkpoints_dir / f"epoch_{epoch}.pt"
        else:
            return self.checkpoints_dir / "latest.pt"

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出所有checkpoint文件

        Returns:
            checkpoint信息列表
        """
        checkpoints = []
        try:
            if not self.checkpoints_dir.exists():
                return checkpoints

            for file in self.checkpoints_dir.glob("*.pt"):
                stat = file.stat()
                checkpoints.append({
                    "name": file.name,
                    "path": str(file),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

            # 排序：best_model.pt在前，然后按epoch编号
            checkpoints.sort(key=lambda x: (
                0 if x["name"] == "best_model.pt" else
                0 if x["name"] == "latest.pt" else 1,
                x["name"]
            ))

        except Exception as e:
            logger.error(f"列出checkpoint失败: {e}")

        return checkpoints

    def get_files_list(self) -> List[Dict[str, Any]]:
        """
        获取实验目录下所有文件列表

        Returns:
            文件信息列表
        """
        files = []
        try:
            if not self.experiment_dir.exists():
                return files

            for file in self.experiment_dir.rglob("*"):
                if file.is_file():
                    relative_path = file.relative_to(self.experiment_dir)
                    stat = file.stat()
                    files.append({
                        "name": file.name,
                        "path": str(relative_path),
                        "full_path": str(file),
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

            # 按路径排序
            files.sort(key=lambda x: x["path"])

        except Exception as e:
            logger.error(f"获取文件列表失败: {e}")

        return files

    def delete_experiment_dir(self) -> bool:
        """
        删除整个实验目录

        Returns:
            是否删除成功
        """
        try:
            if self.experiment_dir.exists():
                shutil.rmtree(self.experiment_dir)
                logger.info(f"实验目录已删除: {self.experiment_dir}")
                return True
            return False
        except Exception as e:
            logger.error(f"删除实验目录失败: {e}")
            return False

    def get_experiment_info(self) -> Dict[str, Any]:
        """
        获取实验信息摘要

        Returns:
            实验信息字典
        """
        info = {
            "experiment_id": self.experiment_id,
            "exists": self.experiment_dir.exists(),
            "path": str(self.experiment_dir) if self.experiment_dir.exists() else None,
            "has_config": self.config_file.exists(),
            "has_metrics": self.metrics_file.exists(),
            "has_log": self.log_file.exists(),
            "checkpoint_count": len(self.list_checkpoints()),
            "total_size": 0
        }

        # 计算总大小
        if self.experiment_dir.exists():
            try:
                for file in self.experiment_dir.rglob("*"):
                    if file.is_file():
                        info["total_size"] += file.stat().st_size
            except Exception:
                pass

        return info

    @classmethod
    def from_experiment_id(cls, experiment_id: int, base_dir: str = "data/experiments") -> 'ExperimentManager':
        """
        根据实验ID创建管理器实例

        Args:
            experiment_id: 实验/训练任务ID
            base_dir: 实验根目录

        Returns:
            ExperimentManager实例
        """
        return cls(experiment_id, base_dir)

    @classmethod
    def list_all_experiments(cls, base_dir: str = "data/experiments") -> List[Dict[str, Any]]:
        """
        列出所有实验

        Args:
            base_dir: 实验根目录

        Returns:
            实验信息列表
        """
        experiments = []
        base_path = Path(base_dir)

        try:
            if not base_path.exists():
                return experiments

            for exp_dir in base_path.glob("exp_*"):
                try:
                    # 提取ID
                    exp_id_str = exp_dir.name.replace("exp_", "")
                    exp_id = int(exp_id_str)

                    manager = cls(exp_id, base_dir)
                    info = manager.get_experiment_info()
                    experiments.append(info)

                except (ValueError, Exception):
                    continue

            # 按ID排序
            experiments.sort(key=lambda x: x["experiment_id"])

        except Exception as e:
            logger.error(f"列出实验失败: {e}")

        return experiments
