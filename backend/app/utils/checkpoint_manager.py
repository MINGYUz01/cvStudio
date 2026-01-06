"""
Checkpoint管理器
负责训练checkpoint的保存、加载、删除和管理
"""

from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import torch
import shutil
from datetime import datetime
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.training import Checkpoint


class CheckpointManager:
    """Checkpoint管理器"""

    def __init__(self, checkpoint_dir: str):
        """
        初始化Checkpoint管理器

        Args:
            checkpoint_dir: checkpoint存储目录
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(component="checkpoint_manager")

    def save_checkpoint(
        self,
        run_id: int,
        epoch: int,
        model_state: dict,
        optimizer_state: dict,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> str:
        """
        保存checkpoint到文件系统和数据库

        Args:
            run_id: 训练任务ID
            epoch: 当前epoch
            model_state: 模型状态字典
            optimizer_state: 优化器状态字典
            metrics: 指标字典
            is_best: 是否为最佳模型

        Returns:
            checkpoint文件路径
        """
        try:
            # 生成checkpoint文件名
            checkpoint_filename = f"checkpoint_epoch_{epoch}.pt"
            checkpoint_path = self.checkpoint_dir / checkpoint_filename

            # 准备checkpoint数据
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model_state,
                "optimizer_state_dict": optimizer_state,
                "metrics": metrics,
                "is_best": is_best,
                "timestamp": datetime.utcnow().isoformat()
            }

            # 保存到文件系统
            torch.save(checkpoint_data, checkpoint_path)

            # 保存到数据库
            self._save_to_db(
                run_id=run_id,
                epoch=epoch,
                checkpoint_path=str(checkpoint_path),
                metrics=metrics,
                is_best=is_best
            )

            self.logger.info(
                f"Checkpoint已保存: {checkpoint_path} (epoch: {epoch}, 最佳: {is_best})"
            )

            return str(checkpoint_path)

        except Exception as e:
            self.logger.error(f"保存checkpoint失败: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        加载checkpoint

        Args:
            checkpoint_path: checkpoint文件路径

        Returns:
            checkpoint数据字典
        """
        try:
            checkpoint = torch.load(checkpoint_path)
            self.logger.info(f"Checkpoint已加载: {checkpoint_path}")
            return checkpoint

        except Exception as e:
            self.logger.error(f"加载checkpoint失败: {e}")
            raise

    def get_best_checkpoint(self, run_id: int) -> Optional[str]:
        """
        获取最佳checkpoint路径

        Args:
            run_id: 训练任务ID

        Returns:
            最佳checkpoint路径，如果不存在返回None
        """
        try:
            db = SessionLocal()
            try:
                checkpoint = db.query(Checkpoint).filter(
                    Checkpoint.run_id == run_id,
                    Checkpoint.is_best == "true"
                ).order_by(Checkpoint.metric_value.desc()).first()

                if checkpoint:
                    self.logger.info(f"找到最佳checkpoint: {checkpoint.path}")
                    return checkpoint.path
                else:
                    self.logger.warning(f"未找到最佳checkpoint (run_id: {run_id})")
                    return None

            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"获取最佳checkpoint失败: {e}")
            return None

    def list_checkpoints(
        self,
        run_id: int,
        db: Optional[Session] = None
    ) -> List[Dict]:
        """
        列出所有checkpoint

        Args:
            run_id: 训练任务ID
            db: 数据库会话（可选，如果不提供则创建新会话）

        Returns:
            checkpoint信息列表
        """
        try:
            close_db = False
            if db is None:
                db = SessionLocal()
                close_db = True

            try:
                checkpoints = db.query(Checkpoint).filter(
                    Checkpoint.run_id == run_id
                ).order_by(Checkpoint.epoch.desc()).all()

                result = [
                    {
                        "id": cp.id,
                        "epoch": cp.epoch,
                        "path": cp.path,
                        "metric_value": cp.metric_value,
                        "is_best": cp.is_best == "true",
                        "file_size": cp.file_size,
                        "created_at": cp.created_at.isoformat() if cp.created_at else None
                    }
                    for cp in checkpoints
                ]

                self.logger.info(f"找到 {len(result)} 个checkpoint (run_id: {run_id})")
                return result

            finally:
                if close_db:
                    db.close()

        except Exception as e:
            self.logger.error(f"列出checkpoint失败: {e}")
            return []

    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """
        删除checkpoint文件

        Args:
            checkpoint_path: checkpoint文件路径

        Returns:
            是否删除成功
        """
        try:
            path = Path(checkpoint_path)
            if path.exists():
                path.unlink()
                self.logger.info(f"Checkpoint文件已删除: {checkpoint_path}")
                return True
            else:
                self.logger.warning(f"Checkpoint文件不存在: {checkpoint_path}")
                return False

        except Exception as e:
            self.logger.error(f"删除checkpoint文件失败: {e}")
            return False

    def delete_checkpoints_by_run(
        self,
        run_id: int,
        delete_files: bool = True
    ) -> int:
        """
        删除训练任务的所有checkpoint

        Args:
            run_id: 训练任务ID
            delete_files: 是否同时删除文件

        Returns:
            删除的checkpoint数量
        """
        try:
            db = SessionLocal()
            try:
                checkpoints = db.query(Checkpoint).filter(
                    Checkpoint.run_id == run_id
                ).all()

                deleted_count = 0

                for cp in checkpoints:
                    # 删除文件
                    if delete_files:
                        self.delete_checkpoint(cp.path)

                    # 删除数据库记录
                    db.delete(cp)
                    deleted_count += 1

                db.commit()
                self.logger.info(f"已删除 {deleted_count} 个checkpoint (run_id: {run_id})")
                return deleted_count

            except Exception as e:
                db.rollback()
                raise
            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"批量删除checkpoint失败: {e}")
            return 0

    def copy_to_weights(
        self,
        checkpoint_path: str,
        weights_dir: str,
        model_name: str
    ) -> str:
        """
        复制checkpoint到权重库

        Args:
            checkpoint_path: 源checkpoint路径
            weights_dir: 目标权重库目录
            model_name: 模型名称

        Returns:
            目标路径
        """
        try:
            weights_path = Path(weights_dir)
            weights_path.mkdir(parents=True, exist_ok=True)

            # 生成目标文件名
            dest_filename = f"{model_name}_best.pt"
            dest_path = weights_path / dest_filename

            # 复制文件
            shutil.copy2(checkpoint_path, dest_path)

            self.logger.info(
                f"Checkpoint已复制到权重库: {checkpoint_path} -> {dest_path}"
            )

            return str(dest_path)

        except Exception as e:
            self.logger.error(f"复制到权重库失败: {e}")
            raise

    def get_checkpoint_info(self, checkpoint_path: str) -> Optional[Dict]:
        """
        获取checkpoint信息（不加载整个文件）

        Args:
            checkpoint_path: checkpoint文件路径

        Returns:
            checkpoint信息字典
        """
        try:
            path = Path(checkpoint_path)
            if not path.exists():
                return None

            # 获取文件信息
            file_stat = path.stat()

            # 加载checkpoint获取元数据
            checkpoint = torch.load(checkpoint_path)

            info = {
                "path": checkpoint_path,
                "file_size": file_stat.st_size,
                "created_time": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "epoch": checkpoint.get("epoch"),
                "metrics": checkpoint.get("metrics", {}),
                "is_best": checkpoint.get("is_best", False),
                "timestamp": checkpoint.get("timestamp")
            }

            return info

        except Exception as e:
            self.logger.error(f"获取checkpoint信息失败: {e}")
            return None

    def cleanup_old_checkpoints(
        self,
        run_id: int,
        keep_best: int = 3,
        keep_last: int = 5
    ) -> int:
        """
        清理旧的checkpoint，只保留最佳和最近的几个

        Args:
            run_id: 训练任务ID
            keep_best: 保留的最佳checkpoint数量
            keep_last: 保留的最新checkpoint数量

        Returns:
            删除的checkpoint数量
        """
        try:
            db = SessionLocal()
            try:
                # 获取所有checkpoint
                all_checkpoints = db.query(Checkpoint).filter(
                    Checkpoint.run_id == run_id
                ).order_by(Checkpoint.epoch.desc()).all()

                if len(all_checkpoints) <= max(keep_best, keep_last):
                    self.logger.info(f"Checkpoint数量不多，无需清理")
                    return 0

                # 保留最佳checkpoint
                best_checkpoints = db.query(Checkpoint).filter(
                    Checkpoint.run_id == run_id,
                    Checkpoint.is_best == "true"
                ).order_by(Checkpoint.metric_value.desc()).limit(keep_best).all()

                best_ids = {cp.id for cp in best_checkpoints}

                # 保留最新的checkpoint
                latest_checkpoints = all_checkpoints[:keep_last]
                latest_ids = {cp.id for cp in latest_checkpoints}

                # 合并要保留的ID
                keep_ids = best_ids | latest_ids

                # 删除其他checkpoint
                deleted_count = 0
                for cp in all_checkpoints:
                    if cp.id not in keep_ids:
                        # 删除文件
                        self.delete_checkpoint(cp.path)
                        # 删除数据库记录
                        db.delete(cp)
                        deleted_count += 1

                db.commit()
                self.logger.info(
                    f"已清理 {deleted_count} 个旧checkpoint "
                    f"(保留最佳: {len(best_ids)}, 最新: {len(latest_ids)})"
                )

                return deleted_count

            except Exception as e:
                db.rollback()
                raise
            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"清理旧checkpoint失败: {e}")
            return 0

    def _save_to_db(
        self,
        run_id: int,
        epoch: int,
        checkpoint_path: str,
        metrics: Dict[str, float],
        is_best: bool
    ):
        """
        保存checkpoint信息到数据库

        Args:
            run_id: 训练任务ID
            epoch: 当前epoch
            checkpoint_path: checkpoint文件路径
            metrics: 指标字典
            is_best: 是否为最佳模型
        """
        try:
            db = SessionLocal()
            try:
                # 获取主要指标值（通常是val_acc或mAP）
                primary_metric = (
                    metrics.get("val_acc") or
                    metrics.get("mAP") or
                    metrics.get("train_acc") or
                    0.0
                )

                # 获取文件大小
                file_size = Path(checkpoint_path).stat().st_size

                checkpoint = Checkpoint(
                    run_id=run_id,
                    epoch=epoch,
                    metric_value=primary_metric,
                    metrics=metrics,
                    path=checkpoint_path,
                    file_size=file_size,
                    is_best="true" if is_best else "false"
                )

                db.add(checkpoint)
                db.commit()

                self.logger.debug(f"Checkpoint信息已保存到数据库: epoch {epoch}")

            except Exception as e:
                db.rollback()
                raise
            finally:
                db.close()

        except Exception as e:
            self.logger.error(f"保存checkpoint到数据库失败: {e}")
            # 数据库保存失败不影响训练继续
