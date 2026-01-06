"""
推理服务
提供推理任务的CRUD操作和控制逻辑
"""

from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from loguru import logger
from pathlib import Path
from datetime import datetime
import asyncio

from app.models.inference import InferenceJob
from app.utils.model_loader import ModelLoader
from app.utils.inference_executor import InferenceExecutor


class InferenceService:
    """推理服务"""

    def __init__(self):
        """初始化推理服务"""
        self.model_loader = ModelLoader()
        self.logger = logger.bind(component="inference_service")

    def create_inference_job(
        self,
        db: Session,
        name: str,
        model_id: int,
        input_path: str,
        output_path: Optional[str],
        inference_type: str,
        confidence_threshold: float,
        iou_threshold: float,
        batch_size: int,
        device: Optional[str],
        created_by: int
    ) -> InferenceJob:
        """
        创建推理任务

        Args:
            db: 数据库会话
            name: 推理任务名称
            model_id: 模型ID
            input_path: 输入路径
            output_path: 输出路径
            inference_type: 推理类型（single/batch）
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值
            batch_size: 批量大小
            device: 设备
            created_by: 创建用户ID

        Returns:
            创建的推理任务对象

        Raises:
            ValueError: 当参数无效时
        """
        try:
            # 验证推理类型
            if inference_type not in ["single", "batch"]:
                raise ValueError(f"无效的推理类型: {inference_type}")

            # 解析输入路径
            input_paths = self._parse_input_paths(input_path, inference_type)
            total_images = len(input_paths) if inference_type == "batch" else 1

            # 创建数据库记录
            inference_job = InferenceJob(
                name=name,
                model_id=model_id,
                input_path=",".join(input_paths) if inference_type == "batch" else input_path,
                output_path=output_path,
                inference_type=inference_type,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                batch_size=batch_size,
                device=device or self.model_loader.device,
                total_images=total_images,
                processed_images=0,
                status="pending",
                fps=None,
                error_message=None,
                results=None,
                start_time=None,
                end_time=None,
                created_by=created_by
            )

            db.add(inference_job)
            db.commit()
            db.refresh(inference_job)

            self.logger.info(f"推理任务已创建: {inference_job.id} - {name}")

            return inference_job

        except Exception as e:
            self.logger.error(f"创建推理任务失败: {e}")
            db.rollback()
            raise ValueError(f"创建推理任务失败: {e}")

    def _parse_input_paths(self, input_path: str, inference_type: str) -> List[str]:
        """
        解析输入路径

        Args:
            input_path: 输入路径（可能是单个文件或逗号分隔的列表）
            inference_type: 推理类型

        Returns:
            文件路径列表
        """
        if inference_type == "single":
            return [input_path]
        else:
            # 批量推理，支持逗号分隔的路径列表
            paths = input_path.split(",")
            return [p.strip() for p in paths if p.strip()]

    async def run_single_inference(
        self,
        model_id: int,
        image_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        运行单图推理（不创建Job）

        Args:
            model_id: 模型ID
            image_path: 图像路径
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值
            device: 推理设备

        Returns:
            推理结果字典

        Raises:
            ValueError: 当推理失败时
        """
        try:
            self.logger.info(
                f"执行单图推理: model_id={model_id}, image={image_path}"
            )

            # TODO: 从数据库获取模型路径
            # 这里先使用一个模拟的路径
            model_path = f"data/models/model_{model_id}.pt"

            # 加载模型
            model_data = self.model_loader.load_model(
                model_path,
                device=device
            )

            # 创建推理执行器
            config = {
                'target_size': 640
            }
            executor = InferenceExecutor(
                model=model_data['model'],
                model_type=model_data['type'],
                device=model_data['device'],
                config=config
            )

            # 执行推理
            result = await executor.infer_single_image(
                image_path,
                confidence_threshold,
                iou_threshold
            )

            self.logger.info(f"单图推理完成: {image_path}")

            return result

        except Exception as e:
            self.logger.error(f"单图推理失败: {e}")
            raise ValueError(f"单图推理失败: {e}")

    async def run_inference_job(
        self,
        db: Session,
        job_id: int
    ) -> Dict[str, Any]:
        """
        运行推理任务（批量推理）

        Args:
            db: 数据库会话
            job_id: 推理任务ID

        Returns:
            推理结果字典

        Raises:
            ValueError: 当任务不存在或推理失败时
        """
        try:
            # 获取推理任务
            job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()

            if not job:
                raise ValueError(f"推理任务不存在: {job_id}")

            # 更新状态
            job.status = "running"
            job.start_time = datetime.utcnow()
            db.commit()

            self.logger.info(f"开始推理任务: {job_id} - {job.name}")

            # TODO: 从数据库获取模型路径
            model_path = f"data/models/model_{job.model_id}.pt"

            # 加载模型
            model_data = self.model_loader.load_model(
                model_path,
                device=job.device
            )

            # 创建推理执行器
            config = {
                'target_size': 640
            }
            executor = InferenceExecutor(
                model=model_data['model'],
                model_type=model_data['type'],
                device=model_data['device'],
                config=config
            )

            # 解析图像路径
            image_paths = job.input_path.split(",") if job.inference_type == "batch" else [job.input_path]

            # 执行批量推理
            results = []
            start_time = datetime.utcnow()

            try:
                # 定义进度回调
                async def progress_callback(progress_data: Dict[str, Any]):
                    """更新推理进度"""
                    job.processed_images = progress_data['processed']
                    job.progress = progress_data['progress']
                    job.fps = progress_data.get('fps')
                    db.commit()

                # 执行推理
                results = await executor.infer_batch(
                    image_paths,
                    job.confidence_threshold,
                    job.iou_threshold,
                    progress_callback
                )

                # 计算总FPS
                total_time = (datetime.utcnow() - start_time).total_seconds()
                avg_fps = len(image_paths) / total_time if total_time > 0 else 0

                # 更新任务状态
                job.status = "completed"
                job.processed_images = len(results)
                job.fps = avg_fps
                job.end_time = datetime.utcnow()
                job.progress = 100.0
                job.results = {
                    "total_detections": sum(len(r.get('results', [])) for r in results),
                    "avg_fps": avg_fps,
                    "total_time": total_time
                }

                db.commit()

                self.logger.info(f"推理任务完成: {job_id}, FPS={avg_fps:.2f}")

                return {
                    "job_id": job_id,
                    "status": "completed",
                    "results": results,
                    "metrics": {
                        "total_images": len(image_paths),
                        "processed_images": len(results),
                        "avg_fps": avg_fps,
                        "total_time": total_time
                    }
                }

            except Exception as e:
                # 更新任务状态为失败
                job.status = "failed"
                job.error_message = str(e)
                job.end_time = datetime.utcnow()
                db.commit()

                self.logger.error(f"推理任务失败: {job_id}, 错误: {e}")
                raise ValueError(f"推理任务失败: {e}")

        except Exception as e:
            self.logger.error(f"运行推理任务失败: {e}")
            db.rollback()
            raise ValueError(f"运行推理任务失败: {e}")

    def control_inference(
        self,
        db: Session,
        job_id: int,
        action: str
    ) -> Dict[str, Any]:
        """
        控制推理任务

        Args:
            db: 数据库会话
            job_id: 推理任务ID
            action: 控制动作 (cancel|pause|resume)

        Returns:
            操作结果字典

        Raises:
            ValueError: 当任务不存在或操作无效时
        """
        try:
            job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()

            if not job:
                raise ValueError(f"推理任务不存在: {job_id}")

            # 验证操作
            valid_actions = ["cancel", "pause", "resume"]
            if action not in valid_actions:
                raise ValueError(f"无效的操作: {action}")

            # 检查当前状态
            if job.status in ["completed", "failed", "cancelled"]:
                raise ValueError(f"任务已{job.status}，无法控制")

            # 执行控制动作
            if action == "cancel":
                job.status = "cancelled"
                job.end_time = datetime.utcnow()
            elif action == "pause":
                if job.status != "running":
                    raise ValueError("只能暂停正在运行的任务")
                job.status = "paused"
            elif action == "resume":
                if job.status != "paused":
                    raise ValueError("只能恢复暂停的任务")
                job.status = "running"

            db.commit()

            self.logger.info(f"推理控制: {job_id} - {action}")

            return {
                "success": True,
                "action": action,
                "job_id": job_id,
                "message": f"任务已{action}"
            }

        except Exception as e:
            self.logger.error(f"控制推理任务失败: {e}")
            db.rollback()
            raise ValueError(f"控制推理任务失败: {e}")

    def get_inference_jobs(
        self,
        db: Session,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[InferenceJob]:
        """
        获取推理任务列表

        Args:
            db: 数据库会话
            skip: 跳过记录数
            limit: 限制返回数量
            status: 状态过滤

        Returns:
            推理任务列表
        """
        try:
            query = db.query(InferenceJob)

            if status:
                query = query.filter(InferenceJob.status == status)

            jobs = query.order_by(
                InferenceJob.created_at.desc()
            ).offset(skip).limit(limit).all()

            self.logger.debug(f"获取推理任务列表: {len(jobs)} 条记录")

            return jobs

        except Exception as e:
            self.logger.error(f"获取推理任务列表失败: {e}")
            return []

    def get_inference_job(
        self,
        db: Session,
        job_id: int
    ) -> Optional[InferenceJob]:
        """
        获取单个推理任务

        Args:
            db: 数据库会话
            job_id: 推理任务ID

        Returns:
            推理任务对象，如果不存在返回None
        """
        try:
            job = db.query(InferenceJob).filter(
                InferenceJob.id == job_id
            ).first()

            return job

        except Exception as e:
            self.logger.error(f"获取推理任务失败: {e}")
            return None

    def update_inference_job(
        self,
        db: Session,
        job_id: int,
        name: Optional[str] = None
    ) -> Optional[InferenceJob]:
        """
        更新推理任务（重命名等）

        Args:
            db: 数据库会话
            job_id: 推理任务ID
            name: 新名称（可选）

        Returns:
            更新后的推理任务对象，如果不存在返回None
        """
        try:
            job = db.query(InferenceJob).filter(
                InferenceJob.id == job_id
            ).first()

            if not job:
                return None

            # 更新字段
            if name:
                job.name = name

            db.commit()
            db.refresh(job)

            self.logger.info(f"推理任务已更新: {job_id}")

            return job

        except Exception as e:
            self.logger.error(f"更新推理任务失败: {e}")
            db.rollback()
            return None

    def delete_inference_job(
        self,
        db: Session,
        job_id: int
    ) -> bool:
        """
        删除推理任务

        Args:
            db: 数据库会话
            job_id: 推理任务ID

        Returns:
            是否删除成功
        """
        try:
            job = db.query(InferenceJob).filter(
                InferenceJob.id == job_id
            ).first()

            if not job:
                return False

            # 删除输出文件（如果有）
            if job.output_path:
                import shutil
                output_dir = Path(job.output_path)
                if output_dir.exists():
                    shutil.rmtree(output_dir)
                    self.logger.info(f"已删除输出目录: {output_dir}")

            # 删除数据库记录
            db.delete(job)
            db.commit()

            self.logger.info(f"推理任务已删除: {job_id}")

            return True

        except Exception as e:
            self.logger.error(f"删除推理任务失败: {e}")
            db.rollback()
            return False

    def get_inference_results(
        self,
        db: Session,
        job_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        获取推理结果

        Args:
            db: 数据库会话
            job_id: 推理任务ID

        Returns:
            推理结果字典，如果任务不存在返回None
        """
        try:
            job = db.query(InferenceJob).filter(
                InferenceJob.id == job_id
            ).first()

            if not job:
                return None

            return {
                "job_id": job_id,
                "status": job.status,
                "results": job.results,
                "metrics": {
                    "fps": job.fps,
                    "total_images": job.total_images,
                    "processed_images": job.processed_images,
                    "progress": job.progress
                },
                "error_message": job.error_message
            }

        except Exception as e:
            self.logger.error(f"获取推理结果失败: {e}")
            return None
