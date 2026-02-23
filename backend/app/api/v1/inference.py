"""
推理相关API路由
提供单图推理、批量推理和任务管理功能
"""

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from loguru import logger

from app.database import get_db
from app.models.inference import InferenceJob
from app.models.weight_library import WeightLibrary
from app.models.generated_code import GeneratedCode
from app.schemas.inference import (
    InferencePredictRequest,
    InferencePredictResponse,
    InferenceBatchRequest,
    BatchInferenceResponse,
    InferenceJobCreate,
    InferenceJobResponse,
    InferenceControlRequest,
    InferenceControlResponse,
    InferenceMetrics
)
from app.utils.model_loader import ModelLoader
from app.utils.inference_executor import InferenceExecutor
from app.services.inference_service import InferenceService

router = APIRouter()

# 全局模型加载器实例
_model_loader: Optional[ModelLoader] = None

# 全局推理服务实例
_inference_service: Optional[InferenceService] = None


def get_model_loader() -> ModelLoader:
    """获取模型加载器单例"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


def get_inference_service() -> InferenceService:
    """获取推理服务单例"""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service


@router.get("/")
async def inference_root():
    """推理模块根路径"""
    return {
        "message": "推理模块",
        "version": "1.0",
        "endpoints": {
            "predict": "POST /api/v1/inference/predict - 单图推理",
            "batch": "POST /api/v1/inference/batch - 批量推理",
            "jobs": "GET /api/v1/inference/jobs - 获取任务列表"
        }
    }


@router.post(
    "/predict",
    response_model=InferencePredictResponse,
    summary="单图推理",
    description="对单张图像进行推理，返回检测结果和性能指标"
)
async def predict_inference(
    request: InferencePredictRequest,
    db: Session = Depends(get_db)
) -> InferencePredictResponse:
    """
    单图推理API

    功能：
    - 从权重库加载指定权重
    - 执行推理
    - 返回检测结果和性能指标
    """
    try:
        logger.bind(component="inference_api").info(
            f"收到单图推理请求: weight_id={request.weight_id}, "
            f"image={request.image_path}"
        )

        # 从权重库查询权重信息
        weight = db.query(WeightLibrary).filter(
            WeightLibrary.id == request.weight_id
        ).first()

        if not weight:
            raise HTTPException(
                status_code=404,
                detail=f"权重不存在: weight_id={request.weight_id}"
            )

        # 检查权重文件是否存在
        weight_path = Path(weight.file_path)
        if not weight_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"权重文件不存在: {weight.file_path}"
            )

        # 获取模型加载器
        loader = get_model_loader()

        # 使用权重路径加载模型
        model_data = loader.load_model(
            str(weight_path),
            device=request.device
        )

        # 准备配置参数
        config = {
            'target_size': weight.input_size if weight.input_size else [640, 640]
        }

        # 创建推理执行器（传入任务类型和类别名称）
        executor = InferenceExecutor(
            model=model_data['model'],
            model_type=model_data['type'],
            task_type=weight.task_type,
            class_names=weight.class_names,
            device=model_data['device'],
            config=config
        )

        # 执行推理
        result = await executor.infer_single_image(
            request.image_path,
            request.confidence_threshold,
            request.iou_threshold
        )

        # 构建响应
        return InferencePredictResponse(
            task_type=weight.task_type,
            results=result['results'],
            metrics=InferenceMetrics(**result['metrics']),
            image_path=request.image_path,
            weight_id=request.weight_id
        )

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"模型或图像文件未找到: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.bind(component="inference_api").error(f"推理失败: {e}")
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@router.post(
    "/batch",
    response_model=BatchInferenceResponse,
    summary="批量推理",
    description="对多张图像进行批量推理，创建推理任务"
)
async def batch_inference(
    request: InferenceBatchRequest,
    db: Session = Depends(get_db)
) -> BatchInferenceResponse:
    """
    批量推理API

    功能：
    - 创建推理任务
    - 异步执行批量推理
    - 支持进度追踪
    """
    try:
        logger.bind(component="inference_api").info(
            f"收到批量推理请求: weight_id={request.weight_id}, "
            f"images={len(request.image_paths)}, "
            f"output_dir={request.output_dir}"
        )

        # 验证权重是否存在
        weight = db.query(WeightLibrary).filter(
            WeightLibrary.id == request.weight_id
        ).first()

        if not weight:
            raise HTTPException(
                status_code=404,
                detail=f"权重不存在: weight_id={request.weight_id}"
            )

        # 创建推理任务记录
        job = InferenceJob(
            name=f"批量推理_{weight.name}",
            model_id=request.weight_id,  # 存储weight_id到model_id字段
            input_path=",".join(request.image_paths),
            output_path=request.output_dir,
            inference_type="batch",
            confidence_threshold=request.confidence_threshold,
            iou_threshold=request.iou_threshold,
            batch_size=request.batch_size,
            device=request.device,
            total_images=len(request.image_paths),
            status="pending"
        )

        db.add(job)
        db.commit()
        db.refresh(job)

        logger.bind(component="inference_api").info(
            f"推理任务已创建: job_id={job.id}"
        )

        # TODO: 提交异步推理任务
        # 这里可以使用Celery或后台任务

        return BatchInferenceResponse(
            job_id=job.id,
            status=job.status,
            total_images=job.total_images,
            message="批量推理任务已创建，正在处理中"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.bind(component="inference_api").error(f"创建批量推理任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


@router.get(
    "/jobs",
    response_model=List[InferenceJobResponse],
    summary="获取推理任务列表",
    description="获取所有推理任务，支持分页和过滤"
)
async def get_inference_jobs(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
) -> List[InferenceJobResponse]:
    """
    获取推理任务列表

    Args:
        skip: 跳过的记录数
        limit: 返回的记录数
        status: 过滤状态（可选）

    Returns:
        推理任务列表
    """
    try:
        query = db.query(InferenceJob)

        if status:
            query = query.filter(InferenceJob.status == status)

        jobs = query.offset(skip).limit(limit).all()
        return jobs

    except Exception as e:
        logger.bind(component="inference_api").error(f"获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务列表失败: {str(e)}")


@router.get(
    "/jobs/{job_id}",
    response_model=InferenceJobResponse,
    summary="获取推理任务详情",
    description="获取指定推理任务的详细信息"
)
async def get_inference_job(
    job_id: int,
    db: Session = Depends(get_db)
) -> InferenceJobResponse:
    """
    获取单个推理任务

    Args:
        job_id: 任务ID

    Returns:
        推理任务详情
    """
    try:
        job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")

        return job

    except HTTPException:
        raise
    except Exception as e:
        logger.bind(component="inference_api").error(f"获取任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取任务详情失败: {str(e)}")


@router.post(
    "/jobs/{job_id}/control",
    response_model=InferenceControlResponse,
    summary="控制推理任务",
    description="控制推理任务（取消/暂停/恢复）"
)
async def control_inference_job(
    job_id: int,
    request: InferenceControlRequest,
    db: Session = Depends(get_db)
) -> InferenceControlResponse:
    """
    控制推理任务

    Args:
        job_id: 任务ID
        request: 控制请求

    Returns:
        控制结果
    """
    try:
        job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")

        # 检查当前状态
        if job.status in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=400,
                detail=f"任务已{job.status}，无法控制"
            )

        # 执行控制动作
        if request.action == "cancel":
            job.status = "cancelled"
        elif request.action == "pause":
            if job.status != "running":
                raise HTTPException(
                    status_code=400,
                    detail="只能暂停正在运行的任务"
                )
            job.status = "paused"
        elif request.action == "resume":
            if job.status != "paused":
                raise HTTPException(
                    status_code=400,
                    detail="只能恢复暂停的任务"
                )
            job.status = "running"

        db.commit()

        return InferenceControlResponse(
            success=True,
            action=request.action,
            job_id=job_id,
            message=f"任务已{request.action}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.bind(component="inference_api").error(f"控制任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"控制任务失败: {str(e)}")


@router.delete(
    "/jobs/{job_id}",
    summary="删除推理任务",
    description="删除指定的推理任务和结果文件"
)
async def delete_inference_job(
    job_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    删除推理任务

    Args:
        job_id: 任务ID

    Returns:
        删除结果
    """
    try:
        job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")

        # 删除输出文件（如果有）
        if job.output_path:
            import shutil
            from pathlib import Path
            output_dir = Path(job.output_path)
            if output_dir.exists():
                shutil.rmtree(output_dir)

        # 删除数据库记录
        db.delete(job)
        db.commit()

        return {
            "success": True,
            "job_id": job_id,
            "message": "任务已删除"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.bind(component="inference_api").error(f"删除任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除任务失败: {str(e)}")


@router.post("/upload", summary="上传图像", description="上传图像文件并返回路径")
async def upload_image(file: UploadFile = File(...)):
    """
    上传图像文件

    Args:
        file: 上传的文件

    Returns:
        文件路径
    """
    try:
        # 确保上传目录存在
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # 保存文件
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(file.file, buffer)

        return {
            "success": True,
            "path": str(file_path),
            "message": "文件上传成功"
        }

    except Exception as e:
        logger.bind(component="inference_api").error(f"文件上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@router.post(
    "/jobs/{job_id}/run",
    summary="运行推理任务",
    description="启动推理任务的批量推理"
)
async def run_inference_job(
    job_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    运行推理任务

    Args:
        job_id: 任务ID
        db: 数据库会话

    Returns:
        运行结果
    """
    try:
        service = get_inference_service()

        logger.bind(component="inference_api").info(f"启动推理任务: {job_id}")

        # 运行推理任务
        result = await service.run_inference_job(db, job_id)

        return result

    except ValueError as e:
        logger.bind(component="inference_api").error(f"运行推理任务失败: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.bind(component="inference_api").error(f"运行推理任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"运行推理任务失败: {str(e)}")


@router.get(
    "/jobs/{job_id}/results",
    summary="获取推理结果",
    description="获取推理任务的结果数据"
)
async def get_inference_results(
    job_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    获取推理结果

    Args:
        job_id: 任务ID
        db: 数据库会话

    Returns:
        推理结果字典
    """
    try:
        service = get_inference_service()

        results = service.get_inference_results(db, job_id)

        if not results:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")

        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.bind(component="inference_api").error(f"获取推理结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取推理结果失败: {str(e)}")


@router.get(
    "/jobs/{job_id}/download",
    summary="下载推理结果",
    description="下载推理任务的输出文件"
)
async def download_inference_results(
    job_id: int,
    format: str = "json",
    db: Session = Depends(get_db)
):
    """
    下载推理结果

    Args:
        job_id: 任务ID
        format: 文件格式（json/yolo/coco）
        db: 数据库会话

    Returns:
        文件下载响应
    """
    try:
        from fastapi.responses import FileResponse

        job = db.query(InferenceJob).filter(InferenceJob.id == job_id).first()

        if not job:
            raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")

        if job.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"任务未完成，当前状态: {job.status}"
            )

        # 构建输出文件路径
        output_path = Path(job.output_path) if job.output_path else Path(f"data/outputs/job_{job_id}")

        # 根据格式返回文件
        if format == "json":
            result_file = output_path / "results.json"
        elif format == "yolo":
            result_file = output_path / "results.txt"
        elif format == "coco":
            result_file = output_path / "results_coco.json"
        else:
            raise HTTPException(status_code=400, detail=f"不支持的格式: {format}")

        if not result_file.exists():
            raise HTTPException(status_code=404, detail="结果文件不存在")

        return FileResponse(
            path=str(result_file),
            filename=result_file.name,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.bind(component="inference_api").error(f"下载推理结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载推理结果失败: {str(e)}")


@router.post(
    "/predict-image",
    response_model=InferencePredictResponse,
    summary="图片上传+推理一体化",
    description="上传图片并直接返回推理结果"
)
async def predict_image_inference(
    weight_id: int = Form(...),
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    iou_threshold: float = Form(0.45),
    top_k: int = Form(5),
    device: Optional[str] = Form(None),
    db: Session = Depends(get_db)
) -> InferencePredictResponse:
    """
    图片上传+推理一体化API

    功能：
    - 接收上传的图片文件
    - 保存到临时目录
    - 执行推理
    - 返回推理结果
    """
    import uuid
    from fastapi.responses import FileResponse

    try:
        logger.bind(component="inference_api").info(
            f"收到图片推理请求: weight_id={weight_id}, "
            f"image={image.filename if image else 'None'}, "
            f"confidence_threshold={confidence_threshold}"
        )

        # 验证 weight_id
        if not weight_id or weight_id <= 0:
            raise HTTPException(
                status_code=422,
                detail="weight_id 必须是有效的正整数"
            )

        # 验证图片文件
        if not image or not image.filename:
            raise HTTPException(
                status_code=422,
                detail="图片文件不能为空"
            )

        # 从权重库查询权重信息
        weight = db.query(WeightLibrary).filter(
            WeightLibrary.id == weight_id
        ).first()

        if not weight:
            raise HTTPException(
                status_code=404,
                detail=f"权重不存在: weight_id={weight_id}"
            )

        # 检查权重文件是否存在
        weight_path = Path(weight.file_path)
        if not weight_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"权重文件不存在: {weight.file_path}"
            )

        # 保存上传的图片到临时目录
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)

        # 生成唯一文件名
        file_ext = Path(image.filename).suffix if image.filename else '.jpg'
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        image_path = upload_dir / unique_filename

        # 保存文件
        with open(image_path, "wb") as buffer:
            import shutil
            shutil.copyfileobj(image.file, buffer)

        # 加载模型
        import torch
        from app.utils.models.factory import ModelFactory

        # 确定类别数
        num_classes = len(weight.class_names) if weight.class_names else 10

        # 如果权重有关联的生成代码，使用 ModelFactory 加载模型
        if weight.generated_code_id:
            generated_code = db.query(GeneratedCode).filter(
                GeneratedCode.id == weight.generated_code_id
            ).first()

            if not generated_code:
                raise HTTPException(
                    status_code=404,
                    detail=f"生成的代码不存在: generated_code_id={weight.generated_code_id}"
                )

            # 检查代码文件是否存在
            code_file_path = Path(generated_code.file_path)
            if not code_file_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"生成的代码文件不存在: {generated_code.file_path}"
                )

            # 使用 ModelFactory 动态加载模型
            architecture_config = {
                "code_path": str(code_file_path),
                "model_class_name": getattr(generated_code, 'meta', {}).get('model_class_name', 'GeneratedModel')
            }

            try:
                model = ModelFactory._load_generated_model(architecture_config, num_classes)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"加载生成的模型失败: {str(e)}"
                )

            # 加载权重 state_dict
            checkpoint = torch.load(str(weight_path), map_location='cpu', weights_only=False)

            # 提取 model_state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # checkpoint 可能本身就是 state_dict
                state_dict = checkpoint

            # 加载 state_dict 到模型
            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                # 尝试非严格模式
                try:
                    model.load_state_dict(state_dict, strict=False)
                except Exception as e2:
                    raise HTTPException(
                        status_code=500,
                        detail=f"加载权重参数失败: {str(e)}, 非严格模式也失败: {str(e2)}"
                    )

            # 确定设备
            if device and device != 'auto':
                target_device = device
            else:
                target_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            model = model.to(target_device)
            model.eval()

            model_data = {
                'model': model,
                'type': 'pytorch',
                'device': target_device
            }

        else:
            # 没有关联的生成代码，尝试直接加载（适用于 ONNX 或完整模型）
            loader = get_model_loader()
            model_data = loader.load_model(
                str(weight_path),
                device=device
            )

        # 准备配置参数
        config = {
            'target_size': weight.input_size if weight.input_size else [640, 640]
        }

        # 创建推理执行器
        executor = InferenceExecutor(
            model=model_data['model'],
            model_type=model_data['type'],
            task_type=weight.task_type,
            class_names=weight.class_names,
            device=model_data['device'],
            config=config
        )

        # 执行推理
        result = await executor.infer_single_image(
            str(image_path),
            confidence_threshold,
            iou_threshold
        )

        # 构建响应
        return InferencePredictResponse(
            task_type=weight.task_type,
            results=result['results'],
            metrics=InferenceMetrics(**result['metrics']),
            image_path=str(image_path),
            weight_id=weight_id
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.bind(component="inference_api").error(f"图片推理失败: {e}")
        raise HTTPException(status_code=500, detail=f"图片推理失败: {str(e)}")

