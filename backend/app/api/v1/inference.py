"""
推理相关API路由
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def inference_root():
    """推理模块根路径"""
    return {"message": "推理模块", "status": "开发中"}