"""
模型相关API路由
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def models_root():
    """模型模块根路径"""
    return {"message": "模型模块", "status": "开发中"}