"""
训练相关API路由
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def training_root():
    """训练模块根路径"""
    return {"message": "训练模块", "status": "开发中"}