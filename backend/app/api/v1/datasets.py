"""
数据集相关API路由
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def datasets_root():
    """数据集模块根路径"""
    return {"message": "数据集模块", "status": "开发中"}