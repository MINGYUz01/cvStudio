"""
用户相关API路由
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def users_root():
    """用户模块根路径"""
    return {"message": "用户模块", "status": "开发中"}