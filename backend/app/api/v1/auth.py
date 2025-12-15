"""
认证相关API路由
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.config import settings
from app.core.security import create_access_token
from app.schemas.user import UserLogin, UserResponse, Token
from app.services.auth_service import authenticate_user
from app.core.security import verify_token

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """
    用户登录

    Args:
        user_credentials: 用户登录凭证
        db: 数据库会话

    Returns:
        访问令牌和用户信息
    """
    user = authenticate_user(db, user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户账户已被禁用"
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.username, expires_delta=access_token_expires
    )

    # 构建用户响应数据
    user_data = UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=str(user.created_at)
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_data
    }


@router.post("/login-form", response_model=Token)
async def login_form(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    表单方式登录（兼容OAuth2）

    Args:
        form_data: OAuth2表单数据
        db: 数据库会话

    Returns:
        访问令牌和用户信息
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户账户已被禁用"
        )

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.username, expires_delta=access_token_expires
    )

    # 构建用户响应数据
    user_data = UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=str(user.created_at)
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_data
    }


@router.get("/me")
async def get_current_user_info_endpoint(
    db: Session = Depends(get_db)
):
    """
    获取当前用户信息（简化版本，暂不需要认证）

    Args:
        db: 数据库会话

    Returns:
        当前用户信息
    """
    # 暂时返回默认用户信息用于测试
    return {
        "message": "用户信息接口正常工作",
        "note": "完整认证功能将在第2天实现"
    }


@router.get("/")
async def auth_root():
    """认证模块根路径"""
    return {
        "message": "认证模块",
        "version": "1.0.0",
        "endpoints": [
            "/login - JSON登录",
            "/login-form - 表单登录",
            "/me - 获取当前用户信息"
        ]
    }