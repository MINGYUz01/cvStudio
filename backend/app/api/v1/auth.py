"""
认证相关API路由
"""

from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.config import settings
from app.core.security import (
    create_access_token,
    create_refresh_token,
    verify_token_type,
    verify_password,
    get_password_hash
)
from app.schemas.user import (
    UserLogin, UserResponse, Token, UserCreate,
    TokenRefresh, UserConfig, UserPasswordChange
)
from app.services.auth_service import authenticate_user, create_user
from app.dependencies import get_current_active_user
from app.models.user import User

router = APIRouter()


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    用户注册

    Args:
        user_data: 用户注册数据
        db: 数据库会话

    Returns:
        创建的用户信息

    Raises:
        HTTPException: 用户名或邮箱已存在时
    """
    user = create_user(db, user_data)
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=str(user.created_at)
    )


@router.post("/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    """
    用户登录

    Args:
        user_credentials: 用户登录凭证
        db: 数据库会话

    Returns:
        访问令牌、刷新令牌和用户信息
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

    # 创建access token和refresh token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.username, expires_delta=access_token_expires
    )

    refresh_token_expires = timedelta(days=7)
    refresh_token = create_refresh_token(
        subject=user.username, expires_delta=refresh_token_expires
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
        "refresh_token": refresh_token,
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


@router.get("/me", response_model=UserResponse)
async def get_current_user_info_endpoint(
    current_user: User = Depends(get_current_active_user)
):
    """
    获取当前用户信息

    Args:
        current_user: 当前认证用户

    Returns:
        当前用户信息
    """
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        created_at=str(current_user.created_at)
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    token_data: TokenRefresh,
    db: Session = Depends(get_db)
):
    """
    刷新访问令牌

    Args:
        token_data: 刷新令牌数据
        db: 数据库会话

    Returns:
        新的访问令牌和刷新令牌
    """
    # 验证refresh token
    username = verify_token_type(token_data.refresh_token, "refresh")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的刷新令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 获取用户信息
    from app.services.auth_service import get_user_by_username
    user = get_user_by_username(db, username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在"
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户账户已被禁用"
        )

    # 创建新的access token和refresh token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        subject=user.username, expires_delta=access_token_expires
    )

    refresh_token_expires = timedelta(days=7)
    new_refresh_token = create_refresh_token(
        subject=user.username, expires_delta=refresh_token_expires
    )

    # 构建用户响应数据
    user_response = UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=str(user.created_at)
    )

    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token,
        "token_type": "bearer",
        "user": user_response
    }


@router.get("/config", response_model=UserConfig)
async def get_user_config(
    current_user: User = Depends(get_current_active_user)
):
    """
    获取用户配置

    Args:
        current_user: 当前认证用户

    Returns:
        用户配置
    """
    config = current_user.config_dict
    return UserConfig(**config)


@router.put("/config", response_model=UserConfig)
async def update_user_config(
    config_data: UserConfig,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    更新用户配置

    Args:
        config_data: 配置数据
        current_user: 当前认证用户
        db: 数据库会话

    Returns:
        更新后的用户配置
    """
    # 更新用户配置
    current_user.config_dict = config_data.model_dump(exclude_unset=True)
    db.commit()

    return UserConfig(**current_user.config_dict)


@router.post("/change-password")
async def change_password(
    password_data: UserPasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    修改密码

    Args:
        password_data: 密码数据
        current_user: 当前认证用户
        db: 数据库会话

    Returns:
        成功消息
    """
    # 验证旧密码
    if not verify_password(password_data.old_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="旧密码错误"
        )

    # 更新密码
    current_user.password_hash = get_password_hash(password_data.new_password)
    db.commit()

    return {"message": "密码修改成功"}


@router.patch("/me", response_model=UserResponse)
async def update_user_profile(
    email: str | None = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    更新用户基本信息（目前仅支持邮箱）

    Args:
        email: 新邮箱
        db: 数据库会话
        current_user: 当前认证用户

    Returns:
        更新后的用户信息
    """
    if email:
        # 检查邮箱是否已被其他用户使用
        from app.services.auth_service import get_user_by_email
        existing_user = get_user_by_email(db, email)
        if existing_user and existing_user.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已被使用"
            )
        current_user.email = email
        db.commit()
        db.refresh(current_user)

    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        created_at=str(current_user.created_at)
    )


@router.get("/")
async def auth_root():
    """认证模块根路径"""
    return {
        "message": "认证模块",
        "version": "1.0.0",
        "endpoints": [
            "/register - 用户注册",
            "/login - JSON登录",
            "/login-form - 表单登录",
            "/refresh - 刷新令牌",
            "/me - 获取当前用户信息",
            "/config - 用户配置管理",
            "/change-password - 修改密码"
        ]
    }