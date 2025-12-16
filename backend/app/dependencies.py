"""
FastAPI依赖注入
"""

from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError
from app.database import get_db
from app.core.config import settings
from app.core.security import verify_token
from app.models.user import User
from app.services.auth_service import get_user_by_username

# HTTP Bearer 认证方案
security = HTTPBearer()

# OAuth2 密码流（用于表单登录）
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login-form")


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    获取当前用户ID（简化版本，返回用户名）

    Args:
        credentials: HTTP认证凭证

    Returns:
        用户ID字符串

    Raises:
        HTTPException: 认证失败时
    """
    token = credentials.credentials
    user_id = verify_token(token)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """
    获取当前认证用户

    Args:
        token: JWT令牌
        db: 数据库会话

    Returns:
        当前用户对象

    Raises:
        HTTPException: 认证失败时抛出401异常
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无法验证凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        username = verify_token(token)
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = get_user_by_username(db, username)
    if user is None:
        raise credentials_exception

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    获取当前活跃用户

    Args:
        current_user: 当前用户

    Returns:
        当前活跃用户对象

    Raises:
        HTTPException: 用户未激活时抛出400异常
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户账户未激活"
        )
    return current_user


def get_current_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    获取当前超级用户

    Args:
        current_user: 当前用户

    Returns:
        当前超级用户对象

    Raises:
        HTTPException: 用户不是超级用户时抛出403异常
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足"
        )
    return current_user


def get_optional_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    获取可选的当前用户（用于不强制认证的接口）

    Args:
        token: JWT令牌（可选）
        db: 数据库会话

    Returns:
        当前用户对象或None
    """
    if not token:
        return None

    try:
        username = verify_token(token)
        if username is None:
            return None

        user = get_user_by_username(db, username)
        return user
    except JWTError:
        return None


def get_database() -> Session:
    """
    获取数据库会话依赖

    Returns:
        数据库会话对象
    """
    return Depends(get_db)


class CommonQueryParams:
    """通用查询参数"""
    
    def __init__(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc"
    ):
        self.page = max(1, page)
        self.page_size = min(100, max(1, page_size))
        self.sort_by = sort_by
        self.sort_order = sort_order.lower()
        
        if self.sort_order not in ["asc", "desc"]:
            self.sort_order = "desc"