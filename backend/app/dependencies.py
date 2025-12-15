"""
FastAPI依赖注入
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.security import verify_token

# HTTP Bearer 认证方案
security = HTTPBearer()


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    获取当前用户ID
    
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