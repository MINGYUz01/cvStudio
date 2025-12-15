"""
用户相关的Pydantic模式
"""

from typing import Optional
from pydantic import BaseModel, EmailStr


class UserBase(BaseModel):
    """用户基础模式"""
    username: str
    email: EmailStr


class UserCreate(UserBase):
    """创建用户模式"""
    password: str


class UserLogin(BaseModel):
    """用户登录模式"""
    username: str
    password: str


class UserResponse(UserBase):
    """用户响应模式"""
    id: int
    is_active: bool
    is_superuser: bool
    created_at: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    """访问令牌模式"""
    access_token: str
    token_type: str
    user: UserResponse


class TokenData(BaseModel):
    """令牌数据模式"""
    username: Optional[str] = None