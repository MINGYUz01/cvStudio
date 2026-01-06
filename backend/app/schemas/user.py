"""
用户相关的Pydantic模式
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


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


class UserUpdate(BaseModel):
    """更新用户模式"""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None


class UserResponse(UserBase):
    """用户响应模式"""
    id: int
    is_active: bool
    is_superuser: bool
    created_at: datetime = Field(..., description="创建时间")

    class Config:
        from_attributes = True


class Token(BaseModel):
    """访问令牌模式"""
    access_token: str
    refresh_token: str
    token_type: str
    user: UserResponse


class TokenRefresh(BaseModel):
    """刷新令牌模式"""
    refresh_token: str


class TokenData(BaseModel):
    """令牌数据模式"""
    username: Optional[str] = None


class UserConfig(BaseModel):
    """用户配置模式"""
    theme: Optional[str] = "dark"
    language: Optional[str] = "zh-CN"
    default_dataset_id: Optional[int] = None
    default_model_id: Optional[int] = None
    notifications_enabled: Optional[bool] = True
    auto_save: Optional[bool] = True


class UserPasswordChange(BaseModel):
    """用户密码修改模式"""
    old_password: str
    new_password: str