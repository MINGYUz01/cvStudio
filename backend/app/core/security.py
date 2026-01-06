"""
安全相关功能
包括密码哈希、JWT令牌等
"""

from datetime import datetime, timedelta
from typing import Optional, Union, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from app.core.config import settings


# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    创建访问令牌

    Args:
        subject: 令牌主题（通常是用户ID或用户名）
        expires_delta: 过期时间增量

    Returns:
        JWT令牌字符串
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "type": "access"
    }
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    创建刷新令牌

    Args:
        subject: 令牌主题（通常是用户ID或用户名）
        expires_delta: 过期时间增量

    Returns:
        JWT刷新令牌字符串
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        # 默认7天有效期
        expire = datetime.utcnow() + timedelta(days=7)

    to_encode = {
        "exp": expire,
        "sub": str(subject),
        "type": "refresh"
    }
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def verify_token(token: str) -> Optional[str]:
    """
    验证JWT令牌

    Args:
        token: JWT令牌字符串

    Returns:
        令牌主题，如果验证失败返回None
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        token_data = payload.get("sub")
        if token_data is None:
            return None
        return str(token_data)
    except JWTError:
        return None


def verify_token_type(token: str, expected_type: str) -> Optional[str]:
    """
    验证JWT令牌及其类型

    Args:
        token: JWT令牌字符串
        expected_type: 期望的令牌类型（"access" 或 "refresh"）

    Returns:
        令牌主题，如果验证失败返回None
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        token_type = payload.get("type")
        token_data = payload.get("sub")

        if token_data is None or token_type != expected_type:
            return None
        return str(token_data)
    except JWTError:
        return None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    验证密码
    
    Args:
        plain_password: 明文密码
        hashed_password: 哈希密码
        
    Returns:
        密码是否匹配
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    获取密码哈希值
    
    Args:
        password: 明文密码
        
    Returns:
        哈希密码字符串
    """
    return pwd_context.hash(password)


def create_password_reset_token(email: str) -> str:
    """
    创建密码重置令牌

    Args:
        email: 用户邮箱

    Returns:
        密码重置令牌
    """
    delta = timedelta(hours=1)  # 默认1小时有效期
    now = datetime.utcnow()
    expires = now + delta
    exp = expires.timestamp()
    encoded_jwt = jwt.encode(
        {"exp": exp, "nbf": now, "sub": email},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    验证密码重置令牌
    
    Args:
        token: 密码重置令牌
        
    Returns:
        用户邮箱，验证失败返回None
    """
    try:
        decoded_token = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        return decoded_token["sub"]
    except JWTError:
        return None


def generate_secure_filename(original_filename: str) -> str:
    """
    生成安全的文件名
    
    Args:
        original_filename: 原始文件名
        
    Returns:
        安全的文件名
    """
    import os
    import uuid
    
    # 获取文件扩展名
    file_extension = os.path.splitext(original_filename)[1]
    
    # 生成UUID作为文件名
    secure_name = str(uuid.uuid4())
    
    return f"{secure_name}{file_extension}"


def validate_file_type(filename: str, allowed_types: list) -> bool:
    """
    验证文件类型
    
    Args:
        filename: 文件名
        allowed_types: 允许的文件类型列表
        
    Returns:
        文件类型是否允许
    """
    import os
    
    file_extension = os.path.splitext(filename)[1].lower().lstrip('.')
    return file_extension in [ext.lower() for ext in allowed_types]


def sanitize_input(text: str) -> str:
    """
    清理用户输入，防止XSS攻击
    
    Args:
        text: 用户输入文本
        
    Returns:
        清理后的文本
    """
    import html
    
    # HTML转义
    sanitized = html.escape(text)
    
    # 移除潜在的脚本标签
    sanitized = sanitized.replace("<script", "").replace("</script>", "")
    
    return sanitized.strip()