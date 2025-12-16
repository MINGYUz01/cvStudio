"""
统一API响应格式工具
"""

from typing import Any, Optional, List
from datetime import datetime
from pydantic import BaseModel
from fastapi import status
import json


class CustomJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理datetime等特殊类型"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class APIResponse(BaseModel):
    """统一API响应格式"""
    success: bool
    message: str
    data: Optional[Any] = None
    code: int = status.HTTP_200_OK

    class Config:
        from_attributes = True


class PaginatedResponse(BaseModel):
    """分页响应格式"""
    success: bool
    message: str
    data: List[Any]
    pagination: dict
    code: int = status.HTTP_200_OK

    class Config:
        from_attributes = True


class ErrorResponse(BaseModel):
    """错误响应格式"""
    success: bool = False
    message: str
    error_detail: Optional[str] = None
    code: int = status.HTTP_400_BAD_REQUEST

    class Config:
        from_attributes = True


def success_response(
    message: str = "操作成功",
    data: Any = None,
    code: int = status.HTTP_200_OK
) -> APIResponse:
    """
    创建成功响应

    Args:
        message: 成功消息
        data: 响应数据
        code: HTTP状态码

    Returns:
        API响应对象
    """
    return APIResponse(
        success=True,
        message=message,
        data=data,
        code=code
    )


def error_response(
    message: str = "操作失败",
    error_detail: Optional[str] = None,
    code: int = status.HTTP_400_BAD_REQUEST
) -> ErrorResponse:
    """
    创建错误响应

    Args:
        message: 错误消息
        error_detail: 详细错误信息
        code: HTTP状态码

    Returns:
        错误响应对象
    """
    return ErrorResponse(
        message=message,
        error_detail=error_detail,
        code=code
    )


def paginated_response(
    message: str = "获取成功",
    data: List[Any] = None,
    page: int = 1,
    page_size: int = 20,
    total: int = 0,
    code: int = status.HTTP_200_OK
) -> PaginatedResponse:
    """
    创建分页响应

    Args:
        message: 成功消息
        data: 数据列表
        page: 当前页码
        page_size: 每页大小
        total: 总记录数
        code: HTTP状态码

    Returns:
        分页响应对象
    """
    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return PaginatedResponse(
        success=True,
        message=message,
        data=data or [],
        pagination={
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1
        },
        code=code
    )