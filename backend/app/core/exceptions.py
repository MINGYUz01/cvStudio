"""
自定义异常和异常处理器
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Union


class CVStudioException(Exception):
    """CV Studio基础异常类"""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Union[dict, list, None] = None
    ):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)


class ValidationError(CVStudioException):
    """数据验证错误"""
    
    def __init__(self, message: str, details: Union[dict, list, None] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details
        )


class NotFoundError(CVStudioException):
    """资源未找到错误"""
    
    def __init__(self, resource: str, identifier: Union[str, int] = None):
        message = f"{resource}未找到"
        if identifier:
            message += f": {identifier}"
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND
        )


class PermissionError(CVStudioException):
    """权限错误"""
    
    def __init__(self, message: str = "权限不足"):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN
        )


class AuthenticationError(CVStudioException):
    """认证错误"""
    
    def __init__(self, message: str = "认证失败"):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED
        )


class DatasetError(CVStudioException):
    """数据集相关错误"""
    
    def __init__(self, message: str, details: Union[dict, list, None] = None):
        super().__init__(
            message=f"数据集错误: {message}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class ModelError(CVStudioException):
    """模型相关错误"""
    
    def __init__(self, message: str, details: Union[dict, list, None] = None):
        super().__init__(
            message=f"模型错误: {message}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class TrainingError(CVStudioException):
    """训练相关错误"""
    
    def __init__(self, message: str, details: Union[dict, list, None] = None):
        super().__init__(
            message=f"训练错误: {message}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class InferenceError(CVStudioException):
    """推理相关错误"""
    
    def __init__(self, message: str, details: Union[dict, list, None] = None):
        super().__init__(
            message=f"推理错误: {message}",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


async def cvstudio_exception_handler(request: Request, exc: CVStudioException):
    """CV Studio自定义异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.message,
            "details": exc.details,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理器"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "message": "请求数据验证失败",
            "details": exc.errors(),
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "path": str(request.url.path)
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "message": "服务器内部错误",
            "details": str(exc) if __debug__ else None,
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "path": str(request.url.path)
        }
    )


def setup_exception_handlers(app: FastAPI):
    """设置异常处理器"""
    app.add_exception_handler(CVStudioException, cvstudio_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)