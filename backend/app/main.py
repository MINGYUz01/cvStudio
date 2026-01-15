"""
CV Studio - 计算机视觉任务管理平台
FastAPI主应用入口
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.core.config import settings
from app.api.v1 import auth, datasets, models, training, inference, users, websocket, training_logs, augmentation, weights
from app.core.exceptions import setup_exception_handlers
from app.utils.metrics_collector import collector
from app.api.websocket import manager
from app.database import create_tables
from app.models import User, Dataset, Model, WeightLibrary, TrainingRun, Checkpoint, InferenceJob, AugmentationStrategy


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print(f"🚀 {settings.PROJECT_NAME} 正在启动...")
    print(f"📍 环境: {settings.ENVIRONMENT}")
    print(f"🌐 服务地址: http://{settings.HOST}:{settings.PORT}")

    # 创建数据库表（如果不存在）
    create_tables()
    print("📊 数据库表已就绪")

    # 启动系统指标收集器
    async def metrics_callback(metrics):
        """指标收集回调函数，将数据推送给订阅者"""
        await manager.send_system_update({
            "type": "system_stats",
            "data": metrics
        })

    await collector.start_collection(callback=metrics_callback)
    print("📊 系统指标收集器已启动")

    yield

    # 关闭时执行
    print("👋 应用正在关闭...")

    # 停止指标收集器
    await collector.stop_collection()
    print("📊 系统指标收集器已停止")


def create_application() -> FastAPI:
    """创建FastAPI应用实例"""
    
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.APP_VERSION,
        description="计算机视觉任务管理平台API",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url=f"{settings.API_V1_STR}/docs",
        redoc_url=f"{settings.API_V1_STR}/redoc",
        lifespan=lifespan
    )
    
    # 设置CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 设置受信任主机
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]
        )
    
    # 设置异常处理器
    setup_exception_handlers(app)
    
    # 注册路由
    app.include_router(
        auth.router,
        prefix=f"{settings.API_V1_STR}/auth",
        tags=["认证"]
    )
    
    app.include_router(
        users.router,
        prefix=settings.API_V1_STR
    )
    
    app.include_router(
        datasets.router,
        prefix=f"{settings.API_V1_STR}/datasets",
        tags=["数据集"]
    )
    
    app.include_router(
        models.router,
        prefix=f"{settings.API_V1_STR}/models",
        tags=["模型"]
    )
    
    app.include_router(
        training.router,
        prefix=f"{settings.API_V1_STR}/training",
        tags=["训练"]
    )
    
    app.include_router(
        inference.router,
        prefix=f"{settings.API_V1_STR}/inference",
        tags=["推理"]
    )

    # WebSocket路由
    app.include_router(
        websocket.router,
        prefix=settings.API_V1_STR,
        tags=["WebSocket"]
    )

    # 训练日志API
    app.include_router(
        training_logs.router,
        prefix=f"{settings.API_V1_STR}/training",
        tags=["训练日志"]
    )

    # 数据增强API
    app.include_router(
        augmentation.router,
        prefix=f"{settings.API_V1_STR}/augmentation",
        tags=["数据增强"]
    )

    # 权重库API
    app.include_router(
        weights.router,
        prefix=f"{settings.API_V1_STR}/weights",
        tags=["权重库"]
    )

    return app


app = create_application()


@app.get("/")
async def root():
    """根路径健康检查"""
    return {
        "message": f"欢迎使用 {settings.PROJECT_NAME}",
        "version": settings.APP_VERSION,
        "status": "运行中",
        "docs": f"{settings.API_V1_STR}/docs"
    }


@app.get("/health")
async def health_check():
    """详细健康检查"""
    return {
        "status": "健康",
        "app_name": settings.PROJECT_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )