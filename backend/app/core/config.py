"""
应用配置管理
使用Pydantic Settings进行配置管理
"""

from functools import lru_cache
from typing import List, Optional
from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置类"""
    
    # 应用基础配置
    APP_NAME: str = "CV Studio"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # 数据库配置
    DATABASE_URL: str = "sqlite:///./cvstudio.db"
    
    # JWT认证配置
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # 日志配置
    LOG_LEVEL: str = "DEBUG"
    LOG_FILE: str = "./logs/app.log"
    
    # 文件存储配置（统一使用 backend/data/ 目录）
    UPLOAD_DIR: str = "./data/uploads"
    DATASETS_DIR: str = "./data/datasets"
    DATASET_STORAGE_PATH: str = "./data/datasets"
    THUMBNAIL_STORAGE_PATH: str = "./data/thumbnails"
    MODELS_DIR: str = "./data/models"
    CHECKPOINTS_DIR: str = "./data/checkpoints"
    TEMP_DIR: str = "./data/temp"
    MAX_UPLOAD_SIZE: int = 1073741824  # 1GB
    
    # 训练配置
    DEFAULT_TRAINING_DEVICE: str = "cpu"
    MAX_TRAINING_PROCESSES: int = 2
    TRAINING_TIMEOUT: int = 3600  # 1小时
    
    # Redis配置
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # CORS配置
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    # API配置
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CV Studio API"
    
    # 分页配置
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    
    # 缓存配置
    CACHE_TTL: int = 3600  # 1小时
    
    # 图像处理配置
    MAX_IMAGE_SIZE: int = 4096
    THUMBNAIL_SIZE: int = 256
    SUPPORTED_IMAGE_FORMATS: List[str] = [
        "jpg", "jpeg", "png", "bmp", "tiff"
    ]
    SUPPORTED_DATASET_FORMATS: List[str] = [
        "yolo", "coco", "voc", "classification"
    ]
    
    # 性能监控
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # 默认管理员账户配置
    DEFAULT_ADMIN_USERNAME: str = "admin"
    DEFAULT_ADMIN_EMAIL: str = "admin@cvstudio.example.com"
    DEFAULT_ADMIN_PASSWORD: str = "admin123"
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        """处理CORS origins配置"""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, list):
            return v
        raise ValueError("ALLOWED_ORIGINS必须是字符串或列表")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """获取配置实例（带缓存）"""
    return Settings()


# 全局配置实例
settings = get_settings()