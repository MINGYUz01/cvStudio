"""
CV Studio 后端启动脚本
"""

import sys
import os

# 添加项目路径到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    import uvicorn
    from app.core.config import settings
    
    print(f"🚀 正在启动 {settings.PROJECT_NAME}...")
    print(f"📍 环境: {settings.ENVIRONMENT}")
    print(f"🌐 服务地址: http://{settings.HOST}:{settings.PORT}")
    print(f"📚 API文档: http://{settings.HOST}:{settings.PORT}{settings.API_V1_STR}/docs")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )