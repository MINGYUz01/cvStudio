"""
数据库连接配置
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基础模型类
Base = declarative_base()


def get_db():
    """
    获取数据库会话

    Yields:
        数据库会话对象
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """创建所有数据表"""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """删除所有数据表"""
    Base.metadata.drop_all(bind=engine)


def init_admin_user():
    """
    初始化默认管理员用户
    如果数据库中不存在任何用户，则创建默认管理员账户
    """
    from app.models.user import User
    from app.core.security import get_password_hash

    db = SessionLocal()
    try:
        # 检查是否已存在用户
        existing_user = db.query(User).first()
        if existing_user:
            print(f"👤 数据库已存在用户，跳过管理员初始化")
            return

        # 创建默认管理员用户
        admin_user = User(
            username=settings.DEFAULT_ADMIN_USERNAME,
            email=settings.DEFAULT_ADMIN_EMAIL,
            password_hash=get_password_hash(settings.DEFAULT_ADMIN_PASSWORD),
            is_active=True,
            is_superuser=True
        )
        db.add(admin_user)
        db.commit()

        print(f"👤 默认管理员账户已创建:")
        print(f"   用户名: {settings.DEFAULT_ADMIN_USERNAME}")
        print(f"   邮箱: {settings.DEFAULT_ADMIN_EMAIL}")
        print(f"   密码: {settings.DEFAULT_ADMIN_PASSWORD}")
        print(f"   ⚠️  请在生产环境中及时修改默认密码！")
    except Exception as e:
        db.rollback()
        print(f"❌ 创建管理员用户失败: {e}")
    finally:
        db.close()