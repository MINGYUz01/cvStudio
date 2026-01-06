"""
用户数据模型
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    config = Column(String(1000), nullable=True)  # JSON字符串存储用户配置
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    # updated_at = Column(DateTime(timezone=True), onupdate=func.now())  # 暂时注释掉

    # 关联关系
    models = relationship("Model", back_populates="creator")
    training_runs = relationship("TrainingRun", back_populates="creator")
    inference_jobs = relationship("InferenceJob", back_populates="creator")

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

    @property
    def config_dict(self):
        """获取配置字典"""
        import json
        if self.config:
            try:
                return json.loads(self.config)
            except:
                return {}
        return {}

    @config_dict.setter
    def config_dict(self, value):
        """设置配置字典"""
        import json
        self.config = json.dumps(value) if value else None