"""
模型数据模型
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.database import Base


class Model(Base):
    """模型模型"""
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    graph_json = Column(JSON, nullable=False)  # 模型图的JSON表示
    code_path = Column(String(500), nullable=True)  # 生成的代码路径
    template_tag = Column(String(50), nullable=True)  # 模板标签
    version = Column(String(20), default="1.0")
    tags = Column(JSON, nullable=True)  # 标签
    is_active = Column(String(10), default="active")  # active, deleted
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # 关联关系
    creator = relationship("User", back_populates="models")
    training_runs = relationship("TrainingRun", back_populates="model")

    def __repr__(self):
        return f"<Model(id={self.id}, name='{self.name}', version='{self.version}')>"